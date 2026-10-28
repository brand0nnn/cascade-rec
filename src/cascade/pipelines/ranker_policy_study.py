"""Controlled score-input x supervised-population MLP study, validation only.

Development, pilot and replication use separate fresh output roots. Existing
candidate archives are immutable inputs; this module never opens Test predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.ranker_meso import group_labels
from cascade.eval.policy_score_diagnostics import measure_score_response, uniform_cohort
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.pipelines.ranker_training import create_ranker, fit_candidate_lists
from cascade.utils.config import load_config
from cascade.utils.io import REPO_ROOT, git_hash, save_run
from cascade.utils.seed import set_seed

ARMS = {
    "A": {"training_population": "all", "retrieval_score_input": "enabled"},
    "B": {"training_population": "all", "retrieval_score_input": "masked"},
    "C": {"training_population": "natural_only", "retrieval_score_input": "enabled"},
    "D": {"training_population": "natural_only", "retrieval_score_input": "masked"},
}
STAGE_SEEDS = {"development": [0], "pilot": [0], "replication": [1, 2]}


def digest(path: Path) -> str:
    """Hash large immutable inputs without reading them all into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _path(value: str | Path) -> Path:
    return (REPO_ROOT / value).resolve()


def _write(path: Path, value: Mapping) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temporary.replace(path)


def validate_study_config(cfg: Mapping) -> dict:
    """Reject protocol changes before any result files are created."""
    if cfg.get("scope") != "validation only" or cfg.get("candidate_counts") != [50, 100, 200, 500]:
        raise ValueError("policy study fixes validation-only K1=[50,100,200,500]")
    if cfg.get("anchor_k1") != 200 or cfg.get("anchor_macro_ndcg_tolerance") != 1e-6:
        raise ValueError("policy study fixes the K200 reproduction tolerance to 1e-6")
    ranker = load_config(_path(cfg["ranker_config"]))
    fixed = {"model": "mlp", "k1": 200, "max_epochs": 30, "batch_size": 2048,
             "learning_rate": 0.001, "loss": "binary_cross_entropy", "optimizer": "adam",
             "history_length": 50, "item_text_dim": 384, "dropout": 0.1,
             "hidden_dims": [256, 128, 64], "feature_groups": ["retrieval", "matching"]}
    if any(ranker.get(key) != value for key, value in fixed.items()):
        raise ValueError("policy study requires the unchanged matching-MLP training recipe")
    if ranker.get("early_stopping") != {"metric": "ndcg@10_macro", "patience": 3}:
        raise ValueError("policy study fixes natural-hit conditional nDCG checkpoint selection")
    if ranker.get("negative_sampling") != {"negatives_per_list": 50, "strategy": "uniform_without_replacement"}:
        raise ValueError("policy study fixes 50 uniform negatives without replacement")
    dev = cfg.get("development", {})
    if dev != {"user_fraction": 0.02, "seed": 0, "max_epochs": 1}:
        raise ValueError("development must use fixed 2% users, seed 0 and one epoch")
    if int(cfg.get("score_batch_pairs", 0)) < 1:
        raise ValueError("score_batch_pairs must be positive")
    return ranker


def required_input_paths(cfg: Mapping) -> list[Path]:
    """Inputs needed locally; Test prediction files are deliberately absent."""
    data_cfg = load_config(_path(cfg["data_config"]))
    processed = _path(data_cfg["processed_dir"])
    candidates = _path(cfg["candidate_dir"])
    return [*(processed / name for name in (
        "sequences.npz", "items.parquet", "item_segments.parquet", "instances.parquet",
        "interactions.parquet", "split.json", "item_text_emb.npy")),
        *(candidates / name for name in ("ranker_train.npz", "val.npz", "val_target_score.npz", "meta.json")),
        _path(cfg["data_config"]), _path(cfg["ranker_config"]),
        _path(cfg["reference_comparison"]), _path(cfg["reference_ranker_config"])]


def select_training_lists(full: Mapping[str, np.ndarray], population: str,
                          users: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Filter complete supervised lists, preserving target labels and padding."""
    if population not in ("all", "natural_only"):
        raise ValueError("unknown supervised training population")
    mask = np.ones(len(full["inst"]), dtype=bool)
    if users is not None:
        mask &= np.isin(full["user"], users)
    if population == "natural_only":
        mask &= ~np.asarray(full["forced"], dtype=bool)
    if not mask.any():
        raise ValueError(f"empty {population} training population")
    return {key: np.asarray(value)[mask] for key, value in full.items()}


def frozen_feature_builder(data: ProcessedData, full_train: Mapping[str, np.ndarray],
                           reference_config: Path) -> RankerFeatureBuilder:
    """Fit the original full-training contract before any supervised filtering."""
    builder = RankerFeatureBuilder.create(data, full_train)
    old = load_config(reference_config)["run"]
    if dict(builder.item_cardinalities) != old["feature_spec"]["cardinalities"] or not np.allclose(
            [builder.score_mean, builder.score_std], [old["score_mean"], old["score_std"]], rtol=0, atol=1e-8):
        raise ValueError("original full-training feature contract could not be reproduced")
    return builder


def population_support(data: ProcessedData, lists: Mapping[str, np.ndarray]) -> dict:
    """Observed support, not an assertion of equal optimization exposure."""
    table = data.instances.set_index("inst").loc[lists["inst"]]
    first = data.items.set_index("item").loc[table["item"].to_numpy(), "first_ts"].to_numpy()
    new = first >= int(data.split["t_ranker_train"])
    return {"n_lists": len(table), "n_users": int(table["user"].nunique()),
            "n_target_items": int(table["item"].nunique()),
            "n_new_in_ranker_period_targets": int(new.sum()),
            "forced_count": int(np.asarray(lists["forced"]).sum()),
            "forced_fraction": float(np.asarray(lists["forced"]).mean())}


def _validate_alignment(data: ProcessedData, lists: Mapping[str, np.ndarray], period: int) -> None:
    ids = np.asarray(lists["inst"])
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("candidate instance IDs must be nonempty and unique")
    table = data.instances.set_index("inst").loc[ids]
    if not np.array_equal(table["user"], lists["user"]) or not (table["period"] == period).all():
        raise ValueError("candidate user/period alignment failed; Test is prohibited")
    cutoff = int(data.split["t_ranker_train"] if period == 1 else data.split["t1"])
    end = int(data.split["t1"] if period == 1 else data.split["t2"])
    if not ((table["ts"] >= cutoff) & (table["ts"] < end)).all():
        raise ValueError("candidate timestamps are outside the permitted training/validation period")
    if np.asarray(lists["cand"]).shape != np.asarray(lists["score"]).shape:
        raise ValueError("candidate/score shapes differ")


def _parameter_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(config_path: Path, run_id: str, *, stage: str, arms: list[str] | None = None,
        seeds: list[int] | None = None, device: str = "cpu", candidate_dir: str | None = None,
        data_config: str | None = None) -> Path:
    """Train selected fresh arms and export selected-checkpoint natural Val prefixes."""
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run ID must be a single directory name")
    if stage not in STAGE_SEEDS:
        raise ValueError("stage must be development, pilot or replication")
    arms = list(ARMS) if arms is None else arms
    seeds = STAGE_SEEDS[stage].copy() if seeds is None else seeds
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("arms must be a unique nonempty subset of A/B/C/D")
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in STAGE_SEEDS[stage] for s in seeds):
        raise ValueError(f"{stage} seeds must be a unique subset of {STAGE_SEEDS[stage]}")
    cfg = load_config(config_path)
    if candidate_dir is not None:
        cfg["candidate_dir"] = candidate_dir
    if data_config is not None:
        cfg["data_config"] = data_config
    ranker_cfg = validate_study_config(cfg)
    development = stage == "development"
    root = REPO_ROOT / "results/ranker_policy" / (run_id + ("_sample" if development else ""))
    if root.exists():
        raise FileExistsError(f"refusing existing study: {root}")
    required = required_input_paths(cfg)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Recover original processed inputs and frozen candidates before training:\n" + "\n".join(missing))
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("policy study supports cpu, cuda or mps devices")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if selected_device.type == "cuda" and selected_device.index is not None and selected_device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    if selected_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    hashes = {str(path): digest(path) for path in required}
    frozen = json.loads(_path(cfg["reference_comparison"]).read_text())["candidate_source"]
    cand_dir = _path(cfg["candidate_dir"])
    if hashes[str(cand_dir / "meta.json")] != frozen["meta_sha256"] or hashes[str(cand_dir / "val.npz")] != frozen["val_sha256"]:
        raise ValueError("candidate metadata/validation hashes differ from the frozen study")
    data = ProcessedData.load(str(_path(cfg["data_config"])))
    full_train = load_candidates(cand_dir, "ranker_train")
    natural_val = load_candidates(cand_dir, "val", k1=500, force=False)
    val = load_candidates(cand_dir, "val", k1=200, force=True)
    _validate_alignment(data, full_train, 1)
    _validate_alignment(data, natural_val, 2)
    _validate_alignment(data, val, 2)
    if full_train["cand"].shape[1] != 201 or natural_val["cand"].shape[1] != 500:
        raise ValueError("expected original forced top200 training and natural top500 validation lists")
    if not np.array_equal(val["inst"], natural_val["inst"]):
        raise ValueError("training-selector and prefix validation populations differ")
    # Critical: fit normalization/vocabularies on FULL original training lists,
    # before either development-user or natural-only selection.
    builder = frozen_feature_builder(data, full_train, _path(cfg["reference_ranker_config"]))
    universe = np.asarray(full_train["inst"]).copy()
    chosen_users = None
    if development:
        chosen_users = np.random.default_rng(0).choice(np.arange(1, data.n_users + 1),
            max(1, round(data.n_users * 0.02)), replace=False)
        mask = np.isin(val["user"], chosen_users)
        val = {key: value[mask] for key, value in val.items()}
        natural_val = {key: value[mask] for key, value in natural_val.items()}
    if not len(val["inst"]) or not (~val["forced"]).any():
        raise ValueError("validation population must include natural retrieval hits")
    training = {arm: select_training_lists(full_train, ARMS[arm]["training_population"], chosen_users) for arm in arms}
    labels = group_labels(data, data.instances.set_index("inst").loc[natural_val["inst"]].reset_index())
    diagnostic_rows, diagnostic_cohort, diagnostic_identity = uniform_cohort(
        natural_val, queries=20 if development else 1000)
    spec = RankerFeatureSpec(cardinalities=builder.item_cardinalities, history_length=50,
                            item_text_dim=384, feature_groups=("retrieval", "matching"))
    source_paths = [Path(__file__), Path(__file__).with_name("ranker_training.py"),
                    Path(__file__).with_name("ranker_k1_sweep.py"),
                    REPO_ROOT / "src/cascade/models/ranker_mlp.py",
                    REPO_ROOT / "src/cascade/data/ranker_features.py",
                    REPO_ROOT / "src/cascade/data/instances.py",
                    REPO_ROOT / "src/cascade/pipelines/build_candidates.py",
                    REPO_ROOT / "src/cascade/eval/ranker.py",
                    REPO_ROOT / "src/cascade/eval/metrics.py",
                    REPO_ROOT / "src/cascade/eval/ranker_meso.py",
                    REPO_ROOT / "src/cascade/eval/bootstrap.py",
                    REPO_ROOT / "src/cascade/eval/policy_score_diagnostics.py",
                    REPO_ROOT / "requirements.txt",
                    REPO_ROOT / "src/cascade/utils/seed.py", config_path.resolve()]
    source_hashes = {str(path): digest(path) for path in source_paths}
    root.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "stage": stage, "development_sample": development,
              "completed": [], "test_predictions_loaded": False,
              "started_at": datetime.now(timezone.utc).isoformat()}
    study = {"scope": "validation only", "stage": stage, "development_sample": development,
             "arms": {arm: ARMS[arm] for arm in arms}, "seeds": seeds,
             "candidate_counts": cfg["candidate_counts"], "anchor_k1": 200,
             "jobs": [], "prior_validation_inspected": True,
             "normalization_population": "full original ranker-training candidates",
             "score_mean": builder.score_mean, "score_std": builder.score_std,
             "sampling_strategy": "instance_keyed", "training_universe_lists": len(universe),
             "candidate_source": frozen,
             "score_diagnostic_cohort": {"file": "uniform_score_cohort.npz",
                 "identity_sha256": diagnostic_identity, "rng_seed": 0,
                 "requested_queries": 20 if development else 1000,
                 "n_queries": len(diagnostic_rows), "includes_retrieval_misses": True},
             "causal_limit": "natural-only is a total recipe intervention with different support and optimizer updates",
             "final_selected_configuration": None, "test_predictions_loaded": False}
    save_run(root, status, cfg)
    _write(root / "status.json", status)
    _write(root / "study.json", study)
    _write(root / "provenance.json", {"input_sha256": hashes, "source_sha256": source_hashes,
                                     "git_hash": git_hash(), "test_predictions_loaded": False})
    labels.to_parquet(root / "validation_groups.parquet", index=False)
    np.savez_compressed(root / "training_universe.npz", inst=universe,
                        selected_users=np.array([], dtype=np.int64) if chosen_users is None else chosen_users)
    np.savez_compressed(root / "uniform_score_cohort.npz", **diagnostic_cohort)
    study["score_diagnostic_cohort"]["file_sha256"] = digest(root / "uniform_score_cohort.npz")
    _write(root / "study.json", study)
    initial_by_seed = {}
    try:
        for seed in seeds:
            for arm in arms:
                for path, expected in source_hashes.items():
                    if digest(Path(path)) != expected:
                        raise RuntimeError(f"source changed during policy study: {path}")
                arm_cfg = {**ranker_cfg, **ARMS[arm], "seed": seed, "policy_arm": arm,
                           "sampling_strategy": "instance_keyed", "development_sample": development,
                           "max_epochs": 1 if development else 30,
                           "run": {"candidate_dir": str(cand_dir), "seed": seed,
                                   "feature_spec": asdict(spec), "score_mean": builder.score_mean,
                                   "score_std": builder.score_std, "normalization_population": "full original training"}}
                set_seed(seed)
                model = create_ranker(spec, arm_cfg)
                initial = _parameter_digest(model)
                if seed in initial_by_seed and initial != initial_by_seed[seed]:
                    raise RuntimeError("paired arms did not receive identical initial parameters")
                initial_by_seed[seed] = initial
                directory = root / f"arm_{arm}_seed{seed}"
                directory.mkdir(exist_ok=False)
                support = population_support(data, training[arm])
                support["batches_per_epoch"] = int(np.ceil(support["n_lists"] / 40))
                save_run(directory, {"status": "running", "development_sample": development}, arm_cfg)
                result = fit_candidate_lists(model, builder, training[arm], val, n_negatives=50,
                    batch_size=2048, learning_rate=0.001, max_epochs=arm_cfg["max_epochs"], patience=3,
                    seed=seed, device=device, checkpoint_path=directory / "model.pt",
                    sampling_strategy="instance_keyed", training_instance_universe=universe)
                ranks, aucs = score_prefixes(model, builder, natural_val, cfg["candidate_counts"],
                    device=device, batch_pairs=int(cfg["score_batch_pairs"]))
                with np.load(directory / "val_ranks.npz", allow_pickle=False) as archive:
                    old_ranks = archive["ranks"]
                    if not np.array_equal(archive["inst"], natural_val["inst"]) or not np.array_equal(archive["user"], natural_val["user"]):
                        raise ValueError("selected checkpoint rank population changed")
                old_value = M.macro_micro(M.ndcg_at(old_ranks, 10), natural_val["user"])["macro"]
                new_value = M.macro_micro(M.ndcg_at(ranks[2], 10), natural_val["user"])["macro"]
                if abs(old_value - new_value) > 1e-6:
                    raise ValueError("selected K200 checkpoint does not reproduce within 1e-6")
                positional = np.stack([np.where((natural_val["target_pos"] >= 0) &
                    (natural_val["target_pos"] < k), natural_val["target_pos"] + 1, np.inf) for k in cfg["candidate_counts"]])
                np.savez_compressed(directory / "prefix_ranks.npz", inst=natural_val["inst"], user=natural_val["user"],
                    candidate_counts=np.array(cfg["candidate_counts"]), ranks=ranks, aucs=aucs,
                    target_pos=natural_val["target_pos"], retrieval_ranks=positional,
                    full_catalogue_ranks=natural_val["target_rank"])
                diagnostic_artifacts = ["score_diagnostics.json"]
                if ARMS[arm]["retrieval_score_input"] == "enabled":
                    diagnostic, raw_diagnostics = measure_score_response(model, builder, natural_val,
                        diagnostic_rows, cohort_identity_sha256=diagnostic_identity,
                        device=device, batch_pairs=int(cfg["score_batch_pairs"]))
                    np.savez_compressed(directory / "score_diagnostics.npz", **raw_diagnostics)
                    diagnostic_artifacts.append("score_diagnostics.npz")
                else:
                    diagnostic = {"status": "not_measured", "reason": "score masking imposes a flat score response",
                        "scope": "uniform validation events including retrieval misses",
                        "n_queries": len(diagnostic_rows), "cohort_identity_sha256": diagnostic_identity,
                        "n_retrieval_misses_at_500": int((diagnostic_cohort["target_pos"] < 0).sum()),
                        "interpretation": "Score-off flat sensitivity is imposed by design, not measured evidence."}
                diagnostic["development_sample"] = development
                _write(directory / "score_diagnostics.json", diagnostic)
                checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
                metrics = {"status": "complete", "development_sample": development,
                           "seed": seed, "arm": arm, **ARMS[arm], "best_epoch": result.best_epoch,
                           "best_val_ndcg10_macro": result.best_val_macro_ndcg10,
                           "best_val_gauc_macro": result.best_val_gauc, "history": result.history,
                           "training_support": support, "initial_parameters_sha256": initial,
                           "selected_optimizer_steps": checkpoint["optimizer_steps"],
                           "total_optimizer_steps": result.history[-1]["optimizer_steps"],
                           "anchor_check": {"selected_macro_ndcg10": old_value,
                               "prefix_macro_ndcg10": new_value, "changed_ranks": int(np.sum(old_ranks != ranks[2]))},
                           "validation_comparison": json.loads((directory / "validation_comparison.json").read_text()),
                           "score_diagnostics": diagnostic,
                           "test_predictions_loaded": False}
                save_run(directory, metrics, arm_cfg)
                record = {"arm": arm, "seed": seed, "directory": directory.name,
                          "status": "complete", "training_support": support,
                          "selected_optimizer_steps": checkpoint["optimizer_steps"],
                          "artifact_sha256": {name: digest(directory / name) for name in [
                              "model.pt", "config.yaml", "metrics.json", "val_ranks.npz", "prefix_ranks.npz",
                              *diagnostic_artifacts]}}
                study["jobs"].append(record)
                status["completed"].append({"arm": arm, "seed": seed})
                _write(root / "study.json", study)
                _write(root / "status.json", status)
        for path, expected in {**hashes, **source_hashes}.items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"input/source changed during policy study: {path}")
        status["status"] = "complete"
        study["status"] = "complete"
        _write(root / "study.json", study)
        _write(root / "status.json", status)
        save_run(root, status, cfg)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write(root / "status.json", status)
        raise
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ranker/policy_study.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=list(STAGE_SEEDS))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--candidate-dir")
    parser.add_argument("--data-config")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = run(_path(args.config), args.run_id, stage=args.stage, arms=args.arms,
               seeds=args.seeds, device=args.device, candidate_dir=args.candidate_dir,
               data_config=args.data_config)
    print(f"Completed policy study: {root}", flush=True)


if __name__ == "__main__":
    main()
