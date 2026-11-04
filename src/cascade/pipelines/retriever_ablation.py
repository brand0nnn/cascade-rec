"""Validation-only paired ID-residual dropout study and frozen 2x2 ID probe.

Pilot: ``--run-id STUDY --phase pilot``; seeds 1/2 require ``--phase replication``.
``--arm`` and ``--seed`` allow separate scheduler jobs. ``--summarize-only``
compares completed pairs; ``--probe-only --probe-run-dir RUN`` only rescored Val.
No path in this module evaluates Test or automatically promotes a pilot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from cascade.data.instances import ProcessedData
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import evaluate
from cascade.eval.metrics import macro_micro, ndcg_at, recall_at
from cascade.eval.saved_results import validate_ranks
from cascade.models.retriever import RetrieverScorer
from cascade.pipelines import train_retriever
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

ARMS = {"control": "configs/retriever/id_dropout_control.yaml", "dropout": "configs/retriever/id_dropout_treatment.yaml"}
PHASE_SEEDS = {"pilot": (0,), "replication": (1, 2)}


def study_paths(run_id: str, sample: bool = False) -> tuple[Path, str]:
    study = train_retriever.safe_run_id(run_id) + ("_sample" if sample else "")
    return REPO_ROOT / "results" / "retriever_ablation" / study, f"id_dropout/{study}"


def fit_path(run_id: str, arm: str, seed: int, sample: bool = False) -> Path:
    _, namespace = study_paths(run_id, sample)
    return REPO_ROOT / "results" / "retriever" / namespace / arm / f"seed{seed}"


def group_masks(data: ProcessedData) -> dict[str, np.ndarray]:
    """Fixed Val groups; release/seen labels remain descriptive, never inputs."""
    inst = data.split_instances("val")
    first = np.r_[np.iinfo(np.int64).max, data.items["first_ts"].values]
    _, seen = train_retriever.retriever_train_log_q(data)
    new = first[inst["item"].values] >= data.split["t1"]
    masks = {"all": np.ones(len(inst), bool), "new_in_period": new, "not_new": ~new,
             "seen_in_retriever_train": seen[inst["item"].values], "unseen_in_retriever_train": ~seen[inst["item"].values]}
    age_days = (inst["ts"].values - data.split["t1"]) / 86400000
    for name, lo, hi in (("days_0_89", 0, 90), ("days_90_179", 90, 180), ("days_180_269", 180, 270), ("days_270_plus", 270, np.inf)):
        masks[name] = (age_days >= lo) & (age_days < hi)
    return masks


def aligned_ranks(run_dir: Path, data: ProcessedData) -> tuple[np.ndarray, np.ndarray]:
    inst = data.split_instances("val")
    with np.load(run_dir / "val_ranks.npz") as z:
        if not np.array_equal(z["inst"], inst["inst"].values) or not np.array_equal(z["user"], inst["user"].values):
            raise ValueError(f"Validation rank alignment differs: {run_dir}")
        ranks = z["rank"]
        if ranks.shape != (len(inst),):
            raise ValueError(f"Validation ranks must have one-dimensional shape ({len(inst)},): {run_dir}")
        validate_ranks(ranks)
        ranks = ranks.astype(np.float64, copy=True)
        if "scorable" in z:
            scorable = z["scorable"]
            if scorable.shape != ranks.shape or scorable.dtype != np.bool_:
                raise ValueError(f"Scorable flags must be aligned one-dimensional booleans: {run_dir}")
            ranks[~scorable] = np.inf
    return ranks, inst["user"].values


def compare_ranks(data: ProcessedData, a: np.ndarray, b: np.ndarray, n_resamples: int) -> dict:
    """B-minus-A comparisons from one or several aligned arrays of target ranks."""
    users = data.split_instances("val")["user"].values
    out = {}
    for group, mask in group_masks(data).items():
        out[group] = {"events": int(mask.sum()), "users": int(len(np.unique(users[mask])))}
        for name, fn, cutoff in (("ndcg10", ndcg_at, 10), ("recall10", recall_at, 10), ("recall200", recall_at, 200)):
            av = np.mean([fn(r, cutoff) for r in np.atleast_2d(a)], axis=0)
            bv = np.mean([fn(r, cutoff) for r in np.atleast_2d(b)], axis=0)
            out[group][name] = {
                "reference": macro_micro(av[mask], users[mask]), "candidate": macro_micro(bv[mask], users[mask]),
                **paired_bootstrap(users[mask], av[mask], bv[mask], n_resamples=n_resamples, seed=0),
            }
    return out


def summarize_pairs(run_id: str, phase: str, data_config: str = "configs/data.yaml", sample: bool = False,
                    n_resamples: int = 1000, text_reference: str | None = None) -> dict:
    """Require both arms; replication aggregates all three completed seed pairs."""
    data = ProcessedData.load(data_config, sample)
    seeds = (0,) if phase == "pilot" else (0, 1, 2)
    controls, treatments, checks = [], [], []
    shared_identity = None
    for seed in seeds:
        paths = [fit_path(run_id, arm, seed, sample) for arm in ARMS]
        metrics = [json.loads((p / "metrics.json").read_text()) for p in paths]
        if any(m.get("status") != "complete" or m.get("test_predictions_loaded") for m in metrics):
            raise ValueError("Pair must contain completed validation-only runs")
        recipes = []
        for arm, path, recorded in zip(ARMS, paths, metrics):
            config = train_retriever.load_config(path / "config.yaml")
            expected_dropout = 0.0 if arm == "control" else 0.5
            if (recorded.get("seed") != seed or config.get("seed") != seed
                    or recorded.get("sample") != sample or config.get("sample") != sample
                    or recorded.get("id_dropout") != expected_dropout
                    or config.get("id_dropout") != expected_dropout
                    or recorded.get("evaluation_splits") != ["val"]
                    or not recorded.get("input_sha256") or not recorded.get("source_sha256")):
                raise ValueError(f"Unexpected seed, population, treatment or provenance: {path}")
            if recorded.get("checkpoint_sha256") != train_retriever.file_sha256(path / "model.pth"):
                raise ValueError(f"Checkpoint identity changed: {path}")
            recipes.append({k: v for k, v in config.items() if k not in ("name", "id_dropout", "seed", "run_id")})
            identity = {k: recorded[k] for k in ("input_sha256", "source_sha256", "data_config_sha256")}
            if shared_identity is None:
                shared_identity = identity
            if identity != shared_identity:
                raise ValueError("Training inputs or source code differ across arms/seeds")
        if recipes[0] != recipes[1]:
            raise ValueError(f"Training recipes differ within seed {seed}")
        if checks and recipes[0] != checks[0]["training_recipe"]:
            raise ValueError("Training recipes differ across seeds")
        for key in ("initial_state_sha256", "input_sha256", "selector"):
            if metrics[0][key] != metrics[1][key]:
                raise ValueError(f"Matched-arm provenance mismatch for seed {seed}: {key}")
        controls.append(aligned_ranks(paths[0], data)[0])
        treatments.append(aligned_ranks(paths[1], data)[0])
        checks.append({"seed": seed, "initial_state_sha256": metrics[0]["initial_state_sha256"],
                       "training_recipe": recipes[0], "matched": True,
                       "val_rank_sha256": {arm: train_retriever.file_sha256(path / "val_ranks.npz")
                                           for arm, path in zip(ARMS, paths)}})
    result = {
        "phase": phase, "seeds": list(seeds), "paired_provenance": checks,
        "shared_provenance": shared_identity,
        "dropout_minus_control": compare_ranks(data, np.stack(controls), np.stack(treatments), n_resamples),
        "test_predictions_loaded": False,
        "interpretation": "Exploratory validation comparison. User intervals condition on the trained seed mean; no noninferiority claim.",
        "text_reference_guard": {"status": "not_checked", "reason": "Supply --text-reference for the same-seed text-only reference."},
    }
    if text_reference is not None:
        for seed in seeds:
            path = Path(text_reference) / f"seed{seed}"
            config = train_retriever.load_config(path / "config.yaml")
            if config.get("seed") != seed or config.get("sample", False) != sample or config.get("item_tower") != "text":
                raise ValueError(f"Text reference seed/population/model mismatch: {path}")
        refs = [aligned_ranks(Path(text_reference) / f"seed{seed}", data)[0] for seed in seeds]
        comparison = compare_ranks(data, np.stack(refs), np.stack(treatments), n_resamples)
        result["dropout_minus_text"] = comparison
        result["text_reference_guard"] = {
            "status": "promising" if comparison["all"]["ndcg10"]["diff"] > 0 and comparison["new_in_period"]["recall200"]["diff"] >= 0 else "not_met",
            "rule": "Higher observed overall nDCG@10 and no observed new-item Recall@200 loss versus text-only; no statistical equivalence claim.",
        }
    output, _ = study_paths(run_id, sample)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{phase}_comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def run_probe(run_dir: str, output_dir: Path, data_config: str = "configs/data.yaml", device: str = "cpu",
              sample: bool = False, n_resamples: int = 1000) -> dict:
    """Frozen checkpoint probe; each cell evaluates the full natural Val catalogue."""
    if output_dir.exists():
        raise FileExistsError(f"Probe output already exists: {output_dir}")
    data = ProcessedData.load(data_config, sample)
    model, cfg = train_retriever.load_retriever(run_dir, data, device, data_config=data_config)
    if cfg["item_tower"] != "id_text":
        raise ValueError("The ID-branch probe requires a text + ID checkpoint")
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    metrics, ranks = {}, {}
    for history_on, candidate_on in ((True, True), (False, True), (True, False), (False, False)):
        cell = f"history{int(history_on)}_candidate{int(candidate_on)}"
        cell_dir = output_dir / cell
        metrics[cell] = evaluate(RetrieverScorer(model, history_use_id=history_on, candidate_use_id=candidate_on),
                                 data, "val", cutoffs=train_retriever.EVAL_CUTOFFS, save_dir=cell_dir)
        ranks[cell], _ = aligned_ranks(cell_dir, data)
    reference = ranks["history1_candidate1"]
    tolerance = 1e-6
    baseline_check = {"status": "unvalidated", "replay_validated": False,
                      "reason": "Source Val ranks not present", "ndcg10_tolerance": tolerance}
    if (Path(run_dir) / "val_ranks.npz").exists():
        original, _ = aligned_ranks(Path(run_dir), data)
        difference = abs(macro_micro(ndcg_at(original, 10), data.split_instances("val")["user"].values)["macro"] - metrics["history1_candidate1"]["ndcg@10"]["macro"])
        validated = bool(np.isfinite(difference) and difference <= tolerance)
        baseline_check = {"status": "validated" if validated else "unvalidated", "replay_validated": validated,
                          "rank_differences": int(np.sum(original != reference)), "ndcg10_absolute_difference": difference,
                          "ndcg10_tolerance": tolerance,
                          "reason": "ID-on macro nDCG reproduces within tolerance; rank differences retained" if validated else "ID-on macro nDCG differs beyond tolerance; original failure replay is unvalidated"}
    result = {
        "source_run": str(run_dir), "checkpoint_sha256": train_retriever.file_sha256(Path(run_dir) / "model.pth"),
        "data_config": str(data_config), "git_hash": git_hash(), "sample": sample, "metrics": metrics,
        "id_on_reproduction": baseline_check,
        "replay_validated": baseline_check["replay_validated"],
        "differences_from_ids_on": {cell: compare_ranks(data, reference, r, n_resamples) for cell, r in ranks.items() if cell != "history1_candidate1"},
        "elapsed_s": time.time() - start, "test_predictions_loaded": False,
        "interpretation": "Artificial frozen-network sensitivity, potentially off distribution; not a controlled retraining effect or causal explanation of ID co-adaptation.",
    }
    if not baseline_check["replay_validated"]:
        result["interpretation"] += " Source replay is unvalidated: preserve current probe contrasts, but do not interpret them as validated explanations of the original recorded failures."
    (output_dir / "probe.json").write_text(json.dumps(result, indent=2))
    return result


def main(run_id: str, phase: str = "pilot", arm: str = "both", seed: int | None = None,
         device: str = "auto", sample: bool = False, data_config: str = "configs/data.yaml",
         summarize_only: bool = False, probe_only: bool = False, probe_run_dir: str | None = None,
         text_reference: str | None = None, n_resamples: int = 1000) -> dict:
    if phase not in PHASE_SEEDS or arm not in (*ARMS, "both"):
        raise ValueError("Unknown study phase/arm")
    if seed is not None and seed not in PHASE_SEEDS[phase]:
        raise ValueError("Pilot uses seed 0; seeds 1/2 require explicit replication phase")
    if summarize_only and probe_only:
        raise ValueError("Choose either summary or probe-only mode")
    if summarize_only:
        return summarize_pairs(run_id, phase, data_config, sample, n_resamples, text_reference)
    output, namespace = study_paths(run_id, sample)
    if probe_only:
        if not probe_run_dir:
            raise ValueError("--probe-only requires --probe-run-dir")
        return run_probe(probe_run_dir, output / "probe", data_config, device, sample, n_resamples)
    if probe_run_dir:
        raise ValueError("Use --probe-only to separate the frozen probe from training jobs")
    if phase == "replication" and not sample:
        pilot_path = output / "pilot_comparison.json"
        if not pilot_path.is_file() or json.loads(pilot_path.read_text())["text_reference_guard"]["status"] != "promising":
            raise ValueError("Replication requires a completed promising pilot comparison against text-only")
    seeds = (seed,) if seed is not None else PHASE_SEEDS[phase]
    arms = tuple(ARMS) if arm == "both" else (arm,)
    planned = [fit_path(run_id, a, s, sample) for s in seeds for a in arms]
    if any(p.exists() for p in planned):
        raise FileExistsError("A planned fit directory already exists; use a fresh study/run ID")
    summaries = []
    for s in seeds:
        initial_hashes = []
        for a in arms:
            try:
                metrics = train_retriever.main(str(REPO_ROOT / ARMS[a]), s, sample, device, splits=("val",),
                                               run_id=f"{namespace}/{a}/seed{s}", data_config=data_config, technical_smoke=sample)
            except Exception as error:
                failure = {"status": "failed", "phase": phase, "arm": a, "seed": s,
                           "error_type": type(error).__name__, "error": str(error), "test_predictions_loaded": False}
                failure_dir = output / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / f"{phase}_{a}_seed{s}_{time.time_ns()}.json").write_text(json.dumps(failure, indent=2))
                metrics_path = fit_path(run_id, a, s, sample) / "metrics.json"
                if metrics_path.exists():
                    old = json.loads(metrics_path.read_text())
                    metrics_path.write_text(json.dumps({**old, **failure}, indent=2))
                raise
            summaries.append({"arm": a, "seed": s, "run_dir": metrics["run_dir"], "initial_state_sha256": metrics["initial_state_sha256"]})
            initial_hashes.append(metrics["initial_state_sha256"])
        if len(set(initial_hashes)) != 1:
            raise ValueError("Control/treatment initialization differs")
    return {"phase": phase, "runs": summaries, "test_predictions_loaded": False,
            "next_step": "Review pilot observations; replication requires an explicit launch. Run --summarize-only after both arms complete."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), default="pilot")
    parser.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--probe-run-dir")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--text-reference", help="Text-only parent directory containing seed0/seed1/seed2")
    parser.add_argument("--n-resamples", type=int, default=1000)
    print(json.dumps(main(**vars(parser.parse_args())), indent=2))
