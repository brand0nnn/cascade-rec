"""Replay frozen Val/Test evidence, then correct the mixed-pool Test scaler.

No fitting or model selection. Old predictions are read only. A small-list smoke
run must pass before a full run; both refuse existing output directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.hybrid_normalization import natural_score_stats
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, git_hash

ORIGINAL = REPO_ROOT / "results/test_final_ranker/20261108"
ARMS = {"mlp_rrf": ("mlp", "rrf"), "din_rrf": ("din", "rrf"),
        "din_rrf_res": ("din", "rrf"), "din_rrfl": ("din", "rrfl")}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-lists", type=int, help="smoke: first N lists, no headline result")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"))
    orig_train = load_candidates(REPO_ROOT / "data/processed/candidates/two_tower_text_seed0", "ranker_train")
    builder = frozen_feature_builder(data, orig_train,
        REPO_ROOT / "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml")
    del orig_train
    report = {"started": datetime.now(timezone.utc).isoformat(), "git_hash": git_hash(),
              "sample_lists": a.sample_lists, "status": "running", "pools": {}, "replays": [],
              "protocol": "frozen Val-selected weights; training-only scaler correction; no fitting or selection",
              "source_hashes": {str(p.relative_to(REPO_ROOT)): sha(p) for p in (
                  Path(__file__).resolve(), REPO_ROOT / "src/cascade/pipelines/hybrid_normalization.py",
                  REPO_ROOT / "scripts/run_hybrid_study.py")}}
    def save():
        (out / "audit.json").write_text(json.dumps(report, indent=2))
    save()
    all_values, old_values, users, inst_ids = {}, {}, None, None
    for pool in ("rrf", "rrfl"):
        candidate_dir = REPO_ROOT / f"data/processed/candidates/p3{pool}_seed0"
        train = load_candidates(candidate_dir, "ranker_train")
        sc = np.asarray(train["score"][:, :-1], np.float32)[train["cand"][:, :-1] > 0]
        old_stats = float(sc.mean()), float(sc.std())
        stats = natural_score_stats(train)
        natural_count = int((~np.asarray(train["forced"], bool)).sum())
        report["pools"][pool] = {"all_lists": len(train["inst"]), "natural_lists": natural_count,
            "old_mean": old_stats[0], "old_std": old_stats[1], "training_mean": stats[0], "training_std": stats[1]}
        print(json.dumps({"pool": pool, **report["pools"][pool]}), flush=True)
        del train, sc
        val = load_candidates(candidate_dir, "val", k1=200, force=False)
        test_dir = REPO_ROOT / "data/processed/candidates/p3rrf_test_seed0" if pool == "rrf" else candidate_dir
        test = load_candidates(test_dir, "test", k1=200, force=False)
        if a.sample_lists:
            val = {k: v[:a.sample_lists] for k, v in val.items()}
            test = {k: v[:a.sample_lists] for k, v in test.items()}
        if inst_ids is not None:
            np.testing.assert_array_equal(inst_ids, test["inst"])
            np.testing.assert_array_equal(users, test["user"])
        users, inst_ids = test["user"], test["inst"]
        for arm, (kind, arm_pool) in ARMS.items():
            if arm_pool != pool:
                continue
            corrected, historical = [], []
            for seed in (0, 1, 2):
                run = REPO_ROOT / f"results/hybrid_study/{pool}-20261108-s{seed}/{arm}_seed{seed}"
                meta = json.loads((run / "metrics.json").read_text())
                if meta["train_lists"] != natural_count:
                    raise ValueError("training population no longer matches saved run")
                spec = RankerFeatureSpec(cardinalities=builder.item_cardinalities, history_length=50,
                    item_text_dim=384, feature_groups=("retrieval", "matching"), extra_numeric=0)
                model = create_ranker(spec, {"model": kind, "dropout": .1, "attention_hidden_dims": (64, 32),
                    "attention_use_softmax": False, "retrieval_score_input": "enabled",
                    "residual_score": arm.endswith("_res")})
                ck_path = run / "model.pt"
                ck_hash = sha(ck_path)
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                if ck["epoch"] != meta["best_epoch"]:
                    raise ValueError("selected epoch mismatch")
                model.load_state_dict(ck["state_dict"], strict=True)
                model.to(a.device).eval()
                builder.score_mean, builder.score_std = stats
                vr, _ = score_prefixes(model, builder, val, [200], device=a.device, batch_pairs=8192)
                with np.load(run / "ranks.npz") as saved:
                    n = len(val["inst"])
                    for key in ("inst", "user", "target_pos"):
                        np.testing.assert_array_equal(val[key], saved[key][:n])
                    np.testing.assert_array_equal(vr[0], saved["ranks"][:n])
                builder.score_mean, builder.score_std = old_stats
                old, _ = score_prefixes(model, builder, test, [200], device=a.device, batch_pairs=8192)
                with np.load(ORIGINAL / f"{arm}_seed{seed}_test_ranks.npz") as saved:
                    n = len(test["inst"])
                    for key in ("inst", "user", "target_pos"):
                        np.testing.assert_array_equal(test[key], saved[key][:n])
                    np.testing.assert_array_equal(old[0], saved["ranks"][:n])
                builder.score_mean, builder.score_std = stats
                ranks, _ = score_prefixes(model, builder, test, [200], device=a.device, batch_pairs=8192)
                np.savez_compressed(out / f"{arm}_seed{seed}_test_ranks.npz", inst=inst_ids, user=users,
                                    ranks=ranks[0], target_pos=test["target_pos"])
                historical.append(M.ndcg_at(old[0], 10))
                corrected.append(M.ndcg_at(ranks[0], 10))
                if sha(ck_path) != ck_hash:
                    raise ValueError("checkpoint changed during replay")
                report["replays"].append({"arm": arm, "seed": seed, "checkpoint_sha256": ck_hash,
                    "val_ranks_exact": True, "old_test_ranks_exact": True, "epoch": ck["epoch"],
                    "old_ndcg10": M.macro_micro(historical[-1], users)["macro"],
                    "corrected_ndcg10": M.macro_micro(corrected[-1], users)["macro"]})
                save()
                print(json.dumps(report["replays"][-1]), flush=True)
            all_values[arm] = np.mean(corrected, axis=0)
            old_values[arm] = np.mean(historical, axis=0)
        del val, test
    base = []
    for seed in (0, 1, 2):
        with np.load(ORIGINAL / f"mlp_tt_seed{seed}_test_ranks.npz") as z:
            n = len(inst_ids)
            np.testing.assert_array_equal(z["inst"][:n], inst_ids)
            np.testing.assert_array_equal(z["user"][:n], users)
            base.append(M.ndcg_at(z["ranks"][:n], 10))
    all_values["mlp_tt"] = np.mean(base, axis=0)
    events = data.instances.set_index("inst").loc[inst_ids]
    items = data.items.set_index("item").loc[events["item"].to_numpy()]
    new = items.first_ts.to_numpy() >= data.split["t2"]
    masks = {"all": np.ones(len(users), bool), "new-in-period": new,
             "warm-head": ~new & (items.segment.astype(str).to_numpy() == "head"),
             "warm-tail": ~new & (items.segment.astype(str).to_numpy() != "head")}
    levels, comparisons = [], []
    for group, mask in masks.items():
        for arm, values in all_values.items():
            levels.append({"system": arm, "group": group,
                "ndcg10": M.macro_micro(values[mask], users[mask])["macro"]})
        for arm, values in old_values.items():
            levels.append({"system": arm + "_historical", "group": group,
                "ndcg10": M.macro_micro(values[mask], users[mask])["macro"]})
        pairs = [("mlp_tt", arm, all_values["mlp_tt"], v) for arm, v in all_values.items() if arm != "mlp_tt"]
        pairs += [(arm + "_historical", arm, old_values[arm], all_values[arm]) for arm in old_values]
        pairs += [("din_rrf", "din_rrfl", all_values["din_rrf"], all_values["din_rrfl"])]
        for an, bn, av, bv in pairs:
            ci = paired_bootstrap(users[mask], av[mask], bv[mask], n_resamples=1000, seed=0)
            base_mean = M.macro_micro(av[mask], users[mask])["macro"]
            comparisons.append({"a": an, "b": bn, "group": group, **ci,
                                "relative": ci["diff"] / base_mean if base_mean else None})
    pd.DataFrame(levels).to_csv(out / "levels.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "comparisons.csv", index=False)
    report.update(status="complete", completed=datetime.now(timezone.utc).isoformat(), n_lists=len(users))
    save()
    print(pd.DataFrame(levels).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
