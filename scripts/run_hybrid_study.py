"""Natural-list ranker study: hybrid candidate pool (two-tower + last-item) and a last-item similarity feature.

Arms (all natural-only training, retrieval + matching features, BCE, 50 negatives, Adam 1e-3,
max 30 epochs, patience 3 on natural-hit Val nDCG@10, instance-keyed sampling, same init per seed):
  mlp_tt     MLP on the original two-tower K200 pool (control)
  mlp_hyb    MLP on the hybrid pool
  mlp_tt_li  MLP on the original pool + cosine(candidate text, last product text) feature
  din_tt     DIN on the original pool (control)
  din_hyb    DIN on the hybrid pool
  mlp_p3 / din_p3      MLP / DIN on the rebuilt P3 pool (recency + two-tower round-robin, Brandon's candidate set)
  mlp_p3li / din_p3li  same with the last 20 P3 slots replaced by last-item products
  mlp_rrf / din_rrf    P3 lists with Brandon's RRF retrieval score (both retrievers) as the score input
  din_rrfl   DIN on P3 lists with the FAST combined score (positions within each retriever's top-200 list)
  mlp_rrf_res / din_rrf_res  same, scored as retrieval score + zero-initialised correction (Brandon's ranker design)
For RRF pools the score standardisation is refitted on that pool's natural training lists (the RRF scale differs).
Feature normalisation and vocabularies are fitted on the ORIGINAL full ranker-training lists, unchanged.
End-to-end Val metric = rank of the target in that arm's own natural 200-candidate list (misses score 0).
Validation only; no Test.

    python scripts/run_hybrid_study.py --run-id hybrid-20261107-s0 --seeds 0 --device cuda [--sample-users 0.02 --max-epochs 1]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.pipelines.ranker_policy_study import frozen_feature_builder, select_training_lists
from cascade.pipelines.ranker_training import create_ranker, fit_candidate_lists
from cascade.utils.io import REPO_ROOT, git_hash
from cascade.utils.seed import set_seed

ORIG = "data/processed/candidates/two_tower_text_seed0"
HYB = "data/processed/candidates/hybrid_tt180_li20_seed0"
P3 = "data/processed/candidates/p3_seed0"
P3LI = "data/processed/candidates/p3li_seed0"
import os
P3RRF = "data/processed/candidates/p3rrf_seed0" + os.environ.get("P3RRF_SUFFIX", "")
P3RRFL = "data/processed/candidates/p3rrfl_seed0"
REF_CFG = "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml"
ARMS = {"mlp_tt": ("mlp", ORIG, 0), "mlp_hyb": ("mlp", HYB, 0), "mlp_tt_li": ("mlp", ORIG, 1),
        "din_tt": ("din", ORIG, 0), "din_hyb": ("din", HYB, 0),
        "mlp_p3": ("mlp", P3, 0), "din_p3": ("din", P3, 0), "mlp_p3li": ("mlp", P3LI, 0), "din_p3li": ("din", P3LI, 0),
        "mlp_rrf": ("mlp", P3RRF, 0), "din_rrf": ("din", P3RRF, 0),
        "mlp_rrf_res": ("mlp", P3RRF, 0), "din_rrf_res": ("din", P3RRF, 0),
        "din_rrfl": ("din", P3RRFL, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample-users", type=float, help="development: fraction of users (fixed RNG 0)")
    ap.add_argument("--max-epochs", type=int, default=30)
    a = ap.parse_args()
    root = REPO_ROOT / "results/hybrid_study" / a.run_id
    root.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"))
    orig_train = load_candidates(REPO_ROOT / ORIG, "ranker_train")
    builder = frozen_feature_builder(data, orig_train, REPO_ROOT / REF_CFG)
    universe = np.asarray(orig_train["inst"]).copy()
    users = None
    if a.sample_users:
        users = np.random.default_rng(0).choice(np.arange(1, data.n_users + 1),
                                                max(1, round(data.n_users * a.sample_users)), replace=False)
    pools = {}
    for pool in {ARMS[x][1] for x in a.arms}:
        train = select_training_lists(load_candidates(REPO_ROOT / pool, "ranker_train"), "natural_only", users)
        train.pop("source_last_item", None)
        val_sel = load_candidates(REPO_ROOT / pool, "val", k1=200, force=True)
        val_nat = load_candidates(REPO_ROOT / pool, "val", k1=200, force=False)
        for d in (val_sel, val_nat):
            d.pop("source_last_item", None)
        if users is not None:
            m = np.isin(val_nat["user"], users)
            val_sel = {k: v[m] for k, v in val_sel.items()}
            val_nat = {k: v[m] for k, v in val_nat.items()}
        if not np.isin(train["inst"], universe).all():
            raise ValueError("training instances outside the original universe")
        pools[pool] = (train, val_sel, val_nat)
    ref_inst = next(iter(pools.values()))[2]["inst"]
    if any(not np.array_equal(p[2]["inst"], ref_inst) for p in pools.values()):
        raise ValueError("validation populations differ between pools")
    status = {"started": datetime.now(timezone.utc).isoformat(), "completed": [], "git_hash": git_hash(),
              "arms": {k: ARMS[k] for k in a.arms}, "seeds": a.seeds, "sample_users": a.sample_users,
              "test_predictions_loaded": False}
    (root / "status.json").write_text(json.dumps(status, indent=2))
    base_stats = (builder.score_mean, builder.score_std)
    for seed in a.seeds:
        for arm in a.arms:
            kind, pool, extra = ARMS[arm]
            train, val_sel, val_nat = pools[pool]
            builder.last_item_cosine = bool(extra)
            if pool in (P3RRF, P3RRFL):
                sc = np.asarray(train["score"][:, :-1], np.float32)[train["cand"][:, :-1] > 0]
                builder.score_mean, builder.score_std = float(sc.mean()), float(sc.std())
            else:
                builder.score_mean, builder.score_std = base_stats
            spec = RankerFeatureSpec(cardinalities=builder.item_cardinalities, history_length=50, item_text_dim=384,
                                     feature_groups=("retrieval", "matching"), extra_numeric=extra)
            cfg = {"model": kind, "dropout": 0.1, "attention_hidden_dims": (64, 32), "attention_use_softmax": False,
                   "retrieval_score_input": "enabled", "residual_score": arm.endswith("_res")}
            set_seed(seed)
            model = create_ranker(spec, cfg)
            out = root / f"{arm}_seed{seed}"
            out.mkdir()
            result = fit_candidate_lists(model, builder, train, val_sel, n_negatives=50, batch_size=2048,
                                         learning_rate=0.001, max_epochs=a.max_epochs, patience=3, seed=seed,
                                         device=a.device, checkpoint_path=out / "model.pt",
                                         sampling_strategy="instance_keyed", training_instance_universe=universe)
            ranks, _ = score_prefixes(model, builder, val_nat, [200], device=a.device, batch_pairs=8192)
            np.savez_compressed(out / "ranks.npz", inst=val_nat["inst"], user=val_nat["user"], ranks=ranks[0],
                                target_pos=val_nat["target_pos"])
            ndcg = M.macro_micro(M.ndcg_at(ranks[0], 10), val_nat["user"])["macro"]
            metrics = {"arm": arm, "seed": seed, "model": kind, "pool": pool, "extra_numeric": extra,
                       "best_epoch": result.best_epoch, "epochs_run": len(result.history),
                       "train_lists": int(len(train["inst"])), "val_recall_at_200": float((val_nat["target_pos"] >= 0).mean()),
                       "end_to_end_ndcg10_macro": ndcg, "spec": asdict(spec), "test_predictions_loaded": False}
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            status["completed"].append({"arm": arm, "seed": seed, "ndcg10": ndcg})
            (root / "status.json").write_text(json.dumps(status, indent=2))
            print(json.dumps({"arm": arm, "seed": seed, "ndcg10": ndcg, "best_epoch": result.best_epoch}), flush=True)
    status["status"] = "complete"
    (root / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
