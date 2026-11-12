"""Score the fast-combined-score DIN (din_rrfl, Val-selected, seeds 0/1/2) once on Test; prediction 6 in predictions.md.

Rebuilds the model as run_hybrid_study.py does (feature builder fitted on the original training lists; score
standardisation refitted on the p3rrfl natural-only training lists), loads each Val-selected checkpoint and scores the p3rrfl Test lists.

    python scripts/score_test_rrfl.py --device cuda --out results/test_final_ranker/FRESH_ID
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.pipelines.matching_recipe import matching_recipe
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.hybrid_normalization import natural_score_stats
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT

POOL = "data/processed/candidates/p3rrfl_seed0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="fresh output directory; original results are preserved")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"))
    builder = frozen_feature_builder(data, load_candidates(REPO_ROOT / "data/processed/candidates/two_tower_text_seed0", "ranker_train"),
                                     REPO_ROOT / "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml")
    tr = load_candidates(REPO_ROOT / POOL, "ranker_train")
    builder.score_mean, builder.score_std = natural_score_stats(tr)
    (out / "score_normalization.json").write_text(json.dumps({
        "score_mean": builder.score_mean, "score_std": builder.score_std,
        "population": "natural_only ranker training", "pool": POOL}, indent=2))
    del tr
    test = load_candidates(REPO_ROOT / POOL, "test", k1=200, force=False)
    rows = []
    for seed in (0, 1, 2):
        spec, cfg = matching_recipe(builder.item_cardinalities, "din")
        m = create_ranker(spec, cfg)
        m.load_state_dict(torch.load(REPO_ROOT / f"results/hybrid_study/rrfl-20261108-s{seed}/din_rrfl_seed{seed}/model.pt", map_location="cpu")["state_dict"])
        ranks, _ = score_prefixes(m.to(a.device).eval(), builder, test, [200], device=a.device, batch_pairs=8192)
        np.savez_compressed(out / f"din_rrfl_seed{seed}_test_ranks.npz", inst=test["inst"], user=test["user"], ranks=ranks[0], target_pos=test["target_pos"])
        rows.append({"seed": seed, "test_ndcg10_macro": M.macro_micro(M.ndcg_at(ranks[0], 10), test["user"])["macro"]})
        print(json.dumps(rows[-1]), flush=True)
    (out / "summary_rrfl.json").write_text(json.dumps({"rows": rows, "recall_at_200": float((test["target_pos"] >= 0).mean())}, indent=2))


if __name__ == "__main__":
    main()
