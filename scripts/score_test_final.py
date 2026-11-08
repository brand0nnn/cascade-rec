"""Final Test run (user-authorised 8 Nov 2026): score frozen Val-selected rankers on natural Test lists, once.

No training, no tuning, no model or setting chosen on Test. Systems and predictions were fixed beforehand in
results/test_final_ranker/20261108/predictions.md. Each model is rebuilt exactly as in run_hybrid_study.py
(same feature builder fitted on the original ranker-training lists; RRF score standardisation refitted on the
p3rrf training lists), its Val-selected checkpoint is loaded, and every Test list is scored.

  systems: mlp_tt, din_tt (two-tower lists, two-tower score); mlp_rrf, din_rrf, din_rrf_res (mixed lists, combined score)

    python scripts/score_test_final.py --out results/test_final_ranker/20261108 --device cuda
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_k1_sweep import score_prefixes
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, git_hash

ORIG = "data/processed/candidates/two_tower_text_seed0"
P3RRF = "data/processed/candidates/p3rrf_seed0"
TEST = {"tt": "data/processed/candidates/tt_test_seed0", "rrf": "data/processed/candidates/p3rrf_test_seed0"}
REF_CFG = "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml"
SYSTEMS = {"mlp_tt": ("mlp", "tt", "hybrid-20261107-s{}"), "din_tt": ("din", "tt", "hybrid-20261107-s{}"),
           "mlp_rrf": ("mlp", "rrf", "rrf-20261108-s{}"), "din_rrf": ("din", "rrf", "rrf-20261108-s{}"),
           "din_rrf_res": ("din", "rrf", "rrf-20261108-s{}")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    if not (out / "predictions.md").exists():
        raise RuntimeError("predictions.md must be written before Test is opened")
    data = ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"))
    orig_train = load_candidates(REPO_ROOT / ORIG, "ranker_train")
    builder = frozen_feature_builder(data, orig_train, REPO_ROOT / REF_CFG)
    base = (builder.score_mean, builder.score_std)
    rtrain = load_candidates(REPO_ROOT / P3RRF, "ranker_train")
    sc = np.asarray(rtrain["score"][:, :-1], np.float32)[rtrain["cand"][:, :-1] > 0]
    rrf_stats = (float(sc.mean()), float(sc.std()))
    del orig_train, rtrain
    tests = {k: load_candidates(REPO_ROOT / v, "test", k1=200, force=False) for k, v in TEST.items()}
    if not np.array_equal(tests["tt"]["inst"], tests["rrf"]["inst"]):
        raise ValueError("Test populations differ between pools")
    (out / "test_opened.json").write_text(json.dumps({"opened": datetime.now(timezone.utc).isoformat(),
                                                      "git_hash": git_hash(), "systems": list(SYSTEMS)}, indent=2))
    summary = []
    for name, (kind, pool, run) in SYSTEMS.items():
        lists = tests[pool]
        builder.score_mean, builder.score_std = rrf_stats if pool == "rrf" else base
        builder.last_item_cosine = False
        for seed in (0, 1, 2):
            spec = RankerFeatureSpec(cardinalities=builder.item_cardinalities, history_length=50, item_text_dim=384,
                                     feature_groups=("retrieval", "matching"), extra_numeric=0)
            cfg = {"model": kind, "dropout": 0.1, "attention_hidden_dims": (64, 32), "attention_use_softmax": False,
                   "retrieval_score_input": "enabled", "residual_score": name.endswith("_res")}
            model = create_ranker(spec, cfg)
            ck = torch.load(REPO_ROOT / "results/hybrid_study" / run.format(seed) / f"{name}_seed{seed}" / "model.pt",
                            map_location="cpu")
            model.load_state_dict(ck["state_dict"])
            model.to(a.device).eval()
            ranks, _ = score_prefixes(model, builder, lists, [200], device=a.device, batch_pairs=8192)
            np.savez_compressed(out / f"{name}_seed{seed}_test_ranks.npz", inst=lists["inst"], user=lists["user"],
                                ranks=ranks[0], target_pos=lists["target_pos"])
            nd = M.macro_micro(M.ndcg_at(ranks[0], 10), lists["user"])["macro"]
            summary.append({"system": name, "seed": seed, "test_ndcg10_macro": nd, "val_epoch": ck["epoch"]})
            print(json.dumps(summary[-1]), flush=True)
    (out / "summary.json").write_text(json.dumps({"rows": summary, "recall_at_200": {
        k: float((v["target_pos"] >= 0).mean()) for k, v in tests.items()}, "n_lists": int(len(tests["tt"]["inst"]))}, indent=2))


if __name__ == "__main__":
    main()
