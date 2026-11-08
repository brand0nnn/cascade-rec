"""Export candidate lists and ranker top 10s for a fixed sample of Val purchases (for list dissection).

Systems (seed 0, Val-selected checkpoints, rebuilt exactly as in run_hybrid_study.py):
  original  MLP on two-tower lists (two-tower score)       best  DIN on mixed lists (combined RRF score)
Sample: 20,000 Val purchases (RNG 0 over list rows; same rows for both systems; ALL purchases, hit or miss).
Writes results/hybrid_study/top10-export-20261108/{original,best}.npz with inst, cand [n,200], top10 [n,10], target_pos.
Validation only; no Test.

    python scripts/export_top10_lists.py --device cuda
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from cascade.data.instances import ProcessedData
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT

ORIG = "data/processed/candidates/two_tower_text_seed0"
P3RRF = "data/processed/candidates/p3rrf_seed0"
REF_CFG = "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml"
SYSTEMS = {"original": ("mlp", ORIG, "hybrid-20261107-s0/mlp_tt_seed0"), "best": ("din", P3RRF, "rrf-20261108-s0/din_rrf_seed0")}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    out = REPO_ROOT / "results/hybrid_study/top10-export-20261108"; out.mkdir(parents=True, exist_ok=True)
    data = ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"))
    builder = frozen_feature_builder(data, load_candidates(REPO_ROOT / ORIG, "ranker_train"), REPO_ROOT / REF_CFG)
    base = (builder.score_mean, builder.score_std)
    rt = load_candidates(REPO_ROOT / P3RRF, "ranker_train")
    sc = np.asarray(rt["score"][:, :-1], np.float32)[rt["cand"][:, :-1] > 0]; rrf = (float(sc.mean()), float(sc.std())); del rt
    rows = None
    for name, (kind, pool, ck) in SYSTEMS.items():
        lists = load_candidates(REPO_ROOT / pool, "val", k1=200, force=False)
        if rows is None:
            rows = np.sort(np.random.default_rng(0).choice(len(lists["inst"]), a.n, replace=False)); inst0 = lists["inst"][rows]
        assert np.array_equal(lists["inst"][rows], inst0)
        builder.score_mean, builder.score_std = rrf if pool == P3RRF else base
        spec = RankerFeatureSpec(cardinalities=builder.item_cardinalities, history_length=50, item_text_dim=384, feature_groups=("retrieval", "matching"))
        model = create_ranker(spec, {"model": kind, "dropout": 0.1, "attention_hidden_dims": (64, 32), "attention_use_softmax": False})
        model.load_state_dict(torch.load(REPO_ROOT / "results/hybrid_study" / ck / "model.pt", map_location="cpu")["state_dict"])
        model.to(a.device).eval()
        top = np.zeros((len(rows), 10), np.int32)
        with torch.inference_mode():
            for s in range(0, len(rows), 40):
                rr = rows[s:s + 40]; cand = lists["cand"][rr]; valid = cand > 0
                pr = np.repeat(rr, cand.shape[1])[valid.ravel()]; pc = np.tile(np.arange(cand.shape[1]), len(rr))[valid.ravel()]
                feats = builder.build(lists, pr, pc)
                v = model({k: torch.as_tensor(x, device=a.device) for k, x in feats.items()}).float().cpu().numpy()
                scores = np.full(cand.shape, -np.inf, np.float32); scores[valid] = v
                top[s:s + 40] = np.take_along_axis(cand, np.argsort(-scores, 1, kind="stable")[:, :10], 1)
        np.savez_compressed(out / f"{name}.npz", inst=inst0, cand=lists["cand"][rows].astype(np.int32), top10=top, target_pos=lists["target_pos"][rows])
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
