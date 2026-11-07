"""Deployable 'router': merge two rankers' saved top-10 lists (no target information used).

final top 10 = first k distinct products of ranker X, then fill from ranker Y's list in order (skipping
duplicates), then from X again if still short. Inputs: seed-0 top-10 lists saved by
analysis_top10_diversity.py over all natural K200 Validation lists. Also a user-history router:
users with <= H prior purchases get list X, others list Y (history length is known before the purchase).
Validation only; no Test; no training.

Usage: .venv/bin/python scripts/analysis_list_merge.py OUTPUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average
from cascade.pipelines.build_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "results/targeted_study/top10-diversity-20261107"
GROUPS = ROOT / "results/ranker_policy/policy-20261106-pilot/validation_groups.parquet"


def merge(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(len(x)):
        seen, row = set(), []
        for src in (x[i, :k], y[i], x[i, k:]):
            for item in src:
                if item > 0 and item not in seen:
                    seen.add(item); row.append(item)
                if len(row) == 10:
                    break
            if len(row) == 10:
                break
        out[i, :len(row)] = row
    return out


def ndcg(top: np.ndarray, target: np.ndarray) -> np.ndarray:
    hit = top == target[:, None]
    pos = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    return np.where(pos > 0, 1 / np.log2(pos + 1), 0.0)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    lists = load_candidates(ROOT / "data/processed/candidates/two_tower_text_seed0", "val", k1=500, force=False)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    inst = data.instances.set_index("inst").loc[lists["inst"]]
    target, users, hist = inst["item"].to_numpy(), inst["user"].to_numpy(), inst["hist_len"].to_numpy()
    g = pd.read_parquet(GROUPS).set_index("inst").loc[lists["inst"]].item_type.to_numpy()
    nat = np.load(TOP / "top10_C_natural_mlp.npy"); din = np.load(TOP / "top10_DIN_forced.npy")
    forced = np.load(TOP / "top10_A_forced_mlp.npy")
    masks = {"all": np.ones(len(g), bool), **{k: g == k for k in ("warm-head", "warm-tail", "new-in-period")}}
    base = ndcg(nat, target)
    systems = {"natural_mlp": nat, "din_forced": din, "forced_mlp": forced}
    for k in (8, 7, 5, 3):
        systems[f"natural{k}+din{10 - k}"] = merge(nat, din, k)
    for h in (2, 5):
        systems[f"route_hist<={h}_natural_else_din"] = np.where((hist <= h)[:, None], nat, din)
    rows = []
    for name, top in systems.items():
        v = ndcg(top, target)
        for grp, m in masks.items():
            ci = paired_bootstrap(users[m], base[m], v[m], n_resamples=1000, seed=0) if name != "natural_mlp" else None
            rows.append(dict(system=name, group=grp, ndcg10=average(v[m], users[m]),
                             diff_vs_natural=None if ci is None else ci["diff"],
                             ci_low=None if ci is None else ci["ci_low"], ci_high=None if ci is None else ci["ci_high"]))
    res = pd.DataFrame(rows)
    res.to_csv(out / "merge_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"seed": 0, "lists": int(len(target)), "test_predictions_loaded": False}))
    pd.set_option("display.width", 200)
    print(res.pivot(index="system", columns="group", values="ndcg10").mul(100).round(3).to_string())
    print(res[res.group == "all"][["system", "diff_vs_natural", "ci_low", "ci_high"]].round(5).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
