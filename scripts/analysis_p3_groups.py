"""Seed-averaged nDCG@10 by product group with paired user-bootstrap CIs, for the hybrid and P3 ranker studies.

Per purchase, nDCG@10 is averaged over seeds 0/1/2, then per-user macro averages; differences use
1,000 paired user resamples. Groups from the policy pilot (warm-head = popular, warm-tail = niche,
new-in-period = launched during Val). Validation only; no Test.

    .venv/bin/python scripts/analysis_p3_groups.py results/targeted_study/p3-groups-20261108
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ROOT / "results/ranker_policy/policy-20261106-pilot/validation_groups.parquet"
RUNS = {"hybrid": "hybrid-20261107-s{}", "p3": "p3-20261108-s{}", "rrf": "rrf-20261108-s{}"}
ARMS = {"mlp_tt": "hybrid", "din_tt": "hybrid", "mlp_hyb": "hybrid", "din_hyb": "hybrid",
        "mlp_p3": "p3", "din_p3": "p3", "mlp_p3li": "p3", "din_p3li": "p3",
        "mlp_rrf": "rrf", "din_rrf": "rrf", "mlp_rrf_res": "rrf", "din_rrf_res": "rrf"}
PAIRS = [("mlp_tt", "mlp_p3"), ("din_tt", "din_p3"), ("mlp_p3", "din_p3"), ("mlp_p3", "mlp_p3li"),
         ("din_p3", "din_p3li"), ("din_p3", "din_hyb"), ("mlp_hyb", "mlp_p3"),
         ("mlp_p3", "mlp_rrf"), ("din_p3", "din_rrf"), ("mlp_rrf", "din_rrf"), ("mlp_rrf", "mlp_rrf_res"),
         ("din_rrf", "din_rrf_res"), ("din_tt", "din_rrf"), ("mlp_tt", "din_rrf")]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    ref, vals = None, {}
    for arm, run in ARMS.items():
        per_seed = []
        for s in (0, 1, 2):
            z = np.load(ROOT / "results/hybrid_study" / RUNS[run].format(s) / f"{arm}_seed{s}" / "ranks.npz")
            if ref is None:
                ref = (z["inst"], z["user"])
            if not np.array_equal(z["inst"], ref[0]):
                raise ValueError(f"{arm} seed {s}: instances not aligned")
            per_seed.append(M.ndcg_at(z["ranks"], 10))
        vals[arm] = np.mean(per_seed, 0)
    inst, users = ref
    g = pd.read_parquet(GROUPS).set_index("inst").loc[inst].item_type.to_numpy()
    masks = {"all": np.ones(len(inst), bool), **{k: g == k for k in ("warm-head", "warm-tail", "new-in-period")}}
    lv = pd.DataFrame([dict(arm=a, group=k, ndcg10=average(v[m], users[m]), events=int(m.sum()))
                       for a, v in vals.items() for k, m in masks.items()])
    lv.to_csv(out / "levels.csv", index=False)
    rows = []
    for a, b in PAIRS:
        for k, m in masks.items():
            ci = paired_bootstrap(users[m], vals[a][m], vals[b][m], n_resamples=1000, seed=0)
            base = average(vals[a][m], users[m])
            rows.append(dict(a=a, b=b, group=k, diff=ci["diff"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                             rel=ci["diff"] / base if base else np.nan))
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(lv.pivot(index="arm", columns="group", values="ndcg10").mul(1000).round(3).to_string())
    print(cmp.round(5).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
