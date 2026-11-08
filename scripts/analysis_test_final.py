"""Final Test analysis (opened once, 8 Nov 2026): seed-averaged nDCG@10 by item group with paired user bootstrap.

Groups as on Val but at Test entry: new-in-period = target first seen at/after T2 (equals cold on Test);
otherwise warm-head / warm-tail by the project segment (counts before T2). Compares against predictions.md.

    .venv/bin/python scripts/analysis_test_final.py results/test_final_ranker/20261108
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average

SYSTEMS = ["mlp_tt", "din_tt", "mlp_rrf", "din_rrf", "din_rrf_res"]
PAIRS = [("mlp_tt", "din_rrf"), ("mlp_rrf", "din_rrf"), ("din_rrf_res", "din_rrf"), ("mlp_tt", "mlp_rrf"), ("din_tt", "din_rrf"), ("mlp_tt", "din_tt")]


def main(d: Path) -> None:
    data = ProcessedData.load("configs/data.yaml")
    vals, ref, cov = {}, None, {}
    for s in SYSTEMS:
        r = []
        for seed in (0, 1, 2):
            z = np.load(d / f"{s}_seed{seed}_test_ranks.npz")
            ref = z if ref is None else ref
            assert np.array_equal(z["inst"], ref["inst"])
            r.append(M.ndcg_at(z["ranks"], 10))
            cov[s] = (z["target_pos"] >= 0).mean()
        vals[s] = np.mean(r, 0)
    inst = data.instances.set_index("inst").loc[ref["inst"]]
    it = data.items.set_index("item").loc[inst["item"].to_numpy()]
    new = it.first_ts.to_numpy() >= data.split["t2"]
    g = np.where(new, "new-in-period", np.where(it.segment.astype(str).to_numpy() == "head", "warm-head", "warm-tail"))
    users = ref["user"]
    masks = {"all": np.ones(len(g), bool), **{k: g == k for k in ("warm-head", "warm-tail", "new-in-period")}}
    lv = pd.DataFrame([dict(system=s, group=k, ndcg10=average(v[m], users[m]), share=m.mean(), coverage=cov[s])
                       for s, v in vals.items() for k, m in masks.items()])
    rows = []
    for a, b in PAIRS:
        for k, m in masks.items():
            ci = paired_bootstrap(users[m], vals[a][m], vals[b][m], n_resamples=1000, seed=0)
            base = average(vals[a][m], users[m])
            rows.append(dict(a=a, b=b, group=k, diff=ci["diff"], ci_low=ci["ci_low"], ci_high=ci["ci_high"], rel=ci["diff"] / base))
    cmp = pd.DataFrame(rows)
    lv.to_csv(d / "levels.csv", index=False); cmp.to_csv(d / "comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(lv.pivot(index="system", columns="group", values="ndcg10").mul(1000).round(3).to_string())
    print({k: round(m.mean(), 3) for k, m in masks.items()}, {s: round(c, 4) for s, c in cov.items()})
    print(cmp.round(5).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
