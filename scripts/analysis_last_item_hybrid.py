"""Combine the two-tower retriever with Brandon's last-item scorer; plus two launch/loyalty checks.

Uses saved full-catalogue Val ranks (results/analysis/20261028-last-item/aligned_val_ranks.npz):
`frozen` = Two-tower (text) seed 0, `last_item` = cosine to the user's most recent product (no training).
Validation only; no Test; no model training.

Hybrid candidate pool of 200 = top-a two-tower + top-(200-a) last-item. A purchase is counted as covered
only if it is within the top-a of one OR the top-(200-a) of the other. Overlaps between sources would let a
real deduplicated pool reach further, so this is a conservative lower bound on hybrid recall.
Hybrid top-10 (no ranker) is treated the same way with a/(10-a) slots.

Checks:
  launch_titles  share of Val-new products whose title mentions iPhone 13 / Galaxy S22, by first-seen week
  brand_loyalty  share of Val purchases whose store already appears in the user's prior history, by item group,
                 and which rankers hit those purchases (seed-0 K200 ranks)

Usage: .venv/bin/python scripts/analysis_last_item_hybrid.py OUTPUT_DIR
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

ROOT = Path(__file__).resolve().parents[1]
LAST = ROOT / "results/analysis/20261028-last-item/aligned_val_ranks.npz"
PILOT = ROOT / "results/ranker_policy/policy-20261106-pilot"
RQ3 = ROOT / "results/ranker_comparison/20261025-rq3/validation_ranks.npz"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    z = np.load(LAST)
    groups = pd.read_parquet(PILOT / "validation_groups.parquet").set_index("inst").loc[z["inst"]]
    users = z["user"]
    tt, li = z["frozen"].astype(float), z["last_item"].astype(float)
    masks = {"all": np.ones(len(tt), bool)}
    masks.update({g: groups.item_type.to_numpy() == g for g in ("warm-head", "warm-tail", "new-in-period")})

    rows = []
    for budget in (10, 200):
        for a in sorted({budget, int(budget * .9), int(budget * .75), budget // 2, int(budget * .25), 0}):
            hit = (tt <= a) | (li <= budget - a)
            base = tt <= budget
            for g, m in masks.items():
                ci = paired_bootstrap(users[m], base[m].astype(float), hit[m].astype(float), n_resamples=1000, seed=0) if a != budget else None
                rows.append(dict(budget=budget, two_tower_slots=a, last_item_slots=budget - a, group=g,
                                 recall=average(hit[m].astype(float), users[m]),
                                 diff_vs_two_tower=None if ci is None else ci["diff"],
                                 ci_low=None if ci is None else ci["ci_low"], ci_high=None if ci is None else ci["ci_high"]))
    hyb = pd.DataFrame(rows)
    hyb.to_csv(out / "hybrid_recall.csv", index=False)

    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    t1, t2 = int(data.split["t1"]), int(data.split["t2"])
    items = data.items.set_index("item")
    new = items[(items.first_ts >= t1) & (items.first_ts < t2)].copy()
    title = new.title.fillna("").str.lower()
    new["iphone13"] = title.str.contains(r"iphone\s*13", regex=True)
    new["s22"] = title.str.contains(r"\bs22\b", regex=True)
    new["week"] = pd.to_datetime(new.first_ts, unit="ms").dt.to_period("W").dt.start_time
    weekly = new.groupby("week").agg(new_products=("iphone13", "size"), iphone13=("iphone13", "sum"), s22=("s22", "sum"))
    weekly.to_csv(out / "launch_titles_weekly.csv")

    val = data.split_instances("val").set_index("inst").loc[z["inst"]]
    store = items.store.fillna("").to_numpy() if False else items.store.reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    loyal = np.zeros(len(val), bool)
    for i, (u, ts, it) in enumerate(zip(val.user.to_numpy(), val.ts.to_numpy(), val["item"].to_numpy())):
        s = slice(data.user_ptr[u], data.user_ptr[u + 1])
        hist = data.seq_item[s][data.seq_ts[s] < ts][-50:]
        st = store[it]
        loyal[i] = bool(st) and st in set(store[hist])
    with np.load(PILOT / "arm_A_seed0/prefix_ranks.npz") as a, np.load(PILOT / "arm_C_seed0/prefix_ranks.npz") as c, np.load(RQ3) as r:
        order = pd.Series(np.arange(len(r["inst"])), index=r["inst"]).loc[z["inst"]].to_numpy()
        ranks = {"forced_mlp": a["ranks"][2], "natural_mlp": c["ranks"][2], "din_forced": r["R3"][0][order],
                 "two_tower_order": tt, "last_item_order": li}
    lrows = []
    for g, m in masks.items():
        for flag, fm in (("same_store_as_history", loyal), ("new_store", ~loyal)):
            mm = m & fm
            row = dict(group=g, purchase=flag, events=int(mm.sum()), share_of_group=float(mm.sum() / m.sum()))
            for name, rk in ranks.items():
                row[f"{name}_recall10"] = average((rk[mm] <= 10).astype(float), users[mm])
            lrows.append(row)
    loy = pd.DataFrame(lrows)
    loy.to_csv(out / "brand_loyalty.csv", index=False)

    summary = {"new_products": int(len(new)), "iphone13_titles": int(new.iphone13.sum()),
               "s22_titles": int(new.s22.sum()), "test_predictions_loaded": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(hyb.round(5).to_string()); print(summary)
    print(weekly.sort_values("new_products", ascending=False).head(6).to_string())
    print(loy.round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
