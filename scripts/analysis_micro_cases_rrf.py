"""Micro cases for the final systems: individual Val purchases with history titles and each system's target rank.

Uses seed-0 ranks of mlp_tt (original) and din_rrf (best) plus labels from analysis_meso_segments_rrf.py.
Picks 2 random cases (fixed RNG 0) per story; prints history (last 5 titles), target title, ranks.
Validation only; no Test.

    .venv/bin/python scripts/analysis_micro_cases_rrf.py results/targeted_study/meso-segments-rrf-20261108
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData

ROOT = Path(__file__).resolve().parents[1]


def main(seg: Path) -> None:
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    lab = pd.read_parquet(seg / "labels.parquet")
    r = {n: np.load(ROOT / "results/hybrid_study" / run / f"{n}_seed0" / "ranks.npz")["ranks"]
         for n, run in (("mlp_tt", "hybrid-20261107-s0"), ("din_rrf", "rrf-20261108-s0"))}
    inst = data.instances.set_index("inst").loc[lab.inst].reset_index()
    titles = data.items.set_index("item").title.fillna("")
    hist = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    tgt = inst["item"].to_numpy()
    ttl = lambda i: titles.get(int(i), "")[:90]
    stories = {
        "upgrade to newer iPhone, launch month, best system finds it, original misses":
            (lab["iPhone upgrade"] == "newer iPhone model") & (lab.period == "iPhone 13 launch month") & (r["din_rrf"] <= 10) & (r["mlp_tt"] > 200),
        "same iPhone model, case: best system top 3, original misses":
            (lab["iPhone upgrade"] == "same iPhone model") & (lab["product type"] == "case") & (r["din_rrf"] <= 3) & (r["mlp_tt"] > 50),
        "screen protector: original top 3, best system drops it":
            (lab["product type"] == "screen protector") & (r["mlp_tt"] <= 3) & (r["din_rrf"] > 10),
        "platform switch: both miss":
            (lab["user vs product"] == "switch platform") & (r["mlp_tt"] > 200) & (r["din_rrf"] > 200),
    }
    rng = np.random.default_rng(0)
    rows = []
    for story, m in stories.items():
        idx = np.flatnonzero(m)
        print(f"\n=== {story}: {len(idx)} purchases ===")
        for i in rng.choice(idx, min(2, len(idx)), replace=False):
            h = [x for x in hist[i] if x > 0][-5:]
            date = pd.to_datetime(inst.ts[i], unit="ms").date()
            print(f"- {date}, user {inst.user[i]}, history {inst.hist_len[i]} purchases")
            for x in h:
                print("    had:", ttl(x))
            print("  BOUGHT:", ttl(tgt[i]))
            print(f"  rank: original {r['mlp_tt'][i]:.0f}, best {r['din_rrf'][i]:.0f}")
            rows.append(dict(story=story, inst=int(lab.inst[i]), date=str(date), history=" | ".join(ttl(x) for x in h),
                             bought=ttl(tgt[i]), rank_original=float(r["mlp_tt"][i]), rank_best=float(r["din_rrf"][i])))
    pd.DataFrame(rows).to_csv(seg / "micro_cases.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
