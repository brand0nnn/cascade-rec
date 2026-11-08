"""Micro level as small event cohorts (tens to a few hundred purchases), original vs best system.

Cohorts are defined by purchase date window + product title pattern (titles are current metadata, so a
pre-launch purchase whose title names the new model is reported as a data-quality check, not a result).
Per cohort: purchases, users, coverage (purchase in the 200 candidates) and seed-averaged nDCG@10 / Hit@10
for original (MLP, two-tower lists) and best (DIN, mixed lists + combined score), paired user bootstrap.
Validation only; no Test.

    .venv/bin/python scripts/analysis_micro_cohorts_rrf.py results/targeted_study/micro-cohorts-rrf-20261108
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average

ROOT = Path(__file__).resolve().parents[1]
SYS = {"orig": ("mlp_tt", "hybrid-20261107-s{}"), "best": ("din_rrf", "rrf-20261108-s{}")}
COHORTS = [
    ("iPhone 13 item, before announcement (title-edit check)", r"iphone\s*13", None, "2021-09-14"),
    ("iPhone 13 item, launch fortnight (14–30 Sep 2021)", r"iphone\s*13", "2021-09-14", "2021-10-01"),
    ("iPhone 13 item, Oct–Dec 2021", r"iphone\s*13", "2021-10-01", "2022-01-01"),
    ("iPhone 13 item, 2022", r"iphone\s*13", "2022-01-01", None),
    ("Galaxy S22 item, launch (9 Feb–15 Mar 2022)", r"\bs22\b", "2022-02-09", "2022-03-16"),
    ("Galaxy S22 item, after launch", r"\bs22\b", "2022-03-16", None),
    ("Apple Watch 41/45mm band, Series 7 launch (15 Oct–30 Nov 2021)", r"(41\s*mm|45\s*mm).*(watch|iwatch)|(watch|iwatch).*(41\s*mm|45\s*mm)", "2021-10-15", "2021-12-01"),
    ("MagSafe accessory", r"magsafe", None, None),
    ("Pixel 6 item (launched 28 Oct 2021)", r"pixel\s*6", "2021-10-28", None),
    ("iPhone SE 2022 item (on sale 18 Mar 2022)", r"se\s*(3|2022|3rd)", "2022-03-18", None),
]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    nd, hit, cov, ref = {}, {}, {}, None
    for k, (arm, run) in SYS.items():
        a, b = [], []
        for s in (0, 1, 2):
            z = np.load(ROOT / "results/hybrid_study" / run.format(s) / f"{arm}_seed{s}" / "ranks.npz")
            ref = z if ref is None else ref
            assert np.array_equal(z["inst"], ref["inst"])
            a.append(M.ndcg_at(z["ranks"], 10)); b.append((z["ranks"] <= 10).astype(float))
            cov[k] = z["target_pos"] >= 0
        nd[k], hit[k] = np.mean(a, 0), np.mean(b, 0)
    inst = data.instances.set_index("inst").loc[ref["inst"]].reset_index()
    users = ref["user"]
    titles = data.items.set_index("item").title.fillna("").reindex(inst["item"].to_numpy()).fillna("").str.lower().to_numpy()
    when = pd.to_datetime(inst.ts.to_numpy(), unit="ms")
    rows = []
    for name, pat, lo, hi in COHORTS:
        m = np.array([bool(re.search(pat, t)) for t in titles])
        if lo: m &= when >= lo
        if hi: m &= when < hi
        if m.sum() == 0:
            rows.append(dict(cohort=name, purchases=0)); continue
        ci = paired_bootstrap(users[m], nd["orig"][m], nd["best"][m], n_resamples=1000, seed=0)
        rows.append(dict(cohort=name, purchases=int(m.sum()), users=int(len(np.unique(users[m]))),
                         cover_orig=cov["orig"][m].mean(), cover_best=cov["best"][m].mean(),
                         hit10_orig=hit["orig"][m].mean(), hit10_best=hit["best"][m].mean(),
                         ndcg_orig=average(nd["orig"][m], users[m]), ndcg_best=average(nd["best"][m], users[m]),
                         diff=ci["diff"], lo=ci["ci_low"], hi=ci["ci_high"]))
    res = pd.DataFrame(rows)
    res.to_csv(out / "cohorts.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    show = res.copy()
    for c in ("cover_orig", "cover_best", "hit10_orig", "hit10_best"):
        show[c] = (show[c] * 100).round(1)
    for c in ("ndcg_orig", "ndcg_best", "diff", "lo", "hi"):
        show[c] = (show[c] * 1000).round(2)
    print(show.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
