"""Launch figure: iPhone 13 accessory purchases by week, original vs best system, and where the hits come from.

Top: seed-averaged nDCG@10 per 2-week bin (per-user macro within bin).
Bottom: share of purchases found in the 200 candidates: two-tower list (original) vs mixed list (best).
The extra coverage of the mixed list over the two-tower list must come from the recency side (the two-tower
half of the mixed list is a subset of the two-tower top 200).
Validation only; titles are current metadata (see the title-edit check in docs/evaluations.md).

Usage: .venv/bin/python scripts/poster_launch_decay_20261108.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.saved_results import average

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/poster_20261108"
ORIG, BEST = "#2a78d6", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#b4b2a9", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def load(arm, run):
    nd, cov, ref = [], None, None
    for s in (0, 1, 2):
        z = np.load(ROOT / "results/hybrid_study" / run.format(s) / f"{arm}_seed{s}" / "ranks.npz")
        nd.append(M.ndcg_at(z["ranks"], 10)); cov = z["target_pos"] >= 0; ref = z
    return np.mean(nd, 0), cov, ref


def main():
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    o, co, ref = load("mlp_tt", "hybrid-20261107-s{}")
    b, cb, _ = load("din_rrf", "rrf-20261108-s{}")
    inst = data.instances.set_index("inst").loc[ref["inst"]].reset_index()
    users = ref["user"]
    title = data.items.set_index("item").title.fillna("").reindex(inst["item"].to_numpy()).fillna("").str.lower()
    m = title.str.contains(r"iphone\s*13", regex=True).to_numpy()
    when = pd.to_datetime(inst.ts.to_numpy(), unit="ms")
    m &= when >= "2021-09-14"
    bins = pd.date_range("2021-09-14", "2022-07-16", freq="14D")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = m & (when >= lo) & (when < hi)
        if k.sum() < 200:
            continue
        rows.append(dict(start=lo, mid=lo + (hi - lo) / 2, n=int(k.sum()),
                         orig=average(o[k], users[k]) * 1000, best=average(b[k], users[k]) * 1000,
                         cov_orig=co[k].mean() * 100, cov_best=cb[k].mean() * 100))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/targeted_study/micro-cohorts-rrf-20261108/iphone13_fortnightly.csv", index=False)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    for ax, (c1, c2), lab in ((a1, ("orig", "best"), "nDCG@10 ×1000"), (a2, ("cov_orig", "cov_best"), "Purchase in the\n200 candidates (%)")):
        ax.plot(df.mid, df[c1], color=ORIG, lw=2, marker="o", ms=6)
        ax.plot(df.mid, df[c2], color=BEST, lw=2, marker="o", ms=6)
        ax.set_ylabel(lab); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_ylim(0, None)
    last = df.iloc[-1]
    a1.text(last.mid + pd.Timedelta(days=4), max(last.best, last.orig + 3.2), "Best: DIN, mixed list\n+ combined score", color=INK, fontsize=11, va="center")
    a1.text(last.mid + pd.Timedelta(days=4), last.orig, "Original: MLP,\ntwo-tower list", color=INK, fontsize=11, va="center")
    a2.text(last.mid + pd.Timedelta(days=4), last.cov_best, "Mixed list", color=INK, fontsize=11, va="center")
    a2.text(last.mid + pd.Timedelta(days=4), last.cov_orig, "Two-tower list", color=INK, fontsize=11, va="center")
    f = df.iloc[0]
    a1.annotate(f"Launch fortnight: ×{f.best / f.orig:.0f}", (f.mid, f.best), (f.mid + pd.Timedelta(days=30), f.best * 0.95),
                fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    a2.text(df.mid.iloc[2], df.cov_best.max() * 0.93, "the extra purchases come from the recency retriever:\n"
            "similar text to the user's own recent purchases", fontsize=10.5, color=INK2)
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10, 12, 2, 4, 6]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("iPhone 13 accessories: the best system's edge is largest at launch and fades", x=0.07, ha="left", fontsize=14, color=INK)
    fig.text(0.07, -0.01, f"Validation, purchases of products whose title mentions iPhone 13, 2-week bins (≥200 purchases each; n = {df.n.sum():,}). "
             "Mean of 3 seeds.", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    fig.savefig(OUT / "fig12_iphone13_launch.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig12_iphone13_launch.svg", bbox_inches="tight")
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
