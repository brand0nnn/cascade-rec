"""fig25: RQ2 K curve to K=500 in the poster palette (Validation, two-tower text lists, 3 seeds).

Left: candidate recall (Recall@K of the two-tower retriever, results/rq1/summary.md).
Right: final nDCG@10 for the natural-list MLP (C) and the forced-list MLP (A) at K = 50/100/200/500
(results/targeted_study/ext-20261107-all-seeds/k_curve.csv).

Usage: .venv/bin/python scripts/poster_k_curve_orange_20261108.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORANGE, GREY, NAVY, INK2, GRID, SURF = "#E8710A", "#9A9A9A", "#232F3E", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "text.color": NAVY,
                     "axes.labelcolor": NAVY, "xtick.color": INK2, "ytick.color": INK2})
k = pd.read_csv(ROOT / "results/targeted_study/ext-20261107-all-seeds/k_curve.csv")
K = [50, 100, 200, 500]
recall = {50: 2.95, 100: 4.54, 200: 6.86, 500: 11.79}
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
a.plot(K, [recall[x] for x in K], color=NAVY, lw=2.5, marker="o", ms=7)
for x in K:
    a.annotate(f"{recall[x]:.1f}%", (x, recall[x]), (0, 9), textcoords="offset points", ha="center", fontsize=11)
a.set_xscale("log"); a.set_xticks(K, [str(x) for x in K]); a.minorticks_off()
a.set_ylim(0, 14); a.set_ylabel("Purchase among the K candidates (%)"); a.set_xlabel("Candidates passed to the ranker (K)")
a.set_title("Recall keeps rising with K", loc="left", fontsize=13)
for arm, col, lab in (("C", ORANGE, "Natural-list ranker"), ("A", GREY, "Forced-list ranker (original)")):
    q = k[k.arm == arm].set_index("k1").loc[K]
    b.plot(K, q.ndcg10 * 1000, color=col, lw=2.5, marker="o", ms=7, label=lab)
b.annotate("peak K=100–200", (150, 8.53), (60, 9.15), fontsize=11, color=NAVY)
b.annotate("−7% at K=500\n(more distractors)", (500, 7.93), (270, 6.3), fontsize=11, color=NAVY, arrowprops=dict(arrowstyle="->", color=INK2))
b.annotate("−72%: forced-list\nshortcut collapses", (500, 2.31), (190, 3.1), fontsize=11, color=NAVY, arrowprops=dict(arrowstyle="->", color=INK2))
b.set_xscale("log"); b.set_xticks(K, [str(x) for x in K]); b.minorticks_off()
b.set_ylim(0, 10); b.set_ylabel("nDCG@10 ×10⁻³"); b.set_xlabel("Candidates passed to the ranker (K)")
b.set_title("Final quality peaks at K≈100–200", loc="left", fontsize=13); b.legend(frameon=False, loc="lower left", fontsize=11)
for ax in (a, b):
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.text(0.01, -0.03, "Validation, two-tower (text) candidates, MLP rankers, mean of 3 seeds. K=500: natural vs its K=100 peak −0.00060 [−0.00079, −0.00041].",
         fontsize=9.5, color=INK2)
fig.tight_layout()
fig.savefig(ROOT / "results/figures/poster_20261108/fig25_k_curve_orange.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "results/figures/poster_20261108/fig25_k_curve_orange.svg", bbox_inches="tight")
