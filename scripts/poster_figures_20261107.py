"""Poster figures from saved validation analyses (no new model outputs).

Usage: .venv/bin/python scripts/poster_figures_20261107.py
Writes PNG (300 dpi) and SVG to results/figures/poster_20261107/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/poster_20261107"
TS = ROOT / "results/targeted_study"
# Reference palette slots 1-4 (validated): blue, orange, aqua, yellow.
COL = {"A": "#2a78d6", "C": "#eb6834", "DIN": "#1baf7a", "R1": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"A": "Forced-list MLP (original)", "C": "Natural-list MLP",
         "DIN": "DIN (forced-list)", "R1": "No ranker (retrieval order)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def k_curve():
    k = pd.read_csv(TS / "e3-20261107-all-seeds/k_curve.csv")
    m = pd.read_csv(TS / "policy-all-seeds-20261107/metrics.csv")
    sd = m[m.axis == "overall"].groupby(["arm", "k1"]).ndcg10.std()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for arm, key in (("A", "A"), ("C", "C")):
        d = k[k.arm == arm].sort_values("k1")
        e = [sd[(arm, kk)] for kk in d.k1]
        ax.errorbar(d.k1, d.ndcg10 * 100, yerr=np.array(e) * 100, color=COL[key], lw=2, marker="o",
                    ms=8, capsize=3, mec=SURF, mew=2, label=LABEL[key])
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xticks([50, 100, 200, 500]); ax.set_xticklabels(["50", "100", "200", "500"])
    ax.set_xlabel("Candidates passed to the ranker (K₁)"); ax.set_ylabel("nDCG@10 (×100)")
    ax.set_ylim(0, 1.0); ax.grid(axis="y", color=GRID, lw=0.8)
    a500 = k[(k.arm == "A") & (k.k1 == 500)].ndcg10.iloc[0] * 100
    c500 = k[(k.arm == "C") & (k.k1 == 500)].ndcg10.iloc[0] * 100
    ax.annotate("collapses −72%", (500, a500), (230, 0.18), color=INK, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("stays flat", (500, c500), (240, 0.93), color=INK, fontsize=12,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    ax.set_title("Bigger candidate pools only hurt the shortcut-trained ranker", loc="left", fontsize=13, color=INK)
    fig.text(0.0, -0.04, "Validation, mean of 3 training seeds (error bars = seed SD). Same retriever and candidates.",
             fontsize=9, color=INK2)
    save(fig, "fig1_k_curve")


def winners_grid():
    t = pd.read_csv(TS / "meso-oracle-20261107/meso_metrics.csv")
    rows = [("R1", "R1_retrieval_order"), ("A", "policy_A"), ("C", "policy_C"), ("DIN", "R3_din")]
    cols = [("warm-head", "Popular"), ("warm-tail", "Niche"), ("new-in-period", "New"), ("all", "Overall")]
    val = np.array([[t[(t.system == s) & (t.group == g)].ndcg10.iloc[0] for g, _ in cols] for _, s in rows])
    rel = val / val[1]  # relative to A (original reference)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#c2410c", "#f1f0ec", "#1c5cab"])
    norm = matplotlib.colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    ax.imshow(np.clip(rel, 0, 2), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            r = rel[i, j]
            txt = "reference" if i == 1 else f"{(r - 1) * 100:+.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if abs(r - 1) > 0.6 else INK, fontweight="bold" if i != 1 else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for _, c in cols])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([LABEL[k] for k, _ in rows])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURF, lw=3); ax.tick_params(which="minor", length=0)
    ax.set_title("No ranker wins everywhere: nDCG@10 vs the original ranker", loc="left", fontsize=13, color=INK)
    fig.text(0.0, -0.06, "Blue = better, red = worse than forced-list MLP. Validation, 3-seed means (retrieval order: deterministic).",
             fontsize=9, color=INK2)
    save(fig, "fig2_winners_grid")


def accuracy_vs_popularity():
    d = json.loads((TS / "top10-diversity-20261107/diversity.json").read_text())["systems"]
    t = pd.read_csv(TS / "meso-oracle-20261107/meso_metrics.csv")
    acc = {"R1": "R1_retrieval_order", "A": "policy_A", "C": "policy_C", "DIN": "R3_din"}
    div = {"R1": "R1_retrieval_order", "A": "A_forced_mlp", "C": "C_natural_mlp", "DIN": "DIN_forced"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for key in ("R1", "A", "C", "DIN"):
        y = t[(t.system == acc[key]) & (t.group == "all")].ndcg10.iloc[0] * 100
        for ax, metric, scale in ((axes[0], "top1pct_share", 100), (axes[1], "new_share", 100)):
            ax.scatter(d[div[key]][metric] * scale, y, s=160, color=COL[key], edgecolor=SURF, lw=2, zorder=3)
            ax.annotate(LABEL[key], (d[div[key]][metric] * scale, y), (8, 6), textcoords="offset points",
                        fontsize=10, color=INK)
    axes[0].set_xlabel("Top-10 slots given to the top-1% bestsellers (%)")
    axes[1].set_xlabel("Top-10 slots given to new products (%)")
    for ax in axes:
        ax.set_ylabel("nDCG@10 (×100)"); ax.grid(color=GRID, lw=0.8); ax.set_ylim(0.4, 1.0)
    fig.suptitle("Accuracy is bought with popularity: the best ranker recommends the most bestsellers",
                 x=0.0, ha="left", fontsize=13, color=INK)
    fig.text(0.0, -0.05, "Validation, natural K=200 lists. Accuracy: 3-seed means; list composition: seed-0 models.",
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig3_accuracy_vs_popularity")


def training_vs_architecture():
    t = pd.read_csv(TS / "meso-oracle-20261107/meso_metrics.csv")
    g = lambda s: t[(t.system == s) & (t.group == "all")].ndcg10.iloc[0]
    base = g("policy_A")
    bars = [("Change the model\n(MLP → DIN)", g("R3_din") / g("R2_mlp") - 1, COL["DIN"]),
            ("Change the training lists\n(forced → natural)", g("policy_C") / base - 1, COL["C"]),
            ("Best of both per purchase\n(oracle, vs natural-list)", g("oracle_A_C") / g("policy_C") - 1, INK2)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (name, v, c) in enumerate(bars):
        ax.barh(i, v * 100, color=c, height=0.55, edgecolor=SURF, lw=2)
        ax.text(v * 100 + (1 if v >= 0 else -1), i, f"{v * 100:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=13, color=INK, fontweight="bold")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars]); ax.invert_yaxis()
    ax.set_xlim(-15, 45); ax.set_xlabel("Change in nDCG@10 (%)"); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("How you train beats which model you pick", loc="left", fontsize=13, color=INK)
    fig.text(0.0, -0.06, "Validation, 3-seed means. Oracle = upper bound, not a deployable system.", fontsize=9, color=INK2)
    save(fig, "fig4_training_vs_architecture")


if __name__ == "__main__":
    k_curve(); winners_grid(); training_vs_architecture()
    if (TS / "top10-diversity-20261107/diversity.json").exists():
        accuracy_vs_popularity()
    print("figures in", OUT)
