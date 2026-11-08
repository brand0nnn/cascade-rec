"""Archetype 'funnel' rows: of 100 purchases, how many reach the ranker (in the 200 candidates) and how many make
the top 10, for the original and best systems. Users (behaviour, prior purchases for the same phone) and products
(store vs history, age at purchase, store size). Validation, seed-averaged; labels from titles are approximate.

Usage: .venv/bin/python scripts/poster_archetype_funnels_20261108.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/poster_20261108"
ORIG, BEST = "#2a78d6", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#b4b2a9", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def light(c, f=0.55):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(c + (1 - c) * f)


def panel(ax, rows, title):
    y = 0; ticks, labels = [], []
    for sec, items in rows:
        if sec:
            ax.text(-0.5, y - 0.15, sec, fontsize=12, fontweight="bold", color=INK2, ha="right", va="center")
            y += 0.9
        for lab, r in items:
            for k, (sys, col) in enumerate((("orig", ORIG), ("best", BEST))):
                yy = y + k * 0.38
                reach = 100 - r[f"{sys}_retrieval_miss"] * 100; hit = r[f"{sys}_hit10"] * 100
                ax.barh(yy, reach, 0.34, color=light(col), edgecolor="none")
                ax.barh(yy, hit, 0.34, color=col, edgecolor="none")
                ax.text(reach + 0.4, yy, f"{reach:.1f} reach · {hit:.1f} top 10", va="center", fontsize=9.5, color=INK2)
            ticks.append(y + 0.19); labels.append(f"{lab}  ({r['share'] * 100:.0f}%)")
            y += 1.05
        y += 0.3
    ax.set_yticks(ticks, labels); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Out of 100 purchases (%)")


def legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=light(ORIG), label="Original: reaches ranker"), Patch(color=ORIG, label="Original: in top 10"),
                        Patch(color=light(BEST), label="Best: reaches ranker"), Patch(color=BEST, label="Best: in top 10")],
               loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=4, frameon=False, fontsize=10.5)


def pick(df, cut, groups, names):
    d = df[df.cut == cut].set_index("group")
    return [(names.get(g, g), d.loc[g]) for g in groups if g in d.index]


def main():
    b = pd.read_csv(ROOT / "results/targeted_study/behaviour-failures-20261108/behaviour_failures.csv")
    p = pd.read_csv(ROOT / "results/targeted_study/product-archetypes-20261108/product_archetypes.csv")
    users = [("What they bought it for", pick(b, "behaviour", ["same device", "other model, same family", "upgrade", "older model",
                                                               "device-agnostic", "other device, same platform", "switch platform"],
                                              {"same device": "Same phone as before", "other model, same family": "Other model, same family",
                                               "upgrade": "Upgrade (newer model)", "older model": "Older model", "device-agnostic": "No phone (cables, chargers…)",
                                               "other device, same platform": "Other phone, same platform", "switch platform": "Switch iPhone ↔ Android"})),
             ("Earlier purchases for that phone", pick(b, "prior purchases for this device", ["0", "1", "2", "3-4", "5+"],
                                                       {"0": "0", "1": "1", "2": "2", "3-4": "3–4", "5+": "5+"}))]
    prods = [("Store", pick(p, "store vs history", ["same store as history", "new store"],
                            {"same store as history": "Store the user bought from before", "new store": "New store"})),
             ("Product age at purchase", pick(p, "product age at purchase", ["<30 days", "30-180 days", "180-365 days", "1-2 years", ">2 years"],
                                              {"30-180 days": "30–180 days", "180-365 days": "180–365 days", "1-2 years": "1–2 years"})),
             ("Store size (products listed)", pick(p, "store size", ["1-9", "10-99", "100-499", "500+"],
                                                   {"1-9": "1–9", "10-99": "10–99", "100-499": "100–499"}))]
    for name, rows, title, xmax in (("fig13_user_archetypes", users, "User side: works for the phone you already accessorise", 25),
                                    ("fig14_product_archetypes", prods, "Product side: works for fresh products from stores you know", 38)):
        n = sum(len(i) for _, i in rows)
        fig, ax = plt.subplots(figsize=(11.5, 1.0 + 0.62 * n + 0.5 * len(rows)))
        panel(ax, rows, title); ax.set_xlim(0, xmax)
        legend(fig)
        fig.text(0.01, 0.995, title, fontsize=14, color=INK, va="top")
        fig.text(0.01, -0.02, "Share of all purchases in brackets. The rest of each 100 never reaches the ranker. Validation, 232,904 purchases, "
                 "mean of 3 seeds. Phone labels come from product titles (approximate).", fontsize=9.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
