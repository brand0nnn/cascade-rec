"""Poster figures for the new meso cuts and micro cases (Validation; from saved analysis outputs only).

fig9   where the improvement goes: nDCG@10 by product type, original vs best system (dumbbell)
fig10  what the system can and cannot do: same platform / switch, same iPhone model / newer model
fig11  four micro cases: history, purchase, rank under each system

Original = MLP on two-tower lists; best = DIN on mixed lists with the combined retrieval score (seed means).

Usage: .venv/bin/python scripts/poster_meso_micro_20261108.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "results/targeted_study/meso-segments-rrf-20261108"
OUT = ROOT / "results/figures/poster_20261108"
ORIG, BEST = "#2a78d6", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#b4b2a9", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def dumbbell(ax, rows, labels):
    y = np.arange(len(rows))[::-1]
    for yi, (_, r) in zip(y, rows.iterrows()):
        o, b = r.mlp_tt * 1000, r.din_rrf * 1000
        ax.plot([o, b], [yi, yi], color=MUTED, lw=2, zorder=1)
        ax.scatter([o], [yi], s=90, color=ORIG, zorder=2, edgecolor=SURF, linewidth=2)
        ax.scatter([b], [yi], s=90, color=BEST, zorder=3, edgecolor=SURF, linewidth=2)
        ratio = b / o
        txt = f"×{ratio:.1f}" if ratio >= 1.5 else f"{(ratio - 1) * 100:+.0f}%"
        ax.text(max(o, b) + 0.8, yi, txt, va="center", fontsize=12, color=INK, fontweight="bold" if ratio >= 1.5 or ratio < 0.95 else None)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def legend(ax, loc="lower right"):
    ax.scatter([], [], s=90, color=ORIG, label="Original: MLP, two-tower list")
    ax.scatter([], [], s=90, color=BEST, label="Best: DIN, mixed list + combined score")
    ax.legend(loc=loc, frameon=False, fontsize=11)


def fig9(seg):
    order = ["case", "car mount/charger", "other", "watch accessory", "charger/cable", "screen protector", "phone"]
    names = {"case": "Phone cases", "car mount/charger": "Car mounts & chargers", "other": "Other accessories",
             "watch accessory": "Watch accessories", "charger/cable": "Chargers & cables",
             "screen protector": "Screen protectors", "phone": "Phones"}
    r = seg[seg.cut == "product type"].set_index("group").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    dumbbell(ax, r, [f"{names[g]}  ({s * 100:.0f}%)" for g, s in zip(r.group, r.share)])
    ax.set_xlabel("nDCG@10 ×1000 (higher is better)")
    ax.set_xlim(0, 38)
    ax.set_title("The gain goes to hard, niche categories; easy popular ones lose a little", loc="left", fontsize=14, color=INK)
    ax.text(0, -0.17, "Share of purchases in brackets. Validation, 232,904 purchases, mean of 3 seeds. "
            "All changes have 95% intervals excluding 0 except chargers & cables.", transform=ax.transAxes, fontsize=10, color=INK2)
    legend(ax, "center right")
    save(fig, "fig9_gain_by_product_type")


def fig10(seg):
    pick = [("user vs product", "same platform", "Same platform as history"),
            ("user vs product", "switch platform", "Switches platform\n(iPhone ↔ Android)"),
            ("iPhone upgrade", "same iPhone model", "Same iPhone model\nas history"),
            ("iPhone upgrade", "newer iPhone model", "Newer iPhone model\n(upgrade)")]
    r = pd.DataFrame([seg[(seg.cut == c) & (seg.group == g)].iloc[0] for c, g, _ in pick])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, sl, title in ((axes[0], slice(0, 2), "Staying on your platform vs switching"),
                          (axes[1], slice(2, 4), "Same iPhone model vs upgrading")):
        rr = r.iloc[sl]
        x = np.arange(len(rr)); w = 0.36
        ax.bar(x - w / 2 - 0.01, rr.mlp_tt * 1000, w, color=ORIG)
        ax.bar(x + w / 2 + 0.01, rr.din_rrf * 1000, w, color=BEST)
        for xi, (_, q) in zip(x, rr.iterrows()):
            for dx, v in ((-w / 2, q.mlp_tt), (w / 2, q.din_rrf)):
                ax.text(xi + dx, v * 1000 + 0.4, f"{v * 1000:.1f}", ha="center", fontsize=11, color=INK)
        ax.set_xticks(x, [p[2] for p in pick[sl]])
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("nDCG@10 ×1000")
    axes[0].text(1, r.iloc[1].din_rrf * 1000 + 3.2, "purchase in the 200\ncandidates only 2–3%\nof the time", ha="center", fontsize=10, color=INK2)
    axes[1].text(1, r.iloc[3].din_rrf * 1000 + 3.2, "DIN no better than MLP\n(+0.15 [−0.30, +0.60])", ha="center", fontsize=10, color=INK2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ORIG, label="Original: MLP, two-tower list"),
                        Patch(color=BEST, label="Best: DIN, mixed list + combined score")],
               loc="upper left", bbox_to_anchor=(0.055, 0.995), frameon=False, fontsize=11, ncol=2)
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.suptitle("Great at “more for the phone you have”, weak at switches and upgrades", x=0.06, y=1.07, ha="left", fontsize=14, color=INK)
    fig.text(0.06, -0.04, "Platform and model labels come from product titles, which sellers edit over time; treat them as approximate. Validation, mean of 3 seeds.",
             fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "fig10_switch_and_upgrade")


def short(t, n=46):
    return t if len(t) <= n else t[:n - 1].rstrip() + "…"


def fig11():
    cases = [
        ("Upgrade caught", "18 Sep 2021, iPhone 13 launch week",
         ["Facbiny case, iPhone 12 Pro Max", "Facbiny bling case, iPhone 12 Pro Max"], "Facbiny case, iPhone 13 Pro", "not found", "5"),
        ("Same store, model family", "8 Apr 2022",
         ["UAG rugged case, iPhone 13"], "UAG rugged case, iPhone 13 Pro", "not found", "1"),
        ("Popular commodity lost", "16 Aug 2021",
         ["JETech screen protector, iPhone 11", "Clear case, iPhone 11 Pro Max"], "Ailun screen protector, iPhone 11 Pro Max", "3", "16"),
        ("Blind spot: platform switch", "24 Oct 2021",
         ["NTG rugged case, iPhone SE", "ESR clear case, iPhone SE", "Wireless charger"], "Ring-stand case, OnePlus Nord N10", "not found", "not found"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06)
    for ax, (title, date, hist, bought, ro, rb) in zip(axes, cases):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f3f2ef", edgecolor=GRID))
        ax.text(0.05, 0.92, title, fontsize=13.5, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.05, 0.84, date, fontsize=10.5, color=INK2, transform=ax.transAxes)
        ax.text(0.05, 0.74, "Owned before:", fontsize=10.5, color=INK2, transform=ax.transAxes)
        for i, h in enumerate(hist):
            ax.text(0.07, 0.66 - i * 0.075, "• " + short(h, 34), fontsize=10.5, color=INK, transform=ax.transAxes)
        ax.text(0.05, 0.38, "Bought next:", fontsize=10.5, color=INK2, transform=ax.transAxes)
        ax.text(0.07, 0.30, short(bought, 34), fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
        for i, (lab, val, c) in enumerate((("Original rank", ro, ORIG), ("Best rank", rb, BEST))):
            yy = 0.16 - i * 0.09
            ax.scatter([0.08], [yy + 0.012], s=60, color=c, transform=ax.transAxes)
            ax.text(0.13, yy, f"{lab}: {val}", fontsize=11.5, color=INK, transform=ax.transAxes,
                    fontweight="bold" if val not in ("not found",) and i == 1 else None)
    fig.suptitle("Four real purchases (Validation): what each system ranked the item that was bought", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, -0.03, "Rank among recommendations, seed 0; “not found” = not in the 200 candidates. Titles shortened.", fontsize=10, color=INK2)
    save(fig, "fig11_micro_cases")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(SEG / "segments.csv")
    fig9(seg); fig10(seg); fig11()


if __name__ == "__main__":
    main()
