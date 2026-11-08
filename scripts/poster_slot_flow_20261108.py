"""fig24: what fills the candidate list and the top 10, by purchase type (best system, Validation sample).

For each purchase type, an alluvial from the 200-candidate list (left bar, share of slots) to the ranker's top 10
(right bar), slots coloured by which phone the product is for relative to the purchase: bought phone / owned phone
/ other phone / no phone. Notes give the same-product-type share and how often the exact purchase is in the top 10.
Data: results/targeted_study/list-dissection-20261108/dissection.csv (20,000 Val purchases, seed 0).
Labels from titles (approximate).

Usage: .venv/bin/python scripts/poster_slot_flow_20261108.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/poster_20261108"
NAVY, INK2, SURF = "#232F3E", "#52514e", "#fcfcfb"
COL = {"bought phone": "#E8710A", "owned phone": "#5F5F5F", "other phone": "#BDBDBD", "no phone": "#F6C9A3"}
LAB = {"bought phone": "for the phone they bought", "owned phone": "for a phone they already own", "other phone": "for some other phone",
       "no phone": "no phone (cables, chargers…)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "figure.facecolor": SURF, "axes.facecolor": SURF})


def band(ax, x0, x1, a0, b0, a1, b1, c):
    t = np.linspace(0, 1, 50); s = 3 * t ** 2 - 2 * t ** 3
    ax.fill_between(x0 + (x1 - x0) * t, a0 + (a1 - a0) * s, b0 + (b1 - b0) * s, color=c, alpha=0.6, lw=0)


def main():
    d = pd.read_csv(ROOT / "results/targeted_study/list-dissection-20261108/dissection.csv")
    d = d[d.system == "best"].set_index("behaviour")
    share = d.n / d.n.sum()
    panels = [("same device", "Same phone as before"), ("upgrade", "Upgrade to a newer model"),
              ("device-agnostic", "General accessory (no phone)"), ("switch platform", "Switch iPhone ↔ Android")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.4))
    for ax, (b, name) in zip(axes, panels):
        r = d.loc[b]
        cats = [c for c in COL if not (b == "device-agnostic" and c == "bought phone")]
        L = np.array([r[f"list_{c}"] for c in cats]); T = np.array([r[f"top_{c}"] for c in cats])
        L, T = L / L.sum(), T / T.sum()
        yl = yt = 1.0
        for c, l, t in zip(cats, L, T):
            ax.add_patch(Rectangle((0, yl - l), 0.18, l, color=COL[c])); ax.add_patch(Rectangle((0.82, yt - t), 0.18, t, color=COL[c]))
            band(ax, 0.18, 0.82, yl - l, yl, yt - t, yt, COL[c])
            if l > 0.04: ax.text(-0.03, yl - l / 2, f"{l * 100:.0f}%", ha="right", va="center", fontsize=11, color=NAVY)
            if t > 0.04: ax.text(1.03, yt - t / 2, f"{t * 100:.0f}%", ha="left", va="center", fontsize=11, color=NAVY,
                                 fontweight="bold" if c == "bought phone" else None)
            yl -= l; yt -= t
        ax.text(0.09, 1.03, "200\ncandidates", ha="center", va="bottom", fontsize=10.5, color=INK2)
        ax.text(0.91, 1.03, "top 10", ha="center", va="bottom", fontsize=10.5, color=INK2)
        ax.set_title(f"{name}\n({share[b] * 100:.0f}% of purchases)", fontsize=13, color=NAVY, pad=38)
        note = (f"Same product type: {r.list_same_type * 100:.0f}% → {r.top_same_type * 100:.0f}%\n"
                f"Purchase reaches the ranker: {r.found * 100:.1f}%\nExact purchase in top 10: {r.hit10 * 100:.1f}%")
        if b != "device-agnostic":
            note = f"Top 10 has ≥1 for the bought phone: {r.top_has_bought_phone * 100:.0f}%\n" + note
        ax.text(0.5, -0.06, note, ha="center", va="top", fontsize=10.5, color=NAVY, linespacing=1.5)
        ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.42, 1.12); ax.axis("off")
    fig.legend(handles=[Patch(color=COL[c], label=LAB[c]) for c in COL], loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=4, frameon=False, fontsize=11)
    fig.text(0.01, 0.99, "What fills the list and the top 10: the ranker moves slots toward the right phone, but cannot add what retrieval missed",
             fontsize=14.5, color=NAVY, va="top")
    fig.text(0.01, -0.01, "Best system (DIN on mixed list + combined score), 20,000 random Validation purchases, seed 0. Slots classified by the phone models product titles name (approximate).",
             fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.72, bottom=0.05, wspace=0.25)
    fig.savefig(OUT / "fig24_slot_flow_orange.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig24_slot_flow_orange.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
