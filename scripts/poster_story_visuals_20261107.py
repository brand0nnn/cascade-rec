"""Story visuals for the STePS poster, drawn from saved data (no new model outputs).

fig5  shortcut: a real natural list vs a real forced list (target position and score)
fig6  where the top 10 comes from at K=500 (forced A vs natural C, uniform 1,000-event cohort)
fig7  new products entering the catalogue per week across Validation (iPhone 13 launch wave)
fig8  per-product-type routing schematic with measured group results

Usage: .venv/bin/python scripts/poster_story_visuals_20261107.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cascade.data.instances import ProcessedData
from cascade.pipelines.build_candidates import load_candidates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/poster_20261107"
TS = ROOT / "results/targeted_study"
COL = {"A": "#2a78d6", "C": "#eb6834", "DIN": "#1baf7a", "R1": "#eda100"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#b4b2a9", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def shortcut(train):
    forced = np.flatnonzero(train["forced"])
    natural = np.flatnonzero(~train["forced"] & (train["target_pos"] >= 5) & (train["target_pos"] <= 15))
    rows = {"Natural list: retriever found the purchase": int(natural[0]),
            "Forced list: purchase appended at the end": int(forced[0])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, row) in zip(axes, rows.items()):
        valid = train["cand"][row] > 0
        s = train["score"][row].astype(float)[valid]
        pos = int(train["target_pos"][row])
        x = np.arange(1, len(s) + 1)
        colors = np.where(x == pos + 1, COL["C"] if "Natural" in title else COL["A"], MUTED)
        ax.bar(x, s - s.min() + 0.02, width=1.0, color=colors, edgecolor="none")
        ax.bar([pos + 1], [s[pos] - s.min() + 0.02], width=4, color=colors[pos], edgecolor="none")
        ax.annotate(f"bought (slot {pos + 1})", (pos + 1, s[pos] - s.min() + 0.04),
                    (pos + 1 + 25, (s.max() - s.min()) * 0.75) if pos < 100 else (140, (s.max() - s.min()) * 0.25),
                    fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Position in the training list (sorted by retrieval score)")
        ax.set_xlim(0, 205); ax.set_yticks([])
    axes[0].set_ylabel("Retrieval score")
    fig.suptitle("The shortcut: in 88% of training lists the purchase sat at the very bottom",
                 x=0.0, ha="left", fontsize=14, color=INK)
    f = np.flatnonzero(train["forced"])
    sc = train["score"][f].astype(np.float32)
    tgt = sc[np.arange(len(f)), train["target_pos"][f]]
    valid = (train["cand"][f] > 0)
    valid[np.arange(len(f)), train["target_pos"][f]] = False
    below_all = np.mean(np.all(np.where(valid, sc > tgt[:, None], True), axis=1)) * 100
    fig.text(0.0, -0.04, f"Two real ranker-training lists. {len(f):,} of {len(train['forced']):,} lists were forced; the appended "
             f"purchase scored below every retrieved candidate in {below_all:.0f}% of them.", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig5_shortcut")


def top10_origin():
    d = pd.read_csv(TS / "policy-all-seeds-20261107/score_diagnostics.csv")
    d = d[d.status == "measured"].groupby("arm").added_top10_share.agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (arm, label) in enumerate((("A", "Forced-list MLP"), ("C", "Natural-list MLP"))):
        share = d.loc[arm, "mean"] * 100
        ax.barh(i, 100 - share, color=MUTED, height=0.55, edgecolor=SURF, lw=2)
        ax.barh(i, share, left=100 - share, color=COL[arm], height=0.55, edgecolor=SURF, lw=2)
        ax.text(100 - share + share / 2, i, f"{share:.0f}%", ha="center", va="center", color="white", fontsize=14)
        ax.text(-2, i, label, ha="right", va="center", fontsize=12, color=INK)
    ax.set_yticks([]); ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.7); ax.invert_yaxis()
    ax.set_xlabel("Share of top-10 slots at K₁=500 (%)")
    ax.text(2, 1.5, "grey: from positions 1–200", fontsize=10, color=INK2)
    ax.text(98, 1.5, "colour: from positions 201–500 (added)", fontsize=10, color=INK2, ha="right")
    ax.spines["left"].set_visible(False)
    ax.set_title("Add 300 weak candidates and the shortcut ranker promotes them", loc="left", fontsize=13, color=INK)
    fig.text(0.0, -0.08, f"1,000 uniform validation events; mean of 3 seeds (A {d.loc['A','min']*100:.0f}–{d.loc['A','max']*100:.0f}%, "
             f"C {d.loc['C','min']*100:.0f}–{d.loc['C','max']*100:.0f}%).", fontsize=9, color=INK2)
    save(fig, "fig6_top10_origin")


def launch_timeline(data):
    t1, t2 = int(data.split["t1"]), int(data.split["t2"])
    items = data.items
    new = items[(items.first_ts >= t1) & (items.first_ts < t2)]
    week = pd.to_datetime(new.first_ts, unit="ms").dt.to_period("W").dt.start_time
    counts = week.value_counts().sort_index()
    val = data.split_instances("val")
    first = items.set_index("item").first_ts.reindex(val["item"]).to_numpy()
    month = pd.to_datetime(val.ts, unit="ms").dt.to_period("M").dt.start_time
    share = pd.Series(first >= t1).groupby(month.to_numpy()).mean()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(counts.index, counts.values, width=6, color=COL["DIN"], edgecolor="none")
    ax.set_ylabel("Products first seen per week")
    launch = pd.Timestamp("2021-09-14")
    peak = counts.loc[counts.index >= launch - pd.Timedelta(days=7)].iloc[:4].max()
    ax.annotate("iPhone 13 launch\n(14 Sep 2021)", (launch, peak), (launch + pd.Timedelta(days=40), peak * 0.95),
                fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    s22 = pd.Timestamp("2022-02-25")
    peak2 = counts.loc[(counts.index >= s22 - pd.Timedelta(days=10)) & (counts.index <= s22 + pd.Timedelta(days=10))].max()
    ax.annotate("Galaxy S22 release\n(25 Feb 2022)", (s22, peak2), (s22 + pd.Timedelta(days=35), peak2 * 0.95),
                fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    first_m, last_m = share.iloc[0] * 100, share.iloc[-1] * 100
    ax.set_title(f"New products keep arriving: their share of purchases grows from {first_m:.0f}% to {last_m:.0f}% across Validation",
                 loc="left", fontsize=13, color=INK)
    fig.text(0.0, -0.04, f"{len(new):,} products first appear during Validation (11 Aug 2021 – 16 Jul 2022); "
             f"overall {np.mean(first >= t1) * 100:.0f}% of Validation purchases are of these products.", fontsize=9, color=INK2)
    save(fig, "fig7_launch_timeline")
    return share


def routing():
    t = pd.read_csv(TS / "meso-oracle-20261107/meso_metrics.csv")
    g = lambda s, grp: t[(t.system == s) & (t.group == grp)].ndcg10.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=SURF, ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=13, color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=10.5, color=INK2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.4))

    box(2, 15, 20, 10, "Purchase", "candidate list", INK2)
    box(33, 26, 30, 10, "Popular product", "47% of purchases", MUTED)
    box(33, 4, 30, 10, "New or niche product", "53% of purchases", MUTED)
    box(72, 26, 26, 10, "Natural-list MLP", f"popular-item score {g('policy_C','warm-head'):.2f}", COL["C"])
    box(72, 4, 26, 10, "DIN", f"new-item score {g('R3_din','new-in-period'):.2f}", COL["DIN"])
    arrow(22.5, 21, 32.5, 30); arrow(22.5, 19, 32.5, 9)
    arrow(63.5, 31, 71.5, 31); arrow(63.5, 9, 71.5, 9)
    ax.text(0, 39, "No single best ranker: choose by product type", fontsize=14, color=INK)
    ax.text(0, -2, "Upper bound if the better ranker were chosen for every purchase: +34% (A/C) to +42% (MLP/DIN). "
            "Score = nDCG@10 ×100. Not a deployed router; it needs a selector built from pre-purchase signals.", fontsize=9, color=INK2)
    save(fig, "fig8_routing")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    train = load_candidates(ROOT / "data/processed/candidates/two_tower_text_seed0", "ranker_train")
    shortcut(train); top10_origin(); print(launch_timeline(data).round(3).to_string()); routing()
    print("done")
