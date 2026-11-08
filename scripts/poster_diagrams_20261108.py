"""Poster diagrams (8 Nov): drawn from results/targeted_study/poster-data-20261108 and earlier saved outputs.

fig15 funnel flow (Test)          fig16 two retrievers, two worlds      fig17 user x product heatmap
fig18 repeat-buying staircase     fig19 waffles of 100 purchases         fig20 launch waves
fig21 what the retriever sees     fig22 titles that name the future      fig23 accuracy vs speed (Test)

Usage: .venv/bin/python scripts/poster_diagrams_20261108.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/targeted_study/poster-data-20261108"
OUT = ROOT / "results/figures/poster_20261108"
DEEP = "#0b6b48"
ORIG, BEST, REC, TT, WARN = "#2a78d6", "#1baf7a", "#eb6834", "#2a78d6", "#eda100"
INK, INK2, MUTED, GRID, SURF, CARD = "#0b0b0b", "#52514e", "#b4b2a9", "#e4e3df", "#fcfcfb", "#f3f2ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.svg", bbox_inches="tight"); plt.close(fig)


def light(c, f=0.6):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(c + (1 - c) * f)


def band(ax, x0, x1, y0a, y0b, y1a, y1b, color):
    t = np.linspace(0, 1, 60); s = 3 * t ** 2 - 2 * t ** 3
    x = x0 + (x1 - x0) * t
    top = y0b + (y1b - y0b) * s; bot = y0a + (y1a - y0a) * s
    ax.fill_between(x, bot, top, color=color, lw=0)


def fig15():
    f = pd.read_csv(D / "funnel.csv"); f = f[f.split == "Test"].set_index("system")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, sys, col, name in ((axes[0], "original", ORIG, "Original: MLP on two-tower list"), (axes[1], "best", BEST, "Best: DIN on mixed list + combined score")):
        r = f.loc[sys]; parts = [("Never reaches the ranker", r.retrieval_miss * 100, MUTED), ("Reaches ranker, ranked below 10", r.ranking_miss * 100, light(col)),
                                 ("In the top 10", r.hit10 * 100, col)]
        ax.add_patch(Rectangle((0, 0), 0.6, 100, color=INK2)); ax.text(-0.1, 50, "100\npurchases", ha="right", va="center", color=INK, fontsize=13, fontweight="bold")
        gap = 6; y_left = 100; y_right = 100 + gap * 2
        for lab, v, c in parts:
            vis = max(v, 1.2)
            band(ax, 0.6, 3.0, y_left - v, y_left, y_right - vis, y_right, light(c, 0.25) if c != MUTED else light(MUTED, 0.3))
            ax.add_patch(Rectangle((3.0, y_right - vis), 0.35, vis, color=c))
            ax.text(3.5, y_right - vis / 2, f"{v:.1f}  {lab}", va="center", fontsize=12, color=INK, fontweight="bold" if c == col else None)
            y_left -= v; y_right -= vis + gap
        ax.set_xlim(-1.6, 7.2); ax.set_ylim(-5, 116); ax.axis("off")
        ax.set_title(name, loc="left", fontsize=13, color=INK)
    fig.suptitle("Where 100 purchases go (Test): most never reach the ranker", x=0.02, ha="left", fontsize=15, color=INK)
    fig.text(0.02, -0.02, "Test period, 297,390 purchases, mean of 3 seeds. Small groups drawn at a minimum visible height; numbers are exact.", fontsize=10, color=INK2)
    save(fig, "fig15_funnel_flow")


def fig16():
    o = json.loads((D / "overlap.json").read_text())
    fig = plt.figure(figsize=(13, 5.4))
    ax = fig.add_axes([0.0, 0.05, 0.45, 0.85]); ax.set_aspect("equal"); ax.axis("off")
    r = 1.0; d = 1.85
    ax.add_patch(Circle((0, 0), r, color=REC, alpha=0.22, lw=0)); ax.add_patch(Circle((d, 0), r, color=TT, alpha=0.22, lw=0))
    ax.add_patch(Circle((0, 0), r, fill=False, ec=REC, lw=2)); ax.add_patch(Circle((d, 0), r, fill=False, ec=TT, lw=2))
    ax.text(-0.15, 0.25, "Recency\nretriever", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(-0.15, -0.45, "your own recent\npurchases, by text", ha="center", fontsize=10.5, color=INK2)
    ax.text(d + 0.15, 0.25, "Two-tower\nretriever", ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(d + 0.15, -0.45, "trained model,\npopular patterns", ha="center", fontsize=10.5, color=INK2)
    ax.annotate(f"only {o['mean_overlap_of_200']:.0f} of 200\nproducts shared", (d / 2, 0), (d / 2, -1.45), ha="center", fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlim(-1.3, d + 1.3); ax.set_ylim(-1.8, 1.3)
    ax2 = fig.add_axes([0.53, 0.18, 0.44, 0.62])
    groups = [("popular", "Popular"), ("niche", "Niche"), ("new", "New")]
    y = np.arange(len(groups))[::-1]
    for yi, (g, lab) in zip(y, groups):
        v = o["groups"][g]; left = 0
        for k, c in (("recency_only", REC), ("both", "#8a6fbf"), ("two_tower_only", TT)):
            ax2.barh(yi, v[k] * 100, 0.55, left=left, color=c, edgecolor=SURF, lw=1.5); left += v[k] * 100
        ax2.text(left + 0.3, yi, f"{left:.1f}% found", va="center", fontsize=11, color=INK2)
    ax2.set_yticks(y, [g[1] for g in groups]); ax2.tick_params(axis="y", length=0); ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("Purchases found among the 200 candidates (%)"); ax2.grid(axis="x", color=GRID); ax2.set_axisbelow(True)
    ax2.legend(handles=[Patch(color=REC, label="only recency finds it"), Patch(color="#8a6fbf", label="both"), Patch(color=TT, label="only two-tower finds it")],
               loc="upper center", bbox_to_anchor=(0.45, 1.22), ncol=3, frameon=False, fontsize=10.5)
    fig.suptitle("Two retrievers, two different worlds: mixing them adds what the other misses", x=0.02, ha="left", fontsize=15, color=INK)
    fig.text(0.02, -0.02, "20,000 random Validation purchases. Two-tower finds popular products; recency finds new and niche ones.", fontsize=10, color=INK2)
    save(fig, "fig16_two_retrievers")


def fig17():
    h = pd.read_csv(D / "heatmap.csv")
    rows = [("same device", "Same phone as before"), ("upgrade", "Upgrade (newer model)"), ("other phone", "Other phone, same platform"),
            ("device-agnostic", "No phone (cables, chargers…)"), ("switch platform", "Switch iPhone ↔ Android")]
    cols = [("store bought from before", "fresh (<6 months)", "Known store\nfresh product"), ("store bought from before", "older (6 months+)", "Known store\nolder product"),
            ("new store", "fresh (<6 months)", "New store\nfresh product"), ("new store", "older (6 months+)", "New store\nolder product")]
    M_ = np.full((len(rows), len(cols)), np.nan); O = M_.copy(); N = M_.copy()
    for i, (b, _) in enumerate(rows):
        for j, (s, a, _) in enumerate(cols):
            q = h[(h.behaviour == b) & (h.store == s) & (h.age == a)]
            if len(q):
                M_[i, j], O[i, j], N[i, j] = q.hit_b.iloc[0] * 100, q.hit_o.iloc[0] * 100, q.purchases.iloc[0]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", [SURF, light(BEST, 0.5), BEST, DEEP])
    im = ax.imshow(M_, cmap=cmap, vmin=0, vmax=22, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v, ov, n = M_[i, j], O[i, j], N[i, j]
            c = "white" if v > 13 else INK
            ax.text(j, i - 0.12, f"{v:.1f}%", ha="center", va="center", fontsize=15, fontweight="bold", color=c)
            ax.text(j, i + 0.22, f"was {ov:.1f}% · n={int(n):,}", ha="center", va="center", fontsize=9.5, color=c if v > 13 else INK2)
    ax.set_xticks(range(len(cols)), [c[2] for c in cols]); ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURF, lw=3); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Best system: purchase in top 10 (%)"); cb.outline.set_visible(False)
    fig.suptitle("User × product: the sweet spot is a fresh product from a known store, for a phone you own", x=0.02, ha="left", fontsize=14.5, color=INK, y=1.03)
    fig.text(0.02, -0.03, "Validation, best system (DIN, mixed list + combined score); 'was' = original system. Mean of 3 seeds. Phone labels from titles (approximate); small cells are noisy.",
             fontsize=9.5, color=INK2)
    save(fig, "fig17_user_x_product")


def phone(ax, x, y, w=0.5, hgt=0.9, c=INK2):
    ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=c, lw=2))
    ax.add_patch(Rectangle((x + 0.06, y + 0.12), w - 0.12, hgt - 0.24, fc=light(c, 0.85), ec="none"))


def fig18():
    b = pd.read_csv(ROOT / "results/targeted_study/behaviour-failures-20261108/behaviour_failures.csv")
    b = b[b.cut == "prior purchases for this device"].set_index("group")
    order = [("0", 0), ("1", 1), ("2", 2), ("3-4", 3), ("5+", 5)]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for k, (g, n) in enumerate(order):
        x = k * 2.0
        hb, ho = b.loc[g, "best_hit10"] * 100, b.loc[g, "orig_hit10"] * 100
        ax.bar(x + 0.25, hb, 0.7, color=BEST, zorder=2); ax.bar(x - 0.5, ho, 0.7, color=ORIG, zorder=2)
        ax.text(x + 0.25, hb + 0.12, f"{hb:.1f}%", ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x - 0.5, ho + 0.12, f"{ho:.1f}%", ha="center", fontsize=10.5, color=INK2)
        phone(ax, x - 0.45, -2.2, 0.55, 1.2)
        for a in range(min(n, 5)):
            ax.add_patch(FancyBboxPatch((x + 0.2 + (a % 3) * 0.24, -1.9 + (a // 3) * 0.42), 0.18, 0.3, boxstyle="round,pad=0.01,rounding_size=0.04", fc=BEST, ec="none"))
        ax.text(x - 0.12, -2.6, {"3-4": "3–4", "5+": "5+"}.get(g, g) + " earlier", ha="center", va="top", fontsize=11, color=INK)
        ax.text(x - 0.12, -3.75, f"{b.loc[g, 'share'] * 100:.0f}% of purchases", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlim(-1.3, 9.3); ax.set_ylim(-4.3, 7)
    ax.set_yticks([0, 2, 4, 6]); ax.set_ylabel("Purchase in the top 10 (%)", loc="top")
    ax.spines["bottom"].set_visible(False); ax.set_xticks([])
    ax.legend(handles=[Patch(color=ORIG, label="Original"), Patch(color=BEST, label="Best")], loc="upper left", frameon=False, ncol=2)
    fig.suptitle("Each earlier purchase for the same phone makes the next one easier to recommend (up to 3–4)", x=0.02, ha="left", fontsize=15, color=INK)
    fig.text(0.02, -0.02, "Validation. Earlier purchases = items in the user's last 50 that name the same phone model as the purchase (from titles, approximate).", fontsize=9.5, color=INK2)
    save(fig, "fig18_repeat_staircase")


def waffle(ax, parts, title):
    cells = []
    for lab, v, c in parts:
        cells += [c] * int(round(v))
    cells = (cells + [parts[0][2]] * 100)[:100]
    cells = cells[::-1]
    for k, c in enumerate(cells):
        ax.add_patch(FancyBboxPatch((k % 10, k // 10), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.15", fc=c, ec="none"))
    ax.set_xlim(-0.2, 10); ax.set_ylim(-0.2, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12, color=INK, loc="left")


def fig19():
    b = pd.read_csv(ROOT / "results/targeted_study/behaviour-failures-20261108/behaviour_failures.csv").set_index(["cut", "group"])
    p = pd.read_csv(ROOT / "results/targeted_study/product-archetypes-20261108/product_archetypes.csv").set_index(["cut", "group"])
    sel = [("Store bought from before", p.loc[("store vs history", "same store as history")]),
           ("Same phone as before", b.loc[("behaviour", "same device")]),
           ("Upgrade to a newer model", b.loc[("behaviour", "upgrade")]),
           ("Switch iPhone ↔ Android", b.loc[("behaviour", "switch platform")])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9))
    for ax, (name, r) in zip(axes, sel):
        hit = r.best_hit10 * 100; reach = (1 - r.best_retrieval_miss) * 100
        parts = [("never", 100 - reach, light(MUTED, 0.35)), ("reach", reach - hit, light(BEST, 0.55)), ("top10", hit, BEST)]
        waffle(ax, parts, name)
        ax.text(0, -0.7, f"{hit:.0f} in top 10 · {reach - hit:.0f} below 10\n{100 - reach:.0f} never reach the ranker", fontsize=10.5, color=INK2, va="top")
    fig.legend(handles=[Patch(color=BEST, label="In the top 10"), Patch(color=light(BEST, 0.55), label="Reaches the ranker, ranked below 10"),
                        Patch(color=light(MUTED, 0.35), label="Never reaches the ranker")], loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Out of 100 purchases, best system", x=0.01, ha="left", fontsize=15, color=INK, y=1.0)
    fig.text(0.01, -0.03, "Validation, mean of 3 seeds; each square = 1 purchase in 100 (rounded). Labels from titles (approximate).", fontsize=9.5, color=INK2)
    fig.subplots_adjust(top=0.78, wspace=0.12)
    save(fig, "fig19_waffles")


def fig20():
    l = pd.read_csv(D / "launches.csv"); l["date"] = pd.to_datetime(l.month) + pd.Timedelta(days=14)
    launch = {"iPhone 13": ("2021-09-14", "announced\n14 Sep 2021"), "Galaxy S22": ("2022-02-09", "announced\n9 Feb 2022"), "Pixel 6": ("2021-10-19", "announced\n19 Oct 2021")}
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True)
    for ax, fam in zip(axes, ["iPhone 13", "Galaxy S22", "Pixel 6"]):
        q = l[l.family == fam]
        ld = pd.Timestamp(launch[fam][0])
        ax.axvspan(pd.Timestamp("2021-08-01"), ld, color=light(WARN, 0.8), lw=0)
        ax.plot(q.date, q.orig * 1000, color=ORIG, lw=2, marker="o", ms=5); ax.plot(q.date, q.best * 1000, color=BEST, lw=2, marker="o", ms=5)
        ax.axvline(ld, color=INK2, lw=1.2, ls="--"); ax.text(ld + pd.Timedelta(days=4), ax.get_ylim()[1] * 0.92, launch[fam][1], fontsize=10, color=INK2, va="top")
        ax.set_ylabel("nDCG@10 ×1000"); ax.set_title(f"{fam} accessories", loc="left", fontsize=12.5, color=INK)
        ax.grid(axis="y", color=GRID); ax.set_ylim(0, None)
    axes[0].legend(handles=[Patch(color=ORIG, label="Original"), Patch(color=BEST, label="Best"), Patch(color=light(WARN, 0.8), label="before announcement: titles edited later")],
                   loc="upper right", frameon=False, fontsize=10, ncol=3)
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[8, 10, 12, 2, 4, 6])); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Launch waves: the best system's edge jumps when a new phone arrives", x=0.02, ha="left", fontsize=15, color=INK)
    fig.text(0.02, -0.01, "Validation, monthly (≥150 purchases per point), mean of 3 seeds. Purchases before an announcement whose title names the phone exist because sellers edit titles.",
             fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig20_launch_waves")


def fig21():
    e = json.loads((D / "upgrader_example.json").read_text())
    up = pd.read_csv(ROOT / "results/targeted_study/behaviour-failures-20261108/upgrader_candidates.csv")
    sh = up[(up.group == "upgrade") & (up.source == "recency")][["old phone", "new model"]].mean() * 100
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0, 0, 1, 0.9]); ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 0.4), 3.6, 5.6, boxstyle="round,pad=0.05,rounding_size=0.25", fc=CARD, ec=GRID))
    ax.text(0.45, 5.6, "Owned before", fontsize=13, fontweight="bold", color=INK)
    for k, t in enumerate(e["history"][-2:]):
        import textwrap
        w = textwrap.wrap(t, 30)[:2]
        ax.text(0.45, 4.9 - k * 0.9, "• " + "\n   ".join(w) + "…", fontsize=10.5, color=INK, va="top")
    ax.text(0.45, 2.6, "Bought next", fontsize=13, fontweight="bold", color=INK)
    import textwrap
    ax.text(0.45, 2.15, "\n".join(textwrap.wrap(e["bought"], 30)[:2]) + "…", fontsize=10.5, color=INK, va="top", fontweight="bold")
    ax.text(0.45, 1.0, f"{e['date']} · an upgrade:\niPhone 6s Plus → iPhone XS Max", fontsize=10, color=INK2)
    ax.annotate("", (4.6, 3.3), (3.9, 3.3), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2))
    ax.add_patch(FancyBboxPatch((4.7, 0.25), 5.2, 6.0, boxstyle="round,pad=0.05,rounding_size=0.35", fc="white", ec=INK2, lw=2.5))
    ax.text(4.95, 5.85, "Recency retriever: 10 closest products", fontsize=12.5, fontweight="bold", color=INK)
    for k, t in enumerate(e["top10"]):
        y = 5.35 - k * 0.5
        c = REC if t["kind"] == "old phone" else BEST if t["kind"] == "new phone" else MUTED
        ax.add_patch(FancyBboxPatch((4.95, y - 0.17), 0.95, 0.34, boxstyle="round,pad=0,rounding_size=0.08", fc=c, ec="none"))
        ax.text(5.42, y, "old phone" if t["kind"] == "old phone" else t["kind"], fontsize=8.5, color="white", ha="center", va="center", fontweight="bold")
        ax.text(6.0, y, t["title"][:44] + "…", fontsize=9.3, color=INK, va="center")
    ax.add_patch(FancyBboxPatch((10.4, 1.4), 3.4, 3.6, boxstyle="round,pad=0.05,rounding_size=0.25", fc=light(REC, 0.85), ec="none"))
    ax.text(10.65, 4.55, "Across 2,000 upgrades", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(10.65, 3.7, f"{sh['old phone']:.0f}%", fontsize=30, fontweight="bold", color=REC)
    ax.text(12.15, 3.85, "of candidates are\nfor the OLD phone", fontsize=10.5, color=INK, va="center")
    ax.text(10.65, 2.35, f"{sh['new model']:.0f}%", fontsize=30, fontweight="bold", color=BEST)
    ax.text(12.15, 2.5, "are for the phone\nthey actually bought", fontsize=10.5, color=INK, va="center")
    ax.text(10.65, 1.65, "The purchase never reached the ranker.", fontsize=10, color=INK2)
    fig.suptitle("Why upgrades fail: your history fills the list with accessories for the phone you are replacing", x=0.02, ha="left", fontsize=14.5, color=INK)
    fig.text(0.02, -0.02, "Validation. One typical upgrade (chosen by fixed random draw among iPhone-only upgraders), next to the 2,000-purchase statistics. Titles shortened.",
             fontsize=9.5, color=INK2)
    save(fig, "fig21_upgrade_list")


def fig22():
    f = json.loads((D / "future_title.json").read_text())
    import re
    fig = plt.figure(figsize=(13, 5.2)); ax = fig.add_axes([0, 0, 1, 0.88]); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 3.4, 4.6, boxstyle="round,pad=0.05,rounding_size=0.25", fc=light(WARN, 0.75), ec="none"))
    ax.text(0.5, 3.7, f"{f['count']}", fontsize=44, fontweight="bold", color=INK)
    ax.text(0.5, 2.95, "purchases made before\nthe iPhone 13 was announced\n(14 Sep 2021) have “iPhone 13”\nin the product title", fontsize=11.5, color=INK, va="top")
    for k, e in enumerate(f["examples"][:4]):
        y = 4.4 - k * 1.05
        ax.add_patch(FancyBboxPatch((4.0, y - 0.38), 8.8, 0.8, boxstyle="round,pad=0.03,rounding_size=0.12", fc=CARD, ec=GRID))
        ax.text(4.2, y + 0.12, f"bought {e['date']}", fontsize=10, color=INK2)
        t = e["title"][:92]
        m = re.search(r"iphone\s*13", t, re.I)
        pre, hit, post = t[:m.start()], t[m.start():m.end()], t[m.end():]
        x = 4.2
        for seg, kw in ((pre, {}), (hit, {"fontweight": "bold", "color": "#b35c00"}), (post + "…", {})):
            txt = ax.text(x, y - 0.18, seg, fontsize=10.5, color=kw.get("color", INK), fontweight=kw.get("fontweight", "normal"))
            fig.canvas.draw(); bb = txt.get_window_extent().transformed(ax.transData.inverted()); x = bb.x1
    fig.suptitle("The data remembers the future: sellers edit titles to add new phones", x=0.02, ha="left", fontsize=15, color=INK)
    fig.text(0.02, -0.02, "Validation. Product titles are today's metadata, not as of purchase, so title-based labels (and text features) carry some future information.",
             fontsize=9.5, color=INK2)
    save(fig, "fig22_future_titles")


def fig23():
    lat = json.loads((ROOT / "results/latency/20261108-final-systems/latency.json").read_text())["results"]
    t = pd.read_csv(ROOT / "results/test_final_ranker/20261108/levels.csv"); t = t[t.group == "all"].set_index("system").ndcg10 * 1000
    pts = [("Original\nMLP, two-tower list", lat["original_mlp"]["total"]["median"], t["mlp_tt"], ORIG, True),
           ("DIN, two-tower list", lat["original_din"]["total"]["median"], t["din_tt"], ORIG, True),
           ("MLP, mixed list", lat["best_mlp"]["total"]["median"], t["mlp_rrf"], BEST, True),
           ("Best\nDIN, mixed list", lat["best_din"]["total"]["median"], t["din_rrf"], BEST, True),
           ("Best, faster combined score\n(accuracy not measured)", lat["best_fast_din"]["total"]["median"], t["din_rrf"], BEST, False)]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axvspan(0, 50, color=light(BEST, 0.88), lw=0); ax.text(49, 4.75, "real-time budget (<50 ms)", ha="right", fontsize=10, color=INK2)
    offs = [(6, -18), (6, 6), (6, -18), (6, 6), (-6, 14)]
    for (lab, x, y, c, filled), (dx, dy) in zip(pts, offs):
        ax.scatter([x], [y], s=150, color=c if filled else SURF, edgecolor=c, lw=2.5, zorder=3)
        ax.annotate(lab, (x, y), (dx, dy), textcoords="offset points", fontsize=10.5, color=INK, ha="left" if dx > 0 else "right")
    ax.set_xlim(0, 55); ax.set_ylim(4.5, 9)
    ax.set_xlabel("Median latency per user, single CPU thread (ms)"); ax.set_ylabel("Test nDCG@10 ×1000")
    ax.grid(color=GRID); ax.set_axisbelow(True)
    fig.suptitle("+59% accuracy for ~14–28 ms more per user", x=0.02, ha="left", fontsize=15, color=INK)
    fig.text(0.02, -0.02, "Test accuracy (mean of 3 seeds); latency on a laptop CPU, 1,000 Validation queries, K₁ = 200, exact search. The hollow point reuses the best system's accuracy, which was not measured for that variant.",
             fontsize=9, color=INK2)
    save(fig, "fig23_accuracy_vs_speed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig15, fig16, fig17, fig18, fig19, fig20, fig21, fig22, fig23):
        f(); print("done", f.__name__)


if __name__ == "__main__":
    main()
