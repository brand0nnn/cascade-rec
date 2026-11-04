"""Figures for the main findings (Val only, exploratory), drawn from saved runs.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.plot_findings --out results/figures/20261029-findings

Every number is read from a saved run listed in SOURCES (or recomputed from saved candidate lists, for the
training-positive head share); each figure is written as PNG plus a CSV of exactly the plotted values.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascade.utils.io import REPO_ROOT, git_hash

# Reference palette (dataviz skill, light mode), validated all-pairs for these three slots.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
FORCED, NATURAL = ORANGE, BLUE
K1S = (50, 100, 200, 500)
ITEM_TYPES = (("item_type=new-in-period", "New products"), ("item_type=warm-tail", "Tail"),
              ("item_type=warm-head", "Popular (head)"))

SOURCES = {
    "k1_forced_mlp": "results/ranker_k1_sweep/20261025-k1/summary.csv",
    "k1_natural_mlp": "results/ranker_k1_sweep/20261028-k1-natural/summary.csv",
    "k1_natural_mlp_k500": "results/index_rank_eval/20261028-k1-ktrain500-r2/results.json",
    "k1_natural_din": "results/index_rank_eval/20261028-k1-din-natural/results.json",
    "k1_natural_din_k500": "results/index_rank_eval/20261028-k1-din-ktrain500/results.json",
    "mlp_compare": "results/ranker_compare/20261028-natural-vs-forced-mlp/comparison.json",
    "din_compare": "results/ranker_compare/20261028-din-natural-vs-din-forced/comparison.json",
    "rank_response": "results/analysis/20261028-permutation-importance-r2/results.json",
    "ivf_din_k500": "results/index_rank_eval/20261028-index-rank-din-ktrain500/results.json",
    "debias_din": "results/index_rank_eval/20261028-debiased-rank-din/results.json",
    "popularity_buckets": "results/analysis/20261028-staleness-popularity/a_by_rtrain_bucket.csv",
    "last_item": "results/analysis/20261028-last-item/comparison_last_item.csv",
    "train_lists": "data/processed/candidates/two_tower_text_seed0/ranker_train.npz",
    "policy_metrics": "results/targeted_study/policy-report-20261029/metrics.csv",
    "policy_comparisons": "results/targeted_study/policy-report-20261029/comparisons.csv",
    "policy_score": "results/targeted_study/policy-report-20261029/score_diagnostics.csv",
}
# Policy-study arms: (label, colour, hatched = retrieval score masked).
ARMS = {"A": ("A  all lists, score on", FORCED, False), "B": ("B  all lists, score masked", FORCED, True),
        "C": ("C  natural lists, score on", NATURAL, False), "D": ("D  natural lists, score masked", NATURAL, True)}


def _load(key: str):
    path = REPO_ROOT / SOURCES[key]
    return pd.read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text())


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2,
        "axes.axisbelow": True, "figure.dpi": 150})


def _save(fig, out: Path, name: str, table: pd.DataFrame, note: str) -> None:
    fig.text(0.01, -0.03, textwrap.fill(note, 150), fontsize=7.5, color=INK_2, ha="left", va="top")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    table.to_csv(out / f"{name}.csv", index=False)


def _k1_from_eval(key: str) -> list[float]:
    by = _load(key)["by_config"]
    return [by[f"two_tower_text_seed0@{k}"]["all"]["ndcg10_macro_mean"] for k in K1S]


def fig_k1(out: Path) -> None:
    forced, natural = _load("k1_forced_mlp").set_index("k1"), _load("k1_natural_mlp").set_index("k1")
    series = [  # (label, values, colour, emphasised)
        ("MLP, forced lists (team)", [forced.loc[k, "ndcg10_macro_mean"] for k in K1S], FORCED, True),
        ("MLP, natural lists", [natural.loc[k, "ndcg10_macro_mean"] for k in K1S], NATURAL, True),
        ("MLP, natural, K_train 500", _k1_from_eval("k1_natural_mlp_k500"), GREY, False),
        ("DIN, natural lists", _k1_from_eval("k1_natural_din"), GREY, False),
        ("DIN, natural, K_train 500", _k1_from_eval("k1_natural_din_k500"), AQUA, True),
    ]
    recall = [natural.loc[k, "retrieval_recall_macro"] for k in K1S]
    r1 = _load("mlp_compare")["overall"]["natural"]["r1_ndcg10_macro"]
    fig, (ax_r, ax) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.9]})
    ax_r.plot(K1S, recall, color=INK_2, marker="o", markersize=5)
    for k, v in zip(K1S, recall):
        last = k == K1S[-1]
        ax_r.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(-8, 0) if last else (6, -10),
                      ha="right" if last else "left", va="center" if last else "baseline", fontsize=8, color=INK_2)
    ax_r.set(xscale="log", xticks=K1S, xticklabels=[str(k) for k in K1S], ylim=(0, 0.14),
             xlabel="K₁ (candidates passed to the ranker)", ylabel="Retrieval Recall@K₁")
    ax_r.set_title("Retrieval ceiling rises 4×")
    ax.axhline(r1, color=GREY, linewidth=1)
    ax.text(52, r1 + 0.00012, f"retrieval order, no ranker ({r1:.4f})", fontsize=8, color=INK_2)
    for label, values, colour, strong in series:
        ax.plot(K1S, values, color=colour, marker="o", markersize=5 if strong else 4,
                linewidth=2 if strong else 1.2, label=label, zorder=3 if strong else 2)
        ax.annotate(label, (K1S[-1], values[-1]), textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=8, color=INK if strong else INK_2)
    ax.annotate("collapse: the ranker promotes\nlow-score candidates", (500, series[0][1][-1]),
                textcoords="offset points", xytext=(-150, 6), fontsize=8, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.8})
    ax.set(xscale="log", xticks=K1S, xticklabels=[str(k) for k in K1S], xlim=(45, 560), ylim=(0, 0.0115),
           xlabel="K₁ (candidates passed to the ranker)", ylabel="End-to-end nDCG@10 (Val, per-user macro)")
    ax.set_title("Whether more candidates help depends on how the ranker was trained")
    ax.legend(loc="lower left", fontsize=8)
    for a in (ax_r, ax):  # after the log scale is set, which resets the locators
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    table = pd.DataFrame({"k1": K1S, "retrieval_recall_at_k1": recall,
                          **{label: values for label, values, _, _ in series}})
    _save(fig, out, "fig1_k1_by_ranker_training", table,
          "Val only, exploratory. Mean of ranker seeds 0–2; frozen Two-tower (text) seed-0 retriever, exact search. "
          "Sources: results/ranker_k1_sweep/*, results/index_rank_eval/20261028-k1-*.")


def fig_rank_response(out: Path) -> None:
    models = _load("rank_response")["models"]
    curves = {}
    for key, block in models.items():
        kind = key.split("/")[0]
        curves.setdefault(kind, []).append(block["score_response"]["mean_centred_logit_by_retrieval_rank_decile"])
    mean = {k: np.mean(v, axis=0) for k, v in curves.items()}
    deciles = np.arange(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, model in zip(axes, ("mlp", "din")):
        for kind, colour, label in ((f"forced_{model}", FORCED, "forced lists"), (f"natural_{model}", NATURAL, "natural lists")):
            ax.plot(deciles, mean[kind], color=colour, marker="o", markersize=5, label=label)
            ax.annotate(label, (10, mean[kind][-1]), textcoords="offset points", xytext=(8, 0), va="center",
                        fontsize=8, color=INK)
        ax.axhline(0, color=GREY, linewidth=1)
        ax.set(xticks=deciles, xlim=(0.6, 12.2), xlabel="Retrieval-rank decile within the top-200 list (1 = top)")
        ax.set_title(f"{model.upper()}")
        ax.legend(loc="upper center", fontsize=8)
    axes[0].set_ylabel("Mean ranker logit, centred within list")
    fig.suptitle("Forced-list rankers boost the bottom of the list too: the shortcut", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", y=1.03)
    table = pd.DataFrame({"decile": deciles, **{k: v for k, v in mean.items()}})
    _save(fig, out, "fig2_rank_response_shortcut", table,
          "Val natural-hit lists (2,000 per checkpoint), mean of seeds 0–2. "
          "Source: results/analysis/20261028-permutation-importance-r2 (score_response).")


def fig_rankers(out: Path) -> None:
    mlp, din = _load("mlp_compare")["overall"], _load("din_compare")["overall"]

    def at200(key):
        block = _load(key)["by_config"]["two_tower_text_seed0@200"]["all"]
        return block["ndcg10_macro_mean"], block["ndcg10_macro_seed_sd"]

    rows = [("Retrieval order (R1), no ranker", mlp["natural"]["r1_ndcg10_macro"], 0.0, GREY),
            ("DIN, forced lists (team)", din["din_forced"]["ndcg10_macro_mean"], din["din_forced"]["ndcg10_macro_seed_sd"], FORCED),
            ("MLP, forced lists (team)", mlp["forced"]["ndcg10_macro_mean"], mlp["forced"]["ndcg10_macro_seed_sd"], FORCED),
            ("MLP, natural lists", mlp["natural"]["ndcg10_macro_mean"], mlp["natural"]["ndcg10_macro_seed_sd"], NATURAL),
            ("MLP, natural, K_train 500", *at200("k1_natural_mlp_k500"), NATURAL),
            ("DIN, natural lists", din["din_natural"]["ndcg10_macro_mean"], din["din_natural"]["ndcg10_macro_seed_sd"], NATURAL),
            ("DIN, natural, K_train 500", *at200("k1_natural_din_k500"), NATURAL)]
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, sd, colour) in zip(y, rows):
        ax.barh(yi, value, height=0.62, color=colour)
        if sd:
            ax.errorbar(value, yi, xerr=sd, color=INK_2, capsize=2, linewidth=1)
        ax.text(value + sd + 0.00012, yi, f"{value:.4f}", va="center", fontsize=8, color=INK)
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xlim=(0, 0.0118),
           xlabel="End-to-end nDCG@10 at K₁ = 200 (Val, per-user macro)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Training on natural lists: +26% (MLP), +53% (DIN); DIN overtakes the MLP")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (FORCED, NATURAL, GREY)]
    ax.legend(handles, ["forced lists", "natural lists", "no ranker"], loc="upper right", fontsize=8)
    table = pd.DataFrame([{"system": r[0], "ndcg10_macro_mean": r[1], "seed_sd": r[2]} for r in rows])
    _save(fig, out, "fig3_ranker_comparison", table,
          "Val only, exploratory. Bars: mean of seeds 0–2; whiskers: seed SD. Identical K₁ = 200 exact candidate lists. "
          "Sources: results/ranker_compare/*, results/index_rank_eval/20261028-k1-*.")


def _training_positive_groups() -> dict[str, dict[str, float]]:
    """Ranker-training positives by item group, as in the 28 Oct report (H6): new-in-period = first interaction at or
    after t_ranker_train; otherwise warm-head / warm-tail by the §4.7 segment."""
    from cascade.data.instances import ProcessedData
    from cascade.pipelines.build_candidates import load_candidates
    data = ProcessedData.load("configs/data.yaml")
    lists = load_candidates(REPO_ROOT / Path(SOURCES["train_lists"]).parent, "ranker_train")
    rows = np.arange(len(lists["inst"]))
    target = np.asarray(lists["cand"])[rows, np.asarray(lists["target_pos"])]
    items = data.items.set_index("item").loc[target]
    new = items["first_ts"].to_numpy() >= data.split["t_ranker_train"]  # segments.split_start covers Val/Test only
    group = np.where(new, "new", np.where(items["segment"].astype(str).to_numpy() == "head", "head", "tail"))
    forced = np.asarray(lists["forced"], dtype=bool)
    return {kind: {g: float(np.mean(group[mask] == g)) for g in ("new", "tail", "head")}
            for kind, mask in (("forced", forced), ("natural", ~forced))}


def fig_new_products(out: Path) -> None:
    by = _load("mlp_compare")["by_group"]
    diffs = [by[g]["natural_minus_forced"] for g, _ in ITEM_TYPES]
    groups = _training_positive_groups()
    share = {kind: g["head"] for kind, g in groups.items()}
    fig, (ax_s, ax) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    for xi, (kind, colour) in enumerate((("forced", FORCED), ("natural", NATURAL))):
        value = share[kind]
        ax_s.bar(xi, value, width=0.55, color=colour)
        ax_s.text(xi, value + 0.02, f"{value:.0%} popular", ha="center", fontsize=9, color=INK)
        ax_s.text(xi, 0.03, f"new: {groups[kind]['new']:.0%}", ha="center", fontsize=8, color=SURFACE)
    ax_s.set(xticks=[0, 1], xticklabels=["forced lists", "natural lists"], ylim=(0, 1),
             ylabel="Training positives that are popular")
    ax_s.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_s.grid(axis="x", visible=False)
    ax_s.set_title("Natural positives are mostly popular")
    x = np.arange(len(ITEM_TYPES))
    for xi, d in zip(x, diffs):
        colour = NATURAL if d["diff"] >= 0 else FORCED
        ax.bar(xi, d["diff"], width=0.55, color=colour)
        ax.errorbar(xi, d["diff"], yerr=[[d["diff"] - d["ci_low"]], [d["ci_high"] - d["diff"]]], color=INK_2, capsize=3, linewidth=1)
        ax.text(xi, d["ci_high"] + 0.0002 if d["diff"] >= 0 else d["ci_low"] - 0.0002, f"{d['diff']:+.5f}",
                ha="center", va="bottom" if d["diff"] >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set(xticks=x, xticklabels=[name for _, name in ITEM_TYPES], ylim=(-0.0016, 0.0052),
           ylabel="Δ nDCG@10, natural − forced MLP")
    ax.grid(axis="x", visible=False)
    ax.set_title("…so the gain is on popular products, with a loss on new and tail ones")
    table = pd.DataFrame([{"group": name, **d} for (_, name), d in zip(ITEM_TYPES, diffs)]
                         + [{"group": f"training positives, {kind} lists, {g}", "diff": v}
                            for kind, shares in groups.items() for g, v in shares.items()])
    _save(fig, out, "fig4_natural_lists_popularity_bias", table,
          "Val only, exploratory. Left: ranker-training positives; new = first reviewed in the ranker-training period, "
          "popular = §4.7 head among the rest. Right: paired user-bootstrap 95% CI, seed-mean MLPs. "
          "Sources: results/ranker_compare/20261028-natural-vs-forced-mlp; ranker_train.npz.")


def fig_looser_retrieval(out: Path) -> None:
    k1 = _load("k1_natural_din_k500")["by_config"]["two_tower_text_seed0@500"]
    ivf = _load("ivf_din_k500")["by_config"]["ivf_nprobe32"]
    deb = _load("debias_din")["by_config"]["beta0.25_minseen"]
    panels = [("K₁ 500 vs 200", k1, "minus_two_tower_text_seed0@200", "DIN, natural, K_train 500"),
              ("Approximate search: IVF-32 vs exact", ivf, "minus_exact", "DIN, natural, K_train 500"),
              ("Popularity correction (β = 0.25) vs none", deb, "minus_beta0", "DIN, natural lists")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    rows = []
    for ax, (title, block, key, model) in zip(axes, panels):
        x = np.arange(len(ITEM_TYPES))
        for xi, (group, name) in zip(x, ITEM_TYPES):
            d = block[group][key]
            colour = NATURAL if d["diff"] >= 0 else FORCED
            ax.bar(xi, d["diff"], width=0.55, color=colour)
            ax.errorbar(xi, d["diff"], yerr=[[d["diff"] - d["ci_low"]], [d["ci_high"] - d["diff"]]], color=INK_2,
                        capsize=3, linewidth=1)
            rows.append({"change": title, "ranker": model, "group": name, **d})
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set(xticks=x, xticklabels=["New", "Tail", "Popular"])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5)
        ax.text(0.02, 0.03, model, transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom")
    axes[0].set_ylabel("Δ end-to-end nDCG@10")
    fig.suptitle("Loosening retrieval moves recommendations from popular to new and tail products",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    _save(fig, out, "fig5_looser_retrieval_shifts_to_new", pd.DataFrame(rows),
          "Val only, exploratory; probes, no setting selected. Paired user-bootstrap 95% CI, seed-mean rankers. "
          "Sources: results/index_rank_eval/20261028-{k1-din-ktrain500, index-rank-din-ktrain500, debiased-rank-din}.")


def fig_retriever_popularity(out: Path) -> None:
    buckets = _load("popularity_buckets")
    last = _load("last_item").set_index("group")
    fig, (ax_b, ax) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    labels = [b.split(":")[1] for b in buckets["rtrain_bucket"]]
    x = np.arange(len(buckets))
    v, lo, hi = buckets["recall200_macro"], buckets["recall200_item_ci_lo"], buckets["recall200_item_ci_hi"]
    ax_b.bar(x, v, width=0.6, color=BLUE)
    ax_b.errorbar(x, v, yerr=[v - lo, hi - v], color=INK_2, capsize=2, linewidth=1, fmt="none")
    for xi, value in zip(x, v):
        ax_b.text(xi, value + 0.03, f"{value:.2f}", ha="center", fontsize=8, color=INK)
    ax_b.set(xticks=x, ylim=(0, 1), xlabel="Target's reviews in the retriever-training period",
             ylabel="Retriever Recall@200 (warm targets)")
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.grid(axis="x", visible=False)
    ax_b.set_title("The retriever finds what was popular when it trained")
    groups = [("item_new-in-period", "New"), ("item_warm-tail", "Tail"), ("item_warm-head", "Popular")]
    x = np.arange(len(groups))
    width = 0.36
    for offset, column, colour, label in ((-width / 2, "frozen_recall@200", BLUE, "trained retriever"),
                                          (width / 2, "scorer_recall@200", AQUA, "similar to last product (no training)")):
        values = [last.loc[g, column] for g, _ in groups]
        ax.bar(x + offset, values, width=width - 0.03, color=colour, label=label)
        for xi, value in zip(x + offset, values):
            ax.text(xi, value + 0.003, f"{value:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set(xticks=x, xticklabels=[name for _, name in groups], ylim=(0, 0.175), ylabel="Recall@200")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("A no-training baseline wins on new and tail")
    table = pd.concat([buckets.assign(panel="recall by training-period count"),
                       last.reset_index().assign(panel="trained vs last-product scorer")], ignore_index=True)
    _save(fig, out, "fig6_retriever_popularity", table,
          "Val only, exploratory; frozen Two-tower (text) seed 0. Left: item-cluster bootstrap 95% CI. Sources: "
          "results/analysis/20261028-staleness-popularity, results/analysis/20261028-last-item.")


def fig_policy_study(out: Path) -> None:
    metrics, comps, score = _load("policy_metrics"), _load("policy_comparisons"), _load("policy_score")
    overall = metrics[(metrics.axis == "overall") & (metrics.group == "all")]
    fig, ((ax_a, ax_k), (ax_g, ax_s)) = plt.subplots(2, 2, figsize=(11, 7.6))
    rows = []

    # (a) K1 = 200 quality per arm: bar = seed mean, dots = seeds.
    at200 = overall[overall.k1 == 200]
    for x, (arm, (label, colour, masked)) in enumerate(ARMS.items()):
        values = at200[at200.arm == arm].sort_values("seed").ndcg10.to_numpy()
        ax_a.bar(x, values.mean(), width=0.62, color=colour, alpha=0.45 if masked else 1, hatch="//" if masked else None,
                 edgecolor=SURFACE if not masked else colour, linewidth=0)
        ax_a.scatter([x - 0.12, x, x + 0.12], values, s=14, color=INK, zorder=3)
        ax_a.text(x, values.max() + 0.0003, f"{values.mean():.4f}", ha="center", fontsize=8.5, color=INK)
        rows += [{"panel": "a", "arm": arm, "seed": s, "k1": 200, "value": v} for s, v in zip((0, 1, 2), values)]
    ax_a.set(xticks=range(4), xticklabels=["A\nall · on", "B\nall · masked", "C\nnatural · on", "D\nnatural · masked"],
             ylim=(0, 0.0105), ylabel="End-to-end nDCG@10 at K₁ = 200")
    ax_a.grid(axis="x", visible=False)
    ax_a.set_title("(a) Natural lists win, with or without the score")

    # (b) K1 sweep per arm, seed mean.
    for arm, (label, colour, masked) in ARMS.items():
        curve = overall[overall.arm == arm].groupby("k1").ndcg10.mean().reindex(K1S)
        ax_k.plot(K1S, curve, color=colour, linestyle="--" if masked else "-", marker="o", markersize=4.5)
        ax_k.annotate(arm, (K1S[-1], curve.iloc[-1]), textcoords="offset points", xytext=(7, 0), va="center",
                      fontsize=9, fontweight="bold", color=INK)
        rows += [{"panel": "b", "arm": arm, "k1": k, "value": v} for k, v in curve.items()]
    ax_k.set(xscale="log", xticks=K1S, xticklabels=[str(k) for k in K1S], xlim=(42, 640), ylim=(0, 0.0105),
             xlabel="K₁ (candidates passed to the ranker)", ylabel="End-to-end nDCG@10")
    ax_k.minorticks_off()
    ax_k.set_title("(b) Only all-list training collapses as K₁ grows")

    # (c) C - A by product group, paired user-bootstrap 95% CI.
    groups = [("overall", "all", "All products"), ("item_type", "new-in-period", "New products"),
              ("item_type", "warm-tail", "Tail"), ("item_type", "warm-head", "Popular (head)")]
    ca = comps[comps.comparison == "C-A"]
    for y, (axis, group, label) in zip(range(len(groups))[::-1], groups):
        r = ca[(ca.axis == axis) & (ca.group == group)].iloc[0]
        ax_g.errorbar(r["diff"], y, xerr=[[r["diff"] - r.ci_low], [r.ci_high - r["diff"]]], fmt="o", color=NATURAL,
                      markersize=7, capsize=3, linewidth=2)
        ax_g.text(r["diff"], y + 0.2, f"{r['diff']:+.5f}".replace("-", "−"), va="bottom", ha="center",
                  fontsize=8.5, color=INK)
        rows.append({"panel": "c", "arm": "C-A", "group": label, "value": r["diff"], "ci_low": r.ci_low,
                     "ci_high": r.ci_high})
    ax_g.axvline(0, color=INK_2, linewidth=1)
    ax_g.set(yticks=range(len(groups))[::-1], yticklabels=[g[2] for g in groups], ylim=(-0.6, 3.6),
             xlim=(-0.0016, 0.0062), xlabel="C − A: change in nDCG@10 (95% CI; right = natural lists better)")
    ax_g.grid(axis="y", visible=False)
    ax_g.set_title("(c) The gain is on popular products; new products lose")

    # (d) Shortcut: share of top-10 slots taken by retrieval ranks 201-500 when ranking 500 candidates.
    measured = score[score.status == "measured"]
    for x, arm in enumerate(("A", "C")):
        values = measured[measured.arm == arm].sort_values("seed").added_top10_share.to_numpy() * 100
        label, colour, _ = ARMS[arm]
        ax_s.bar(x, values.mean(), width=0.55, color=colour)
        ax_s.scatter([x - 0.1, x, x + 0.1], values, s=14, color=INK, zorder=3)
        ax_s.text(x, values.max() + 3, f"{values.mean():.0f}%", ha="center", fontsize=9.5, color=INK)
        rows += [{"panel": "d", "arm": arm, "seed": s, "k1": 500, "value": v} for s, v in zip((0, 1, 2), values)]
    ax_s.axhline(60, color=GREY, linewidth=1, linestyle="--")
    ax_s.text(1.45, 61.5, "60% = if retrieval\nrank were ignored", fontsize=8, color=INK_2, ha="right", va="bottom")
    ax_s.set(xticks=[0, 1], xticklabels=["A\nall lists", "C\nnatural lists"], xlim=(-0.6, 1.6), ylim=(0, 108),
             ylabel="Top-10 slots from retrieval ranks 201–500 (%)")
    ax_s.grid(axis="x", visible=False)
    ax_s.set_title("(d) The shortcut: all-list training fills the top 10\nfrom the retriever's weakest candidates")

    handles = [plt.Rectangle((0, 0), 1, 1, color=FORCED), plt.Rectangle((0, 0), 1, 1, color=NATURAL),
               plt.Rectangle((0, 0), 1, 1, facecolor=GREY, alpha=0.45, hatch="//", edgecolor=GREY, linewidth=0)]
    fig.legend(handles, ["trained on all lists (missed purchases added)", "trained on natural lists only",
                         "retrieval score masked"], loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=3,
               fontsize=8.5)
    fig.suptitle("Controlled policy study: same MLP, same starting weights, only the training lists and score input change",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2, w_pad=2.5)
    _save(fig, out, "fig7_policy_study", pd.DataFrame(rows),
          "Val only, exploratory. MLP ranker, seeds 0–2; within a seed all four arms share initial parameters. "
          "(a, b) per-user macro nDCG@10, bars/lines = seed mean, dots = seeds. (c) paired user bootstrap (1,000 "
          "resamples). (d) 1,000 uniform Val queries ranked over 500 retrieved candidates; arms B/D not measured "
          "(score masked by construction). Source: results/targeted_study/policy-report-20261029.")


FIGURES = {"k1": fig_k1, "rank_response": fig_rank_response, "rankers": fig_rankers, "new_products": fig_new_products,
           "looser_retrieval": fig_looser_retrieval, "retriever_popularity": fig_retriever_popularity,
           "policy": fig_policy_study}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/figures/20261029-findings")
    ap.add_argument("--figures", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    _style()
    for name in args.figures:
        FIGURES[name](out)
    (out / "sources.json").write_text(json.dumps({"label": "validation only, exploratory", "git_hash": git_hash(),
                                                  "sources": SOURCES}, indent=2))
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
