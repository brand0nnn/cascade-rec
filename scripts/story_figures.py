"""Story figures for the 5-7 Nov results (reads saved results only; no training or scoring).

    PYTHONPATH=src python scripts/story_figures.py

Writes PNGs and an index to ``results/figures/20261107-story/``.  Colour roles (validated with the dataviz
palette checker): new products = blue, established products = orange, Val = muted gray, Test = blue.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figures/20261107-story'
TEST = ROOT/'results/test_final/20261107'
VAL_TIMING = ROOT/'results/analysis/20261107-statistics-timing'

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE = '#2a78d6', '#eb6834'
RAMP = ('#86b6ef', '#3987e5', '#1c5cab')
NEW, WARM, VAL, TST = BLUE, ORANGE, MUTED, BLUE

plt.rcParams.update({
    'font.family': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 10, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'savefig.dpi': 200, 'savefig.facecolor': SURFACE})


def titled(fig, title, subtitle):
    fig.text(0.02, 0.97, title, fontsize=13, fontweight='bold', color=INK, va='top')
    fig.text(0.02, 0.905, subtitle, fontsize=9.5, color=INK2, va='top')


def save(fig, name):
    fig.savefig(OUT/name, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def hgrid(ax):
    ax.grid(axis='x'); ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(axis='y'); ax.set_axisbelow(True)


def tm(system, group='all', metric='ndcg10'):
    m = pd.read_csv(TEST/'metrics.csv')
    return float(m.loc[(m.system == system) & (m.group == group) & (m.metric == metric), 'value'].iloc[0])


def tc(label, group='all', ref=None, cand=None):
    c = pd.read_csv(TEST/'contrasts.csv')
    c = c[(c.contrast == label) & (c.group == group) & (c.metric == 'ndcg10')]
    if ref: c = c[(c.reference == ref) & (c.candidate == cand)]
    return c.iloc[0]


def test_diff(ref, cand):
    """cand − ref on Test, computed like the registered contrasts: seed-mean events, 1,000 paired-user draws, seed 0."""
    from cascade.analysis.freshness_val import _metric_values
    from cascade.eval.bootstrap import paired_bootstrap
    bulk = ROOT/'results/local_experiments/20261107-test-final/ranks'
    mean = lambda s: np.mean([_metric_values(np.load(bulk/f'{s}/seed{k}.npz')['rank'])['ndcg10'] for k in (0, 1, 2)], axis=0)
    users = np.load(bulk/'S1/seed0.npz')['user']
    return paired_bootstrap(users, mean(ref), mean(cand), n_resamples=1000, seed=0)


# ---------------------------------------------------------------------------------------------- figures
def fig_funnel():
    bulk = ROOT/'results/local_experiments/20261107-test-final'
    retrieval = np.load(bulk/'candidates/P1/test/retrieval_rank.npy')
    top10 = np.mean([(np.load(bulk/f'ranks/S1/seed{s}.npz')['rank'] <= 10).sum() for s in (0, 1, 2)])
    n = len(retrieval); found = int(np.isfinite(retrieval).sum())
    stages = [('All products users bought next', n), ('…that were among the 200 candidates', found), ('…that the ranker put in the top 10', top10)]
    fig, ax = plt.subplots(figsize=(8, 3.2)); fig.subplots_adjust(top=0.72)
    for i, ((label, value), colour) in enumerate(zip(stages, RAMP)):
        ax.barh(i, value/n, height=0.5, color=colour)
        ax.text(value/n + 0.01, i, f'{value/n:.1%}  ({value:,.0f})', va='center', color=INK, fontsize=10)
    ax.set_yticks(range(3), [s[0] for s in stages]); ax.invert_yaxis()
    ax.set_xlim(0, 1.25); ax.set_xticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%']); hgrid(ax)
    titled(fig, 'Most recommendations fail before the ranker sees the right product',
           'Test period, current pipeline (B1). About 94% of targets never reach the 200 candidates; on Val, 87.6% of targets\nare missed by both retrievers we tried.')
    save(fig, '01_where_failures_happen.png')


def fig_effects():
    team = pd.read_csv(ROOT/'results/steps_final/20261104/test_paired_differences.csv')
    team = team[(team.split == 'test') & (team.metric == 'ndcg10') & (team.group == 'all')]
    level = pd.read_csv(ROOT/'results/steps_final/20261104/test_comparison.csv')
    base = lambda m: float(level.loc[(level.split == 'test') & (level.method == m) & (level.group == 'all'), 'ndcg10'].iloc[0])
    def pick(ref, cand):
        hit = team[(team.reference == ref) & (team.candidate == cand)]
        r = hit.iloc[0] if len(hit) else None
        if r is None:
            r = team[(team.reference == cand) & (team.candidate == ref)].iloc[0]
            return {'diff': -r['diff'], 'ci_low': -r.ci_high, 'ci_high': -r.ci_low, 'base': base(ref)}
        return {'diff': r['diff'], 'ci_low': r.ci_low, 'ci_high': r.ci_high, 'base': base(ref)}
    ours = lambda r, ref: {'diff': r['diff'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'], 'base': tm(ref)}
    rows = [('Pairwise instead of BCE loss', pick('bce_text', 'pairwise_text'), MUTED),
            ('Pairwise instead of BCE loss\n(ranker with co-review inputs)', pick('bce_coreview', 'pairwise_coreview'), MUTED),
            ('Add co-review inputs\n(stale statistics)', pick('bce_text', 'bce_coreview'), MUTED),
            ('Statistics recomputed at the\nperiod start', ours(tc('C1'), 'S1'), BLUE),
            ('Mixed candidate lists\n(two retrievers)', ours(tc('C2'), 'S1'), BLUE),
            ('Add co-review inputs\n(consistent monthly statistics)', ours(tc('C3'), 'S4'), BLUE),
            ('All fixes combined', ours(tc('C4'), 'S1'), BLUE)]
    fig, ax = plt.subplots(figsize=(8.8, 5.0)); fig.subplots_adjust(top=0.8)
    for i, (label, r, colour) in enumerate(rows):
        pct, lo, hi = (100*r[k]/r['base'] for k in ('diff', 'ci_low', 'ci_high'))
        ax.barh(i, pct, height=0.6, color=colour)
        ax.errorbar(pct, i, xerr=[[pct - lo], [hi - pct]], fmt='none', ecolor=INK, elinewidth=1.1, capsize=3)
        ax.text(max(hi, 0) + 1.5, i, f'{pct:+.0f}%', va='center', fontsize=10, color=INK, fontweight='bold' if i == len(rows) - 1 else None)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(-12, 75); ax.set_xlabel('Change in Test nDCG@10 (%)'); hgrid(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED, label='Changing the ranker'), Patch(color=BLUE, label='Changing what the ranker is given')],
              loc='upper right', fontsize=9)
    titled(fig, 'Changing what the ranker is given matters far more than changing the ranker',
           'Change in Test nDCG@10 from one change, relative to the system without it. Error bars: 95% intervals over users.\n'
           'Co-review inputs: how often people who reviewed the user\'s past products also reviewed the candidate.')
    save(fig, '02_what_moves_the_needle.png')


def fig_pools():
    m = pd.read_csv(TEST/'metrics.csv')
    test = {p: float(m.loc[(m.system == f'{p}_retrieval') & (m.group == 'all') & (m.metric == 'recall200'), 'value'].iloc[0]) for p in ('P1', 'P2', 'P3')}
    val = {'P1': 0.0547, 'P2': 0.0687, 'P3': 0.0795}  # results/analysis/20261106-pools/tables.md
    labels = {'P1': 'Recency retriever (P1)', 'P2': 'Two-tower retriever (P2)', 'P3': 'Mixed: alternate both lists (P3)'}
    fig, ax = plt.subplots(figsize=(8, 3.6)); fig.subplots_adjust(top=0.74)
    y = np.arange(3)
    for offset, (name, data, colour) in zip((-0.15, 0.15), (('Val', val, VAL), ('Test', test, TST))):
        vals = [data[p] for p in labels]
        ax.barh(y + offset, vals, height=0.28, color=colour, label=name)
        for yi, v in zip(y + offset, vals): ax.text(v + 0.001, yi, f'{v:.1%}', va='center', fontsize=9, color=INK)
    ax.set_yticks(y, list(labels.values())); ax.invert_yaxis(); ax.set_xlim(0, 0.095)
    ax.set_xticks([0, .02, .04, .06, .08], ['0%', '2%', '4%', '6%', '8%']); ax.set_xlabel('Share of targets among the 200 candidates (Recall@200)')
    ax.legend(loc='upper right'); hgrid(ax)
    titled(fig, 'The two retrievers find different products, so mixing their lists finds more',
           'Recall@200 of each retriever\'s 200 candidates. On Val the two lists share only about 12 of 200: the recency\nretriever finds new products, the two-tower retriever finds established ones.')
    save(fig, '03_two_retrievers_find_different_products.png')


def fig_waterfall():
    bars = [('Current pipeline\n(P1 lists, B1)', 0.008141, MUTED, None),
            ('Same ranker,\nmixed lists\n(not retrained)', 0.008811, RAMP[0], '23% of the gain'),
            ('Retrained on\nmixed lists,\nsame amount of data', 0.010387, RAMP[1], '+55%'),
            ('Retrained on\nmixed lists,\nall the data (D1)', 0.011019, RAMP[2], '+22%')]
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); fig.subplots_adjust(top=0.76)
    for i, (label, value, colour, note) in enumerate(bars):
        ax.bar(i, value, width=0.55, color=colour)
        ax.text(i, value + 0.0002, f'{value:.4f}', ha='center', fontsize=10, color=INK)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
    ax.set_ylim(0, 0.0125); ax.set_ylabel('Val nDCG@10'); vgrid(ax)
    titled(fig, 'The gain needs the ranker to learn on the lists it will rank',
           'Val nDCG@10. Of the +0.0029 gain from mixed lists, 23% comes from the better candidates alone, 55% from retraining\n'
           'the ranker on mixed lists (with the same amount of training data as before) and 22% from the extra training examples.')
    save(fig, '04_mixed_pool_gain_explained.png')


def fig_cue():
    groups = ['Training\n(B, T0)', 'Val at T0\n(a year old)', 'Val at T1\n(period start)', 'Val monthly\n(very fresh)']
    bought, other = [0.431, 0.713, 0.406, 0.095], [0.203, 0.444, 0.153, 0.014]
    new, warm = [0.008666, 0.011721, 0.006181], [0.007887, 0.008231, 0.009570]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.25, 1]}); fig.subplots_adjust(top=0.74, wspace=0.3)
    x = np.arange(4)
    a.bar(x - 0.17, bought, width=0.3, color=BLUE, label='Product the user bought')
    a.bar(x + 0.17, other, width=0.3, color=MUTED, label='Other candidates')
    for xi, v in zip(x - 0.17, bought): a.text(xi, v + 0.012, f'{v:.0%}', ha='center', fontsize=8.5, color=INK)
    for xi, v in zip(x + 0.17, other): a.text(xi, v + 0.012, f'{v:.0%}', ha='center', fontsize=8.5, color=INK)
    a.axvspan(-0.5, 0.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    a.set_xticks(x, groups, fontsize=8.8); a.set_ylim(0, 0.82); a.set_yticks([0, .2, .4, .6, .8], ['0%', '20%', '40%', '60%', '80%'])
    a.set_title('Share showing "no reviews yet"', fontsize=10, color=INK, loc='left'); a.legend(loc='upper right', fontsize=8.5); vgrid(a)
    x3 = np.arange(3)
    b.bar(x3 - 0.17, new, width=0.3, color=NEW, label='New products (33% of purchases)'); b.bar(x3 + 0.17, warm, width=0.3, color=WARM, label='Established products (67%)')
    b.set_xticks(x3, ['T0', 'T1', 'Monthly']); b.set_ylabel('Val nDCG@10 (B1, same frozen model)')
    b.set_title('Recommendation quality', fontsize=10, color=INK, loc='left'); b.legend(loc='upper left', fontsize=8.5); b.set_ylim(0, 0.0152); vgrid(b)
    titled(fig, 'The ranker learned "no reviews yet = likely purchase", and it only works with statistics as old as in training',
           'Left: at T1 the cue looks as it did in training; at T0 it fires too often; monthly statistics erase it. Right: the same frozen\n'
           'ranker gains most at T1; monthly statistics help established products but hurt new ones.')
    save(fig, '05_learned_rule_and_statistics_timing.png')


def fig_swap():
    val = pd.read_csv(VAL_TIMING/'metrics.csv'); test = pd.read_csv(TEST/'timing/metrics.csv')
    def share(frame, arm, end):
        g = lambda a: float(frame.loc[(frame.arm == a) & (frame.method == 'bce_text') & (frame.group == 'all'), 'ndcg10'].iloc[0])
        return (g(arm) - g('T0'))/(g(end) - g('T0'))
    arms = [('X1_known', 'Only the "has any reviews" flag'), ('X2_known_degree', 'Flag + review count'),
            ('X7_products_released_in_B', 'All statistics, only for products\nreleased after T0'), ('X6_old_products', 'All statistics, only for products\nreleased before T0')]
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); fig.subplots_adjust(top=0.76)
    y = np.arange(len(arms))
    for offset, (name, frame, end, colour) in zip((-0.15, 0.15), (('Val (T0 → T1)', val, 'T1', VAL), ('Test (T0 → T2)', test, 'T2', TST))):
        vals = [share(frame, a, end) for a, _ in arms]
        ax.barh(y + offset, vals, height=0.28, color=colour, label=name)
        for yi, v in zip(y + offset, vals): ax.text(max(v, 0) + 0.02, yi, f'{v:.0%}', va='center', fontsize=9, color=INK)
    ax.axvline(1, color=AXIS, lw=0.8); ax.text(1.005, -0.62, 'full gain', fontsize=8.5, color=INK2)
    ax.set_yticks(y, [a[1] for a in arms]); ax.invert_yaxis(); ax.set_xlim(0, 1.35)
    ax.set_xticks([0, .25, .5, .75, 1, 1.25], ['0%', '25%', '50%', '75%', '100%', '125%']); ax.set_xlabel('Share of the gain from fresh statistics recovered (B1, all products)')
    ax.legend(loc='lower right'); hgrid(ax)
    titled(fig, 'The stale-statistics problem is products released after T0 being misread as new',
           'Start from T0 statistics and update one piece at a time. Updating only products released after T0 recovers the whole gain;\n'
           'updating older products recovers little. Same frozen model, rescored only.')
    save(fig, '06_where_the_stale_statistics_gain_comes_from.png')


def fig_lag():
    frame = pd.read_csv(TEST/'timing/metrics.csv')
    lags = [0, 1, 3, 6, 9, 12]
    fig, ax = plt.subplots(figsize=(8.4, 4.6)); fig.subplots_adjust(top=0.76)
    ax.axvline(6.1, color=AXIS, lw=1.2)
    for group, colour, label in (('new', NEW, 'New products (44% of purchases)'), ('warm', WARM, 'Established products (56%)')):
        v = [float(frame.loc[(frame.arm == f'lag{l}') & (frame.method == 'bce_text') & (frame.group == group), 'ndcg10'].iloc[0]) for l in lags]
        ax.plot(lags, v, color=colour, lw=2.2, marker='o', ms=7, mec=SURFACE, mew=2, label=label)
        best = int(np.argmax(v)); ax.annotate('best', (lags[best], v[best]), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8.5, color=INK2)
    ax.text(5.85, 0.0085, 'The ranker was trained with\nstatistics about 6 months old →', fontsize=8.5, color=INK2, va='center', ha='right')
    ax.set_xticks(lags, ['0\n(up to date)', '1', '3', '6', '9', '12']); ax.set_xlim(-0.6, 12.8)
    ax.set_xlabel('How old the product statistics are when the ranker uses them (months)')
    ax.set_ylabel('Test nDCG@10 (same ranker)'); ax.legend(loc='center right', fontsize=9); vgrid(ax)
    titled(fig, 'For new products, statistics work best when they are as old as in training',
           'The same trained ranker, given product statistics (e.g. review counts) of different ages. New products are recommended best\n'
           'with ~6-month-old statistics, the age it learned from; established products are best with up-to-date ones. Val shows the same.')
    save(fig, '07_statistics_age_curve.png')


def fig_coreview():
    rows = [('Stale: trained T0,\nscored T0 (Val)', 0.000157, 0.000065, 0.000255, MUTED),
            ('Trained T0,\nscored T1 (Val)', 0.000492, 0.000375, 0.000609, MUTED),
            ('Monthly in training\nand scoring (Val)', 0.001308, 0.001129, 0.001483, BLUE)]
    c3 = tc('C3'); rows.append(('Monthly in training\nand scoring (Test)', c3['diff'], c3.ci_low, c3.ci_high, BLUE))
    fig, ax = plt.subplots(figsize=(8, 4.0)); fig.subplots_adjust(top=0.76)
    for i, (label, d, lo, hi, colour) in enumerate(rows):
        ax.bar(i, d, width=0.45, color=colour); ax.plot([i, i], [lo, hi], color=INK, lw=1.2)
        ax.text(i, hi + 0.00006, f'+{d:.4f}', ha='center', fontsize=9.5, color=INK)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9); ax.set_ylabel('Co-review − text, nDCG@10'); ax.set_ylim(0, 0.0021)
    ax.set_yticks([0, .0005, .001, .0015, .002], ['0', '0.0005', '0.0010', '0.0015', '0.0020']); vgrid(ax)
    titled(fig, 'Co-review statistics only help when their timing is consistent',
           'Gain from adding co-review ("bought together") statistics to the BCE ranker on P1 lists, with 95% intervals.\n'
           'The team concluded co-review barely helps because it was measured with stale statistics.')
    save(fig, '08_coreview_needs_consistent_timing.png')


def fig_confirm():
    names = {'C1': 'Statistics at the\nperiod start', 'C2': 'Mixed candidate\nlists', 'C3': 'Co-review inputs,\nmonthly statistics',
             'C4': 'Combined system\nvs current pipeline'}
    val = {'C1': 0.009432 - 0.008141, 'C2': 0.002878, 'C3': 0.010613 - 0.009304, 'C4': 0.013084 - 0.008141}
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); fig.subplots_adjust(top=0.76)
    x = np.arange(4)
    for i, c in enumerate(names):
        d = tc(c)
        ax.bar(i - 0.18, val[c], width=0.34, color=VAL, label='Val' if i == 0 else None)
        ax.bar(i + 0.18, d['diff'], width=0.34, color=TST, label='Test (with 95% interval)' if i == 0 else None)
        ax.plot([i + 0.18, i + 0.18], [d.ci_low, d.ci_high], color=INK, lw=1.2)
        ax.text(i - 0.18, val[c] + 0.00008, f'+{val[c]:.4f}', ha='center', fontsize=8.5, color=INK)
        ax.text(i + 0.18, d.ci_high + 0.00008, f'+{d["diff"]:.4f}', ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(x, list(names.values()), fontsize=9); ax.set_ylabel('Change in nDCG@10')
    ax.set_ylim(0, 0.0058); ax.legend(loc='upper left', fontsize=8.5); vgrid(ax)
    titled(fig, 'The locked Test run confirms every main effect',
           'The four comparisons written down before Test was opened. The mixed-list gain is smaller on Test, as predicted,\n'
           'because 44% of Test purchases are new products (33% on Val).')
    save(fig, '09_test_confirms_val.png')


def fig_winners():
    systems = [('S1', 'Current pipeline\n(B1, T0)'), ('S3', 'Same model,\nstatistics at T2'), ('S5', 'P1 + monthly\nco-review (B10)'),
               ('S6', 'Mixed pool\n(D1, T0)'), ('S8', 'Mixed pool + monthly\nco-review (D7)')]
    fig, ax = plt.subplots(figsize=(9, 4.2)); fig.subplots_adjust(top=0.76)
    x = np.arange(len(systems))
    for offset, group, colour, label in ((-0.17, 'new', NEW, 'New products'), (0.17, 'warm', WARM, 'Established products')):
        v = [tm(s, group) for s, _ in systems]
        ax.bar(x + offset, v, width=0.3, color=colour, label=label)
    for i, (s, _) in enumerate(systems): ax.text(i, max(tm(s, 'new'), tm(s, 'warm')) + 0.0004, f'all {tm(s):.4f}', ha='center', fontsize=8.5, color=INK2)
    ax.set_xticks(x, [s[1] for s in systems], fontsize=8.8); ax.set_ylabel('Test nDCG@10'); ax.set_ylim(0, 0.0152)
    ax.legend(loc='upper left', fontsize=8.5); vgrid(ax)
    titled(fig, 'The best system overall is not the best for new products',
           'Test nDCG@10 by product type (44% of Test targets are new). The mixed pool helps established products; the P1 lists\n'
           'with monthly co-review statistics are best for new products.')
    save(fig, '10_who_wins_on_test.png')


INDEX = """# Story figures (7 Nov 2026)

Made by `scripts/story_figures.py` from saved results only (no training or scoring). Colours: new products blue,
established products orange; Val gray, Test blue. All numbers are nDCG@10, per-user macro, seed means.

| # | File | What it shows | Source |
|---|---|---|---|
| 1 | `01_where_failures_happen.png` | Most targets never reach the 200 candidates, so the ranker cannot fix them | Test ranks (S1); Val case study |
| 2 | `02_what_moves_the_needle.png` | Ranker loss choices move results by at most 0.0005; statistics inputs (with consistent timing), statistics timing and retrieval by +0.0014 to +0.0039 | `results/test_final/20261107`, team `results/steps_final/20261104` |
| 3 | `03_two_retrievers_find_different_products.png` | The two retrievers overlap little, so mixing them raises Recall@200 | `results/analysis/20261106-pools`, Test metrics |
| 4 | `04_mixed_pool_gain_explained.png` | Bar chart: current pipeline, same ranker on mixed lists, retrained with the same amount of data, retrained with all data | `results/analysis/20261107-pool-size-control` |
| 5 | `05_learned_rule_and_statistics_timing.png` | The "no reviews yet" rule, and why T1 helps while monthly hurts new products | `results/analysis/20261106-freshness-mechanism`, `-freshness-val` |
| 6 | `06_where_the_stale_statistics_gain_comes_from.png` | Column swap: the gain comes from products released after T0 being misread as new (Val and Test) | `results/analysis/20261107-statistics-timing`, `results/test_final/20261107/timing` |
| 7 | `07_statistics_age_curve.png` | Same ranker given statistics of different ages (Test): new products best at the training age (~6 months), established products best when up to date | `results/test_final/20261107/timing` |
| 8 | `08_coreview_needs_consistent_timing.png` | Co-review only helps with consistent monthly timing | `-refit-draws`, `-rolling-ranking`, Test C3 |
| 9 | `09_test_confirms_val.png` | Grouped bars: Val vs Test for the four pre-registered comparisons | `results/test_final/20261107` |
| 10 | `10_who_wins_on_test.png` | Best overall (D7) vs best for new products (B10) | `results/test_final/20261107` |

Caveats: Val was reused to choose directions (Test confirms the main effects); product titles leak future device
names; the 5-core filter flatters "no reviews yet"; monthly statistics depart from the period-start rule (no future data).
"""


ORANGE_SCHEME = {'BLUE': '#d9730d', 'NEW': '#d9730d', 'WARM': '#5f5e5a', 'TST': '#d9730d',
                 'RAMP': ('#eba154', '#d9730d', '#a35400')}  # Amazon-style poster scheme, validated with the dataviz checker


def orange_variants(out, dpi=300, svg=True):
    """Figures 02, 04, 05 and 07 in the orange poster scheme: orange highlight, dark gray for the comparison series, no blue."""
    global OUT
    saved = {k: globals()[k] for k in ORANGE_SCHEME}
    globals().update(ORANGE_SCHEME); OUT = Path(out); OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams['savefig.dpi'] = dpi
    def save_both(fig, name):
        fig.savefig(OUT/name, bbox_inches='tight', pad_inches=0.25)
        if svg: fig.savefig(OUT/name.replace('.png', '.svg'), bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)
    original_save = globals()['save']; globals()['save'] = save_both
    try:
        for f in (fig_effects, fig_waterfall, fig_cue, fig_lag): f()
    finally:
        globals().update(saved); globals()['save'] = original_save


def main():
    import sys
    if '--orange' in sys.argv:
        orange_variants(ROOT/'results/figures/20261107-story-orange')
        orange_variants(ROOT/'results/poster_pack_20261107/figures')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_funnel, fig_effects, fig_pools, fig_waterfall, fig_cue, fig_swap, fig_lag, fig_coreview, fig_confirm, fig_winners):
        f(); print(json.dumps({'figure': f.__name__}), flush=True)
    (OUT/'README.md').write_text(INDEX)


if __name__ == '__main__':
    main()
