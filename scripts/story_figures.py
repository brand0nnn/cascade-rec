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
    def pick(ref, cand):
        hit = team[(team.reference == ref) & (team.candidate == cand)]
        if len(hit): return hit.iloc[0]
        r = team[(team.reference == cand) & (team.candidate == ref)].iloc[0]  # stored the other way round
        return {'diff': -r['diff'], 'ci_low': -r.ci_high, 'ci_high': -r.ci_low}
    groups = [('Ranker loss (the only change to the ranker model)', MUTED,
               [('Pairwise instead of BCE (text inputs)', pick('bce_text', 'pairwise_text')),
                ('Pairwise instead of BCE (with co-review inputs)', pick('bce_coreview', 'pairwise_coreview'))]),
              ('Statistics given to the ranker', BLUE,
               [('Add co-review inputs, stale T0 statistics', pick('bce_text', 'bce_coreview')),
                ('Add co-review inputs, monthly statistics', tc('C3'))]),
              ('Timing of the statistics', BLUE,
               [('Statistics at the period start (T2)', tc('C1')),
                ('Monthly statistics in training and scoring', test_diff('S1', 'S4'))]),
              ('Retrieval', BLUE, [('Mixed candidate pool (two retrievers)', tc('C2'))]),
              ('Combined', BLUE, [('Mixed pool + monthly co-review (D7)', tc('C4'))])]
    fig, ax = plt.subplots(figsize=(8.8, 6.4)); fig.subplots_adjust(top=0.79)
    y, ticks, labels = 0.0, [], []
    for header, colour, rows in groups:
        ax.text(-0.00068, y - 0.7, header, fontsize=9, color=INK2, fontweight='bold')
        for label, r in rows:
            c = MUTED if (colour == BLUE and 'stale' in label) else colour
            ax.barh(y, r['diff'], height=0.5, color=c)
            ax.plot([r['ci_low'], r['ci_high']], [y, y], color=INK, lw=1.2, solid_capstyle='butt')
            ax.text(max(r['diff'], r['ci_high'], 0) + 0.00008, y, f'{r["diff"]:+.4f}' if abs(r['diff']) >= 5e-4 else f'{r["diff"]:+.5f}',
                    va='center', fontsize=9.5, color=INK)
            ticks.append(y); labels.append(label); y += 1
        y += 0.8
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ticks, labels); ax.invert_yaxis()
    ax.set_xlim(-0.0007, 0.0047); ax.set_xlabel('Change in Test nDCG@10 from making the change'); hgrid(ax)
    titled(fig, 'Changing the ranker barely matters; what it is given, and when, matters most',
           'Each bar compares the same system with and without one change, on Test (current pipeline 0.0064); lines are 95% paired-user\n'
           'intervals. Co-review inputs: how often people who reviewed the user\'s past products also reviewed the candidate. Gray = no\n'
           'gain. The first three rows use the team\'s trained weights; the rest use the rebuilt B1 as reference.')
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
    steps = [('Current pipeline\n(P1 lists, B1)', 0.008141, 'total'), ('Give it the mixed\nlists (no retraining)', 0.000670, 'step'),
             ('Retrain on mixed lists\n(same amount of data)', 0.001576, 'step'), ('Use all the extra\ntraining examples', 0.000632, 'step'),
             ('Mixed-pool ranker\n(D1)', 0.011019, 'total')]
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); fig.subplots_adjust(top=0.76)
    level = 0.0
    for i, (label, value, kind) in enumerate(steps):
        if kind == 'total':
            ax.bar(i, value, width=0.5, color=MUTED if i == 0 else BLUE); ax.text(i, value + 0.0002, f'{value:.4f}', ha='center', fontsize=9.5, color=INK)
            level = value
        else:
            ax.bar(i, value, bottom=level, width=0.5, color=RAMP[1])
            ax.text(i, level + value + 0.0002, f'+{value:.4f}\n({value/0.002878:.0%})', ha='center', fontsize=9, color=INK)
            level += value
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=8.8); ax.set_ylim(0.006, 0.0124)
    ax.set_ylabel('Val nDCG@10'); vgrid(ax)
    titled(fig, 'The gain needs the ranker to learn on the lists it will rank',
           'Val nDCG@10. Splitting the +35% mixed-pool gain with a size-matched control (D1s, 20,676 positives, as for P1).\n'
           'Note the y-axis starts at 0.006.')
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
    b.bar(x3 - 0.17, new, width=0.3, color=NEW, label='New products'); b.bar(x3 + 0.17, warm, width=0.3, color=WARM, label='Established products')
    b.set_xticks(x3, ['T0', 'T1', 'Monthly']); b.set_ylabel('Val nDCG@10 (B1, same frozen model)')
    b.set_title('Recommendation quality', fontsize=10, color=INK, loc='left'); b.legend(loc='upper left', fontsize=8.5, ncol=2); b.set_ylim(0, 0.0148); vgrid(b)
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False); fig.subplots_adjust(top=0.74, wspace=0.25)
    for ax, (name, path) in zip(axes, (('Val', VAL_TIMING/'metrics.csv'), ('Test', TEST/'timing/metrics.csv'))):
        frame = pd.read_csv(path)
        arms = (['monthly'] if name == 'Val' else ['lag0']) + [f'lag{l}' for l in (1, 3, 6, 9, 12)]
        lags = [0, 1, 3, 6, 9, 12]
        at6 = []
        for group, colour, label in (('new', NEW, 'New products'), ('warm', WARM, 'Established products')):
            v = [float(frame.loc[(frame.arm == a) & (frame.method == 'bce_text') & (frame.group == group), 'ndcg10'].iloc[0]) for a in arms]
            ax.plot(lags, v, color=colour, lw=2, marker='o', ms=6, mec=SURFACE, mew=2, label=label, solid_capstyle='round')
            ax.text(12.3, v[-1], label.split()[0], color=INK2, fontsize=8.5, va='center'); at6.append(v[3])
        ax.axvline(6.1, color=AXIS, lw=1); ax.text(6.3, sum(at6)/2, 'average age of the\nstatistics in training\n(6.1 months)', fontsize=8, color=INK2, va='center')
        ax.set_xticks(lags, [f'{l}' for l in lags]); ax.set_xlabel('Age of statistics: months before the query\'s month')
        ax.set_title(name, fontsize=10, color=INK, loc='left'); ax.set_xlim(-0.5, 14); vgrid(ax)
        if name == 'Val': ax.set_ylabel('nDCG@10 (B1, same frozen model)'); ax.legend(loc='lower right', fontsize=8.5)
    titled(fig, 'New products need statistics as old as in training; established products want fresh ones',
           'Each query gets statistics cut a fixed number of months before its month. New-product quality peaks near the training age;\n'
           'established-product quality is best with the freshest statistics. Consistent monthly training removes the conflict.')
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
    names = {'C1': 'Statistics at the period start\n(same model)', 'C2': 'Mixed candidate pool', 'C3': 'Co-review with monthly statistics',
             'C4': 'Combined system (D7)\nvs current pipeline'}
    val = {'C1': 0.009432 - 0.008141, 'C2': 0.002878, 'C3': 0.010613 - 0.009304, 'C4': 0.013084 - 0.008141}
    fig, ax = plt.subplots(figsize=(8.6, 3.8)); fig.subplots_adjust(top=0.74)
    for i, c in enumerate(names):
        d = tc(c)
        ax.plot([d.ci_low, d.ci_high], [i, i], color=TST, lw=1.4, alpha=0.6)
        ax.plot([val[c], d['diff']], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(val[c], i, s=60, color=VAL, edgecolor=SURFACE, linewidth=2, zorder=3, label='Val' if i == 0 else None)
        ax.scatter(d['diff'], i, s=60, color=TST, edgecolor=SURFACE, linewidth=2, zorder=3, label='Test (line: 95% interval)' if i == 0 else None)
        ax.text(max(val[c], d.ci_high) + 0.00012, i, f'Test {d["diff"]:+.4f}', va='center', fontsize=9, color=INK)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(4), list(names.values())); ax.invert_yaxis(); ax.set_xlim(-0.0002, 0.0062)
    ax.set_xlabel('Change in nDCG@10'); ax.legend(loc='upper right', fontsize=8.5); hgrid(ax)
    titled(fig, 'The locked Test run confirms every main effect',
           'Primary contrasts, written down before Test was opened. The mixed-pool gain is smaller on Test, as predicted,\n'
           'because 44% of Test targets are new products (18% on Val).')
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
| 4 | `04_mixed_pool_gain_explained.png` | Most of the +35% needs the ranker retrained on the mixed lists (size-matched control) | `results/analysis/20261107-pool-size-control` |
| 5 | `05_learned_rule_and_statistics_timing.png` | The "no reviews yet" rule, and why T1 helps while monthly hurts new products | `results/analysis/20261106-freshness-mechanism`, `-freshness-val` |
| 6 | `06_where_the_stale_statistics_gain_comes_from.png` | Column swap: the gain comes from products released after T0 being misread as new (Val and Test) | `results/analysis/20261107-statistics-timing`, `results/test_final/20261107/timing` |
| 7 | `07_statistics_age_curve.png` | New products peak with statistics as old as in training; established products prefer fresh ones (Val and Test) | same |
| 8 | `08_coreview_needs_consistent_timing.png` | Co-review only helps with consistent monthly timing | `-refit-draws`, `-rolling-ranking`, Test C3 |
| 9 | `09_test_confirms_val.png` | The four pre-registered Test contrasts against Val | `results/test_final/20261107` |
| 10 | `10_who_wins_on_test.png` | Best overall (D7) vs best for new products (B10) | `results/test_final/20261107` |

Caveats: Val was reused to choose directions (Test confirms the main effects); product titles leak future device
names; the 5-core filter flatters "no reviews yet"; monthly statistics depart from the period-start rule (no future data).
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_funnel, fig_effects, fig_pools, fig_waterfall, fig_cue, fig_swap, fig_lag, fig_coreview, fig_confirm, fig_winners):
        f(); print(json.dumps({'figure': f.__name__}), flush=True)
    (OUT/'README.md').write_text(INDEX)


if __name__ == '__main__':
    main()
