"""Presentation of saved-results measurements; no model or dataset access."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def table(frame: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    def fmt(value):
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return '—'
        if isinstance(value, (float, np.floating)):
            return f'{value:.6f}'
        return str(value).replace('|', '/')
    return ['| ' + ' | '.join(columns.values()) + ' |',
            '| ' + ' | '.join('---' for _ in columns) + ' |',
            *['| ' + ' | '.join(fmt(row.get(k)) for k in columns) + ' |' for _, row in frame.iterrows()]]


def write_report(root: Path, aggregate: pd.DataFrame, frames: dict, contrast: dict,
                 completion: list, sample: bool, provenance: dict) -> None:
    headline = aggregate[aggregate.axis == 'overall']
    intervals = frames['uncertainty']
    lines = ['# Evaluation of saved full-data experiments', '',
             '**Development subset only — these are analysis checks, not research results.**' if sample else
             '**Full validation evaluation from existing predictions. No models executed; no Test predictions loaded.**', '',
             f"Population: {provenance['n_instances']:,} events / {provenance['n_users']:,} users. "
             'Every end-to-end metric averages events within user, then users. Retrieval misses count as zero.', '',
             'Protocol: docs/evaluations.md. Sources, training seeds, candidate provenance and analysis hashes are in provenance.json; '
             'per_seed.csv retains each run. SD is sample SD across training seeds; deterministic/single-seed rows have no estimated seed SD. '
             'User intervals condition on the trained runs and do not include retriever retraining variation.', '',
             '## 1. Which systems perform better?', '']
    for period in ('retriever_train', 'all_train'):
        lines += [f'### Retrieval / baselines trained on {period}', '',
                  'All-training baselines are context; they are not the matched-training RQ1 comparison.' if period == 'all_train' else
                  'RQ1: all rows train on the same retriever-training period.', '']
        rows = headline[(headline.family == 'retrieval') & (headline.train_on == period)]
        lines += table(rows, {'label': 'System', 'ndcg10_macro_mean': 'nDCG@10', 'ndcg10_macro_std': 'Seed SD',
                            'recall10_macro_mean': 'Recall@10', 'recall200_macro_mean': 'Recall@200'}) + ['']
    lines += ['### Ranking on the same K1=200 candidates', '',
              'MLP score/statistics are single-seed discovery; matching/attributes/DIN have three seeds. '
              'Historical-position R1 is retained alongside the pessimistic-rank companion. '
              'Ranker variation is conditional on the frozen text seed-0 retriever.', '']
    lines += table(headline[headline.family.isin(['rq3', 'ranker'])],
                   {'label': 'System', 'ndcg10_macro_mean': 'nDCG@10', 'ndcg10_macro_std': 'Seed SD',
                    'recall10_macro_mean': 'Recall@10', 'conditional_ndcg10_macro_mean': 'Conditional nDCG@10',
                    'n_retrieved_mean': 'Retrieved targets', 'n_retrieved_users_mean': 'Retrieved users'}) + ['']
    lines += ['### Archived natural-list GAUC', '',
              'These are the original full-validation K200 retrieved-target estimates, including when this report is a development check. '
              'Per-event RQ3 AUCs were not archived, so no subgroup/development GAUC is inferred. '
              'GAUC is secondary to the headline @10 metrics.', '']
    rows = frames['reused_gauc'].groupby('system', sort=False).agg(
        gauc_macro=('gauc_macro', 'mean'), seed_sd=('gauc_macro', 'std'),
        n_instances=('n_instances', 'first'), n_users=('n_users', 'first')).reset_index()
    lines += table(rows, {'system': 'System', 'gauc_macro': 'GAUC', 'seed_sd': 'Seed SD',
                         'n_instances': 'Retrieved events', 'n_users': 'Retrieved users'}) + ['']
    lines += ['### Paired differences', '',
              f"{provenance['bootstrap_resamples']} user resamples, seed {provenance['bootstrap_seed']}, 95% percentile intervals. "
              'Metrics are averaged across training seeds per event before paired user averaging. '
              'An interval containing zero is inconclusive, not evidence of equivalence.', '']
    rows = intervals[(intervals.axis == 'overall') | ((intervals.comparison == 'text_id_minus_text') & (intervals.axis == 'new_in_period'))]
    lines += table(rows, {'comparison': 'Comparison', 'group': 'Population', 'metric': 'Metric', 'diff': 'Difference',
                         'ci_low': 'CI low', 'ci_high': 'CI high', 'n_users': 'Users'}) + ['']
    lines += ['## 2. Where is the target lost?', '',
              'The five outcomes partition every event. Shares and nDCG contributions retain the full per-user denominator. '
              'Contributions sum to final-minus-Stage-1 nDCG; rescued share minus demoted share equals the Recall@10 change. '
              'Conditional rescue/demotion rates use their respective at-risk events/users, recorded in per_seed.csv.', '']
    funnel = frames['funnel']
    rows = funnel[(funnel.axis == 'overall') & funnel.system.isin(['matching', 'din'])]
    rows = rows.groupby(['system', 'outcome'], sort=False)[['count', 'macro_share', 'ndcg_contribution']].mean().reset_index()
    lines += ['Counts below are means of seed-specific raw counts, not one combined/ensemble prediction.', '']
    lines += table(rows, {'system': 'Ranker', 'outcome': 'Outcome', 'count': 'Mean count',
                         'macro_share': 'Macro share', 'ndcg_contribution': 'nDCG contribution'}) + ['']
    lines += ['### Retrieval-rank reconciliation', '',
              'The archive stores actual returned positions; full-catalogue ranks use pessimistic ties. '
              'For a naturally retrieved target the pessimistic within-prefix rank is min(full-catalogue rank, K1). '
              'Natural membership, recovered consistently from the saved ranker ranks, defines coverage. '
              'A target may be returned at a tied boundary even when its full-catalogue rank exceeds K1. '
              'Per-K discrepancies are archived in boundary_k*.npz; historical artifacts remain unchanged.', '']
    lines += table(frames['rank_reconciliation'], {'k1': 'K1', 'membership_vs_full_rank_count': 'Membership / full-rank discrepancies',
              'retrieved_position_vs_full_rank_count': 'Position / full-rank differences (K200)',
              'ndcg10_changed_instances': 'Changed Stage-1 nDCG events', 'recall10_changed_instances': 'Changed Stage-1 hits'}) + ['']
    lines += ['## 3. For whom does performance change?', '',
              'Each slice averages its own qualifying events within user, then its qualifying users. Slice means do not add to the overall mean. '
              'groups.csv contains all systems, support, conditional denominators and at-risk rates; funnel.csv contains every group/seed outcome. '
              'Canonical pre-T2 cold/tail/head aggregates are separately reused in reused_segments.csv because their per-event masks were not shared. '
              'They overlap with new-in-period labels and are not model inputs. Reused aggregates retain their original scorable population.', '']
    for axis in ('item_type', 'history_length', 'history_diversity', 'validation_window'):
        lines += [f'### {axis}', '']
        rows = aggregate[(aggregate.axis == axis) & aggregate.system.isin(['R1', 'matching', 'din'])]
        lines += table(rows, {'system': 'System', 'group': 'Group', 'n_instances_mean': 'Events', 'n_users_mean': 'Users',
              'n_retrieved_mean': 'Retrieved targets', 'n_retrieved_users_mean': 'Retrieved users',
              'ndcg10_macro_mean': 'nDCG@10', 'recall10_macro_mean': 'Recall@10'}) + ['']
    rows = intervals[(intervals.comparison == 'din_minus_matching') & intervals.axis.isin(['item_type', 'history_diversity'])]
    lines += ['### DIN-minus-MLP group differences', '']
    lines += table(rows, {'group': 'Group', 'diff': 'nDCG difference', 'ci_low': 'CI low', 'ci_high': 'CI high', 'n_users': 'Users'}) + ['']
    lines += ['### Direct mixed-minus-homogeneous effect contrast', '',
              'This compares the actual DIN-minus-MLP effects, rather than significance in two separate groups. '
              'Shared user draws preserve correlation when a user contributes to both groups. '
              'Sparse homogeneous-history support limits this comparison.', '',
              f"Difference: {contrast['diff']:.6f}; 95% CI [{contrast['ci_low']:.6f}, {contrast['ci_high']:.6f}]. "
              f"Mixed: {contrast['mixed_instances']:,} events / {contrast['mixed_users']:,} users; "
              f"homogeneous: {contrast['homogeneous_instances']:,} events / {contrast['homogeneous_users']:,} users. "
              f"Valid bootstrap draws: {contrast['valid_resamples']}/{contrast['n_resamples']}.", '']
    lines += ['## 4. Why does increasing K1 change the outcome?', '',
              'Weights and training preprocessing remain frozen. Larger pools improve target availability but introduce competitors. '
              'K500 changes the evaluation candidate distribution beyond the training pool. Conditional metrics at different K use different target populations.', '']
    lines += table(headline[headline.family == 'sweep'], {'k1': 'K1', 'retrieval_coverage_macro_mean': 'Membership Recall@K1',
                  'ndcg10_macro_mean': 'nDCG@10', 'ndcg10_macro_std': 'Seed SD', 'recall10_macro_mean': 'Recall@10',
                  'conditional_ndcg10_macro_mean': 'Conditional nDCG', 'gauc_macro_mean': 'Natural-list GAUC'}) + ['']
    rows = frames['expansion'].groupby(['k_low', 'k_high'], sort=False).mean(numeric_only=True).reset_index()
    lines += table(rows, {'k_low': 'From K', 'k_high': 'To K', 'newly_retrieved': 'New targets',
                         'new_target_contribution': 'New-target nDCG gain', 'existing_target_contribution': 'Existing-target change',
                         'total_delta_ndcg10': 'Overall change'}) + ['']
    lines += ['K100 is the observed validation quality maximum in the historical sweep. '
              '**No final K1 or cost-optimal setting is selected.** No acceptable-loss margin was recorded before inspecting the sweep. '
              'The historical CI-containment heuristic is not a non-inferiority/equivalence result.', '',
              '## 5. Supported explanations and remaining work', '',
              'The saved score audit reports 87.69% force-added training positives. '
              'Raw-score target-versus-alternative AUC is about 0.080 over constructed training lists and 0.629 over naturally retrieved validation targets. '
              'The existing one-input probe shows sensitivity to retrieval score. These observations support a mismatch hypothesis; '
              'they do not establish that the training policy caused the learned response. '
              'See saved_diagnostic_report.md for the original numeric tables and three already selected product examples. '
              'These examples are reused illustrations, not newly selected cases or independent evidence.', '',
              'Diagnostics keep their original populations: 315,114 training lists; 1,000 uniform validation events; '
              'a 5,814-event rescored union including selected demotions. Overtaker counts are seed-event-candidate occurrences, '
              'not headline user-macro estimates. Unobserved alternatives are not known dislikes.', '',
              'saved_latency.json retains the original Apple M1 Max, single-thread ranker-stage measurements '
              '(1,000 common queries after 50 warm-ups). They exclude retrieval, disk I/O, checkpoint loading and one-time tables; '
              'they are not timings measured by this evaluator. This remains true for a development-subset analysis.', '']
    lines += table(pd.DataFrame(completion, columns=['work', 'status', 'gap']), {'work': 'Work', 'status': 'Status', 'gap': 'Evidence / remaining gap'}) + ['']
    lines += ['### Next phase, after reviewing this evaluation', '',
              '1. Obtain the original processed data, embeddings and frozen-retriever candidates. Regenerate only if needed, with provenance and reproduction checks.',
              '2. Complete candidate-level profiles/parser audits and any new representative examples; retain unknowns and freeze their definitions on Val.',
              '3. Obtain the team’s final ranker/index/K1 choice, acceptable-loss/cost rule and matched validation evidence. Do not manufacture a post-hoc margin.',
              '4. Freeze artifact identities and analysis protocol, then perform the final F0/F1/F2 Test comparison and full-cascade timing. Existing retrieval Test exposure remains disclosed.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    plot(root, headline, funnel, frames['expansion'], sample)


def plot(root: Path, headline: pd.DataFrame, funnel: pd.DataFrame, expansion: pd.DataFrame, sample: bool) -> None:
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.mpl-cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(root / '.cache'))
    (Path(os.environ['XDG_CACHE_HOME']) / 'fontconfig').mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = 'Development subset' if sample else 'Full validation'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    rows = headline.set_index('system').loc[['R0', 'R1', 'matching', 'din']]
    axes[0].bar(['Popularity', 'Retrieval', 'MLP', 'DIN'], rows.ndcg10_macro_mean,
                yerr=rows.ndcg10_macro_std.fillna(0), capsize=3, color=['#9ca3af', '#6b7280', '#2563eb', '#d97706'])
    axes[0].set(title=f'{title}: same K1=200 candidates', ylabel='User-macro nDCG@10')
    axes[0].tick_params(axis='x', labelrotation=20)
    rows = headline[headline.family == 'sweep'].sort_values('k1')
    axes[1].errorbar(rows.k1, rows.ndcg10_macro_mean, yerr=rows.ndcg10_macro_std, marker='o', capsize=3)
    axes[1].set(title='Frozen MLP: quality / candidate count', xlabel='K1 (no final selection)', ylabel='User-macro nDCG@10')
    rows = funnel[(funnel.axis == 'overall') & funnel.system.isin(['matching', 'din'])]
    rows = rows.groupby(['system', 'outcome']).ndcg_contribution.mean().unstack().reindex(['matching', 'din'])
    xs = np.arange(2)
    for i, outcome in enumerate(['rescued', 'demoted', 'retained']):
        axes[2].bar(xs + (i - 1)*.24, rows[outcome], .24, label=outcome)
    axes[2].axhline(0, color='black', linewidth=.7)
    axes[2].set(xticks=xs, xticklabels=['MLP', 'DIN'], title='Additive changes versus retrieval', ylabel='Macro nDCG contribution')
    axes[2].legend(fontsize=8)
    fig.savefig(root / 'overview.png', dpi=160)
    fig.savefig(root / 'overview.svg')
    plt.close(fig)
