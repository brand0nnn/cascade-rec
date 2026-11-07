"""Publish compact, audited replication evidence and a checked comparison figure."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd

from cascade.eval.saved_results import digest, write_json
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average
from cascade.pipelines.full_data_ranking import METHODS, arrays, data_for, groups_for, read, study_path
from cascade.utils.config import REPO_ROOT


LABELS = {'retrieval': 'Frozen retrieval', 'bce_text': 'BCE text', 'pairwise_text': 'Pairwise text',
          'pairwise_coreview': 'Pairwise + co-review', 'pairwise_combined': 'Pairwise combined'}


def historical_context(root, destination, meta):
    """Empirical context, explicitly outside the matched primary interventions."""
    data = data_for(meta); inst = data.split_instances('val'); users = inst.user.to_numpy()
    ranks = {'archived_MLP_K100': [], 'archived_MLP_K200': [], 'archived_SASRec_all_train': []}
    inputs = {}
    for seed in (0, 1, 2):
        path = REPO_ROOT/f'results/ranker_k1_sweep/20261025-k1/seed{seed}_ranks.npz'
        inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
        with np.load(path) as z:
            if not np.array_equal(z['inst'], inst.inst) or not np.array_equal(z['user'], users): raise ValueError('historical sweep alignment differs')
            for k in (100, 200): ranks[f'archived_MLP_K{k}'].append(z['ranks'][list(z['candidate_counts']).index(k)])
        path = REPO_ROOT/f'results/baselines/sasrec/all_train/seed{seed}/val_ranks.npz'
        inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
        with np.load(path) as z:
            if not np.array_equal(z['inst'], inst.inst) or not np.array_equal(z['user'], users): raise ValueError('historical baseline alignment differs')
            ranks['archived_SASRec_all_train'].append(np.where(z['scorable'], z['rank'], np.inf))
    for method in METHODS:
        ranks[method] = []
        for seed in (0, 1, 2):
            path = root/f'evaluation/{method}/seed{seed}/val_ranks.npz'
            inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
            with np.load(path) as z: ranks[method].append(z['rank'])
    ranks['retrieval'] = [np.asarray(arrays(root/'candidates/val')['retrieval_rank'])]
    retrieval_path = root/'candidates/val/retrieval_rank.npy'
    inputs[str(retrieval_path.relative_to(REPO_ROOT))] = digest(retrieval_path)
    groups = groups_for(data, inst)
    metrics = []; contrasts = []; secondary = []
    for group in ('all', 'new_in_val', 'warm_in_val'):
        mask = groups[group]
        values = {m: np.mean([M.ndcg_at(r, 10) for r in rs], axis=0) for m, rs in ranks.items()}
        for method, val in values.items():
            metrics.append({'group': group, 'method': method, 'ndcg10': average(val[mask], users[mask]),
                            'events': int(mask.sum()), 'users': int(np.unique(users[mask]).size)})
        for reference in ('archived_MLP_K100', 'archived_MLP_K200', 'archived_SASRec_all_train'):
            for candidate in METHODS:
                ci = paired_bootstrap(users[mask], values[reference][mask], values[candidate][mask], n_resamples=1000, seed=0)
                contrasts.append({'group': group, 'reference': reference, 'candidate': candidate, **ci})
        # Complete the missing arm pairs as transparently secondary analyses.
        for reference, candidate in [('retrieval', 'bce_text'), ('pairwise_text', 'pairwise_combined'),
                                     ('bce_text', 'pairwise_coreview'), ('bce_text', 'pairwise_combined')]:
            ci = paired_bootstrap(users[mask], values[reference][mask], values[candidate][mask], n_resamples=1000, seed=0)
            secondary.append({'group': group, 'reference': reference, 'candidate': candidate,
                              'analysis': 'secondary exploratory contrast, not registered primary', **ci})
    pd.DataFrame(metrics).to_csv(destination/'historical_context_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(destination/'historical_context_differences.csv', index=False)
    pd.DataFrame(secondary).to_csv(destination/'secondary_differences.csv', index=False)
    lines = ['# Historical pipeline context', '',
             'Aligned full-Val comparison, exploratory and unadjusted. These are different pipelines/candidate pools, '
             'training policies and encoder artifacts, not matched objective or feature ablations. '
             'Archived K100 was the observed validation maximum, without final K selection. '
             'No historical model was retrained or Test prediction loaded.', '',
             '| Method | Overall nDCG@10 | New | Warm |', '|---|---:|---:|---:|']
    table = pd.DataFrame(metrics).pivot(index='method', columns='group', values='ndcg10')
    for method, row in table.iterrows():
        lines.append(f'| {method} | {row["all"]:.6f} | {row.new_in_val:.6f} | {row.warm_in_val:.6f} |')
    lines += ['', 'Paired-user uncertainty for every new arm versus every reference is in historical_context_differences.csv.', '']
    (destination/'historical_context.md').write_text('\n'.join(lines))
    write_json(destination/'historical_context_provenance.json', {'input_sha256': inputs,
        'source_sha256': digest(Path(__file__)), 'test_predictions_loaded': False,
        'output_sha256': {name: digest(destination/name) for name in
                          ('historical_context_metrics.csv', 'historical_context_differences.csv', 'secondary_differences.csv', 'historical_context.md')}})


def publish(root, destination, preview=False):
    audit = read(root/'completion_audit.json')
    if audit['status'] != 'pass' or audit['sample']:
        raise ValueError('only a complete audited full-data result can be published')
    meta = read(root/'study.json')
    frame = pd.read_csv(root/'report/metrics.csv')
    contrasts = pd.read_csv(root/'report/paired_differences.csv')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/cascade-full-ranking-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'text.color': '#172b3a', 'axes.labelcolor': '#172b3a'})
    order = ['retrieval', *METHODS]
    colors = ['#96a7b3', '#d17a30', '#247ba0', '#188977', '#8064a2']
    def rows(group):
        return frame.loc[(frame.group == group) & (frame.metric == 'ndcg10')].set_index('method').loc[order]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.16, h_pad=.42, wspace=.12, hspace=.13)
    fig.suptitle('Full-data replication on fixed frozen-recency K200 candidates', fontsize=19, fontweight='bold')
    ax = axes[0, 0]
    all_rows = rows('all')
    ax.barh(np.arange(5), all_rows.value, height=.55, color=colors)
    ax.set_yticks(np.arange(5), [LABELS[m] for m in order]); ax.invert_yaxis()
    maximum = all_rows.value.max()
    for i, row in enumerate(all_rows.itertuples()):
        ax.text(maximum*1.30, i, f'{row.value:.6f}', va='center', ha='right', fontsize=10)
        if row.Index != 'retrieval':
            ax.scatter([getattr(row, f'seed{s}') for s in (0,1,2)], [i]*3, c='#172b3a', s=15, zorder=3)
    ax.set_xlim(0, maximum*1.35); ax.set_xlabel('End-to-end nDCG@10 · dots: training seeds')
    ax.set_title('A  Overall quality', loc='left', pad=12, fontweight='bold')
    ax.xaxis.grid(True, color='#e8edf0'); ax.set_axisbelow(True)
    ax = axes[0, 1]
    for off, group, label, color in [(-.18, 'new_in_val', 'New in validation', '#188977'),
                                     (.18, 'warm_in_val', 'Warm in validation', '#247ba0')]:
        ax.bar(np.arange(5)+off, rows(group).value, width=.34, label=label, color=color)
    ax.set_xticks(np.arange(5), ['Retrieval', 'BCE', 'Pairwise\ntext', '+ co-review', 'Combined'])
    ax.set_ylabel('End-to-end nDCG@10'); ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, max(rows('new_in_val').value.max(), rows('warm_in_val').value.max())*1.33)
    ax.set_title('B  New / warm trade-off', loc='left', pad=12, fontweight='bold')
    ax.yaxis.grid(True, color='#e8edf0'); ax.set_axisbelow(True)
    ax = axes[1, 0]
    selected = contrasts.loc[(contrasts.group == 'all') & (contrasts.metric == 'ndcg10')]
    labels = ['Pairwise − retrieval', 'Pairwise − BCE', 'Co-review − pairwise',
              'Combined − co-review', 'Co-review − retrieval', 'Combined − retrieval']
    for i, row in enumerate(selected.itertuples()):
        color = '#188977' if row.ci_low > 0 else '#d17a30' if row.ci_high < 0 else '#96a7b3'
        ax.errorbar(row.diff, i, xerr=[[row.diff-row.ci_low], [row.ci_high-row.diff]], fmt='o',
                    color=color, capsize=4, ms=6, lw=2)
    ax.axvline(0, lw=1, color='#7f8b94'); ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
    ax.set_xlabel('nDCG@10 difference · paired-user 95% interval')
    ax.set_title('C  Registered contrasts', loc='left', pad=12, fontweight='bold')
    ax.xaxis.grid(True, color='#e8edf0'); ax.set_axisbelow(True); ax.xaxis.set_major_locator(MaxNLocator(5))
    ax = axes[1, 1]
    monthly = frame.loc[frame.group.str.fullmatch(r'month_\d{4}-\d{2}') & (frame.metric == 'ndcg10')]
    months = sorted(monthly.group.unique())
    for method, color, marker in zip(order, colors, ['o', 's', '^', 'D', 'v']):
        rr = monthly.loc[monthly.method == method].set_index('group').loc[months]
        ax.plot(np.arange(len(months)), rr.value, label=LABELS[method], color=color, marker=marker, ms=4, lw=1.5)
    ticks = np.arange(0, len(months), 2)
    ax.set_xticks(ticks, [months[i].replace('month_', '') for i in ticks], rotation=25)
    ax.set_ylabel('End-to-end nDCG@10'); ax.set_xlabel('Validation month · UTC; boundary months partial')
    ax.set_title('D  Monthly outcomes · same fitted models', loc='left', pad=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, ncol=2); ax.yaxis.grid(True, color='#e8edf0'); ax.set_axisbelow(True)
    fig.text(.02, .016, '232,904 events / 124,416 users · 3 training seeds · exploratory reused Val; unadjusted user-bootstrap intervals\nNatural candidates, unchanged filtering and strictly pre-T0 graph · monthly slices are not independent temporal retraining · no held-out Test', fontsize=10)
    figure = root/'report/comparison.png'
    fig.savefig(figure, dpi=160)
    fig.savefig(root/'report/comparison.pdf')
    plt.close(fig)
    if preview:
        print(f'Comparison figure prepared: {figure}', flush=True)
        return
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('metrics.csv', 'paired_differences.csv', 'failure_funnel.csv', 'fits.csv', 'report.md', 'provenance.json'):
        shutil.copyfile(root/'report'/name, destination/name)
    for name in ('protocol.json', 'config.yaml', 'git_hash.txt', 'selection_lock.json', 'completion_audit.json'):
        shutil.copyfile(root/name, destination/name)
    historical_context(root, destination, meta)
    # Relative paths and hashes only; no machine/user paths or per-event identities.
    write_json(destination/'study_provenance.json', {key: meta[key] for key in
        ('sample', 'config', 'split', 'git_hash', 'source_sha256', 'input_sha256', 'protocol_sha256',
         'text_path', 'device', 'test_predictions_loaded', 'inner_split', 'artifact_sha256')})
    write_json(destination/'environment.json', {'python': platform.python_version(), 'system': platform.system(),
        'architecture': platform.machine(), 'macos': platform.mac_ver()[0], 'training_device': meta['device'],
        'libraries': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pandas', 'scipy', 'matplotlib')}})
    figure_destination = REPO_ROOT/'docs/figures/20261103-full-promising-ranking.png'
    shutil.copyfile(figure, figure_destination)
    write_json(destination/'publication_manifest.json', {
        'local_study': str(root.relative_to(REPO_ROOT)), 'source_sha256': digest(Path(__file__)),
        'audit_sha256': digest(root/'completion_audit.json'),
        'shared_sha256': {p.name: digest(p) for p in destination.iterdir() if p.is_file()},
        'figure_sha256': digest(figure_destination), 'test_predictions_loaded': False})
    print(f'Published compact evidence: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-full-promising-ranking')
    parser.add_argument('--destination', default='results/full_data_replication/20261103-promising')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    publish(study_path(args.study), REPO_ROOT/args.destination, args.preview)
