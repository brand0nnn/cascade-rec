"""Factorial effects, joint-user novelty contrast and retained-score audit."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines.objective_coreview import CELLS, OLD, checked
from cascade.utils.config import REPO_ROOT


def interaction(values):
    """Per-event graph gain under pairwise minus its gain under BCE."""
    return (values['pairwise_coreview']-values['pairwise_text'])-(values['bce_coreview']-values['bce_text'])


def novelty_difference(users, values, new, n_resamples=1000, seed=0):
    """Jointly resample users, preserving separate new/warm macro denominators."""
    users, values, new = np.asarray(users), np.asarray(values), np.asarray(new, bool)
    unique, inverse = np.unique(users, return_inverse=True)
    per_group = []
    for mask in (new, ~new):
        counts = np.bincount(inverse[mask], minlength=len(unique))
        sums = np.bincount(inverse[mask], weights=values[mask], minlength=len(unique))
        per_group.append(np.divide(sums, counts, out=np.full(len(unique), np.nan), where=counts > 0))
    if any(not np.isfinite(v).any() for v in per_group): raise ValueError('both novelty groups require support')
    rng = np.random.default_rng(seed); boots = []
    for start in range(0, n_resamples, 32):
        idx = rng.integers(0, len(unique), size=(min(32, n_resamples-start), len(unique)))
        a, b = per_group[0][idx], per_group[1][idx]
        if ((np.isfinite(a).sum(1) == 0) | (np.isfinite(b).sum(1) == 0)).any():
            raise ValueError('bootstrap draw has no support in a novelty group')
        boots.extend(np.nanmean(a, axis=1)-np.nanmean(b, axis=1))
    low, high = np.percentile(boots, [2.5, 97.5])
    return {'diff': float(np.nanmean(per_group[0])-np.nanmean(per_group[1])),
            'ci_low': float(low), 'ci_high': float(high), 'n_users': len(unique), 'n_resamples': n_resamples}


def load_ranks(root, meta, parent):
    archive = P.arrays(parent/'candidates/val')
    ranks = {'retrieval': np.asarray(archive['retrieval_rank'])[None, :]}
    inputs = {}
    for method in CELLS:
        values = []
        for seed in meta['config']['seeds']:
            path = root/f'evaluation/seed{seed}/val_ranks.npz' if method == 'bce_coreview' else parent/f'evaluation/{method}/seed{seed}/val_ranks.npz'
            inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
            with np.load(path) as z:
                if not np.array_equal(z['inst'], archive['inst']) or not np.array_equal(z['user'], archive['user']):
                    raise ValueError('factorial event alignment changed')
                if not np.array_equal(np.isfinite(z['rank']), np.isfinite(archive['retrieval_rank'])):
                    raise ValueError('factorial candidate membership changed')
                values.append(z['rank'])
        ranks[method] = np.stack(values)
    return archive, ranks, inputs


def report(root):
    meta, parent, previous = checked(root)
    if P.read(root/'evaluation/completion.json')['status'] != 'complete': raise ValueError('outer scoring incomplete')
    out = root/'report'; out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out/'analysis_source.py')
    archive, ranks, inputs = load_ranks(root, meta, parent)
    data = P.data_for(previous); inst = data.split_instances('val'); users = archive['user']
    groups = P.groups_for(data, inst); rows = []; contrasts = []; interactions = []; funnels = []
    functions = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall10': lambda r: M.recall_at(r, 10),
                 'recall200': lambda r: M.recall_at(r, 200)}
    all_values = {k: {m: np.stack([fn(r) for r in rr]) for m, rr in ranks.items()} for k, fn in functions.items()}
    found = np.isfinite(archive['retrieval_rank']); old_hit = archive['retrieval_rank'] <= 10
    for group, mask in groups.items():
        for metric, values in all_values.items():
            means = {m: v.mean(0) for m, v in values.items()}
            for method, v in values.items():
                seeds = [average(x[mask], users[mask]) for x in v]
                row = {'group': group, 'method': method, 'metric': metric, 'events': int(mask.sum()),
                       'users': int(np.unique(users[mask]).size), 'natural_targets': int((mask & found).sum()),
                       'value': float(np.mean(seeds)), 'seed_std_ddof0': float(np.std(seeds))}
                if method != 'retrieval': row.update({f'seed{s}': value for s, value in zip(meta['config']['seeds'], seeds)})
                if metric == 'ndcg10': row['conditional_ndcg10'] = float(np.mean([average(x[mask & found], users[mask & found]) for x in v]))
                rows.append(row)
            for reference, candidate in meta['config']['primary_contrasts']:
                ci = paired_bootstrap(users[mask], means[reference][mask], means[candidate][mask],
                                      n_resamples=meta['config']['bootstrap_resamples'], seed=meta['config']['bootstrap_seed'])
                contrasts.append({'group': group, 'metric': metric, 'reference': reference, 'candidate': candidate, **ci})
            effect = interaction(means)
            ci = paired_bootstrap(users[mask], np.zeros(mask.sum()), effect[mask],
                                  n_resamples=meta['config']['bootstrap_resamples'], seed=meta['config']['bootstrap_seed'])
            interactions.append({'group': group, 'metric': metric, 'contrast': meta['config']['interaction'], **ci})
        for method, rr in ranks.items():
            for seed, r in enumerate(rr):
                hit = r <= 10
                flags = {'retrieval_miss': ~found, 'unrescued': found & ~old_hit & ~hit,
                         'rescued': ~old_hit & hit, 'demoted': old_hit & ~hit, 'retained': old_hit & hit}
                if not np.all(np.stack(list(flags.values())).sum(0) == 1): raise ValueError('invalid factorial funnel')
                funnels.append({'group': group, 'method': method, 'seed': seed if method != 'retrieval' else None,
                                **{k: int((v & mask).sum()) for k, v in flags.items()}, 'top10_hits': int((mask & hit).sum())})
    effect = interaction({m: v.mean(0) for m, v in all_values['ndcg10'].items()})
    novelty = novelty_difference(users, effect, groups['new_in_val'], meta['config']['bootstrap_resamples'], meta['config']['bootstrap_seed'])
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_differences.csv', index=False)
    pd.DataFrame(interactions).to_csv(out/'interactions.csv', index=False)
    pd.DataFrame(funnels).to_csv(out/'failure_funnel.csv', index=False)
    write_json(out/'novelty_interaction.json', {'contrast': meta['config']['novelty_contrast'], **novelty,
        'bootstrap': 'joint user-cluster draws with separate group user-macro means',
        'interpretation': 'groups have different qualifying populations; descriptive effect contrast'})
    frame = pd.DataFrame(rows)
    def value(method, group, metric):
        return float(frame.loc[(frame.method == method) & (frame.group == group) & (frame.metric == metric), 'value'].iloc[0])
    lines = [('# Technical sample workflow check' if meta['sample'] else '# Training objective x additional co-review evidence'), '',
             'Exploratory completion of a previously inspected validation factorial. Three new BCE+co-review fits '
             'reuse the exact parent candidates/features/plans and nine existing locked control fits. '
             'All cells retain basic metadata matching and nine common support/degree inputs. Only four additional '
             'co-review columns are switched. No Test predictions or graph retrieval experiment.', '',
             '| Objective | Additional co-review | Overall nDCG@10 | Recall@10 | New | Warm |',
             '|---|---|---:|---:|---:|---:|']
    for method in CELLS:
        vals = [value(method, 'all', 'ndcg10'), value(method, 'all', 'recall10'),
                value(method, 'new_in_val', 'ndcg10'), value(method, 'warm_in_val', 'ndcg10')]
        lines.append('| '+('BCE' if method.startswith('bce') else 'Pairwise')+' | '+('On' if method.endswith('coreview') else 'Off')+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines += ['', 'Interaction = graph gain under pairwise minus graph gain under BCE. '
              'A positive value favours the incremental graph effect under pairwise. '
              'Inference uses the direct paired contrast, not comparison of significance across cells.', '',
              '| Group | nDCG@10 interaction [95% CI] |', '|---|---:|']
    for r in pd.DataFrame(interactions).query('metric == "ndcg10" and group in ["all", "new_in_val", "warm_in_val"]').itertuples():
        lines.append(f'| {r.group} | {r.diff:+.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}] |')
    lines += ['', f'New-minus-warm interaction: **{novelty["diff"]:+.6f} [{novelty["ci_low"]:.6f}, {novelty["ci_high"]:.6f}]**.', '',
              'All intervals use 1,000 user-cluster resamples, seed 0, after event-metric averaging over training seeds. '
              'They are exploratory/unadjusted on reused Val and conditional on these trained runs/candidates. '
              'Selection follows the same rule but selected epochs/update counts can differ; this tests the whole '
              'objective/selection/refit recipe. Novelty-group means retain their own qualifying-user denominators. '
              'Monthly tables are descriptive slices, not independent temporal replications.', '']
    (out/'report.md').write_text('\n'.join(lines))
    write_json(out/'provenance.json', {'source_sha256': digest(Path(__file__)), 'input_sha256': inputs,
        'study_sha256': digest(root/'study.json'), 'selection_lock_sha256': digest(root/'selection_lock.json'),
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}, 'test_predictions_loaded': False})
    print('\n'.join(lines), flush=True)


def audit(root):
    from cascade.pipelines.full_data_ranking_audit import user_mean, ndcg
    meta, parent, previous = checked(root)
    lock = P.read(root/'selection_lock.json'); counts = {'inner_replays': 0, 'outer_replays': 0, 'metrics': 0, 'main_intervals': 0}
    if P.read(root/'evaluation/completion.json')['selection_lock_sha256'] != digest(root/'selection_lock.json'):
        raise ValueError('selection lock changed after scoring')
    train = P.arrays(parent/'candidates/ranker_train'); tune = np.flatnonzero(np.load(parent/'split/tune.npy'))
    retained = {}
    for seed in meta['config']['seeds']:
        run = root/f'runs/bce_coreview/seed{seed}'; fit = P.read(run/'metrics.json')
        if digest(run/'model.pth') != fit['model_sha256']: raise ValueError('new model changed')
        values = []
        for row in P.read(run/'inner_history.json'):
            e = row['epoch']; scores = np.load(run/f'inner/epoch{e}_logits.npy', mmap_mode='r')
            rr, tt = P.ranks_and_top10(scores, train['cand'][tune], train['target'][tune])
            with np.load(run/f'inner/epoch{e}_ranks.npz') as z:
                if not np.array_equal(rr, z['rank']) or not np.array_equal(tt, z['top10']): raise ValueError('inner replay differs')
            v = user_mean(ndcg(rr), train['user'][tune]); values.append(v)
            if not np.isclose(v, row['ndcg10'], atol=1e-15, rtol=0): raise ValueError('inner metric differs')
            counts['inner_replays'] += 1
        if int(np.argmax(values)) != fit['selected_epoch'] or len(P.read(run/'refit_history.json')) != fit['selected_epoch']:
            raise ValueError('new selection or refit differs')
        out = root/f'evaluation/seed{seed}'; proof = P.read(out/'provenance.json')
        if (digest(out/'logits.npy') != proof['logits_sha256'] or digest(run/'model.pth') != proof['model_sha256']
                or digest(out/'val_ranks.npz') != proof['ranks_sha256']):
            raise ValueError('new evaluation identity differs')
        archive = P.arrays(parent/'candidates/val')
        rr, tt = P.ranks_and_top10(np.load(out/'logits.npy', mmap_mode='r'), archive['cand'], archive['target'])
        with np.load(out/'val_ranks.npz') as z:
            if not np.array_equal(rr, z['rank']) or not np.array_equal(tt, z['top10']): raise ValueError('outer replay differs')
        counts['outer_replays'] += 1
        for method in CELLS:
            m = lock[f'{method}/seed{seed}']; owner = REPO_ROOT/m['owner']
            actual = P.read(owner/f'runs/{method}/seed{seed}/metrics.json')
            if ({'owner': m['owner'], **actual} != m
                    or digest(owner/f'runs/{method}/seed{seed}/model.pth') != m['model_sha256']):
                raise ValueError('factorial selected lock changed')
        for path in run.rglob('*'):
            if path.is_file(): retained[str(path.relative_to(root))] = digest(path)
        for path in out.iterdir(): retained[str(path.relative_to(root))] = digest(path)
    archive, ranks, inputs = load_ranks(root, meta, parent)
    data = P.data_for(previous); groups = P.groups_for(data, data.split_instances('val')); users = archive['user']
    vectors = {metric: {m: np.mean([ndcg(r) if metric == 'ndcg10' else (r <= (10 if metric == 'recall10' else 200)).astype(float) for r in rr], axis=0)
                        for m, rr in ranks.items()} for metric in ('ndcg10', 'recall10', 'recall200')}
    for row in pd.read_csv(root/'report/metrics.csv').itertuples():
        mask = groups[row.group]
        if not np.isclose(user_mean(vectors[row.metric][row.method][mask], users[mask]), row.value, atol=1e-14, rtol=0):
            raise ValueError('factorial metric reconstruction differs')
        counts['metrics'] += 1
    for filename in ('paired_differences.csv', 'interactions.csv'):
        frame = pd.read_csv(root/'report'/filename)
        for row in frame.loc[frame.metric.eq('ndcg10') & frame.group.isin(['all', 'new_in_val', 'warm_in_val'])].itertuples():
            mask = groups[row.group]; _, inv = np.unique(users[mask], return_inverse=True)
            v = vectors['ndcg10']
            delta = (v[row.candidate]-v[row.reference])[mask] if filename.startswith('paired') else interaction(v)[mask]
            per_user = np.bincount(inv, weights=delta)/np.bincount(inv)
            rng = np.random.default_rng(0); draws = []
            for start in range(0, 1000, 32):
                indices = rng.integers(0, len(per_user), size=(min(32, 1000-start), len(per_user)))
                draws.extend(per_user[indices].mean(1))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            if not np.allclose([per_user.mean(), lo, hi], [row.diff, row.ci_low, row.ci_high], atol=1e-14, rtol=0):
                raise ValueError('factorial paired interval differs')
            counts['main_intervals'] += 1
    effect = interaction(vectors['ndcg10'])
    table = pd.DataFrame({'user': users, 'new': groups['new_in_val'], 'effect': effect}).groupby(['user','new']).effect.mean().unstack('new')
    new, warm = table[True].to_numpy(), table[False].to_numpy()
    rng = np.random.default_rng(0); draws = []
    for start in range(0, 1000, 32):
        idx = rng.integers(0, len(table), size=(min(32, 1000-start), len(table)))
        draws.extend(np.nanmean(new[idx], axis=1)-np.nanmean(warm[idx], axis=1))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    saved = P.read(root/'report/novelty_interaction.json')
    if not np.allclose([np.nanmean(new)-np.nanmean(warm),lo,hi], [saved['diff'],saved['ci_low'],saved['ci_high']], atol=1e-14, rtol=0):
        raise ValueError('joint-user novelty contrast reconstruction differs')
    counts['joint_novelty_interval'] = 1
    prov = P.read(root/'report/provenance.json')
    for n, h in prov['output_sha256'].items():
        if digest(root/'report'/n) != h: raise ValueError('factorial report changed')
    write_json(root/'completion_audit.json', {'status': 'pass', 'sample': meta['sample'], **counts,
        'retained_sha256': retained, 'parent_completion_audit_sha256': digest(parent/'completion_audit.json'),
        'source_sha256': digest(Path(__file__)), 'test_predictions_loaded': False})
    print({'audit': 'pass', **counts}, flush=True)


def publish(root, preview=False):
    meta, parent, previous = checked(root)
    audit_meta = P.read(root/'completion_audit.json')
    if audit_meta['status'] != 'pass' or meta['sample']: raise ValueError('audited full result required')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/cascade-objective-coreview-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(root/'report/metrics.csv'); effects = pd.read_csv(root/'report/interactions.csv')
    fig = plt.figure(figsize=(14, 8), layout='constrained'); grid = fig.add_gridspec(2, 3)
    fig.set_constrained_layout_pads(w_pad=.14, h_pad=.35, wspace=.16, hspace=.15)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    values = frame.loc[frame.metric.eq('ndcg10') & frame.group.isin(['all', 'new_in_val', 'warm_in_val']) & frame.method.isin(CELLS)]
    low, high = values.value.min(), values.value.max(); pad = (high-low)*.35
    for ax, group, label in zip(axes, ('all', 'new_in_val', 'warm_in_val'), ('All validation', 'New products', 'Warm products')):
        table = values.loc[values.group.eq(group)].set_index('method')
        for methods, color, name in [(('bce_text','bce_coreview'), '#d17a30', 'BCE'),
                                     (('pairwise_text','pairwise_coreview'), '#247ba0', 'Pairwise')]:
            y = [table.loc[m, 'value'] for m in methods]
            ax.plot([0,1], y, 'o-', color=color, label=name, lw=2)
            other = ('pairwise_text','pairwise_coreview') if name == 'BCE' else ('bce_text','bce_coreview')
            for x, v in enumerate(y):
                higher = v > table.loc[other[x], 'value'] or (v == table.loc[other[x], 'value'] and name == 'BCE')
                ax.annotate(f'{v:.6f}', (x,v), xytext=(0,8 if higher else -16), textcoords='offset points', ha='center', fontsize=10)
        ax.set_xticks([0,1], ['Co-review off', 'Co-review on']); ax.set_ylim(low-pad, high+pad)
        ax.set_xlim(-.25,1.25); ax.set_title(label); ax.set_ylabel('End-to-end nDCG@10'); ax.grid(axis='y', alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False)
    ax = fig.add_subplot(grid[1, :]); selected = effects.loc[effects.metric.eq('ndcg10') & effects.group.isin(['all','new_in_val','warm_in_val'])]
    for i, r in enumerate(selected.itertuples()):
        ax.errorbar(r.diff, i, xerr=[[r.diff-r.ci_low],[r.ci_high-r.diff]], fmt='o', capsize=4, color='#247ba0')
    ax.set_yticks(range(3), ['All validation','New products','Warm products']); ax.invert_yaxis(); ax.axvline(0,color='#888888',lw=1)
    ax.set_xlabel('Interaction: pairwise graph gain minus BCE graph gain · paired-user 95% interval')
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Does training objective change the value of co-review?', fontsize=18)
    fig.text(.02,.015,'Three training seeds · fixed full-data K200 candidates · reused Val, exploratory/unadjusted intervals · no held-out Test',fontsize=10)
    fig.savefig(root/'report/comparison.png',dpi=160); fig.savefig(root/'report/comparison.pdf'); plt.close(fig)
    if preview:
        print(f'Prepared figure: {root / "report/comparison.png"}', flush=True)
        return
    destination = REPO_ROOT/'results/full_data_replication/20261104-objective-coreview'
    destination.mkdir(parents=True, exist_ok=False)
    for n in ('study.json','protocol.json','config.yaml','selection_lock.json','completion_audit.json'):
        shutil.copyfile(root/n, destination/n)
    for p in (root/'report').iterdir():
        if p.suffix in ('.csv','.json','.md') or p.name == 'analysis_source.py': shutil.copyfile(p,destination/p.name)
    figure = REPO_ROOT/'docs/figures/20261104-objective-coreview.png'; shutil.copyfile(root/'report/comparison.png',figure)
    write_json(destination/'publication_manifest.json', {'source_sha256': digest(Path(__file__)),
        'files_sha256': {p.name:digest(p) for p in destination.iterdir() if p.is_file()},
        'figure_sha256':digest(figure), 'test_predictions_loaded':False})
    print(f'Shared audited factorial: {destination}',flush=True)
