"""Validation-only seed replication and meso analysis from aligned saved ranks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.segments import hist_bucket, new_in_period
from cascade.utils.io import REPO_ROOT

DISCOVERY = '20261024-feature-ladder-seed0'
MODELS = {'matching': 1, 'attributes': 2}


def group_labels(data: ProcessedData, instances: pd.DataFrame) -> pd.DataFrame:
    """Use existing item segments; history diversity uses only last 50 prior items."""
    item_table = data.items.set_index('item')
    targets = item_table.loc[instances['item'].to_numpy()]
    new = new_in_period(targets['first_ts'].to_numpy(), 'val', data.split)
    existing = targets['segment'].astype(str).to_numpy()
    item_group = np.where(new, 'new-in-period', np.where(existing == 'head', 'warm-head', 'warm-tail'))
    histories = data.histories(instances['user'].to_numpy(), instances['hist_len'].to_numpy(), data.max_hist_len)
    names = data.items['leaf_cat'].fillna('').astype(str)
    codes, _ = pd.factorize(names, sort=True)
    lut = np.zeros(data.n_items + 1, dtype=np.int32)
    lut[data.items['item'].to_numpy()] = np.where(names.to_numpy() != '', codes + 2, 1)
    categories = np.sort(lut[histories], axis=1)
    distinct = np.sum((categories > 1) & np.concatenate([
        np.ones((len(categories), 1), dtype=bool), np.diff(categories, axis=1) != 0
    ], axis=1), axis=1)
    length = hist_bucket(instances['hist_len'].to_numpy())
    diversity = length.astype(object).copy()
    long = instances['hist_len'].to_numpy() >= 11
    diversity[long & (distinct >= 3)] = '11+ / mixed'
    diversity[long & (distinct >= 1) & (distinct <= 2)] = '11+ / homogeneous'
    diversity[long & (distinct == 0)] = '11+ / unknown categories'
    days = (instances['ts'].to_numpy() - int(data.split['t1'])) / 86400000.0
    window = np.select([days < 90, days < 180, days < 270], ['days 0-89', 'days 90-179', 'days 180-269'], default='days 270+')
    return pd.DataFrame({'inst': instances['inst'].to_numpy(), 'user': instances['user'].to_numpy(),
                         'item_type': item_group, 'history_length': length,
                         'history_diversity': diversity, 'validation_window': window,
                         'new_at_val_entry': new, 'known_history_categories': distinct})


def aligned_archive(path: Path, inst: np.ndarray | None = None, users: np.ndarray | None = None) -> dict:
    """Fail on mismatched cases/order instead of comparing different populations."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if inst is not None and not np.array_equal(a['inst'], inst):
        raise ValueError(f'Instance alignment mismatch: {path}')
    if users is not None and not np.array_equal(a['user'], users):
        raise ValueError(f'User alignment mismatch: {path}')
    return a


def summarize_group(users: np.ndarray, baseline: np.ndarray, ranks: np.ndarray, mask: np.ndarray,
                    n_resamples: int = 1000) -> dict:
    """Means across seed-specific metrics, never an ensemble of scores/ranks."""
    u = users[mask]; r0 = baseline[mask]; rr = ranks[:, mask]
    baseline_ndcg = M.ndcg_at(r0, 10)
    values = np.stack([M.ndcg_at(r, 10) for r in rr])
    found = np.isfinite(r0)
    r0hit = r0 <= 10
    per_seed = []
    for r, v in zip(rr, values):
        per_seed.append({
            'ndcg10_macro': M.macro_micro(v, u)['macro'], 'ndcg10_micro': M.macro_micro(v, u)['micro'],
            'recall10_macro': M.macro_micro(M.recall_at(r, 10), u)['macro'],
            'conditional_ndcg10_macro': M.macro_micro(v[found], u[found])['macro'],
            'rescued': int(np.sum(found & ~r0hit & (r <= 10))),
            'demoted': int(np.sum(found & r0hit & (r > 10))),
            'retained': int(np.sum(found & r0hit & (r <= 10))),
            'unrescued': int(np.sum(found & ~r0hit & (r > 10))),
        })
    base = M.macro_micro(baseline_ndcg, u)
    macro = np.array([s['ndcg10_macro'] for s in per_seed])
    ci = paired_bootstrap(u, baseline_ndcg, values.mean(axis=0), n_resamples=n_resamples, seed=0)
    return {
        'n_instances': int(mask.sum()), 'n_users': int(np.unique(u).size),
        'n_retrieved': int(found.sum()), 'n_retrieved_users': int(np.unique(u[found]).size),
        'retrieval_recall200_micro': float(found.mean()),
        'retrieval_recall200_macro': M.macro_micro(found.astype(float), u)['macro'],
        'baseline_ndcg10_macro': base['macro'], 'baseline_ndcg10_micro': base['micro'],
        'baseline_conditional_ndcg10_macro': M.macro_micro(baseline_ndcg[found], u[found])['macro'],
        'ndcg10_macro_mean': float(macro.mean()),
        'ndcg10_macro_seed_std': float(macro.std(ddof=1)) if len(macro) > 1 else None,
        'ndcg10_micro_mean': float(np.mean([s['ndcg10_micro'] for s in per_seed])),
        'conditional_ndcg10_macro_mean': float(np.mean([s['conditional_ndcg10_macro'] for s in per_seed])),
        'delta_vs_r1_user_bootstrap': ci,
        'positive_seeds_vs_r1': int(np.sum(macro > base['macro'])),
        'rescued_mean': float(np.mean([s['rescued'] for s in per_seed])),
        'demoted_mean': float(np.mean([s['demoted'] for s in per_seed])),
        'per_seed': per_seed,
    }


def fmt(x: float | None, digits: int = 6) -> str:
    return '—' if x is None or not np.isfinite(x) else f'{x:.{digits}f}'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeat-id', default='20261024-shortlist-repeat')
    ap.add_argument('--din-id', help='include completed DIN seeds; save a separate study report')
    args = ap.parse_args()
    root = REPO_ROOT / 'results' / ('ranker_din_study' if args.din_id else 'ranker_repeat') / (args.din_id or args.repeat_id)
    root.mkdir(parents=True, exist_ok=True)
    models = {}; reference = None
    runs = {
        model: [(seed, REPO_ROOT / 'results' / 'ranker_mlp' /
                 f"{DISCOVERY if seed == 0 else f'{args.repeat_id}-seed{seed}'}-rung{rung}_{model}")
                for seed in (0, 1, 2)]
        for model, rung in MODELS.items()
    }
    if args.din_id:
        runs['din'] = [(s, REPO_ROOT / 'results' / 'ranker_din' / f'{args.din_id}-seed{s}') for s in (0, 1, 2)]
    source_runs = {}
    for model, model_runs in runs.items():
        archives = []; seeds = []; metrics = []
        for seed, run in model_runs:
            mp = run / 'metrics.json'
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            if m.get('status') == 'running' or m.get('development_sample'):
                continue
            a = aligned_archive(run / 'val_ranks.npz', None if reference is None else reference['inst'], None if reference is None else reference['user'])
            if reference is None:
                reference = a
            if not np.array_equal(a['retrieval_ranks'], reference['retrieval_ranks']):
                raise ValueError('Retrieval baseline changed across runs')
            if not np.array_equal(np.isfinite(a['ranks']), np.isfinite(a['retrieval_ranks'])):
                raise ValueError('Ranker must preserve retrieval misses')
            archives.append(a['ranks']); seeds.append(seed); metrics.append(m)
            source_runs[f'{model}_seed{seed}'] = {
                'run_dir': str(run),
                'metrics_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
                'ranks_sha256': hashlib.sha256((run / 'val_ranks.npz').read_bytes()).hexdigest(),
            }
        if archives:
            models[model] = {'seeds': seeds, 'ranks': np.stack(archives), 'metrics': metrics}
    if reference is None:
        raise RuntimeError('No completed validation ranks found')
    data = ProcessedData.load()
    instances = data.instances.set_index('inst').loc[reference['inst']].reset_index()
    if not np.array_equal(instances['user'], reference['user']):
        raise ValueError('Saved ranks do not match dataset users')
    if not ((instances['period'] == 2) & (instances['ts'] >= data.split['t1']) & (instances['ts'] < data.split['t2'])).all():
        raise ValueError('This analysis is validation-only')
    labels = group_labels(data, instances)
    labels.to_parquet(root / 'validation_groups.parquet', index=False)
    masks = {'overall': {'all': np.ones(len(labels), bool)}}
    for axis in ('item_type', 'history_length', 'history_diversity', 'validation_window'):
        masks[axis] = {str(v): labels[axis].to_numpy() == v for v in sorted(labels[axis].unique())}
    output = {'scope': 'validation only', 'planned_seeds': [0, 1, 2], 'models': {}, 'novelty_by_window': [],
              'source_runs': source_runs}
    for axis, groups in masks.items():
        for label, mask in groups.items():
            if axis == 'validation_window':
                output['novelty_by_window'].append({'window': label, 'n_instances': int(mask.sum()),
                                                    'new_at_val_entry_fraction': float(labels.loc[mask, 'new_at_val_entry'].mean())})
    for model, m in models.items():
        print(f'Analyze {model}, completed seeds {m["seeds"]}', flush=True)
        report = {'seeds': m['seeds'], 'groups': {}}
        for axis, groups in masks.items():
            report['groups'][axis] = {}
            for label, mask in groups.items():
                report['groups'][axis][label] = summarize_group(reference['user'], reference['retrieval_ranks'], m['ranks'], mask)
        report['global_gauc_macro_by_seed'] = [x['best_val_gauc_macro'] for x in m['metrics']]
        output['models'][model] = report
    # Paired comparisons use common seeds, identical cases, and mean metrics.
    pairs = [('attributes', 'matching')]
    if 'din' in models:
        pairs.append(('din', 'matching'))
    for treatment, control in pairs:
        if treatment not in models or control not in models:
            continue
        common = sorted(set(models[control]['seeds']) & set(models[treatment]['seeds']))
        if not common:
            continue
        key = f'{treatment}_minus_{control}'
        output[key] = {'common_seeds': common, 'groups': {}}
        values = {}
        for name in (control, treatment):
            values[name] = np.stack([
                M.ndcg_at(models[name]['ranks'][models[name]['seeds'].index(s)], 10) for s in common
            ]).mean(axis=0)
        for axis, groups in masks.items():
            output[key]['groups'][axis] = {
                label: paired_bootstrap(reference['user'][mask], values[control][mask], values[treatment][mask], n_resamples=1000, seed=0)
                for label, mask in groups.items()
            }
    # Entry novelty at the two development/training windows, without inspecting Test.
    first = data.items.set_index('item')['first_ts']
    output['window_entry_novelty'] = []
    for code, name, start in [(1, 'ranker_train', data.split['t_ranker_train']), (2, 'val', data.split['t1'])]:
        cases = data.instances[data.instances['period'] == code]
        novelty = first.loc[cases['item']].to_numpy() >= start
        output['window_entry_novelty'].append({'period': name, 'n_instances': len(cases), 'new_in_period_fraction': float(novelty.mean())})
    output['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output['analysis_sources'] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ('ranker_meso.py', 'plot_ranker_meso.py', 'bootstrap.py', 'metrics.py', 'segments.py')
    }
    jp = root / 'meso.json'; jp.with_suffix('.tmp').write_text(json.dumps(output, indent=2) + '\n'); jp.with_suffix('.tmp').replace(jp)
    lines = ['# Ranker replication and group analysis', '', 'Validation only. Checkpoint rule and candidate lists are fixed. No Test evaluation.', '',
             'Training seeds vary the ranker; the retriever remains the same frozen seed-0 retriever.', '',
             '| Setup | Completed seeds | Macro nDCG@10 mean ± seed SD | R1 baseline | Difference, user-bootstrap 95% CI |',
             '|---|---|---:|---:|---|']
    for model, m in output['models'].items():
        g=m['groups']['overall']['all']; ci=g['delta_vs_r1_user_bootstrap']
        lines.append(f"| {model} | {m['seeds']} | {fmt(g['ndcg10_macro_mean'])} ± {fmt(g['ndcg10_macro_seed_std'])} | {fmt(g['baseline_ndcg10_macro'])} | {fmt(ci['diff'])} [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}] |")
    lines += ['', '## Who benefits?', '', 'Each row averages within user inside that group, then across users. Group means do not add up to the overall mean. Differences compare identical cases against retriever order. Rescue/demotion columns are mean raw counts across completed seeds.', '']
    for axis in ('item_type', 'history_length', 'history_diversity', 'validation_window'):
        lines += [f'### {axis}', '', '| Setup / group | Cases / users | Retrieved % | R1 nDCG@10 | Ranker nDCG@10 | Difference [95% CI] | Positive seeds | Rescued / demoted |', '|---|---:|---:|---:|---:|---|---:|---:|']
        for model, m in output['models'].items():
            for label, g in m['groups'][axis].items():
                ci=g['delta_vs_r1_user_bootstrap']
                lines.append(f"| {model} / {label} | {g['n_instances']:,} / {g['n_users']:,} | {g['retrieval_recall200_micro']*100:.2f} | {fmt(g['baseline_ndcg10_macro'])} | {fmt(g['ndcg10_macro_mean'])} | {fmt(ci['diff'])} [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}] | {g['positive_seeds_vs_r1']}/{len(m['seeds'])} | {g['rescued_mean']:.1f} / {g['demoted_mean']:.1f} |")
        lines += ['']
    for treatment, control in pairs:
        comparison = output.get(f'{treatment}_minus_{control}')
        if comparison is None:
            continue
        lines += [f'## {treatment} versus {control}', '',
                  f"Common seeds: {comparison['common_seeds']}. Paired end-to-end macro nDCG@10; positive favours {treatment}.", '',
                  '| Group | Difference [95% user CI] |', '|---|---|']
        for axis in ('overall', 'item_type', 'history_diversity'):
            for label, ci in comparison['groups'][axis].items():
                lines.append(f"| {label} | {fmt(ci['diff'])} [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}] |")
        lines += ['']
    if args.din_id:
        lines += ['DIN uses item/category candidate embeddings inside unnormalised target attention. In matching-only MLP these embeddings are used for history pooling, so this comparison includes the candidate identity interactions introduced by attention. It does not isolate attention independently of that information. Full recipe and reference links are in docs/ranker_experiment.md.', '']
    lines += ['', '## Definitions and limits', '',
              '- New-in-period means target first interaction at/after validation start: cold at window entry. Warm tail/head retain the project item labels based on counts before T2, after separating new-in-period targets. These are descriptive evaluation groups; the before-T2 counts are never passed to the ranker.',
              '- History length is prior interactions at target time. Diversity counts known leaf categories in the last 50 prior items; long histories are homogeneous at 1–2 distinct categories or mixed at 3+. Missing categories have an explicit unknown group when none is known.',
              '- Validation windows are elapsed days since T1: 0–89, 90–179, 180–269, 270+. Entry novelty remains defined relative to T1 throughout.',
              '- Conditional metrics use retrieved-target cases only and are saved in meso.json alongside end-to-end macro/micro metrics and support counts.',
              '- Seed SD describes training variability. CIs resample paired users on mean per-instance metrics across completed seeds; they do not capture training-seed uncertainty. Averaging metrics across runs is not an ensemble.',
              '- CIs are exploratory and unadjusted across groups. Group-specific gains do not establish causes or prove that gains differ between groups. Test replication and the planned three seeds are still needed.',
              '- Any report with fewer than three completed seeds is provisional. Thresholds and definitions were specified before examining these group results.', '']
    (root/'meso.md').write_text('\n'.join(lines))
    from cascade.eval.plot_ranker_meso import plot_groups
    plot_groups(output, root)
    print(f'Saved {root / "meso.md"}', flush=True)


if __name__ == '__main__':
    main()
