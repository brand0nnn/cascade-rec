"""Statistics timing on Val: T0 -> T1 column swap and a statistics-age curve (Val only, exploratory; no training).

The four frozen P1 rankers trained with T0 statistics are rescored on the Val natural-hit rows with product
statistics that differ only in timing.  Run 1 swaps T1 values into the T0 features column by column, or product
group by product group.  Run 2 varies the cut date (fixed date T1 - k months; fixed lag of L months before each
query's month).  Features come from ``freshness_val`` (same builder, cut-rule and reference checks, parent T0
scaler); scoring is ``freshness_val._score``.  Hypotheses: ``results/analysis/20261107-statistics-timing/README.md``.

    PYTHONPATH=src python -m cascade.analysis.statistics_timing --phase {features,score,analyse}
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from cascade.analysis import freshness_val as F
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, write_json
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

STUDY = REPO_ROOT/'results/local_experiments/20261107-statistics-timing'
COMPACT = REPO_ROOT/'results/analysis/20261107-statistics-timing'
FRESH = REPO_ROOT/F.STUDY
GRAPH_COMMON = [0, 1, 2, 7, 8]  # graph-derived common columns; 3-6 are snapshot-free metadata
SWAPS = {'X1_known': ('common', [1], None), 'X2_known_degree': ('common', [0, 1], None),
         'X3_all_common': ('common', GRAPH_COMMON, None), 'X6_old_products': ('both', None, 'old'),
         'X7_products_released_in_B': ('both', None, 'B')}
DATES = (3, 6, 9)        # fixed cut date T1 - k months (T1 and T0 come from freshness_val)
LAGS = (1, 3, 6, 9, 12)  # fixed lag before each query's month start (lag 0 = freshness_val monthly)


def shift_months(ms, months):
    """Millisecond timestamps moved back by whole calendar months (UTC)."""
    stamps = pd.to_datetime(np.asarray(ms, np.int64), unit='ms', utc=True)
    return np.asarray([int((t - pd.DateOffset(months=months)).value // 10**6) for t in stamps], np.int64)


def lag_cutoffs(ts, lag):
    starts = F.month_start_ms(ts)
    if lag == 0: return starts
    unique = np.unique(starts)
    return shift_months(unique, lag)[np.searchsorted(unique, starts)]


def swap(t0, t1, block, columns, product_mask):
    """T0 features with a subset replaced by T1 values (by column, or by candidate slot)."""
    out = {k: np.array(v, copy=True) for k, v in t0.items()}
    if product_mask is not None:
        for key in out: out[key][product_mask] = t1[key][product_mask]
    else:
        out[block][..., columns] = t1[block][..., columns]
    return out


def unknown_stats(common, cand, target, scaler):
    """Share of 'no reviews before the cut' among target and other candidate slots, and its AUC as a score."""
    mean, std = scaler['common']['mean'][1], scaler['common']['std'][1]
    known = (np.asarray(common[..., 1], np.float64)*std + mean) > 0.5
    valid = cand > 0; is_target = cand == target[:, None]
    unknown = ~known
    pos, neg = unknown[is_target], unknown[valid & ~is_target]
    p, n = pos.mean(), neg.mean()
    return {'target_unknown': float(p), 'other_unknown': float(n), 'auc_unknown': float(p*(1-n) + 0.5*(p*n + (1-p)*(1-n)))}


def phase_features():
    torch.set_num_threads(3)
    P, meta, data, inst, archive, hit = F._context()
    scaler = P.read(REPO_ROOT/F.PARENT/'features/manifest.json')['scaler']
    sub = F._sub_archive(archive, hit); ts = inst.ts.to_numpy(np.int64)
    STUDY.mkdir(parents=True, exist_ok=False)
    record = {'label': 'Val only, exploratory', 'hit_rows': int(len(hit))}
    for k in DATES:
        cut = int(shift_months([F.T1], k)[0])
        record[f'date_T1-{k}m'] = F._build_arm(P, data, inst, sub, hit, np.full(len(inst), cut, np.int64), scaler, STUDY, f'date_T1-{k}m')
    for lag in LAGS:
        cuts = lag_cutoffs(ts, lag)
        if (cuts > ts).any() or (cuts < F.T0).any(): raise ValueError('lag cutoff outside [T0, query)')
        record[f'lag{lag}'] = F._build_arm(P, data, inst, sub, hit, cuts, scaler, STUDY, f'lag{lag}')
    write_json(STUDY/'snapshots.json', record)


def phase_score():
    torch.set_num_threads(3)
    P, meta, data, inst, archive, hit = F._context()
    sub = F._sub_archive(archive, hit)
    first = np.full(data.n_items+1, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    cand_first = first[sub['cand']]
    t0 = {k: np.asarray(P.arrays(REPO_ROOT/F.PARENT/'features/val')[k][hit]) for k in ('common', 'graph')}
    t1 = {k: np.load(FRESH/f'features/T1/{k}.npy') for k in ('common', 'graph')}
    product = {'old': cand_first < F.T0, 'B': (cand_first >= F.T0) & (cand_first < F.T1)}
    built = {name: swap(t0, t1, block, cols, None if group is None else product[group]) for name, (block, cols, group) in SWAPS.items()}
    built.update({f'date_T1-{k}m': {key: np.load(STUDY/f'features/date_T1-{k}m/{key}.npy') for key in ('common', 'graph')} for k in DATES})
    built.update({f'lag{lag}': {key: np.load(STUDY/f'features/lag{lag}/{key}.npy') for key in ('common', 'graph')} for lag in LAGS})
    for name, feats in built.items():
        for method in F.METHODS:
            for seed in F.SEEDS:
                dest = STUDY/f'ranks/{name}/{method}/seed{seed}.npz'
                if dest.exists(): continue
                started = time.perf_counter()
                ranks, _ = F._score(P, meta, method, seed, sub, feats)
                full = np.full(len(inst), np.inf); full[hit] = ranks
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, inst=archive['inst'], user=archive['user'], rank=full)
                print(json.dumps({'scored': name, 'method': method, 'seed': seed, 'seconds': round(time.perf_counter()-started, 1)}), flush=True)


def rank_path(arm, method, seed):
    base = FRESH/'ranks' if arm in ('T0', 'T1', 'monthly') else STUDY/'ranks'
    return base/f'{arm}/{method}/seed{seed}.npz'


def phase_analyse():
    import yaml
    P, meta, data, inst, archive, hit = F._context()
    scaler = P.read(REPO_ROOT/F.PARENT/'features/manifest.json')['scaler']
    users = inst.user.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= F.T1
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    curve = ['T0', *[f'date_T1-{k}m' for k in sorted(DATES, reverse=True)], 'T1']
    lags = ['monthly', *[f'lag{lag}' for lag in LAGS]]
    all_arms = ['T0', 'T1', *SWAPS, *curve[1:-1], *lags]
    values, rows = {}, []
    for arm in all_arms:
        for method in F.METHODS:
            seeds = []
            for seed in F.SEEDS:
                with np.load(rank_path(arm, method, seed)) as z:
                    if not np.array_equal(z['inst'], inst.inst): raise ValueError('rank alignment')
                    seeds.append(F._metric_values(z['rank']))
            values[arm, method] = {m: np.mean([s[m] for s in seeds], axis=0) for m in seeds[0]}
            for group, mask in groups.items():
                per = [average(s['ndcg10'][mask], users[mask]) for s in seeds]
                rows.append({'arm': arm, 'method': method, 'group': group, 'ndcg10': float(np.mean(per)), 'seed_std_ddof0': float(np.std(per))})
    contrasts = []
    pairs = [(f'{arm} - T0', 'T0', arm) for arm in [*SWAPS, 'T1']] + [(f'T1 - {arm}', arm, 'T1') for arm in SWAPS]
    pairs += [(f'{b} - {a}', a, b) for a, b in zip(curve[:-1], curve[1:])]
    pairs += [(f'{arm} - monthly', 'monthly', arm) for arm in lags[1:]]
    for label, ref, cand in pairs:
        for method in F.METHODS:
            for group, mask in groups.items():
                ci = paired_bootstrap(users[mask], values[ref, method]['ndcg10'][mask], values[cand, method]['ndcg10'][mask], n_resamples=1000, seed=0)
                contrasts.append({'contrast': label, 'method': method, 'group': group, **ci})
        print(json.dumps({'contrast': label}), flush=True)
    sub = F._sub_archive(archive, hit); ts = inst.ts.to_numpy(np.int64)[hit]
    stats = []
    for arm in [*curve, *lags]:
        path = FRESH/f'features/{arm}/common.npy' if arm in ('T1', 'monthly') else STUDY/f'features/{arm}/common.npy'
        common = np.asarray(P.arrays(REPO_ROOT/F.PARENT/'features/val')['common'][hit]) if arm == 'T0' else np.load(path)
        if arm == 'T0': cut = np.full(len(hit), F.T0)
        elif arm == 'T1': cut = np.full(len(hit), F.T1)
        elif arm.startswith('date'): cut = np.full(len(hit), shift_months([F.T1], int(arm.split('-')[1][:-1]))[0])
        else: cut = lag_cutoffs(ts, 0 if arm == 'monthly' else int(arm[3:]))
        age = (ts - cut)/86400000/30.44
        stats.append({'arm': arm, 'age_months_mean': float(age.mean()), 'age_months_min': float(age.min()), 'age_months_max': float(age.max()),
                      **unknown_stats(common, sub['cand'], sub['target'], scaler)})
    b = P.arrays(REPO_ROOT/F.PARENT/'candidates/ranker_train'); found = np.isfinite(b['retrieval_rank'])
    b_ts = data.split_instances('ranker_train').ts.to_numpy(np.int64)[found]
    b_age = (b_ts - F.T0)/86400000/30.44
    training = {'arm': 'training (B natural lists, T0)', 'age_months_mean': float(b_age.mean()), 'age_months_min': float(b_age.min()),
                'age_months_max': float(b_age.max()), 'target_unknown': 0.431, 'other_unknown': 0.203, 'auc_unknown': 0.614}
    frame, cons, stat = pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame([training, *stats])
    COMPACT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPACT/'metrics.csv', index=False); cons.to_csv(COMPACT/'contrasts.csv', index=False); stat.to_csv(COMPACT/'age_and_unknown.csv', index=False)
    get = lambda a, m, g: float(frame.loc[(frame.arm == a) & (frame.method == m) & (frame.group == g), 'ndcg10'].iloc[0])
    c = lambda label, m, g: cons.loc[(cons.contrast == label) & (cons.method == m) & (cons.group == g)].iloc[0]
    fmt = lambda d: f'{d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}]'
    lines = ['## Results (computed after the hypotheses above)', '', '### Run 1: T0 -> T1 column swap, nDCG@10 (all / new / warm)', '',
             '| Arm | ' + ' | '.join(F.METHODS) + ' |', '|---|' + '---|'*len(F.METHODS)]
    for arm in ['T0', *SWAPS, 'T1']:
        lines.append(f'| {arm} | ' + ' | '.join(f'{get(arm, m, "all"):.6f} / {get(arm, m, "new"):.6f} / {get(arm, m, "warm"):.6f}' for m in F.METHODS) + ' |')
    lines += ['', 'Share of the T0 -> T1 gain recovered by each swap (arm − T0) / (T1 − T0):', '', '| Arm | ' + ' | '.join(f'{m} all / new' for m in F.METHODS) + ' |', '|---|' + '---|'*len(F.METHODS)]
    for arm in SWAPS:
        cells = []
        for m in F.METHODS:
            share = [(get(arm, m, g)-get('T0', m, g))/(get('T1', m, g)-get('T0', m, g)) for g in ('all', 'new')]
            cells.append(f'{share[0]:.0%} / {share[1]:.0%}')
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Arm − T0, nDCG@10 all [95% CI]:', '', '| Arm | ' + ' | '.join(F.METHODS) + ' |', '|---|' + '---|'*len(F.METHODS)]
    for arm in [*SWAPS, 'T1']:
        lines.append(f'| {arm} | ' + ' | '.join(fmt(c(f'{arm} - T0', m, 'all')) for m in F.METHODS) + ' |')
    lines += ['', '### Run 2a: fixed cut date, nDCG@10 (all / new / warm)', '', '| Cut date | ' + ' | '.join(F.METHODS) + ' |', '|---|' + '---|'*len(F.METHODS)]
    for arm in curve:
        lines.append(f'| {arm} | ' + ' | '.join(f'{get(arm, m, "all"):.6f} / {get(arm, m, "new"):.6f} / {get(arm, m, "warm"):.6f}' for m in F.METHODS) + ' |')
    lines += ['', '### Run 2b: fixed lag before each query month, nDCG@10 (all / new / warm)', '', '| Lag | ' + ' | '.join(F.METHODS) + ' |', '|---|' + '---|'*len(F.METHODS)]
    for arm in lags:
        lines.append(f'| {arm} | ' + ' | '.join(f'{get(arm, m, "all"):.6f} / {get(arm, m, "new"):.6f} / {get(arm, m, "warm"):.6f}' for m in F.METHODS) + ' |')
    lines += ['', 'Lag − monthly (lag 0), nDCG@10 new [95% CI]:', '', '| Lag | ' + ' | '.join(F.METHODS) + ' |', '|---|' + '---|'*len(F.METHODS)]
    for arm in lags[1:]:
        lines.append(f'| {arm} | ' + ' | '.join(fmt(c(f'{arm} - monthly', m, 'new')) for m in F.METHODS) + ' |')
    lines += ['', '### Age of statistics and the "no reviews yet" cue (Val natural-hit rows)', '',
              '| Arm | age mean (months) | age range | target unknown | other unknown | AUC of unknown |', '|---|---:|---|---:|---:|---:|']
    for r in stat.itertuples():
        lines.append(f'| {r.arm} | {r.age_months_mean:.1f} | {r.age_months_min:.1f}–{r.age_months_max:.1f} | {r.target_unknown:.3f} | {r.other_unknown:.3f} | {r.auc_unknown:.3f} |')
    lines += ['', 'Intervals: 1,000 paired-user draws (seed 0) on seed-mean event metrics, unadjusted. No Test file was opened.']
    readme = COMPACT/'README.md'
    text = readme.read_text()
    if '## Results' in text: raise ValueError('results already written')
    readme.write_text(text + '\n' + '\n'.join(lines) + '\n')
    (COMPACT/'git_hash.txt').write_text(git_hash()+'\n')
    (COMPACT/'config.yaml').write_text(yaml.safe_dump({'label': 'Val only, exploratory', 'models': list(F.METHODS), 'seeds': list(F.SEEDS),
        'swaps': {k: list(map(str, v)) for k, v in SWAPS.items()}, 'dates_months_before_T1': list(DATES), 'lags_months': list(LAGS),
        'scaler': 'parent T0 prefix scaler', 'rows': 'Val natural-hit rows', 'test_opened': False}, sort_keys=False))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('features', 'score', 'analyse'))
    args = parser.parse_args()
    {'features': phase_features, 'score': phase_score, 'analyse': phase_analyse}[args.phase]()


if __name__ == '__main__':
    main()
