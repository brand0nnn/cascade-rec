"""Test replication of the statistics-timing diagnostics (addendum ``docs/test_protocol_2026-11-07_addendum_timing.md``).

Same frozen B1/B4/B5/B6 checkpoints, builder, cut rule, parent T0 scaler and scorer as the Val study
(``statistics_timing``), on the already-built Test P1 natural-hit rows.  T2 (Test start) plays the role of T1.
No training; nothing is selected; each arm is scored once.

    PYTHONPATH=src python -m cascade.analysis.statistics_timing_test --phase {features,score,analyse}
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from cascade.analysis import freshness_val as F
from cascade.analysis import statistics_timing as ST
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import test_final as TF
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

ADDENDUM = REPO_ROOT/'docs/test_protocol_2026-11-07_addendum_timing.md'
ADDENDUM_SHA = 'cd8228c4ea766fdb767e9081476ae58d979f47d0d6ede2b73089a49237a301a9'
BULK = TF.BULK/'timing'
OUT = TF.OUT/'timing'
DATES = (3, 6, 9, 12, 18)
LAGS = (0, 1, 3, 6, 9, 12)


def context():
    if digest(ADDENDUM) != ADDENDUM_SHA: raise ValueError('the frozen addendum changed')
    meta = P.verify(TF.PARENT, artifacts=False)
    data = P.data_for(meta)
    inst = data.split_instances('test')
    archive = P.arrays(TF.BULK/'candidates/P1/test')
    if not np.array_equal(archive['inst'], inst.inst): raise ValueError('Test alignment')
    hit = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    for arm in ('P1_parent_T0', 'P1_parent_T2'):
        if not np.array_equal(np.load(TF.BULK/f'features/{arm}/hit.npy'), hit): raise ValueError('hit rows differ')
    return meta, data, inst, archive, hit


def scaler():
    return P.read(TF.PARENT/'features/manifest.json')['scaler']


def phase_features():
    torch.set_num_threads(3)
    _, data, inst, archive, hit = context()
    frame = inst.iloc[hit]; cand = np.asarray(archive['cand'][hit]); ts = frame.ts.to_numpy(np.int64)
    BULK.mkdir(parents=True, exist_ok=False); audits = {}
    arms = {f'date_T2-{k}m': np.full(len(hit), ST.shift_months([TF.T2], k)[0], np.int64) for k in DATES}
    arms.update({f'lag{lag}': ST.lag_cutoffs(ts, lag) for lag in LAGS})
    for name, cuts in arms.items():
        if (cuts > ts).any() or (cuts < F.T0).any(): raise ValueError('cutoff outside [T0, query)')
        feats, audits[name] = TF.build_features(data, frame, cand, cuts, scaler(), name)
        directory = BULK/f'features/{name}'; directory.mkdir(parents=True)
        for key, value in feats.items(): np.save(directory/f'{key}.npy', value)
    write_json(BULK/'snapshots.json', audits)


def all_features(data, archive, hit):
    t0 = {k: np.load(TF.BULK/f'features/P1_parent_T0/{k}.npy') for k in ('common', 'graph')}
    t2 = {k: np.load(TF.BULK/f'features/P1_parent_T2/{k}.npy') for k in ('common', 'graph')}
    first = np.full(data.n_items+1, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    cand_first = first[np.asarray(archive['cand'][hit])]
    product = {'old': cand_first < F.T0, 'B': (cand_first >= F.T0) & (cand_first < TF.T2)}
    out = {'T0': t0, 'T2': t2}
    out.update({name: ST.swap(t0, t2, block, cols, None if group is None else product[group])
                for name, (block, cols, group) in ST.SWAPS.items()})
    for name in [*[f'date_T2-{k}m' for k in DATES], *[f'lag{lag}' for lag in LAGS]]:
        out[name] = {k: np.load(BULK/f'features/{name}/{k}.npy') for k in ('common', 'graph')}
    return out


def phase_score():
    torch.set_num_threads(3)
    meta, data, inst, archive, hit = context()
    sub = F._sub_archive(archive, hit)
    for name, feats in all_features(data, archive, hit).items():
        for method in F.METHODS:
            for seed in F.SEEDS:
                dest = BULK/f'ranks/{name}/{method}/seed{seed}.npz'
                if dest.exists(): raise ValueError(f'{name} {method} seed {seed} already scored: each arm is scored once')
                started = time.perf_counter()
                ranks, _ = F._score(P, meta, method, seed, sub, feats)
                full = np.full(len(inst), np.inf); full[hit] = ranks
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, inst=np.asarray(archive['inst']), user=np.asarray(archive['user']), rank=full)
                print(json.dumps({'scored': name, 'method': method, 'seed': seed, 'seconds': round(time.perf_counter()-started, 1)}), flush=True)


def phase_analyse():
    import yaml
    _, data, inst, archive, hit = context()
    users = inst.user.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= TF.T2
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    curve = ['T0', *[f'date_T2-{k}m' for k in sorted(DATES, reverse=True)], 'T2']
    lags = [f'lag{lag}' for lag in LAGS]
    arms = ['T0', 'T2', *ST.SWAPS, *curve[1:-1], *lags]
    values, rows = {}, []
    for arm in arms:
        for method in F.METHODS:
            seeds = []
            for seed in F.SEEDS:
                with np.load(BULK/f'ranks/{arm}/{method}/seed{seed}.npz') as z:
                    if not np.array_equal(z['inst'], inst.inst): raise ValueError('rank alignment')
                    seeds.append(F._metric_values(z['rank'])['ndcg10'])
            values[arm, method] = np.mean(seeds, axis=0)
            for group, mask in groups.items():
                per = [average(s[mask], users[mask]) for s in seeds]
                rows.append({'arm': arm, 'method': method, 'group': group, 'ndcg10': float(np.mean(per)), 'seed_std_ddof0': float(np.std(per))})
    pairs = [(f'{a} - T0', 'T0', a) for a in [*ST.SWAPS, 'T2']] + [(f'{b} - {a}', a, b) for a, b in zip(curve[:-1], curve[1:])]
    pairs += [(f'{a} - lag0', 'lag0', a) for a in lags[1:]]
    contrasts = []
    for label, ref, cand in pairs:
        for method in F.METHODS:
            for group, mask in groups.items():
                ci = paired_bootstrap(users[mask], values[ref, method][mask], values[cand, method][mask], n_resamples=1000, seed=0)
                contrasts.append({'contrast': label, 'method': method, 'group': group, **ci})
        print(json.dumps({'contrast': label}), flush=True)
    sub = F._sub_archive(archive, hit); ts = inst.ts.to_numpy(np.int64)[hit]; stats = []
    for arm in [*curve, *lags]:
        path = TF.BULK/f'features/P1_parent_{arm}/common.npy' if arm in ('T0', 'T2') else BULK/f'features/{arm}/common.npy'
        if arm in ('T0', 'T2'): cut = np.full(len(hit), F.T0 if arm == 'T0' else TF.T2)
        elif arm.startswith('date'): cut = np.full(len(hit), ST.shift_months([TF.T2], int(arm.split('-')[1][:-1]))[0])
        else: cut = ST.lag_cutoffs(ts, int(arm[3:]))
        age = (ts - cut)/86400000/30.44
        stats.append({'arm': arm, 'age_months_mean': float(age.mean()), 'age_months_min': float(age.min()), 'age_months_max': float(age.max()),
                      **ST.unknown_stats(np.load(path), sub['cand'], sub['target'], scaler())})
    frame, cons, stat = pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame(stats)
    OUT.mkdir(parents=True, exist_ok=False)
    frame.to_csv(OUT/'metrics.csv', index=False); cons.to_csv(OUT/'contrasts.csv', index=False); stat.to_csv(OUT/'age_and_unknown.csv', index=False)
    get = lambda a, m, g: float(frame.loc[(frame.arm == a) & (frame.method == m) & (frame.group == g), 'ndcg10'].iloc[0])
    share = lambda a, m, g: (get(a, m, g)-get('T0', m, g))/(get('T2', m, g)-get('T0', m, g))
    verdicts = {
        'R1a known flag recovers >= 1/3 of the new-target gain (all models)': all(share('X1_known', m, 'new') >= 1/3 for m in F.METHODS),
        'R1b X7 recovers >= 75% overall and X6 < 1/3 (all models)': all(share('X7_products_released_in_B', m, 'all') >= .75 and share('X6_old_products', m, 'all') < 1/3 for m in F.METHODS),
        'R2a overall rises monotonically T0 -> T2 (all models)': all(all(get(b, m, 'all') > get(a, m, 'all') for a, b in zip(curve[:-1], curve[1:])) for m in F.METHODS),
        'R2b new lowest at lag 0, peak in {3,6,9}, lag 12 below peak (all models)': all(
            min(lags, key=lambda a: get(a, m, 'new')) == 'lag0' and max(lags, key=lambda a: get(a, m, 'new')) in ('lag3', 'lag6', 'lag9')
            and get('lag12', m, 'new') < max(get(a, m, 'new') for a in lags) for m in F.METHODS),
        'R2c warm higher at lag 0 than lag 12 (all models)': all(get('lag0', m, 'warm') > get('lag12', m, 'warm') for m in F.METHODS)}
    fmt3 = lambda a, m: f'{get(a, m, "all"):.6f} / {get(a, m, "new"):.6f} / {get(a, m, "warm"):.6f}'
    head = '| Arm | ' + ' | '.join(F.METHODS) + ' |'; sep = '|---|' + '---|'*len(F.METHODS)
    lines = ['# Test replication of the statistics-timing diagnostics (7 Nov 2026)', '',
             f'Addendum `docs/test_protocol_2026-11-07_addendum_timing.md` (sha256 {ADDENDUM_SHA}). Frozen T0-trained P1 rankers, '
             f'{len(hit):,} Test natural-hit rows of {len(inst):,} events; new = target first review ≥ T2. nDCG@10 per-user macro, seed means.', '',
             '## Predictions', '', '| Prediction | Verdict |', '|---|---|']
    lines += [f'| {k} | {"held" if v else "not held"} |' for k, v in verdicts.items()]
    lines += ['', '## Column swap T0 -> T2, nDCG@10 (all / new / warm)', '', head, sep]
    lines += [f'| {a} | ' + ' | '.join(fmt3(a, m) for m in F.METHODS) + ' |' for a in ['T0', *ST.SWAPS, 'T2']]
    lines += ['', 'Share of the T0 -> T2 gain recovered (all / new):', '', head, sep]
    lines += [f'| {a} | ' + ' | '.join(f'{share(a, m, "all"):.0%} / {share(a, m, "new"):.0%}' for m in F.METHODS) + ' |' for a in ST.SWAPS]
    lines += ['', '## Fixed cut date, nDCG@10 (all / new / warm)', '', head, sep]
    lines += [f'| {a} | ' + ' | '.join(fmt3(a, m) for m in F.METHODS) + ' |' for a in curve]
    lines += ['', '## Fixed lag before each query month, nDCG@10 (all / new / warm)', '', head, sep]
    lines += [f'| {a} | ' + ' | '.join(fmt3(a, m) for m in F.METHODS) + ' |' for a in lags]
    lines += ['', '## Age of statistics and the "no reviews yet" cue', '', '| Arm | age mean (months) | range | target unknown | other unknown | AUC |', '|---|---:|---|---:|---:|---:|']
    lines += [f'| {r.arm} | {r.age_months_mean:.1f} | {r.age_months_min:.1f}–{r.age_months_max:.1f} | {r.target_unknown:.3f} | {r.other_unknown:.3f} | {r.auc_unknown:.3f} |' for r in stat.itertuples()]
    lines += ['', 'Intervals for every contrast: `contrasts.csv` (1,000 paired-user draws, seed 0). Training reference (B natural lists, T0): age mean 6.1 months, target / other unknown 0.431 / 0.203.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'verdicts.json', verdicts)
    (OUT/'git_hash.txt').write_text(git_hash()+'\n')
    (OUT/'config.yaml').write_text(yaml.safe_dump({'addendum_sha256': ADDENDUM_SHA, 'models': list(F.METHODS), 'seeds': list(F.SEEDS),
        'dates_months_before_T2': list(DATES), 'lags_months': list(LAGS), 'test_predictions_loaded': True, 'training': 'none'}, sort_keys=False))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('features', 'score', 'analyse'))
    {'features': phase_features, 'score': phase_score, 'analyse': phase_analyse}[parser.parse_args().phase]()


if __name__ == '__main__':
    main()
