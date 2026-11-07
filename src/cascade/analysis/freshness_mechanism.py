"""Does the frozen ranker read candidate knownness as a newness cue? (Val only, exploratory; diagnostic.)

Mixed-snapshot arms are made from the saved standardised T1 and monthly features of
:mod:`cascade.analysis.freshness_val`. Every column comes from a snapshot that ends before every query using it,
so all arms are leak-free. They diagnose what the frozen models use and are not a serving recipe.

* ``H1``: monthly for every column except common[1] ``candidate_graph_known`` (from T1).
* ``H2``: as H1, also common[0] ``candidate_log_degree`` from T1.
* ``H3``: T1 for every column except ``candidate_graph_known`` (from monthly).

Also reported: in the T0-era training lists (B natural-hit rows, T0 snapshot), how often positives and negatives
are unknown; and a frozen-model probe, which flips only ``candidate_graph_known`` 0->1 in standardised units on
10,000 fixed Val hit rows. The head is per-candidate, so flipping all unknown slots at once equals flipping each.

    PYTHONPATH=src OMP_NUM_THREADS=4 python -m cascade.analysis.freshness_mechanism --phase {score,training,probe,analyse}
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from cascade.analysis import freshness_val as F
from cascade.pipelines.local_evidence_export import ranks_and_top10

LABEL = 'Val only, exploratory; mixed snapshots are a diagnostic of frozen-model use, not a serving recipe'
STUDY = 'results/local_experiments/20261106-freshness-mechanism'
COMPACT = 'results/analysis/20261106-freshness-mechanism'
HYBRIDS = ('H1', 'H2', 'H3')
KNOWN, DEGREE = 1, 0


def hybrid(arm: str, t1: dict, monthly: dict) -> dict:
    """Column-swapped feature dict; the inputs are not modified."""
    base, other, cols = {'H1': (monthly, t1, [KNOWN]), 'H2': (monthly, t1, [DEGREE, KNOWN]),
                         'H3': (t1, monthly, [KNOWN])}[arm]
    common = np.array(base['common'], copy=True)
    common[:, :, cols] = other['common'][:, :, cols]
    return {'common': common, 'graph': np.array(base['graph'], copy=True)}


def binary_auc(p_pos: float, p_neg: float) -> float:
    """AUC of a 0/1 score with P(score=1|pos)=p_pos and P(score=1|neg)=p_neg (ties count half)."""
    return p_pos*(1-p_neg) + 0.5*(p_pos*p_neg + (1-p_pos)*(1-p_neg))


def unknown_stats(cand: np.ndarray, target: np.ndarray, unknown: np.ndarray) -> dict:
    """Pooled and per-list summaries of the unknown indicator for positives vs non-target candidates."""
    pos = cand == target[:, None]
    neg = (cand > 0) & ~pos
    if not (pos.sum(1) == 1).all(): raise ValueError('each list needs exactly one positive')
    p1, p0 = float(unknown[pos].mean()), float(unknown[neg].mean())
    u_pos = unknown[pos]
    neg_unknown = (unknown & neg).sum(1); neg_count = neg.sum(1)
    q = neg_unknown/np.maximum(neg_count, 1)
    per_list = np.where(u_pos, (1-q) + 0.5*q, 0.5*(1-q))  # P(pos unk > neg unk) + 0.5 tie
    return {'lists': int(len(cand)), 'share_positives_unknown': p1, 'share_negatives_unknown': p0,
            'target_rate_among_unknown_slots': float((unknown & pos).sum()/max((unknown & (pos | neg)).sum(), 1)),
            'target_rate_among_known_slots': float((~unknown & pos).sum()/max((~unknown & (pos | neg)).sum(), 1)),
            'pooled_auc_unknown_as_score': binary_auc(p1, p0), 'mean_within_list_auc_unknown_as_score': float(per_list.mean())}


def _raw_known(z, scaler):
    raw = z*scaler['common']['std'][KNOWN] + scaler['common']['mean'][KNOWN]
    if not np.allclose(raw, np.round(raw), atol=1e-4): raise ValueError('knownness is not binary after unscaling')
    return np.round(raw).astype(bool)


def _arms():
    out = F._root()/F.STUDY
    return {arm: {k: np.load(out/f'features/{arm}/{k}.npy') for k in ('common', 'graph')} for arm in ('T1', 'monthly')}


def phase_score():
    P, meta, data, inst, archive, hit = F._context()
    out = F._root()/STUDY; out.mkdir(parents=True, exist_ok=True)
    sub = F._sub_archive(archive, hit)
    base = _arms()
    # Determinism control: rescoring the saved monthly arm must reproduce its saved ranks.
    ranks, _ = F._score(P, meta, 'pairwise_coreview', 0, sub, base['monthly'])
    with np.load(F._root()/F.STUDY/'ranks/monthly/pairwise_coreview/seed0.npz') as z:
        control = bool(np.array_equal(z['rank'][hit], ranks))
    if not control: raise ValueError('monthly replay differs')
    checks = {'monthly_replay_exact': control}
    for arm in HYBRIDS:
        feats = hybrid(arm, base['T1'], base['monthly'])
        src, oth = (base['monthly'], base['T1']) if arm != 'H3' else (base['T1'], base['monthly'])
        swapped = [KNOWN] if arm != 'H2' else [DEGREE, KNOWN]
        kept = [c for c in range(9) if c not in swapped]
        if not (np.array_equal(feats['common'][:, :, kept], src['common'][:, :, kept]) and
                np.array_equal(feats['common'][:, :, swapped], oth['common'][:, :, swapped]) and
                np.array_equal(feats['graph'], src['graph'])):
            raise ValueError('hybrid construction differs from its definition')
        checks[arm] = {'base': 'monthly' if arm != 'H3' else 'T1', 'swapped_common_columns': swapped,
                       'columns_swapped_from': 'T1' if arm != 'H3' else 'monthly',
                       'slots_changed': int((feats['common'] != src['common']).any(-1).sum())}
        for method in F.METHODS:
            for seed in F.SEEDS:
                dest = out/f'ranks/{arm}/{method}/seed{seed}.npz'
                if dest.exists(): continue
                started = time.perf_counter()
                ranks, _ = F._score(P, meta, method, seed, sub, feats)
                full = np.full(len(inst), np.inf); full[hit] = ranks
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, inst=archive['inst'], user=archive['user'], rank=full)
                print(json.dumps({'scored': arm, 'method': method, 'seed': seed, 'seconds': time.perf_counter()-started}), flush=True)
        del feats
    F._write_json(out/'score_checks.json', checks)


def phase_training():
    P, meta, data, inst, archive, hit = F._context()
    out = F._root()/STUDY; out.mkdir(parents=True, exist_ok=True)
    parent = F._root()/F.PARENT
    scaler = P.read(parent/'features/manifest.json')['scaler']
    train = P.arrays(parent/'candidates/ranker_train')
    common = np.load(parent/'features/ranker_train/common.npy', mmap_mode='r')
    prefix = np.load(parent/'split/prefix.npy')
    found = np.isfinite(train['retrieval_rank'])
    first = np.zeros(data.n_items+1, np.int64); first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    rows = []
    for name, mask in (('B_prefix_natural (inner fit)', prefix & found), ('B_all_natural (refit)', found)):
        idx = np.flatnonzero(mask)
        cand = np.asarray(train['cand'][idx]); target = np.asarray(train['target'][idx])
        unknown = ~_raw_known(np.asarray(common[idx, :, KNOWN]), scaler) & (cand > 0)
        if not np.array_equal(unknown[cand == target[:, None]], first[target] >= F.T0):
            raise ValueError('T0 knownness of B positives disagrees with first review time')
        rows.append({'population': name, 'snapshot': 'T0', **unknown_stats(cand, target, unknown)})
    b_prefix = np.flatnonzero(prefix)
    rows.append({'population': 'B_prefix_all_events (incl. retrieval misses)', 'snapshot': 'T0',
                 'lists': int(len(b_prefix)),
                 'share_positives_unknown': float((first[np.asarray(train['target'][b_prefix])] >= F.T0).mean())})
    # Val comparison under T1 (natural-hit rows), same statistic.
    cand = np.asarray(archive['cand'][hit]); target = np.asarray(archive['target'][hit])
    t1 = np.load(F._root()/F.STUDY/'features/T1/common.npy', mmap_mode='r')
    monthly = np.load(F._root()/F.STUDY/'features/monthly/common.npy', mmap_mode='r')
    old = np.load(parent/'features/val/common.npy', mmap_mode='r')
    for arm, z in (('T0', np.asarray(old[hit, :, KNOWN])), ('T1', np.asarray(t1[:, :, KNOWN])), ('monthly', np.asarray(monthly[:, :, KNOWN]))):
        unknown = ~_raw_known(z, scaler) & (cand > 0)
        rows.append({'population': 'Val_natural_hits', 'snapshot': arm, **unknown_stats(cand, target, unknown)})
    pd.DataFrame(rows).to_csv(out/'training_knownness.csv', index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)


def phase_probe():
    P, meta, data, inst, archive, hit = F._context()
    out = F._root()/STUDY; out.mkdir(parents=True, exist_ok=True)
    scaler = P.read(F._root()/F.PARENT/'features/manifest.json')['scaler']
    pick = np.sort(np.random.default_rng(20261106).choice(len(hit), 10000, replace=False))
    sub = {k: v[pick] for k, v in F._sub_archive(archive, hit).items()}
    base = {k: np.load(F._root()/F.STUDY/f'features/T1/{k}.npy', mmap_mode='r')[pick] for k in ('common', 'graph')}
    valid = sub['cand'] > 0
    unknown = ~_raw_known(base['common'][:, :, KNOWN], scaler) & valid
    flipped = {'common': base['common'].copy(), 'graph': base['graph']}
    mean, std = scaler['common']['mean'][KNOWN], scaler['common']['std'][KNOWN]
    flipped['common'][:, :, KNOWN] = np.where(unknown, np.float32((1-mean)/std), base['common'][:, :, KNOWN])
    first = data.items.set_index('item').first_ts.loc[sub['target']].to_numpy()
    new_row = first >= F.T1
    is_target = sub['cand'] == sub['target'][:, None]
    rows = []
    for method in F.METHODS:
        for seed in F.SEEDS:
            _, a = F._score(P, meta, method, seed, sub, base)
            _, b = F._score(P, meta, method, seed, sub, flipped)
            delta = (b - a)
            if not np.allclose(delta[valid & ~unknown], 0, atol=1e-5):
                raise ValueError('probe changed a known candidate (head is not per-candidate)')
            spread = np.nanmean(np.where(valid, a, np.nan).std(axis=1, where=valid))
            ranks_base, _ = ranks_and_top10(a, sub['cand'], sub['target'])
            ranks_flip, _ = ranks_and_top10(b, sub['cand'], sub['target'])
            tgt_unknown = (unknown & is_target).any(1)
            for slot, m in (('all_unknown_slots', unknown), ('unknown_non_target', unknown & ~is_target),
                            ('unknown_target_new_row', unknown & is_target & new_row[:, None])):
                rows.append({'method': method, 'seed': seed, 'slots': slot, 'count': int(m.sum()),
                             'mean_delta_logit': float(delta[m].mean()) if m.any() else np.nan,
                             'median_delta_logit': float(np.median(delta[m])) if m.any() else np.nan,
                             'mean_within_list_logit_std': float(spread),
                             'flip_z_from': float(-mean/std), 'flip_z_to': float((1-mean)/std)})
            rows.append({'method': method, 'seed': seed, 'slots': 'unknown_targets_rank_change',
                         'count': int(tgt_unknown.sum()),
                         'mean_rank_base': float(ranks_base[tgt_unknown].mean()),
                         'mean_rank_flipped': float(ranks_flip[tgt_unknown].mean()),
                         'top10_base': float((ranks_base[tgt_unknown] <= 10).mean()),
                         'top10_flipped': float((ranks_flip[tgt_unknown] <= 10).mean())})
            print(json.dumps({'probe': method, 'seed': seed, 'mean_delta_all': rows[-4]['mean_delta_logit']}), flush=True)
    pd.DataFrame(rows).to_csv(out/'knownness_probe.csv', index=False)


def phase_analyse():
    import yaml
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.saved_results import average
    from cascade.utils.io import git_hash
    P, meta, data, inst, archive, hit = F._context()
    out = F._root()/STUDY; compact = F._root()/COMPACT; compact.mkdir(parents=True, exist_ok=True)
    users = inst.user.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= F.T1
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    paths = {arm: F._root()/F.STUDY/'ranks'/arm for arm in ('T1', 'monthly')}
    paths.update({arm: out/'ranks'/arm for arm in HYBRIDS})
    values, rows = {}, []
    for arm, path in paths.items():
        for method in F.METHODS:
            per_seed = []
            for seed in F.SEEDS:
                with np.load(path/f'{method}/seed{seed}.npz') as z:
                    if not np.array_equal(z['inst'], inst.inst): raise ValueError('rank alignment')
                    per_seed.append(F._metric_values(z['rank']))
            values[arm, method] = {m: np.mean([s[m] for s in per_seed], axis=0) for m in per_seed[0]}
            for group, mask in groups.items():
                for metric in per_seed[0]:
                    seeds = [average(s[metric][mask], users[mask]) for s in per_seed]
                    rows.append({'arm': arm, 'method': method, 'group': group, 'metric': metric,
                                 'value': float(np.mean(seeds)), 'seed_std_ddof0': float(np.std(seeds)),
                                 **{f'seed{s}': v for s, v in zip(F.SEEDS, seeds)}})
    pairs = [(m, ((ref, m), (cand, m))) for m in F.METHODS for cand, ref in
             (('H1', 'T1'), ('H1', 'monthly'), ('H2', 'T1'), ('H2', 'monthly'), ('H3', 'T1'))]
    for obj, text, co in (('bce', 'bce_text', 'bce_coreview'), ('pairwise', 'pairwise_text', 'pairwise_coreview')):
        pairs += [(obj, ((arm, text), (arm, co))) for arm in ('H1', 'H2')]
    contrasts = []
    for family, (ref, cand) in pairs:
        for group, mask in groups.items():
            for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
                ci = paired_bootstrap(users[mask], values[ref][metric][mask], values[cand][metric][mask], n_resamples=1000, seed=0)
                contrasts.append({'family': family, 'reference': '/'.join(ref), 'candidate': '/'.join(cand),
                                  'group': group, 'metric': metric, 'events': int(mask.sum()), **ci})
    pd.DataFrame(rows).to_csv(compact/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(compact/'contrasts.csv', index=False)
    for name in ('training_knownness.csv', 'knownness_probe.csv', 'score_checks.json'):
        (compact/name).write_text((out/name).read_text())
    (compact/'git_hash.txt').write_text(git_hash()+'\n')
    (compact/'config.yaml').write_text(yaml.safe_dump({
        'label': LABEL, 'study': STUDY, 'feature_source': F.STUDY, 'methods': list(F.METHODS), 'seeds': list(F.SEEDS),
        'arms': {'H1': 'monthly except candidate_graph_known from T1',
                 'H2': 'monthly except candidate_graph_known and candidate_log_degree from T1',
                 'H3': 'T1 except candidate_graph_known from monthly'},
        'leak_free': 'every column from a snapshot ending before each of its queries (T1 or query month start)',
        'probe': {'rows': 10000, 'row_seed': 20261106, 'base': 'T1 features', 'flip': 'candidate_graph_known 0->1 standardised'},
        'metrics': ['ndcg10', 'recall10', 'hr3', 'mrr10'], 'bootstrap': {'resamples': 1000, 'seed': 0, 'unit': 'user'},
        'device': 'cpu', 'training_performed': False, 'test_opened': False, 'selection': 'none'}, sort_keys=False))
    print(json.dumps({'analysed': str(compact)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('score', 'training', 'probe', 'analyse'))
    args = parser.parse_args(); torch.set_num_threads(4)
    {'score': phase_score, 'training': phase_training, 'probe': phase_probe, 'analyse': phase_analyse}[args.phase]()


if __name__ == '__main__':
    main()
