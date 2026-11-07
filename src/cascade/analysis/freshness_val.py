"""Graph/common feature freshness on Val (Val only, exploratory; selects nothing).

Question: the frozen residual rankers use nine common and six graph inputs built from the review graph
strictly before T0. Does rebuilding those inputs from a fresher snapshot help on Val, with correct
sequencing (every snapshot strictly precedes each of its queries)? Arms:

* ``T0``: the saved parent features (all reviews strictly before T0), rescored on CPU.
* ``T1``: all reviews strictly before T1 (Val start).
* ``monthly``: for a Val query, all reviews strictly before the start of its UTC calendar month.

Candidates, histories, checkpoints, prefix scalers (fitted on T0-period ranker-train features) and the
post-standardisation graph masks stay fixed. Only rows whose target is inside the frozen K200 list can
change a target rank; every natural retrieval miss keeps rank infinity, so only the 14,916 natural-hit
rows are re-featurised and rescored. Test is never opened: every snapshot is asserted to end before T2,
only ``split_instances('val')`` is used and no Test artifact is read.

Run from the repo root (CPU only):
    PYTHONPATH=src OMP_NUM_THREADS=4 python -m cascade.analysis.freshness_val --phase reproduce
    ... --phase features ; --phase score ; --phase analyse
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.data.local_graph_features import FrozenGraphFeatures

LABEL = 'Val only, exploratory'
STUDY = 'results/local_experiments/20261106-freshness-val'
COMPACT = 'results/analysis/20261106-freshness-val'
PARENT = 'results/local_experiments/20261103-full-promising-ranking'
FACTORIAL = 'results/local_experiments/20261104-objective-coreview'
METHODS = ('bce_text', 'pairwise_text', 'pairwise_coreview', 'bce_coreview')
ARMS = ('T0', 'T1', 'monthly')
SEEDS = (0, 1, 2)
T0, T1 = 1595800947930, 1628643414042


# ----------------------------------------------------------------------------------------------------------
# Snapshot logic (pure; unit-tested on toy data)
# ----------------------------------------------------------------------------------------------------------
def month_start_ms(ts: np.ndarray) -> np.ndarray:
    """UTC calendar-month start (ms) of each millisecond timestamp."""
    ts = np.asarray(ts, dtype=np.int64)
    return ts.astype('datetime64[ms]').astype('datetime64[M]').astype('datetime64[ms]').astype(np.int64)


class SnapshotFeatures(BatchedGraphFeatures):
    """Same common/graph features as the T0 builder, from all reviews strictly before ``cutoff``.

    The query-user exclusion is the T0 builder's algebra (inherited ``build``): the query user's own
    snapshot edges are subtracted from history degrees, and full-history-filtered candidates never carry a
    query-user edge, which :func:`assert_cut_rule` checks explicitly.
    """

    def __init__(self, data, cutoff: int):  # noqa: D107 - deliberately bypasses the T0-only parent init
        cutoff = int(cutoff)
        if cutoff > int(data.split['t2']):
            raise ValueError('snapshot would include Test-period reviews')
        self.data = data
        self.strict_before_ms = cutoff
        users = np.repeat(np.arange(len(data.user_ptr)-1), np.diff(data.user_ptr))
        keep = data.seq_ts < cutoff
        reviews = pd.DataFrame({'user': users[keep], 'item': data.seq_item[keep].astype(np.int64),
                                'ts': data.seq_ts[keep]})
        if len(reviews) and int(reviews.ts.max()) >= cutoff:
            raise ValueError('snapshot contains a review at or after its cutoff')
        if reviews.duplicated(['user', 'item']).any():
            raise ValueError('duplicate graph edge')
        self.matrix = sparse.csc_matrix((np.ones(len(reviews), np.float32), (reviews.user.to_numpy(), reviews.item.to_numpy())),
                                        shape=(data.n_users+1, data.n_items+1))
        self.by_user = self.matrix.tocsr()
        self.degree = np.asarray(self.matrix.sum(axis=0)).ravel().astype(np.float32)
        self.leaf, self.store = self._metadata(data)
        self.leaf_available = self._neighbour_available(self.leaf)
        self.store_available = self._neighbour_available(self.store)
        self.cooccurrence = (self.matrix.T @ self.matrix).tocsr()
        self.support = self.cooccurrence.copy(); self.support.data.fill(1)
        self.audit = {'strict_before_ms': cutoff, 'review_edges': int(self.matrix.nnz),
                      'max_review_timestamp': int(reviews.ts.max()) if len(reviews) else None,
                      'query_user_excluded': True, 'self_history_paths_excluded': True, 'labels_read': False,
                      'history_limit': 50}


def assert_cut_rule(builder, instances: pd.DataFrame, candidates: np.ndarray) -> dict:
    """No snapshot review at/after any query; the query user's snapshot edges are history-only.

    Raises if the snapshot's latest review is at/after a query timestamp, if any query user has a snapshot
    edge to one of its candidates or targets, or if a query user's snapshot item is absent from the user's
    strictly-earlier history (which would mean the exclusion algebra no longer removes it).
    """
    data = builder.data
    ts = instances['ts'].to_numpy(np.int64)
    latest = builder.audit['max_review_timestamp']
    if len(ts) and latest is not None and latest >= ts.min():
        raise ValueError('snapshot contains a review at or after a query')
    if len(ts) and builder.strict_before_ms > ts.min():
        raise ValueError('snapshot cutoff is after a query')
    users = instances['user'].to_numpy(np.int64)
    lengths = instances['hist_len'].to_numpy(np.int64)
    cand = np.asarray(candidates)
    rows = np.repeat(users, cand.shape[1]); cols = cand.ravel()
    if np.asarray(builder.by_user[rows, cols]).ravel()[cols > 0].any():
        raise ValueError('query user has a snapshot edge to a candidate')
    if 'item' in instances and np.asarray(builder.by_user[users, instances['item'].to_numpy(np.int64)]).ravel().any():
        raise ValueError('query user has a snapshot edge to the target')
    for user, length in zip(users, lengths):
        own = builder.by_user.indices[builder.by_user.indptr[user]:builder.by_user.indptr[user+1]]
        begin = data.user_ptr[user]
        if not np.isin(own, data.seq_item[begin:begin+length]).all():
            raise ValueError('query user snapshot edge is outside its strict history')
    return {'queries': int(len(ts)), 'snapshot_max_ts': latest, 'min_query_ts': int(ts.min()) if len(ts) else None}


# ----------------------------------------------------------------------------------------------------------
# Study plumbing (full data; Val only)
# ----------------------------------------------------------------------------------------------------------
def _root():
    from cascade.utils.config import REPO_ROOT
    return REPO_ROOT


def _context():
    from cascade.pipelines import full_data_ranking as P
    root = _root(); parent = root/PARENT
    meta = P.read(parent/'study.json')
    for name, expected in meta['source_sha256'].items():
        from cascade.eval.saved_results import digest
        if digest(root/name) != expected:
            raise ValueError(f'parent source changed: {name}')
    if meta['split']['t_ranker_train'] != T0 or meta['split']['t1'] != T1:
        raise ValueError('split constants differ from the brief')
    data = P.data_for(meta)
    inst = data.split_instances('val')
    archive = P.arrays(parent/'candidates/val')
    if not (np.array_equal(archive['inst'], inst.inst) and np.array_equal(archive['user'], inst.user)
            and np.array_equal(archive['target'], inst.item)):
        raise ValueError('Val alignment differs')
    if len(inst) != 232904 or inst.user.nunique() != 124416:
        raise ValueError('unexpected Val population')
    if (inst.ts.to_numpy() >= data.split['t2']).any() or (inst.ts.to_numpy() < data.split['t1']).any():
        raise ValueError('non-Val query')
    hit = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    return P, meta, data, inst, archive, hit


def _weights(method, seed):
    owner = FACTORIAL if method == 'bce_coreview' else PARENT
    return _root()/owner/f'runs/{method}/seed{seed}/model.pth'


def _saved_ranks(method, seed):
    path = (_root()/FACTORIAL/f'evaluation/seed{seed}/val_ranks.npz' if method == 'bce_coreview'
            else _root()/PARENT/f'evaluation/{method}/seed{seed}/val_ranks.npz')
    with np.load(path) as z:
        return z['inst'], z['rank']


def _mask(method):
    return 'pairwise_coreview' if method == 'bce_coreview' else method


def _sub_archive(archive, hit):
    return {k: np.asarray(archive[k][hit]) for k in ('cand', 'score', 'history', 'target', 'inst', 'user')}


def _score(P, meta, method, seed, sub, feats):
    import torch
    model = P.model_for(meta, _root()/PARENT, seed, 'cpu').eval()
    model.load_state_dict(torch.load(_weights(method, seed), map_location='cpu', weights_only=True), strict=True)
    before = P.state_hash(model)
    ranks, _, logits = P.predict(model, sub, feats, np.arange(len(sub['cand'])), _mask(method), 'cpu')
    if P.state_hash(model) != before: raise ValueError('scoring modified weights')
    del model; gc.collect()
    return ranks, logits


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float)+'\n')


def _standardise(values, scaler, key):
    return ((values.astype(np.float64)-np.asarray(scaler[key]['mean']))/np.asarray(scaler[key]['std'])).astype(np.float32)


def phase_reproduce():
    P, meta, data, inst, archive, hit = _context()
    out = _root()/STUDY; out.mkdir(parents=True, exist_ok=True)
    if (out/'reproduce.json').exists(): raise ValueError('reproduction already recorded')
    old = P.arrays(_root()/PARENT/'features/val')
    sub = _sub_archive(archive, hit)
    feats = {k: np.asarray(old[k][hit]) for k in ('common', 'graph')}
    started = time.perf_counter()
    ranks, _ = _score(P, meta, 'bce_text', 0, sub, feats)
    full = np.full(len(inst), np.inf); full[hit] = ranks
    saved_inst, saved = _saved_ranks('bce_text', 0)
    if not np.array_equal(saved_inst, archive['inst']): raise ValueError('saved rank alignment')
    exact = bool(np.array_equal(full, saved))
    # Own builder at the T0 cutoff must reproduce the saved T0 features (raw -> prefix scaler).
    scaler = P.read(_root()/PARENT/'features/manifest.json')['scaler']
    rows = np.sort(np.random.default_rng(20261106).choice(len(hit), 256, replace=False))
    builder = SnapshotFeatures(data, T0)
    edges_match = builder.audit['review_edges'] == P.read(_root()/PARENT/'features/manifest.json')['audit']['review_edges']
    raw = builder.build(inst.iloc[hit[rows]], sub['cand'][rows])
    feat_err = {k: float(np.max(np.abs(_standardise(v, scaler, k)-feats[k][rows]))) for k, v in zip(('common', 'graph'), raw)}
    record = {'label': LABEL, 'checkpoint': 'bce_text/seed0', 'device': 'cpu', 'events': len(inst),
              'natural_hit_rows_rescored': int(len(hit)), 'ranks_equal_saved_exactly': exact,
              'rank_mismatches': int((full != saved).sum()), 'seconds': time.perf_counter()-started,
              'own_T0_builder_edges_match_parent': bool(edges_match), 'own_T0_builder_rows_checked': int(len(rows)),
              'own_T0_builder_max_abs_error_standardised': feat_err}
    _write_json(out/'reproduce.json', record)
    print(json.dumps(record), flush=True)
    if not exact or max(feat_err.values()) > 2e-5 or not edges_match:
        raise SystemExit('reproduction failed: stop')


def _build_arm(P, data, inst, sub, hit, cutoffs, scaler, out, arm):
    """Features for hit rows grouped by cutoff, with cut-rule and reference checks per snapshot."""
    common = np.zeros((len(hit), 200, 9), np.float32); graph = np.zeros((len(hit), 200, 6), np.float32)
    support = np.zeros(len(inst), np.float32)  # target snapshot degree (query user cannot be a reviewer)
    audits = []
    hit_cut = cutoffs[hit]
    for cutoff in np.unique(cutoffs):
        started = time.perf_counter()
        builder = SnapshotFeatures(data, int(cutoff))
        events = np.flatnonzero(cutoffs == cutoff)
        rows = np.flatnonzero(hit_cut == cutoff)
        frame = inst.iloc[hit[rows]]
        cut = assert_cut_rule(builder, frame, sub['cand'][rows])
        everyone = inst.iloc[events]
        if builder.audit['max_review_timestamp'] >= everyone.ts.min(): raise ValueError('snapshot overlaps a query')
        if np.asarray(builder.by_user[everyone.user.to_numpy(), everyone.item.to_numpy()]).ravel().any():
            raise ValueError('query user reviewed the target inside the snapshot')
        support[events] = builder.degree[everyone.item.to_numpy()]
        check = rows[np.sort(np.random.default_rng(int(cutoff) % 2**32).choice(len(rows), min(24, len(rows)), replace=False))]
        fast = builder.build(inst.iloc[hit[check]], sub['cand'][check])
        ref = FrozenGraphFeatures.build(builder, inst.iloc[hit[check]], sub['cand'][check])
        err = [float(np.max(np.abs(a-b))) for a, b in zip(fast, ref)]
        if max(err) > 2e-5: raise ValueError('batched snapshot differs from the explicit-exclusion reference')
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            values = builder.build(inst.iloc[hit[part]], sub['cand'][part])
            common[part] = _standardise(values[0], scaler, 'common')
            graph[part] = _standardise(values[1], scaler, 'graph')
        audits.append({**builder.audit, **cut, 'cutoff_utc': str(np.datetime64(int(cutoff), 'ms')),
                       'val_events': int(len(events)), 'hit_rows': int(len(rows)),
                       'reference_rows': int(len(check)), 'reference_max_abs_error': err,
                       'seconds': time.perf_counter()-started})
        print(json.dumps({'arm': arm, **{k: audits[-1][k] for k in ('cutoff_utc', 'review_edges', 'val_events', 'hit_rows', 'seconds')}}), flush=True)
        del builder; gc.collect()
    directory = out/'features'/arm; directory.mkdir(parents=True, exist_ok=False)
    np.save(directory/'common.npy', common); np.save(directory/'graph.npy', graph)
    np.save(directory/'target_degree.npy', support)
    return audits


def phase_features():
    P, meta, data, inst, archive, hit = _context()
    out = _root()/STUDY
    if not P.read(out/'reproduce.json')['ranks_equal_saved_exactly']: raise ValueError('reproduction must pass first')
    scaler = P.read(_root()/PARENT/'features/manifest.json')['scaler']
    sub = _sub_archive(archive, hit)
    ts = inst.ts.to_numpy(np.int64)
    record = {'label': LABEL, 'hit_rows': int(len(hit))}
    t0 = SnapshotFeatures(data, T0)
    (out/'features/T0').mkdir(parents=True, exist_ok=False)
    np.save(out/'features/T0/target_degree.npy', t0.degree[inst.item.to_numpy()].astype(np.float32))
    record['T0'] = [t0.audit]; del t0; gc.collect()
    record['T1'] = _build_arm(P, data, inst, sub, hit, np.full(len(inst), T1, np.int64), scaler, out, 'T1')
    monthly = month_start_ms(ts)
    if (monthly > ts).any(): raise ValueError('month start after query')
    record['monthly'] = _build_arm(P, data, inst, sub, hit, monthly, scaler, out, 'monthly')
    record['monthly_note'] = 'August 2021 queries use 1 Aug 2021, which precedes T1 (11 Aug 2021)'
    old = P.arrays(_root()/PARENT/'features/val')
    for arm in ('T1', 'monthly'):
        new = np.load(out/f'features/{arm}/common.npy', mmap_mode='r')
        record[f'{arm}_nongraph_common_unchanged'] = bool(np.array_equal(new[:, :, 3:7], np.asarray(old['common'][hit])[:, :, 3:7]))
    _write_json(out/'snapshots.json', record)


def phase_score():
    P, meta, data, inst, archive, hit = _context()
    out = _root()/STUDY
    sub = _sub_archive(archive, hit)
    old = P.arrays(_root()/PARENT/'features/val')
    arms = {'T0': {k: np.asarray(old[k][hit]) for k in ('common', 'graph')}}
    for arm in ('T1', 'monthly'):
        arms[arm] = {k: np.load(out/f'features/{arm}/{k}.npy') for k in ('common', 'graph')}
    controls = []
    for method in METHODS:
        for seed in SEEDS:
            for arm, feats in arms.items():
                dest = out/f'ranks/{arm}/{method}/seed{seed}.npz'
                if dest.exists(): continue
                started = time.perf_counter()
                ranks, _ = _score(P, meta, method, seed, sub, feats)
                full = np.full(len(inst), np.inf); full[hit] = ranks
                if arm == 'T0':
                    saved_inst, saved = _saved_ranks(method, seed)
                    if not np.array_equal(saved_inst, archive['inst']): raise ValueError('saved alignment')
                    controls.append({'method': method, 'seed': seed, 'exact': bool(np.array_equal(full, saved)),
                                     'mismatches': int((full != saved).sum())})
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, inst=archive['inst'], user=archive['user'], rank=full)
                print(json.dumps({'scored': arm, 'method': method, 'seed': seed, 'seconds': time.perf_counter()-started}), flush=True)
    if controls: _write_json(out/'t0_controls.json', controls)


def _metric_values(rank):
    from cascade.eval import metrics as M
    mrr = np.where(rank <= 10, 1.0/rank, 0.0)
    return {'ndcg10': M.ndcg_at(rank, 10), 'recall10': M.recall_at(rank, 10), 'hr3': M.recall_at(rank, 3), 'mrr10': mrr}


def phase_analyse():
    import yaml
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.saved_results import average
    from cascade.utils.io import git_hash
    P, meta, data, inst, archive, hit = _context()
    out = _root()/STUDY; compact = _root()/COMPACT
    users = inst.user.to_numpy()
    first = data.items.set_index('item').first_ts.loc[inst.item].to_numpy()
    new = first >= T1
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    values = {}  # (arm, method) -> metric -> seed-mean event values
    seed_rows = []
    for arm in ARMS:
        for method in METHODS:
            per_seed = []
            for seed in SEEDS:
                with np.load(out/f'ranks/{arm}/{method}/seed{seed}.npz') as z:
                    if not np.array_equal(z['inst'], inst.inst): raise ValueError('rank alignment')
                    per_seed.append(_metric_values(z['rank']))
            values[arm, method] = {m: np.mean([s[m] for s in per_seed], axis=0) for m in per_seed[0]}
            for group, mask in groups.items():
                for metric in per_seed[0]:
                    seeds = [average(s[metric][mask], users[mask]) for s in per_seed]
                    seed_rows.append({'arm': arm, 'method': method, 'group': group, 'metric': metric,
                                      'events': int(mask.sum()), 'users': int(np.unique(users[mask]).size),
                                      'value': float(np.mean(seeds)), 'seed_std_ddof0': float(np.std(seeds)),
                                      **{f'seed{s}': v for s, v in zip(SEEDS, seeds)}})
    contrasts = []
    pairs = [(m, (('T0', m), ('T1', m))) for m in METHODS] + [(m, (('T1', m), ('monthly', m))) for m in METHODS]
    pairs += [(m, (('T0', m), ('monthly', m))) for m in METHODS]
    for obj, text, co in (('bce', 'bce_text', 'bce_coreview'), ('pairwise', 'pairwise_text', 'pairwise_coreview')):
        pairs += [(obj, ((arm, text), (arm, co))) for arm in ARMS]
    for family, (ref, cand) in pairs:
        for group, mask in groups.items():
            for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
                ci = paired_bootstrap(users[mask], values[ref][metric][mask], values[cand][metric][mask], n_resamples=1000, seed=0)
                contrasts.append({'family': family, 'reference': '/'.join(ref), 'candidate': '/'.join(cand),
                                  'group': group, 'metric': metric, 'events': int(mask.sum()), **ci})
        print(json.dumps({'contrast': '/'.join(cand)+' - '+'/'.join(ref)}), flush=True)
    # Feature shifts on natural-hit rows (only rows whose target rank can change).
    old = P.arrays(_root()/PARENT/'features/val')
    cand = np.asarray(archive['cand'][hit]); target = np.asarray(archive['target'][hit])
    is_target = cand == target[:, None]; valid = cand > 0
    hit_new = new[hit]
    names = {'common': list(FrozenGraphFeatures.common_names), 'graph': list(FrozenGraphFeatures.graph_names)}
    shift_rows = []
    for key in ('common', 'graph'):
        base = np.asarray(old[key][hit])
        for arm in ('T1', 'monthly'):
            fresh = np.load(out/f'features/{arm}/{key}.npy', mmap_mode='r')
            delta = np.asarray(fresh) - base
            for slot, smask in (('target', is_target), ('non_target', valid & ~is_target)):
                for group, gmask in (('all', np.ones(len(hit), bool)), ('new', hit_new), ('warm', ~hit_new)):
                    m = smask & gmask[:, None]
                    for i, name in enumerate(names[key]):
                        shift_rows.append({'arm': arm, 'block': key, 'feature': name, 'slot': slot, 'group': group,
                                           'slots': int(m.sum()), 'mean_z_T0': float(base[..., i][m].mean()),
                                           'mean_z_arm': float(fresh[..., i][m].mean()),
                                           'mean_shift_z': float(delta[..., i][m].mean()),
                                           'share_changed': float((delta[..., i][m] != 0).mean())})
            del delta, fresh
    support_rows = []
    for arm in ARMS:
        degree = np.load(out/f'features/{arm}/target_degree.npy')
        for population, pmask in (('all_val_events', np.ones(len(inst), bool)), ('natural_hit_rows', np.isfinite(archive['retrieval_rank']))):
            for group, gmask in groups.items():
                m = pmask & gmask
                support_rows.append({'arm': arm, 'population': population, 'group': group, 'events': int(m.sum()),
                                     'share_target_graph_known': float((degree[m] > 0).mean()),
                                     'mean_log1p_target_degree': float(np.log1p(degree[m]).mean())})
    compact.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(compact/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(compact/'contrasts.csv', index=False)
    pd.DataFrame(shift_rows).to_csv(compact/'feature_shifts.csv', index=False)
    pd.DataFrame(support_rows).to_csv(compact/'target_support.csv', index=False)
    for name in ('reproduce.json', 'snapshots.json', 't0_controls.json'):
        (compact/name).write_text((out/name).read_text())
    (compact/'git_hash.txt').write_text(git_hash()+'\n')
    (compact/'config.yaml').write_text(yaml.safe_dump({
        'label': LABEL, 'study': STUDY, 'parent': PARENT, 'factorial': FACTORIAL, 'methods': list(METHODS),
        'seeds': list(SEEDS), 'arms': {'T0': 'all reviews strictly before T0 (saved parent features)',
                                       'T1': 'all reviews strictly before T1', 'monthly': 'all reviews strictly before the UTC month start of each query'},
        'split': {'t0': T0, 't1': T1}, 'fixed': 'candidates, histories, checkpoints, prefix scalers, post-standardisation masks',
        'scored_rows': 'natural K200 hits only (misses keep rank inf)', 'device': 'cpu',
        'metrics': ['ndcg10', 'recall10', 'hr3', 'mrr10'], 'aggregation': 'seed-mean event metric, then per-user macro over all Val events',
        'bootstrap': {'resamples': 1000, 'seed': 0, 'unit': 'user'}, 'groups': {'new': 'target first_ts >= T1', 'warm': 'otherwise'},
        'test_opened': False, 'selection': 'none'}, sort_keys=False))
    print(json.dumps({'analysed': str(compact)}), flush=True)


def phase_sensitivity():
    """monthly - T1 excluding August 2021, whose month-start snapshot precedes T1 (nDCG@10 only)."""
    from cascade.eval.bootstrap import paired_bootstrap
    P, meta, data, inst, archive, hit = _context()
    out = _root()/STUDY; compact = _root()/COMPACT
    users = inst.user.to_numpy(); ts = inst.ts.to_numpy(np.int64)
    keep = month_start_ms(ts) >= T1
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= T1
    rows = []
    for method in METHODS:
        vals = {}
        for arm in ('T1', 'monthly'):
            seeds = []
            for seed in SEEDS:
                with np.load(out/f'ranks/{arm}/{method}/seed{seed}.npz') as z: seeds.append(_metric_values(z['rank'])['ndcg10'])
            vals[arm] = np.mean(seeds, axis=0)
        for group, g in (('all', keep), ('new', keep & new), ('warm', keep & ~new)):
            ci = paired_bootstrap(users[g], vals['T1'][g], vals['monthly'][g], n_resamples=1000, seed=0)
            rows.append({'method': method, 'contrast': 'monthly - T1, Sep 2021 onward', 'group': group,
                         'metric': 'ndcg10', 'events': int(g.sum()), **ci})
    pd.DataFrame(rows).to_csv(compact/'sensitivity_excluding_august.csv', index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('reproduce', 'features', 'score', 'analyse', 'sensitivity'))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    {'reproduce': phase_reproduce, 'features': phase_features, 'score': phase_score, 'analyse': phase_analyse,
     'sensitivity': phase_sensitivity}[args.phase]()


if __name__ == '__main__':
    main()
