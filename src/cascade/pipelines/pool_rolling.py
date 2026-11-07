"""P3 fused pool x rolling monthly co-review features (Val only, exploratory).

Combines two existing studies without editing either: the P3 candidate pool, split and negative plans from
``pool_ranking`` (``20261106-pool-ranking-P3``) and the rolling monthly snapshot rule from ``rolling_ranking``
(B queries take their nine common + six graph features from all reviews strictly before their UTC month start,
floored at T0; Val queries likewise, with a T1 diagnostic arm; query user excluded). Prefix scalers are refitted
on the rolling P3 prefix rows. The recipe is the source BCE recipe; ``bce_coreview`` enables graph columns
0/1/2/5 (``objective_coreview``), ``bce_text`` disables them. No Test file is opened; nothing is selected.

Phases (repo root, PYTHONPATH=src):
    prepare  (CPU, <=3 threads)  rolling P3 B features, scaler, Val monthly/T1 natural-hit features
    fit      (MPS)               --method {bce_coreview,bce_text} --seed {0,1,2}
    evaluate (CPU)               every fitted checkpoint x {val_monthly, val_T1}
    analyse  (CPU)               metrics, contrasts, report.md
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from cascade.analysis import freshness_val as F
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import objective_coreview as OC
from cascade.pipelines import pool_ranking as PR
from cascade.pipelines import rolling_ranking as RR
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.utils.config import REPO_ROOT

LABEL = 'Val only, exploratory'
STUDY = REPO_ROOT/'results/local_experiments/20261107-pool-rolling'
COMPACT = REPO_ROOT/'results/analysis/20261107-pool-rolling'
P3 = REPO_ROOT/'results/local_experiments/20261106-pool-ranking-P3'
POOL = REPO_ROOT/PR.POOL_BASE/'candidates'/PR.POOLS['P3']
ROLL = REPO_ROOT/'results/local_experiments/20261106-rolling-ranking'
COMBO_T1 = REPO_ROOT/'results/local_experiments/20261106-pool-combo-T1'
COMBO_CTX = REPO_ROOT/'results/local_experiments/20261106-pool-combo-context'
METHODS = ('bce_coreview', 'bce_text')
SEEDS = (0, 1, 2)
ARMS = ('val_monthly', 'val_T1')
SOURCE = 'src/cascade/pipelines/pool_rolling.py'
HYPOTHESIS = ('Yes on both (beats P3 bce_text T1 0.011451 and P1 rolling bce_coreview 0.010613), sub-additively; '
              'co-review - text on P3 under rolling features is positive.')


def mask_for(method):
    return RR.mask_for(method)


def _p3():
    """P3 study metadata, with the pool archive, split and plans checked against its recorded digests."""
    meta, _ = PR.verify(P3, artifacts=False, post_evaluation=True)
    if meta['pool'] != 'P3' or meta['technical_smoke'] or meta['sample']:
        raise ValueError('unexpected P3 study')
    if meta['split']['t_ranker_train'] != F.T0 or meta['split']['t1'] != F.T1:
        raise ValueError('split constants changed')
    for name, expected in meta['pool_sha256'].items():
        if digest(REPO_ROOT/name) != expected: raise ValueError(f'pool archive changed: {name}')
    for name, expected in meta['artifact_sha256'].items():
        if name.startswith(('plans/', 'split/', 'metadata/')) and digest(P3/name) != expected:
            raise ValueError(f'P3 artifact changed: {name}')
    return meta


def _archive(split):
    arc = P.arrays(POOL/split)
    if set(arc) != set(PR.ARCHIVE_KEYS): raise ValueError('pool archive files differ')
    return arc


class NaturalRows:
    """B features stored for natural queries only (disk budget), served by full B row index.

    A query whose target is not in its P3 list has rank = inf whatever its scores, and the negative plans only
    draw natural lists, so its features never affect training, inner selection or metrics; zeros are served.
    """

    def __init__(self, array, natural):
        self.array = array
        self.position = np.full(len(natural), -1, np.int64); self.position[natural] = np.arange(len(natural.nonzero()[0]))

    def __getitem__(self, rows):
        rows = np.atleast_1d(np.asarray(rows))
        pos = self.position[rows]; out = np.zeros((len(rows), *self.array.shape[1:]), np.float32)
        keep = pos >= 0
        if keep.any(): out[keep] = np.asarray(self.array[np.sort(pos[keep])], np.float32)[np.argsort(np.argsort(pos[keep]))]
        return out


def b_features():
    natural = np.isfinite(np.load(POOL/'ranker_train/retrieval_rank.npy'))
    stored = np.load(STUDY/'features/ranker_train/natural_rows.npy')
    if not np.array_equal(stored, np.flatnonzero(natural)): raise ValueError('natural rows differ')
    return {k: NaturalRows(np.load(STUDY/f'features/ranker_train/{k}.npy', mmap_mode='r'), natural) for k in ('common', 'graph')}


def val_features(arm):
    return {k: np.load(STUDY/f'features/{arm}/{k}.npy') for k in ('common', 'graph')}


def free_gb():
    import shutil as _sh
    return _sh.disk_usage('/Users').free/1e9


def feature_bytes(n_b_natural, n_val_hit):
    return (n_b_natural + 2*n_val_hit)*200*15*4


def prepare():
    """One pass per snapshot: every prefix query feeds the scaler (all 200 slots, raw float64 sums, the source
    rule); natural B queries are stored (float32, as the source pipeline), then standardised in place."""
    torch.set_num_threads(3)
    meta = _p3()
    root = STUDY; root.mkdir(parents=True, exist_ok=True)
    if sorted(p.name for p in root.iterdir()) not in ([], ['study.json']): raise ValueError('bulk dir not empty')
    b_found = np.isfinite(np.load(POOL/'ranker_train/retrieval_rank.npy'))
    v_found = np.isfinite(np.load(POOL/'val/retrieval_rank.npy'))
    need = feature_bytes(int(b_found.sum()), int(v_found.sum()))/1e9
    print(json.dumps({'estimated_feature_gb': need, 'free_gb': free_gb()}), flush=True)
    if free_gb()-need < 18.5: raise SystemExit(f'disk: writing {need:.2f} GB would leave < 18.5 GB free')
    write_json(root/'study.json', {'label': LABEL, 'hypothesis': HYPOTHESIS, 'p3_study': str(P3.relative_to(REPO_ROOT)),
        'pool': str(POOL.relative_to(REPO_ROOT)), 'rolling_study': str(ROLL.relative_to(REPO_ROOT)),
        'storage': 'float32; B features stored for natural queries only (others cannot be ranked); Val natural-hit rows',
        'estimated_feature_gb': need, 'source_sha256': {SOURCE: digest(REPO_ROOT/SOURCE)}, 'test_predictions_loaded': False})
    data = P.data_for(meta)
    inst = data.split_instances('ranker_train')
    archive = _archive('ranker_train')
    if not (np.array_equal(archive['inst'], inst.inst) and np.array_equal(archive['target'], inst.item)):
        raise ValueError('B alignment differs')
    prefix = np.load(P3/'split/prefix.npy')
    if len(prefix) != len(inst): raise ValueError('prefix length')
    ts = inst.ts.to_numpy(np.int64)
    if (ts < F.T0).any() or (ts >= F.T1).any(): raise ValueError('non-B query')
    cutoffs = RR.rolling_cutoffs(ts, F.T0)
    natural = np.flatnonzero(b_found); slot = np.full(len(inst), -1, np.int64); slot[natural] = np.arange(len(natural))
    p3_scaler = P.read(P3/'features/manifest.json')['scaler']
    p3_common = np.load(P3/'features/ranker_train/common.npy', mmap_mode='r')
    (root/'features/ranker_train').mkdir(parents=True)
    np.save(root/'features/ranker_train/natural_rows.npy', natural)
    out = {key: open_memmap(root/f'features/ranker_train/{key}.npy', mode='w+', dtype=np.float32,
                            shape=(len(natural), 200, dim)) for key, dim in (('common', 9), ('graph', 6))}
    sums = {key: np.zeros(dim) for key, dim in (('common', 9), ('graph', 6))}
    squares = {key: np.zeros_like(v) for key, v in sums.items()}
    count = 0; audits = []; meta_err = 0.0
    meta_rows = set(np.random.default_rng(20261107).choice(natural, 256, replace=False).tolist())
    started = time.perf_counter()
    for cutoff in np.unique(cutoffs):
        t = time.perf_counter()
        builder = F.SnapshotFeatures(data, int(cutoff))
        rows = np.flatnonzero((cutoffs == cutoff) & (prefix | b_found)); t0_err = 0.0
        every = np.flatnonzero(cutoffs == cutoff)
        for start in range(0, len(every), 2048):  # cut rule on every query of the snapshot, stored or not
            part = every[start:start+2048]
            F.assert_cut_rule(builder, inst.iloc[part], np.asarray(archive['cand'][part]))
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            frame = inst.iloc[part]; cand = np.asarray(archive['cand'][part])
            common, graph = builder.build(frame, cand)
            pre = prefix[part]
            for key, val in (('common', common), ('graph', graph)):
                v = val[pre].astype(np.float64); sums[key] += v.sum(axis=(0, 1)); squares[key] += np.square(v).sum(axis=(0, 1))
            count += int(pre.sum())*200
            keep = slot[part] >= 0
            out['common'][slot[part][keep]] = common[keep]; out['graph'][slot[part][keep]] = graph[keep]
            if int(cutoff) == F.T0:  # partial first month reproduces the P3 study's T0 common features
                z = F._standardise(common, p3_scaler, 'common'); ref = np.asarray(p3_common[part])
                if not np.allclose(z, ref, atol=2e-5, rtol=2e-5): raise ValueError('T0 rows differ from P3')
                t0_err = max(t0_err, float(np.max(np.abs(z-ref))))
            sel = [i for i, r in enumerate(part) if int(r) in meta_rows]
            if sel:  # snapshot-free metadata columns 3-6 equal the P3 raw values
                raw_old = np.asarray(p3_common[part[sel]]).astype(np.float64)*np.asarray(p3_scaler['common']['std'])+np.asarray(p3_scaler['common']['mean'])
                err = float(np.max(np.abs(common[sel][..., 3:7]-raw_old[..., 3:7])))
                if err > 1e-4: raise ValueError('metadata columns changed')
                meta_err = max(meta_err, err)
        check = rows[np.sort(np.random.default_rng(int(cutoff) % 2**32).choice(len(rows), min(24, len(rows)), replace=False))]
        fast = builder.build(inst.iloc[check], np.asarray(archive['cand'][check]))
        ref = FrozenGraphFeatures.build(builder, inst.iloc[check], np.asarray(archive['cand'][check]))
        err = [float(np.max(np.abs(a-b))) for a, b in zip(fast, ref)]
        if max(err) > 2e-5: raise ValueError('batched rolling snapshot differs from reference')
        audits.append({**builder.audit, 'cutoff_utc': str(np.datetime64(int(cutoff), 'ms')),
                       'queries': int((cutoffs == cutoff).sum()), 'prefix_queries': int(prefix[cutoffs == cutoff].sum()),
                       'natural_queries': int(b_found[cutoffs == cutoff].sum()), 'min_query_ts': int(ts[cutoffs == cutoff].min()),
                       'reference_max_abs_error': err, 't0_common_max_abs_error_vs_p3': t0_err if int(cutoff) == F.T0 else None,
                       'seconds': time.perf_counter()-t,
                       'note': 'period start T0 (partial first month 26-31 Jul 2020)' if int(cutoff) == F.T0 else 'month start'})
        print(json.dumps({k: audits[-1][k] for k in ('cutoff_utc', 'review_edges', 'queries', 'natural_queries', 'seconds')}), flush=True)
        del builder; gc.collect()
    for v in out.values(): v.flush()
    del out; gc.collect()
    if count != int(prefix.sum())*200: raise ValueError('prefix slot count')
    scaler = {}
    for key in sums:
        mean = sums[key]/count
        std = np.sqrt(np.maximum(squares[key]/count-mean*mean, 0)); std[std < 1e-8] = 1
        scaler[key] = {'mean': mean.tolist(), 'std': std.tolist()}
        arr = np.load(root/f'features/ranker_train/{key}.npy', mmap_mode='r+')
        for start in range(0, len(arr), 256):
            value = ((arr[start:start+256].astype(np.float64)-mean)/std).astype(np.float32)
            if not np.isfinite(value).all(): raise ValueError('nonfinite scaled features')
            arr[start:start+256] = value
        arr.flush(); del arr
    gc.collect()
    vinst = data.split_instances('val'); varchive = _archive('val')
    if not (np.array_equal(varchive['inst'], vinst.inst) and np.array_equal(varchive['target'], vinst.item)):
        raise ValueError('Val alignment')
    hit = np.flatnonzero(np.isfinite(varchive['retrieval_rank'])); sub = F._sub_archive(varchive, hit)
    vts = vinst.ts.to_numpy(np.int64)
    val_audit = {'val_monthly': F._build_arm(P, data, vinst, sub, hit, F.month_start_ms(vts), scaler, root, 'val_monthly'),
                 'val_T1': F._build_arm(P, data, vinst, sub, hit, np.full(len(vinst), F.T1, np.int64), scaler, root, 'val_T1')}
    np.save(root/'features/val_hit.npy', hit)
    manifest = {'label': LABEL, 'scaler': scaler, 'prefix_slots': count, 'b_snapshots': audits, 'val': val_audit,
                'b_queries': int(len(inst)), 'b_prefix_queries': int(prefix.sum()), 'b_natural_queries': int(len(natural)),
                'val_hit_rows': int(len(hit)), 'estimated_feature_gb': need,
                'storage': 'float32; B features stored for natural queries only (non-natural queries rank = inf and are never drawn by plans)',
                'metadata_columns_max_abs_error_vs_p3': meta_err,
                'first_month_rule': 'B queries 26-31 Jul 2020 use T0; their common features equal the P3 study T0 features',
                'val_month_rule': 'Val August 2021 queries use 1 Aug 2021 (precedes T1)',
                'p3_scaler': p3_scaler, 'seconds': time.perf_counter()-started, 'test_opened': False}
    write_json(root/'features/manifest.json', manifest)
    print(json.dumps({'prepared': str(root), 'seconds': manifest['seconds']}), flush=True)


def fit(method, seed, device):
    torch.set_num_threads(3)
    meta = _p3(); config = meta['config']
    if not (STUDY/'features/manifest.json').exists(): raise ValueError('prepare first')
    run = STUDY/f'runs/{method}/seed{seed}'; run.mkdir(parents=True, exist_ok=False)
    archive = _archive('ranker_train')
    features = b_features()
    tune = np.flatnonzero(np.load(P3/'split/tune.npy'))
    mask = mask_for(method)
    p3_initial = P.read(P3/f'runs/bce_text/seed{seed}/metrics.json')['initial_state_sha256']
    source_initial = P.read(RR.source_run(method, seed)/'metrics.json')['initial_state_sha256']
    model = P.model_for(meta, P3, seed, device); initial = P.state_hash(model)
    if initial != p3_initial or initial != source_initial:
        raise ValueError('initial state differs from the P3 bce_text / source fits')
    write_json(run/'config.json', {'label': LABEL, 'method': method, 'seed': seed, 'objective': 'bce', 'feature_mask': mask,
        'pool': 'P3', 'features': 'rolling monthly B snapshots on P3 candidates, rolling P3 prefix scaler',
        'plans': str((P3/'plans').relative_to(REPO_ROOT)), 'device': device, 'source_sha256': digest(REPO_ROOT/SOURCE),
        **{k: config[k] for k in ('max_epochs', 'patience', 'batch_lists', 'negative_alternatives', 'learning_rate')}})
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    data = P.data_for(meta); first = np.zeros(len(model.text), np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy(); del data; gc.collect()
    cold = first[archive['target'][tune]] >= F.T0
    found = np.isfinite(archive['retrieval_rank'][tune]); users = archive['user'][tune]
    (run/'inner').mkdir(); history = []; best = -np.inf; selected = 0; stale = 0
    with np.load(P3/f'plans/seed{seed}_inner.npz') as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter()
    for epoch in range(config['max_epochs']+1):
        loss = None if epoch == 0 else OC.train_epoch(model, optimizer, archive, features, rows[epoch-1], columns[epoch-1], mask, device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = P.predict(model, archive, features, tune, mask, device)
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or not np.array_equal(ranks, archive['retrieval_rank'][tune])
                           or not np.array_equal(top, archive['top10'][tune])):
            raise ValueError('epoch zero differs from retrieval')
        value = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(value, users),
               'unseen_before_T0_ndcg10': average(value[cold], users[cold]), 'conditional_ndcg10': average(value[found], users[found])}
        history.append(row); write_json(run/'inner_history.json', history)
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        print(json.dumps({'method': method, 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        del logits
        if row['ndcg10'] > best: best = row['ndcg10']; selected = epoch; stale = 0
        elif epoch: stale += 1
        if stale >= config['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected}.pth', map_location='cpu', weights_only=True), strict=True)
    with np.load(run/f'inner/epoch{selected}_ranks.npz') as z:
        if not np.array_equal(P.predict(model, archive, features, tune, mask, device)[0], z['rank']):
            raise ValueError('selected checkpoint replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = P.model_for(meta, P3, seed, device)
    if P.state_hash(model) != initial: raise ValueError('refit initial state changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    with np.load(P3/f'plans/seed{seed}_refit.npz') as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter(); refit = []
    for epoch in range(selected):
        loss = OC.train_epoch(model, optimizer, archive, features, rows[epoch], columns[epoch], mask, device)
        refit.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': method, 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth'); write_json(run/'refit_history.json', refit)
    before = P.predict(model, archive, features, tune[:128], mask, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    if not np.array_equal(before, P.predict(model, archive, features, tune[:128], mask, device)[2]):
        raise ValueError('portable state reload differs')
    p3_fit = P.read(P3/f'runs/bce_text/seed{seed}/metrics.json')
    write_json(run/'metrics.json', {'status': 'fitted', 'method': method, 'seed': seed, 'selected_epoch': selected,
        'p3_bce_text_selected_epoch': p3_fit['selected_epoch'], 'inner_ndcg10': best, 'p3_bce_text_inner_ndcg10': p3_fit['inner_ndcg10'],
        'initial_state_sha256': initial, 'initial_state_matches_p3_bce_text': True, 'initial_state_matches_source': True,
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds,
        'refit_seconds': refit_seconds, 'portable_reload_exact': True, 'epoch0_equals_retrieval_on_tune': True,
        'test_predictions_loaded': False})
    print(json.dumps({'fitted': method, 'seed': seed, 'selected_epoch': selected, 'p3_text_epoch': p3_fit['selected_epoch']}), flush=True)


def fitted(method):
    return all((STUDY/f'runs/{method}/seed{s}/metrics.json').exists() for s in SEEDS)


def evaluate():
    torch.set_num_threads(3)
    meta = _p3()
    archive = _archive('val')
    hit = np.load(STUDY/'features/val_hit.npy')
    if not np.array_equal(hit, np.flatnonzero(np.isfinite(archive['retrieval_rank']))): raise ValueError('hit rows differ')
    sub = F._sub_archive(archive, hit); sub['retrieval_rank'] = np.asarray(archive['retrieval_rank'][hit])
    arms = {arm: val_features(arm) for arm in ARMS}
    zero_path = STUDY/'evaluation/epoch0_check.json'
    if not zero_path.exists():  # untrained model on Val hit rows reproduces the P3 retrieval order, both arms, both masks
        model = P.model_for(meta, P3, 0, 'cpu').eval(); zero = {}
        for arm, feats in arms.items():
            for method in METHODS:
                ranks, _, logits = P.predict(model, sub, feats, np.arange(len(hit)), mask_for(method), 'cpu')
                zero[f'{arm}/{method}'] = bool(np.array_equal(logits, sub['score']) and np.array_equal(ranks, sub['retrieval_rank']))
        if not all(zero.values()): raise ValueError(f'epoch-zero Val differs from retrieval: {zero}')
        zero_path.parent.mkdir(parents=True, exist_ok=True); write_json(zero_path, zero)
        del model; gc.collect()
    for method in METHODS:
        if not fitted(method):
            print(json.dumps({'skipped': method, 'reason': 'not all seeds fitted'}), flush=True); continue
        for seed in SEEDS:
            run = STUDY/f'runs/{method}/seed{seed}'; record = P.read(run/'metrics.json')
            if digest(run/'model.pth') != record['model_sha256']: raise ValueError('checkpoint changed')
            for arm, feats in arms.items():
                dest = STUDY/f'evaluation/{arm}/{method}/seed{seed}.npz'
                if dest.exists(): continue
                model = P.model_for(meta, P3, seed, 'cpu').eval()
                model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
                before = P.state_hash(model)
                ranks, top, logits = P.predict(model, sub, feats, np.arange(len(hit)), mask_for(method), 'cpu')
                if P.state_hash(model) != before: raise ValueError('scoring modified weights')
                rr, _ = ranks_and_top10(logits, sub['cand'], sub['target'])
                if not np.array_equal(rr, ranks): raise ValueError('rank replay')
                full = np.full(len(archive['inst']), np.inf); full[hit] = ranks
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, inst=np.asarray(archive['inst']), user=np.asarray(archive['user']), rank=full)
                print(json.dumps({'evaluated': arm, 'method': method, 'seed': seed,
                                  'ndcg10': average(M.ndcg_at(full, 10), np.asarray(archive['user']))}), flush=True)
                del model, logits; gc.collect()


# ---------------------------------------------------------------- analysis
def reference_paths():
    """Existing seed rank files (read only) used as comparison systems."""
    return {'P3_bce_text_T0': lambda s: P3/f'evaluation/bce_text/seed{s}/val_ranks.npz',
            'P3_bce_text_T1': lambda s: COMBO_T1/f'ranks/T1/seed{s}.npz',
            'P3_context_T1': lambda s: COMBO_CTX/f'evaluation/common_T1/seed{s}.npz',
            'P1_rolling_bce_coreview_monthly': lambda s: ROLL/f'evaluation/val_monthly/bce_coreview/seed{s}.npz',
            'P1_rolling_bce_text_monthly': lambda s: ROLL/f'evaluation/val_monthly/bce_text/seed{s}.npz',
            'P1_bce_text_T0': lambda s: REPO_ROOT/F.PARENT/f'evaluation/bce_text/seed{s}/val_ranks.npz'}


def additive_terms():
    """Interaction = observed P3 rolling co-review minus (P3 text T0 + P1 rolling co-review - P1 text T0)."""
    return ('P3_rolling_bce_coreview_monthly', 'P1_bce_text_T0'), ('P3_bce_text_T0', 'P1_rolling_bce_coreview_monthly')


def contrast_plan(h3):
    main = 'P3_rolling_bce_coreview_monthly'
    plan = [('vs_P3_text_T1', 'P3_bce_text_T1', main), ('vs_P1_rolling_coreview', 'P1_rolling_bce_coreview_monthly', main),
            ('vs_P3_text_T0', 'P3_bce_text_T0', main), ('vs_P3_context_T1', 'P3_context_T1', main),
            ('serving_monthly_minus_T1', 'P3_rolling_bce_coreview_T1', main)]
    if h3:
        plan += [('coreview_minus_text_P3_rolling', 'P3_rolling_bce_text_monthly', main),
                 ('coreview_minus_text_P3_rolling_T1', 'P3_rolling_bce_text_T1', 'P3_rolling_bce_coreview_T1'),
                 ('text_rolling_vs_P3_text_T1', 'P3_bce_text_T1', 'P3_rolling_bce_text_monthly'),
                 ('text_rolling_vs_P3_text_T0', 'P3_bce_text_T0', 'P3_rolling_bce_text_monthly')]
    return plan


def _fmt(row):
    return f"{row['diff']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"


def _rel(path):
    try: return str(path.relative_to(REPO_ROOT))
    except ValueError: return str(path)


def analyse():
    import yaml
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.utils.io import git_hash
    _, meta, data, inst, archive, hit = F._context()
    users = inst.user.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= F.T1
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    h3 = fitted('bce_text') and all((STUDY/f'evaluation/{a}/bce_text/seed{s}.npz').exists() for a in ARMS for s in SEEDS)
    paths = dict(reference_paths())
    for method in METHODS if h3 else METHODS[:1]:
        for arm, tag in (('val_monthly', 'monthly'), ('val_T1', 'T1')):
            paths[f'P3_rolling_{method}_{tag}'] = (lambda a, m: (lambda s: STUDY/f'evaluation/{a}/{m}/seed{s}.npz'))(arm, method)
    values, rows = {}, []
    for system, path in paths.items():
        per_seed = []
        for seed in SEEDS:
            with np.load(path(seed)) as z:
                if not np.array_equal(z['inst'], inst.inst): raise ValueError(f'rank alignment: {system}')
                per_seed.append(F._metric_values(z['rank']))
        values[system] = {m: np.mean([s[m] for s in per_seed], axis=0) for m in per_seed[0]}
        for group, mask in groups.items():
            for metric in per_seed[0]:
                seeds = [average(s[metric][mask], users[mask]) for s in per_seed]
                rows.append({'system': system, 'group': group, 'metric': metric, 'value': float(np.mean(seeds)),
                             'seed_std_ddof0': float(np.std(seeds)), **{f'seed{s}': v for s, v in zip(SEEDS, seeds)}})
    contrasts = []
    plan = contrast_plan(h3)
    (pos_a, pos_b), (neg_a, neg_b) = additive_terms()
    for group, mask in groups.items():
        for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
            for family, ref, cand in plan:
                ci = paired_bootstrap(users[mask], values[ref][metric][mask], values[cand][metric][mask], n_resamples=1000, seed=0)
                contrasts.append({'family': family, 'reference': ref, 'candidate': cand, 'group': group, 'metric': metric,
                                  'events': int(mask.sum()), **ci})
            ref_sum = values[neg_a][metric][mask]+values[neg_b][metric][mask]
            cand_sum = values[pos_a][metric][mask]+values[pos_b][metric][mask]
            ci = paired_bootstrap(users[mask], ref_sum, cand_sum, n_resamples=1000, seed=0)
            contrasts.append({'family': 'interaction_pool_x_rolling', 'reference': f'({neg_a} + {neg_b})',
                              'candidate': f'{pos_a} + {pos_b}', 'group': group, 'metric': metric,
                              'events': int(mask.sum()), **ci})
    epochs = [{k: v for k, v in P.read(STUDY/f'runs/{m}/seed{s}/metrics.json').items() if k in
               ('method', 'seed', 'selected_epoch', 'p3_bce_text_selected_epoch', 'inner_ndcg10', 'p3_bce_text_inner_ndcg10',
                'fit_seconds', 'refit_seconds')}
              for m in METHODS if fitted(m) for s in SEEDS]
    COMPACT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows); cons = pd.DataFrame(contrasts)
    metrics.to_csv(COMPACT/'metrics.csv', index=False); cons.to_csv(COMPACT/'contrasts.csv', index=False)
    pd.DataFrame(epochs).to_csv(COMPACT/'epochs.csv', index=False)
    manifest = P.read(STUDY/'features/manifest.json')
    write_json(COMPACT/'features_manifest.json', manifest)
    (COMPACT/'inner_history').mkdir(exist_ok=True)
    for m in METHODS:
        if fitted(m):
            for s in SEEDS:
                write_json(COMPACT/f'inner_history/{m}_seed{s}.json', P.read(STUDY/f'runs/{m}/seed{s}/inner_history.json'))
    for name in ('epoch0_check.json',):
        write_json(COMPACT/name, P.read(STUDY/'evaluation'/name))
    (COMPACT/'git_hash.txt').write_text(git_hash()+'\n')
    (COMPACT/'config.yaml').write_text(yaml.safe_dump({
        'label': LABEL, 'study': _rel(STUDY), 'pool': _rel(POOL),
        'p3_study': _rel(P3), 'methods_fitted': [m for m in METHODS if fitted(m)], 'seeds': list(SEEDS),
        'h3_text_control_ran': bool(h3),
        'recipe': 'source recipe (BCE, 50 negatives, Adam 1e-3, batch 32 lists, <=10 epochs, patience 2, inner selection with epoch 0, all-B refit)',
        'masks': {m: mask_for(m) for m in METHODS}, 'plans': _rel((P3/'plans')),
        'features': {'B': 'rolling monthly snapshot per query (month start, floored at T0) on P3 candidates',
                     'scaler': 'refitted on rolling P3 B prefix slots (all 200)',
                     'Val': {'val_monthly': 'month start of each query (consistent)', 'val_T1': 'T1 snapshot (diagnostic)'}},
        'references': {k: _rel(v(0)).replace('seed0', 'seed{s}') for k, v in reference_paths().items()},
        'interaction': 'P3_rolling_coreview - P3_text_T0 - P1_rolling_coreview + P1_text_T0 (per event, seed means)',
        'metrics': ['ndcg10', 'recall10', 'hr3', 'mrr10'], 'bootstrap': {'resamples': 1000, 'seed': 0, 'unit': 'user'},
        'fit_device': 'mps', 'eval_device': 'cpu', 'test_opened': False, 'selection': 'none',
        'hypothesis': HYPOTHESIS}, sort_keys=False))
    write_report(metrics, cons, pd.DataFrame(epochs), manifest, h3)
    print(json.dumps({'analysed': str(COMPACT), 'h3': bool(h3)}), flush=True)


def write_report(metrics, cons, epochs, manifest, h3):
    get = lambda s, g='all', m='ndcg10': float(metrics[(metrics.system == s) & (metrics.group == g) & (metrics.metric == m)].value.iloc[0])
    con = lambda f, g='all', m='ndcg10': cons[(cons.family == f) & (cons.group == g) & (cons.metric == m)].iloc[0]
    sign = lambda r: 'positive' if r['ci_low'] > 0 else 'negative' if r['ci_high'] < 0 else 'inconclusive'
    lines = ['# P3 fused pool x rolling monthly co-review features: results (Val only, exploratory)', '',
             f'Pre-stated (README, before computing): {HYPOTHESIS}', '',
             '232,904 Val events / 124,416 users; per-user macro, misses = 0; seed means (0/1/2). Intervals: 1,000 '
             'paired-user draws, seed 0, on seed-mean event values; unadjusted. Fits on MPS, scoring on CPU. No Test '
             'file opened; nothing selected. These are reused-Val exploratory comparisons.', '',
             f"Rolling P3 features: {len(manifest['b_snapshots'])} B snapshots, {manifest['val_hit_rows']:,} Val natural-hit rows "
             f"(P3 recall@200 ceiling). Build {manifest['seconds']/60:.1f} min.", '', '## Systems', '',
             '| System | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 | nDCG@10 seed sd |', '|---|---|---:|---:|---:|---:|---:|']
    for system in metrics.system.unique():
        for g in ('all', 'new', 'warm'):
            sd = float(metrics[(metrics.system == system) & (metrics.group == g) & (metrics.metric == 'ndcg10')].seed_std_ddof0.iloc[0])
            lines.append(f"| {system} | {g} | " + ' | '.join(f'{get(system, g, m):.6f}' for m in ('ndcg10', 'recall10', 'hr3', 'mrr10'))
                         + f' | {sd:.6f} |')
    lines += ['', '## Paired differences (candidate − reference) [95% CI]', '',
              '| Contrast | Candidate − reference | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |', '|---|---|---|---:|---:|---:|---:|']
    for family in cons.family.unique():
        for g in ('all', 'new', 'warm'):
            r = con(family, g)
            lines.append(f"| {family} | {r['candidate']} − {r['reference']} | {g} | "
                         + ' | '.join(_fmt(con(family, g, m)) for m in ('ndcg10', 'recall10', 'hr3', 'mrr10')) + ' |')
    lines += ['', '## Selected epochs', '', epochs.to_markdown(index=False) if len(epochs) else '(none)', '',
              '## Pre-stated expectation versus outcome (nDCG@10, all events)', '']
    a, b, i = con('vs_P3_text_T1'), con('vs_P1_rolling_coreview'), con('interaction_pool_x_rolling')
    lines += [f"- Beats P3 bce_text T1 (0.011451): {_fmt(a)} → {sign(a)}.",
              f"- Beats P1 rolling bce_coreview (0.010613): {_fmt(b)} → {sign(b)}.",
              f"- Additivity: interaction (observed − [P3 text T0 + P1 rolling co-review − P1 text T0]) {_fmt(i)} → {sign(i)} "
              '(negative = sub-additive). The pool gain is measured with bce_text at T0 and the rolling gain with co-review '
              'on P1, so this interaction also absorbs any method-by-pool difference.']
    if h3:
        c = con('coreview_minus_text_P3_rolling')
        lines.append(f"- Co-review − text within P3 rolling (monthly): {_fmt(c)} → {sign(c)}.")
    else:
        lines.append('- Co-review − text within P3 rolling: H3 text control did not run (time gate); not measured.')
    lines += ['', 'Groups: new = target first review at/after T1; warm = otherwise. New/warm rows are shown in the tables above.', '']
    (COMPACT/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'fit', 'evaluate', 'analyse'))
    parser.add_argument('--method', choices=METHODS); parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    if args.phase == 'prepare': return prepare()
    if args.phase == 'fit':
        from cascade.pipelines.train_baseline import pick_device
        return fit(args.method, args.seed, str(pick_device(args.device)))
    return {'evaluate': evaluate, 'analyse': analyse}[args.phase]()


if __name__ == '__main__':
    main()
