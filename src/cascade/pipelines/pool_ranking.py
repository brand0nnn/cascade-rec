"""Fixed ``bce_text`` residual ranker on alternative candidate pools (Val only, exploratory).

Question: holding the registered ``bce_text`` recipe fixed, how does the candidate pool
change what the ranker delivers?  Recipe reused unchanged by import from
``full_data_ranking`` (GraphLatentRanker R2 with graph columns disabled, BCE, 50 uniform
real non-target negatives per natural list, Adam 1e-3, 32 lists, <=10 epochs, patience 2,
chronological inner selection with epoch 0 eligible, fixed-epoch all-B refit from the same
initial state).  Per pool: new natural-list plans, the nine common columns built with the
same pre-T0 ``BatchedGraphFeatures`` builder, standardised over all 200 slots of the
chronological-prefix B queries (the source study's rule for its own pool).

Pools are read-only archives built by ``cascade.analysis.pools_build``.  Column order is
not assumed to be retrieval order anywhere: ranks are pessimistic counts and top-10 lists
come from a stable argsort of the stored score (P3 stores interleave order with RRF score).

The registered P3t variant (RRF minus 1e-6 x interleave position) has a gate: its epoch-0
order must reproduce the interleave-order metrics.  ``check_p3t`` evaluates the gate and
writes no archive.  Test data is never opened.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines.graph_latent_study import negative_plan, split_cutoff, state_hash
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.local_rankers import metadata_luts
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

METHOD = 'bce_text'
SOURCE = '20261103-full-promising-ranking'
POOL_BASE = 'results/local_experiments/20261106-pools'
POOLS = {'P3': 'P3_fused_rrf', 'P2': 'P2_two_tower'}
P3T_DIR = 'P3t_fused_rrf_tiebreak'
P3T_EPS = 1e-6
INTERLEAVE_NDCG10 = 0.007408  # results/analysis/20261106-pools/tables.md, P3_interleave_order overall
ANALYSIS = 'results/analysis/20261106-pool-ranking'
SPLITS = ('ranker_train', 'val')
ARCHIVE_KEYS = ('cand', 'score', 'inst', 'user', 'target', 'retrieval_rank', 'top10', 'history')
COMMON, GRAPH = 9, 6
SOURCES = ('src/cascade/pipelines/pool_ranking.py', *fdr.SOURCES)
HYPOTHESIS = ('Pre-stated (before fitting): the ranker on the fused pool beats the ranker on P1 overall by clearly more '
              'than about 0.0003 nDCG@10, mostly through warm targets; new-item quality may fall. A larger natural '
              'training set (P2/P3 about 38.8k vs 20.7k positives, skewed warm) may also help.')


def study_name(pool, smoke=False):
    return '20261106-pool-ranking-smoke' if smoke else f'20261106-pool-ranking-{pool}'


class ZeroGraph:
    """Graph block for ``bce_text``: every graph column is disabled, so supply zeros on demand."""

    def __getitem__(self, rows):
        return np.zeros((len(np.atleast_1d(rows)), 200, GRAPH), np.float32)


def features_for(root, split):
    return {'common': np.load(root/f'features/{split}/common.npy', mmap_mode='r'), 'graph': ZeroGraph()}


def candidate_dir(meta, split):
    return REPO_ROOT/meta['candidate_dirs'][split]


# ---------------------------------------------------------------- archive checks
def check_archive(arc, inst, first_ts, reference=None):
    """Structural checks; column order is never assumed to follow the score."""
    if set(arc) != set(ARCHIVE_KEYS):
        raise ValueError('unexpected archive files')
    n = len(inst)
    if arc['cand'].shape != (n, 200) or arc['score'].shape != (n, 200) or not np.issubdtype(arc['cand'].dtype, np.integer):
        raise ValueError('archive shape/dtype differs')
    if not np.array_equal(arc['inst'], inst.inst.to_numpy()) or not np.array_equal(arc['user'], inst.user.to_numpy()) \
            or not np.array_equal(arc['target'], inst.item.to_numpy()):
        raise ValueError('archive events differ from the harness instances')
    if reference is not None:
        for key in ('inst', 'user', 'target', 'history'):
            if not np.array_equal(arc[key], reference[key]):
                raise ValueError(f'archive {key} differs from the P1 archive')
    ts = inst.ts.to_numpy()
    out = {'rows': n, 'padding_slots': 0, 'unsorted_rows': 0, 'natural': 0}
    for start in range(0, n, 8192):
        c = np.asarray(arc['cand'][start:start+8192]); s = np.asarray(arc['score'][start:start+8192])
        t = np.asarray(arc['target'][start:start+8192])
        real = c > 0
        out['padding_slots'] += int((~real).sum())
        if ((real.sum(1)) < 51).any():
            raise ValueError('fewer than 51 real candidates')
        o = np.sort(c, axis=1)
        if ((np.diff(o, axis=1) == 0) & (o[:, 1:] > 0)).any():
            raise ValueError('duplicate candidate')
        if (first_ts[c][real] > np.broadcast_to(ts[start:start+8192, None], c.shape)[real]).any():
            raise ValueError('unreleased candidate')
        if not np.isfinite(s[real]).all():
            raise ValueError('nonfinite candidate score')
        out['unsorted_rows'] += int((np.diff(np.where(real, s, -np.inf), axis=1) > 0).any(1).sum())
        r, top = ranks_and_top10(s, c, t)
        if not np.array_equal(r, np.asarray(arc['retrieval_rank'][start:start+8192])) or \
                not np.array_equal(top, np.asarray(arc['top10'][start:start+8192])):
            raise ValueError('stored retrieval_rank/top10 do not follow the stored score')
        out['natural'] += int(np.isfinite(r).sum())
    return out


def slice_archive(arc, rows, directory):
    directory.mkdir(parents=True, exist_ok=False)
    for key in ARCHIVE_KEYS:
        np.save(directory/f'{key}.npy', np.asarray(arc[key][rows]))
    np.save(directory/'source_rows.npy', rows)


# ---------------------------------------------------------------- phases
FROZEN_FUNCTIONS = ('ZeroGraph', 'features_for', 'candidate_dir', 'check_archive', 'slice_archive',
                    'p1_feature_crosscheck', 'prepare', 'fit', 'evaluate')


def function_sources(path, names=FROZEN_FUNCTIONS):
    import ast
    text = Path(path).read_text()
    tree = ast.parse(text)
    return {n.name: ast.get_source_segment(text, n) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}


def verify(root, artifacts=True, post_evaluation=False):
    """``post_evaluation=True`` (cross_score/report only) accepts a later extension of this module if every
    function used by prepare/fit/evaluate is identical to the study's frozen source copy."""
    meta = fdr.read(root/'study.json')
    if meta['test_predictions_loaded'] or meta['method'] != METHOD:
        raise ValueError('test exposure or method changed')
    for name, expected in meta['source_sha256'].items():
        frozen = root/'source'/Path(name).name
        if digest(frozen) != expected:
            raise ValueError(f'frozen source copy changed: {name}')
        if digest(REPO_ROOT/name) != expected:
            same = function_sources(REPO_ROOT/name) == function_sources(frozen) and \
                len(function_sources(frozen)) == len(FROZEN_FUNCTIONS)
            if not (post_evaluation and name == SOURCES[0] and same):
                raise ValueError(f'implementation changed: {name}')
    if digest(root/'protocol.json') != meta['protocol_sha256']:
        raise ValueError('registered protocol changed')
    if artifacts:
        for name, expected in {**meta.get('artifact_sha256', {}), **meta['pool_sha256']}.items():
            path = root/name if name in meta.get('artifact_sha256', {}) else REPO_ROOT/name
            if digest(path) != expected:
                raise ValueError(f'artifact changed: {name}')
    source_root = study_path(meta['source_study'])
    return meta, source_root


def p1_feature_crosscheck(builder, data, source_root, rows_per_split=256):
    """The same builder on P1 candidates, scaled by the P1 scaler, reproduces stored P1 features."""
    scaler = fdr.read(source_root/'features/manifest.json')['scaler']['common']
    mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    proof = {}
    for split in SPLITS:
        inst = data.split_instances(split)
        rows = np.sort(np.random.default_rng(7).choice(len(inst), rows_per_split, replace=False))
        cand = np.load(source_root/f'candidates/{split}/cand.npy', mmap_mode='r')[rows]
        stored = np.load(source_root/f'features/{split}/common.npy', mmap_mode='r')[rows]
        common = builder.build(inst.iloc[rows], cand)[0]
        rebuilt = ((common.astype(np.float64)-mean)/std).astype(np.float32)
        err = float(np.max(np.abs(rebuilt-stored)))
        if not np.allclose(rebuilt, stored, rtol=1e-5, atol=1e-5):
            raise ValueError(f'builder does not reproduce P1 common features on {split}: {err}')
        proof[split] = {'queries': len(rows), 'max_abs_error': err}
    return proof


def prepare(pool, root, smoke_rows=None):
    source_root = study_path(SOURCE)
    source_meta = fdr.verify(source_root, artifacts=False)
    if source_meta['sample']:
        raise ValueError('full-data source expected')
    cfg = source_meta['config']
    data = fdr.data_for(source_meta)
    pool_dir = REPO_ROOT/POOL_BASE/'candidates'/POOLS[pool]
    first = np.full(data.n_items+1, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    first[0] = np.iinfo(np.int64).min
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    for name in SOURCES:
        shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    smoke = smoke_rows is not None
    protocol = {'method': METHOD, 'pool': pool, 'pool_archive': f'{POOL_BASE}/candidates/{POOLS[pool]}',
                'source_study': SOURCE, 'technical_smoke': smoke, 'smoke_rows_per_split': smoke_rows,
                'recipe': 'bce_text from full_data_ranking unchanged (imported); graph columns disabled',
                'features': 'nine common columns, BatchedGraphFeatures with the strictly pre-T0 graph (same builder)',
                'scaler': 'mean/std over all 200 slots of chronological-prefix B queries of this pool',
                'split': 'source study chronological prefix/tune (split_cutoff of B)',
                'plans': 'negative_plan per seed on this pool: inner = prefix natural, refit = all-B natural',
                'initial_state': 'model_for(seed) must equal the rebuilt P1 bce_text initial_state_sha256',
                'column_order': 'not assumed; pessimistic ranks and stable-argsort top-10 from the stored score',
                'hypothesis': HYPOTHESIS, 'label': 'Val only, exploratory', 'selection': 'none',
                'outer_val': 'once after all three seeds are locked; reused Val population'}
    write_json(root/'protocol.json', protocol)
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    pool_sha = {f'{POOL_BASE}/candidates/{POOLS[pool]}/{s}/{k}.npy': digest(pool_dir/s/f'{k}.npy')
                for s in SPLITS for k in ARCHIVE_KEYS}
    meta = {'status': 'preparing', 'method': METHOD, 'pool': pool, 'sample': False, 'technical_smoke': smoke,
            'source_study': SOURCE, 'config': {**cfg, 'methods': [METHOD]}, 'split': data.split,
            'text_path': source_meta['text_path'], 'git_hash': git_hash(),
            'source_sha256': {n: digest(REPO_ROOT/n) for n in SOURCES}, 'protocol_sha256': digest(root/'protocol.json'),
            'pool_sha256': pool_sha, 'test_predictions_loaded': False}
    write_json(root/'study.json', meta)
    checks, insts, archives, dirs = {}, {}, {}, {}
    for split in SPLITS:
        inst = data.split_instances(split)
        arc = fdr.arrays(pool_dir/split)
        checks[split] = check_archive(arc, inst, first, fdr.arrays(source_root/f'candidates/{split}'))
        rows = np.arange(len(inst))
        if smoke:
            rows = np.sort(np.random.default_rng(0).choice(len(inst), smoke_rows, replace=False))
            slice_archive(arc, rows, root/'candidates'/split)
            arc = fdr.arrays(root/'candidates'/split)
            arc.pop('source_rows')
            dirs[split] = str((root/'candidates'/split).relative_to(REPO_ROOT))
        else:
            dirs[split] = f'{POOL_BASE}/candidates/{POOLS[pool]}/{split}'
        insts[split], archives[split] = (inst.iloc[rows], rows), arc
        print(json.dumps({'checked': split, **checks[split]}), flush=True)
    b_inst, b_rows = insts['ranker_train']
    cutoff, prefix_full = split_cutoff(data.split_instances('ranker_train'))
    if not np.array_equal(prefix_full, np.load(source_root/'split/prefix.npy')) or \
            cutoff != source_meta['inner_split']['timestamp_ms']:
        raise ValueError('chronological split differs from the source study')
    prefix = prefix_full[b_rows]
    found = np.isfinite(archives['ranker_train']['retrieval_rank'])
    split_info = {'timestamp_ms': cutoff, 'prefix_queries': int(prefix.sum()), 'tune_queries': int((~prefix).sum()),
                  'prefix_natural': int((prefix & found).sum()), 'tune_natural': int((~prefix & found).sum()),
                  'all_B_natural': int(found.sum()), 'val_natural': int(np.isfinite(archives['val']['retrieval_rank']).sum())}
    fdr.save_arrays(root/'split', {'prefix': prefix, 'tune': ~prefix})
    (root/'plans').mkdir()
    for seed in cfg['seeds']:
        for stage, eligible in (('inner', np.flatnonzero(prefix & found)), ('refit', np.flatnonzero(found))):
            r, c = negative_plan(archives['ranker_train'], eligible, seed, epochs=cfg['max_epochs'])
            np.savez_compressed(root/f'plans/seed{seed}_{stage}.npz', rows=r, columns=c)
    print(json.dumps({'plans': 'done', **split_info}), flush=True)
    builder = BatchedGraphFeatures(data)
    proof = {'audit': builder.audit, 'archive_checks': checks, 'split': split_info, 'reference_comparisons': {},
             'p1_crosscheck': p1_feature_crosscheck(builder, data, source_root)}
    (root/'features').mkdir()
    total, square, count = np.zeros(COMMON), np.zeros(COMMON), 0
    for split in SPLITS:
        inst, _ = insts[split]
        cand = archives[split]['cand']
        check = np.sort(np.random.default_rng(104729).choice(len(inst), min(24, len(inst)), replace=False))
        actual = builder.build(inst.iloc[check], cand[check])
        reference = FrozenGraphFeatures.build(builder, inst.iloc[check], cand[check])
        if not all(np.allclose(a, b, rtol=2e-6, atol=2e-6) for a, b in zip(actual, reference)):
            raise ValueError('batched graph construction differs from reference')
        proof['reference_comparisons'][split] = {'queries': len(check),
                                                 'max_abs_error': [float(np.max(np.abs(a-b))) for a, b in zip(actual, reference)]}
        (root/'features'/split).mkdir()
        out = open_memmap(root/f'features/{split}/common.npy', mode='w+', dtype=np.float32, shape=(*cand.shape, COMMON))
        started = time.perf_counter()
        for start in range(0, len(inst), 128):
            stop = min(start+128, len(inst))
            value = builder.build(inst.iloc[start:stop], cand[start:stop])[0]  # graph block unused by bce_text
            out[start:stop] = value
            if split == 'ranker_train':
                sel = value[prefix[start:stop]].astype(np.float64)
                total += sel.sum(axis=(0, 1)); square += np.square(sel).sum(axis=(0, 1))
                count += int(prefix[start:stop].sum())*200
            if start % (128*200) == 0:
                print(json.dumps({'features': split, 'done': stop, 'total': len(inst),
                                  'seconds': time.perf_counter()-started}), flush=True)
        out.flush(); del out
    mean = total/count
    std = np.sqrt(np.maximum(square/count-mean*mean, 0)); std[std < 1e-8] = 1
    for split in SPLITS:
        out = np.load(root/f'features/{split}/common.npy', mmap_mode='r+')
        for start in range(0, len(out), 256):
            value = ((out[start:start+256].astype(np.float64)-mean)/std).astype(np.float32)
            if not np.isfinite(value).all(): raise ValueError('nonfinite scaled features')
            out[start:start+256] = value
        out.flush(); del out
    p1_scaler = fdr.read(source_root/'features/manifest.json')['scaler']['common']
    proof.update(scaler={'common': {'mean': mean.tolist(), 'std': std.tolist()}}, p1_scaler_common=p1_scaler,
                 common_names=list(builder.common_names), graph='disabled (bce_text); not stored')
    write_json(root/'features/manifest.json', proof)
    leaf, store = metadata_luts(data)
    if not np.array_equal(leaf, np.load(source_root/'metadata/leaf.npy')) or \
            not np.array_equal(store, np.load(source_root/'metadata/store.npy')):
        raise ValueError('metadata lookups differ from the source study')
    fdr.save_arrays(root/'metadata', {'leaf': leaf, 'store': store})
    paths = [p for d in ('candidates', 'features', 'plans', 'split', 'metadata') if (root/d).exists()
             for p in (root/d).rglob('*') if p.is_file()]
    meta.update(status='prepared', inner_split=split_info, candidate_dirs=dirs,
                artifact_sha256={str(p.relative_to(root)): digest(p) for p in paths})
    write_json(root/'study.json', meta)
    verify(root)
    print(json.dumps({'prepared': str(root), **split_info}), flush=True)


def fit(root, seed, device):
    meta, source_root = verify(root)
    cfg = meta['config']
    if seed not in cfg['seeds']: raise ValueError('unregistered seed')
    if (root/'selection_lock.json').exists(): raise ValueError('cannot fit after Val lock')
    run = root/f'runs/{METHOD}/seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    archive, feature = fdr.arrays(candidate_dir(meta, 'ranker_train')), features_for(root, 'ranker_train')
    archive.pop('source_rows', None)
    tune = np.flatnonzero(np.load(root/'split/tune.npy'))
    max_epochs = 1 if meta['technical_smoke'] else cfg['max_epochs']
    write_json(run/'config.json', {**cfg, 'method': METHOD, 'pool': meta['pool'], 'seed': seed, 'device': device,
                                   'technical_smoke': meta['technical_smoke'], 'git_hash': meta['git_hash'],
                                   'protocol_sha256': meta['protocol_sha256']})
    model = fdr.model_for(meta, root, seed, device)
    initial = state_hash(model)
    control = fdr.read(source_root/f'runs/{METHOD}/seed{seed}/metrics.json')['initial_state_sha256']
    if initial != control: raise ValueError('initial state differs from the rebuilt P1 bce_text fit')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    found = np.isfinite(archive['retrieval_rank'][tune])
    data = fdr.data_for(meta)
    first = np.zeros(len(model.text), np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    cold = first[archive['target'][tune]] >= meta['split']['t_ranker_train']
    del data
    users = archive['user'][tune]
    (run/'inner').mkdir()
    history = []; best = -np.inf; selected_epoch = 0; stale = 0
    started = time.perf_counter()
    with np.load(root/f'plans/seed{seed}_inner.npz') as z: inner_rows, inner_cols = z['rows'], z['columns']
    for epoch in range(max_epochs+1):
        loss = None if epoch == 0 else fdr.train_epoch(model, optimizer, archive, feature, inner_rows[epoch-1],
                                                       inner_cols[epoch-1], METHOD, device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = fdr.predict(model, archive, feature, tune, METHOD, device, run/f'inner/epoch{epoch}_logits.npy')
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or
                           not np.array_equal(ranks, archive['retrieval_rank'][tune]) or
                           not np.array_equal(top, archive['top10'][tune])):
            raise ValueError('epoch-zero function differs from retrieval')
        values = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(values, users),
               'unseen_before_T0_ndcg10': average(values[cold], users[cold]) if cold.any() else None,
               'conditional_ndcg10': average(values[found], users[found])}
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        history.append(row); write_json(run/'inner_history.json', history)
        print(json.dumps({'pool': meta['pool'], 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if row['ndcg10'] > best:
            best = row['ndcg10']; selected_epoch = epoch; stale = 0
        elif epoch > 0: stale += 1
        if stale >= cfg['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected_epoch}.pth', map_location='cpu', weights_only=True), strict=True)
    replay = fdr.predict(model, archive, feature, tune, METHOD, device)[0]
    with np.load(run/f'inner/epoch{selected_epoch}_ranks.npz') as z:
        if not np.array_equal(replay, z['rank']): raise ValueError('selected inner replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = fdr.model_for(meta, root, seed, device)
    if state_hash(model) != initial: raise ValueError('refit initial state differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    with np.load(root/f'plans/seed{seed}_refit.npz') as z: refit_rows, refit_cols = z['rows'], z['columns']
    refit_log = []; started = time.perf_counter()
    for epoch in range(selected_epoch):
        loss = fdr.train_epoch(model, optimizer, archive, feature, refit_rows[epoch], refit_cols[epoch], METHOD, device)
        refit_log.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': meta['pool'], 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth')
    before = fdr.predict(model, archive, feature, tune[:128], METHOD, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    after = fdr.predict(model, archive, feature, tune[:128], METHOD, device)[2]
    if not np.array_equal(before, after): raise ValueError('portable checkpoint replay differs')
    write_json(run/'refit_history.json', refit_log)
    write_json(run/'metrics.json', {'status': 'fitted', 'method': METHOD, 'pool': meta['pool'], 'seed': seed,
        'technical_smoke': meta['technical_smoke'], 'selected_epoch': selected_epoch, 'inner_ndcg10': best,
        'initial_state_sha256': initial, 'p1_initial_state_sha256': control, 'initial_state_matches_p1': True,
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'portable_reload_exact': True,
        'epoch0_equals_retrieval_on_tune': True, 'test_predictions_loaded': False})
    print(json.dumps({'fitted': meta['pool'], 'seed': seed, 'selected_epoch': selected_epoch,
                      'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds}), flush=True)


def evaluate(root, device, seeds=None):
    meta, source_root = verify(root)
    seeds = list(meta['config']['seeds'] if seeds is None else seeds)
    lock = {}
    for seed in seeds:
        run = root/f'runs/{METHOD}/seed{seed}'
        record = fdr.read(run/'metrics.json')
        if record['status'] != 'fitted' or digest(run/'model.pth') != record['model_sha256']:
            raise ValueError('all registered fits must finish before outer evaluation')
        lock[f'{METHOD}/seed{seed}'] = record
    write_json(root/'selection_lock.json', lock)
    out = root/'evaluation'; out.mkdir(exist_ok=False)
    archive, feature = fdr.arrays(candidate_dir(meta, 'val')), features_for(root, 'val')
    archive.pop('source_rows', None)
    rows = np.arange(len(archive['inst']))
    # Epoch-0 check on the full outer population: the untrained ranker is the pool's retrieval order.
    model = fdr.model_for(meta, root, seeds[0], device)
    ranks, top, logits = fdr.predict(model, archive, feature, rows, METHOD, device)
    zero = {'seed': seeds[0], 'logits_equal_score': bool(np.array_equal(logits, archive['score'])),
            'ranks_equal_retrieval_rank': bool(np.array_equal(ranks, archive['retrieval_rank'])),
            'top10_equal': bool(np.array_equal(top, archive['top10'])),
            'ndcg10': average(M.ndcg_at(ranks, 10), archive['user'])}
    if not all(zero[k] for k in ('logits_equal_score', 'ranks_equal_retrieval_rank', 'top10_equal')):
        raise ValueError('epoch-zero Val function differs from retrieval')
    write_json(out/'epoch0_check.json', zero)
    del model, logits; gc.collect()
    for seed in seeds:
        run = root/f'runs/{METHOD}/seed{seed}'
        destination = out/METHOD/f'seed{seed}'; destination.mkdir(parents=True)
        model = fdr.model_for(meta, root, seed, device)
        model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
        before = state_hash(model); started = time.perf_counter()
        ranks, top, _ = fdr.predict(model, archive, feature, rows, METHOD, device, destination/'logits.npy')
        if state_hash(model) != before: raise ValueError('outer scoring modified parameters')
        rr, tt = ranks_and_top10(np.load(destination/'logits.npy', mmap_mode='r'), archive['cand'], archive['target'])
        if not np.array_equal(rr, ranks) or not np.array_equal(tt, top): raise ValueError('saved outer scores do not replay')
        np.savez_compressed(destination/'val_ranks.npz', inst=archive['inst'], user=archive['user'], rank=ranks, top10=top)
        write_json(destination/'provenance.json', {'weights_sha256': digest(run/'model.pth'),
            'logits_sha256': digest(destination/'logits.npy'), 'rank_sha256': digest(destination/'val_ranks.npz'),
            'score_replay_exact': True, 'parameters_unchanged': True, 'seconds': time.perf_counter()-started,
            'test_predictions_loaded': False})
        print(json.dumps({'evaluated': meta['pool'], 'seed': seed, 'ndcg10': average(M.ndcg_at(ranks, 10), archive['user'])}), flush=True)
        del model; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    write_json(out/'completion.json', {'status': 'complete', 'seeds': seeds, 'epoch0_check': zero,
                                     'selection_lock_sha256': digest(root/'selection_lock.json'),
                                     'test_predictions_loaded': False})


# ---------------------------------------------------------------- P3t gate
def tiebreak_scores(score):
    """P3t: RRF score minus 1e-6 x (0-based interleave position), float32."""
    score = np.asarray(score)
    return (score.astype(np.float64)-P3T_EPS*np.arange(score.shape[1])[None, :]).astype(np.float32)


def lexicographic_ranks(score, cand, target):
    """Pessimistic-free rank under (score desc, column asc): a pure tie-break by column order."""
    match = cand == target[:, None]; found = match.any(1); col = match.argmax(1)
    t = score[np.arange(len(score)), col]
    pos = np.arange(score.shape[1])[None, :]
    rank = (score > t[:, None]).sum(1)+((score == t[:, None]) & (pos <= col[:, None])).sum(1)
    return np.where(found, rank, np.inf)


def check_p3t():
    """Evaluate the registered P3t gate on Val and B; writes a JSON record only (no archive)."""
    out = REPO_ROOT/ANALYSIS/'p3t_gate.json'
    if out.exists(): raise ValueError('gate record exists')
    record = {'definition': 'P3t score = P3 RRF score - 1e-6 x interleave position (0-based)',
              'gate': f'P3t epoch-0 Val nDCG@10 must equal the P3 interleave-order value {INTERLEAVE_NDCG10}',
              'splits': {}}
    for split in SPLITS:
        d = REPO_ROOT/POOL_BASE/'candidates/P3_fused_rrf'/split
        cand, score = np.load(d/'cand.npy'), np.load(d/'score.npy')
        target, user = np.load(d/'target.npy'), np.load(d/'user.npy')
        inter = np.load(REPO_ROOT/POOL_BASE/'aux/P3_fused_rrf'/split/'interleave_rank.npy')
        st = tiebreak_scores(score)
        r, _ = ranks_and_top10(st, cand, target)
        lex = lexicographic_ranks(score, cand, target)
        order_t = np.argsort(-st, axis=1, kind='stable'); order_l = np.argsort(-score, axis=1, kind='stable')
        f = np.isfinite(r)
        record['splits'][split] = {
            'ndcg10_rrf_pessimistic': average(M.ndcg_at(np.load(d/'retrieval_rank.npy'), 10), user),
            'ndcg10_p3t': average(M.ndcg_at(r, 10), user),
            'ndcg10_interleave_order': average(M.ndcg_at(inter, 10), user),
            'ndcg10_rrf_lexicographic_tiebreak': average(M.ndcg_at(lex, 10), user),
            'natural_targets': int(f.sum()),
            'targets_rank_p3t_ne_interleave': int((r[f] != inter[f]).sum()),
            'targets_rank_p3t_ne_lexicographic': int((r[f] != lex[f]).sum()),
            'rows_full_order_p3t_eq_lexicographic': float((order_t == order_l).all(1).mean()),
            'rows_p3t_order_eq_interleave_order': float((np.diff(st, axis=1) < 0).all(1).mean())}
    v = record['splits']['val']
    record['gate_passed'] = bool(round(v['ndcg10_p3t'], 6) == INTERLEAVE_NDCG10)
    record['reason'] = ('RRF order differs from interleave order beyond ties (items in both sources move up), and the '
                        '1e-6 x position offset (up to 1.99e-4) also reorders distinct RRF values whose gap is smaller, '
                        'so P3t is neither the interleave order nor a pure tie-break.')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    print(json.dumps(record, indent=1), flush=True)
    return record


# ---------------------------------------------------------------- decomposition (no training)
CROSS = '20261106-pool-ranking-P1onP3'


def overlap_index(cand_a, cand_b):
    """For each slot of ``cand_a``, the column of the same item in ``cand_b`` (-1 if absent)."""
    eq = cand_a[:, :, None] == cand_b[:, None, :]
    return np.where(eq.any(2), eq.argmax(2), -1)


def cross_score(device):
    """Score the locked P1 bce_text checkpoints on the P3 Val lists, keeping the P1 input function.

    The skip score is the P1 retrieval function (90-day recency text mean, cosine) for every P3 candidate,
    and the nine common columns are re-standardised with the P1 scaler. No training. Overlapping
    (event, item) slots must reproduce the P1 archive's score and features.
    """
    from cascade.eval.harness import make_batch
    from cascade.pipelines.local_followups import TextMeanScorer
    root = study_path(CROSS); root.mkdir(parents=True, exist_ok=False)
    p3_root = study_path(study_name('P3'))
    p3_meta, source_root = verify(p3_root, artifacts=False, post_evaluation=True)
    source_meta = fdr.verify(source_root, artifacts=False)
    data = fdr.data_for(source_meta); inst = data.split_instances('val')
    p3 = fdr.arrays(candidate_dir(p3_meta, 'val')); p1 = fdr.arrays(source_root/'candidates/val')
    if not np.array_equal(p3['inst'], inst.inst.to_numpy()) or not np.array_equal(p1['inst'], p3['inst']):
        raise ValueError('event order differs')
    write_json(root/'protocol.json', {'purpose': 'decompose P3 ranker gain: P1-trained checkpoints on P3 Val candidates',
        'skip_score': 'P1 retrieval function (TextMeanScorer half-life 90 on the P1 frozen text vectors) for every P3 candidate',
        'features': 'P3 common columns un-standardised with the P3 scaler and re-standardised with the P1 scaler',
        'checkpoints': 'results/local_experiments/20261103-full-promising-ranking/runs/bce_text/seed*/model.pth (locked)',
        'training': 'none', 'label': 'Val only, exploratory', 'test_predictions_loaded': False})
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    shutil.copyfile(REPO_ROOT/SOURCES[0], root/'pool_ranking.py')
    n = len(inst)
    scorer = TextMeanScorer(data, np.load(REPO_ROOT/source_meta['text_path']), device, 90)
    score = open_memmap(root/'score.npy', mode='w+', dtype=np.float32, shape=(n, 200))
    for start in range(0, n, 128):
        stop = min(start+128, n)
        batch = make_batch(data, inst.iloc[start:stop], 50)
        with torch.no_grad():
            full = scorer.score_all(batch).numpy()
        score[start:stop] = np.take_along_axis(full, np.asarray(p3['cand'][start:stop]).astype(np.int64), axis=1)
        if start % (128*300) == 0: print(json.dumps({'cross_score': start, 'total': n}), flush=True)
    score.flush()
    m3 = fdr.read(p3_root/'features/manifest.json')['scaler']['common']
    m1 = fdr.read(source_root/'features/manifest.json')['scaler']['common']
    mean3, std3, mean1, std1 = (np.asarray(v) for v in (m3['mean'], m3['std'], m1['mean'], m1['std']))
    f3 = np.load(p3_root/'features/val/common.npy', mmap_mode='r')
    f1 = np.load(source_root/'features/val/common.npy', mmap_mode='r')
    common = open_memmap(root/'common.npy', mode='w+', dtype=np.float32, shape=(n, 200, COMMON))
    check = {'overlap_slots': 0, 'score_exact_slots': 0, 'score_max_abs_diff': 0.0, 'feature_max_abs_diff': 0.0}
    for start in range(0, n, 1024):
        stop = min(start+1024, n)
        common[start:stop] = (((f3[start:stop].astype(np.float64)*std3+mean3)-mean1)/std1).astype(np.float32)
        idx = overlap_index(np.asarray(p3['cand'][start:stop]), np.asarray(p1['cand'][start:stop]))
        r, c = np.nonzero(idx >= 0)
        a, b = score[start:stop][r, c], np.asarray(p1['score'][start:stop])[r, idx[r, c]]
        fa, fb = common[start:stop][r, c], np.asarray(f1[start:stop])[r, idx[r, c]]
        check['overlap_slots'] += len(r); check['score_exact_slots'] += int((a == b).sum())
        check['score_max_abs_diff'] = max(check['score_max_abs_diff'], float(np.abs(a-b).max(initial=0)))
        check['feature_max_abs_diff'] = max(check['feature_max_abs_diff'], float(np.abs(fa-fb).max(initial=0)))
    common.flush()
    if check['score_max_abs_diff'] > 1e-5 or check['feature_max_abs_diff'] > 1e-4:
        raise ValueError(f'cross inputs do not reproduce P1 on overlapping slots: {check}')
    archive = {'cand': p3['cand'], 'score': np.load(root/'score.npy', mmap_mode='r'), 'target': p3['target'],
               'history': p3['history']}
    feature = {'common': np.load(root/'common.npy', mmap_mode='r'), 'graph': ZeroGraph()}
    rows = np.arange(n)
    r0 = ranks_and_top10(np.asarray(archive['score']), np.asarray(p3['cand']), p3['target'])[0]
    np.savez_compressed(root/'recency_order_ranks.npz', inst=p3['inst'], user=p3['user'], rank=r0)
    # Replay guard: this scoring path reproduces the stored P1 outer logits on P1 lists.
    p1_feature = {'common': f1, 'graph': ZeroGraph()}
    lock = fdr.read(source_root/'selection_lock.json')
    for seed in source_meta['config']['seeds']:
        run = source_root/f'runs/{METHOD}/seed{seed}'
        if digest(run/'model.pth') != lock[f'{METHOD}/seed{seed}']['model_sha256']: raise ValueError('P1 checkpoint changed')
        model = fdr.model_for(source_meta, source_root, seed, device)
        model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
        guard = np.arange(2048)
        replay = fdr.predict(model, p1, p1_feature, guard, METHOD, device)[2]
        stored = np.load(source_root/f'evaluation/{METHOD}/seed{seed}/logits.npy', mmap_mode='r')[guard]
        check[f'seed{seed}_p1_replay_max_abs_diff'] = float(np.nanmax(np.abs(np.where(np.isfinite(stored), replay-stored, 0))))
        if check[f'seed{seed}_p1_replay_max_abs_diff'] > 1e-5: raise ValueError('P1 logits do not replay')
        dest = root/f'evaluation/seed{seed}'; dest.mkdir(parents=True)
        ranks, top, _ = fdr.predict(model, archive, feature, rows, METHOD, device, dest/'logits.npy')
        np.savez_compressed(dest/'val_ranks.npz', inst=p3['inst'], user=p3['user'], rank=ranks, top10=top)
        check[f'seed{seed}_ndcg10'] = average(M.ndcg_at(ranks, 10), p3['user'])
        print(json.dumps({'cross_scored': seed, 'ndcg10': check[f'seed{seed}_ndcg10']}), flush=True)
        del model; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    check['recency_order_on_P3_ndcg10'] = average(M.ndcg_at(r0, 10), p3['user'])
    write_json(root/'checks.json', {**check, 'test_predictions_loaded': False})
    print(json.dumps(check), flush=True)


# ---------------------------------------------------------------- report
METRICS = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall10': lambda r: M.recall_at(r, 10),
           'hr3': lambda r: M.recall_at(r, 3), 'mrr10': lambda r: np.where(r <= 10, 1/np.asarray(r, float), 0.)}
GROUPS = ('all', 'new_in_val', 'warm_in_val')
FUNNEL = ('retrieval_miss', 'unrescued', 'rescued', 'demoted', 'retained')


def funnel_masks(retrieval_rank, final_rank):
    found = np.isfinite(retrieval_rank); old = retrieval_rank <= 10; hit = final_rank <= 10
    if (hit & ~found).any(): raise ValueError('ranker hit outside its pool')
    masks = {'retrieval_miss': ~found, 'unrescued': found & ~old & ~hit, 'rescued': ~old & hit & found,
             'demoted': old & ~hit, 'retained': old & hit}
    if not np.all(np.stack(list(masks.values())).sum(0) == 1): raise ValueError('funnel does not partition population')
    return masks


def load_ranks(path, inst, users):
    with np.load(path) as z:
        if not np.array_equal(z['inst'], inst) or not np.array_equal(z['user'], users):
            raise ValueError(f'event alignment differs: {path}')
        return z['rank']


def report(pools):
    out = REPO_ROOT/ANALYSIS
    for name in ('metrics.csv', 'paired_differences.csv', 'funnel.csv', 'fits.csv', 'inner_curves.csv', 'tables.md'):
        if (out/name).exists(): raise ValueError(f'refusing to overwrite {name}')
    source_root = study_path(SOURCE)
    source_meta = fdr.verify(source_root, artifacts=False)
    data = fdr.data_for(source_meta); inst = data.split_instances('val'); users = inst.user.to_numpy()
    ids = inst.inst.to_numpy()
    groups = {k: v for k, v in fdr.groups_for(data, inst).items() if k in GROUPS}
    seeds = source_meta['config']['seeds']
    ranks, retrieval_of = {}, {}
    p1_ret = np.load(source_root/'candidates/val/retrieval_rank.npy')
    ranks['P1_retrieval'] = p1_ret[None]
    for s in seeds:
        prov = fdr.read(source_root/f'evaluation/{METHOD}/seed{s}/provenance.json')
        if digest(source_root/f'evaluation/{METHOD}/seed{s}/val_ranks.npz') != prov['rank_sha256']:
            raise ValueError('P1 evaluation changed')
    ranks['P1_ranker'] = np.stack([load_ranks(source_root/f'evaluation/{METHOD}/seed{s}/val_ranks.npz', ids, users) for s in seeds])
    retrieval_of['P1_ranker'] = p1_ret
    fits, curves = [], []
    for s in seeds:
        run = source_root/f'runs/{METHOD}/seed{s}'
        fits.append({**fdr.read(run/'metrics.json'), 'pool': 'P1'})
        curves += [{'pool': 'P1', 'seed': s, **r} for r in fdr.read(run/'inner_history.json')]
    epoch0 = {}
    for pool in pools:
        root = study_path(study_name(pool))
        meta, _ = verify(root, artifacts=False, post_evaluation=True)
        done = fdr.read(root/'evaluation/completion.json')
        if done['status'] != 'complete' or done['seeds'] != seeds or digest(root/'selection_lock.json') != done['selection_lock_sha256']:
            raise ValueError(f'{pool} evaluation incomplete')
        epoch0[pool] = done['epoch0_check']
        ret = np.load(candidate_dir(meta, 'val')/'retrieval_rank.npy')
        ranks[f'{pool}_retrieval'] = ret[None]
        ranks[f'{pool}_ranker'] = np.stack([load_ranks(root/f'evaluation/{METHOD}/seed{s}/val_ranks.npz', ids, users) for s in seeds])
        retrieval_of[f'{pool}_ranker'] = ret
        for s in seeds:
            run = root/f'runs/{METHOD}/seed{s}'
            fits.append({**fdr.read(run/'metrics.json'), 'pool': pool})
            curves += [{'pool': pool, 'seed': s, **r} for r in fdr.read(run/'inner_history.json')]
    pool_of = {m: m.split('_')[0] for m in ranks}
    if 'P3' in pools:
        ranks['P3_interleave_order'] = np.load(REPO_ROOT/POOL_BASE/'aux/P3_fused_rrf/val/interleave_rank.npy')[None]
        pool_of['P3_interleave_order'] = 'P3'
        cross = study_path(CROSS)
        if (cross/'checks.json').exists():
            ranks['P1ranker_on_P3'] = np.stack([load_ranks(cross/f'evaluation/seed{s}/val_ranks.npz', ids, users) for s in seeds])
            ranks['P3_recency_order'] = load_ranks(cross/'recency_order_ranks.npz', ids, users)[None]
            retrieval_of['P1ranker_on_P3'] = ranks['P3_retrieval'][0]
            pool_of.update(P1ranker_on_P3='P3', P3_recency_order='P3')
    found_of = {m.split('_')[0]: np.isfinite(v[0]) for m, v in ranks.items() if m.endswith('_retrieval')}
    if {'P1', 'P3'} <= set(found_of):
        both = found_of['P1'] & found_of['P3']
        for g in GROUPS:
            groups['both_'+g] = groups[g] & both
    contrasts = [('P1_retrieval', 'P1_ranker')]
    for pool in pools:
        contrasts += [(f'{pool}_retrieval', f'{pool}_ranker'), ('P1_ranker', f'{pool}_ranker')]
    if 'P3' in pools:
        contrasts += [('P3_interleave_order', 'P3_ranker')]
    if {'P2', 'P3'} <= set(pools):
        contrasts += [('P2_ranker', 'P3_ranker')]
    if 'P1ranker_on_P3' in ranks:
        contrasts += [('P1_ranker', 'P1ranker_on_P3'), ('P1ranker_on_P3', 'P3_ranker'), ('P3_recency_order', 'P1ranker_on_P3')]
    rows, diffs = [], []
    for metric, fn in METRICS.items():
        values = {m: np.stack([fn(r) for r in rs]) for m, rs in ranks.items()}
        for group, mask in groups.items():
            for m, v in values.items():
                seed_means = [average(x[mask], users[mask]) for x in v]
                fm = found_of[pool_of[m]] & mask
                row = {'group': group, 'system': m, 'metric': metric, 'events': int(mask.sum()),
                       'users': int(np.unique(users[mask]).size), 'value': float(np.mean(seed_means)),
                       'seed_std_ddof0': float(np.std(seed_means)),
                       **({f'seed{s}': x for s, x in zip(seeds, seed_means)} if len(v) > 1 else {})}
                if metric == 'ndcg10':
                    row['conditional_ndcg10'] = float(np.mean([average(x[fm], users[fm]) for x in v]))
                    row['natural_targets'] = int(fm.sum())
                rows.append(row)
            for ref, cand in contrasts:
                ci = paired_bootstrap(users[mask], values[ref].mean(0)[mask], values[cand].mean(0)[mask],
                                      n_resamples=source_meta['config']['bootstrap_resamples'],
                                      seed=source_meta['config']['bootstrap_seed'])
                diffs.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand,
                              'events': int(mask.sum()), **ci})
            print(json.dumps({'metric': metric, 'group': group, 'done': True}), flush=True)
    funnels = []
    for m, ret in retrieval_of.items():
        for si, r in enumerate(ranks[m]):
            masks = funnel_masks(ret, r)
            for group, gm in groups.items():
                funnels.append({'system': m, 'seed': seeds[si], 'group': group, 'events': int(gm.sum()),
                                **{k: int((v & gm).sum()) for k, v in masks.items()},
                                'top10_hits': int((r[gm] <= 10).sum())})
    for m in ranks:
        if m.endswith('_retrieval') or m == 'P3_interleave_order':
            r = ranks[m][0]
            for group, gm in groups.items():
                funnels.append({'system': m, 'seed': None, 'group': group, 'events': int(gm.sum()),
                                'retrieval_miss': int((~np.isfinite(r) & gm).sum()), 'top10_hits': int((r[gm] <= 10).sum())})
    frame, contrast, funnel = pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(funnels)
    fits, curves = pd.DataFrame(fits), pd.DataFrame(curves)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'metrics.csv', index=False); contrast.to_csv(out/'paired_differences.csv', index=False)
    funnel.to_csv(out/'funnel.csv', index=False); fits.to_csv(out/'fits.csv', index=False)
    curves.to_csv(out/'inner_curves.csv', index=False)
    get = lambda m, g, k: float(frame.loc[(frame.system == m) & (frame.group == g) & (frame.metric == k), 'value'].iloc[0])
    lines = ['# Fixed bce_text ranker on alternative pools (Val only, exploratory)', '', HYPOTHESIS, '',
             f'{len(ids):,} Val events / {len(np.unique(users)):,} users; per-user macro, misses = 0; ranker values are '
             'seed means (0/1/2). Intervals: 1,000 paired-user draws, seed 0, on seed-mean event metrics; unadjusted. '
             'No Test file was opened. Nothing is selected.', '',
             '## Systems', '', '| System | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 | cond. nDCG@10 | natural targets |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    show = [g for g in groups]
    for m in ranks:
        for g in show:
            c = frame.loc[(frame.system == m) & (frame.group == g) & (frame.metric == 'ndcg10')].iloc[0]
            lines.append(f'| {m} | {g} | ' + ' | '.join(f'{get(m, g, k):.6f}' for k in METRICS)
                         + f' | {c.conditional_ndcg10:.6f} | {int(c.natural_targets):,} |')
    lines += ['', '## Paired differences (candidate minus reference)', '',
              '| Contrast | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for ref, cand in contrasts:
        for g in show:
            cells = []
            for k in METRICS:
                d = contrast.loc[(contrast.reference == ref) & (contrast.candidate == cand) & (contrast.group == g) & (contrast.metric == k)].iloc[0]
                cells.append(f'{d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}]')
            lines.append(f'| {cand} − {ref} | {g} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Funnel (seed-mean event counts; each ranker against its own pool retrieval order)', '',
              '| Ranker | Group | events | retrieval miss | unrescued | rescued | demoted | retained | top-10 hits |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in retrieval_of:
        for g in GROUPS:
            sub = funnel.loc[(funnel.system == m) & (funnel.group == g)]
            lines.append(f'| {m} | {g} | {int(sub.events.iloc[0]):,} | ' + ' | '.join(f'{sub[k].mean():,.1f}' for k in (*FUNNEL, 'top10_hits')) + ' |')
    lines += ['', '## Fits', '', '| Pool | Seed | Selected epoch | Inner nDCG@10 by epoch (0, 1, ...) |', '|---|---:|---:|---|']
    for r in fits.itertuples():
        curve = curves.loc[(curves.pool == r.pool) & (curves.seed == r.seed)].sort_values('epoch')
        lines.append(f'| {r.pool} | {r.seed} | {r.selected_epoch} | ' + ', '.join(f'{v:.6f}' for v in curve.ndcg10) + ' |')
    lines += ['', '## Checks', '', f'Epoch-0 Val (untrained ranker = pool retrieval order): {json.dumps(epoch0)}',
              'Initial state hashes equal the rebuilt P1 bce_text fits for every seed (see fits.csv).', '']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'report_provenance.json', {'pools': list(pools), 'test_predictions_loaded': False,
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.suffix in ('.csv', '.md') and p.name != 'README.md'}})
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check_p3t', 'prepare', 'fit', 'evaluate', 'cross_score', 'report'))
    parser.add_argument('--pool', choices=tuple(POOLS), default='P3')
    parser.add_argument('--pools', nargs='+', choices=tuple(POOLS), default=list(POOLS))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(0, 1, 2))
    parser.add_argument('--smoke-rows', type=int, help='technical CPU smoke on a seeded slice (separate study)')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    if args.phase == 'check_p3t': return check_p3t()
    if args.phase == 'report': return report(args.pools)
    if args.phase == 'cross_score': return cross_score(str(pick_device(args.device)))
    root = study_path(study_name(args.pool, args.smoke_rows is not None))
    if args.phase == 'prepare': return prepare(args.pool, root, args.smoke_rows)
    if fdr.read(root/'study.json')['technical_smoke'] != (args.smoke_rows is not None):
        raise ValueError('study smoke flag mismatch')
    device = str(pick_device(args.device))
    if args.phase == 'fit': return fit(root, args.seed, device)
    if args.seeds and not fdr.read(root/'study.json')['technical_smoke']:
        raise ValueError('full evaluation scores every registered seed')
    evaluate(root, device, args.seeds)


if __name__ == '__main__':
    main()
