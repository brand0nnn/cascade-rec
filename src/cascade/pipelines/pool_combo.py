"""Additivity test: pool (P3) x period-start freshness (T1 common columns) x leak-safe context.

Val only, exploratory.  This is an additivity test, NOT the selection of a final pipeline.

Arms (all on the P3 fused pool, recipe = ``bce_text`` / ``bce_context`` unchanged, reused by import):

* ``T1`` (no training): the three locked P3 ``bce_text`` checkpoints score Val with the nine common columns
  rebuilt from all reviews strictly before T1 (``freshness_val.SnapshotFeatures``; query-user exclusion and
  cut-rule checks), standardised with the unchanged P3 prefix scaler.  Only natural-hit rows can change a
  target rank, so only they are rebuilt and rescored (CPU); misses keep rank infinity.
* ``context`` (training): ``context_ranking`` columns (63; period-start statistics T0 for B, T1 for Val; same
  leak checks) on the P3 pool with the P3 plans/split/common features; seeds 0/1/2; shared parameters start
  from the P3 (= P1) ``bce_text`` initial state, new head columns from the context arm's seeded draw.
  Evaluated with (a) common columns at T0 (as trained) and (b) common columns at T1 (all inputs at the
  period start).

Test data is never opened.
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

from cascade.analysis import freshness_val as fv
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import context_ranking as cr
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines import pool_ranking as pr
from cascade.pipelines.graph_latent_study import state_hash
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.ranking_objective_study import objective_loss  # noqa: F401  (recipe provenance)
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

LABEL = 'Val only, exploratory; additivity test, not a final-pipeline selection'
T1_STUDY, CTX_STUDY = '20261106-pool-combo-T1', '20261106-pool-combo-context'
ANALYSIS = 'results/analysis/20261106-pool-combo'
P3_STUDY = pr.study_name('P3')
CONTEXT_STUDY = cr.FULL_STUDY  # P1 bce_context
FRESH = '20261106-freshness-val'
FOLLOWUP = 'results/analysis/20261106-context/followup'
METHOD = cr.METHOD
SEEDS = (0, 1, 2)
HYPOTHESIS = ('Pre-stated (before any combo scoring or fitting): each factor keeps most of its gain when combined; '
              'the combined P3+context+T1 ranker beats the P3 ranker(T0) overall, with new-item quality recovering '
              'part of P3\'s new-item loss (context/newness helps new items).')


def p3_context():
    root = study_path(P3_STUDY)
    meta, source_root = pr.verify(root, artifacts=False, post_evaluation=True)
    return root, meta, source_root


class RowView:
    """Feature rows stored for a subset of events, indexed by original Val row numbers."""

    def __init__(self, values, rows, n):
        self.values = values
        self.pos = np.full(n, -1, np.int64); self.pos[rows] = np.arange(len(rows))

    def __getitem__(self, rows):
        idx = self.pos[np.asarray(rows)]
        if (idx < 0).any(): raise ValueError('row outside the stored subset')
        return self.values[idx]


# ---------------------------------------------------------------- arm 1: T1 common columns (no training)
def arm_t1(threads=3):
    torch.set_num_threads(threads)
    root = study_path(T1_STUDY); root.mkdir(parents=True, exist_ok=False)
    p3_root, p3_meta, source_root = p3_context()
    source_meta = fdr.verify(source_root, artifacts=False)
    data = fdr.data_for(source_meta); inst = data.split_instances('val')
    if data.split['t1'] != fv.T1 or data.split['t_ranker_train'] != fv.T0: raise ValueError('period starts differ')
    archive = fdr.arrays(pr.candidate_dir(p3_meta, 'val'))
    if not np.array_equal(archive['inst'], inst.inst.to_numpy()): raise ValueError('event order differs')
    hit = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    sub = {k: np.asarray(archive[k][hit]) for k in ('cand', 'score', 'history', 'target', 'retrieval_rank', 'top10')}
    scaler = fdr.read(p3_root/'features/manifest.json')['scaler']
    stored = np.asarray(np.load(p3_root/'features/val/common.npy', mmap_mode='r')[hit])
    write_json(root/'protocol.json', {'label': LABEL, 'arm': 'P3 bce_text checkpoints, common columns at T1',
        'snapshot': 'freshness_val.SnapshotFeatures(data, T1): all reviews strictly before T1',
        'scaler': 'P3 prefix scaler unchanged', 'rows': 'natural-hit Val rows only; misses keep rank infinity',
        'device': 'cpu', 'hypothesis': HYPOTHESIS, 'test_predictions_loaded': False})
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    shutil.copyfile(Path(__file__), root/'pool_combo.py')
    record = {'label': LABEL, 'hit_rows': int(len(hit))}
    # Own T0 snapshot builder reproduces the stored P3 T0 features.
    t0 = fv.SnapshotFeatures(data, fv.T0)
    check = np.sort(np.random.default_rng(20261106).choice(len(hit), 256, replace=False))
    raw = t0.build(inst.iloc[hit[check]], sub['cand'][check])[0]
    record['own_T0_builder_max_abs_error'] = float(np.max(np.abs(fv._standardise(raw, scaler, 'common')-stored[check])))
    record['own_T0_review_edges'] = t0.audit['review_edges']
    if record['own_T0_builder_max_abs_error'] > 2e-5: raise ValueError('own T0 builder does not reproduce P3 features')
    del t0; gc.collect()
    builder = fv.SnapshotFeatures(data, fv.T1)
    frame = inst.iloc[hit]
    record['cut_rule'] = fv.assert_cut_rule(builder, frame, sub['cand'])
    if builder.audit['max_review_timestamp'] >= inst.ts.min(): raise ValueError('snapshot overlaps a query')
    ref_rows = np.sort(np.random.default_rng(104729).choice(len(hit), 24, replace=False))
    fast = builder.build(frame.iloc[ref_rows], sub['cand'][ref_rows])
    ref = FrozenGraphFeatures.build(builder, frame.iloc[ref_rows], sub['cand'][ref_rows])
    record['reference_max_abs_error'] = [float(np.max(np.abs(a-b))) for a, b in zip(fast, ref)]
    if max(record['reference_max_abs_error']) > 2e-5: raise ValueError('batched snapshot differs from reference')
    common = np.zeros((len(hit), 200, pr.COMMON), np.float32)
    started = time.perf_counter()
    for s in range(0, len(hit), 128):
        common[s:s+128] = fv._standardise(builder.build(frame.iloc[s:s+128], sub['cand'][s:s+128])[0], scaler, 'common')
    record.update(snapshot=builder.audit, feature_seconds=time.perf_counter()-started,
                  nongraph_common_unchanged=bool(np.array_equal(common[..., 3:7], stored[..., 3:7])),
                  changed_slots_fraction=float(np.mean(np.any(common != stored, axis=-1))))
    if not record['nongraph_common_unchanged']: raise ValueError('non-graph common columns changed')
    del builder; gc.collect()
    (root/'features').mkdir()
    np.save(root/'features/common_T1.npy', common); np.save(root/'features/hit.npy', hit)
    lock = fdr.read(p3_root/'selection_lock.json')
    n = len(inst)
    for seed in SEEDS:
        run = p3_root/f'runs/bce_text/seed{seed}'
        if digest(run/'model.pth') != lock[f'bce_text/seed{seed}']['model_sha256']: raise ValueError('locked checkpoint changed')
        model = fdr.model_for(p3_meta, p3_root, seed, 'cpu')
        model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
        with np.load(p3_root/f'evaluation/bce_text/seed{seed}/val_ranks.npz') as z: saved = z['rank']
        out = {}
        for arm, feats in (('T0_cpu', stored), ('T1', common)):
            ranks = fdr.predict(model, sub, {'common': feats, 'graph': pr.ZeroGraph()}, np.arange(len(hit)), 'bce_text', 'cpu')[0]
            full = np.full(n, np.inf); full[hit] = ranks
            out[arm] = full
            (root/f'ranks/{arm}').mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root/f'ranks/{arm}/seed{seed}.npz', inst=archive['inst'], user=archive['user'], rank=full)
        record[f'seed{seed}'] = {'T0_cpu_equals_saved_mps': bool(np.array_equal(out['T0_cpu'], saved)),
                                 'T0_cpu_mismatches': int((out['T0_cpu'] != saved).sum()),
                                 'T0_cpu_top10_flips': int(((out['T0_cpu'] <= 10) != (saved <= 10)).sum()),
                                 'ndcg10_T0_saved': average(M.ndcg_at(saved, 10), archive['user']),
                                 'ndcg10_T1': average(M.ndcg_at(out['T1'], 10), archive['user'])}
        print(json.dumps({'seed': seed, **record[f'seed{seed}']}), flush=True)
        del model; gc.collect()
    write_json(root/'checks.json', {**record, 'test_predictions_loaded': False})


# ---------------------------------------------------------------- arm 2: context columns on P3
def view_root(root, p3_root, p3_meta):
    """Read-only symlink view so ``context_ranking.build_context`` reads the P3 candidates/split."""
    view = root/'view'
    for split in pr.SPLITS:
        (view/'candidates'/split).mkdir(parents=True)
        for key in ('cand', 'target', 'inst'):
            (view/'candidates'/split/f'{key}.npy').symlink_to(pr.candidate_dir(p3_meta, split)/f'{key}.npy')
    (view/'split').mkdir()
    (view/'split/prefix.npy').symlink_to(p3_root/'split/prefix.npy')
    return view


def ctx_verify(root):
    meta = fdr.read(root/'study.json')
    if meta['test_predictions_loaded']: raise ValueError('test exposure recorded')
    for name, expected in meta['artifact_sha256'].items():
        if digest(root/name) != expected: raise ValueError(f'context artifact changed: {name}')
    for name in ('src/cascade/pipelines/context_ranking.py', 'src/cascade/pipelines/full_data_ranking.py',
                 'src/cascade/models/graph_latent_ranker.py', 'src/cascade/data/ranker_features.py'):
        if digest(REPO_ROOT/name) != meta['source_sha256'][name]: raise ValueError(f'implementation changed: {name}')
    return meta


def prepare_context():
    root = study_path(CTX_STUDY); root.mkdir(parents=True, exist_ok=False)
    p3_root, p3_meta, source_root = p3_context()
    data = fdr.data_for(p3_meta)
    view = view_root(root, p3_root, p3_meta)
    built = cr.build_context(data, p3_meta, view)  # same columns, period-start tables and leak checks
    (root/'context').mkdir()
    for s, values in built['splits'].items():
        fdr.save_arrays(root/'context'/s, values)
    p1_ctx = fdr.read(study_path(CONTEXT_STUDY)/'context/manifest.json')
    if not np.allclose(built['cuts'], p1_ctx['price_cut_points']): raise ValueError('price cut points differ from P1 context')
    leak = built['leak']
    manifest = {'columns': cr.column_names(), 'price_cut_points': built['cuts'].tolist(),
                'scaler': {'mean': built['mean'].tolist(), 'std': built['std'].tolist()},
                'scaler_rule': 'one-hot over all 200 slots of chronological-prefix B queries of the P3 pool',
                'cutoff_proof': built['cutoff_proof'], 'new_candidate_leak_check': leak,
                'p1_same_ts_reference': cr.FULL_VAL_SAME_TS_TARGETS,
                'note': ('The 656 same-timestamp count is specific to the P1 lists. On P3 the same check runs '
                         '(every same-timestamp first-review candidate must be the target and be indistinguishable '
                         'from other new candidates); its count is recorded here.')}
    write_json(root/'context/manifest.json', manifest)
    sources = ('src/cascade/pipelines/pool_combo.py', 'src/cascade/pipelines/context_ranking.py',
               'src/cascade/pipelines/full_data_ranking.py', 'src/cascade/models/graph_latent_ranker.py',
               'src/cascade/data/ranker_features.py', 'src/cascade/pipelines/pool_ranking.py')
    (root/'source').mkdir()
    for name in sources: shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    write_json(root/'protocol.json', {'label': LABEL, 'method': METHOD, 'pool': 'P3', 'p3_study': P3_STUDY,
        'recipe': 'context_ranking bce_context (imported); P3 plans, split, common features', 'hypothesis': HYPOTHESIS,
        'initial_state': 'shared params = P3 bce_text initial state; new head columns generator 1000003+seed',
        'evaluation': '(a) Val common columns at T0 (as trained); (b) at T1 from 20261106-pool-combo-T1',
        'test_predictions_loaded': False})
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    paths = [p for p in (root/'context').rglob('*') if p.is_file()]
    write_json(root/'study.json', {'status': 'prepared', 'label': LABEL, 'p3_study': P3_STUDY, 'seeds': list(SEEDS),
        'source_sha256': {n: digest(REPO_ROOT/n) for n in sources},
        'artifact_sha256': {str(p.relative_to(root)): digest(p) for p in paths},
        'same_ts_candidates_val': leak['val']['same_ts_candidates'], 'test_predictions_loaded': False})
    print(json.dumps({'prepared': str(root), 'leak': {s: {k: v for k, v in r.items() if 'levels' not in k and 'codes' not in k}
                                                       for s, r in leak.items()}}), flush=True)


def fit_context(seed, device):
    root = study_path(CTX_STUDY); ctx_verify(root)
    p3_root, p3_meta, _ = p3_context()
    cfg = p3_meta['config']
    run = root/f'runs/{METHOD}/seed{seed}'; run.mkdir(parents=True, exist_ok=False)
    archive = fdr.arrays(pr.candidate_dir(p3_meta, 'ranker_train'))
    feature = {'common': np.load(p3_root/'features/ranker_train/common.npy', mmap_mode='r')}
    context = cr.load_context(root, 'ranker_train', device)
    tune = np.flatnonzero(np.load(p3_root/'split/tune.npy'))
    write_json(run/'config.json', {**cfg, 'method': METHOD, 'pool': 'P3', 'seed': seed, 'device': device,
                                   'common_dim': cr.BASE_COMMON+cr.CONTEXT_DIM})
    model, base_hash = cr.build_model(p3_meta, p3_root, seed, device)
    p3_initial = fdr.read(p3_root/f'runs/bce_text/seed{seed}/metrics.json')['initial_state_sha256']
    if base_hash != p3_initial: raise ValueError('shared initial state differs from the P3 bce_text fit')
    initial = state_hash(model)
    p1_ctx_initial = fdr.read(study_path(CONTEXT_STUDY)/f'runs/{METHOD}/seed{seed}/metrics.json')['initial_state_sha256']
    if initial != p1_ctx_initial: raise ValueError('full initial state differs from the P1 bce_context fit')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    found = np.isfinite(archive['retrieval_rank'][tune]); users = archive['user'][tune]
    (run/'inner').mkdir()
    history = []; best = -np.inf; selected = 0; stale = 0; started = time.perf_counter()
    with np.load(p3_root/f'plans/seed{seed}_inner.npz') as z: inner_rows, inner_cols = z['rows'], z['columns']
    for epoch in range(cfg['max_epochs']+1):
        loss = None if epoch == 0 else cr.train_epoch(model, optimizer, archive, feature, context,
                                                      inner_rows[epoch-1], inner_cols[epoch-1], device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = cr.predict(model, archive, feature, context, tune, device, run/f'inner/epoch{epoch}_logits.npy')
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or
                           not np.array_equal(ranks, archive['retrieval_rank'][tune])):
            raise ValueError('epoch-zero function differs from retrieval')
        values = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(values, users),
               'conditional_ndcg10': average(values[found], users[found])}
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        history.append(row); write_json(run/'inner_history.json', history)
        print(json.dumps({'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if row['ndcg10'] > best: best = row['ndcg10']; selected = epoch; stale = 0
        elif epoch > 0: stale += 1
        if stale >= cfg['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected}.pth', map_location='cpu', weights_only=True), strict=True)
    with np.load(run/f'inner/epoch{selected}_ranks.npz') as z:
        if not np.array_equal(cr.predict(model, archive, feature, context, tune, device)[0], z['rank']):
            raise ValueError('selected inner replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model, _ = cr.build_model(p3_meta, p3_root, seed, device)
    if state_hash(model) != initial: raise ValueError('refit initial state differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    with np.load(p3_root/f'plans/seed{seed}_refit.npz') as z: refit_rows, refit_cols = z['rows'], z['columns']
    refit_log = []; started = time.perf_counter()
    for epoch in range(selected):
        loss = cr.train_epoch(model, optimizer, archive, feature, context, refit_rows[epoch], refit_cols[epoch], device)
        refit_log.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    torch.save(model.state_dict(), run/'model.pth')
    before = cr.predict(model, archive, feature, context, tune[:128], device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    if not np.array_equal(before, cr.predict(model, archive, feature, context, tune[:128], device)[2]):
        raise ValueError('portable checkpoint replay differs')
    write_json(run/'refit_history.json', refit_log)
    write_json(run/'metrics.json', {'status': 'fitted', 'method': METHOD, 'pool': 'P3', 'seed': seed,
        'selected_epoch': selected, 'inner_ndcg10': best, 'initial_state_sha256': initial,
        'shared_initial_equals_P3_bce_text': True, 'initial_equals_P1_bce_context': True,
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds,
        'refit_seconds': time.perf_counter()-started, 'test_predictions_loaded': False})
    print(json.dumps({'fitted': seed, 'selected_epoch': selected}), flush=True)


def evaluate_context(device):
    root = study_path(CTX_STUDY); ctx_verify(root)
    p3_root, p3_meta, _ = p3_context()
    lock = {}
    for seed in SEEDS:
        rec = fdr.read(root/f'runs/{METHOD}/seed{seed}/metrics.json')
        if rec['status'] != 'fitted' or digest(root/f'runs/{METHOD}/seed{seed}/model.pth') != rec['model_sha256']:
            raise ValueError('all three fits must finish first')
        lock[f'{METHOD}/seed{seed}'] = rec
    write_json(root/'selection_lock.json', lock)
    out = root/'evaluation'; out.mkdir(exist_ok=False)
    archive = fdr.arrays(pr.candidate_dir(p3_meta, 'val'))
    n = len(archive['inst'])
    context = cr.load_context(root, 'val', device)
    t1_root = study_path(T1_STUDY)
    hit = np.load(t1_root/'features/hit.npy')
    if not np.array_equal(hit, np.flatnonzero(np.isfinite(archive['retrieval_rank']))): raise ValueError('hit rows differ')
    arms = {'common_T0': ({'common': np.load(p3_root/'features/val/common.npy', mmap_mode='r')}, np.arange(n)),
            'common_T1': ({'common': RowView(np.load(t1_root/'features/common_T1.npy', mmap_mode='r'), hit, n)}, hit)}
    record = {}
    for seed in SEEDS:
        model, _ = cr.build_model(p3_meta, p3_root, seed, device)
        model.load_state_dict(torch.load(root/f'runs/{METHOD}/seed{seed}/model.pth', map_location='cpu', weights_only=True), strict=True)
        before = state_hash(model)
        for arm, (feature, rows) in arms.items():
            ranks = cr.predict(model, archive, feature, context, rows, device)[0]
            full = np.full(n, np.inf); full[rows] = ranks
            (out/arm).mkdir(exist_ok=True)
            np.savez_compressed(out/arm/f'seed{seed}.npz', inst=archive['inst'], user=archive['user'], rank=full)
            record[f'{arm}/seed{seed}'] = average(M.ndcg_at(full, 10), archive['user'])
            print(json.dumps({'evaluated': arm, 'seed': seed, 'ndcg10': record[f'{arm}/seed{seed}']}), flush=True)
        if state_hash(model) != before: raise ValueError('scoring modified parameters')
        del model; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    write_json(out/'completion.json', {'status': 'complete', 'seeds': list(SEEDS), 'ndcg10': record,
        'selection_lock_sha256': digest(root/'selection_lock.json'), 'test_predictions_loaded': False})


# ---------------------------------------------------------------- report
METRICS = pr.METRICS
GROUPS = pr.GROUPS


def additivity(gains, observed):
    """Sum of individual gains vs the observed combined gain (all relative to one base)."""
    total = float(sum(gains.values()))
    return {'sum_individual': total, 'observed_combined': float(observed), 'interaction': float(observed-total),
            'observed_over_sum': float(observed/total) if total else None}


def report():
    out = REPO_ROOT/ANALYSIS
    for name in ('metrics.csv', 'paired_differences.csv', 'additivity.csv', 'tables.md'):
        if (out/name).exists(): raise ValueError(f'refusing to overwrite {name}')
    p3_root, p3_meta, source_root = p3_context()
    source_meta = fdr.verify(source_root, artifacts=False)
    data = fdr.data_for(source_meta); inst = data.split_instances('val')
    ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
    groups = {k: v for k, v in fdr.groups_for(data, inst).items() if k in GROUPS}
    load = lambda p: pr.load_ranks(p, ids, users)

    def stack(paths): return np.stack([load(p) for p in paths])
    ctx_root, t1_root = study_path(CTX_STUDY), study_path(T1_STUDY)
    fresh = study_path(FRESH)
    with np.load(REPO_ROOT/FOLLOWUP/'ranks_consistent_T1.npz') as z:
        if not np.array_equal(z['inst'], ids): raise ValueError('follow-up alignment')
        p1_ctx_t1 = z['rank']
    with np.load(REPO_ROOT/FOLLOWUP/'ranks_full.npz') as z:
        p1_ctx_t0_cpu = z['rank']
    ranks = {
        'P1_bce_text_T0': stack([source_root/f'evaluation/bce_text/seed{s}/val_ranks.npz' for s in SEEDS]),
        'P1_bce_text_T1': stack([fresh/f'ranks/T1/bce_text/seed{s}.npz' for s in SEEDS]),
        'P1_context_T0': stack([study_path(CONTEXT_STUDY)/f'evaluation/{METHOD}/seed{s}/val_ranks.npz' for s in SEEDS]),
        'P1_context_T1': p1_ctx_t1,
        'P3_bce_text_T0': stack([p3_root/f'evaluation/bce_text/seed{s}/val_ranks.npz' for s in SEEDS]),
        'P3_bce_text_T0_cpu': stack([t1_root/f'ranks/T0_cpu/seed{s}.npz' for s in SEEDS]),
        'P3_bce_text_T1': stack([t1_root/f'ranks/T1/seed{s}.npz' for s in SEEDS]),
        'P3_context_T0': stack([ctx_root/f'evaluation/common_T0/seed{s}.npz' for s in SEEDS]),
        'P3_context_T1': stack([ctx_root/f'evaluation/common_T1/seed{s}.npz' for s in SEEDS]),
    }
    contrasts = [('P3_bce_text_T0', 'P3_bce_text_T1'), ('P3_bce_text_T0_cpu', 'P3_bce_text_T1'),
                 ('P3_bce_text_T0', 'P3_context_T0'), ('P3_bce_text_T0', 'P3_context_T1'),
                 ('P3_context_T0', 'P3_context_T1'), ('P3_bce_text_T1', 'P3_context_T1'),
                 ('P1_bce_text_T0', 'P3_bce_text_T1'), ('P1_bce_text_T0', 'P3_context_T0'),
                 ('P1_bce_text_T0', 'P3_context_T1'), ('P1_context_T1', 'P3_context_T1'),
                 ('P1_context_T0', 'P3_context_T0'), ('P1_bce_text_T0', 'P1_bce_text_T1'),
                 ('P1_bce_text_T1', 'P1_context_T1'), ('P1_bce_text_T0', 'P3_bce_text_T0')]
    rows, diffs, values = [], [], {}
    for metric, fn in METRICS.items():
        values[metric] = {m: np.stack([fn(r) for r in rs]).mean(0) for m, rs in ranks.items()}
        for group, mask in groups.items():
            for m, rs in ranks.items():
                seed_means = [average(fn(r)[mask], users[mask]) for r in rs]
                rows.append({'group': group, 'system': m, 'metric': metric, 'events': int(mask.sum()),
                             'value': float(np.mean(seed_means)), 'seed_std_ddof0': float(np.std(seed_means)),
                             **{f'seed{s}': v for s, v in zip(SEEDS, seed_means)}})
            for ref, cand in contrasts:
                ci = paired_bootstrap(users[mask], values[metric][ref][mask], values[metric][cand][mask], n_resamples=1000, seed=0)
                diffs.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand, **ci})
        print(json.dumps({'metric': metric, 'done': True}), flush=True)
    frame, contrast = pd.DataFrame(rows), pd.DataFrame(diffs)
    get = lambda m, g, k: float(frame.loc[(frame.system == m) & (frame.group == g) & (frame.metric == k), 'value'].iloc[0])
    dif = lambda a, b, g, k: contrast.loc[(contrast.reference == a) & (contrast.candidate == b) & (contrast.group == g) & (contrast.metric == k)].iloc[0]
    add_rows = []
    for metric in METRICS:
        for g in GROUPS:
            base = get('P1_bce_text_T0', g, metric)
            gain = lambda m: get(m, g, metric)-base
            # Individual factors measured on P1 (or pool on bce_text at T0), each from the same base.
            pool = gain('P3_bce_text_T0'); fresh_g = gain('P1_bce_text_T1')
            context_t1 = get('P1_context_T1', g, metric)-get('P1_bce_text_T1', g, metric)
            for combo, factors, observed in (
                    ('pool + freshness', {'pool': pool, 'freshness': fresh_g}, gain('P3_bce_text_T1')),
                    ('pool + context (Val common T0)', {'pool': pool, 'context_T0common': gain('P1_context_T0')}, gain('P3_context_T0')),
                    ('pool + freshness + context (all T1)', {'pool': pool, 'freshness': fresh_g, 'context_at_T1': context_t1},
                     gain('P3_context_T1'))):
                add_rows.append({'metric': metric, 'group': g, 'combination': combo, 'base': 'P1_bce_text_T0',
                                 **{f'gain_{k}': v for k, v in factors.items()}, **additivity(factors, observed)})
    adds = pd.DataFrame(add_rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'metrics.csv', index=False); contrast.to_csv(out/'paired_differences.csv', index=False)
    adds.to_csv(out/'additivity.csv', index=False)
    fits = [{**fdr.read(ctx_root/f'runs/{METHOD}/seed{s}/metrics.json'),
             'curve': [r['ndcg10'] for r in fdr.read(ctx_root/f'runs/{METHOD}/seed{s}/inner_history.json')]} for s in SEEDS]
    pd.DataFrame(fits).to_csv(out/'fits.csv', index=False)
    lines = [f'# Pool x freshness x context additivity ({LABEL})', '', HYPOTHESIS, '',
             f'{len(ids):,} Val events / {len(np.unique(users)):,} users; per-user macro, misses = 0; seed means (0/1/2). '
             'Intervals: 1,000 paired-user draws, seed 0, on seed-mean event metrics; unadjusted. No Test file opened.', '',
             '## Systems', '', '| System | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for m in ranks:
        for g in GROUPS:
            lines.append(f'| {m} | {g} | ' + ' | '.join(f'{get(m, g, k):.6f}' for k in METRICS) + ' |')
    lines += ['', '## Paired differences (candidate − reference) [95% CI]', '',
              '| Contrast | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for a, b in contrasts:
        for g in GROUPS:
            lines.append(f'| {b} − {a} | {g} | ' + ' | '.join(
                f"{dif(a, b, g, k)['diff']:+.6f} [{dif(a, b, g, k).ci_low:+.6f}, {dif(a, b, g, k).ci_high:+.6f}]" for k in METRICS) + ' |')
    lines += ['', '## Additivity (nDCG@10; gains relative to P1 bce_text at T0)', '',
              '| Combination | Group | individual gains | sum | observed | observed − sum | observed / sum |', '|---|---|---|---:|---:|---:|---:|']
    for r in adds.loc[adds.metric == 'ndcg10'].itertuples():
        parts = ', '.join(f'{c[5:]} {getattr(r, c):+.6f}' for c in adds.columns if c.startswith('gain_') and pd.notna(getattr(r, c)))
        lines.append(f'| {r.combination} | {r.group} | {parts} | {r.sum_individual:+.6f} | {r.observed_combined:+.6f} | '
                     f'{r.interaction:+.6f} | {r.observed_over_sum:.2f} |')
    lines += ['', '## Context-on-P3 fits', '', '| Seed | Selected epoch | Inner nDCG@10 by epoch |', '|---:|---:|---|']
    for f in fits:
        lines.append(f"| {f['seed']} | {f['selected_epoch']} | " + ', '.join(f'{v:.6f}' for v in f['curve']) + ' |')
    lines += ['', 'Checks: `checks_T1.json` (arm 1), `context_manifest.json` (leak checks), `completion.json`.',
              f'P1 context at Val common T0 uses the saved MPS ranks; the CPU rescore in the follow-up differs on '
              f'{int((p1_ctx_t0_cpu != ranks["P1_context_T0"]).sum())} seed-event ranks.', '']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    shutil.copyfile(t1_root/'checks.json', out/'checks_T1.json')
    shutil.copyfile(ctx_root/'context/manifest.json', out/'context_manifest.json')
    shutil.copyfile(ctx_root/'evaluation/completion.json', out/'completion.json')
    write_json(out/'report_provenance.json', {'label': LABEL, 'test_predictions_loaded': False,
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.suffix in ('.csv', '.md', '.json') and p.name != 'README.md'}})
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('arm_t1', 'prepare_context', 'fit', 'evaluate', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    if args.phase == 'arm_t1': return arm_t1(args.threads)
    if args.phase == 'prepare_context': return prepare_context()
    if args.phase == 'report': return report()
    device = str(pick_device(args.device))
    if args.phase == 'fit': return fit_context(args.seed, device)
    evaluate_context(device)


if __name__ == '__main__':
    main()
