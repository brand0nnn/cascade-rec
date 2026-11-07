"""Random mixing of missed B targets into their natural K200 lists (Val only, exploratory).

TRAINING-LABEL CHANGE in an isolated study, pending team review; never replaces natural-list results. Selects nothing.

Question: forced lists let rankers recognise appended positives by low retrieval score / last position. If a missed
target instead replaces the candidate at a uniformly random rank r in [1, 200] (``default_rng([seed, inst])``) and
takes that candidate's retrieval score, does the shortcut disappear, and do the extra positives help or hurt?

The ``bce_text`` recipe is imported unchanged from ``full_data_ranking`` (GraphLatentRanker R2, graph columns
disabled, BCE, 50 uniform real non-target negatives, Adam 1e-3, 32 lists, <=10 epochs, patience 2, chronological
inner selection on the NATURAL tuning lists with epoch 0 eligible, fixed-epoch all-B refit, identical initial state).
Mixed lists are virtual rows appended after the natural B archive (``MixedView``): the stored score array is reused
as is, the target replaces one candidate ID, and its nine common columns are rebuilt with the same pre-T0
``BatchedGraphFeatures`` builder and the P1 scaler. Arms: M1 = all mixed lists; M2 = one fixed seeded subsample of
20,676 mixed lists. Val stays natural. Test is never opened.

    PYTHONPATH=src python -m cascade.pipelines.random_mix --phase prepare
    ... --phase diagnose
    ... --phase plans --arm M1          # only if M1 is run
    ... --phase fit --arm M2 --seed 0 --device mps
    ... --phase evaluate --arm M2 --device mps
    ... --phase report
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
from torch.nn import functional as F

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines import pool_ranking as PR
from cascade.pipelines.graph_latent_study import negative_plan, state_hash
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

LABEL = ('Val only, exploratory; TRAINING-LABEL CHANGE in an isolated study, pending team review; '
         'never replaces natural-list results')
METHOD = 'bce_text'
SOURCE = '20261103-full-promising-ranking'
STUDY = '20261106-random-mix'
ANALYSIS = 'results/analysis/20261106-random-mix'
ARMS = ('M1', 'M2')
M2_SIZE, M2_SEED = 20676, 20261106
COMMON = 9
SOURCES = ('src/cascade/pipelines/random_mix.py', *fdr.SOURCES, 'src/cascade/pipelines/pool_ranking.py')
FROZEN_FUNCTIONS = ('MixedView', 'insertion_slots', 'training_views', 'arm_rows', 'build_plans',
                    'prepare', 'fit', 'evaluate')
USHAPE_LISTS, USHAPE_SEED = 50000, 0
PR_USED = ('ZeroGraph', 'features_for', 'p1_feature_crosscheck', 'function_sources', 'funnel_masks', 'load_ranks')


# ---------------------------------------------------------------- mixed rows
class MixedView:
    """Rows ``[0, n)`` are the base array; row ``n + j`` is base row ``src[j]`` with ``value[j]`` placed at column
    ``slot[j]`` (or an unchanged copy when ``slot`` is None). Read-only; returns copies."""

    def __init__(self, base, src, slot=None, value=None):
        self.base, self.src = base, np.asarray(src, np.int64)
        self.slot = None if slot is None else np.asarray(slot, np.int64)
        self.value = None if value is None else np.asarray(value)
        if self.slot is not None and (len(self.slot) != len(self.src) or len(self.value) != len(self.src)):
            raise ValueError('mixed slot/value arrays must align with source rows')
        self.n = len(base)
        self.shape = (self.n+len(self.src), *base.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        scalar = np.ndim(rows) == 0
        rows = np.atleast_1d(np.asarray(rows, np.int64))
        if (rows < 0).any() or (rows >= self.shape[0]).any():
            raise IndexError('mixed view row out of range')
        mixed = rows >= self.n
        j = np.where(mixed, rows-self.n, 0)
        source = np.where(mixed, self.src[j] if len(self.src) else 0, rows)
        out = np.array(self.base[source])
        if self.slot is not None and mixed.any():
            out[np.flatnonzero(mixed), self.slot[j[mixed]]] = self.value[j[mixed]]
        return out[0] if scalar else out


def insertion_slots(inst_ids, seed, n_real):
    """0-based slot = r - 1 with r uniform in [1, n_real] from ``default_rng([seed, inst])``."""
    return np.asarray([np.random.default_rng([int(seed), int(i)]).integers(0, int(n)) for i, n in zip(inst_ids, n_real)],
                      np.int16)


def training_views(root, seed, source_root=None):
    """Archive/feature dicts over natural B rows followed by every mixed row (seed-specific insertion slots)."""
    source_root = source_root or study_path(SOURCE)
    base = fdr.arrays(source_root/'candidates/ranker_train')
    mix = fdr.arrays(root/'mixed')
    src, slot = np.asarray(mix['src_rows']), np.asarray(mix[f'slot_seed{seed}'])
    archive = {'cand': MixedView(base['cand'], src, slot, base['target'][src]),
               'score': MixedView(base['score'], src), 'history': MixedView(base['history'], src),
               'target': MixedView(base['target'], src), 'inst': base['inst'], 'user': base['user'],
               'retrieval_rank': base['retrieval_rank'], 'top10': base['top10']}
    common = np.load(source_root/'features/ranker_train/common.npy', mmap_mode='r')
    feature = {'common': MixedView(common, src, slot, np.asarray(mix['target_common'])), 'graph': PR.ZeroGraph()}
    return archive, feature


def arm_rows(root, arm):
    """Indices j into the mixed rows used by an arm (M1 all, M2 the fixed seeded subsample)."""
    n = len(np.load(root/'mixed/src_rows.npy', mmap_mode='r'))
    if arm == 'M1':
        return np.arange(n)
    if arm == 'M2':
        return np.load(root/'mixed/m2_subset.npy')
    raise ValueError('unregistered arm')


def build_plans(root, arm, seeds, source_root=None):
    source_root = source_root or study_path(SOURCE)
    prefix = np.load(source_root/'split/prefix.npy')
    found = np.isfinite(np.load(source_root/'candidates/ranker_train/retrieval_rank.npy'))
    src = np.load(root/'mixed/src_rows.npy')
    chosen = arm_rows(root, arm)
    n = len(found)
    counts = {}
    for seed in seeds:
        archive, _ = training_views(root, seed, source_root)
        for stage in ('inner', 'refit'):
            path = root/f'plans/{arm}_seed{seed}_{stage}.npz'
            if path.exists(): raise ValueError(f'refusing to overwrite {path}')
            natural = np.flatnonzero(prefix & found) if stage == 'inner' else np.flatnonzero(found)
            mixed = chosen[prefix[src[chosen]]] if stage == 'inner' else chosen
            eligible = np.r_[natural, n+mixed]
            started = time.perf_counter()
            rows, columns = negative_plan(archive, eligible, seed, epochs=fdr.read(root/'study.json')['config']['max_epochs'])
            np.savez_compressed(path, rows=rows, columns=columns)
            counts[f'seed{seed}_{stage}'] = {'natural_lists': int(len(natural)), 'mixed_lists': int(len(mixed)),
                                             'seconds': time.perf_counter()-started}
            print(json.dumps({'plan': arm, 'seed': seed, 'stage': stage, **counts[f'seed{seed}_{stage}']}), flush=True)
    return counts


# ---------------------------------------------------------------- provenance
def verify(root, artifacts=True):
    meta = fdr.read(root/'study.json')
    if meta['test_predictions_loaded'] or meta['method'] != METHOD:
        raise ValueError('test exposure or method changed')
    for name, expected in meta['source_sha256'].items():
        frozen = root/'source'/Path(name).name
        if digest(frozen) != expected:
            raise ValueError(f'frozen source copy changed: {name}')
        if digest(REPO_ROOT/name) != expected:
            # This module may gain report code, pool_ranking may gain phases: only the functions used must match.
            used = FROZEN_FUNCTIONS if name == SOURCES[0] else PR_USED if name == SOURCES[-1] else None
            same = used is not None and PR.function_sources(REPO_ROOT/name, used) == PR.function_sources(frozen, used) \
                and len(PR.function_sources(frozen, used)) == len(used)
            if not same:
                raise ValueError(f'implementation changed: {name}')
    if digest(root/'protocol.json') != meta['protocol_sha256']:
        raise ValueError('registered protocol changed')
    if artifacts:
        for name, expected in meta.get('artifact_sha256', {}).items():
            if digest(root/name) != expected:
                raise ValueError(f'artifact changed: {name}')
    return meta


def prepare(root):
    source_root = study_path(SOURCE)
    source_meta = fdr.verify(source_root, artifacts=False)
    if source_meta['sample']: raise ValueError('full-data source expected')
    used = [n for n in source_meta['artifact_sha256'] if n.startswith(('candidates/', 'split/', 'metadata/'))
            or n in ('features/manifest.json', 'features/ranker_train/common.npy', 'features/val/common.npy')]
    for name in used:  # every parent artifact this study reads (graph blocks are unused by bce_text)
        if digest(source_root/name) != source_meta['artifact_sha256'][name]: raise ValueError(f'parent artifact changed: {name}')
    if 'features/ranker_train/common.npy' not in used: raise ValueError('parent artifact list differs')
    cfg = source_meta['config']
    data = fdr.data_for(source_meta)
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    for name in SOURCES:
        shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    protocol = {'label': LABEL, 'method': METHOD, 'source_study': SOURCE, 'arms': list(ARMS),
                'mixing': 'missed B target replaces the candidate at uniform rank r in [1, n_real] '
                          '(default_rng([seed, inst])); target takes that candidate score; replaced candidate dropped',
                'features': 'target common columns from BatchedGraphFeatures (pre-T0 graph), P1 scaler; text/leaf/'
                            'store/cosine computed in-model from IDs',
                'M1': 'natural + all mixed lists', 'M2': f'natural + fixed subsample of {M2_SIZE} mixed lists '
                      f'(default_rng({M2_SEED})), shared by all seeds',
                'inner': 'prefix natural + prefix mixed of the arm; selection on natural tune lists only',
                'refit': 'all-B natural + all mixed of the arm, fixed selected epochs, identical initial state',
                'initial_state': 'must equal the rebuilt P1 bce_text initial_state_sha256 per seed',
                'val': 'natural P1 Val lists and features; once after the arm locks', 'selection': 'none',
                'm1_rule': 'if one M1 fit would exceed 25 min on mps, M2 first and M1 reported as not run',
                'test_predictions_loaded': False}
    write_json(root/'protocol.json', protocol)
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    meta = {'status': 'preparing', 'method': METHOD, 'sample': False, 'source_study': SOURCE,
            'config': {**cfg, 'methods': [METHOD]}, 'split': data.split, 'text_path': source_meta['text_path'],
            'git_hash': git_hash(), 'source_sha256': {n: digest(REPO_ROOT/n) for n in SOURCES},
            'protocol_sha256': digest(root/'protocol.json'), 'test_predictions_loaded': False}
    write_json(root/'study.json', meta)
    inst = data.split_instances('ranker_train')
    arc = fdr.arrays(source_root/'candidates/ranker_train')
    for key, col in (('inst', 'inst'), ('user', 'user'), ('target', 'item')):
        if not np.array_equal(arc[key], inst[col].to_numpy()): raise ValueError(f'B archive {key} misaligned')
    found = np.isfinite(arc['retrieval_rank'])
    if int(found.sum()) != source_meta['inner_split']['all_B_natural']: raise ValueError('natural B count differs')
    src = np.flatnonzero(~found)
    cand, target = np.asarray(arc['cand'][src]), np.asarray(arc['target'][src])
    n_real = (cand > 0).sum(1)
    if (cand == target[:, None]).any(): raise ValueError('a missed target is in its list')
    if (n_real < 51).any(): raise ValueError('fewer than 51 real candidates')
    first = np.zeros(data.n_items+1, np.int64); first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    if (first[target] > inst.ts.to_numpy()[src]).any(): raise ValueError('unreleased target')
    del cand
    builder = BatchedGraphFeatures(data)
    scaler = fdr.read(source_root/'features/manifest.json')['scaler']['common']
    mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    proof = {'builder_audit': builder.audit, 'p1_crosscheck': PR.p1_feature_crosscheck(builder, data, source_root)}
    # Single-column builds equal the stored P1 feature of the same (event, item): per-candidate independence.
    nat = np.sort(np.random.default_rng(11).choice(np.flatnonzero(found), 512, replace=False))
    col = (np.asarray(arc['cand'][nat]) == np.asarray(arc['target'][nat])[:, None]).argmax(1)
    single = builder.build(inst.iloc[nat], np.asarray(arc['target'][nat])[:, None])[0][:, 0]
    stored = np.load(source_root/'features/ranker_train/common.npy', mmap_mode='r')[nat][np.arange(len(nat)), col]
    rebuilt = ((single.astype(np.float64)-mean)/std).astype(np.float32)
    proof['single_column_vs_stored_natural_target'] = {'rows': len(nat), 'max_abs_error': float(np.abs(rebuilt-stored).max())}
    if not np.allclose(rebuilt, stored, rtol=1e-5, atol=1e-5): raise ValueError('single-column build differs')
    target_common = np.empty((len(src), COMMON), np.float32)
    started = time.perf_counter()
    for start in range(0, len(src), 2048):
        rows = src[start:start+2048]
        value = builder.build(inst.iloc[rows], target[start:start+2048, None])[0][:, 0]
        target_common[start:start+2048] = ((value.astype(np.float64)-mean)/std).astype(np.float32)
        if start % (2048*40) == 0:
            print(json.dumps({'target_features': start, 'total': len(src), 'seconds': time.perf_counter()-started}), flush=True)
    if not np.isfinite(target_common).all(): raise ValueError('nonfinite target features')
    slots = {f'slot_seed{s}': insertion_slots(arc['inst'][src], s, n_real) for s in cfg['seeds']}
    for key, value in slots.items():
        if (value < 0).any() or (value >= n_real).any(): raise ValueError('insertion slot outside real candidates')
    subset = np.sort(np.random.default_rng(M2_SEED).choice(len(src), M2_SIZE, replace=False))
    fdr.save_arrays(root/'mixed', {'src_rows': src, 'target_common': target_common, 'n_real': n_real,
                                   'm2_subset': subset, **slots})
    prefix = np.load(source_root/'split/prefix.npy')
    # Structural check of the virtual rows for every seed on a seeded slice.
    check_rows = np.sort(np.random.default_rng(3).choice(len(src), 4096, replace=False))
    for s in cfg['seeds']:
        archive, feature = training_views(root, s, source_root)
        c = archive['cand'][len(found)+check_rows]
        if not np.array_equal((c == target[check_rows, None]).sum(1), np.ones(len(check_rows))):
            raise ValueError('mixed row must contain exactly one target')
        if not np.array_equal((c == target[check_rows, None]).argmax(1), slots[f'slot_seed{s}'][check_rows]):
            raise ValueError('target not at its slot')
        if not np.array_equal(archive['score'][len(found)+check_rows], np.asarray(arc['score'][src[check_rows]])):
            raise ValueError('mixed scores differ from the natural score array')
        o = np.sort(c, axis=1)
        if ((np.diff(o, axis=1) == 0) & (o[:, 1:] > 0)).any(): raise ValueError('duplicate candidate in mixed row')
        f = feature['common'][len(found)+check_rows]
        if not np.array_equal(f[np.arange(len(check_rows)), slots[f'slot_seed{s}'][check_rows]], target_common[check_rows]):
            raise ValueError('target features not at the slot')
    counts = {'B_lists': int(len(found)), 'natural_B': int(found.sum()), 'mixed_lists': int(len(src)),
              'mixed_prefix': int(prefix[src].sum()), 'mixed_tune': int((~prefix[src]).sum()),
              'natural_prefix': int((prefix & found).sum()), 'natural_tune': int((~prefix & found).sum()),
              'M2_mixed': int(len(subset)), 'M2_mixed_prefix': int(prefix[src[subset]].sum()),
              'M2_mixed_tune': int((~prefix[src[subset]]).sum()), 'padded_mixed_lists': int((n_real < 200).sum()),
              'mean_slot_seed0_1based': float(slots['slot_seed0'].mean()+1)}
    proof['counts'] = counts
    write_json(root/'mixed_manifest.json', proof)
    print(json.dumps(counts), flush=True)
    (root/'plans').mkdir()
    meta.update(status='mixed_built', counts=counts)
    write_json(root/'study.json', meta)
    meta['plan_counts'] = {'M2': build_plans(root, 'M2', cfg['seeds'], source_root)}
    paths = [p for d in ('mixed', 'plans') for p in (root/d).rglob('*') if p.is_file()]
    meta.update(status='prepared', artifact_sha256={str(p.relative_to(root)): digest(p) for p in paths})
    write_json(root/'study.json', meta)
    verify(root)


def plans_phase(root, arm):
    meta = verify(root)
    meta.setdefault('plan_counts', {})[arm] = build_plans(root, arm, meta['config']['seeds'])
    paths = [p for p in (root/'plans').glob(f'{arm}_seed*.npz')]
    meta['artifact_sha256'].update({str(p.relative_to(root)): digest(p) for p in paths})
    write_json(root/'study.json', meta)


# ---------------------------------------------------------------- fit / evaluate
def fit(root, arm, seed, device, max_minutes=None):
    meta = verify(root)
    cfg = meta['config']
    source_root = study_path(SOURCE)
    if seed not in cfg['seeds'] or arm not in ARMS: raise ValueError('unregistered fit')
    if (root/f'evaluation/{arm}').exists(): raise ValueError('cannot fit after the arm was evaluated')
    if not (root/f'plans/{arm}_seed{seed}_inner.npz').exists(): raise ValueError('plans missing')
    run = root/f'runs/{arm}/seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    archive, feature = training_views(root, seed, source_root)
    tune = np.flatnonzero(np.load(source_root/'split/tune.npy'))
    write_json(run/'config.json', {**cfg, 'method': METHOD, 'arm': arm, 'seed': seed, 'device': device,
                                   'git_hash': meta['git_hash'], 'protocol_sha256': meta['protocol_sha256'], 'label': LABEL})
    model = fdr.model_for(meta, source_root, seed, device)
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
    with np.load(root/f'plans/{arm}_seed{seed}_inner.npz') as z: inner_rows, inner_cols = z['rows'], z['columns']
    for epoch in range(cfg['max_epochs']+1):
        t0 = time.perf_counter()
        loss = None if epoch == 0 else fdr.train_epoch(model, optimizer, archive, feature, inner_rows[epoch-1],
                                                       inner_cols[epoch-1], METHOD, device)
        train_seconds = time.perf_counter()-t0
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = fdr.predict(model, archive, feature, tune, METHOD, device, run/f'inner/epoch{epoch}_logits.npy')
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or
                           not np.array_equal(ranks, archive['retrieval_rank'][tune]) or
                           not np.array_equal(top, archive['top10'][tune])):
            raise ValueError('epoch-zero function differs from retrieval')
        values = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(values, users),
               'unseen_before_T0_ndcg10': average(values[cold], users[cold]),
               'conditional_ndcg10': average(values[found], users[found]), 'train_seconds': train_seconds,
               'epoch_seconds': time.perf_counter()-t0}
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        history.append(row); write_json(run/'inner_history.json', history)
        print(json.dumps({'arm': arm, 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if max_minutes and epoch == 1:
            projected = (row['epoch_seconds']*cfg['max_epochs']+row['train_seconds']*cfg['max_epochs']*
                         (len(np.load(root/f'plans/{arm}_seed{seed}_refit.npz')['rows'][0])/len(inner_rows[0])))/60
            write_json(run/'projection.json', {'projected_minutes_upper': projected, 'limit_minutes': max_minutes})
            if projected > max_minutes:
                raise SystemExit(f'projected fit {projected:.1f} min exceeds {max_minutes} min; stopping (not run)')
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
    model = fdr.model_for(meta, source_root, seed, device)
    if state_hash(model) != initial: raise ValueError('refit initial state differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    with np.load(root/f'plans/{arm}_seed{seed}_refit.npz') as z: refit_rows, refit_cols = z['rows'], z['columns']
    refit_log = []; started = time.perf_counter()
    for epoch in range(selected_epoch):
        loss = fdr.train_epoch(model, optimizer, archive, feature, refit_rows[epoch], refit_cols[epoch], METHOD, device)
        refit_log.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': arm, 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth')
    before = fdr.predict(model, archive, feature, tune[:128], METHOD, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    after = fdr.predict(model, archive, feature, tune[:128], METHOD, device)[2]
    if not np.array_equal(before, after): raise ValueError('portable checkpoint replay differs')
    write_json(run/'refit_history.json', refit_log)
    write_json(run/'metrics.json', {'status': 'fitted', 'method': METHOD, 'arm': arm, 'seed': seed,
        'selected_epoch': selected_epoch, 'inner_ndcg10': best, 'initial_state_sha256': initial,
        'p1_initial_state_sha256': control, 'initial_state_matches_p1': True,
        'inner_lists_per_epoch': int(len(inner_rows[0])), 'refit_lists_per_epoch': int(len(refit_rows[0])),
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'portable_reload_exact': True,
        'epoch0_equals_retrieval_on_tune': True, 'device': device, 'test_predictions_loaded': False})
    print(json.dumps({'fitted': arm, 'seed': seed, 'selected_epoch': selected_epoch,
                      'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds}), flush=True)


def evaluate(root, arm, device):
    meta = verify(root, artifacts=False)
    source_root = study_path(SOURCE)
    seeds = meta['config']['seeds']
    lock = {}
    for seed in seeds:
        run = root/f'runs/{arm}/seed{seed}'
        record = fdr.read(run/'metrics.json')
        if record['status'] != 'fitted' or digest(run/'model.pth') != record['model_sha256']:
            raise ValueError('all seeds of the arm must finish before Val scoring')
        lock[f'{arm}/seed{seed}'] = record
    out = root/'evaluation'/arm; out.mkdir(parents=True, exist_ok=False)
    write_json(out/'selection_lock.json', lock)
    archive = fdr.arrays(source_root/'candidates/val')
    feature = PR.features_for(source_root, 'val')
    rows = np.arange(len(archive['inst']))
    # Guard: this scoring path reproduces the stored P1 outer logits.
    model = fdr.model_for(meta, source_root, seeds[0], device)
    model.load_state_dict(torch.load(source_root/f'runs/{METHOD}/seed{seeds[0]}/model.pth', map_location='cpu',
                                     weights_only=True), strict=True)
    replay = fdr.predict(model, archive, feature, rows[:2048], METHOD, device)[2]
    stored = np.load(source_root/f'evaluation/{METHOD}/seed{seeds[0]}/logits.npy', mmap_mode='r')[:2048]
    guard = float(np.nanmax(np.abs(np.where(np.isfinite(stored), replay-stored, 0))))
    if guard > 1e-5: raise ValueError('P1 logits do not replay on this path')
    del model; gc.collect()
    for seed in seeds:
        run = root/f'runs/{arm}/seed{seed}'
        destination = out/f'seed{seed}'; destination.mkdir()
        model = fdr.model_for(meta, source_root, seed, device)
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
        print(json.dumps({'evaluated': arm, 'seed': seed, 'ndcg10': average(M.ndcg_at(ranks, 10), archive['user'])}), flush=True)
        del model; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    write_json(out/'completion.json', {'status': 'complete', 'seeds': seeds, 'p1_replay_max_abs_diff': guard,
                                     'selection_lock_sha256': digest(out/'selection_lock.json'),
                                     'test_predictions_loaded': False})


# ---------------------------------------------------------------- diagnostics (CPU, no training)
def list_auc(values, positive_col, valid):
    """Per-list AUC of the positive against the list's other valid candidates (ties count one half)."""
    values = np.asarray(values, np.float64)
    pos = values[np.arange(len(values)), positive_col][:, None]
    neg = valid.copy(); neg[np.arange(len(values)), positive_col] = False
    greater = ((pos > values) & neg).sum(1); ties = ((pos == values) & neg).sum(1)
    return (greater+0.5*ties)/neg.sum(1)


def head_inputs(text, leaf, store, cand, history):
    """Numeric head inputs exactly as ``GraphLatentRanker.forward`` computes them (torch, CPU)."""
    c, h = torch.as_tensor(np.asarray(cand, np.int64)), torch.as_tensor(np.asarray(history, np.int64))
    valid = h.ne(0)
    cl, ll = leaf[c], leaf[h[:, -1]]
    leaf_known = cl.gt(0) & ll[:, None].gt(0)
    leaf_match = leaf_known & cl.eq(ll[:, None])
    cs, hs = store[c], store[h]
    store_known = cs.gt(0) & (hs.gt(0) & valid).any(1)[:, None]
    store_match = store_known & ((cs[:, :, None] == hs[:, None, :]) & valid[:, None, :]).any(-1)
    raw = (text[h]*valid.unsqueeze(-1)).sum(1)/valid.sum(1, keepdim=True).clamp_min(1)
    cosine = F.cosine_similarity(text[c], raw[:, None, :], dim=-1)
    return {'leaf_match': leaf_match.numpy(), 'leaf_known': leaf_known.numpy(), 'store_match': store_match.numpy(),
            'store_known': store_known.numpy(), 'cosine_to_history': cosine.numpy()}


def diagnose(root):
    meta = verify(root)
    source_root = study_path(SOURCE)
    out = REPO_ROOT/ANALYSIS
    if (out/'diagnostics.json').exists(): raise ValueError('diagnostics exist')
    names = fdr.read(source_root/'features/manifest.json')['common_names']
    text = torch.as_tensor(np.load(REPO_ROOT/meta['text_path']))
    leaf = torch.as_tensor(np.load(source_root/'metadata/leaf.npy'), dtype=torch.long)
    store = torch.as_tensor(np.load(source_root/'metadata/store.npy'), dtype=torch.long)
    archive, feature = training_views(root, 0, source_root)
    n = archive['cand'].n
    natural = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    mixed = n+arm_rows(root, 'M2')
    rows_all = []
    per_list = {}
    for label, rows in (('natural', natural), ('inserted', mixed)):
        aucs, means = {}, {}
        for start in range(0, len(rows), 256):
            idx = rows[start:start+256]
            cand, score = archive['cand'][idx], archive['score'][idx]
            common = feature['common'][idx]
            valid = cand > 0
            pcol = (cand == archive['target'][idx][:, None]).argmax(1)
            hi = head_inputs(text, leaf, store, cand, archive['history'][idx])
            groups = {'skip_score': score, **hi, 'score_minus_cosine': score-hi['cosine_to_history'],
                      **{f'common{i}_{nm}': common[..., i] for i, nm in enumerate(names)}}
            for key, value in groups.items():
                value = np.asarray(value, np.float64)
                aucs.setdefault(key, []).append(list_auc(value, pcol, valid))
                negm = valid.copy(); negm[np.arange(len(idx)), pcol] = False
                means.setdefault(key, []).append(np.c_[value[np.arange(len(idx)), pcol],
                                                       (value*negm).sum(1)/negm.sum(1)])
            aucs.setdefault('_slot', []).append(pcol+1)
        slot = np.concatenate(aucs.pop('_slot'))
        per_list[label] = {k: np.concatenate(v) for k, v in aucs.items()}
        for key, v in per_list[label].items():
            m = np.concatenate(means[key])
            rows_all.append({'positives': label, 'input': key, 'lists': len(v), 'mean_auc': float(v.mean()),
                             'auc_q25': float(np.quantile(v, .25)), 'auc_q75': float(np.quantile(v, .75)),
                             'positive_mean': float(m[:, 0].mean()), 'negative_mean': float(m[:, 1].mean())})
        per_list[label]['slot'] = slot
    frame = pd.DataFrame(rows_all)
    # Rank-decile view of the cosine gap for inserted positives: by insertion rank.
    ins = per_list['inserted']
    dec = (ins['slot']-1)//20
    by_rank = pd.DataFrame({'insertion_rank_decile': dec+1, 'cosine_auc': ins['cosine_to_history'],
                            'score_auc': ins['skip_score'], 'score_minus_cosine_auc': ins['score_minus_cosine']}
                           ).groupby('insertion_rank_decile').mean().reset_index()
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'diagnostics_auc.csv', index=False)
    by_rank.to_csv(out/'diagnostics_inserted_by_rank_decile.csv', index=False)
    get = lambda p, k: float(frame.loc[(frame.positives == p) & (frame.input == k), 'mean_auc'].iloc[0])
    h2 = {'inserted_cosine_auc': get('inserted', 'cosine_to_history'),
          'inserted_score_minus_cosine_auc': get('inserted', 'score_minus_cosine'),
          'natural_cosine_auc': get('natural', 'cosine_to_history'),
          'natural_score_minus_cosine_auc': get('natural', 'score_minus_cosine'),
          'inserted_skip_score_auc': get('inserted', 'skip_score')}
    h2['h2_diagnostic_condition'] = bool(h2['inserted_cosine_auc'] < .40 or h2['inserted_score_minus_cosine_auc'] > .60)
    write_json(out/'diagnostics.json', {'label': LABEL, 'natural_lists': int(len(natural)), 'inserted_lists': int(len(mixed)),
        'inserted_source': 'M2 subsample, seed-0 insertion slots', 'auc': 'per-list positive vs the other real '
        'candidates of its own list (ties 1/2), mean over lists', 'h2': h2, 'test_predictions_loaded': False})
    print(frame.to_string(), flush=True); print(by_rank.to_string(), flush=True); print(json.dumps(h2), flush=True)


# ---------------------------------------------------------------- report (Val)
def ushape_checks(per_seed, arms):
    groups = ('head', 'tail', 'new')
    result = {}
    for arm in arms:
        d = per_seed[per_seed['arm'] == arm]
        piv = d.pivot_table(index=['seed', 'group'], columns='decile', values='mean_centred_logit')
        rise = {g: (piv.xs(g, level='group')[10]-piv.xs(g, level='group')[6]).tolist() for g in groups}
        contrast = {}
        for g in ('tail', 'new'):
            diff = piv.xs(g, level='group')-piv.xs('head', level='group')
            contrast[g] = {'mean_over_deciles_per_seed': diff.mean(axis=1).tolist(),
                           'deciles_positive_on_seed_mean': int((diff.mean(axis=0) > 0).sum())}
        result[arm] = {'low_rank_rise_d10_minus_d6_per_seed': rise,
                       'low_rank_rise_holds_all_groups': bool(all(min(v) > 0 for v in rise.values())),
                       'unpopular_minus_head': contrast,
                       'unpopular_preference_holds': bool(all(min(c['mean_over_deciles_per_seed']) > 0 and
                                                              c['deciles_positive_on_seed_mean'] >= 8 for c in contrast.values()))}
    return result


def ushape(data, archive, logits_paths, out):
    from cascade.analysis import ushape_popularity as U
    lab, _, info = U.candidate_groups(data)
    n = len(archive['inst'])
    rows = np.sort(np.random.default_rng(USHAPE_SEED).choice(n, USHAPE_LISTS, replace=False))
    cand = np.asarray(archive['cand'][rows]); score = np.asarray(archive['score'][rows])
    valid = cand > 0
    if (np.diff(np.where(valid, score, -np.inf), axis=1) > 0).any(): raise ValueError('P1 Val columns not in score order')
    ret_rank = np.broadcast_to(np.arange(1, 201), valid.shape)
    groups = lab[cand]
    hits = np.isfinite(np.asarray(archive['retrieval_rank'])[rows])
    frames = []
    for (arm, seed), path in logits_paths.items():
        lg = np.asarray(np.load(path, mmap_mode='r')[rows])
        for subset, mask in (('all_lists', np.ones(len(rows), bool)), ('natural_hit_lists', hits)):
            t = U.decile_table(lg[mask], groups[mask], valid[mask], ret_rank[mask])
            t.insert(0, 'subset', subset); t.insert(0, 'seed', seed); t.insert(0, 'arm', arm)
            frames.append(t)
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed.to_csv(out/'ushape_deciles_per_seed.csv', index=False)
    seed_mean = (per_seed.groupby(['subset', 'arm', 'decile', 'ranks', 'group'], sort=False)
                 .agg(share_mean=('share', 'mean'), share_sd=('share', 'std'),
                      mean_centred_logit_mean=('mean_centred_logit', 'mean'),
                      mean_centred_logit_sd=('mean_centred_logit', 'std'), n_seeds=('seed', 'nunique')).reset_index())
    seed_mean.to_csv(out/'ushape_deciles_seed_mean.csv', index=False)
    arms = list(dict.fromkeys(a for a, _ in logits_paths))
    checks = {s: ushape_checks(per_seed[per_seed.subset == s], arms) for s in ('all_lists', 'natural_hit_lists')}
    primary = seed_mean[seed_mean.subset == 'all_lists']
    U.composition_decomposition(primary).to_csv(out/'ushape_composition_decomposition.csv', index=False)
    from cascade.analysis.plot_findings import AQUA, BLUE, GREY, ORANGE, _save, _style
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style()
    colours = {'head': BLUE, 'tail': ORANGE, 'new': AQUA}
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2*len(arms), 3.9), sharey=True)
    for ax, arm in zip(np.atleast_1d(axes), arms):
        d = primary[primary.arm == arm]
        for g in ('head', 'tail', 'new'):
            s = d[d.group == g].sort_values('decile')
            ax.errorbar(s.decile, s.mean_centred_logit_mean, yerr=s.mean_centred_logit_sd, color=colours[g],
                        marker='o', markersize=4, capsize=2, label=g)
        s = d[d.group == 'all'].sort_values('decile')
        ax.plot(s.decile, s.mean_centred_logit_mean, color=GREY, linestyle='--', linewidth=1.2, label='all')
        ax.axhline(0, color=GREY, linewidth=.8)
        ax.set(xticks=range(1, 11), xlabel='Retrieval-rank decile (1 = ranks 1-20)', title=arm)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel('Mean logit, centred within list')
    _save(fig, out, 'fig_ushape_random_mix', primary, f'{LABEL}. {USHAPE_LISTS:,} seeded natural Val K200 lists '
          '(rng 0); mean ± SD over seeds 0-2; head/tail/new from pre-T1 counts (ushape_popularity.candidate_groups).')
    return {'lists': USHAPE_LISTS, 'users': int(np.unique(np.asarray(archive['user'])[rows]).size),
            'natural_hit_lists': int(hits.sum()), 'candidate_groups': info, 'checks': checks}


def report(root):
    meta = verify(root, artifacts=False)
    source_root = study_path(SOURCE)
    source_meta = fdr.verify(source_root, artifacts=False)
    out = REPO_ROOT/ANALYSIS
    for name in ('metrics.csv', 'paired_differences.csv', 'funnel.csv', 'tables.md', 'results.json'):
        if (out/name).exists(): raise ValueError(f'refusing to overwrite {name}')
    data = fdr.data_for(source_meta); inst = data.split_instances('val')
    users, ids = inst.user.to_numpy(), inst.inst.to_numpy()
    groups = {k: v for k, v in fdr.groups_for(data, inst).items() if k in PR.GROUPS}
    seeds = meta['config']['seeds']
    archive = fdr.arrays(source_root/'candidates/val')
    ret = np.asarray(archive['retrieval_rank'])
    ranks = {'retrieval': ret[None]}
    logits_paths = {}
    for s in seeds:
        prov = fdr.read(source_root/f'evaluation/{METHOD}/seed{s}/provenance.json')
        if digest(source_root/f'evaluation/{METHOD}/seed{s}/val_ranks.npz') != prov['rank_sha256']:
            raise ValueError('P1 evaluation changed')
        logits_paths[('P1_bce_text', s)] = source_root/f'evaluation/{METHOD}/seed{s}/logits.npy'
    ranks['P1_bce_text'] = np.stack([PR.load_ranks(source_root/f'evaluation/{METHOD}/seed{s}/val_ranks.npz', ids, users)
                                     for s in seeds])
    fits = [{**fdr.read(source_root/f'runs/{METHOD}/seed{s}/metrics.json'), 'arm': 'P1_bce_text'} for s in seeds]
    curves = [{'arm': 'P1_bce_text', 'seed': s, **r} for s in seeds
              for r in fdr.read(source_root/f'runs/{METHOD}/seed{s}/inner_history.json')]
    done = []
    for arm in ARMS:
        comp = root/f'evaluation/{arm}/completion.json'
        if not comp.exists(): continue
        c = fdr.read(comp)
        if c['status'] != 'complete' or digest(root/f'evaluation/{arm}/selection_lock.json') != c['selection_lock_sha256']:
            raise ValueError(f'{arm} evaluation incomplete')
        for s in seeds:
            prov = fdr.read(root/f'evaluation/{arm}/seed{s}/provenance.json')
            if digest(root/f'evaluation/{arm}/seed{s}/val_ranks.npz') != prov['rank_sha256']: raise ValueError('ranks changed')
            logits_paths[(arm, s)] = root/f'evaluation/{arm}/seed{s}/logits.npy'
            fits.append({**fdr.read(root/f'runs/{arm}/seed{s}/metrics.json'), 'arm': arm})
            curves += [{'arm': arm, 'seed': s, **r} for r in fdr.read(root/f'runs/{arm}/seed{s}/inner_history.json')]
        ranks[arm] = np.stack([PR.load_ranks(root/f'evaluation/{arm}/seed{s}/val_ranks.npz', ids, users) for s in seeds])
        done.append(arm)
    if not done: raise ValueError('no arm evaluated')
    contrasts = [('retrieval', 'P1_bce_text')] + [('P1_bce_text', a) for a in done] + [('retrieval', a) for a in done]
    if set(ARMS) <= set(done): contrasts.append(('M2', 'M1'))
    found = np.isfinite(ret)
    rows, diffs = [], []
    for metric, fn in PR.METRICS.items():
        values = {m: np.stack([fn(r) for r in rs]) for m, rs in ranks.items()}
        for group, mask in groups.items():
            for m, v in values.items():
                seed_means = [average(x[mask], users[mask]) for x in v]
                row = {'group': group, 'system': m, 'metric': metric, 'events': int(mask.sum()),
                       'users': int(np.unique(users[mask]).size), 'value': float(np.mean(seed_means)),
                       'seed_std_ddof0': float(np.std(seed_means)),
                       **({f'seed{s}': x for s, x in zip(seeds, seed_means)} if len(v) > 1 else {})}
                if metric == 'ndcg10':
                    fm = found & mask
                    row['conditional_ndcg10'] = float(np.mean([average(x[fm], users[fm]) for x in v]))
                rows.append(row)
            for ref, cand in contrasts:
                ci = paired_bootstrap(users[mask], values[ref].mean(0)[mask], values[cand].mean(0)[mask],
                                      n_resamples=meta['config']['bootstrap_resamples'], seed=meta['config']['bootstrap_seed'])
                diffs.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand,
                              'events': int(mask.sum()), **ci})
        print(json.dumps({'metric': metric, 'done': True}), flush=True)
    funnels = []
    for m in [k for k in ranks if k != 'retrieval']:
        for si, r in enumerate(ranks[m]):
            masks = PR.funnel_masks(ret, r)
            for group, gm in groups.items():
                funnels.append({'system': m, 'seed': seeds[si], 'group': group, 'events': int(gm.sum()),
                                **{k: int((v & gm).sum()) for k, v in masks.items()}, 'top10_hits': int((r[gm] <= 10).sum())})
    frame, contrast, funnel = pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(funnels)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'metrics.csv', index=False); contrast.to_csv(out/'paired_differences.csv', index=False)
    funnel.to_csv(out/'funnel.csv', index=False); pd.DataFrame(fits).to_csv(out/'fits.csv', index=False)
    pd.DataFrame(curves).to_csv(out/'inner_curves.csv', index=False)
    shape = ushape(data, archive, logits_paths, out)
    get = lambda m, g, k: float(frame.loc[(frame.system == m) & (frame.group == g) & (frame.metric == k), 'value'].iloc[0])
    ci = lambda r, c, g, k: contrast.loc[(contrast.reference == r) & (contrast.candidate == c) & (contrast.group == g)
                                         & (contrast.metric == k)].iloc[0]
    diag = fdr.read(out/'diagnostics.json')
    verdict = {}
    for arm in done:
        no_rise = not shape['checks']['all_lists'][arm]['low_rank_rise_holds_all_groups']
        new_up = ci('P1_bce_text', arm, 'new_in_val', 'ndcg10').ci_low > 0
        overall_down = ci('P1_bce_text', arm, 'all', 'ndcg10').ci_high < 0
        verdict[arm] = {'H1_no_low_rank_rise': no_rise, 'H1_new_ndcg10_up': bool(new_up), 'H1_holds': bool(no_rise and new_up),
                        'H2_diagnostic': diag['h2']['h2_diagnostic_condition'], 'H2_overall_ndcg10_down': bool(overall_down),
                        'H2_holds': bool(diag['h2']['h2_diagnostic_condition'] and overall_down)}
    lines = ['# Random mixing of missed B targets — Val results', '', f'**{LABEL}.** Nothing is selected. No Test file opened.', '',
             f'{len(ids):,} Val events / {len(np.unique(users)):,} users; natural P1 K200 lists; per-user macro, misses = 0; '
             'ranker values are seed means (0/1/2). Intervals: 1,000 paired-user draws, seed 0, on seed-mean event metrics; '
             'unadjusted.', '', '| System | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 | cond. nDCG@10 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for m in ranks:
        for g in groups:
            cond = frame.loc[(frame.system == m) & (frame.group == g) & (frame.metric == 'ndcg10'), 'conditional_ndcg10'].iloc[0]
            lines.append(f'| {m} | {g} | '+' | '.join(f'{get(m, g, k):.6f}' for k in PR.METRICS)+f' | {cond:.6f} |')
    lines += ['', '| Contrast | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for r, c in contrasts:
        for g in groups:
            cells = [f'{d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}]' for d in (ci(r, c, g, k) for k in PR.METRICS)]
            lines.append(f'| {c} − {r} | {g} | '+' | '.join(cells)+' |')
    lines += ['', '| Ranker | Group | retrieval miss | unrescued | rescued | demoted | retained | top-10 hits |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for m in [k for k in ranks if k != 'retrieval']:
        for g in groups:
            sub = funnel.loc[(funnel.system == m) & (funnel.group == g)]
            lines.append(f'| {m} | {g} | '+' | '.join(f'{sub[k].mean():,.1f}' for k in (*PR.FUNNEL, 'top10_hits'))+' |')
    lines += ['', '| Arm | Seed | Selected epoch | Inner lists/epoch | Refit lists/epoch | Inner nDCG@10 by epoch |', '|---|---:|---:|---:|---:|---|']
    cf = pd.DataFrame(curves)
    for f in fits:
        curve = cf.loc[(cf.arm == f['arm']) & (cf.seed == f['seed'])].sort_values('epoch')
        lines.append(f"| {f['arm']} | {f['seed']} | {f['selected_epoch']} | {f.get('inner_lists_per_epoch', 16819)} | "
                     f"{f.get('refit_lists_per_epoch', 20676)} | "+', '.join(f'{v:.6f}' for v in curve.ndcg10)+' |')
    lines += ['', '## U-shape (mean within-list-centred logit by retrieval-rank decile; all lists)', '',
              '| Arm | Group | ' + ' | '.join(f'd{d}' for d in range(1, 11)) + ' | d10−d6 per seed |', '|---|---|' + '---:|'*11]
    sm = pd.read_csv(out/'ushape_deciles_seed_mean.csv')
    for arm in dict.fromkeys(a for a, _ in logits_paths):
        for g in ('all', 'head', 'tail', 'new'):
            s = sm[(sm.subset == 'all_lists') & (sm.arm == arm) & (sm.group == g)].sort_values('decile')
            rise = shape['checks']['all_lists'][arm]['low_rank_rise_d10_minus_d6_per_seed'].get(g)
            lines.append(f'| {arm} | {g} | '+' | '.join(f'{v:+.3f}' for v in s.mean_centred_logit_mean)
                         + f" | {', '.join(f'{v:+.3f}' for v in rise) if rise else '—'} |")
    lines += ['', f'Verdict: {json.dumps(verdict)}', '']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'results.json', {'label': LABEL, 'arms_evaluated': done, 'arms_not_run': [a for a in ARMS if a not in done],
        'diagnostics_h2': diag['h2'], 'ushape': shape, 'verdict': verdict, 'test_predictions_loaded': False,
        'git_hash': git_hash()})
    write_json(out/'report_provenance.json', {'study': str(root.relative_to(REPO_ROOT)), 'test_predictions_loaded': False,
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.suffix in ('.csv', '.md', '.json', '.png')
                          and p.name not in ('README.md', 'report_provenance.json')}})
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'diagnose', 'plans', 'fit', 'evaluate', 'report'))
    parser.add_argument('--arm', choices=ARMS, default='M2')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--max-minutes', type=float, help='stop after epoch 1 if the projected fit exceeds this')
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    root = study_path(STUDY)
    if args.phase == 'prepare': return prepare(root)
    if args.phase == 'diagnose': return diagnose(root)
    if args.phase == 'plans': return plans_phase(root, args.arm)
    if args.phase == 'report': return report(root)
    device = str(pick_device(args.device))
    if args.phase == 'fit': return fit(root, args.arm, args.seed, device, args.max_minutes)
    evaluate(root, args.arm, device)


if __name__ == '__main__':
    main()
