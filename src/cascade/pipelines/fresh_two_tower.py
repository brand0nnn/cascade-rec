"""Fresh Two-tower (text) on all data before T1, served to the frozen P2/P3 rankers (Val only, exploratory).

Question: graph/support freshness helped every ranker; does retriever freshness help too? The frozen
Two-tower (text) seed 0 was trained before T0. Here the registered recipe is retrained on all pre-T1
interactions (``all_train``) and used only to serve fresher Val candidates.

Phases (``--phase``):
  train    ``train_retriever.main`` unchanged, with its three internal ``retriever_train`` interaction reads
           (training windows, static logQ counts, q audit) mapped to retriever_train + ranker_train.
  pools    Val P2' (fresh natural top 200, exact search, §4.6 filters) and P3' (``pools_build.interleave_fuse``
           of P1 recency and P2'). Check: the same path on the frozen checkpoint reproduces the stored P2 Val list.
  features nine common columns with the same pre-T0 ``BatchedGraphFeatures`` builder, standardised with the frozen
           P2 / P3 prefix scalers. Check: the same path on stored P3/P2 candidates reproduces stored features.
  score    frozen P3 rankers (seeds 0/1/2) on P3', frozen P2 rankers on P2'. Check first: stored P3/P2 Val
           ranks are reproduced exactly by this scoring path.
  report   pool and ranker metrics, paired user bootstraps.

HARD RULES: Val only; no Test file is opened; nothing is selected; no existing artifact is modified.

    PYTHONPATH=src .venv/bin/python -m cascade.pipelines.fresh_two_tower --phase train [--sample --technical-smoke]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

LABEL = 'Val only, exploratory'
CONFIG = 'results/analysis/20261106-fresh-two-tower/two_tower_text_all_train.yaml'
RUN_ID = 'local/20261106-two-tower-text-all-train-seed0'
SMOKE_RUN_ID = 'local/20261106-two-tower-text-all-train-sample-smoke'
FRESH_RUN = REPO_ROOT/'results/retriever'/RUN_ID
FROZEN_RUN = REPO_ROOT/'results/retriever/two_tower_text/seed0'
BULK = REPO_ROOT/'results/local_experiments/20261106-fresh-two-tower'
COMPACT = REPO_ROOT/'results/analysis/20261106-fresh-two-tower'
POOLS = REPO_ROOT/'results/local_experiments/20261106-pools'
RECENCY = REPO_ROOT/'results/local_experiments/20261103-full-promising-ranking/candidates'
RANKER_STUDY = {'P2': '20261106-pool-ranking-P2', 'P3': '20261106-pool-ranking-P3'}
STORED_POOL = {'P2': 'P2_two_tower', 'P3': 'P3_fused_rrf'}
FRESH_POOL = {'P2': 'P2f_two_tower_all_train', 'P3': 'P3f_fused_rrf_all_train'}
ARCHIVE_KEYS = ('cand', 'score', 'inst', 'user', 'target', 'retrieval_rank', 'top10', 'history')
K, COMMON, METHOD = 200, 9, 'bce_text'
FRESH_PERIODS = ['retriever_train', 'ranker_train']


def guard(path) -> Path:
    path = Path(path); low = str(path).lower()
    if 'test' in path.name.lower() or '/test/' in low:
        raise ValueError(f'refusing to touch a Test artifact: {path}')
    return path


def load_archive(directory):
    return {k: np.load(guard(Path(directory)/f'{k}.npy'), mmap_mode='r') for k in ARCHIVE_KEYS}


# ------------------------------------------------------------------------------------------------ train
class AllTrainView:
    """Delegating view of ProcessedData whose ``retriever_train`` interaction read covers all data before T1."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)

    def interactions(self, periods):
        if list(periods) != ['retriever_train']:
            raise ValueError(f'unexpected period request {periods}')
        out = self._data.interactions(FRESH_PERIODS)
        if len(out) and out.ts.max() >= self._data.split['t1']:
            raise ValueError('training interaction at or after T1')
        return out


def train(sample: bool, technical_smoke: bool, device: str) -> dict:
    from cascade.pipelines import train_retriever as tr
    orig_windows, orig_logq, orig_audit = tr.training_windows, tr.retriever_train_log_q, tr.q_target_audit

    def windows(data, max_len):
        return orig_windows(AllTrainView(data), max_len)

    def logq(data):
        return orig_logq(AllTrainView(data))

    def audit(data, seq_tgt):
        summary, arrays = orig_audit(AllTrainView(data), seq_tgt)
        summary['q_definition'] = 'all interactions before T1 (retriever_train + ranker_train); fresh_two_tower wrapper'
        return summary, arrays

    tr.training_windows, tr.retriever_train_log_q, tr.q_target_audit = windows, logq, audit
    try:
        metrics = tr.main(str(REPO_ROOT/CONFIG), 0, sample, device, splits=('val',),
                          run_id=SMOKE_RUN_ID if technical_smoke else RUN_ID, technical_smoke=technical_smoke)
    finally:
        tr.training_windows, tr.retriever_train_log_q, tr.q_target_audit = orig_windows, orig_logq, orig_audit
    run_dir = Path(metrics['run_dir'])
    write_json(run_dir/'fresh_training_period.json', {
        'label': LABEL, 'training_periods': FRESH_PERIODS, 'logq_periods': FRESH_PERIODS,
        'wrapper': 'src/cascade/pipelines/fresh_two_tower.py', 'wrapper_sha256': digest(Path(__file__)),
        'train_retriever_sha256': digest(Path(tr.__file__)), 'config': CONFIG, 'evaluation_splits': ['val'],
        'test_predictions_loaded': False, 'git_hash': git_hash()})
    print(json.dumps({k: metrics[k] for k in ('best_epoch', 'best_val_subset_recall@100', 'train_time_s')}), flush=True)
    return metrics


# ------------------------------------------------------------------------------------------------ pools
def data_checked():
    from cascade.analysis.pools_build import load_data
    return load_data()


def top200(run_dir, data, device, batch_size=512):
    """Natural float32 top 200 after §4.6 filters, plus the full-catalogue filtered target rank (pools_build.regen path)."""
    from cascade.analysis.pools_build import pool_ranks
    from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
    from cascade.models.retriever import RetrieverScorer
    from cascade.pipelines.local_evidence_export import check_candidates
    from cascade.pipelines.train_retriever import load_retriever
    model, _ = load_retriever(str(run_dir), data, device)
    scorer = RetrieverScorer(model)
    first = first_ts_tensor(data); first_np = first.numpy()
    inst = data.split_instances('val')
    n = len(inst)
    cand = np.empty((n, K), np.int32); score = np.empty((n, K), np.float32)
    ranks = np.empty(n); top = np.empty((n, 10), np.int32); full = np.empty(n)
    started = time.perf_counter()
    for s in range(0, n, batch_size):
        e = min(s+batch_size, n)
        b = make_batch(data, inst.iloc[s:e], data.max_hist_len)
        with torch.no_grad():
            sc = apply_filters(scorer.score_all(b), b, first)
            vals, ids = torch.topk(sc, K, dim=1)
            tg = sc[torch.arange(e-s), torch.as_tensor(b.targets.astype(np.int64))]
            full[s:e] = (sc >= tg[:, None]).sum(1).numpy()
        c, v = ids.numpy().astype(np.int32), vals.numpy()
        if not np.isfinite(v).all():
            raise ValueError('fewer than 200 eligible items')
        check_candidates(c, b, first_np)
        cand[s:e], score[s:e] = c, v
        ranks[s:e], top[s:e] = pool_ranks(v, c, b.targets)
        if (s//batch_size) % 100 == 0:
            print(json.dumps({'top200': str(run_dir.name), 'done': e, 'total': n, 's': round(time.perf_counter()-started)}), flush=True)
    return inst, {'cand': cand, 'score': score, 'retrieval_rank': ranks, 'top10': top}, full


def save_archive(out, values):
    out.mkdir(parents=True, exist_ok=False)
    for key in ARCHIVE_KEYS:
        np.save(out/f'{key}.npy', values[key])


def pools(device):
    from cascade.analysis.pools_build import filter_violations, interleave_fuse, pool_ranks
    from cascade.eval.harness import first_ts_tensor, make_batch
    torch.set_num_threads(3)
    data = data_checked()
    rec = load_archive(RECENCY/'val')
    record = {'label': LABEL, 'device': device, 'fresh_checkpoint_sha256': digest(FRESH_RUN/'model.pth'),
              'frozen_checkpoint_sha256': digest(FROZEN_RUN/'model.pth'), 'test_predictions_loaded': False}
    # check: the same path on the frozen checkpoint reproduces the stored P2 Val archive
    inst, frozen, frozen_full = top200(FROZEN_RUN, data, device)
    stored = load_archive(POOLS/'candidates'/STORED_POOL['P2']/'val')
    record['frozen_repro'] = {k: bool(np.array_equal(frozen[k], np.asarray(stored[k]))) for k in ('cand', 'score', 'retrieval_rank', 'top10')}
    record['frozen_repro']['full_rank_equal_aux'] = bool(np.array_equal(
        frozen_full, np.load(POOLS/'aux/P2_two_tower/val/full_catalogue_target_rank.npy')))
    if not all(record['frozen_repro'].values()):
        raise ValueError(f'frozen P2 Val archive not reproduced: {record["frozen_repro"]}')
    del frozen, stored; gc.collect()
    print(json.dumps({'frozen_repro': record['frozen_repro']}), flush=True)
    # fresh P2'
    inst, fresh, fresh_full = top200(FRESH_RUN, data, device)
    if not (np.array_equal(inst.inst.to_numpy(), rec['inst']) and np.array_equal(inst.item.to_numpy(), rec['target'])):
        raise ValueError('instance order differs from the recency archive')
    with np.load(guard(FRESH_RUN/'val_ranks.npz')) as z:
        keys = list(z.files)
        saved_rank = z[[k for k in keys if 'rank' in k][0]] if any('rank' in k for k in keys) else None
    record['fresh_val_ranks_keys'] = keys
    common = {'inst': inst.inst.to_numpy(), 'user': inst.user.to_numpy(), 'target': inst.item.to_numpy(),
              'history': np.asarray(rec['history'])}
    out = BULK/'candidates'/FRESH_POOL['P2']/'val'
    save_archive(out, {**common, **fresh})
    aux = BULK/'aux'/FRESH_POOL['P2']/'val'; aux.mkdir(parents=True, exist_ok=False)
    np.save(aux/'full_catalogue_target_rank.npy', fresh_full)
    if saved_rank is not None and len(saved_rank) == len(fresh_full):
        record['fresh_full_rank_equals_training_run_val_ranks'] = int((saved_rank == fresh_full).sum())
    record['P2f'] = {'rows': len(inst), 'natural_hits': int(np.isfinite(fresh['retrieval_rank']).sum()),
                     'pool_rank_equals_full_rank_inside200': int((fresh['retrieval_rank'][np.isfinite(fresh['retrieval_rank'])]
                                                                  == fresh_full[np.isfinite(fresh['retrieval_rank'])]).sum())}
    # fused P3'; check the fusion path reproduces stored P3 from stored P1/P2 first
    p2 = load_archive(POOLS/'candidates/P2_two_tower/val'); p3 = load_archive(POOLS/'candidates/P3_fused_rrf/val')
    n = len(inst)
    fused = {'cand': np.empty((n, K), np.int32), 'score': np.empty((n, K), np.float32),
             'retrieval_rank': np.empty(n), 'top10': np.empty((n, 10), np.int32)}
    src_rec, src_tt, inter = np.empty((n, K), np.int16), np.empty((n, K), np.int16), np.empty(n)
    repro = True
    for s in range(0, n, 4096):
        e = min(s+4096, n)
        r = np.asarray(rec['cand'][s:e])
        f0 = interleave_fuse(r, np.asarray(p2['cand'][s:e]))
        repro &= np.array_equal(f0['cand'], p3['cand'][s:e]) and np.array_equal(f0['score'], p3['score'][s:e])
        f = interleave_fuse(r, fresh['cand'][s:e])
        fused['cand'][s:e], fused['score'][s:e] = f['cand'], f['score']
        src_rec[s:e], src_tt[s:e] = f['rec_rank'], f['tt_rank']
        fused['retrieval_rank'][s:e], fused['top10'][s:e] = pool_ranks(f['score'], f['cand'], common['target'][s:e])
        hit = f['cand'] == common['target'][s:e, None]
        inter[s:e] = np.where(hit.any(1), hit.argmax(1)+1, np.inf)
    record['fusion_reproduces_stored_P3'] = bool(repro)
    if not repro:
        raise ValueError('fusion path does not reproduce stored P3')
    save_archive(BULK/'candidates'/FRESH_POOL['P3']/'val', {**common, **fused})
    aux = BULK/'aux'/FRESH_POOL['P3']/'val'; aux.mkdir(parents=True, exist_ok=False)
    np.save(aux/'source_rank_recency.npy', src_rec); np.save(aux/'source_rank_two_tower.npy', src_tt)
    np.save(aux/'interleave_rank.npy', inter)
    record['P3f'] = {'rows': n, 'natural_hits': int(np.isfinite(fused['retrieval_rank']).sum()),
                     'composition_mean_per_list': {'both': float(((src_rec > 0) & (src_tt > 0)).sum(1).mean()),
                                                   'recency_only': float(((src_rec > 0) & (src_tt == 0)).sum(1).mean()),
                                                   'two_tower_only': float(((src_rec == 0) & (src_tt > 0)).sum(1).mean())}}
    ov = np.array([len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(np.asarray(p2['cand']), fresh['cand'])])
    record['P2_P2f_top200_overlap'] = {'mean': float(ov.mean()), 'median': float(np.median(ov))}
    # §4.6 filter check on a seeded 10K sample (same function as pools_build.filters)
    first = first_ts_tensor(data).numpy()
    rows = np.sort(np.random.default_rng(0).choice(n, 10000, replace=False))
    b = make_batch(data, inst.iloc[rows], data.max_hist_len)
    record['filter_check'] = {name: filter_violations(c[rows], b.ts, b.mask_rows, b.mask_items, first)
                              for name, c in (('P2f', fresh['cand']), ('P3f', fused['cand']))}
    record['filter_check']['history_matches_harness'] = bool(np.array_equal(common['history'][rows], b.hist.numpy()))
    if any(v for r in record['filter_check'].values() if isinstance(r, dict) for v in r.values()) or \
            not record['filter_check']['history_matches_harness']:
        raise ValueError(f'filter violations: {record["filter_check"]}')
    write_json(COMPACT/'pools_check.json', record)
    print(json.dumps(record, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ features
def ranker_meta(pool):
    from cascade.pipelines import full_data_ranking as fdr
    from cascade.pipelines import pool_ranking as pr
    from cascade.pipelines.local_experiments import study_path
    root = study_path(RANKER_STUDY[pool])
    meta, source_root = pr.verify(root, artifacts=False, post_evaluation=True)
    return root, meta, source_root, fdr


def scaler(root):
    from cascade.pipelines import full_data_ranking as fdr
    s = fdr.read(root/'features/manifest.json')['scaler']['common']
    return np.asarray(s['mean']), np.asarray(s['std'])


def build_common(builder, inst, cand, mean, std, out, chunk=128):
    started = time.perf_counter()
    for start in range(0, len(inst), chunk):
        stop = min(start+chunk, len(inst))
        value = builder.build(inst.iloc[start:stop], np.asarray(cand[start:stop]))[0]
        scaled = ((value.astype(np.float64)-mean)/std).astype(np.float32)
        if not np.isfinite(scaled).all(): raise ValueError('nonfinite scaled features')
        out[start:stop] = scaled
        if start % (chunk*200) == 0:
            print(json.dumps({'features': str(getattr(out, 'filename', 'mem')), 'done': stop, 'total': len(inst),
                              'seconds': round(time.perf_counter()-started)}), flush=True)


def features(pools_=('P3', 'P2')):
    from cascade.data.batched_graph_features import BatchedGraphFeatures
    record = {'label': LABEL, 'test_predictions_loaded': False}
    data = None
    for pool in pools_:
        root, meta, source_root, fdr = ranker_meta(pool)
        if data is None:
            data = fdr.data_for(fdr.verify(source_root, artifacts=False))
            builder = BatchedGraphFeatures(data)
            inst = data.split_instances('val')
        mean, std = scaler(root)
        # check: rebuilding the stored pool's Val features with this path reproduces them
        stored_c = np.load(REPO_ROOT/meta['candidate_dirs']['val']/'cand.npy', mmap_mode='r')
        stored_f = np.load(root/'features/val/common.npy', mmap_mode='r')
        rows = np.sort(np.random.default_rng(7).choice(len(inst), 512, replace=False))
        rebuilt = np.empty((len(rows), K, COMMON), np.float32)
        build_common(builder, inst.iloc[rows], stored_c[rows], mean, std, rebuilt)
        err = float(np.abs(rebuilt-stored_f[rows]).max())
        record[pool] = {'stored_feature_rebuild_rows': len(rows), 'max_abs_error': err,
                        'exact': bool(np.array_equal(rebuilt, stored_f[rows])), 'scaler_from': str(root.relative_to(REPO_ROOT))}
        if not np.allclose(rebuilt, stored_f[rows], rtol=1e-5, atol=1e-5):
            raise ValueError(f'{pool}: builder does not reproduce stored features ({err})')
        fresh = load_archive(BULK/'candidates'/FRESH_POOL[pool]/'val')
        if not np.array_equal(fresh['inst'], inst.inst.to_numpy()): raise ValueError('event order differs')
        d = BULK/'features'/FRESH_POOL[pool]/'val'; d.mkdir(parents=True, exist_ok=False)
        out = open_memmap(d/'common.npy', mode='w+', dtype=np.float32, shape=(len(inst), K, COMMON))
        build_common(builder, inst, fresh['cand'], mean, std, out)
        out.flush(); del out
        # overlap check: identical (event, item) slots get identical features
        from cascade.pipelines.pool_ranking import overlap_index
        fresh_f = np.load(d/'common.npy', mmap_mode='r')
        sub = rows[:256]
        idx = overlap_index(np.asarray(fresh['cand'][sub]), np.asarray(stored_c[sub]))
        r, c = np.nonzero(idx >= 0)
        record[pool]['overlap_slots_checked'] = int(len(r))
        record[pool]['overlap_feature_max_abs_diff'] = float(np.abs(fresh_f[sub][r, c]-stored_f[sub][r, idx[r, c]]).max(initial=0))
        record[pool]['feature_sha256'] = digest(d/'common.npy')
        print(json.dumps({pool: record[pool]}), flush=True)
        gc.collect()
    write_json(COMPACT/'features_check.json', record)


# ------------------------------------------------------------------------------------------------ score
class ZeroGraph:
    def __getitem__(self, rows):
        return np.zeros((len(np.atleast_1d(rows)), K, 6), np.float32)


def score(device, pools_=('P3', 'P2')):
    from cascade.pipelines.graph_latent_study import state_hash
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    torch.set_num_threads(3)
    record = {'label': LABEL, 'device': device, 'test_predictions_loaded': False}
    for pool in pools_:
        root, meta, source_root, fdr = ranker_meta(pool)
        stored = fdr.arrays(REPO_ROOT/meta['candidate_dirs']['val'])
        stored_feature = {'common': np.load(root/'features/val/common.npy', mmap_mode='r'), 'graph': ZeroGraph()}
        fresh = load_archive(BULK/'candidates'/FRESH_POOL[pool]/'val')
        fresh_feature = {'common': np.load(BULK/'features'/FRESH_POOL[pool]/'val/common.npy', mmap_mode='r'), 'graph': ZeroGraph()}
        rows = np.arange(len(fresh['inst']))
        lock = fdr.read(root/'selection_lock.json')
        record[pool] = {}
        for seed in meta['config']['seeds']:
            run = root/f'runs/{METHOD}/seed{seed}'
            if digest(run/'model.pth') != lock[f'{METHOD}/seed{seed}']['model_sha256']:
                raise ValueError('ranker checkpoint changed')
            model = fdr.model_for(meta, root, seed, device)
            model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
            before = state_hash(model)
            ev = root/f'evaluation/{METHOD}/seed{seed}'
            with np.load(ev/'val_ranks.npz') as z: s_rank, s_top = z['rank'], z['top10']
            r0, t0, _ = fdr.predict(model, stored, stored_feature, rows, METHOD, device)
            repro = {'rank_exact': bool(np.array_equal(r0, s_rank)), 'top10_exact': bool(np.array_equal(t0, s_top))}
            if not all(repro.values()): raise ValueError(f'{pool} seed {seed}: stored ranks not reproduced {repro}')
            dest = BULK/'evaluation'/FRESH_POOL[pool]/f'seed{seed}'; dest.mkdir(parents=True, exist_ok=False)
            ranks, top, _ = fdr.predict(model, fresh, fresh_feature, rows, METHOD, device, dest/'logits.npy')
            if state_hash(model) != before: raise ValueError('scoring modified parameters')
            rr, tt = ranks_and_top10(np.load(dest/'logits.npy', mmap_mode='r'), fresh['cand'], fresh['target'])
            if not np.array_equal(rr, ranks) or not np.array_equal(tt, top): raise ValueError('saved scores do not replay')
            np.savez_compressed(dest/'val_ranks.npz', inst=fresh['inst'], user=fresh['user'], rank=ranks, top10=top)
            record[pool][f'seed{seed}'] = {'stored_repro': repro, 'weights_sha256': digest(run/'model.pth'),
                                           'logits_sha256': digest(dest/'logits.npy'),
                                           'ndcg10_stored_pool': average(M.ndcg_at(r0, 10), stored['user']),
                                           'ndcg10_fresh_pool': average(M.ndcg_at(ranks, 10), fresh['user'])}
            print(json.dumps({pool: seed, **record[pool][f'seed{seed}']}), flush=True)
            del model; gc.collect()
            if device == 'mps': torch.mps.empty_cache()
        # epoch-0 identity on the fresh pool (untrained ranker = pool retrieval order)
        model = fdr.model_for(meta, root, meta['config']['seeds'][0], device)
        r0, t0, lg = fdr.predict(model, fresh, fresh_feature, rows, METHOD, device)
        record[pool]['epoch0_fresh'] = {'logits_equal_score': bool(np.array_equal(lg, fresh['score'])),
                                        'ranks_equal_retrieval_rank': bool(np.array_equal(r0, fresh['retrieval_rank'])),
                                        'top10_equal': bool(np.array_equal(t0, fresh['top10']))}
        del model; gc.collect()
    write_json(COMPACT/'score_check.json', record)


# ------------------------------------------------------------------------------------------------ report
METRICS = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall10': lambda r: M.recall_at(r, 10),
           'hr3': lambda r: M.recall_at(r, 3), 'mrr10': lambda r: np.where(r <= 10, 1/np.asarray(r, float), 0.),
           'recall200': lambda r: np.isfinite(r).astype(float)}


def report():
    from cascade.pipelines import full_data_ranking as fdr
    from cascade.pipelines.local_experiments import study_path
    out = COMPACT
    for name in ('metrics.csv', 'paired_differences.csv', 'tables.md'):
        if (out/name).exists(): raise ValueError(f'refusing to overwrite {name}')
    root3, meta3, source_root, _ = ranker_meta('P3')
    data = fdr.data_for(fdr.verify(source_root, artifacts=False))
    inst = data.split_instances('val'); users = inst.user.to_numpy(); ids = inst.inst.to_numpy()
    g = fdr.groups_for(data, inst)
    first = data.items.set_index('item').first_ts.loc[inst.item].to_numpy()
    groups = {'all': g['all'], 'new_in_val': g['new_in_val'], 'warm_in_val': g['warm_in_val'],
              'first_T0_T1': (first >= data.split['t_ranker_train']) & (first < data.split['t1']),
              'first_before_T0': first < data.split['t_ranker_train']}
    seeds = meta3['config']['seeds']

    def ranks_of(path):
        with np.load(guard(path)) as z:
            if not np.array_equal(z['inst'], ids) or not np.array_equal(z['user'], users): raise ValueError(f'alignment {path}')
            return z['rank']
    sys = {}
    for pool in ('P2', 'P3'):
        root, meta, _, _ = ranker_meta(pool)
        sys[f'{pool}_retrieval'] = np.load(REPO_ROOT/meta['candidate_dirs']['val']/'retrieval_rank.npy')[None]
        sys[f"{pool}'_retrieval"] = np.load(BULK/'candidates'/FRESH_POOL[pool]/'val/retrieval_rank.npy')[None]
        sys[f'{pool}_ranker'] = np.stack([ranks_of(root/f'evaluation/{METHOD}/seed{s}/val_ranks.npz') for s in seeds])
        sys[f"{pool}'_ranker"] = np.stack([ranks_of(BULK/'evaluation'/FRESH_POOL[pool]/f'seed{s}/val_ranks.npz') for s in seeds])
    sys['P1_retrieval'] = np.load(RECENCY/'val/retrieval_rank.npy')[None]
    sys['P3_interleave_order'] = np.load(POOLS/'aux/P3_fused_rrf/val/interleave_rank.npy')[None]
    sys["P3'_interleave_order"] = np.load(BULK/'aux'/FRESH_POOL['P3']/'val/interleave_rank.npy')[None]
    # full-catalogue retriever ranks (Recall@500 etc. for the retrievers themselves)
    full = {'frozen_full': np.load(POOLS/'aux/P2_two_tower/val/full_catalogue_target_rank.npy'),
            'fresh_full': np.load(BULK/'aux'/FRESH_POOL['P2']/'val/full_catalogue_target_rank.npy')}
    contrasts = [("P2_retrieval", "P2'_retrieval"), ("P3_retrieval", "P3'_retrieval"),
                 ("P3_interleave_order", "P3'_interleave_order"),
                 ("P2_ranker", "P2'_ranker"), ("P3_ranker", "P3'_ranker"),
                 ("P2'_retrieval", "P2'_ranker"), ("P3'_retrieval", "P3'_ranker")]
    rows, diffs = [], []
    for metric, fn in METRICS.items():
        values = {m: np.stack([fn(r) for r in rs]) for m, rs in sys.items()}
        for group, mask in groups.items():
            for m, v in values.items():
                sm = [average(x[mask], users[mask]) for x in v]
                rows.append({'group': group, 'system': m, 'metric': metric, 'events': int(mask.sum()),
                             'users': int(np.unique(users[mask]).size), 'value': float(np.mean(sm)),
                             'seed_std_ddof0': float(np.std(sm)), **({f'seed{s}': x for s, x in zip(seeds, sm)} if len(v) > 1 else {})})
            for ref, cand in contrasts:
                if metric == 'recall200' and 'ranker' in cand and 'ranker' in ref:
                    continue
                ci = paired_bootstrap(users[mask], values[ref].mean(0)[mask], values[cand].mean(0)[mask], n_resamples=1000, seed=0)
                diffs.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand, 'events': int(mask.sum()), **ci})
        print(json.dumps({'metric': metric, 'done': True}), flush=True)
    for name, r in full.items():
        for group, mask in groups.items():
            for c in (100, 200, 500):
                rows.append({'group': group, 'system': name, 'metric': f'recall{c}_full_catalogue', 'events': int(mask.sum()),
                             'users': int(np.unique(users[mask]).size), 'value': average((r <= c).astype(float)[mask], users[mask])})
    frame, contrast = pd.DataFrame(rows), pd.DataFrame(diffs)
    frame.to_csv(out/'metrics.csv', index=False); contrast.to_csv(out/'paired_differences.csv', index=False)
    get = lambda m, gname, k: float(frame.loc[(frame.system == m) & (frame.group == gname) & (frame.metric == k), 'value'].iloc[0])
    fm = fdr.read(FRESH_RUN/'metrics.json'); fl = fdr.read(FRESH_RUN/'train_log.json')
    zm = fdr.read(FROZEN_RUN/'metrics.json')
    lines = ['# Fresh two-tower (all pre-T1) served to frozen P2/P3 rankers (Val only, exploratory)', '',
             f'{len(ids):,} Val events / {len(np.unique(users)):,} users; per-user macro, misses = 0; ranker values are seed means '
             '(0/1/2). Intervals: 1,000 paired-user draws, seed 0, on seed-mean event metrics; unadjusted. No Test file opened. '
             'Nothing selected. Hypothesis and caveats: README.md (written before computing).', '',
             '## Retrievers', '',
             f"Fresh: selected epoch {fm['best_epoch']} of {len(fl)} run (patience 5), subset R@100 {fm['best_val_subset_recall@100']:.6f}, "
             f"train {fm['train_time_s']/60:.1f} min on {fm['device']}. Frozen seed 0: epoch {zm['best_epoch']}, subset R@100 "
             f"{zm['best_val_subset_recall@100']:.6f}.", '',
             '| Retriever (full catalogue, filtered) | Group | events | R@100 | R@200 | R@500 |', '|---|---|---:|---:|---:|---:|']
    for name in full:
        for gname in groups:
            lines.append(f'| {name} | {gname} | {int(groups[gname].sum()):,} | ' +
                         ' | '.join(f"{get(name, gname, f'recall{c}_full_catalogue'):.6f}" for c in (100, 200, 500)) + ' |')
    lines += ['', '## Systems', '', '| System | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 | pool R@200 |', '|---|---|---:|---:|---:|---:|---:|']
    for m in sys:
        for gname in groups:
            lines.append(f'| {m} | {gname} | ' + ' | '.join(f'{get(m, gname, k):.6f}' for k in METRICS) + ' |')
    lines += ['', '## Paired differences (candidate minus reference)', '',
              '| Contrast | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 | R@200 |', '|---|---|---:|---:|---:|---:|---:|']
    for ref, cand in contrasts:
        for gname in groups:
            cells = []
            for k in METRICS:
                d = contrast.loc[(contrast.reference == ref) & (contrast.candidate == cand) & (contrast.group == gname) & (contrast.metric == k)]
                cells.append('—' if d.empty else f'{d["diff"].iloc[0]:+.6f} [{d.ci_low.iloc[0]:+.6f}, {d.ci_high.iloc[0]:+.6f}]')
            lines.append(f'| {cand} − {ref} | {gname} | ' + ' | '.join(cells) + ' |')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'report_provenance.json', {'label': LABEL, 'git_hash': git_hash(), 'source_sha256': digest(Path(__file__)),
                                              'test_predictions_loaded': False,
                                              'output_sha256': {p: digest(out/p) for p in ('metrics.csv', 'paired_differences.csv', 'tables.md')}})
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', required=True, choices=('train', 'pools', 'features', 'score', 'report'))
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--technical-smoke', action='store_true')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.phase == 'train': return train(a.sample, a.technical_smoke, a.device)
    if a.phase == 'pools': return pools(a.device)
    if a.phase == 'features': return features()
    if a.phase == 'score': return score(a.device)
    return report()


if __name__ == '__main__':
    main()
