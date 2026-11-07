"""Locked Test run of the 5-7 Nov systems (protocol ``docs/test_protocol_2026-11-07.md``, approved 7 Nov 2026).

Frozen models only: nothing is trained, tuned or selected here, and every system is scored once.  Candidate
lists and product statistics are rebuilt for Test with the same builders as Val; ``check`` first proves on Val that
this module's builders and scorers reproduce the saved Val lists, features and ranks exactly.

Phases (repo root, PYTHONPATH=src): check -> candidates -> features -> score -> report.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time

import numpy as np
import pandas as pd
import torch

from cascade.analysis import freshness_val as F
from cascade.analysis import pools_build as PB
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import pool_ranking as PR
from cascade.pipelines import pool_rolling as PL
from cascade.pipelines import rolling_ranking as RR
from cascade.pipelines.local_evidence_export import check_candidates, ranks_and_top10
from cascade.pipelines.local_followups import TextMeanScorer
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

PROTOCOL = REPO_ROOT/'docs/test_protocol_2026-11-07.md'
PROTOCOL_SHA = 'ed463472b4f437ae27a768975fbc622b69a2a49b2d5792cd3cf60be45fd30ac7'
BULK = REPO_ROOT/'results/local_experiments/20261107-test-final'
OUT = REPO_ROOT/'results/test_final/20261107'
PARENT, ROLL, P3, PROLL = REPO_ROOT/F.PARENT, RR.STUDY, PL.P3, PL.STUDY
T2 = 1658002729837
SEEDS = (0, 1, 2)
DEVICE = 'mps'  # device of the saved Val recency and two-tower lists; ``check`` proves exact reproduction

# Frozen models: Val row, pool, method, run folder, model-construction root, scoring device (the device of each
# study's own Val evaluation; the P3 study scored on MPS, where CPU flips 1 of 20,441 tied Val rows).
MODELS = {'M1': ('B1', 'P1', 'bce_text', PARENT/'runs/bce_text', PARENT, 'cpu'),
          'M2': ('B9', 'P1', 'bce_text', ROLL/'runs/bce_text', PARENT, 'cpu'),
          'M3': ('B10', 'P1', 'bce_coreview', ROLL/'runs/bce_coreview', PARENT, 'cpu'),
          'M4': ('D1', 'P3', 'bce_text', P3/'runs/bce_text', P3, 'mps'),
          'M5': ('D6', 'P3', 'bce_text', PROLL/'runs/bce_text', P3, 'cpu'),
          'M6': ('D7', 'P3', 'bce_coreview', PROLL/'runs/bce_coreview', P3, 'cpu')}
# Feature arms: pool, scaler source, statistics rule.
ARMS = {'P1_parent_T0': ('P1', 'parent', 'T0'), 'P1_parent_T1': ('P1', 'parent', 'T1'),
        'P1_parent_T2': ('P1', 'parent', 'T2'), 'P1_roll_monthly': ('P1', 'roll', 'monthly'),
        'P3_p3_T0': ('P3', 'p3', 'T0'), 'P3_proll_monthly': ('P3', 'proll', 'monthly')}
# Systems from the protocol: model and feature arm (S0 = P1 retrieval order).
SYSTEMS = {'S1': ('M1', 'P1_parent_T0'), 'S2': ('M1', 'P1_parent_T1'), 'S3': ('M1', 'P1_parent_T2'),
           'S4': ('M2', 'P1_roll_monthly'), 'S5': ('M3', 'P1_roll_monthly'), 'S6': ('M4', 'P3_p3_T0'),
           'S7': ('M5', 'P3_proll_monthly'), 'S8': ('M6', 'P3_proll_monthly')}
PRIMARY = {'C1': ('S1', 'S3'), 'C2': ('S1', 'S6'), 'C3': ('S4', 'S5'), 'C4': ('S1', 'S8')}
SECONDARY = [('S0', 'S1'), ('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4'), ('S3', 'S5'), ('S6', 'S7'), ('S7', 'S8'),
             ('S5', 'S8'), ('S3', 'S8')]
VAL_REFERENCE = {'C1': 0.009432 - 0.008141, 'C2': 0.002878, 'C3': 0.010613 - 0.009304, 'C4': 0.013084 - 0.008141}


def guard():
    if digest(PROTOCOL) != PROTOCOL_SHA: raise ValueError('the frozen Test protocol changed')


def mask_for(method):
    return RR.mask_for(method)


def scalers():
    p3 = P.read(P3/'features/manifest.json')['scaler']
    # bce_text on P3 never used graph columns (zeroed by its mask); an identity graph scaler keeps shapes only.
    return {'parent': P.read(PARENT/'features/manifest.json')['scaler'],
            'roll': P.read(ROLL/'features/manifest.json')['scaler'],
            'p3': {'common': p3['common'], 'graph': {'mean': [0.0]*6, 'std': [1.0]*6}},
            'proll': P.read(PROLL/'features/manifest.json')['scaler']}


def context():
    parent = P.verify(PARENT, artifacts=False)
    p3 = PL._p3()
    return parent, p3, P.data_for(parent)


def unrestricted(data):
    """Shallow view whose split lets a snapshot end inside Test; ``assert_cut_rule`` still enforces past-only."""
    view = copy.copy(data); view.split = {**data.split, 't2': np.iinfo(np.int64).max}
    return view


def cutoffs_for(rule, ts):
    if rule == 'monthly': return F.month_start_ms(ts)
    return np.full(len(ts), {'T0': F.T0, 'T1': F.T1, 'T2': T2}[rule], np.int64)


def build_features(data, frame, cand, cutoffs, scaler, label):
    """Standardised common/graph features for ``frame`` rows, one snapshot per cutoff, with reference checks."""
    view = unrestricted(data)
    common = np.zeros((len(frame), 200, 9), np.float32); graph = np.zeros((len(frame), 200, 6), np.float32)
    audits = []
    for cutoff in np.unique(cutoffs):
        started = time.perf_counter()
        builder = F.SnapshotFeatures(view, int(cutoff))
        rows = np.flatnonzero(cutoffs == cutoff)
        cut = F.assert_cut_rule(builder, frame.iloc[rows], cand[rows])
        check = rows[np.sort(np.random.default_rng(int(cutoff) % 2**32).choice(len(rows), min(24, len(rows)), replace=False))]
        fast = builder.build(frame.iloc[check], cand[check])
        ref = FrozenGraphFeatures.build(builder, frame.iloc[check], cand[check])
        err = [float(np.max(np.abs(a-b))) for a, b in zip(fast, ref)]
        if max(err) > 2e-5: raise ValueError('batched snapshot differs from the explicit-exclusion reference')
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            values = builder.build(frame.iloc[part], cand[part])
            common[part] = F._standardise(values[0], scaler, 'common')
            graph[part] = F._standardise(values[1], scaler, 'graph')
        audits.append({**builder.audit, **cut, 'cutoff_utc': str(np.datetime64(int(cutoff), 'ms')), 'rows': int(len(rows)),
                       'reference_max_abs_error': err, 'seconds': time.perf_counter()-started})
        print(json.dumps({'features': label, **{k: audits[-1][k] for k in ('cutoff_utc', 'review_edges', 'rows', 'seconds')}}), flush=True)
        del builder; gc.collect()
    return {'common': common, 'graph': graph}, audits


def load_model(model_id, seed, metas):
    _, pool, method, runs, root, device = MODELS[model_id]
    run = runs/f'seed{seed}'
    if digest(run/'model.pth') != P.read(run/'metrics.json')['model_sha256']: raise ValueError(f'{model_id} weights changed')
    model = P.model_for(metas[pool], root, seed, 'cpu').eval()
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    return model.to(device)


def score(model_id, seed, metas, sub, feats):
    model = load_model(model_id, seed, metas)
    before = P.state_hash(model)
    ranks, _, logits = P.predict(model, sub, feats, np.arange(len(sub['cand'])), mask_for(MODELS[model_id][2]), MODELS[model_id][5])
    if P.state_hash(model) != before: raise ValueError('scoring modified weights')
    if not np.array_equal(ranks_and_top10(logits, sub['cand'], sub['target'])[0], ranks): raise ValueError('rank replay')
    del model; gc.collect()
    if MODELS[model_id][5] == 'mps': torch.mps.empty_cache()
    return ranks


def recency_lists(data, text, frame, device):
    """The frozen 90-day recency retriever, same loop as ``full_data_ranking.candidate_archive``."""
    scorer = TextMeanScorer(data, text, device, 90); first = first_ts_tensor(data)
    cand = np.empty((len(frame), 200), np.int32); score_ = np.empty((len(frame), 200), np.float32)
    for start in range(0, len(frame), 128):
        batch = make_batch(data, frame.iloc[start:start+128], 50)
        with torch.no_grad():
            vals, ids = torch.topk(apply_filters(scorer.score_all(batch), batch, first), 200, dim=1)
            ids[~torch.isfinite(vals)] = 0
        c, s = ids.numpy().astype(np.int32), vals.numpy()
        check_candidates(c, batch, first.numpy())
        cand[start:start+len(c)], score_[start:start+len(c)] = c, s
    return cand, score_


def two_tower_lists(data, frame, device, batch_size=512):
    """The frozen two-tower (text) seed 0, same loop as ``pools_build.regen``."""
    from cascade.models.retriever import RetrieverScorer
    from cascade.pipelines.train_retriever import load_retriever
    model, _ = load_retriever(str(PB.TT_RUN), data, device)
    scorer = RetrieverScorer(model); first = first_ts_tensor(data); first_np = first.numpy()
    cand = np.empty((len(frame), 200), np.int32); score_ = np.empty((len(frame), 200), np.float32)
    for start in range(0, len(frame), batch_size):
        batch = make_batch(data, frame.iloc[start:start+batch_size], data.max_hist_len)
        with torch.no_grad():
            vals, ids = torch.topk(apply_filters(scorer.score_all(batch), batch, first), 200, dim=1)
        c, v = ids.numpy().astype(np.int32), vals.numpy()
        if not np.isfinite(v).all(): raise ValueError('fewer than 200 eligible items')
        check_candidates(c, batch, first_np)
        cand[start:start+len(c)], score_[start:start+len(c)] = c, v
    return cand, score_


def fuse(rec, tt, target, chunk=4096):
    out = {'cand': np.empty_like(rec), 'score': np.empty(rec.shape, np.float32)}
    for s in range(0, len(rec), chunk):
        f = PB.interleave_fuse(rec[s:s+chunk], tt[s:s+chunk])
        out['cand'][s:s+chunk], out['score'][s:s+chunk] = f['cand'], f['score']
    out['retrieval_rank'], out['top10'] = PB.pool_ranks(out['score'], out['cand'], target)
    return out


# ---------------------------------------------------------------- phases
def phase_check():
    """On Val: every builder and scorer used for Test reproduces the saved Val lists, features and ranks."""
    guard()
    parent, p3, data = context(); metas = {'P1': parent, 'P3': p3}; sc = scalers()
    inst = data.split_instances('val'); out = {'label': 'Val reproduction of the Test pipeline', 'test_opened': False}
    rows = np.arange(2048)
    text = np.load(REPO_ROOT/parent['text_path'])
    a1, a2, a3 = P.arrays(PARENT/'candidates/val'), P.arrays(PB.BULK/'candidates/P2_two_tower/val'), P.arrays(PL.POOL/'val')
    c1, s1 = recency_lists(data, text, inst.iloc[rows], DEVICE)
    out['P1_lists_exact'] = bool(np.array_equal(c1, a1['cand'][rows]) and np.array_equal(s1, a1['score'][rows]))
    tt_data = PB.load_data()
    if not np.array_equal(tt_data.split_instances('val').inst, inst.inst): raise ValueError('two-tower data differs')
    c2, s2 = two_tower_lists(tt_data, tt_data.split_instances('val').iloc[rows], DEVICE)
    out['P2_lists_exact'] = bool(np.array_equal(c2, a2['cand'][rows]) and np.array_equal(s2, a2['score'][rows]))
    del tt_data; gc.collect()
    f3 = fuse(np.asarray(a1['cand'][rows]), np.asarray(a2['cand'][rows]), inst.item.to_numpy()[rows])
    out['P3_fusion_exact'] = bool(np.array_equal(f3['cand'], a3['cand'][rows]) and np.array_equal(f3['score'], a3['score'][rows]))
    saved = {'P1_parent_T0': lambda h, r: {k: np.asarray(P.arrays(PARENT/'features/val')[k][h[r]]) for k in ('common', 'graph')},
             'P1_parent_T1': lambda h, r: {k: np.load(REPO_ROOT/F.STUDY/f'features/T1/{k}.npy', mmap_mode='r')[r] for k in ('common', 'graph')},
             'P1_roll_monthly': lambda h, r: {k: np.load(ROLL/f'features/val_monthly/{k}.npy', mmap_mode='r')[r] for k in ('common', 'graph')},
             'P3_p3_T0': lambda h, r: {'common': np.load(P3/'features/val/common.npy', mmap_mode='r')[h[r]]},
             'P3_proll_monthly': lambda h, r: {k: np.load(PROLL/f'features/val_monthly/{k}.npy', mmap_mode='r')[r] for k in ('common', 'graph')}}
    out['features_max_abs_error'] = {}
    for arm, load in saved.items():
        pool, scaler, rule = ARMS[arm]
        archive = a1 if pool == 'P1' else a3
        hit = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
        r = np.sort(np.random.default_rng(7).choice(len(hit), 256, replace=False))
        frame = inst.iloc[hit[r]]
        feats, _ = build_features(data, frame, np.asarray(archive['cand'][hit[r]]), cutoffs_for(rule, frame.ts.to_numpy(np.int64)), sc[scaler], f'val-check {arm}')
        ref = load(hit, r)
        err = {k: float(np.max(np.abs(feats[k]-np.asarray(ref[k])))) for k in ref}
        out['features_max_abs_error'][arm] = err
        if max(err.values()) > 2e-5: raise ValueError(f'{arm} features differ from the saved Val features')
    saved_ranks = {'M1': (PARENT/'evaluation/bce_text/seed0/val_ranks.npz', lambda h: {k: np.asarray(P.arrays(PARENT/'features/val')[k][h]) for k in ('common', 'graph')}),
                   'M2': (ROLL/'evaluation/val_monthly/bce_text/seed0.npz', lambda h: {k: np.load(ROLL/f'features/val_monthly/{k}.npy') for k in ('common', 'graph')}),
                   'M3': (ROLL/'evaluation/val_monthly/bce_coreview/seed0.npz', lambda h: {k: np.load(ROLL/f'features/val_monthly/{k}.npy') for k in ('common', 'graph')}),
                   'M4': (P3/'evaluation/bce_text/seed0/val_ranks.npz', lambda h: {'common': np.asarray(np.load(P3/'features/val/common.npy', mmap_mode='r')[h]), 'graph': np.zeros((len(h), 200, 6), np.float32)}),
                   'M5': (PROLL/'evaluation/val_monthly/bce_text/seed0.npz', lambda h: PL.val_features('val_monthly')),
                   'M6': (PROLL/'evaluation/val_monthly/bce_coreview/seed0.npz', lambda h: PL.val_features('val_monthly'))}
    out['val_rank_replay_exact'] = {}
    for model_id, (path, feats) in saved_ranks.items():
        archive = a1 if MODELS[model_id][1] == 'P1' else a3
        hit = np.flatnonzero(np.isfinite(archive['retrieval_rank'])); sub = F._sub_archive(archive, hit)
        ranks = np.full(len(inst), np.inf); ranks[hit] = score(model_id, 0, metas, sub, feats(hit))
        with np.load(path) as z: exact = bool(np.array_equal(z['rank'], ranks))
        out['val_rank_replay_exact'][model_id] = exact
        print(json.dumps({'replay': model_id, 'exact': exact}), flush=True)
    passed = out['P1_lists_exact'] and out['P2_lists_exact'] and out['P3_fusion_exact'] and all(out['val_rank_replay_exact'].values())
    out['passed'] = bool(passed)
    OUT.mkdir(parents=True, exist_ok=True); write_json(OUT/'val_reproduction_check.json', out)
    print(json.dumps(out), flush=True)
    if not passed: raise SystemExit('Val reproduction failed: stop before opening Test')


def phase_candidates():
    guard()
    if not P.read(OUT/'val_reproduction_check.json')['passed']: raise ValueError('run check first')
    parent, _, data = context()
    inst = data.split_instances('test')
    ts = inst.ts.to_numpy(np.int64)
    if (ts < T2).any(): raise ValueError('non-Test query')
    write_json(OUT/'test_opened.json', {'opened_utc': pd.Timestamp.utcnow().isoformat(), 'protocol_sha256': PROTOCOL_SHA,
                                        'events': int(len(inst)), 'users': int(inst.user.nunique()), 'git_hash': git_hash()})
    target = inst.item.to_numpy(); history = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    started = time.perf_counter()
    c1, s1 = recency_lists(data, np.load(REPO_ROOT/parent['text_path']), inst, DEVICE)
    print(json.dumps({'P1': 'done', 'seconds': time.perf_counter()-started}), flush=True)
    tt_data = PB.load_data()
    if not np.array_equal(tt_data.split_instances('test').inst, inst.inst): raise ValueError('two-tower data differs')
    c2, s2 = two_tower_lists(tt_data, tt_data.split_instances('test'), DEVICE)
    del tt_data; gc.collect()
    print(json.dumps({'P2': 'done', 'seconds': time.perf_counter()-started}), flush=True)
    if (c1 <= 0).any(): raise ValueError('recency list padding: fusion needs 200 real items')
    pools = {'P1': {'cand': c1, 'score': s1}, 'P2': {'cand': c2, 'score': s2}}
    for name in ('P1', 'P2'):
        pools[name]['retrieval_rank'], pools[name]['top10'] = ranks_and_top10(pools[name]['score'], pools[name]['cand'], target)
    pools['P3'] = fuse(c1, c2, target)
    for name, arc in pools.items():
        directory = BULK/f'candidates/{name}/test'; directory.mkdir(parents=True, exist_ok=False)
        for key, value in {**arc, 'inst': inst.inst.to_numpy(), 'user': inst.user.to_numpy(), 'target': target, 'history': history}.items():
            np.save(directory/f'{key}.npy', value)
        print(json.dumps({'pool': name, 'natural_hits': int(np.isfinite(arc['retrieval_rank']).sum())}), flush=True)


def phase_features():
    guard()
    _, _, data = context(); sc = scalers()
    inst = data.split_instances('test'); audits = {}
    for arm, (pool, scaler, rule) in ARMS.items():
        archive = P.arrays(BULK/f'candidates/{pool}/test')
        if not np.array_equal(archive['inst'], inst.inst): raise ValueError('Test alignment')
        hit = np.flatnonzero(np.isfinite(archive['retrieval_rank'])); frame = inst.iloc[hit]
        feats, audits[arm] = build_features(data, frame, np.asarray(archive['cand'][hit]), cutoffs_for(rule, frame.ts.to_numpy(np.int64)), sc[scaler], arm)
        directory = BULK/f'features/{arm}'; directory.mkdir(parents=True, exist_ok=False)
        for key, value in {**feats, 'hit': hit}.items(): np.save(directory/f'{key}.npy', value)
    write_json(BULK/'features/manifest.json', {'arms': {k: list(v) for k, v in ARMS.items()}, 'audits': audits,
                                                'monthly_rule': 'reviews strictly before the UTC month start of each Test query (July 2022 queries use 1 Jul 2022)'})


def phase_score():
    guard()
    parent, p3, _ = context(); metas = {'P1': parent, 'P3': p3}
    for system, (model_id, arm) in SYSTEMS.items():
        pool = ARMS[arm][0]
        archive = P.arrays(BULK/f'candidates/{pool}/test')
        hit = np.load(BULK/f'features/{arm}/hit.npy'); sub = F._sub_archive(archive, hit)
        feats = {k: np.load(BULK/f'features/{arm}/{k}.npy') for k in ('common', 'graph')}
        for seed in SEEDS:
            dest = BULK/f'ranks/{system}/seed{seed}.npz'
            if dest.exists(): raise ValueError(f'{system} seed {seed} already scored: Test is scored once')
            ranks = np.full(len(archive['inst']), np.inf); ranks[hit] = score(model_id, seed, metas, sub, feats)
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, inst=np.asarray(archive['inst']), user=np.asarray(archive['user']), rank=ranks)
            print(json.dumps({'scored': system, 'seed': seed, 'ndcg10': average(F._metric_values(ranks)['ndcg10'], np.asarray(archive['user']))}), flush=True)


def judge(diff, low, high, rule):
    """Mechanical verdict of one written prediction."""
    if rule == 'positive': return 'held' if low > 0 else 'not held' if high < 0 else 'inconclusive'
    if rule == 'negative': return 'held' if high < 0 else 'not held' if low > 0 else 'inconclusive'
    if rule == 'nonpositive': return 'held' if diff <= 0 else 'not held'
    raise ValueError(rule)


def phase_report():
    guard()
    _, _, data = context()
    inst = data.split_instances('test'); users = inst.user.to_numpy(); ids = inst.inst.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= T2
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    values, rows = {}, []
    retrieval = {name: np.load(BULK/f'candidates/{name}/test/retrieval_rank.npy') for name in ('P1', 'P2', 'P3')}
    per_system = {'S0': [retrieval['P1']]}
    for system in SYSTEMS:
        per_system[system] = []
        for seed in SEEDS:
            with np.load(BULK/f'ranks/{system}/seed{seed}.npz') as z:
                if not np.array_equal(z['inst'], ids): raise ValueError('rank alignment')
                per_system[system].append(z['rank'])
    for system, ranks in per_system.items():
        seeds = [F._metric_values(r) for r in ranks]
        values[system] = {m: np.mean([s[m] for s in seeds], axis=0) for m in seeds[0]}
        pool = 'P1' if system == 'S0' else ARMS[SYSTEMS[system][1]][0]
        for group, mask in groups.items():
            for metric in seeds[0]:
                per = [average(s[metric][mask], users[mask]) for s in seeds]
                rows.append({'system': system, 'pool': pool, 'group': group, 'metric': metric, 'value': float(np.mean(per)),
                             'seed_std_ddof0': float(np.std(per)), 'events': int(mask.sum()),
                             **({f'seed{s}': v for s, v in zip(SEEDS, per)} if len(per) > 1 else {})})
            rows.append({'system': system, 'pool': pool, 'group': group, 'metric': 'recall200',
                         'value': average((retrieval[pool] <= 200)[mask].astype(float), users[mask]), 'events': int(mask.sum())})
    for name, r in retrieval.items():
        for group, mask in groups.items():
            rows.append({'system': f'{name}_retrieval', 'pool': name, 'group': group, 'metric': 'ndcg10',
                         'value': average(F._metric_values(r)['ndcg10'][mask], users[mask]), 'events': int(mask.sum())})
            rows.append({'system': f'{name}_retrieval', 'pool': name, 'group': group, 'metric': 'recall200',
                         'value': average((r <= 200)[mask].astype(float), users[mask]), 'events': int(mask.sum())})
    contrasts = []
    plan = [(c, ref, cand, ('ndcg10', 'recall10', 'hr3', 'mrr10')) for c, (ref, cand) in PRIMARY.items()]
    plan += [('secondary', ref, cand, ('ndcg10',)) for ref, cand in SECONDARY]
    for label, ref, cand, metrics in plan:
        for group, mask in groups.items():
            for metric in metrics:
                ci = paired_bootstrap(users[mask], values[ref][metric][mask], values[cand][metric][mask], n_resamples=1000, seed=0)
                contrasts.append({'contrast': label, 'reference': ref, 'candidate': cand, 'group': group, 'metric': metric, **ci})
            print(json.dumps({'contrast': label, 'ref': ref, 'cand': cand, 'group': group}), flush=True)
    frame, cons = pd.DataFrame(rows), pd.DataFrame(contrasts)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT/'metrics.csv', index=False); cons.to_csv(OUT/'contrasts.csv', index=False)
    get = lambda s, g, m='ndcg10': float(frame.loc[(frame.system == s) & (frame.group == g) & (frame.metric == m), 'value'].iloc[0])
    con = lambda label, g, ref=None, cand=None: cons.loc[(cons.contrast == label) & (cons.group == g) & (cons.metric == 'ndcg10')
                                                       & ((cons.reference == ref) if ref else True) & ((cons.candidate == cand) if cand else True)].iloc[0]
    preds = []
    def add(text, label, group, rule):
        d = con(label, group); preds.append({'prediction': text, 'contrast': label, 'group': group, 'diff': d['diff'],
                                             'ci_low': d.ci_low, 'ci_high': d.ci_high, 'verdict': judge(d['diff'], d.ci_low, d.ci_high, rule)})
    add('1. Period-start statistics help frozen models (C1 > 0)', 'C1', 'all', 'positive')
    add('2a. Mixed pool helps warm targets (C2 warm > 0)', 'C2', 'warm', 'positive')
    add('2b. Mixed pool hurts new targets (C2 new < 0)', 'C2', 'new', 'negative')
    add('3. Co-review helps with monthly statistics (C3 > 0)', 'C3', 'all', 'positive')
    add('4a. Combined system helps warm targets (C4 warm > 0)', 'C4', 'warm', 'positive')
    add('4b. Combined system does not help new targets (C4 new <= 0)', 'C4', 'new', 'nonpositive')
    extra = {'1b. S3 >= S2 overall (T2 at least as good as T1)': bool(get('S3', 'all') >= get('S2', 'all')),
             '2c. C2 overall smaller than on Val (+0.002878)': bool(con('C2', 'all')['diff'] < VAL_REFERENCE['C2']),
             '4c. C4 overall smaller than on Val (+0.004943)': bool(con('C4', 'all')['diff'] < VAL_REFERENCE['C4'])}
    pd.DataFrame(preds).to_csv(OUT/'predictions.csv', index=False)
    names = {'S0': 'P1 recency order (no ranker)', 'S1': 'B1 current pipeline, T0', 'S2': 'B1, T1', 'S3': 'B1, T2',
             'S4': 'B9 text, monthly', 'S5': 'B10 co-review, monthly', 'S6': 'D1 mixed pool, T0',
             'S7': 'D6 mixed pool text, monthly', 'S8': 'D7 mixed pool co-review, monthly'}
    lines = ['# Locked Test results (7 Nov 2026)', '', f'Protocol `docs/test_protocol_2026-11-07.md` (sha256 {PROTOCOL_SHA}). '
             f'{len(inst):,} Test events / {len(np.unique(users)):,} users; {int(new.sum()):,} new-target events '
             f'({new.mean():.1%}). Frozen models, each system scored once; per-user macro, misses = 0, seed means (0/1/2). '
             'Intervals: 1,000 paired-user draws, seed 0, unadjusted.', '',
             '## Systems, nDCG@10', '', '| System | Description | all | new | warm | Recall@10 all | Recall@200 of its pool |',
             '|---|---|---:|---:|---:|---:|---:|']
    for s in per_system:
        lines.append(f'| {s} | {names[s]} | {get(s, "all"):.6f} | {get(s, "new"):.6f} | {get(s, "warm"):.6f} | '
                     f'{get(s, "all", "recall10"):.6f} | {get(s, "all", "recall200"):.6f} |')
    lines += ['', '## Primary contrasts, nDCG@10 (candidate − reference) [95% CI]', '', '| # | Contrast | all | new | warm | Val (all) |', '|---|---|---|---|---|---:|']
    for label, (ref, cand) in PRIMARY.items():
        cells = [con(label, g) for g in ('all', 'new', 'warm')]
        lines.append(f'| {label} | {cand} − {ref} | ' + ' | '.join(f'{d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}]' for d in cells)
                     + f' | {VAL_REFERENCE[label]:+.6f} |')
    lines += ['', '## Secondary contrasts, nDCG@10', '', '| Contrast | all | new | warm |', '|---|---|---|---|']
    for ref, cand in SECONDARY:
        cells = [con('secondary', g, ref, cand) for g in ('all', 'new', 'warm')]
        lines.append(f'| {cand} − {ref} | ' + ' | '.join(f'{d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}]' for d in cells) + ' |')
    lines += ['', '## Written predictions', '', '| Prediction | Result [95% CI] | Verdict |', '|---|---|---|']
    lines += [f'| {p["prediction"]} | {p["diff"]:+.6f} [{p["ci_low"]:+.6f}, {p["ci_high"]:+.6f}] | {p["verdict"]} |' for p in preds]
    lines += [f'| {k} | — | {"held" if v else "not held"} |' for k, v in extra.items()]
    team = {'S0 nDCG@10 (team 4 Nov Test: 0.006043)': get('S0', 'all'), 'P1 Recall@200 (team: 0.053114)': get('P1_retrieval', 'all', 'recall200'),
            'S1 nDCG@10 (team bce_text T0, their weights: 0.006532)': get('S1', 'all'), 'S2 nDCG@10 (team bce_text T1, their weights: 0.007506)': get('S2', 'all')}
    lines += ['', '## Consistency with the team\'s 4 Nov Test run', '', '| Quantity | This run |', '|---|---:|']
    lines += [f'| {k} | {v:.6f} |' for k, v in team.items()]
    lines += ['', '## Checks', '', 'See `val_reproduction_check.json` (builders and scorers reproduce saved Val lists, features and ranks '
              'exactly before Test was opened), `test_opened.json` and `report_provenance.json`.', '']
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    (OUT/'git_hash.txt').write_text(git_hash()+'\n')
    write_json(OUT/'report_provenance.json', {'protocol_sha256': PROTOCOL_SHA, 'test_predictions_loaded': True, 'training': 'none',
        'selection': 'none', 'ranks_sha256': {f'{s}/seed{k}': digest(BULK/f'ranks/{s}/seed{k}.npz') for s in SYSTEMS for k in SEEDS},
        'source_sha256': digest(REPO_ROOT/'src/cascade/pipelines/test_final.py')})
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'candidates', 'features', 'score', 'report'))
    args = parser.parse_args(); torch.set_num_threads(3)
    {'check': phase_check, 'candidates': phase_candidates, 'features': phase_features,
     'score': phase_score, 'report': phase_report}[args.phase]()


if __name__ == '__main__':
    main()
