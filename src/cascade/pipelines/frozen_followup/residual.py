"""R1 residual operations, extracted without changing the historical calculations."""
from __future__ import annotations

import gc
import json
import numpy as np
import torch
from .runtime import digest, log, protocol_guard, record_test_open, write_json


def mlp_context():
    from cascade.pipelines import test_final as TF
    from cascade.pipelines import pool_ranking as PR
    from cascade.pipelines.local_experiments import study_path
    parent, p3, data = TF.context()
    p2_root = study_path('20261106-pool-ranking-P2')
    p2_meta, _ = PR.verify(p2_root, artifacts=False, post_evaluation=True)
    return TF, PR, data, {'P1': parent, 'P3': p3}, p2_root, p2_meta


def p2_scaler(p2_root):
    from cascade.pipelines import full_data_ranking as P
    s = P.read(p2_root/'features/manifest.json')['scaler']
    return {'common': s['common'], 'graph': {'mean': [0.0]*6, 'std': [1.0]*6}}


def score_p2(p2_root, p2_meta, seed, sub, feats, device='mps'):
    from cascade.pipelines import full_data_ranking as P
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    lock = P.read(p2_root/'selection_lock.json'); run = p2_root/f'runs/bce_text/seed{seed}'
    if digest(run/'model.pth') != lock[f'bce_text/seed{seed}']['model_sha256']: raise ValueError('P2 checkpoint changed')
    model = P.model_for(p2_meta, p2_root, seed, device)
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    before = P.state_hash(model)
    ranks, _, logits = P.predict(model, sub, feats, np.arange(len(sub['cand'])), 'bce_text', device)
    if P.state_hash(model) != before: raise ValueError('scoring modified weights')
    if not np.array_equal(ranks_and_top10(logits, sub['cand'], sub['target'])[0], ranks): raise ValueError('rank replay')
    del model; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    return ranks


def recency_scores(ctx, data, parent, frame, cand, device='mps'):
    """P1 retrieval function (90-day recency text mean) for given candidates: M1's skip-score input on P3."""
    from cascade.eval.harness import make_batch
    from cascade.pipelines.local_followups import TextMeanScorer
    scorer = TextMeanScorer(data, np.load(ctx.repo_root/parent['text_path']), device, 90)
    out = np.empty(cand.shape, np.float32)
    for s in range(0, len(frame), 128):
        b = make_batch(data, frame.iloc[s:s+128], 50)
        with torch.no_grad():
            full = scorer.score_all(b).numpy()
        out[s:s+len(b.targets)] = np.take_along_axis(full, cand[s:s+128].astype(np.int64), axis=1)
    return out


def restandardise_p3_to_parent(common_p3, TF):
    sc = TF.scalers(); m3, m1 = sc['p3']['common'], sc['parent']['common']
    raw = common_p3.astype(np.float64)*np.asarray(m3['std'])+np.asarray(m3['mean'])
    return ((raw-np.asarray(m1['mean']))/np.asarray(m1['std'])).astype(np.float32)


def fresh_lists(ctx, TF, tt_data, frame, run=None):
    """Fresh two-tower top 200 in archive format (same loop as test_final.two_tower_lists, fresh checkpoint)."""
    from cascade.analysis import pools_build as PB
    orig = PB.TT_RUN
    try:
        PB.TT_RUN = run or ctx.fresh_run
        return TF.two_tower_lists(tt_data, frame, ctx.device_retrieval)
    finally:
        PB.TT_RUN = orig


def m1_on_mixed(TF, seed, metas, sub, feats):
    """M1 on mixed lists is scored on MPS, the device of the saved Val P1onP3 study (CPU flips one near-tied Val
    row, logits 4.8e-7 apart; MPS replays all three seeds exactly). The locked S1 Test ranks were scored on CPU."""
    saved = TF.MODELS['M1']
    try:
        TF.MODELS['M1'] = (*saved[:5], 'mps')
        return TF.score('M1', seed, metas, sub, feats)
    finally:
        TF.MODELS['M1'] = saved


def phase_mlp_check(ctx):
    from cascade.analysis import freshness_val as F
    from cascade.analysis import pools_build as PB
    from cascade.pipelines import full_data_ranking as P
    torch.set_num_threads(3)
    TF, PR, data, metas, p2_root, p2_meta = mlp_context()
    inst = data.split_instances('val'); out = {'label': 'Val reproduction of the Test pipeline', 'test_opened': False}
    # (a) fresher-retriever list builder reproduces the fresh study's Val pool
    tt_data = PB.load_data(); rows = np.arange(2048)
    if not np.array_equal(tt_data.split_instances('val').inst, inst.inst): raise ValueError('two-tower data differs')
    c, s = fresh_lists(ctx, TF, tt_data, tt_data.split_instances('val').iloc[rows])
    out['fresh_lists_exact'] = bool(np.array_equal(c, np.load(ctx.fresh_val/'cand.npy', mmap_mode='r')[rows])
                                    and np.array_equal(s, np.load(ctx.fresh_val/'score.npy', mmap_mode='r')[rows]))
    del tt_data; gc.collect()
    log(fresh_lists_exact=out['fresh_lists_exact'])
    # (b) P2 rankers: features rebuilt at T0 with the P2 scaler on all Val P2 hits, then ranks replay exactly
    p2 = P.arrays(ctx.pools/'P2_two_tower/val')
    hit = np.flatnonzero(np.isfinite(p2['retrieval_rank'])); frame = inst.iloc[hit]
    feats, _ = TF.build_features(data, frame, np.asarray(p2['cand'][hit]), TF.cutoffs_for('T0', frame.ts.to_numpy(np.int64)), p2_scaler(p2_root), 'val-check P2')
    stored = np.load(p2_root/'features/val/common.npy', mmap_mode='r')
    out['P2_feature_max_abs_error'] = float(np.max(np.abs(feats['common']-np.asarray(stored[hit]))))
    sub = F._sub_archive(p2, hit); out['P2_rank_replay_exact'] = {}
    for seed in ctx.seeds:
        r = np.full(len(inst), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
        with np.load(p2_root/f'evaluation/bce_text/seed{seed}/val_ranks.npz') as z:
            out['P2_rank_replay_exact'][seed] = bool(np.array_equal(z['rank'], r))
    log(P2=out['P2_rank_replay_exact'], err=out['P2_feature_max_abs_error'])
    # (c) recency ranker M1 on the mixed lists: recency skip score + P3 features re-standardised to P1 -> P1onP3 ranks
    p3 = P.arrays(ctx.pools/'P3_fused_rrf/val')
    hit3 = np.flatnonzero(np.isfinite(p3['retrieval_rank'])); frame3 = inst.iloc[hit3]
    sub3 = F._sub_archive(p3, hit3)
    sub3['score'] = recency_scores(ctx, data, metas['P1'], frame3, sub3['cand'])
    cross = ctx.repo_root/'results/local_experiments/20261106-pool-ranking-P1onP3'
    out['M1_skip_score_exact'] = bool(np.array_equal(sub3['score'], np.load(cross/'score.npy', mmap_mode='r')[hit3]))
    common = restandardise_p3_to_parent(np.load(ctx.repo_root/'results/local_experiments/20261106-pool-ranking-P3/features/val/common.npy', mmap_mode='r')[hit3], TF)
    out['M1_feature_max_abs_error'] = float(np.max(np.abs(common-np.load(cross/'common.npy', mmap_mode='r')[hit3])))
    feats3 = {'common': common, 'graph': np.zeros((len(hit3), 200, 6), np.float32)}
    out['M1_rank_replay_exact'] = {}
    for seed in ctx.seeds:
        r = np.full(len(inst), np.inf); r[hit3] = m1_on_mixed(TF, seed, metas, sub3, feats3)
        with np.load(cross/f'evaluation/seed{seed}/val_ranks.npz') as z:
            out['M1_rank_replay_exact'][seed] = int(np.sum(z['rank'] != r))
    log(M1=out['M1_rank_replay_exact'], skip=out['M1_skip_score_exact'], err=out['M1_feature_max_abs_error'])
    out['passed'] = bool(out['fresh_lists_exact'] and out['P2_feature_max_abs_error'] <= 2e-5 and all(out['P2_rank_replay_exact'].values())
                         and out['M1_skip_score_exact'] and out['M1_feature_max_abs_error'] <= 1e-4
                         and not any(out['M1_rank_replay_exact'].values()))
    write_json(ctx.out/'mlp_val_reproduction_check.json', out)
    if not out['passed']: raise SystemExit('Val reproduction failed: stop before opening Test')


def phase_mlp_test(ctx):
    from cascade.analysis import freshness_val as F
    from cascade.analysis import pools_build as PB
    from cascade.pipelines import full_data_ranking as P
    protocol_guard(ctx); torch.set_num_threads(3)
    if not json.loads((ctx.out/'mlp_val_reproduction_check.json').read_text())['passed']: raise ValueError('run mlp-check first')
    TF, PR, data, metas, p2_root, p2_meta = mlp_context()
    inst = data.split_instances('test')
    if (inst.ts.to_numpy(np.int64) < ctx.t2).any(): raise ValueError('non-Test query')
    record_test_open(ctx, 'mlp')
    ids, users, target = inst.inst.to_numpy(), inst.user.to_numpy(), inst.item.to_numpy()
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    # fresher retriever Test pool P2f
    tt_data = PB.load_data()
    if not np.array_equal(tt_data.split_instances('test').inst, inst.inst): raise ValueError('two-tower data differs')
    c, s = fresh_lists(ctx, TF, tt_data, tt_data.split_instances('test')); del tt_data; gc.collect()
    rr, top = ranks_and_top10(s, c, target)
    d = ctx.bulk/'mlp_candidates/P2f/test'; d.mkdir(parents=True, exist_ok=False)
    history = np.load(ctx.test_final/'candidates/P2/test/history.npy')
    for k, v in {'cand': c, 'score': s, 'retrieval_rank': rr, 'top10': top, 'inst': ids, 'user': users, 'target': target, 'history': history}.items():
        np.save(d/f'{k}.npy', v)
    log(P2f_test_hits=int(np.isfinite(rr).sum()))
    sc = p2_scaler(p2_root)
    for pool, directory in (('P2', ctx.test_final/'candidates/P2/test'), ('P2f', d)):
        arc = P.arrays(directory)
        if not np.array_equal(arc['inst'], ids): raise ValueError('Test alignment')
        hit = np.flatnonzero(np.isfinite(arc['retrieval_rank'])); frame = inst.iloc[hit]
        feats, audit = TF.build_features(data, frame, np.asarray(arc['cand'][hit]), TF.cutoffs_for('T0', frame.ts.to_numpy(np.int64)), sc, f'test {pool}')
        sub = F._sub_archive(arc, hit)
        for seed in ctx.seeds:
            r = np.full(len(ids), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
            dest = ctx.bulk/f'ranks/test/{pool}__p2ranker_seed{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists(): raise ValueError('scored once')
            np.savez_compressed(dest, inst=ids, user=users, rank=r)
            log(scored=pool, seed=seed)
    # recency ranker M1 on the mixed Test lists
    p3 = P.arrays(ctx.test_final/'candidates/P3/test')
    hit3 = np.load(ctx.test_final/'features/P3_p3_T0/hit.npy')
    if not np.array_equal(hit3, np.flatnonzero(np.isfinite(p3['retrieval_rank']))): raise ValueError('P3 hit rows')
    sub3 = F._sub_archive(p3, hit3)
    sub3['score'] = recency_scores(ctx, data, metas['P1'], inst.iloc[hit3], sub3['cand'])
    common = restandardise_p3_to_parent(np.load(ctx.test_final/'features/P3_p3_T0/common.npy'), TF)
    feats3 = {'common': common, 'graph': np.zeros((len(hit3), 200, 6), np.float32)}
    for seed in ctx.seeds:
        r = np.full(len(ids), np.inf); r[hit3] = m1_on_mixed(TF, seed, metas, sub3, feats3)
        dest = ctx.bulk/f'ranks/test/P3__m1_seed{seed}.npz'
        if dest.exists(): raise ValueError('scored once')
        np.savez_compressed(dest, inst=ids, user=users, rank=r)
        log(scored='M1 on P3', seed=seed)
