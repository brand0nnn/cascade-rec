"""Frozen rankers on better candidate lists: Val evidence for DIN and a post-hoc Test check (9 Nov 2026).

Question: does a ranker that is NOT retrained gain from a different, higher-recall candidate list?  Three list
changes, two ranker families, all frozen (nothing is trained, tuned or selected here):

* family ``mlp`` (the poster's Val tests, Brandon's score-plus-correction MLP with period-start common inputs):
  - fresher retriever: pool-ranking-P2 bce_text rankers (trained on the frozen two-tower's lists) on lists from
    the two-tower retrained on all data before T1 (P2f) versus their own two-tower lists (P2);
  - mixed lists: the recency-list ranker (full-promising bce_text, M1/B1) on the mixed recency + two-tower lists
    (P3) versus its own recency lists (P1), and versus the ranker retrained on P3 (locked Test system S6).
* family ``din`` (natural-list rankers on the two-tower K200 lists, retrieval + matching inputs, no statistics):
  policy-study arm C MLP (29 Oct) and DIN natural (28 Oct), 3 seeds each, on
  - 500 instead of 200 two-tower candidates (nested prefixes of the same list),
  - the fresher retriever's top 200,
  - the mixed lists (P3 candidates carrying their two-tower score, the score these rankers were trained on).

Protocol and written predictions: ``docs/test_protocol_2026-11-09.md``; its sha256 is locked (phase ``lock``)
before any Test phase, and every Test phase re-checks it.  Val phases never load a Test file.

Phases (repo root, PYTHONPATH=src):
  din-val     build Val fresh/mixed lists in the candidate-file format, score both rankers, anchor-check K200
  mlp-check   prove on Val that every Test builder/scorer of family ``mlp`` reproduces saved Val outputs
  lock        hash the written protocol (after reading the Val results)
  din-test    Test lists + scoring for family ``din``      (opens Test)
  mlp-test    Test lists/features + scoring for family ``mlp`` (opens Test)
  report      tables for Val and Test, mechanical verdicts of the written predictions
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

BULK = REPO_ROOT/'results/local_experiments/20261109-frozen-ranker-followup'
OUT = REPO_ROOT/'results/analysis/20261109-frozen-ranker-followup'
PROTOCOL = REPO_ROOT/'docs/test_protocol_2026-11-09.md'
T1, T2 = 1628643414042, 1658002729837
SEEDS = (0, 1, 2)
TT_RUN = REPO_ROOT/'results/retriever/two_tower_text/seed0'
FRESH_RUN = REPO_ROOT/'results/retriever/local/20261106-two-tower-text-all-train-seed0'
FRESH_RUN_S1 = REPO_ROOT/'results/retriever/local/20261106-two-tower-text-all-train-seed1'  # seed 2 never completed
TT_DIR = REPO_ROOT/'data/processed/candidates/two_tower_text_seed0'
POOLS = REPO_ROOT/'results/local_experiments/20261106-pools/candidates'
FRESH_VAL = REPO_ROOT/'results/local_experiments/20261106-fresh-two-tower/candidates/P2f_two_tower_all_train/val'
TEST_FINAL = REPO_ROOT/'results/local_experiments/20261107-test-final'
DIN_RUNS = {'armC_mlp': [REPO_ROOT/'results/ranker_policy/policy-pilot-20261029/arm_C_seed0',
                         REPO_ROOT/'results/ranker_policy/policy-replication-20261029/arm_C_seed1',
                         REPO_ROOT/'results/ranker_policy/policy-replication-20261029/arm_C_seed2'],
            'din': [REPO_ROOT/f'results/ranker_din/20261028-din-natural-seed{s}' for s in SEEDS]}
RANKER_DEVICE = 'cpu'  # overridden by --ranker-device
DEVICE_RETRIEVAL = 'mps'  # device of the saved two-tower Val lists (rebuilt on MPS 28 Oct; pools on MPS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float)+'\n')


def log(**kw):
    print(json.dumps(kw, default=float), flush=True)


def protocol_guard():
    lock = BULK/'protocol_lock.json'
    if not lock.exists(): raise ValueError('lock the written protocol before opening Test')
    if digest(PROTOCOL) != json.loads(lock.read_text())['protocol_sha256']:
        raise ValueError('the locked Test protocol changed')


def record_test_open(family):
    path = BULK/f'test_opened_{family}.json'
    if path.exists(): raise ValueError(f'{family}: Test already opened; each system is scored once')
    write_json(path, {'opened_utc': pd.Timestamp.utcnow().isoformat(), 'family': family,
                      'protocol_sha256': digest(PROTOCOL), 'git_hash': git_hash()})


# ============================================================== family din (candidate-file format)
def din_context():
    from cascade.data.instances import ProcessedData
    from cascade.data.ranker_features import RankerFeatureBuilder
    from cascade.pipelines.build_candidates import load_candidates
    data = ProcessedData.load(str(REPO_ROOT/'configs/data.yaml'))
    builder = RankerFeatureBuilder.create(data, load_candidates(TT_DIR, 'ranker_train'))
    return data, builder


def as_lists(inst, user, cand, score16, target):
    hit = cand == target[:, None]
    pos = np.where(hit.any(1), hit.argmax(1), -1).astype(np.int64)
    label = np.zeros(cand.shape, np.int8); r = np.flatnonzero(pos >= 0); label[r, pos[r]] = 1
    return {'inst': np.asarray(inst), 'user': np.asarray(user), 'cand': np.asarray(cand, np.int32),
            'score': np.asarray(score16, np.float16), 'target_pos': pos, 'label': label}


def save_lists(name, split, lists):
    d = BULK/'lists'/name; d.mkdir(parents=True, exist_ok=True)
    path = d/f'{split}.npz'
    if path.exists(): raise ValueError(f'refusing to overwrite {path}')
    np.savez(path, **{k: v for k, v in lists.items() if k != 'label'})
    return path


def load_lists(name, split, k1=None):
    with np.load(BULK/'lists'/name/f'{split}.npz') as z:
        lists = {k: z[k] for k in z.files}
    if k1 is not None:
        lists['cand'], lists['score'] = lists['cand'][:, :k1], lists['score'][:, :k1]
        lists['target_pos'] = np.where(lists['target_pos'] < k1, lists['target_pos'], -1)
    label = np.zeros(lists['cand'].shape, np.int8); r = np.flatnonzero(lists['target_pos'] >= 0)
    label[r, lists['target_pos'][r]] = 1; lists['label'] = label
    return lists


def scorer_for(run, data):
    from cascade.models.retriever import RetrieverScorer
    from cascade.pipelines.train_retriever import load_retriever
    model, _ = load_retriever(str(run), data, DEVICE_RETRIEVAL)
    return RetrieverScorer(model)


def two_tower_scores_for(scorer, data, inst, cand, batch_size=512):
    """Two-tower score of every given candidate (float32), same scorer as the candidate builder."""
    from cascade.eval.harness import make_batch
    out = np.zeros(cand.shape, np.float32)
    for s in range(0, len(inst), batch_size):
        b = make_batch(data, inst.iloc[s:s+batch_size], data.max_hist_len)
        with torch.no_grad():
            full = scorer.score_all(b)
        out[s:s+len(b.targets)] = torch.gather(full, 1, torch.as_tensor(cand[s:s+batch_size].astype(np.int64))).numpy()
    return out


def mixed_lists(data, scorer, inst, p3_archive):
    """P3 candidates (recency + two-tower round-robin) re-ordered to ``inst`` and given their two-tower score."""
    order = pd.Series(np.arange(len(p3_archive['inst'])), index=np.asarray(p3_archive['inst'])).loc[inst.inst.to_numpy()].to_numpy()
    cand = np.asarray(p3_archive['cand'])[order]
    if (cand <= 0).any(): raise ValueError('mixed list padding')
    score = two_tower_scores_for(scorer, data, inst, cand)
    if not np.isfinite(score).all(): raise ValueError('nonfinite two-tower score on a mixed-list candidate')
    return as_lists(inst.inst.to_numpy(), inst.user.to_numpy(), cand, score.astype(np.float16), inst.item.to_numpy().astype(np.int32)), order


def load_ranker(run, builder):
    from cascade.pipelines.index_rank_eval import load_ranker as lr
    rec = json.loads((run/'metrics.json').read_text())
    if rec.get('development_sample') or rec.get('status') == 'running': raise ValueError(f'not a completed full-data fit: {run}')
    return lr(run, builder)


def score_family(builder, lists, counts, device=None):
    device = device or RANKER_DEVICE
    from cascade.pipelines.ranker_k1_sweep import score_prefixes
    out = {}
    for name, runs in DIN_RUNS.items():
        per_seed = []
        for run in runs:
            model, seed = load_ranker(run, builder)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            ranks, _ = score_prefixes(model, builder, lists, counts, device=device, batch_pairs=8000)
            if any(not torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items()): raise ValueError('weights changed')
            per_seed.append(ranks); log(scored=name, seed=seed, counts=counts)
            del model; gc.collect()
        out[name] = np.stack(per_seed)  # [seed, count, n]
    return out


def save_ranks(split, list_name, counts, inst, user, ranks):
    for name, value in ranks.items():
        path = BULK/'ranks'/split/f'{list_name}__{name}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise ValueError(f'already scored: {path}')
        np.savez_compressed(path, inst=inst, user=user, counts=np.array(counts), ranks=value)


def phase_din_val():
    from cascade.pipelines.build_candidates import load_candidates, retrieve
    torch.set_num_threads(3)
    data, builder = din_context(); inst = data.split_instances('val')
    users, ids = inst.user.to_numpy(), inst.inst.to_numpy()
    rec = {'label': 'Val only, exploratory follow-up', 'test_predictions_loaded': False, 'ranker_device': RANKER_DEVICE}
    tt = load_candidates(TT_DIR, 'val')  # stored two-tower top 500, never forced
    if not np.array_equal(tt['inst'], ids): raise ValueError('Val order differs from the stored lists')
    # anchor first: frozen rankers on the stored two-tower K200 Val lists reproduce their saved ranks exactly
    ranks = score_family(builder, tt, [200, 500])
    rec['anchor_rank_differences'] = {}
    for name, runs in DIN_RUNS.items():
        for si, run in enumerate(runs):
            with np.load(run/'val_ranks.npz') as z:
                if not np.array_equal(z['inst'], ids): raise ValueError('stored rank order')
                rec['anchor_rank_differences'][f'{name}_seed{si}'] = int(np.sum(z['ranks'] != ranks[name][si, 0]))
    log(anchor=rec['anchor_rank_differences'])
    if any(rec['anchor_rank_differences'].values()): write_json(OUT/'din_val_checks.json', rec); raise SystemExit('anchor failed')
    save_ranks('val', 'tt', [200, 500], ids, users, ranks)
    # check: the retrieval path reproduces the stored two-tower Val lists and their float16 scores
    scorer = scorer_for(TT_RUN, data); rows = np.arange(2048)
    again = retrieve(scorer, data, inst.iloc[rows], 500, show_progress=False)
    rec['tt_rebuild_exact'] = bool(np.array_equal(again['cand'], tt['cand'][rows]) and np.array_equal(again['score'], tt['score'][rows]))
    sc = two_tower_scores_for(scorer, data, inst.iloc[rows], tt['cand'][rows]).astype(np.float16)
    rec['tt_score_regather_exact'] = bool(np.array_equal(sc, tt['score'][rows]))
    log(tt_rebuild_exact=rec['tt_rebuild_exact'], tt_score_regather_exact=rec['tt_score_regather_exact'])
    if not (rec['tt_rebuild_exact'] and rec['tt_score_regather_exact']): write_json(OUT/'din_val_checks.json', rec); raise SystemExit(rec)
    # mixed lists (Brandon's P3 Val pool, the same lists as the mlp family), each candidate with its two-tower score
    p3 = {k: np.load(POOLS/f'P3_fused_rrf/val/{k}.npy', mmap_mode='r') for k in ('inst', 'cand')}
    mixed, _ = mixed_lists(data, scorer, inst, p3); save_lists('mixed', 'val', mixed)
    sub = np.arange(4096)
    eq = mixed['cand'][sub, :, None] == tt['cand'][sub, None, :]
    r, c = np.nonzero(eq.any(2)); j = eq.argmax(2)[r, c]
    rec['mixed_overlap_slots'] = int(len(r))
    rec['mixed_overlap_score_exact'] = bool(np.array_equal(mixed['score'][sub][r, c], tt['score'][sub][r, j]))
    del scorer; gc.collect(); torch.mps.empty_cache()
    # fresher retriever top 200 in the candidate-file format; must equal the fresh study's Val pool
    fscorer = scorer_for(FRESH_RUN, data)
    f = retrieve(fscorer, data, inst, 200)
    fresh = as_lists(f['inst'], f['user'], f['cand'], f['score'], inst.item.to_numpy().astype(np.int32)); save_lists('fresh', 'val', fresh)
    rec['fresh_equals_fresh_study_pool'] = bool(np.array_equal(np.load(FRESH_VAL/'inst.npy'), fresh['inst'])
                                                and np.array_equal(np.load(FRESH_VAL/'cand.npy'), fresh['cand']))
    del fscorer; gc.collect(); torch.mps.empty_cache()
    rec['recall_in_list'] = {'tt200': float(((tt['target_pos'] >= 0) & (tt['target_pos'] < 200)).mean()),
                             'tt500': float((tt['target_pos'] >= 0).mean()), 'fresh': float((fresh['target_pos'] >= 0).mean()),
                             'mixed': float((mixed['target_pos'] >= 0).mean())}
    log(**{k: v for k, v in rec.items() if k != 'anchor_rank_differences'})
    for list_name, lists in (('fresh', fresh), ('mixed', mixed)):
        save_ranks('val', list_name, [200], ids, users, score_family(builder, lists, [200]))
    write_json(OUT/'din_val_checks.json', rec)


def phase_din_test():
    from cascade.pipelines.build_candidates import retrieve
    protocol_guard(); torch.set_num_threads(3)
    data, builder = din_context(); inst = data.split_instances('test')
    if (inst.ts.to_numpy(np.int64) < T2).any(): raise ValueError('non-Test query')
    record_test_open('din')
    ids, users, target = inst.inst.to_numpy(), inst.user.to_numpy(), inst.item.to_numpy().astype(np.int32)
    scorer = scorer_for(TT_RUN, data)
    t = retrieve(scorer, data, inst, 500)
    tt = as_lists(t['inst'], t['user'], t['cand'], t['score'], target); save_lists('tt', 'test', tt)
    p3 = {k: np.load(TEST_FINAL/f'candidates/P3/test/{k}.npy', mmap_mode='r') for k in ('inst', 'cand')}
    mixed, _ = mixed_lists(data, scorer, inst, p3); save_lists('mixed', 'test', mixed)
    del scorer; gc.collect(); torch.mps.empty_cache()
    fscorer = scorer_for(FRESH_RUN, data)
    f = retrieve(fscorer, data, inst, 200)
    fresh = as_lists(f['inst'], f['user'], f['cand'], f['score'], target); save_lists('fresh', 'test', fresh)
    del fscorer; gc.collect(); torch.mps.empty_cache()
    p2 = np.load(TEST_FINAL/'candidates/P2/test/cand.npy', mmap_mode='r'); p2i = np.load(TEST_FINAL/'candidates/P2/test/inst.npy')
    write_json(BULK/'din_test_lists_check.json', {
        'tt_top200_equals_locked_test_P2': bool(np.array_equal(p2i, ids) and np.array_equal(np.asarray(p2), tt['cand'][:, :200])),
        'recall200': {'tt': float(((tt['target_pos'] >= 0) & (tt['target_pos'] < 200)).mean()), 'tt500': float((tt['target_pos'] >= 0).mean()),
                      'fresh': float((fresh['target_pos'] >= 0).mean()), 'mixed': float((mixed['target_pos'] >= 0).mean())}})
    save_ranks('test', 'tt', [200, 500], ids, users, score_family(builder, tt, [200, 500]))
    for list_name, lists in (('fresh', fresh), ('mixed', mixed)):
        save_ranks('test', list_name, [200], ids, users, score_family(builder, lists, [200]))


# ============================================================== family mlp (archive format, period-start inputs)
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


def recency_scores(data, parent, frame, cand, device='mps'):
    """P1 retrieval function (90-day recency text mean) for given candidates: M1's skip-score input on P3."""
    from cascade.eval.harness import make_batch
    from cascade.pipelines.local_followups import TextMeanScorer
    scorer = TextMeanScorer(data, np.load(REPO_ROOT/parent['text_path']), device, 90)
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


def fresh_lists(TF, tt_data, frame, run=None):
    """Fresh two-tower top 200 in archive format (same loop as test_final.two_tower_lists, fresh checkpoint)."""
    from cascade.analysis import pools_build as PB
    orig = PB.TT_RUN
    try:
        PB.TT_RUN = run or FRESH_RUN
        return TF.two_tower_lists(tt_data, frame, DEVICE_RETRIEVAL)
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


def phase_mlp_check():
    from cascade.analysis import freshness_val as F
    from cascade.analysis import pools_build as PB
    from cascade.pipelines import full_data_ranking as P
    torch.set_num_threads(3)
    TF, PR, data, metas, p2_root, p2_meta = mlp_context()
    inst = data.split_instances('val'); out = {'label': 'Val reproduction of the Test pipeline', 'test_opened': False}
    # (a) fresher-retriever list builder reproduces the fresh study's Val pool
    tt_data = PB.load_data(); rows = np.arange(2048)
    if not np.array_equal(tt_data.split_instances('val').inst, inst.inst): raise ValueError('two-tower data differs')
    c, s = fresh_lists(TF, tt_data, tt_data.split_instances('val').iloc[rows])
    out['fresh_lists_exact'] = bool(np.array_equal(c, np.load(FRESH_VAL/'cand.npy', mmap_mode='r')[rows])
                                    and np.array_equal(s, np.load(FRESH_VAL/'score.npy', mmap_mode='r')[rows]))
    del tt_data; gc.collect()
    log(fresh_lists_exact=out['fresh_lists_exact'])
    # (b) P2 rankers: features rebuilt at T0 with the P2 scaler on all Val P2 hits, then ranks replay exactly
    p2 = P.arrays(POOLS/'P2_two_tower/val')
    hit = np.flatnonzero(np.isfinite(p2['retrieval_rank'])); frame = inst.iloc[hit]
    feats, _ = TF.build_features(data, frame, np.asarray(p2['cand'][hit]), TF.cutoffs_for('T0', frame.ts.to_numpy(np.int64)), p2_scaler(p2_root), 'val-check P2')
    stored = np.load(p2_root/'features/val/common.npy', mmap_mode='r')
    out['P2_feature_max_abs_error'] = float(np.max(np.abs(feats['common']-np.asarray(stored[hit]))))
    sub = F._sub_archive(p2, hit); out['P2_rank_replay_exact'] = {}
    for seed in SEEDS:
        r = np.full(len(inst), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
        with np.load(p2_root/f'evaluation/bce_text/seed{seed}/val_ranks.npz') as z:
            out['P2_rank_replay_exact'][seed] = bool(np.array_equal(z['rank'], r))
    log(P2=out['P2_rank_replay_exact'], err=out['P2_feature_max_abs_error'])
    # (c) recency ranker M1 on the mixed lists: recency skip score + P3 features re-standardised to P1 -> P1onP3 ranks
    p3 = P.arrays(POOLS/'P3_fused_rrf/val')
    hit3 = np.flatnonzero(np.isfinite(p3['retrieval_rank'])); frame3 = inst.iloc[hit3]
    sub3 = F._sub_archive(p3, hit3)
    sub3['score'] = recency_scores(data, metas['P1'], frame3, sub3['cand'])
    cross = REPO_ROOT/'results/local_experiments/20261106-pool-ranking-P1onP3'
    out['M1_skip_score_exact'] = bool(np.array_equal(sub3['score'], np.load(cross/'score.npy', mmap_mode='r')[hit3]))
    common = restandardise_p3_to_parent(np.load(REPO_ROOT/'results/local_experiments/20261106-pool-ranking-P3/features/val/common.npy', mmap_mode='r')[hit3], TF)
    out['M1_feature_max_abs_error'] = float(np.max(np.abs(common-np.load(cross/'common.npy', mmap_mode='r')[hit3])))
    feats3 = {'common': common, 'graph': np.zeros((len(hit3), 200, 6), np.float32)}
    out['M1_rank_replay_exact'] = {}
    for seed in SEEDS:
        r = np.full(len(inst), np.inf); r[hit3] = m1_on_mixed(TF, seed, metas, sub3, feats3)
        with np.load(cross/f'evaluation/seed{seed}/val_ranks.npz') as z:
            out['M1_rank_replay_exact'][seed] = int(np.sum(z['rank'] != r))
    log(M1=out['M1_rank_replay_exact'], skip=out['M1_skip_score_exact'], err=out['M1_feature_max_abs_error'])
    out['passed'] = bool(out['fresh_lists_exact'] and out['P2_feature_max_abs_error'] <= 2e-5 and all(out['P2_rank_replay_exact'].values())
                         and out['M1_skip_score_exact'] and out['M1_feature_max_abs_error'] <= 1e-4
                         and not any(out['M1_rank_replay_exact'].values()))
    write_json(OUT/'mlp_val_reproduction_check.json', out)
    if not out['passed']: raise SystemExit('Val reproduction failed: stop before opening Test')


def phase_mlp_test():
    from cascade.analysis import freshness_val as F
    from cascade.analysis import pools_build as PB
    from cascade.pipelines import full_data_ranking as P
    protocol_guard(); torch.set_num_threads(3)
    if not json.loads((OUT/'mlp_val_reproduction_check.json').read_text())['passed']: raise ValueError('run mlp-check first')
    TF, PR, data, metas, p2_root, p2_meta = mlp_context()
    inst = data.split_instances('test')
    if (inst.ts.to_numpy(np.int64) < T2).any(): raise ValueError('non-Test query')
    record_test_open('mlp')
    ids, users, target = inst.inst.to_numpy(), inst.user.to_numpy(), inst.item.to_numpy()
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    # fresher retriever Test pool P2f
    tt_data = PB.load_data()
    if not np.array_equal(tt_data.split_instances('test').inst, inst.inst): raise ValueError('two-tower data differs')
    c, s = fresh_lists(TF, tt_data, tt_data.split_instances('test')); del tt_data; gc.collect()
    rr, top = ranks_and_top10(s, c, target)
    d = BULK/'mlp_candidates/P2f/test'; d.mkdir(parents=True, exist_ok=False)
    history = np.load(TEST_FINAL/'candidates/P2/test/history.npy')
    for k, v in {'cand': c, 'score': s, 'retrieval_rank': rr, 'top10': top, 'inst': ids, 'user': users, 'target': target, 'history': history}.items():
        np.save(d/f'{k}.npy', v)
    log(P2f_test_hits=int(np.isfinite(rr).sum()))
    sc = p2_scaler(p2_root)
    for pool, directory in (('P2', TEST_FINAL/'candidates/P2/test'), ('P2f', d)):
        arc = P.arrays(directory)
        if not np.array_equal(arc['inst'], ids): raise ValueError('Test alignment')
        hit = np.flatnonzero(np.isfinite(arc['retrieval_rank'])); frame = inst.iloc[hit]
        feats, audit = TF.build_features(data, frame, np.asarray(arc['cand'][hit]), TF.cutoffs_for('T0', frame.ts.to_numpy(np.int64)), sc, f'test {pool}')
        sub = F._sub_archive(arc, hit)
        for seed in SEEDS:
            r = np.full(len(ids), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
            dest = BULK/f'ranks/test/{pool}__p2ranker_seed{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists(): raise ValueError('scored once')
            np.savez_compressed(dest, inst=ids, user=users, rank=r)
            log(scored=pool, seed=seed)
    # recency ranker M1 on the mixed Test lists
    p3 = P.arrays(TEST_FINAL/'candidates/P3/test')
    hit3 = np.load(TEST_FINAL/'features/P3_p3_T0/hit.npy')
    if not np.array_equal(hit3, np.flatnonzero(np.isfinite(p3['retrieval_rank']))): raise ValueError('P3 hit rows')
    sub3 = F._sub_archive(p3, hit3)
    sub3['score'] = recency_scores(data, metas['P1'], inst.iloc[hit3], sub3['cand'])
    common = restandardise_p3_to_parent(np.load(TEST_FINAL/'features/P3_p3_T0/common.npy'), TF)
    feats3 = {'common': common, 'graph': np.zeros((len(hit3), 200, 6), np.float32)}
    for seed in SEEDS:
        r = np.full(len(ids), np.inf); r[hit3] = m1_on_mixed(TF, seed, metas, sub3, feats3)
        dest = BULK/f'ranks/test/P3__m1_seed{seed}.npz'
        if dest.exists(): raise ValueError('scored once')
        np.savez_compressed(dest, inst=ids, user=users, rank=r)
        log(scored='M1 on P3', seed=seed)


# ============================================================== report
def verdicts(levels, cons):
    """Mechanical verdicts of the predictions written in docs/test_protocol_2026-11-09.md."""
    protocol_guard()
    lv = lambda r, l: float(levels.query("split=='test' and ranker==@r and list==@l and group=='all'").ndcg10.iloc[0])
    def con(ref, cand, group='all'):
        row = cons.query("split=='test' and reference==@ref and candidate==@cand and group==@group")
        return row.iloc[0]
    din_lists = json.loads((BULK/'din_test_lists_check.json').read_text())['recall200']
    inlist = lambda path: float(np.isfinite(np.load(path)).mean())
    recall = {'tt200': din_lists['tt'], 'tt500': din_lists['tt500'], 'fresh': din_lists['fresh'], 'mixed': din_lists['mixed'],
              'P2': inlist(TEST_FINAL/'candidates/P2/test/retrieval_rank.npy'), 'P2f': inlist(BULK/'mlp_candidates/P2f/test/retrieval_rank.npy'),
              'P1': inlist(TEST_FINAL/'candidates/P1/test/retrieval_rank.npy'), 'P3': inlist(TEST_FINAL/'candidates/P3/test/retrieval_rank.npy')}
    out = []
    cells = [('armC_mlp', 'tt200', 'tt500'), ('din', 'tt200', 'tt500'), ('armC_mlp', 'tt200', 'fresh'), ('din', 'tt200', 'fresh'),
             ('armC_mlp', 'tt200', 'mixed'), ('din', 'tt200', 'mixed'), ('p2ranker', 'P2', 'P2f'), ('m1', 'P1', 'P3')]
    for r, a, b in cells:
        rel_n = lv(r, b)/lv(r, a)-1; rel_r = recall[b]/recall[a]-1
        out.append({'prediction': f'U {r}: {b} vs {a}', 'value': rel_n, 'threshold': rel_r/3,
                    'detail': f'nDCG {rel_n:+.1%} vs purchases in list {rel_r:+.1%}', 'verdict': 'held' if rel_n < rel_r/3 else 'not held'})
    def directional(label, ref, cand, group, rule):
        c = con(ref, cand, group); lo, hi = c.ci_low, c.ci_high
        if rule == 'negative': v = 'held' if hi < 0 else 'not held' if lo > 0 else 'inconclusive'
        else: v = 'held' if lo > 0 else 'not held' if hi < 0 else 'inconclusive'
        out.append({'prediction': label, 'value': c['diff'], 'ci_low': lo, 'ci_high': hi, 'verdict': v})
    directional('D1 armC_mlp tt500 - tt200 < 0', 'armC_mlp/tt200', 'armC_mlp/tt500', 'all', 'negative')
    directional('D2 armC_mlp fresh - tt200 < 0', 'armC_mlp/tt200', 'armC_mlp/fresh', 'all', 'negative')
    directional('D3 din fresh - tt200 < 0', 'din/tt200', 'din/fresh', 'all', 'negative')
    directional('D4 din mixed - tt200 < 0', 'din/tt200', 'din/mixed', 'all', 'negative')
    directional('D5 p2ranker P2f - P2 < 0 (new)', 'p2ranker/P2', 'p2ranker/P2f', 'new', 'negative')
    directional('D6 p2ranker gain < retrieval-order gain', 'P2f gain of retrieval order', 'P2f gain of frozen P2 ranker', 'all', 'negative')
    directional('D7 m1 P3 - P1 < 0 (new)', 'm1/P1', 'm1/P3', 'new', 'negative')
    directional('D8 retrained on P3 beats m1 frozen on P3', 'm1/P3', 'retrained/P3', 'all', 'positive')
    pd.DataFrame(out).to_csv(OUT/'predictions.csv', index=False)
    write_json(OUT/'recall_in_list_test.json', recall)
    print(pd.DataFrame(out).to_string())


def phase_report():
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import first_ts_tensor
    from cascade.data.instances import ProcessedData
    data = ProcessedData.load(str(REPO_ROOT/'configs/data.yaml')); first = first_ts_tensor(data).numpy()
    rows, cons = [], []
    nd = lambda r: M.ndcg_at(np.asarray(r, float), 10)
    avg = lambda v, u: M.macro_micro(np.asarray(v, float), u)['macro']

    def din_values(split):
        out = {}
        for path in sorted((BULK/'ranks'/split).glob('*__armC_mlp.npz'))+sorted((BULK/'ranks'/split).glob('*__din.npz')):
            lst, name = path.stem.split('__')
            with np.load(path) as z:
                for ci, k in enumerate(z['counts']):
                    out[(name, f'{lst}{"" if lst != "tt" else int(k)}')] = (z['ranks'][:, ci], z['inst'], z['user'])
        return out

    plans = []
    splits = [('val', T1)] + ([('test', T2)] if list(BULK.glob('test_opened_*.json')) else [])
    for split, start in splits:
        inst = data.split_instances(split); ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
        new = first[inst.item.to_numpy()] >= start
        groups = {'all': np.ones(len(ids), bool), 'new': new, 'warm': ~new}
        systems = {}
        for key, (ranks, i, u) in din_values(split).items():
            if not np.array_equal(i, ids): raise ValueError('alignment')
            systems[key] = np.mean([nd(r) for r in ranks], axis=0)
        if split == 'val' and (BULK/'ranks/val/P2f_s1__p2ranker_seed0.npz').exists():
            p2_root = REPO_ROOT/'results/local_experiments/20261106-pool-ranking-P2'
            systems[('p2ranker', 'P2')] = np.mean([nd(np.load(p2_root/f'evaluation/bce_text/seed{s}/val_ranks.npz')['rank']) for s in SEEDS], axis=0)
            systems[('p2ranker', 'P2f_s1')] = np.mean([nd(np.load(BULK/f'ranks/val/P2f_s1__p2ranker_seed{s}.npz')['rank']) for s in SEEDS], axis=0)
            systems[('retrieval', 'P2')] = nd(np.load(POOLS/'P2_two_tower/val/retrieval_rank.npy'))
            systems[('retrieval', 'P2f_s1')] = nd(np.load(BULK/'mlp_candidates/P2f_s1/val/retrieval_rank.npy'))
        if split == 'test' and (BULK/'ranks/test/P2__p2ranker_seed0.npz').exists():
            load = lambda p: np.load(p)['rank']
            systems[('p2ranker', 'P2')] = np.mean([nd(load(BULK/f'ranks/test/P2__p2ranker_seed{s}.npz')) for s in SEEDS], axis=0)
            systems[('p2ranker', 'P2f')] = np.mean([nd(load(BULK/f'ranks/test/P2f__p2ranker_seed{s}.npz')) for s in SEEDS], axis=0)
            systems[('retrieval', 'P2')] = nd(np.load(TEST_FINAL/'candidates/P2/test/retrieval_rank.npy'))
            systems[('retrieval', 'P2f')] = nd(np.load(BULK/'mlp_candidates/P2f/test/retrieval_rank.npy'))
            systems[('m1', 'P3')] = np.mean([nd(load(BULK/f'ranks/test/P3__m1_seed{s}.npz')) for s in SEEDS], axis=0)
            for sysname, lab in (('S1', ('m1', 'P1')), ('S6', ('retrained', 'P3'))):
                vals = []
                for s in SEEDS:
                    with np.load(TEST_FINAL/f'ranks/{sysname}/seed{s}.npz') as z:
                        if not np.array_equal(z['inst'], ids): raise ValueError('locked rank alignment')
                        vals.append(nd(z['rank']))
                systems[lab] = np.mean(vals, axis=0)
        for key, v in systems.items():
            for g, m in groups.items():
                rows.append({'split': split, 'ranker': key[0], 'list': key[1], 'group': g, 'ndcg10': avg(v[m], users[m]), 'events': int(m.sum())})
        pairs = [(('armC_mlp', 'tt200'), ('armC_mlp', 'tt500')), (('din', 'tt200'), ('din', 'tt500')),
                 (('armC_mlp', 'tt200'), ('armC_mlp', 'fresh')), (('din', 'tt200'), ('din', 'fresh')),
                 (('armC_mlp', 'tt200'), ('armC_mlp', 'mixed')), (('din', 'tt200'), ('din', 'mixed')),
                 (('armC_mlp', 'tt200'), ('armC_mlp', 'fresh_s1')), (('din', 'tt200'), ('din', 'fresh_s1')),
                 (('p2ranker', 'P2'), ('p2ranker', 'P2f_s1')), (('retrieval', 'P2'), ('retrieval', 'P2f_s1')),
                 (('p2ranker', 'P2'), ('p2ranker', 'P2f')), (('retrieval', 'P2'), ('retrieval', 'P2f')),
                 (('m1', 'P1'), ('m1', 'P3')), (('m1', 'P3'), ('retrained', 'P3')), (('m1', 'P1'), ('retrained', 'P3'))]
        for a, b in pairs:
            if a in systems and b in systems:
                for g, m in groups.items():
                    ci = paired_bootstrap(users[m], systems[a][m], systems[b][m], n_resamples=1000, seed=0)
                    cons.append({'split': split, 'reference': '/'.join(a), 'candidate': '/'.join(b), 'group': g, **ci})
        for fl in ('P2f', 'P2f_s1'):  # frozen ranker gain minus retrieval-order gain (difference in differences)
            if ('p2ranker', fl) not in systems: continue
            dd = (systems[('p2ranker', fl)]-systems[('p2ranker', 'P2')])-(systems[('retrieval', fl)]-systems[('retrieval', 'P2')])
            suffix = '' if fl == 'P2f' else ' (seed 1)'
            for g, m in groups.items():
                ci = paired_bootstrap(users[m], np.zeros(m.sum()), dd[m], n_resamples=1000, seed=0)
                cons.append({'split': split, 'reference': f'P2f gain of retrieval order{suffix}', 'candidate': f'P2f gain of frozen P2 ranker{suffix}', 'group': g, **ci})
        log(report_split=split, systems=len(systems))
    if splits[-1][0] == 'test':
        verdicts(pd.DataFrame(rows), pd.DataFrame(cons))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'levels.csv', index=False); pd.DataFrame(cons).to_csv(OUT/'contrasts.csv', index=False)
    (OUT/'git_hash.txt').write_text(git_hash()+'\n')
    print(pd.DataFrame(rows).pivot_table(index=['split', 'ranker', 'list'], columns='group', values='ndcg10').mul(1000).round(3).to_string())
    print(pd.DataFrame(cons).assign(diff=lambda d: d['diff']*1000, ci_low=lambda d: d.ci_low*1000, ci_high=lambda d: d.ci_high*1000)
          [['split', 'reference', 'candidate', 'group', 'diff', 'ci_low', 'ci_high']].round(3).to_string())


def phase_fresh_seed1_val():
    """Second fresher-retriever seed, Val only: both families on its top 200 (no Test file is loaded)."""
    from cascade.analysis import freshness_val as F
    from cascade.analysis import pools_build as PB
    from cascade.pipelines import full_data_ranking as P
    from cascade.pipelines.build_candidates import retrieve
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    torch.set_num_threads(3)
    if json.loads((FRESH_RUN_S1/'metrics.json').read_text()).get('status') == 'running': raise ValueError('seed 1 incomplete')
    rec = {'label': 'Val only, exploratory follow-up', 'test_predictions_loaded': False, 'retriever': str(FRESH_RUN_S1.relative_to(REPO_ROOT))}
    # family A (candidate-file format)
    data, builder = din_context(); inst = data.split_instances('val'); ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
    f = retrieve(scorer_for(FRESH_RUN_S1, data), data, inst, 200)
    lists = as_lists(f['inst'], f['user'], f['cand'], f['score'], inst.item.to_numpy().astype(np.int32)); save_lists('fresh_s1', 'val', lists)
    rec['family_A_in_list'] = float((lists['target_pos'] >= 0).mean())
    save_ranks('val', 'fresh_s1', [200], ids, users, score_family(builder, lists, [200]))
    gc.collect(); torch.mps.empty_cache()
    # family B (archive format, P2 rankers, T0 common inputs with the P2 scaler)
    TF, PR, data, metas, p2_root, p2_meta = mlp_context(); inst = data.split_instances('val')
    tt_data = PB.load_data()
    if not np.array_equal(tt_data.split_instances('val').inst, inst.inst): raise ValueError('two-tower data differs')
    c, sc = fresh_lists(TF, tt_data, tt_data.split_instances('val'), FRESH_RUN_S1); del tt_data; gc.collect()
    target = inst.item.to_numpy(); rr, top = ranks_and_top10(sc, c, target)
    if not np.array_equal(c, lists['cand']): rec['family_A_B_list_rows_differing'] = int((c != lists['cand']).any(1).sum())
    d = BULK/'mlp_candidates/P2f_s1/val'; d.mkdir(parents=True, exist_ok=False)
    history = np.load(POOLS/'P2_two_tower/val/history.npy')
    for k, v in {'cand': c, 'score': sc, 'retrieval_rank': rr, 'top10': top, 'inst': inst.inst.to_numpy(), 'user': inst.user.to_numpy(),
                 'target': target, 'history': history}.items():
        np.save(d/f'{k}.npy', v)
    arc = P.arrays(d); hit = np.flatnonzero(np.isfinite(rr)); frame = inst.iloc[hit]
    feats, _ = TF.build_features(data, frame, np.asarray(arc['cand'][hit]), TF.cutoffs_for('T0', frame.ts.to_numpy(np.int64)), p2_scaler(p2_root), 'val P2f_s1')
    sub = F._sub_archive(arc, hit)
    for seed in SEEDS:
        r = np.full(len(inst), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
        dest = BULK/f'ranks/val/P2f_s1__p2ranker_seed{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, inst=inst.inst.to_numpy(), user=inst.user.to_numpy(), rank=r)
        log(scored='P2 ranker on P2f_s1', seed=seed)
    rec['family_B_in_list'] = float(np.isfinite(rr).mean())
    write_json(OUT/'fresh_seed1_val_checks.json', rec); log(**rec)


def phase_sasrec_ci():
    """Paired intervals for SASRec (all_train, 3 seeds) against saved Test systems; reuses saved ranks only."""
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.data.instances import ProcessedData
    data = ProcessedData.load(str(REPO_ROOT/'configs/data.yaml')); inst = data.split_instances('test')
    ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
    nd = lambda r: M.ndcg_at(np.asarray(r, float), 10)
    def aligned(path, key):
        with np.load(path) as z:
            order = pd.Series(np.arange(len(z['inst'])), index=z['inst']).loc[ids].to_numpy()
            return nd(z[key][order])
    sas = [aligned(REPO_ROOT/f'results/baselines/sasrec/all_train/seed{s}/test_ranks.npz', 'rank') for s in SEEDS]
    per_seed = [M.macro_micro(v, users)['macro'] for v in sas]; sas = np.mean(sas, axis=0)
    systems = {'natural MLP on two-tower top 200 (planned recipe, arm C)': np.mean([nd(np.load(BULK/'ranks/test/tt__armC_mlp.npz')['ranks'][s, 0]) for s in range(3)], axis=0),
               'DIN on two-tower top 200': np.mean([nd(np.load(BULK/'ranks/test/tt__din.npz')['ranks'][s, 0]) for s in range(3)], axis=0),
               'S6 mixed lists, ranker retrained': np.mean([aligned(TEST_FINAL/f'ranks/S6/seed{s}.npz', 'rank') for s in SEEDS], axis=0),
               'S8 mixed lists + timed statistics + co-review (best)': np.mean([aligned(TEST_FINAL/f'ranks/S8/seed{s}.npz', 'rank') for s in SEEDS], axis=0)}
    for name in ('tt__armC_mlp', 'tt__din'):
        if not np.array_equal(np.load(BULK/f'ranks/test/{name}.npz')['inst'], ids): raise ValueError('alignment')
    rows = [{'system': 'SASRec (all_train)', 'ndcg10': M.macro_micro(sas, users)['macro'], 'per_seed': per_seed}]
    for name, v in systems.items():
        ci = paired_bootstrap(users, sas, v, n_resamples=1000, seed=0)
        lvl = M.macro_micro(v, users)['macro']
        rows.append({'system': name, 'ndcg10': lvl, 'ratio_to_sasrec': lvl/rows[0]['ndcg10'], **ci})
    pd.DataFrame(rows).to_csv(OUT/'sasrec_paired_test.csv', index=False)
    print(pd.DataFrame(rows).to_string())


def phase_tt_r3_val():
    """Two-tower version of R3, Val only: the frozen two-tower-list score-plus-correction MLP (pool-ranking-P2
    bce_text, 3 seeds) on its own two-tower Val lists, with the same nine common inputs computed from reviews before
    T0 (as trained), before T1 (start of Val, the period-start rule) and before each query's month start. Scalers
    and weights frozen; only the statistics' date changes. Mirrors the recency-ranker R3 (locked Test S1-S3)."""
    from cascade.analysis import freshness_val as F
    from cascade.pipelines import full_data_ranking as P
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import first_ts_tensor
    torch.set_num_threads(3)
    TF, PR, data, metas, p2_root, p2_meta = mlp_context(); inst = data.split_instances('val')
    users = inst.user.to_numpy(); ids = inst.inst.to_numpy()
    p2 = P.arrays(POOLS/'P2_two_tower/val')
    hit = np.flatnonzero(np.isfinite(p2['retrieval_rank'])); frame = inst.iloc[hit]; sub = F._sub_archive(p2, hit)
    rec = {'label': 'Val only, exploratory follow-up (two-tower R3)', 'test_predictions_loaded': False, 'hit_rows': int(len(hit))}
    values = {}
    for arm in ('T0', 'T1', 'monthly'):
        feats, audit = TF.build_features(data, frame, np.asarray(p2['cand'][hit]), TF.cutoffs_for(arm, frame.ts.to_numpy(np.int64)), p2_scaler(p2_root), f'val tt-R3 {arm}')
        per = []
        for seed in SEEDS:
            r = np.full(len(inst), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
            if arm == 'T0':
                with np.load(p2_root/f'evaluation/bce_text/seed{seed}/val_ranks.npz') as z:
                    if not np.array_equal(z['rank'], r): raise ValueError('T0 replay failed')
            dest = BULK/f'ranks/val/P2_stats{arm}__p2ranker_seed{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, inst=ids, user=users, rank=r); per.append(M.ndcg_at(r, 10))
        values[arm] = np.mean(per, axis=0); log(arm=arm, ndcg10=M.macro_micro(values[arm], users)['macro'])
        del feats; gc.collect()
    new = first_ts_tensor(data).numpy()[inst.item.to_numpy()] >= T1
    groups = {'all': np.ones(len(ids), bool), 'new': new, 'warm': ~new}
    rows = []
    for arm, v in values.items():
        for g, m in groups.items():
            rows.append({'statistics': arm, 'group': g, 'ndcg10': M.macro_micro(v[m], users[m])['macro']})
    for a, b in (('T0', 'T1'), ('T1', 'monthly'), ('T0', 'monthly')):
        for g, m in groups.items():
            ci = paired_bootstrap(users[m], values[a][m], values[b][m], n_resamples=1000, seed=0)
            rows.append({'statistics': f'{b} - {a}', 'group': g, **ci})
    pd.DataFrame(rows).to_csv(OUT/'two_tower_r3_val.csv', index=False)
    write_json(OUT/'two_tower_r3_val_checks.json', {**rec, 'T0_replay_exact': True})
    print(pd.DataFrame(rows).to_string())


SWAP_DOC = REPO_ROOT/'docs/score_swap_val_2026-11-10.md'


def phase_score_swap_val():
    """Score-swap test for R1, Val only (predictions: ``docs/score_swap_val_2026-11-10.md``). Family-A rankers on the
    fresher retriever's products with the original two-tower score (C), with the fresher order but the original
    score values (D), and on the original products with the fresher scores (E). Resumable: saved arms are reused."""
    from cascade.pipelines.build_candidates import load_candidates
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import first_ts_tensor
    torch.set_num_threads(3)
    rec = {'label': 'Val only, exploratory (score swap)', 'test_predictions_loaded': False,
           'predictions_sha256': digest(SWAP_DOC), 'ranker_device': RANKER_DEVICE, 'git_hash': git_hash()}
    data, builder = din_context(); inst = data.split_instances('val')
    ids, users, target = inst.inst.to_numpy(), inst.user.to_numpy(), inst.item.to_numpy().astype(np.int32)
    tt = load_candidates(TT_DIR, 'val')
    if not np.array_equal(tt['inst'], ids): raise ValueError('Val order differs from the stored lists')
    tt_cand, tt_score = np.asarray(tt['cand'][:, :200]), np.asarray(tt['score'][:, :200])
    del tt; gc.collect()
    fresh = load_lists('fresh', 'val')
    if not np.array_equal(fresh['inst'], ids): raise ValueError('fresh list order')
    # check 1: the saved fresher-list ranks replay exactly on the first 4,096 rows
    sub = {k: v[:4096] for k, v in fresh.items()}
    again = score_family(builder, sub, [200])
    rec['fresh_replay_differences'] = {n: int(np.sum(np.load(BULK/f'ranks/val/fresh__{n}.npz')['ranks'][:, 0, :4096] != again[n][:, 0]))
                                       for n in DIN_RUNS}
    log(fresh_replay_differences=rec['fresh_replay_differences'])
    if any(rec['fresh_replay_differences'].values()): write_json(OUT/'score_swap_val_checks.json', rec); raise SystemExit('replay failed')

    def overlap_exact(cand_a, score_a, cand_b, score_b, n=4096):
        eq = cand_a[:n, :, None] == cand_b[:n, None, :]
        r, c = np.nonzero(eq.any(2)); j = eq.argmax(2)[r, c]
        return int(len(r)), bool(np.array_equal(score_a[:n][r, c], score_b[:n][r, j]))

    arms = {}
    if not (BULK/'lists/fresh_oldscore/val.npz').exists():
        scorer = scorer_for(TT_RUN, data)
        c = two_tower_scores_for(scorer, data, inst, fresh['cand']).astype(np.float16)
        del scorer; gc.collect(); torch.mps.empty_cache()
        rec['C_overlap_slots'], rec['C_overlap_score_exact'] = overlap_exact(fresh['cand'], c, tt_cand, tt_score)
        save_lists('fresh_oldscore', 'val', as_lists(ids, users, fresh['cand'], c, target))
    if not (BULK/'lists/tt_freshscore/val.npz').exists():
        fscorer = scorer_for(FRESH_RUN, data)
        e = two_tower_scores_for(fscorer, data, inst, tt_cand).astype(np.float16)
        del fscorer; gc.collect(); torch.mps.empty_cache()
        rec['E_overlap_slots'], rec['E_overlap_score_exact'] = overlap_exact(tt_cand, e, fresh['cand'], fresh['score'])
        save_lists('tt_freshscore', 'val', as_lists(ids, users, tt_cand, e, target))
    if not (BULK/'lists/fresh_qmap/val.npz').exists():
        order = np.argsort(-fresh['score'].astype(np.float32), axis=1, kind='stable')
        d = np.empty(fresh['score'].shape, np.float16)
        np.put_along_axis(d, order, -np.sort(-tt_score.astype(np.float32), axis=1).astype(np.float16), axis=1)
        save_lists('fresh_qmap', 'val', as_lists(ids, users, fresh['cand'], d, target))
    log(**{k: v for k, v in rec.items() if k.startswith(('C_', 'E_'))})
    if rec.get('C_overlap_score_exact') is False or rec.get('E_overlap_score_exact') is False:
        write_json(OUT/'score_swap_val_checks.json', rec); raise SystemExit('score regather check failed')
    for name in ('fresh_oldscore', 'fresh_qmap', 'tt_freshscore'):
        if all((BULK/f'ranks/val/{name}__{n}.npz').exists() for n in DIN_RUNS): continue
        lists = load_lists(name, 'val')
        save_ranks('val', name, [200], ids, users, score_family(builder, lists, [200]))
        del lists; gc.collect()

    # ---- report
    nd = lambda r: M.ndcg_at(np.asarray(r, float), 10)
    new = first_ts_tensor(data).numpy()[inst.item.to_numpy()] >= T1
    groups = {'all': np.ones(len(ids), bool), 'new': new, 'warm': ~new}
    arm_files = {'A_tt200': 'tt', 'B_fresh': 'fresh', 'C_fresh_oldscore': 'fresh_oldscore', 'D_fresh_qmap': 'fresh_qmap', 'E_tt_freshscore': 'tt_freshscore'}
    values, rows = {}, []
    for ranker in DIN_RUNS:
        for arm, f in arm_files.items():
            with np.load(BULK/f'ranks/val/{f}__{ranker}.npz') as z:
                if not np.array_equal(z['inst'], ids): raise ValueError('rank alignment')
                values[ranker, arm] = np.mean([nd(z['ranks'][s, 0]) for s in range(len(SEEDS))], axis=0)
    # retrieval order of each list's score column (no ranker); ties count in the target's favour
    def retrieval_order(cand, score):
        hit = cand == target[:, None]; found = hit.any(1)
        t = np.where(found, score.astype(np.float32)[np.arange(len(cand)), hit.argmax(1)], np.nan)
        r = np.where(found, 1+(score.astype(np.float32) > t[:, None]).sum(1), np.inf)
        return nd(r)
    lists_cache = {'A_tt200': (tt_cand, tt_score), 'B_fresh': (fresh['cand'], fresh['score'])}
    for arm in ('C_fresh_oldscore', 'D_fresh_qmap', 'E_tt_freshscore'):
        L = load_lists(arm_files[arm], 'val'); lists_cache[arm] = (L['cand'], L['score'])
    for arm, (c, s) in lists_cache.items():
        values['retrieval_order', arm] = retrieval_order(c, s)
    rec['in_list'] = {arm: float((c == target[:, None]).any(1).mean()) for arm, (c, _) in lists_cache.items()}
    for (ranker, arm), v in values.items():
        for g, m in groups.items():
            rows.append({'ranker': ranker, 'arm': arm, 'group': g, 'ndcg10_e3': 1e3*M.macro_micro(v[m], users[m])['macro']})
    pd.DataFrame(rows).to_csv(OUT/'score_swap_val_levels.csv', index=False)
    crow = []
    for ranker in (*DIN_RUNS, 'retrieval_order'):
        for cand_arm, ref_arm in (('C_fresh_oldscore', 'B_fresh'), ('D_fresh_qmap', 'B_fresh'), ('E_tt_freshscore', 'A_tt200'),
                                  ('C_fresh_oldscore', 'A_tt200'), ('D_fresh_qmap', 'A_tt200'), ('B_fresh', 'A_tt200')):
            for g, m in groups.items():
                ci = paired_bootstrap(users[m], values[ranker, ref_arm][m], values[ranker, cand_arm][m], n_resamples=1000, seed=0)
                crow.append({'ranker': ranker, 'contrast': f'{cand_arm} - {ref_arm}', 'group': g, **ci})
    cons = pd.DataFrame(crow); cons.to_csv(OUT/'score_swap_val_contrasts.csv', index=False)
    print(cons.to_string())
    verdict = {}
    for label, cand_arm, ref_arm, sign in (('S1', 'C_fresh_oldscore', 'B_fresh', +1), ('S2', 'D_fresh_qmap', 'B_fresh', +1),
                                           ('S3', 'E_tt_freshscore', 'A_tt200', -1), ('S4', 'C_fresh_oldscore', 'A_tt200', +1)):
        for ranker in DIN_RUNS:
            r = cons[(cons.ranker == ranker) & (cons.contrast == f'{cand_arm} - {ref_arm}') & (cons.group == 'all')].iloc[0]
            lo, hi = r.ci_low, r.ci_high
            held = lo > 0 if sign > 0 else hi < 0
            opposite = hi < 0 if sign > 0 else lo > 0
            verdict[f'{label} {ranker}'] = 'held' if held else ('not held' if opposite else 'inconclusive')
    rec['verdicts'] = verdict
    write_json(OUT/'score_swap_val_checks.json', rec); log(**rec)


def phase_lock():
    if not PROTOCOL.exists(): raise ValueError('write the protocol first')
    lock = BULK/'protocol_lock.json'
    if lock.exists(): raise ValueError('protocol already locked')
    write_json(lock, {'protocol_sha256': digest(PROTOCOL), 'locked_utc': pd.Timestamp.utcnow().isoformat(), 'git_hash': git_hash()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', required=True, choices=('din-val', 'mlp-check', 'lock', 'din-test', 'mlp-test', 'report', 'fresh-seed1-val', 'sasrec-ci', 'tt-r3-val', 'score-swap-val'))
    ap.add_argument('--ranker-device', default='cpu', help='device for the din-family rankers (the K200 anchor must replay exactly)')
    a = ap.parse_args()
    global RANKER_DEVICE
    RANKER_DEVICE = a.ranker_device
    BULK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    {'din-val': phase_din_val, 'mlp-check': phase_mlp_check, 'lock': phase_lock, 'din-test': phase_din_test,
     'mlp-test': phase_mlp_test, 'report': phase_report,
     'fresh-seed1-val': phase_fresh_seed1_val, 'sasrec-ci': phase_sasrec_ci, 'tt-r3-val': phase_tt_r3_val, 'score-swap-val': phase_score_swap_val}[a.phase]()
    log(phase=a.phase, seconds=round(time.perf_counter()-started))


if __name__ == '__main__':
    main()
