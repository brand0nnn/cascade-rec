"""R1 matching operations, extracted without changing the historical calculations."""
from __future__ import annotations

import gc
import json
import numpy as np
import torch
from .runtime import log, protocol_guard, record_test_open, write_json
from .candidates import as_lists, mixed_lists, save_lists, save_ranks, two_tower_scores_for


def din_context(ctx):
    from cascade.data.instances import ProcessedData
    from cascade.data.ranker_features import RankerFeatureBuilder
    from cascade.pipelines.build_candidates import load_candidates
    data = ProcessedData.load(str(ctx.repo_root/'configs/data.yaml'))
    builder = RankerFeatureBuilder.create(data, load_candidates(ctx.tt_dir, 'ranker_train'))
    return data, builder


def scorer_for(ctx, run, data):
    from cascade.models.retriever import RetrieverScorer
    from cascade.pipelines.train_retriever import load_retriever
    model, _ = load_retriever(str(run), data, ctx.device_retrieval)
    return RetrieverScorer(model)


def load_ranker(run, builder):
    from cascade.pipelines.index_rank_eval import load_ranker as lr
    rec = json.loads((run/'metrics.json').read_text())
    if rec.get('development_sample') or rec.get('status') == 'running': raise ValueError(f'not a completed full-data fit: {run}')
    return lr(run, builder)


def score_family(ctx, builder, lists, counts, device=None):
    device = device or ctx.ranker_device
    from cascade.pipelines.ranker_k1_sweep import score_prefixes
    out = {}
    for name, runs in ctx.din_runs.items():
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


def phase_din_val(ctx):
    from cascade.pipelines.build_candidates import load_candidates, retrieve
    torch.set_num_threads(3)
    data, builder = din_context(ctx); inst = data.split_instances('val')
    users, ids = inst.user.to_numpy(), inst.inst.to_numpy()
    rec = {'label': 'Val only, exploratory follow-up', 'test_predictions_loaded': False, 'ranker_device': ctx.ranker_device}
    tt = load_candidates(ctx.tt_dir, 'val')  # stored two-tower top 500, never forced
    if not np.array_equal(tt['inst'], ids): raise ValueError('Val order differs from the stored lists')
    # anchor first: frozen rankers on the stored two-tower K200 Val lists reproduce their saved ranks exactly
    ranks = score_family(ctx, builder, tt, [200, 500])
    rec['anchor_rank_differences'] = {}
    for name, runs in ctx.din_runs.items():
        for si, run in enumerate(runs):
            with np.load(run/'val_ranks.npz') as z:
                if not np.array_equal(z['inst'], ids): raise ValueError('stored rank order')
                rec['anchor_rank_differences'][f'{name}_seed{si}'] = int(np.sum(z['ranks'] != ranks[name][si, 0]))
    log(anchor=rec['anchor_rank_differences'])
    if any(rec['anchor_rank_differences'].values()): write_json(ctx.out/'din_val_checks.json', rec); raise SystemExit('anchor failed')
    save_ranks(ctx, 'val', 'tt', [200, 500], ids, users, ranks)
    # check: the retrieval path reproduces the stored two-tower Val lists and their float16 scores
    scorer = scorer_for(ctx, ctx.tt_run, data); rows = np.arange(2048)
    again = retrieve(scorer, data, inst.iloc[rows], 500, show_progress=False)
    rec['tt_rebuild_exact'] = bool(np.array_equal(again['cand'], tt['cand'][rows]) and np.array_equal(again['score'], tt['score'][rows]))
    sc = two_tower_scores_for(scorer, data, inst.iloc[rows], tt['cand'][rows]).astype(np.float16)
    rec['tt_score_regather_exact'] = bool(np.array_equal(sc, tt['score'][rows]))
    log(tt_rebuild_exact=rec['tt_rebuild_exact'], tt_score_regather_exact=rec['tt_score_regather_exact'])
    if not (rec['tt_rebuild_exact'] and rec['tt_score_regather_exact']): write_json(ctx.out/'din_val_checks.json', rec); raise SystemExit(rec)
    # mixed lists (Brandon's P3 Val pool, the same lists as the mlp family), each candidate with its two-tower score
    p3 = {k: np.load(ctx.pools/f'P3_fused_rrf/val/{k}.npy', mmap_mode='r') for k in ('inst', 'cand')}
    mixed, _ = mixed_lists(data, scorer, inst, p3); save_lists(ctx, 'mixed', 'val', mixed)
    sub = np.arange(4096)
    eq = mixed['cand'][sub, :, None] == tt['cand'][sub, None, :]
    r, c = np.nonzero(eq.any(2)); j = eq.argmax(2)[r, c]
    rec['mixed_overlap_slots'] = int(len(r))
    rec['mixed_overlap_score_exact'] = bool(np.array_equal(mixed['score'][sub][r, c], tt['score'][sub][r, j]))
    del scorer; gc.collect(); torch.mps.empty_cache()
    # fresher retriever top 200 in the candidate-file format; must equal the fresh study's Val pool
    fscorer = scorer_for(ctx, ctx.fresh_run, data)
    f = retrieve(fscorer, data, inst, 200)
    fresh = as_lists(f['inst'], f['user'], f['cand'], f['score'], inst.item.to_numpy().astype(np.int32)); save_lists(ctx, 'fresh', 'val', fresh)
    rec['fresh_equals_fresh_study_pool'] = bool(np.array_equal(np.load(ctx.fresh_val/'inst.npy'), fresh['inst'])
                                                and np.array_equal(np.load(ctx.fresh_val/'cand.npy'), fresh['cand']))
    del fscorer; gc.collect(); torch.mps.empty_cache()
    rec['recall_in_list'] = {'tt200': float(((tt['target_pos'] >= 0) & (tt['target_pos'] < 200)).mean()),
                             'tt500': float((tt['target_pos'] >= 0).mean()), 'fresh': float((fresh['target_pos'] >= 0).mean()),
                             'mixed': float((mixed['target_pos'] >= 0).mean())}
    log(**{k: v for k, v in rec.items() if k != 'anchor_rank_differences'})
    for list_name, lists in (('fresh', fresh), ('mixed', mixed)):
        save_ranks(ctx, 'val', list_name, [200], ids, users, score_family(ctx, builder, lists, [200]))
    write_json(ctx.out/'din_val_checks.json', rec)


def phase_din_test(ctx):
    from cascade.pipelines.build_candidates import retrieve
    protocol_guard(ctx); torch.set_num_threads(3)
    data, builder = din_context(ctx); inst = data.split_instances('test')
    if (inst.ts.to_numpy(np.int64) < ctx.t2).any(): raise ValueError('non-Test query')
    record_test_open(ctx, 'din')
    ids, users, target = inst.inst.to_numpy(), inst.user.to_numpy(), inst.item.to_numpy().astype(np.int32)
    scorer = scorer_for(ctx, ctx.tt_run, data)
    t = retrieve(scorer, data, inst, 500)
    tt = as_lists(t['inst'], t['user'], t['cand'], t['score'], target); save_lists(ctx, 'tt', 'test', tt)
    p3 = {k: np.load(ctx.test_final/f'candidates/P3/test/{k}.npy', mmap_mode='r') for k in ('inst', 'cand')}
    mixed, _ = mixed_lists(data, scorer, inst, p3); save_lists(ctx, 'mixed', 'test', mixed)
    del scorer; gc.collect(); torch.mps.empty_cache()
    fscorer = scorer_for(ctx, ctx.fresh_run, data)
    f = retrieve(fscorer, data, inst, 200)
    fresh = as_lists(f['inst'], f['user'], f['cand'], f['score'], target); save_lists(ctx, 'fresh', 'test', fresh)
    del fscorer; gc.collect(); torch.mps.empty_cache()
    p2 = np.load(ctx.test_final/'candidates/P2/test/cand.npy', mmap_mode='r'); p2i = np.load(ctx.test_final/'candidates/P2/test/inst.npy')
    write_json(ctx.bulk/'din_test_lists_check.json', {
        'tt_top200_equals_locked_test_P2': bool(np.array_equal(p2i, ids) and np.array_equal(np.asarray(p2), tt['cand'][:, :200])),
        'recall200': {'tt': float(((tt['target_pos'] >= 0) & (tt['target_pos'] < 200)).mean()), 'tt500': float((tt['target_pos'] >= 0).mean()),
                      'fresh': float((fresh['target_pos'] >= 0).mean()), 'mixed': float((mixed['target_pos'] >= 0).mean())}})
    save_ranks(ctx, 'test', 'tt', [200, 500], ids, users, score_family(ctx, builder, tt, [200, 500]))
    for list_name, lists in (('fresh', fresh), ('mixed', mixed)):
        save_ranks(ctx, 'test', list_name, [200], ids, users, score_family(ctx, builder, lists, [200]))
