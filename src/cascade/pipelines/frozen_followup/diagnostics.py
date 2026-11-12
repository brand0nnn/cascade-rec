"""R1 diagnostics operations, extracted without changing the historical calculations."""
from __future__ import annotations

import gc
import json
import numpy as np
import pandas as pd
import torch
from cascade.utils.io import git_hash
from .runtime import digest, log, write_json
from .candidates import as_lists, load_lists, save_lists, save_ranks, two_tower_scores_for
from .matching import din_context, score_family, scorer_for
from .residual import fresh_lists, mlp_context, p2_scaler, score_p2


def phase_fresh_seed1_val(ctx):
    """Second fresher-retriever seed, Val only: both families on its top 200 (no Test file is loaded)."""
    from cascade.analysis import freshness_val as F
    from cascade.analysis import pools_build as PB
    from cascade.pipelines import full_data_ranking as P
    from cascade.pipelines.build_candidates import retrieve
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    torch.set_num_threads(3)
    if json.loads((ctx.fresh_run_s1/'metrics.json').read_text()).get('status') == 'running': raise ValueError('seed 1 incomplete')
    rec = {'label': 'Val only, exploratory follow-up', 'test_predictions_loaded': False, 'retriever': str(ctx.fresh_run_s1.relative_to(ctx.repo_root))}
    # family A (candidate-file format)
    data, builder = din_context(ctx); inst = data.split_instances('val'); ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
    f = retrieve(scorer_for(ctx, ctx.fresh_run_s1, data), data, inst, 200)
    lists = as_lists(f['inst'], f['user'], f['cand'], f['score'], inst.item.to_numpy().astype(np.int32)); save_lists(ctx, 'fresh_s1', 'val', lists)
    rec['family_A_in_list'] = float((lists['target_pos'] >= 0).mean())
    save_ranks(ctx, 'val', 'fresh_s1', [200], ids, users, score_family(ctx, builder, lists, [200]))
    gc.collect(); torch.mps.empty_cache()
    # family B (archive format, P2 rankers, T0 common inputs with the P2 scaler)
    TF, PR, data, metas, p2_root, p2_meta = mlp_context(); inst = data.split_instances('val')
    tt_data = PB.load_data()
    if not np.array_equal(tt_data.split_instances('val').inst, inst.inst): raise ValueError('two-tower data differs')
    c, sc = fresh_lists(ctx, TF, tt_data, tt_data.split_instances('val'), ctx.fresh_run_s1); del tt_data; gc.collect()
    target = inst.item.to_numpy(); rr, top = ranks_and_top10(sc, c, target)
    if not np.array_equal(c, lists['cand']): rec['family_A_B_list_rows_differing'] = int((c != lists['cand']).any(1).sum())
    d = ctx.bulk/'mlp_candidates/P2f_s1/val'; d.mkdir(parents=True, exist_ok=False)
    history = np.load(ctx.pools/'P2_two_tower/val/history.npy')
    for k, v in {'cand': c, 'score': sc, 'retrieval_rank': rr, 'top10': top, 'inst': inst.inst.to_numpy(), 'user': inst.user.to_numpy(),
                 'target': target, 'history': history}.items():
        np.save(d/f'{k}.npy', v)
    arc = P.arrays(d); hit = np.flatnonzero(np.isfinite(rr)); frame = inst.iloc[hit]
    feats, _ = TF.build_features(data, frame, np.asarray(arc['cand'][hit]), TF.cutoffs_for('T0', frame.ts.to_numpy(np.int64)), p2_scaler(p2_root), 'val P2f_s1')
    sub = F._sub_archive(arc, hit)
    for seed in ctx.seeds:
        r = np.full(len(inst), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
        dest = ctx.bulk/f'ranks/val/P2f_s1__p2ranker_seed{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, inst=inst.inst.to_numpy(), user=inst.user.to_numpy(), rank=r)
        log(scored='P2 ranker on P2f_s1', seed=seed)
    rec['family_B_in_list'] = float(np.isfinite(rr).mean())
    write_json(ctx.out/'fresh_seed1_val_checks.json', rec); log(**rec)


def phase_tt_r3_val(ctx):
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
    p2 = P.arrays(ctx.pools/'P2_two_tower/val')
    hit = np.flatnonzero(np.isfinite(p2['retrieval_rank'])); frame = inst.iloc[hit]; sub = F._sub_archive(p2, hit)
    rec = {'label': 'Val only, exploratory follow-up (two-tower R3)', 'test_predictions_loaded': False, 'hit_rows': int(len(hit))}
    values = {}
    for arm in ('T0', 'T1', 'monthly'):
        feats, audit = TF.build_features(data, frame, np.asarray(p2['cand'][hit]), TF.cutoffs_for(arm, frame.ts.to_numpy(np.int64)), p2_scaler(p2_root), f'val tt-R3 {arm}')
        per = []
        for seed in ctx.seeds:
            r = np.full(len(inst), np.inf); r[hit] = score_p2(p2_root, p2_meta, seed, sub, feats)
            if arm == 'T0':
                with np.load(p2_root/f'evaluation/bce_text/seed{seed}/val_ranks.npz') as z:
                    if not np.array_equal(z['rank'], r): raise ValueError('T0 replay failed')
            dest = ctx.bulk/f'ranks/val/P2_stats{arm}__p2ranker_seed{seed}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, inst=ids, user=users, rank=r); per.append(M.ndcg_at(r, 10))
        values[arm] = np.mean(per, axis=0); log(arm=arm, ndcg10=M.macro_micro(values[arm], users)['macro'])
        del feats; gc.collect()
    new = first_ts_tensor(data).numpy()[inst.item.to_numpy()] >= ctx.t1
    groups = {'all': np.ones(len(ids), bool), 'new': new, 'warm': ~new}
    rows = []
    for arm, v in values.items():
        for g, m in groups.items():
            rows.append({'statistics': arm, 'group': g, 'ndcg10': M.macro_micro(v[m], users[m])['macro']})
    for a, b in (('T0', 'T1'), ('T1', 'monthly'), ('T0', 'monthly')):
        for g, m in groups.items():
            ci = paired_bootstrap(users[m], values[a][m], values[b][m], n_resamples=1000, seed=0)
            rows.append({'statistics': f'{b} - {a}', 'group': g, **ci})
    pd.DataFrame(rows).to_csv(ctx.out/'two_tower_r3_val.csv', index=False)
    write_json(ctx.out/'two_tower_r3_val_checks.json', {**rec, 'T0_replay_exact': True})
    print(pd.DataFrame(rows).to_string())


def phase_score_swap_val(ctx):
    """Score-swap test for R1, Val only (predictions: ``docs/score_swap_val_2026-11-10.md``). Family-A rankers on the
    fresher retriever's products with the original two-tower score (C), with the fresher order but the original
    score values (D), and on the original products with the fresher scores (E). Resumable: saved arms are reused."""
    from cascade.pipelines.build_candidates import load_candidates
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import first_ts_tensor
    torch.set_num_threads(3)
    rec = {'label': 'Val only, exploratory (score swap)', 'test_predictions_loaded': False,
           'predictions_sha256': digest(ctx.swap_doc), 'ranker_device': ctx.ranker_device, 'git_hash': git_hash()}
    data, builder = din_context(ctx); inst = data.split_instances('val')
    ids, users, target = inst.inst.to_numpy(), inst.user.to_numpy(), inst.item.to_numpy().astype(np.int32)
    tt = load_candidates(ctx.tt_dir, 'val')
    if not np.array_equal(tt['inst'], ids): raise ValueError('Val order differs from the stored lists')
    tt_cand, tt_score = np.asarray(tt['cand'][:, :200]), np.asarray(tt['score'][:, :200])
    del tt; gc.collect()
    fresh = load_lists(ctx, 'fresh', 'val')
    if not np.array_equal(fresh['inst'], ids): raise ValueError('fresh list order')
    # check 1: the saved fresher-list ranks replay exactly on the first 4,096 rows
    sub = {k: v[:4096] for k, v in fresh.items()}
    again = score_family(ctx, builder, sub, [200])
    rec['fresh_replay_differences'] = {n: int(np.sum(np.load(ctx.bulk/f'ranks/val/fresh__{n}.npz')['ranks'][:, 0, :4096] != again[n][:, 0]))
                                       for n in ctx.din_runs}
    log(fresh_replay_differences=rec['fresh_replay_differences'])
    if any(rec['fresh_replay_differences'].values()): write_json(ctx.out/'score_swap_val_checks.json', rec); raise SystemExit('replay failed')

    def overlap_exact(cand_a, score_a, cand_b, score_b, n=4096):
        eq = cand_a[:n, :, None] == cand_b[:n, None, :]
        r, c = np.nonzero(eq.any(2)); j = eq.argmax(2)[r, c]
        return int(len(r)), bool(np.array_equal(score_a[:n][r, c], score_b[:n][r, j]))

    arms = {}
    if not (ctx.bulk/'lists/fresh_oldscore/val.npz').exists():
        scorer = scorer_for(ctx, ctx.tt_run, data)
        c = two_tower_scores_for(scorer, data, inst, fresh['cand']).astype(np.float16)
        del scorer; gc.collect(); torch.mps.empty_cache()
        rec['C_overlap_slots'], rec['C_overlap_score_exact'] = overlap_exact(fresh['cand'], c, tt_cand, tt_score)
        save_lists(ctx, 'fresh_oldscore', 'val', as_lists(ids, users, fresh['cand'], c, target))
    if not (ctx.bulk/'lists/tt_freshscore/val.npz').exists():
        fscorer = scorer_for(ctx, ctx.fresh_run, data)
        e = two_tower_scores_for(fscorer, data, inst, tt_cand).astype(np.float16)
        del fscorer; gc.collect(); torch.mps.empty_cache()
        rec['E_overlap_slots'], rec['E_overlap_score_exact'] = overlap_exact(tt_cand, e, fresh['cand'], fresh['score'])
        save_lists(ctx, 'tt_freshscore', 'val', as_lists(ids, users, tt_cand, e, target))
    if not (ctx.bulk/'lists/fresh_qmap/val.npz').exists():
        order = np.argsort(-fresh['score'].astype(np.float32), axis=1, kind='stable')
        d = np.empty(fresh['score'].shape, np.float16)
        np.put_along_axis(d, order, -np.sort(-tt_score.astype(np.float32), axis=1).astype(np.float16), axis=1)
        save_lists(ctx, 'fresh_qmap', 'val', as_lists(ids, users, fresh['cand'], d, target))
    log(**{k: v for k, v in rec.items() if k.startswith(('C_', 'E_'))})
    if rec.get('C_overlap_score_exact') is False or rec.get('E_overlap_score_exact') is False:
        write_json(ctx.out/'score_swap_val_checks.json', rec); raise SystemExit('score regather check failed')
    for name in ('fresh_oldscore', 'fresh_qmap', 'tt_freshscore'):
        if all((ctx.bulk/f'ranks/val/{name}__{n}.npz').exists() for n in ctx.din_runs): continue
        lists = load_lists(ctx, name, 'val')
        save_ranks(ctx, 'val', name, [200], ids, users, score_family(ctx, builder, lists, [200]))
        del lists; gc.collect()

    # ---- report
    nd = lambda r: M.ndcg_at(np.asarray(r, float), 10)
    new = first_ts_tensor(data).numpy()[inst.item.to_numpy()] >= ctx.t1
    groups = {'all': np.ones(len(ids), bool), 'new': new, 'warm': ~new}
    arm_files = {'A_tt200': 'tt', 'B_fresh': 'fresh', 'C_fresh_oldscore': 'fresh_oldscore', 'D_fresh_qmap': 'fresh_qmap', 'E_tt_freshscore': 'tt_freshscore'}
    values, rows = {}, []
    for ranker in ctx.din_runs:
        for arm, f in arm_files.items():
            with np.load(ctx.bulk/f'ranks/val/{f}__{ranker}.npz') as z:
                if not np.array_equal(z['inst'], ids): raise ValueError('rank alignment')
                values[ranker, arm] = np.mean([nd(z['ranks'][s, 0]) for s in range(len(ctx.seeds))], axis=0)
    # retrieval order of each list's score column (no ranker); ties count in the target's favour
    def retrieval_order(cand, score):
        hit = cand == target[:, None]; found = hit.any(1)
        t = np.where(found, score.astype(np.float32)[np.arange(len(cand)), hit.argmax(1)], np.nan)
        r = np.where(found, 1+(score.astype(np.float32) > t[:, None]).sum(1), np.inf)
        return nd(r)
    lists_cache = {'A_tt200': (tt_cand, tt_score), 'B_fresh': (fresh['cand'], fresh['score'])}
    for arm in ('C_fresh_oldscore', 'D_fresh_qmap', 'E_tt_freshscore'):
        L = load_lists(ctx, arm_files[arm], 'val'); lists_cache[arm] = (L['cand'], L['score'])
    for arm, (c, s) in lists_cache.items():
        values['retrieval_order', arm] = retrieval_order(c, s)
    rec['in_list'] = {arm: float((c == target[:, None]).any(1).mean()) for arm, (c, _) in lists_cache.items()}
    for (ranker, arm), v in values.items():
        for g, m in groups.items():
            rows.append({'ranker': ranker, 'arm': arm, 'group': g, 'ndcg10_e3': 1e3*M.macro_micro(v[m], users[m])['macro']})
    pd.DataFrame(rows).to_csv(ctx.out/'score_swap_val_levels.csv', index=False)
    crow = []
    for ranker in (*ctx.din_runs, 'retrieval_order'):
        for cand_arm, ref_arm in (('C_fresh_oldscore', 'B_fresh'), ('D_fresh_qmap', 'B_fresh'), ('E_tt_freshscore', 'A_tt200'),
                                  ('C_fresh_oldscore', 'A_tt200'), ('D_fresh_qmap', 'A_tt200'), ('B_fresh', 'A_tt200')):
            for g, m in groups.items():
                ci = paired_bootstrap(users[m], values[ranker, ref_arm][m], values[ranker, cand_arm][m], n_resamples=1000, seed=0)
                crow.append({'ranker': ranker, 'contrast': f'{cand_arm} - {ref_arm}', 'group': g, **ci})
    cons = pd.DataFrame(crow); cons.to_csv(ctx.out/'score_swap_val_contrasts.csv', index=False)
    print(cons.to_string())
    verdict = {}
    for label, cand_arm, ref_arm, sign in (('S1', 'C_fresh_oldscore', 'B_fresh', +1), ('S2', 'D_fresh_qmap', 'B_fresh', +1),
                                           ('S3', 'E_tt_freshscore', 'A_tt200', -1), ('S4', 'C_fresh_oldscore', 'A_tt200', +1)):
        for ranker in ctx.din_runs:
            r = cons[(cons.ranker == ranker) & (cons.contrast == f'{cand_arm} - {ref_arm}') & (cons.group == 'all')].iloc[0]
            lo, hi = r.ci_low, r.ci_high
            held = lo > 0 if sign > 0 else hi < 0
            opposite = hi < 0 if sign > 0 else lo > 0
            verdict[f'{label} {ranker}'] = 'held' if held else ('not held' if opposite else 'inconclusive')
    rec['verdicts'] = verdict
    write_json(ctx.out/'score_swap_val_checks.json', rec); log(**rec)
