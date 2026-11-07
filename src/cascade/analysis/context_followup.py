"""No-training follow-ups for the context arm (Val only, exploratory; selects nothing).

1. attribution: neutralise one context group (standardised value 0 = prefix-B mean) at a time;
2. leak re-check: newness buckets vs target rate, and gap(query - first review) among new candidates;
3. consistent snapshot: common columns at T1 (20261106-freshness-val) + context at T1.

Only natural-hit Val rows are re-scored (misses keep rank infinity). Scoring runs on CPU; the
full-context arm is re-scored too and compared with the saved MPS ranks.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, write_json
from cascade.pipelines import context_ranking as C
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.local_experiments import study_path
from cascade.utils.config import REPO_ROOT

OUT = REPO_ROOT/'results/analysis/20261106-context/followup'
FRESH = study_path('20261106-freshness-val')
GROUPS = {**C.ITEM_FIELDS, **C.USER_FIELDS}
CAND_PERIOD = ('review_count', 'mean_rating', 'item_age')


def columns(group):
    names = list(GROUPS)
    start = sum(GROUPS[n] for n in names[:names.index(group)])
    return np.arange(start, start+GROUPS[group])


def context_variant(base, groups):
    item, user = base['item'].clone(), base['user'].clone()
    for g in groups:
        cols = columns(g)
        if g in C.ITEM_FIELDS: item[:, cols] = 0
        else: user[:, cols-C.ITEM_DIM] = 0
    return {'item': item, 'user': user}


def score(model, sub, common, context, user_rows):
    """Logits for hit rows; ``user_rows`` index the full-Val user table."""
    out = np.empty(sub['cand'].shape, np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(out), 64):
            cand = torch.as_tensor(sub['cand'][s:s+64])
            item = context['item'][cand.long()]
            user = context['user'][torch.as_tensor(user_rows[s:s+64])][:, None, :].expand(-1, cand.shape[1], -1)
            com = torch.cat([torch.as_tensor(common[s:s+64]), item, user], -1)
            graph = torch.zeros((*cand.shape, C.GRAPH_DIM))
            out[s:s+64] = model(cand, torch.as_tensor(sub['history'][s:s+64]), torch.as_tensor(sub['score'][s:s+64]),
                                com, graph).numpy()
    return ranks_and_top10(out, sub['cand'], sub['target'])[0]


def main():
    torch.set_num_threads(3)
    OUT.mkdir(parents=True, exist_ok=True)
    root = study_path(C.FULL_STUDY)
    meta, source_root, source_meta = C.verify(root, artifacts=False)
    data = fdr.data_for(source_meta)
    inst = data.split_instances('val')
    users = inst.user.to_numpy()
    archive = fdr.arrays(source_root/'candidates/val')
    hit = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    sub = {k: np.asarray(archive[k][hit]) for k in ('cand', 'score', 'history', 'target')}
    common_t0 = np.asarray(np.load(source_root/'features/val/common.npy', mmap_mode='r')[hit])
    common_t1 = np.load(FRESH/'features/T1/common.npy')
    if common_t1.shape != common_t0.shape: raise ValueError('freshness T1 rows differ')
    if not np.array_equal(common_t1[..., 3:7], common_t0[..., 3:7]): raise ValueError('non-graph common columns differ')
    base = C.load_context(root, 'val', 'cpu')
    first = np.full(data.n_items+1, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    t1 = data.split['t1']
    new = first[inst.item.to_numpy()] >= t1
    arms = {'full': ([], common_t0), **{f'no_{g}': ([g], common_t0) for g in GROUPS},
            'no_candidate_period': (list(CAND_PERIOD), common_t0), 'consistent_T1': ([], common_t1)}
    ranks, repro = {a: [] for a in arms}, []
    started = time.perf_counter()
    for seed in meta['seeds']:
        model, _ = C.build_model(source_meta, source_root, seed, 'cpu')
        model.load_state_dict(torch.load(root/f'runs/{C.METHOD}/seed{seed}/model.pth', map_location='cpu', weights_only=True))
        for arm, (groups, common) in arms.items():
            full = np.full(len(inst), np.inf)
            full[hit] = score(model, sub, common, context_variant(base, groups), hit)
            ranks[arm].append(full)
            if arm == 'full':
                with np.load(root/f'evaluation/{C.METHOD}/seed{seed}/val_ranks.npz') as z: saved = z['rank']
                repro.append({'seed': seed, 'cpu_equals_saved_mps': bool(np.array_equal(full, saved)),
                              'mismatches': int((full != saved).sum()), 'max_top10_flips': int(((full <= 10) != (saved <= 10)).sum())})
            print(json.dumps({'seed': seed, 'arm': arm, 'seconds': time.perf_counter()-started}), flush=True)
    for arm, r in ranks.items():
        np.savez_compressed(OUT/f'ranks_{arm}.npz', inst=archive['inst'], rank=np.stack(r))
    ref = {'bce_text_T0': [], 'bce_text_T1': []}
    for s in meta['seeds']:
        with np.load(source_root/f'evaluation/bce_text/seed{s}/val_ranks.npz') as z: ref['bce_text_T0'].append(z['rank'])
        with np.load(FRESH/f'ranks/T1/bce_text/seed{s}.npz') as z:
            if not np.array_equal(z['inst'], archive['inst']): raise ValueError('freshness alignment')
            ref['bce_text_T1'].append(z['rank'])
    allranks = {**{k: np.stack(v) for k, v in ranks.items()}, **{k: np.stack(v) for k, v in ref.items()}}
    masks = {'all': np.ones(len(inst), bool), 'new_in_val': new, 'warm_in_val': ~new}
    gap = inst.ts.to_numpy()-first[inst.item.to_numpy()]
    masks['new_same_ts'] = new & (gap == 0)
    masks['new_gap_gt0'] = new & (gap > 0)
    metric_fns = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall10': lambda r: M.recall_at(r, 10)}
    vals = {k: {m: np.stack([f(x) for x in v]).mean(0) for m, f in metric_fns.items()} for k, v in allranks.items()}
    levels = []
    for g, mask in masks.items():
        for k in allranks:
            levels.append({'group': g, 'arm': k, 'events': int(mask.sum()),
                           **{m: average(vals[k][m][mask], users[mask]) for m in metric_fns}})
    contrasts = [('full', a) for a in arms if a != 'full'] + [
        ('bce_text_T1', 'consistent_T1'), ('bce_text_T0', 'consistent_T1'), ('bce_text_T0', 'full'),
        ('bce_text_T1', 'full'), ('bce_text_T0', 'bce_text_T1')]
    diffs = []
    for g, mask in masks.items():
        for a, b in contrasts:
            for m in metric_fns:
                if m == 'recall10' and not a.startswith('bce_text'): continue
                ci = paired_bootstrap(users[mask], vals[a][m][mask], vals[b][m][mask], n_resamples=1000, seed=0)
                diffs.append({'group': g, 'metric': m, 'reference': a, 'candidate': b, **ci})
    pd.DataFrame(levels).to_csv(OUT/'levels.csv', index=False)
    pd.DataFrame(diffs).to_csv(OUT/'paired_differences.csv', index=False)
    leak = leak_recheck(root, archive, inst, first, t1)
    write_json(OUT/'leak_recheck.json', leak)
    write_json(OUT/'provenance.json', {'label': 'Val only, exploratory, no training', 'device': 'cpu',
        'hit_rows_rescored': int(len(hit)), 'reproduction': repro, 'test_predictions_loaded': False,
        'common_T1_source': str((FRESH/'features/T1/common.npy').relative_to(REPO_ROOT)),
        'neutralisation': 'standardised value 0 = chronological-prefix B mean'})
    print(json.dumps({'reproduction': repro}), flush=True)


def leak_recheck(root, archive, inst, first, t1):
    """Newness buckets vs target rate; gap among new candidates; context constancy across gaps."""
    codes = np.load(root/'context/val/item_codes.npy').astype(np.int64)
    table = np.load(root/'context/val/item_table.npy')
    query = inst.ts.to_numpy()
    edges = [(0, 0), (1, DAY), (DAY+1, 7*DAY), (7*DAY+1, 30*DAY), (30*DAY+1, 90*DAY), (90*DAY+1, np.inf)]
    labels = ['0 (same ts)', '(0,1d]', '(1d,7d]', '(7d,30d]', '(30d,90d]', '>90d']
    fields = ('price',)+CAND_PERIOD
    totals = {f: [np.zeros(GROUPS[f], np.int64), np.zeros(GROUPS[f], np.int64)] for f in fields}
    by_gap = {l: {'candidates': 0, 'targets': 0, 'items': []} for l in labels}
    for s in range(0, len(query), 8192):
        cand = np.asarray(archive['cand'][s:s+8192])
        real = cand > 0
        is_t = real & (cand == np.asarray(archive['target'][s:s+8192])[:, None])
        for i, f in enumerate(fields):
            totals[f][0] += np.bincount(codes[cand[real], i], minlength=GROUPS[f])
            totals[f][1] += np.bincount(codes[cand[is_t], i], minlength=GROUPS[f])
        new = real & (first[cand] >= t1)
        gap = query[s:s+8192, None]-first[cand]
        for (lo, hi), l in zip(edges, labels):
            m = new & (gap >= lo) & (gap <= hi)
            by_gap[l]['candidates'] += int(m.sum()); by_gap[l]['targets'] += int((m & is_t).sum())
            by_gap[l]['items'].append(np.unique(cand[m]))
    time_cols = slice(C.ITEM_FIELDS['price'], C.ITEM_DIM)
    out = {'per_bucket_target_rate': {f: [{'level': i, 'candidates': int(n), 'targets': int(k),
                                          'target_rate': float(k/n) if n else None}
                                         for i, (n, k) in enumerate(zip(*totals[f]))] for f in fields},
           'new_candidates_by_gap': []}
    period_rows = []
    for l, v in by_gap.items():
        items = np.unique(np.concatenate(v['items']))
        rows = np.unique(table[items][:, time_cols], axis=0) if len(items) else np.zeros((0, 1))
        period_rows.append(rows)
        out['new_candidates_by_gap'].append({'gap': l, 'candidates': v['candidates'], 'targets': v['targets'],
            'target_rate': v['targets']/v['candidates'] if v['candidates'] else None,
            'distinct_period_column_rows': int(len(rows))})
    merged = np.unique(np.concatenate([r for r in period_rows if len(r)]), axis=0)
    out['period_columns_constant_across_gaps'] = bool(len(merged) == 1)
    out['note'] = ('item columns depend only on the item and the period start; the gap to the query is not an input. '
                   'User columns are constant within a list.')
    return out


DAY = C.DAY_MS

if __name__ == '__main__':
    main()
