"""Step A0: candidate pools for a fixed ranker (Val only, exploratory).

Builds and verifies three K200 candidate pools for ranker_train (B) and Val in the archive format of
``results/local_experiments/20261103-full-promising-ranking/candidates/<split>/`` (cand, score, inst, user,
target, retrieval_rank, top10, history), so the existing feature builder can consume them:

  P1  frozen MiniLM + 90-day recency, natural top 200 (referenced in place, not copied)
  P2  frozen two-tower (text) seed 0, natural top 200, float32 scores regenerated from the checkpoint
  P3  fixed round-robin interleave (recency r1, two-tower r1, recency r2, ...), duplicates skipped, until
      200 unique items; score = RRF sum over sources containing the item of 1/(60 + rank_in_source)

Then reports retrieval-stage metrics and a target-conditioned retriever oracle (upper bound only).

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.pools_build {verify,regen,fuse,filters,report}

HARD RULES: only the ranker_train and val splits are ever loaded; no Test file is opened. Nothing is selected.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.eval.saved_results import digest, write_json
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

SPLITS = ('ranker_train', 'val')
K = 200
RRF_K = 60
BULK = REPO_ROOT / 'results/local_experiments/20261106-pools'
COMPACT = REPO_ROOT / 'results/analysis/20261106-pools'
RECENCY = REPO_ROOT / 'results/local_experiments/20261103-full-promising-ranking/candidates'
TT_RUN = REPO_ROOT / 'results/retriever/two_tower_text/seed0'
TT_SAVED = REPO_ROOT / 'data/processed/candidates/two_tower_text_seed0'
ARCHIVE_KEYS = ('cand', 'score', 'inst', 'user', 'target', 'retrieval_rank', 'top10', 'history')
LABEL = 'Val only, exploratory'
N_BOOT, BOOT_SEED = 1000, 0


def _guard(path: Path) -> Path:
    """Refuse any path that names a Test artifact."""
    low = str(path).lower()
    if 'test' in Path(low).name or '/test/' in low:
        raise ValueError(f'refusing to touch a Test artifact: {path}')
    return path


def load_archive(directory: Path) -> dict:
    return {k: np.load(_guard(directory / f'{k}.npy'), mmap_mode='r') for k in ARCHIVE_KEYS}


def pool_ranks(scores: np.ndarray, cand: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pessimistic target rank within the pool (inf if absent) and the stable-sorted top 10.

    Same rule as ``local_evidence_export.ranks_and_top10`` used by the recency archive.
    """
    matches = cand == targets[:, None]
    found = matches.any(1)
    column = matches.argmax(1)
    target = scores[np.arange(len(scores)), column]
    ranks = np.where(found, (scores >= target[:, None]).sum(1), np.inf)
    order = np.argsort(-scores, axis=1, kind='stable')[:, :10]
    return ranks.astype(np.float64), np.take_along_axis(cand, order, axis=1).astype(np.int32)


# ----------------------------------------------------------------------------------------------- fusion rule

def interleave_reference(rec: list[int], tt: list[int], k: int = K, rrf_k: int = RRF_K):
    """Plain-Python reference of the fixed P3 rule (used by tests)."""
    seen, pool = set(), []
    for r, t in zip(rec, tt):
        for item in (r, t):
            if item not in seen and len(pool) < k:
                seen.add(item)
                pool.append(item)
    rec_rank = {item: i + 1 for i, item in enumerate(rec)}
    tt_rank = {item: i + 1 for i, item in enumerate(tt)}
    score = [(1 / (rrf_k + rec_rank[i]) if i in rec_rank else 0.0) + (1 / (rrf_k + tt_rank[i]) if i in tt_rank else 0.0)
             for i in pool]
    return pool, score, [rec_rank.get(i, 0) for i in pool], [tt_rank.get(i, 0) for i in pool]


def interleave_fuse(rec: np.ndarray, tt: np.ndarray, k: int = K, rrf_k: int = RRF_K) -> dict[str, np.ndarray]:
    """Vectorised P3 rule over rows. ``rec``/``tt`` [n, m] are rank-ordered unique positive item ids.

    Returns cand [n, k] (interleave order), RRF score float32, and 1-based ranks in each source (0 = absent
    from that source's m items).
    """
    rec, tt = np.asarray(rec), np.asarray(tt)
    n, m = rec.shape
    if tt.shape != (n, m) or (rec <= 0).any() or (tt <= 0).any():
        raise ValueError('sources must be equal-shaped lists of positive item ids')
    seq = np.empty((n, 2 * m), dtype=np.int64)
    seq[:, 0::2], seq[:, 1::2] = rec, tt
    own = np.repeat(np.arange(1, m + 1), 2)[None, :]          # rank of the slot within its own source
    is_rec = np.tile(np.array([True, False]), m)[None, :]
    rec_rank = np.where(is_rec, own, 0).repeat(n, 0)
    tt_rank = np.where(~is_rec, own, 0).repeat(n, 0)
    order = np.argsort(seq, axis=1, kind='stable')
    s = np.take_along_axis(seq, order, axis=1)
    dup = np.zeros_like(s, dtype=bool)
    dup[:, 1:] = s[:, 1:] == s[:, :-1]
    if (dup[:, 1:] & dup[:, :-1]).any():
        raise ValueError('an item appears more than twice: sources are not unique lists')
    rows, later = np.nonzero(dup)
    first_pos, later_pos = order[rows, later - 1], order[rows, later]   # stable sort: first_pos < later_pos
    # cross-fill the other source's rank onto the first occurrence; mark the later occurrence as skipped
    rec_rank[rows, first_pos] = np.maximum(rec_rank[rows, first_pos], rec_rank[rows, later_pos])
    tt_rank[rows, first_pos] = np.maximum(tt_rank[rows, first_pos], tt_rank[rows, later_pos])
    keep = np.ones_like(seq, dtype=bool)
    keep[rows, later_pos] = False
    keep &= np.cumsum(keep, axis=1) <= k
    if not (keep.sum(1) == k).all():
        raise ValueError('fewer than k unique items')
    cols = np.nonzero(keep)[1].reshape(n, k)
    cand = np.take_along_axis(seq, cols, axis=1)
    rr = np.take_along_axis(rec_rank, cols, axis=1)
    tr = np.take_along_axis(tt_rank, cols, axis=1)
    score = (np.where(rr > 0, 1.0 / (rrf_k + rr), 0.0) + np.where(tr > 0, 1.0 / (rrf_k + tr), 0.0)).astype(np.float32)
    return {'cand': cand.astype(np.int32), 'score': score, 'rec_rank': rr.astype(np.int16), 'tt_rank': tr.astype(np.int16)}


def filter_violations(cand: np.ndarray, ts: np.ndarray, hist_rows: np.ndarray, hist_items: np.ndarray,
                      first_ts: np.ndarray) -> dict[str, int]:
    """Count §4.6 violations: padding, duplicates, history items (full history before t), unreleased items."""
    cand = np.asarray(cand)
    ordered = np.sort(cand, axis=1)
    hist = np.zeros(cand.shape, dtype=bool)
    if len(hist_rows):
        key = np.unique(hist_rows.astype(np.int64) * (1 << 32) + hist_items.astype(np.int64))
        flat = (np.arange(len(cand))[:, None].astype(np.int64) * (1 << 32) + cand.astype(np.int64))
        hist = np.isin(flat, key)
    return {'padding': int((cand <= 0).sum()),
            'duplicates': int((np.diff(ordered, axis=1) == 0).sum()),
            'history_items': int(hist.sum()),
            'unreleased_items': int((first_ts[cand] > ts[:, None]).sum())}


# ------------------------------------------------------------------------------------------------ steps

def load_data():
    from cascade.pipelines.frozen_text_study import checked
    from cascade.pipelines.local_experiments import study_path
    return checked(study_path('20261102-full-frozen-text'))[0]


def saved_two_tower(split: str) -> dict[str, np.ndarray]:
    if split not in SPLITS:
        raise ValueError('only ranker_train and val may be loaded')
    with np.load(_guard(TT_SAVED / f'{split}.npz')) as z:
        out = {k: z[k] for k in z.files if k != 'cand' and k != 'score'}
        out['cand_full'] = z['cand']
        out['score_full'] = z['score']
    return out


def verify() -> None:
    """Task 1: structure of the saved two-tower lists and alignment with the recency archive."""
    result = {'label': LABEL}
    for split in SPLITS:
        z = saved_two_tower(split)
        rec = load_archive(RECENCY / split)
        cand = z['cand_full']
        nat, nat_s = cand[:, :K], z['score_full'][:, :K].astype(np.float32)
        target = np.asarray(rec['target'])
        r = {'rows': int(len(cand)), 'columns': int(cand.shape[1]),
             'inst_equal_recency': bool(np.array_equal(z['inst'], rec['inst'])),
             'user_equal_recency': bool(np.array_equal(z['user'], rec['user'])),
             'natural_zero_padding': int((nat <= 0).sum()),
             'natural_duplicate_rows': int((np.diff(np.sort(nat, 1), axis=1) == 0).any(1).sum()),
             'natural_score_not_nonincreasing_rows': int((np.diff(nat_s, axis=1) > 0).any(1).sum())}
        in_nat = (nat == target[:, None])
        found = in_nat.any(1)
        pos = z['target_pos']
        trank = z['target_rank']
        if split == 'ranker_train':
            forced = z['forced'].astype(bool)
            r.update(forced_rows=int(forced.sum()), forced_fraction=float(forced.mean()),
                     forced_target_pos_not_200=int((pos[forced] != K).sum()),
                     forced_col200_not_target=int((cand[forced, K] != target[forced]).sum()),
                     forced_target_in_natural=int(found[forced].sum()),
                     nonforced_col200_nonzero=int((cand[~forced, K] != 0).sum()),
                     nonforced_target_not_in_natural=int((~found[~forced]).sum()),
                     forced_equals_not_found=bool(np.array_equal(forced, ~found)))
        else:
            forced = np.zeros(len(cand), bool)
        nf = ~forced & found
        col = in_nat.argmax(1)
        r.update(natural_hits_top200=int(found.sum()),
                 target_pos_equals_column=int((pos[nf] == col[nf]).sum()),
                 target_pos_mismatch=int((pos[nf] != col[nf]).sum()),
                 target_rank_equals_pos_plus1=int((trank[nf] == col[nf] + 1).sum()),
                 target_rank_gt_pos_plus1_ties=int((trank[nf] > col[nf] + 1).sum()),
                 target_rank_lt_pos_plus1=int((trank[nf] < col[nf] + 1).sum()),
                 found_vs_target_rank_le200_mismatch=int((found != (trank <= K)).sum()),
                 val_list_columns_beyond_200_contain_target=int((cand[:, K:] == target[:, None]).any(1).sum())
                 if split == 'val' else None)
        result[split] = r
        del z, cand, nat, nat_s
    passed = all(result[s]['inst_equal_recency'] and result[s]['user_equal_recency'] and result[s]['natural_zero_padding'] == 0
                 and result[s]['natural_duplicate_rows'] == 0 and result[s]['natural_score_not_nonincreasing_rows'] == 0
                 and result[s]['target_pos_mismatch'] == 0 for s in SPLITS)
    b = result['ranker_train']
    passed &= (b['forced_target_pos_not_200'] == 0 and b['forced_col200_not_target'] == 0 and b['forced_target_in_natural'] == 0
               and b['nonforced_col200_nonzero'] == 0 and b['nonforced_target_not_in_natural'] == 0)
    result['passed'] = bool(passed)
    write_json(COMPACT / 'verify_saved_two_tower.json', result)
    print(json.dumps(result, indent=1), flush=True)


def regen(device: str, batch_size: int = 512) -> None:
    """Task 2: float32 two-tower top-200 from the frozen checkpoint, with §4.6 filtering; writes P2."""
    from cascade.models.retriever import RetrieverScorer
    from cascade.pipelines.local_evidence_export import check_candidates
    from cascade.pipelines.train_retriever import load_retriever
    torch.set_num_threads(3)
    data = load_data()
    model, cfg = load_retriever(str(TT_RUN), data, device)
    scorer = RetrieverScorer(model)
    first = first_ts_tensor(data)
    first_np = first.numpy()
    summary = {'label': LABEL, 'device': device, 'checkpoint_sha256': digest(TT_RUN / 'model.pth'),
               'text_emb_sha256': digest(data.dir / 'item_text_emb.npy'), 'config': cfg}
    for split in SPLITS:
        inst = data.split_instances(split)
        rec = load_archive(RECENCY / split)
        if not (np.array_equal(inst.inst.to_numpy(), rec['inst']) and np.array_equal(inst.item.to_numpy(), rec['target'])):
            raise ValueError('instance order differs from the recency archive')
        z = saved_two_tower(split)
        saved_c = np.ascontiguousarray(z['cand_full'][:, :K]); saved_s = np.ascontiguousarray(z['score_full'][:, :K])
        saved_trank = z['target_rank']
        del z
        out = BULK / 'candidates/P2_two_tower' / split
        out.mkdir(parents=True, exist_ok=False)
        cand = open_memmap(out / 'cand.npy', mode='w+', dtype=np.int32, shape=(len(inst), K))
        score = open_memmap(out / 'score.npy', mode='w+', dtype=np.float32, shape=(len(inst), K))
        ranks = np.empty(len(inst)); top = np.empty((len(inst), 10), np.int32); full_rank = np.empty(len(inst))
        stats = dict(pos_diff=0, rows_order_diff=0, rows_set_diff=0, set_items_diff=0, f16_score_mismatch=0,
                     f16_score_mismatch_gt1ulp=0, max_gap_at_diff=0.0, max_abs_saved_item_score_err=0.0,
                     full_target_rank_mismatch=0, natural_membership_mismatch=0, order_diff_not_tie=0)
        started = time.perf_counter()
        for s in range(0, len(inst), batch_size):
            e = min(s + batch_size, len(inst))
            b = make_batch(data, inst.iloc[s:e], data.max_hist_len)
            with torch.no_grad():
                sc = apply_filters(scorer.score_all(b), b, first)
                vals, ids = torch.topk(sc, K, dim=1)
                tg = sc[torch.arange(e - s), torch.as_tensor(b.targets.astype(np.int64))]
                fr = (sc >= tg[:, None]).sum(1).numpy()
                sv = sc.gather(1, torch.as_tensor(saved_c[s:e].astype(np.int64))).numpy()
            c, v = ids.numpy().astype(np.int32), vals.numpy()
            if not np.isfinite(v).all():
                raise ValueError('fewer than 200 eligible items')
            check_candidates(c, b, first_np)
            cand[s:e], score[s:e] = c, v
            ranks[s:e], top[s:e] = pool_ranks(v, c, b.targets)
            full_rank[s:e] = fr
            sc_saved, ss = saved_c[s:e], saved_s[s:e]
            diff = c != sc_saved
            stats['pos_diff'] += int(diff.sum()); stats['rows_order_diff'] += int(diff.any(1).sum())
            setdiff = np.array([len(np.setdiff1d(a, bb, assume_unique=True)) for a, bb in zip(c[diff.any(1)], sc_saved[diff.any(1)])])
            stats['rows_set_diff'] += int((setdiff > 0).sum()); stats['set_items_diff'] += int(setdiff.sum())
            f16 = v.astype(np.float16)
            stats['f16_score_mismatch'] += int((f16 != ss).sum())
            ulp = np.abs(np.spacing(ss.astype(np.float16))).astype(np.float32)
            stats['f16_score_mismatch_gt1ulp'] += int((np.abs(f16.astype(np.float32) - ss.astype(np.float32)) > ulp).sum())
            if diff.any():
                gap = np.abs(v - sv)[diff]   # regenerated score at a position vs regenerated score of the saved item there
                stats['max_gap_at_diff'] = max(stats['max_gap_at_diff'], float(gap.max()))
                stats['order_diff_not_tie'] += int((gap > 1e-5).sum())
            stats['max_abs_saved_item_score_err'] = max(stats['max_abs_saved_item_score_err'],
                                                        float(np.abs(sv - ss.astype(np.float32)).max()))
            stats['full_target_rank_mismatch'] += int((fr != saved_trank[s:e]).sum())
            stats['natural_membership_mismatch'] += int(((c == b.targets[:, None]).any(1) != (sc_saved == b.targets[:, None]).any(1)).sum())
            if (s // batch_size) % 100 == 0:
                print(json.dumps({'regen': split, 'done': e, 'total': len(inst), 's': round(time.perf_counter() - started)}), flush=True)
        cand.flush(); score.flush()
        for key, value in {'inst': inst.inst.to_numpy(), 'user': inst.user.to_numpy(), 'target': inst.item.to_numpy(),
                           'retrieval_rank': ranks, 'top10': top}.items():
            np.save(out / f'{key}.npy', value)
        np.save(out / 'history.npy', np.asarray(rec['history']))
        aux = BULK / 'aux/P2_two_tower' / split
        aux.mkdir(parents=True, exist_ok=False)
        saved_rank, _ = pool_ranks(saved_s.astype(np.float32), saved_c, inst.item.to_numpy())
        np.save(aux / 'full_catalogue_target_rank.npy', full_rank)
        np.save(aux / 'saved_f16_pool_rank.npy', saved_rank)
        stats.update(rows=len(inst), seconds=round(time.perf_counter() - started),
                     regen_natural_hits=int(np.isfinite(ranks).sum()), saved_natural_hits=int(np.isfinite(saved_rank).sum()),
                     pool_rank_equals_saved_f16_pessimistic=int((ranks == saved_rank).sum()),
                     regen_pool_rank_equals_full_rank_inside200=int((ranks[np.isfinite(ranks)] == full_rank[np.isfinite(ranks)]).sum()))
        summary[split] = stats
        print(json.dumps({split: stats}), flush=True)
        del saved_c, saved_s
    write_json(COMPACT / 'regen_two_tower.json', summary)


def fuse(chunk: int = 4096) -> None:
    """Task 4: P3 from P1 (recency) and P2 (regenerated two-tower)."""
    for split in SPLITS:
        rec, tt = load_archive(RECENCY / split), load_archive(BULK / 'candidates/P2_two_tower' / split)
        if not np.array_equal(rec['inst'], tt['inst']):
            raise ValueError('P1/P2 instance order differs')
        n = len(rec['inst'])
        out = BULK / 'candidates/P3_fused_rrf' / split
        aux = BULK / 'aux/P3_fused_rrf' / split
        out.mkdir(parents=True, exist_ok=False); aux.mkdir(parents=True, exist_ok=False)
        mm = {k: open_memmap(p, mode='w+', dtype=d, shape=(n, K)) for k, p, d in
              [('cand', out / 'cand.npy', np.int32), ('score', out / 'score.npy', np.float32),
               ('rec_rank', aux / 'source_rank_recency.npy', np.int16), ('tt_rank', aux / 'source_rank_two_tower.npy', np.int16)]}
        ranks = np.empty(n); top = np.empty((n, 10), np.int32); inter = np.empty(n)
        target = np.asarray(rec['target'])
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            f = interleave_fuse(np.asarray(rec['cand'][s:e]), np.asarray(tt['cand'][s:e]))
            for k in mm:
                mm[k][s:e] = f[k]
            ranks[s:e], top[s:e] = pool_ranks(f['score'], f['cand'], target[s:e])
            hit = f['cand'] == target[s:e, None]
            inter[s:e] = np.where(hit.any(1), hit.argmax(1) + 1, np.inf)
        for v in mm.values():
            v.flush()
        for key in ('inst', 'user', 'target', 'history'):
            np.save(out / f'{key}.npy', np.asarray(rec[key]))
        np.save(out / 'retrieval_rank.npy', ranks); np.save(out / 'top10.npy', top)
        np.save(aux / 'interleave_rank.npy', inter)
        print(json.dumps({'fused': split, 'rows': n, 'natural_hits': int(np.isfinite(ranks).sum())}), flush=True)


def filters(n_rows: int = 10000, seed: int = 0) -> None:
    """Task 3: identical §4.6 filters on a seeded sample, for every pool and the saved two-tower list."""
    data = load_data()
    first = first_ts_tensor(data).numpy()
    result = {'label': LABEL, 'rows_per_split': n_rows, 'seed': seed}
    rng = np.random.default_rng(seed)
    for split in SPLITS:
        inst = data.split_instances(split)
        rows = np.sort(rng.choice(len(inst), n_rows, replace=False))
        sub = inst.iloc[rows]
        b = make_batch(data, sub, data.max_hist_len)
        z = saved_two_tower(split)
        sources = {'P1_recency': load_archive(RECENCY / split)['cand'][rows],
                   'P2_two_tower_regen': load_archive(BULK / 'candidates/P2_two_tower' / split)['cand'][rows],
                   'P3_fused_rrf': load_archive(BULK / 'candidates/P3_fused_rrf' / split)['cand'][rows],
                   'two_tower_saved_natural200': z['cand_full'][rows, :K]}
        del z
        r = {name: filter_violations(np.asarray(c), b.ts, b.mask_rows, b.mask_items, first) for name, c in sources.items()}
        # archive histories equal the harness histories (last 50, left padded)
        for name, d in [('P1_recency', RECENCY), ('P2_two_tower_regen', BULK / 'candidates/P2_two_tower'),
                        ('P3_fused_rrf', BULK / 'candidates/P3_fused_rrf')]:
            r[name]['history_matches_harness'] = bool(np.array_equal(load_archive(d / split)['history'][rows], b.hist.numpy()))
        r['full_history_events_checked'] = int(len(b.mask_items))
        result[split] = r
    result['passed'] = all(all(v == 0 for k, v in r.items() if k != 'history_matches_harness') and r.get('history_matches_harness', True)
                           for split in SPLITS for name, r in result[split].items() if isinstance(r, dict))
    write_json(COMPACT / 'filter_check.json', result)
    print(json.dumps(result, indent=1), flush=True)


def _metrics(rank: np.ndarray, users: np.ndarray, masks: dict) -> dict:
    out = {}
    for g, m in masks.items():
        row = {'events': int(m.sum()), 'users': int(len(np.unique(users[m])))}
        for name, v in [('recall200', np.isfinite(rank).astype(float)), ('recall10', M.recall_at(rank, 10)),
                        ('ndcg10', M.ndcg_at(rank, 10))]:
            mm = M.macro_micro(v[m], users[m])
            row[f'{name}_macro'], row[f'{name}_micro'] = mm['macro'], mm['micro']
        out[g] = row
    return out


def report() -> None:
    """Tasks 5-6: pool metrics, B natural positives, retriever oracle and paired user bootstraps."""
    from cascade.pipelines.graph_latent_study import split_cutoff
    data = load_data()
    t1 = data.split['t1']
    first = data.items.set_index('item').first_ts
    paths = {'P1_recency': RECENCY, 'P2_two_tower': BULK / 'candidates/P2_two_tower', 'P3_fused_rrf': BULK / 'candidates/P3_fused_rrf'}
    result = {'label': LABEL, 'pools': {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()}, 'B_natural_positives': {}}
    # B natural positives
    b_inst = data.split_instances('ranker_train')
    _, prefix = split_cutoff(b_inst)
    for name, d in paths.items():
        found = np.isfinite(np.asarray(load_archive(d / 'ranker_train')['retrieval_rank']))
        result['B_natural_positives'][name] = {'all_B': int(found.sum()), 'of_rows': int(len(found)),
                                               'inner_prefix': int((found & prefix).sum()), 'inner_tune': int((found & ~prefix).sum())}
    rec_b = np.isfinite(np.asarray(load_archive(RECENCY / 'ranker_train')['retrieval_rank']))
    tt_b = np.isfinite(np.asarray(load_archive(paths['P2_two_tower'] / 'ranker_train')['retrieval_rank']))
    result['B_natural_positives']['union_either_top200'] = int((rec_b | tt_b).sum())
    result['B_natural_positives']['both_top200'] = int((rec_b & tt_b).sum())
    # Val
    inst = data.split_instances('val')
    users = inst.user.to_numpy()
    new = first.loc[inst.item].to_numpy() >= t1
    masks = {'overall': np.ones(len(inst), bool), 'new_in_val': new, 'warm': ~new}
    arch = {k: load_archive(v / 'val') for k, v in paths.items()}
    for a in arch.values():
        if not np.array_equal(a['inst'], inst.inst.to_numpy()) or not np.array_equal(a['user'], users):
            raise ValueError('val alignment differs')
    rank = {k: np.asarray(a['retrieval_rank'], float) for k, a in arch.items()}
    rank['P3_interleave_order'] = np.load(BULK / 'aux/P3_fused_rrf/val/interleave_rank.npy')
    rank['P2_saved_float16_pessimistic'] = np.load(BULK / 'aux/P2_two_tower/val/saved_f16_pool_rank.npy')
    # oracles (target-conditioned, upper bound only)
    n1, n2 = M.ndcg_at(rank['P1_recency'], 10), M.ndcg_at(rank['P2_two_tower'], 10)
    choose_tt = n2 > n1     # ties -> recency
    oracle_rank = np.where(choose_tt, rank['P2_two_tower'], rank['P1_recency'])
    rank['oracle_retriever_ndcg10'] = oracle_rank
    cand_oracle = np.isfinite(rank['P1_recency']) | np.isfinite(rank['P2_two_tower'])
    table = {k: _metrics(r, users, masks) for k, r in rank.items()}
    for g, m in masks.items():
        mm = M.macro_micro(cand_oracle[m].astype(float), users[m])
        table['oracle_candidate_either_top200'] = table.get('oracle_candidate_either_top200', {})
        table['oracle_candidate_either_top200'][g] = {'events': int(m.sum()), 'recall200_macro': mm['macro'], 'recall200_micro': mm['micro']}
        table['oracle_retriever_ndcg10'][g]['share_events_two_tower_chosen'] = float(choose_tt[m].mean())
        table['oracle_retriever_ndcg10'][g]['note'] = 'recall200 column of this row is not an oracle quantity; see oracle_candidate_either_top200'
    result['val_metrics'] = table
    # overlap diagnostics
    p3_rec = np.load(BULK / 'aux/P3_fused_rrf/val/source_rank_recency.npy', mmap_mode='r')
    p3_tt = np.load(BULK / 'aux/P3_fused_rrf/val/source_rank_two_tower.npy', mmap_mode='r')
    both = np.zeros(len(inst)); only_rec = np.zeros(len(inst)); only_tt = np.zeros(len(inst))
    for s in range(0, len(inst), 8192):
        a, bb = np.asarray(p3_rec[s:s + 8192]) > 0, np.asarray(p3_tt[s:s + 8192]) > 0
        both[s:s + 8192], only_rec[s:s + 8192], only_tt[s:s + 8192] = (a & bb).sum(1), (a & ~bb).sum(1), (~a & bb).sum(1)
    result['P3_val_composition_mean_per_list'] = {'both_sources': float(both.mean()), 'recency_only': float(only_rec.mean()),
                                                  'two_tower_only': float(only_tt.mean())}
    ov = np.zeros(len(inst))
    for s in range(0, len(inst), 8192):
        a, bb = np.asarray(arch['P1_recency']['cand'][s:s + 8192]), np.asarray(arch['P2_two_tower']['cand'][s:s + 8192])
        ov[s:s + 8192] = [len(np.intersect1d(x, y, assume_unique=True)) for x, y in zip(a, bb)]
    result['P1_P2_val_top200_overlap'] = {'mean': float(ov.mean()), 'median': float(np.median(ov))}
    # bootstraps
    vals = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall200': lambda r: np.isfinite(r).astype(float),
            'recall10': lambda r: M.recall_at(r, 10)}
    contrasts = [('P2_two_tower', 'P1_recency', ('ndcg10', 'recall200', 'recall10')),
                 ('P3_fused_rrf', 'P1_recency', ('ndcg10', 'recall200', 'recall10')),
                 ('P3_interleave_order', 'P1_recency', ('ndcg10', 'recall10')),
                 ('P3_fused_rrf', 'P2_two_tower', ('ndcg10', 'recall200')),
                 ('oracle_retriever_ndcg10', 'P1_recency', ('ndcg10', 'recall10'))]
    boots = []
    for b_name, a_name, ms in contrasts:
        for metric in ms:
            for g, m in masks.items():
                ci = paired_bootstrap(users[m], vals[metric](rank[a_name])[m], vals[metric](rank[b_name])[m], n_resamples=N_BOOT, seed=BOOT_SEED)
                boots.append({'contrast': f'{b_name} - {a_name}', 'metric': metric, 'group': g, **ci})
    for g, m in masks.items():
        ci = paired_bootstrap(users[m], np.isfinite(rank['P1_recency'])[m].astype(float), cand_oracle[m].astype(float), n_resamples=N_BOOT, seed=BOOT_SEED)
        boots.append({'contrast': 'oracle_candidate_either_top200 - P1_recency', 'metric': 'recall200', 'group': g, **ci})
    result['paired_user_bootstrap'] = {'resamples': N_BOOT, 'seed': BOOT_SEED, 'unit': 'user (macro over users in group)', 'rows': boots}
    write_json(COMPACT / 'pool_report.json', result)
    np.savez_compressed(COMPACT / 'val_event_ranks.npz', inst=inst.inst.to_numpy(), user=users, new_in_val=new,
                        **{k: v for k, v in rank.items()})
    lines = ['| pool | group | events | R@200 macro | R@200 micro | R@10 macro | nDCG@10 macro | nDCG@10 micro |', '|---|---|---|---|---|---|---|---|']
    for k, rows in table.items():
        for g, r in rows.items():
            lines.append(f"| {k} | {g} | {r['events']} | {r.get('recall200_macro', float('nan')):.6f} | {r.get('recall200_micro', float('nan')):.6f} | "
                         f"{r.get('recall10_macro', float('nan')):.6f} | {r.get('ndcg10_macro', float('nan')):.6f} | {r.get('ndcg10_micro', float('nan')):.6f} |")
    lines += ['', '| contrast | metric | group | diff (macro) | 95% CI |', '|---|---|---|---|---|']
    lines += [f"| {b['contrast']} | {b['metric']} | {b['group']} | {b['diff']:+.6f} | [{b['ci_low']:+.6f}, {b['ci_high']:+.6f}] |" for b in boots]
    (COMPACT / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(json.dumps({k: result[k] for k in ('B_natural_positives', 'P3_val_composition_mean_per_list', 'P1_P2_val_top200_overlap')}, indent=1))


def provenance() -> None:
    """config.yaml and git_hash.txt for the compact folder (never overwritten)."""
    for name in ('config.yaml', 'git_hash.txt'):
        if (COMPACT / name).exists():
            raise FileExistsError(name)
    (COMPACT / 'git_hash.txt').write_text(git_hash() + '\n')
    (COMPACT / 'config.yaml').write_text(
        '# Step A0 candidate pools (Val only, exploratory). Fixed rules; nothing selected.\n'
        f'k: {K}\nrrf_k: {RRF_K}\nsplits: [ranker_train, val]\n'
        'P1: results/local_experiments/20261103-full-promising-ranking/candidates (frozen MiniLM + 90-day recency, natural top 200)\n'
        'P2: results/retriever/two_tower_text/seed0 regenerated float32, exact search, §4.6 filters, natural top 200\n'
        'P3: round-robin interleave recency r1, two-tower r1, recency r2, ... skip duplicates until 200 unique; '
        'score = sum over sources containing the item (within their 200) of 1/(60 + rank)\n'
        'retrieval_rank: pessimistic target rank by stored score within the pool (inf if absent)\n'
        'oracle: per event better of P1/P2 by nDCG@10 (ties -> recency); candidate oracle: target in either top 200\n'
        f'bootstrap: {{resamples: {N_BOOT}, seed: {BOOT_SEED}, unit: user}}\n'
        'data: 20261102-full-frozen-text checked inputs (data/processed)\ntest_loaded: false\n'
        f'source_sha256: {digest(Path(__file__))}\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('verify', 'regen', 'fuse', 'filters', 'report', 'provenance'))
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    {'verify': verify, 'regen': lambda: regen(a.device), 'fuse': fuse, 'filters': filters,
     'report': report, 'provenance': provenance}[a.step]()


if __name__ == '__main__':
    main()
