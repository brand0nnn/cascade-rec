"""Ranker validation on the shared single-target metric definitions."""
from __future__ import annotations
from typing import Mapping
import numpy as np
from cascade.eval import metrics as M


def candidate_ranks(scores: np.ndarray, lists: Mapping[str, np.ndarray]) -> np.ndarray:
    """Pessimistic target ranks; forced targets remain retrieval misses."""
    candidates = lists['cand']
    if scores.shape != candidates.shape:
        raise ValueError('scores and candidates must have the same shape')
    n = len(candidates)
    pos = np.asarray(lists['target_pos'], dtype=np.int64)
    forced = np.asarray(lists.get('forced', np.zeros(n, dtype=bool)), dtype=bool)
    hits = np.flatnonzero((pos >= 0) & ~forced)
    ranks = np.full(n, np.inf, dtype=np.float64)
    target_scores = scores[hits, pos[hits]]
    if not np.isfinite(scores[hits][candidates[hits] > 0]).all():
        raise ValueError('retrieved candidate scores must be finite')
    ranks[hits] = np.sum((candidates[hits] > 0) & (scores[hits] >= target_scores[:, None]), axis=1)
    return ranks


def ranker_comparison(ranks: np.ndarray, lists: Mapping[str, np.ndarray]) -> dict:
    """Overall and hit-only metrics plus exhaustive top-K rescue/demotion counts."""
    cutoff = 10
    pos = np.asarray(lists['target_pos'], dtype=np.int64)
    forced = np.asarray(lists.get('forced', np.zeros(len(pos), dtype=bool)), dtype=bool)
    found = (pos >= 0) & ~forced
    original = np.where(found, pos + 1, np.inf)
    users = np.asarray(lists['user'])
    before = original <= cutoff
    after = ranks <= cutoff
    masks = {
        'retrieval_miss': ~found,
        'unrescued': found & ~before & ~after,
        'demoted': found & before & ~after,
        'retained': found & before & after,
        'rescued': found & ~before & after,
    }
    return {
        'n_lists': int(len(pos)), 'n_retrieved': int(found.sum()),
        'retrieval_order': {
            'ndcg@10': M.macro_micro(M.ndcg_at(original, cutoff), users),
            'recall@10': M.macro_micro(M.recall_at(original, cutoff), users),
            'conditional_ndcg@10': M.macro_micro(M.ndcg_at(original[found], cutoff), users[found]),
        },
        'ranker': {
            'ndcg@10': M.macro_micro(M.ndcg_at(ranks, cutoff), users),
            'recall@10': M.macro_micro(M.recall_at(ranks, cutoff), users),
            'conditional_ndcg@10': M.macro_micro(M.ndcg_at(ranks[found], cutoff), users[found]),
        },
        'outcomes': {name: {'count': int(mask.sum()), 'fraction_of_all_lists': float(mask.mean())} for name, mask in masks.items()},
    }
