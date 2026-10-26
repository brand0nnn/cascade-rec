"""Metric definitions (CLAUDE.md §5). Single relevant item per instance; ``rank`` is 1-based, inf if absent."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recall_at(ranks: np.ndarray, cutoff: int) -> np.ndarray:
    """Recall@K per instance: 1 if rank <= K else 0."""
    return (ranks <= cutoff).astype(np.float64)


def ndcg_at(ranks: np.ndarray, cutoff: int) -> np.ndarray:
    """nDCG@K per instance: 1/log2(rank+1) if rank <= K else 0."""
    out = np.zeros(len(ranks), dtype=np.float64)
    hit = ranks <= cutoff
    out[hit] = 1.0 / np.log2(ranks[hit] + 1.0)
    return out


def macro_micro(values: np.ndarray, users: np.ndarray) -> dict[str, float]:
    """Per-user macro average (mean within user, then across users) and plain per-instance micro average."""
    if len(values) == 0:
        return {"macro": float("nan"), "micro": float("nan")}
    per_user = pd.Series(values).groupby(users).mean()
    return {"macro": float(per_user.mean()), "micro": float(np.mean(values))}


def instance_auc(scores: np.ndarray, target_idx: int) -> float:
    """AUC of the target vs every other candidate in one list (ties count 0.5)."""
    others = np.delete(scores, target_idx)
    if len(others) == 0:
        return float("nan")
    t = scores[target_idx]
    return float(((others < t).sum() + 0.5 * (others == t).sum()) / len(others))


def gauc(instance_aucs: np.ndarray, users: np.ndarray) -> float:
    """GAUC: instance AUCs averaged per user, then over users (NaNs dropped)."""
    m = ~np.isnan(instance_aucs)
    return float(pd.Series(instance_aucs[m]).groupby(users[m]).mean().mean())


def coverage(top_lists: np.ndarray, n_items: int) -> float:
    """#distinct items in any top-K list / N (padding id 0 ignored)."""
    u = np.unique(top_lists)
    return float(len(u[u > 0]) / n_items)


def popularity_concentration(top_lists: np.ndarray, is_top_pop: np.ndarray) -> float:
    """Share of all top-K slots filled by top-1% items. ``is_top_pop`` is indexed by item id (index 0 = padding)."""
    return float(is_top_pop[top_lists].mean())
