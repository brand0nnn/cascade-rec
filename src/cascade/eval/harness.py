"""The single evaluation harness used by every model (CLAUDE.md §5).

Full-ranking over the filtered catalogue (never sampled negatives). For every instance the harness removes
(1) items in the user's full history before t and (2) items whose first interaction is after t (§4.6),
then ranks the target. Ties are broken pessimistically: rank = 1 + #other unmasked items with score >= target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from tqdm import tqdm

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.segments import HIST_BUCKETS, ITEM_SEGMENTS, hist_bucket, new_in_period
from cascade.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CUTOFFS = (10, 20, 50, 100, 200, 500)


@dataclass
class InstanceBatch:
    inst: np.ndarray          # [B] instance ids
    users: np.ndarray         # [B] user ids
    targets: np.ndarray       # [B] target item ids
    ts: np.ndarray            # [B] timestamps (ms)
    hist_len: np.ndarray      # [B] full history length
    hist: torch.LongTensor    # [B, L] last L history items, left-padded with 0
    mask_rows: np.ndarray     # full-history (row, item) pairs used for filtering
    mask_items: np.ndarray


@runtime_checkable
class Recommender(Protocol):
    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        """Scores for every item id, shape [B, N+1] (column 0 = padding, ignored)."""
        ...


def make_batch(data: ProcessedData, inst_df, max_hist_len: int) -> InstanceBatch:
    users = inst_df["user"].values
    hl = inst_df["hist_len"].values
    starts = data.user_ptr[users]
    rows = np.repeat(np.arange(len(users)), hl)
    offs = np.arange(hl.sum()) - np.repeat(np.cumsum(hl) - hl, hl)
    items = data.seq_item[np.repeat(starts, hl) + offs]
    return InstanceBatch(
        inst=inst_df["inst"].values,
        users=users,
        targets=inst_df["item"].values,
        ts=inst_df["ts"].values,
        hist_len=hl,
        hist=torch.from_numpy(data.histories(users, hl, max_hist_len)),
        mask_rows=rows,
        mask_items=items,
    )


def filtered_ranks(scores: torch.Tensor, batch: InstanceBatch, first_ts: torch.Tensor, top_n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Apply §4.6 filtering in place and return (1-based target ranks, top-n item lists)."""
    scores = torch.nan_to_num(scores.float(), nan=-torch.inf)
    scores[:, 0] = -torch.inf
    ts = torch.from_numpy(batch.ts)
    scores.masked_fill_(first_ts.unsqueeze(0) > ts.unsqueeze(1), -torch.inf)
    scores[torch.from_numpy(batch.mask_rows), torch.from_numpy(batch.mask_items.astype(np.int64))] = -torch.inf
    ar = torch.arange(len(batch.targets))
    tgt = scores[ar, torch.from_numpy(batch.targets.astype(np.int64))]
    ranks = (scores >= tgt.unsqueeze(1)).sum(1)  # includes the target itself
    top = torch.topk(scores, top_n, dim=1).indices
    return ranks.numpy().astype(np.float64), top.numpy()


def evaluate(
    model: Recommender,
    data: ProcessedData,
    split: str,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    batch_size: int = 512,
    max_instances: int | None = None,
    subset_seed: int = 0,
    save_dir: Path | None = None,
    show_progress: bool = True,
) -> dict:
    """Evaluate ``model`` on the ``split`` instances ('val' or 'test'); returns a metrics dict.

    ``max_instances`` evaluates a fixed random subset (for early stopping / quick checks only).
    Models may expose ``scorable(batch) -> bool array``; unscorable instances (e.g. BPR users unseen in
    training) are skipped, and metrics are also reported with them counted as misses.
    """
    assert split in ("val", "test")
    inst = data.split_instances(split)
    if max_instances is not None and max_instances < len(inst):
        rng = np.random.default_rng(subset_seed)
        inst = inst.iloc[np.sort(rng.choice(len(inst), max_instances, replace=False))].reset_index(drop=True)
    first_ts = torch.from_numpy(np.concatenate([[np.iinfo(np.int64).max], data.items["first_ts"].values]))
    n = len(inst)
    ranks = np.full(n, np.inf)
    scorable = np.ones(n, dtype=bool)
    top10 = np.zeros((n, 10), dtype=np.int64)
    it = range(0, n, batch_size)
    for s in tqdm(it, desc=f"eval {split}", disable=not show_progress, mininterval=10):
        b = make_batch(data, inst.iloc[s : s + batch_size], data.max_hist_len)
        with torch.no_grad():
            scores = model.score_all(b)
        r, t = filtered_ranks(scores, b, first_ts)
        ok = model.scorable(b) if hasattr(model, "scorable") else np.ones(len(r), dtype=bool)
        r[~ok] = np.inf
        ranks[s : s + len(r)], top10[s : s + len(r)], scorable[s : s + len(r)] = r, t, ok

    seg_map = np.concatenate([["pad"], data.items["segment"].values])
    top_pop = np.concatenate([[False], data.items["top1pct"].values])
    target_seg = seg_map[inst["item"].values]
    hb = hist_bucket(inst["hist_len"].values)
    users = inst["user"].values

    def block(m: np.ndarray) -> dict:
        out = {"n_instances": int(m.sum()), "n_users": int(len(np.unique(users[m])))}
        for c in cutoffs:
            out[f"recall@{c}"] = M.macro_micro(M.recall_at(ranks[m], c), users[m])
            out[f"ndcg@{c}"] = M.macro_micro(M.ndcg_at(ranks[m], c), users[m])
        return out

    res = {"split": split, **block(scorable), "n_unscorable": int((~scorable).sum())}
    if (~scorable).any():
        res["with_unscorable_as_miss"] = block(np.ones(n, dtype=bool))
    res["coverage@10"] = M.coverage(top10[scorable], data.n_items)
    res["pop_concentration@10"] = M.popularity_concentration(top10[scorable], top_pop)
    first_ts_np = first_ts.numpy()
    is_new = new_in_period(first_ts_np[inst["item"].values], split, data.split)
    res["segments"] = {
        **{f"item_{s}": block(scorable & (target_seg == s)) for s in ITEM_SEGMENTS},
        "item_new_in_period": block(scorable & is_new),
        "item_not_new": block(scorable & ~is_new),
        **{f"hist_{h}": block(scorable & (hb == h)) for h in HIST_BUCKETS},
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / f"{split}_ranks.npz", inst=inst["inst"].values, user=users, rank=ranks, scorable=scorable, top10=top10.astype(np.int32))
    return res
