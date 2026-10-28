"""End-to-end single-query CPU latency of the cascade (CLAUDE.md §5 "Latency").

One query = one Val instance passed through the whole pipeline, one at a time, single-threaded:

  user_history   last-L history ids → tensor                 ┐ user tower
  user_forward   user-tower forward pass (u, L2-normalised)  ┘
  faiss_search   FAISS search for K₁ + buffer results        ┐ retrieval (FAISS search incl. §4.6 filtering)
  filter         §4.6 filtering, keep the first K₁           ┘
  rank_prepare   ranker features for the K₁ candidates + tensors  ┐
  rank_score     ranker forward pass                              │ ranker
  rank_order     sort by ranker score, keep the top 10            ┘

Every phase boundary is one ``perf_counter_ns`` stamp, so ``total`` (last stamp − first stamp) is the directly
timed end-to-end latency and equals the sum of the phases. Disk I/O, checkpoint loading, index building and
feature-table creation happen before timing. Per-query outputs (candidates, top 10) are copied out after the
last stamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable, Sequence

import torch  # noqa: I001 — must be imported before faiss on macOS (libomp clash)
import faiss
import numpy as np
import pandas as pd

from cascade.index.faiss_index import search

PHASES = ("user_history", "user_forward", "faiss_search", "filter", "rank_prepare", "rank_score", "rank_order")
COMPONENTS = {
    "user_tower": ("user_history", "user_forward"),
    "retrieval": ("faiss_search", "filter"),
    "ranker": ("rank_prepare", "rank_score", "rank_order"),
    "total": PHASES,
}
FINAL_LIST = 10


def sample_instances(n: int, queries: int, warmup: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed random (warm-up rows, measured rows) out of ``n`` instances, disjoint, without replacement."""
    if queries + warmup > n:
        raise ValueError(f"need {queries + warmup} instances, only {n} available")
    rows = np.random.default_rng(seed).choice(n, size=queries + warmup, replace=False)
    return rows[:warmup], rows[warmup:]


def timed_chain(steps: Sequence[Callable[[Any], Any]], x: Any) -> tuple[Any, np.ndarray]:
    """Run ``x = step(x)`` for each step; return the result and len(steps) + 1 ns stamps (one per boundary)."""
    stamps = np.empty(len(steps) + 1, dtype=np.int64)
    stamps[0] = perf_counter_ns()
    for i, step in enumerate(steps):
        x = step(x)
        stamps[i + 1] = perf_counter_ns()
    return x, stamps


def stamps_to_ms(stamps: np.ndarray) -> np.ndarray:
    """[n, P+1] ns stamps → [n, P] phase durations in ms."""
    return np.diff(np.asarray(stamps, dtype=np.int64), axis=-1) / 1e6


def percentile_summary(values_ms: np.ndarray) -> dict[str, float]:
    """Median, p95 (numpy linear interpolation), mean, min and max of per-query timings."""
    v = np.asarray(values_ms, dtype=np.float64)
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "mean": float(v.mean()),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def summarise_times(times_ms: np.ndarray, phases: Sequence[str] = PHASES, components: dict[str, Sequence[str]] = COMPONENTS) -> dict:
    """Per-phase and per-component summaries; a component is the per-query sum of its phases (then percentiles)."""
    col = {p: i for i, p in enumerate(phases)}
    return {"phases": {p: percentile_summary(times_ms[:, i]) for p, i in col.items()},
            "components": {c: percentile_summary(times_ms[:, [col[p] for p in ps]].sum(1)) for c, ps in components.items()}}


def filter_candidates(ids: np.ndarray, scores: np.ndarray, history: np.ndarray, t: int, first_ts: np.ndarray, k1: int) -> tuple[np.ndarray, np.ndarray]:
    """§4.6 filtering of one query's raw FAISS results (same rule as ``faiss_index.filtered_search``).

    Drops padding/missing ids, items first seen after ``t`` and items in the user's full history; keeps the first
    ``k1`` survivors in search order. Returns (ids int32, scores float32), length ≤ k1 (no padding).
    """
    keep = (ids > 0) & (first_ts[ids] <= t) & ~np.isin(ids, history)
    sel = np.flatnonzero(keep)[:k1]
    return ids[sel].astype(np.int32), scores[sel].astype(np.float32)


@dataclass
class CascadeQuery:
    """In-memory state for timing one (index, K₁, ranker) configuration; each step maps a state dict to itself."""

    data: Any                 # ProcessedData
    retriever: torch.nn.Module
    index: faiss.Index
    ranker: torch.nn.Module
    builder: Any              # RankerFeatureBuilder
    inst: pd.DataFrame        # Val instances (inst, user, item, ts, hist_len)
    first_ts: np.ndarray      # [N+1] first interaction time per item id (0 = padding = never released)
    k1: int
    n_requested: int          # K₁ + buffer
    _cols: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._cols = {c: self.inst[c].to_numpy() for c in ("inst", "user", "item", "ts", "hist_len")}

    @property
    def steps(self) -> tuple[Callable[[dict], dict], ...]:
        return (self.user_history, self.user_forward, self.faiss_search, self.filter, self.rank_prepare, self.rank_score, self.rank_order)

    def run(self, row: int) -> tuple[dict, np.ndarray]:
        """Run one query; returns (final state, ns stamps [len(PHASES) + 1])."""
        return timed_chain(self.steps, {"row": row})

    def user_history(self, s: dict) -> dict:
        r = s["row"]
        hist = self.data.histories(self._cols["user"][r : r + 1], self._cols["hist_len"][r : r + 1], self.data.max_hist_len)
        s["hist"] = torch.from_numpy(hist)
        return s

    def user_forward(self, s: dict) -> dict:
        s["u"] = self.retriever.user_vectors(s["hist"]).float().numpy()
        return s

    def faiss_search(self, s: dict) -> dict:
        s["raw_scores"], s["raw_ids"] = search(self.index, s["u"], self.n_requested)
        return s

    def filter(self, s: dict) -> dict:
        r = s["row"]
        start = self.data.user_ptr[self._cols["user"][r]]
        history = self.data.seq_item[start : start + self._cols["hist_len"][r]]  # full history before t (§4.6)
        s["cand"], s["cand_score"] = filter_candidates(s["raw_ids"][0], s["raw_scores"][0], history, self._cols["ts"][r], self.first_ts, self.k1)
        return s

    def rank_prepare(self, s: dict) -> dict:
        n = len(s["cand"])
        # Same layout as the stored candidate lists the rankers were trained/evaluated on (scores stored as float16).
        lists = {"inst": self._cols["inst"][s["row"] : s["row"] + 1], "cand": s["cand"][None], "score": s["cand_score"].astype(np.float16)[None]}
        feats = self.builder.build(lists, np.zeros(n, dtype=np.int64), np.arange(n, dtype=np.int64))
        s["batch"] = {k: torch.as_tensor(v) for k, v in feats.items()}
        return s

    def rank_score(self, s: dict) -> dict:
        s["rank_scores"] = self.ranker(s["batch"]).numpy()
        return s

    def rank_order(self, s: dict) -> dict:
        s["top"] = s["cand"][np.argsort(-s["rank_scores"], kind="stable")[:FINAL_LIST]]
        return s


def measure(query: CascadeQuery, warmup_rows: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Warm up, then time each row one at a time. Returns (times [n, len(PHASES)] ms, per-query outputs).

    Call inside ``torch.inference_mode()`` and single-thread limits; models must already be in eval mode.
    """
    for r in warmup_rows:
        query.run(int(r))
    stamps = np.empty((len(rows), len(PHASES) + 1), dtype=np.int64)
    cand = np.zeros((len(rows), query.k1), dtype=np.int32)
    top = np.zeros((len(rows), FINAL_LIST), dtype=np.int32)
    for i, r in enumerate(rows):
        s, stamps[i] = query.run(int(r))
        if len(s["top"]) != min(FINAL_LIST, len(s["cand"])):
            raise RuntimeError("pipeline failed to return the final list")
        cand[i, : len(s["cand"])] = s["cand"]
        top[i, : len(s["top"])] = s["top"]
    return stamps_to_ms(stamps), {"cand": cand, "top": top}
