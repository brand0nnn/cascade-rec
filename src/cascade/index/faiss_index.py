"""FAISS indexes over retriever item vectors (CLAUDE.md §8).

Item vectors are L2-normalised, so inner product = cosine. Row r of the index holds item id r+1 (padding id 0 is
not indexed); ``search`` returns item ids.

Approximate search returns the top hits *before* §4.6 filtering, so ``filtered_search`` asks for K₁ + buffer
results (buffer = max_hist_len + 200 by default) and keeps the first K₁ that survive filtering.
"""
from __future__ import annotations

import sys

import faiss
import numpy as np

# On macOS, torch and faiss each bundle their own libomp; using both multi-threaded in one process segfaults.
# Single-threaded faiss avoids it (latency is measured single-threaded anyway, §5). Linux is unaffected.
if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)


def build_index(item_vectors: np.ndarray, kind: str, nlist: int = 1024, hnsw_m: int = 32, ef_construction: int = 200, seed: int = 0) -> faiss.Index:
    """Build ``kind`` in {exact, hnsw, ivf} over ``item_vectors`` [N+1, d] (row 0 = padding, skipped)."""
    x = np.ascontiguousarray(item_vectors[1:], dtype=np.float32)
    d = x.shape[1]
    if kind == "exact":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = seed
        index.train(x)
    else:
        raise ValueError(kind)
    index.add(x)
    return index


def set_search_params(index: faiss.Index, ef_search: int | None = None, nprobe: int | None = None) -> None:
    if ef_search is not None:
        index.hnsw.efSearch = ef_search
    if nprobe is not None:
        index.nprobe = nprobe


def search(index: faiss.Index, queries: np.ndarray, n_results: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``n_results`` (scores, item ids) per query. Missing results have id 0 and score -inf.

    For HNSW, efSearch must be >= n_results (CLAUDE.md §8).
    """
    if isinstance(index, faiss.IndexHNSWFlat):
        assert index.hnsw.efSearch >= n_results, f"efSearch {index.hnsw.efSearch} < {n_results} results requested"
    scores, rows = index.search(np.ascontiguousarray(queries, dtype=np.float32), n_results)
    ids = np.where(rows >= 0, rows + 1, 0)
    return np.where(rows >= 0, scores, -np.inf), ids


def filtered_search(index: faiss.Index, queries: np.ndarray, k1: int, blocked: list[set[int]], ts: np.ndarray, first_ts: np.ndarray, buffer: int = 250) -> tuple[np.ndarray, np.ndarray, int]:
    """§4.6-filtered top-K₁: search K₁ + buffer, drop blocked (history) items and items with first_ts > t.

    Returns (scores [B, K₁], ids [B, K₁], number of queries left with fewer than K₁ candidates).
    ``first_ts`` is indexed by item id (index 0 = padding).
    """
    scores, ids = search(index, queries, k1 + buffer)
    out_s = np.full((len(queries), k1), -np.inf, dtype=np.float32)
    out_i = np.zeros((len(queries), k1), dtype=np.int64)
    short = 0
    for r in range(len(queries)):
        keep = (ids[r] > 0) & (first_ts[ids[r]] <= ts[r]) & ~np.isin(ids[r], list(blocked[r]))
        sel = np.flatnonzero(keep)[:k1]
        out_s[r, : len(sel)], out_i[r, : len(sel)] = scores[r, sel], ids[r, sel]
        short += len(sel) < k1
    return out_s, out_i, short


def recall_vs_exact(approx_ids: np.ndarray, exact_ids: np.ndarray) -> float:
    """Fraction of the exact top-K₁ that the approximate index also returned (per query, then averaged)."""
    hits = [len(set(a[a > 0]) & set(e[e > 0])) / max(1, (e > 0).sum()) for a, e in zip(approx_ids, exact_ids)]
    return float(np.mean(hits))
