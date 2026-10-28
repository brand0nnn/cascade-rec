"""Frozen score-response diagnostics on uniform natural validation events.

These describe logits and artificial one-input probes, not accuracy, calibrated
probabilities, or the causal effect of the training policy.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
from scipy.stats import rankdata
import torch

PROBE_GRID = np.arange(-3, 4, dtype=np.float32)


def uniform_cohort(lists: Mapping[str, np.ndarray], queries: int = 1000,
                   seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Select events without conditioning on target retrieval or model output."""
    n = len(lists["inst"])
    if queries < 1 or n < 1 or seed != 0:
        raise ValueError("uniform diagnostics require nonempty events/queries and fixed RNG seed 0")
    rows = np.sort(np.random.default_rng(0).choice(n, min(queries, n), replace=False))
    cohort = {"list_rows": rows, "inst": np.asarray(lists["inst"])[rows],
              "user": np.asarray(lists["user"])[rows],
              "candidates": np.asarray(lists["cand"])[rows],
              "raw_scores": np.asarray(lists["score"])[rows],
              "target_pos": np.asarray(lists["target_pos"])[rows]}
    identity = hashlib.sha256()
    for name, array in cohort.items():
        value = np.ascontiguousarray(array)
        identity.update(name.encode())
        identity.update(str(value.dtype).encode())
        identity.update(str(value.shape).encode())
        identity.update(value.tobytes())
    return rows, cohort, identity.hexdigest()


def _spearman(raw: np.ndarray, logits: np.ndarray) -> float:
    if len(raw) < 2 or np.all(raw == raw[0]) or np.all(logits == logits[0]):
        return np.nan
    a, b = rankdata(raw, method="average"), rankdata(logits, method="average")
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def _finite_mean(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else None


def _probe_difference(logits: np.ndarray, low: int, high: int) -> dict:
    values = logits[:, low] - logits[:, high]
    valid = values[np.isfinite(values)]
    return {"z_low": float(PROBE_GRID[low]), "z_high": float(PROBE_GRID[high]),
            "mean_logit_difference": float(valid.mean()) if len(valid) else None,
            "fraction_lower_score_favoured": float((valid > 0).mean()) if len(valid) else None,
            "fraction_tied": float((valid == 0).mean()) if len(valid) else None,
            "valid_queries": len(valid)}


def measure_score_response(model, builder, lists: Mapping[str, np.ndarray],
                           rows: np.ndarray, *, cohort_identity_sha256: str,
                           device: str = "cpu", batch_pairs: int = 8192) -> tuple[dict, dict]:
    """Score all valid candidates, including retrieval-miss events, then probe input 0.

    Descriptive top ten uses stable descending-logit order. Correlations average
    query-level values, not user-weighted accuracy. Constant ranks are undefined.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows) or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(lists["inst"])):
        raise ValueError("diagnostic rows must be nonempty unique valid event positions")
    if batch_pairs < 1 or np.asarray(lists["cand"]).shape[1] != 500:
        raise ValueError("diagnostics require natural top500 candidates and positive batch size")
    if not bool(getattr(model, "_retrieval_score_enabled", True)):
        raise ValueError("masked score response is imposed and must not be reported as measured")
    if np.asarray(lists.get("forced", np.zeros(len(lists["inst"]), bool)))[rows].any():
        raise ValueError("diagnostics require natural lists without force-inserted targets")
    candidates = np.asarray(lists["cand"])[rows]
    valid = candidates > 0
    raw = np.asarray(lists["score"])[rows].astype(np.float32)
    if not np.isfinite(raw[valid]).all():
        raise ValueError("nonfinite raw retrieval scores on valid diagnostic candidates")
    n, width = candidates.shape
    logits = np.full((n, width), np.nan, dtype=np.float32)
    model = model.to(device).eval()
    batch_lists = max(1, batch_pairs // width)
    with torch.inference_mode():
        for start in range(0, n, batch_lists):
            local = np.arange(start, min(start + batch_lists, n))
            local_rows, columns = np.nonzero(valid[local])
            if not len(local_rows):
                continue
            query_rows = local[local_rows]
            features = builder.build(lists, rows[query_rows], columns)
            batch = {key: torch.as_tensor(value, device=device) for key, value in features.items()}
            values = model(batch).float().cpu().numpy()
            if values.shape != (len(query_rows),) or not np.isfinite(values).all():
                raise ValueError("nonfinite or misaligned model logits on valid diagnostic candidates")
            logits[query_rows, columns] = values
        probe_valid = valid[:, 99]
        probe_rows = np.flatnonzero(probe_valid)
        probe_logits = np.full((n, len(PROBE_GRID)), np.nan, dtype=np.float32)
        probe_original = np.full(n, np.nan, dtype=np.float32)
        probe_features = {}
        if len(probe_rows):
            features = builder.build(lists, rows[probe_rows], np.full(len(probe_rows), 99, dtype=np.int64))
            probe_features = {f"probe_feature_{key}": np.asarray(value) for key, value in features.items()}
            batch = {key: torch.as_tensor(value, device=device) for key, value in features.items()}
            if batch["numeric"].ndim != 2 or batch["numeric"].shape[1] != 4:
                raise ValueError("probe expects the documented four numeric inputs")
            original = model(batch).float().cpu().numpy()
            if not np.isfinite(original).all():
                raise ValueError("nonfinite rank100 original probe logits")
            probe_original[probe_rows] = original
            for column, value in enumerate(PROBE_GRID):
                modified = dict(batch)
                modified["numeric"] = batch["numeric"].clone()
                modified["numeric"][:, 0] = float(value)
                values = model(modified).float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite artificial score-input probe logits")
                probe_logits[probe_rows, column] = values
    correlations = np.full((n, 3), np.nan, dtype=np.float64)
    top10_columns = np.full((n, 10), -1, dtype=np.int32)
    added_share = np.full(n, np.nan, dtype=np.float64)
    bands = ((0, 500), (0, 200), (200, 500))
    for row in range(n):
        for column, (lo, hi) in enumerate(bands):
            good = valid[row, lo:hi]
            correlations[row, column] = _spearman(raw[row, lo:hi][good], logits[row, lo:hi][good])
        available = np.flatnonzero(valid[row])
        if len(available):
            top = available[np.argsort(-logits[row, available], kind="stable")[:10]]
            top10_columns[row, :len(top)] = top
            added_share[row] = float((top >= 200).mean())
    spearman = {}
    for column, name in enumerate(("all500", "original200", "added300")):
        values = correlations[:, column]
        defined = values[np.isfinite(values)]
        spearman[name] = {"mean": _finite_mean(values),
                          "median": float(np.median(defined)) if len(defined) else None,
                          "defined_queries": len(defined)}
    summary = {"status": "measured", "scope": "uniform validation events including retrieval misses",
               "n_queries": n, "n_retrieval_misses_at_500": int((np.asarray(lists["target_pos"])[rows] < 0).sum()),
               "cohort_identity_sha256": cohort_identity_sha256,
               "mean_spearman": spearman["all500"]["mean"], "spearman_by_band": spearman,
               "added_top10_share": _finite_mean(added_share), "top10_defined_queries": int(np.isfinite(added_share).sum()),
               "probe_candidate_retrieval_rank": 100, "probe_valid_queries": int(probe_valid.sum()),
               "probe_grid": PROBE_GRID.tolist(),
               "mean_probe_logits": [_finite_mean(probe_logits[:, i]) for i in range(len(PROBE_GRID))],
               "low_vs_mid": _probe_difference(probe_logits, 2, 3),
               "extreme_low_vs_high": _probe_difference(probe_logits, 0, 6),
               "interpretation": "Query-level associations and artificial one-input logit sensitivity; not accuracy, calibrated probabilities or a causal training-policy effect. Missing/padded rank100 candidates are excluded from probes; constant-score correlations are undefined."}
    arrays = {"inst": np.asarray(lists["inst"])[rows], "user": np.asarray(lists["user"])[rows],
              "list_rows": rows, "candidates": candidates, "raw_scores": raw, "valid": valid,
              "target_pos": np.asarray(lists["target_pos"])[rows], "logits": logits,
              "spearman_by_query": correlations, "added_top10_share": added_share,
              "top10_columns": top10_columns, "probe_valid": probe_valid,
              "probe_candidate": candidates[:, 99], "probe_original_logits": probe_original,
              "probe_grid": PROBE_GRID, "probe_logits": probe_logits, **probe_features}
    return summary, arrays
