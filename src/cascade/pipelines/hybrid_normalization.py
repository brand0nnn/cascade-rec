"""Retrieval-score statistics for the natural-only hybrid ranker recipe."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def natural_score_stats(lists: Mapping[str, np.ndarray]) -> tuple[float, float]:
    """Match training: exclude forced lists, padding, and the appended target slot.

    Only ranker-training lists may be passed here. Validation and Test always
    reuse these statistics; their own score distribution must not be fitted.
    Float32 reductions intentionally match the original training implementation.
    """
    natural = ~np.asarray(lists["forced"], dtype=bool)
    cand = np.asarray(lists["cand"])[natural, :-1]
    scores = np.asarray(lists["score"])[natural, :-1].astype(np.float32)
    values = scores[cand > 0]
    if not values.size or not np.isfinite(values).all():
        raise ValueError("natural training scores must be nonempty and finite")
    mean, std = float(values.mean()), float(values.std())
    if std <= 0:
        raise ValueError("natural training scores must have positive standard deviation")
    return mean, std
