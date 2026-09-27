"""Segment labels for analysis (CLAUDE.md §4.7)."""
from __future__ import annotations

import numpy as np

HIST_BUCKETS = ["1-2", "3-5", "6-10", "11+"]
ITEM_SEGMENTS = ["cold", "tail", "head"]


def hist_bucket(hist_len: np.ndarray) -> np.ndarray:
    """User segment by history length at instance time."""
    return np.select([hist_len <= 2, hist_len <= 5, hist_len <= 10], HIST_BUCKETS[:3], default=HIST_BUCKETS[3])
