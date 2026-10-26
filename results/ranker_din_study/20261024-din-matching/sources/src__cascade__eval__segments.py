"""Segment labels for analysis (CLAUDE.md §4.7)."""
from __future__ import annotations

import numpy as np

HIST_BUCKETS = ["1-2", "3-5", "6-10", "11+"]
ITEM_SEGMENTS = ["cold", "tail", "head"]


def hist_bucket(hist_len: np.ndarray) -> np.ndarray:
    """User segment by history length at instance time."""
    return np.select([hist_len <= 2, hist_len <= 5, hist_len <= 10], HIST_BUCKETS[:3], default=HIST_BUCKETS[3])


def split_start(split: str, split_info: dict) -> int:
    """First timestamp of an evaluation split (T1 for Val, T2 for Test)."""
    return {"val": split_info["t1"], "test": split_info["t2"]}[split]


def new_in_period(target_first_ts: np.ndarray, split: str, split_info: dict) -> np.ndarray:
    """True if the target item's first interaction is at/after the split start (released during this period).

    On Test this equals the ``cold`` segment; on Val it marks products released during Val.
    """
    return target_first_ts >= split_start(split, split_info)
