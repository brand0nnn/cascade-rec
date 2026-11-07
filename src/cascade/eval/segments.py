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


def segment_masks(target_segment: np.ndarray, is_new: np.ndarray, hist_len: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean instance masks for every §4.7 segment, keyed as in the harness output (``item_cold`` … ``hist_11+``).

    ``target_segment``: cold/tail/head label of each instance's target; ``is_new``: from ``new_in_period``.
    """
    hb = hist_bucket(hist_len)
    return {
        **{f"item_{s}": target_segment == s for s in ITEM_SEGMENTS},
        "item_new_in_period": is_new,
        "item_not_new": ~is_new,
        **{f"hist_{h}": hb == h for h in HIST_BUCKETS},
    }
