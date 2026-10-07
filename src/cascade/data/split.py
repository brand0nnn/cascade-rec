"""Chronological A/B/validation/test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.utils.config import load_config
from cascade.data.preprocess import output_dir

PERIOD_A = 0
PERIOD_B = 1
PERIOD_VAL = 2
PERIOD_TEST = 3
PERIOD_NAMES = ("A", "B", "val", "test")


def compute_t_ab(timestamps: np.ndarray, t1: int, period_b_fraction: float) -> int:
    """First timestamp in B, with roughly ``period_b_fraction`` of train rows in B."""
    if not 0 < period_b_fraction < 1:
        raise ValueError("period_b_fraction must lie between zero and one")
    train = np.asarray(timestamps, dtype=np.int64)
    train = train[train < t1]
    if len(train) == 0:
        raise ValueError("no interactions before t1")
    return int(np.quantile(train, 1 - period_b_fraction, method="higher"))


def assign_periods(timestamps: np.ndarray, t_ab: int, t1: int, t2: int) -> np.ndarray:
    """Assign integer period IDs using half-open time ranges."""
    if not t_ab < t1 < t2:
        raise ValueError("expected t_ab < t1 < t2")
    return np.searchsorted(np.array([t_ab, t1, t2], dtype=np.int64), timestamps, side="right").astype(np.int8)


def main(config_path: str, toy: bool = False) -> Path:
    cfg = load_config(config_path)
    directory = output_dir(cfg, toy)
    path = directory / "interactions.parquet"
    frame = pd.read_parquet(path)
    ts = frame["timestamp"].to_numpy(dtype=np.int64)
    t1, t2 = int(cfg["split"]["t1"]), int(cfg["split"]["t2"])
    t_ab = compute_t_ab(ts, t1, float(cfg["split"]["period_b_fraction"]))
    frame["period"] = assign_periods(ts, t_ab, t1, t2)
    frame.to_parquet(path, index=False)
    split = {"t_ab": t_ab, "t1": t1, "t2": t2,
             "period_b_fraction": float(cfg["split"]["period_b_fraction"]),
             "interactions_per_period": {name: int((frame["period"] == code).sum()) for code, name in enumerate(PERIOD_NAMES)}}
    (directory / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    main(args.config, args.toy)
