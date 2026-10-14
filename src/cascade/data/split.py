"""Global time-based split into periods A / B / Val / Test (CLAUDE.md §4.4) and item segments (§4.7)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cascade.data.preprocess import output_dir
from cascade.utils.config import load_config
from cascade.utils.logging import get_logger

log = get_logger(__name__)

PERIOD_A, PERIOD_B, PERIOD_VAL, PERIOD_TEST = 0, 1, 2, 3
PERIOD_NAMES = {PERIOD_A: "A", PERIOD_B: "B", PERIOD_VAL: "val", PERIOD_TEST: "test"}


def ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def compute_t_ab(timestamps: np.ndarray, t1: int, period_b_fraction: float) -> int:
    """T_AB = timestamp at the (1 - period_b_fraction) quantile of training-period (< T1) interactions."""
    train_ts = np.sort(timestamps[timestamps < t1])
    idx = int(np.floor((1.0 - period_b_fraction) * len(train_ts)))
    return int(train_ts[min(idx, len(train_ts) - 1)])


def assign_periods(timestamps: np.ndarray, t_ab: int, t1: int, t2: int) -> np.ndarray:
    """Period code per interaction: A (< T_AB), B ([T_AB, T1)), Val ([T1, T2)), Test (>= T2)."""
    p = np.full(len(timestamps), PERIOD_A, dtype=np.int8)
    p[timestamps >= t_ab] = PERIOD_B
    p[timestamps >= t1] = PERIOD_VAL
    p[timestamps >= t2] = PERIOD_TEST
    return p


def period_start(period: int, split: dict) -> int:
    """First timestamp of a period (A starts at -inf, represented as the int64 minimum)."""
    return {PERIOD_A: np.iinfo(np.int64).min, PERIOD_B: split["t_ab"], PERIOD_VAL: split["t1"], PERIOD_TEST: split["t2"]}[period]


def item_segments(inter: pd.DataFrame, n_items: int, t2: int, head_fraction: float = 0.2, top_pop_fraction: float = 0.01) -> pd.DataFrame:
    """cold/tail/head by interaction count before T2; plus a top-1% popularity flag (for concentration)."""
    cnt = np.bincount(inter.loc[inter["timestamp"] < t2, "item"].values, minlength=n_items + 1)[1:]
    items = np.arange(1, n_items + 1)
    seg = np.full(n_items, "cold", dtype=object)
    warm = np.where(cnt > 0)[0]
    # order warm items by count desc, item id asc (deterministic tie-break)
    order = warm[np.lexsort((items[warm], -cnt[warm]))]
    n_head = int(np.ceil(head_fraction * len(warm)))
    seg[order[:n_head]] = "head"
    seg[order[n_head:]] = "tail"
    top = np.zeros(n_items, dtype=bool)
    top[order[: int(np.ceil(top_pop_fraction * len(warm)))]] = True
    return pd.DataFrame({"item": items, "cnt_before_t2": cnt, "segment": seg, "top1pct": top})


def main(config_path: str, toy: bool) -> None:
    cfg = load_config(config_path)
    d = output_dir(cfg, toy)
    inter = pd.read_parquet(d / "interactions.parquet")
    n_items = int(inter["item"].max())
    s = cfg["split"]
    t_ab = compute_t_ab(inter["timestamp"].values, s["t1"], s["period_b_fraction"])
    assert inter["timestamp"].min() < t_ab < s["t1"] < s["t2"]
    inter["period"] = assign_periods(inter["timestamp"].values, t_ab, s["t1"], s["t2"])
    inter.to_parquet(d / "interactions.parquet", index=False)

    split = {"t_ab": t_ab, "t1": s["t1"], "t2": s["t2"], "period_b_fraction": s["period_b_fraction"]}
    split["dates"] = {k: ms_to_date(split[k]) for k in ("t_ab", "t1", "t2")}
    split["dates"]["min_ts"] = ms_to_date(int(inter["timestamp"].min()))
    split["dates"]["max_ts"] = ms_to_date(int(inter["timestamp"].max()))
    split["interactions_per_period"] = {PERIOD_NAMES[k]: int(v) for k, v in inter["period"].value_counts().sort_index().items()}
    (d / "split.json").write_text(json.dumps(split, indent=2))
    item_segments(inter, n_items, s["t2"]).to_parquet(d / "item_segments.parquet", index=False)
    log.info("split: %s", json.dumps(split))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()
    main(a.config, a.toy)
