"""Instances = (user, history, target, timestamp, period) (CLAUDE.md §4.5) and the shared data container.

History of an instance at time t = all of the user's interactions with timestamp strictly < t,
truncated to the most recent ``max_hist_len`` when fed to models. Instances with empty history are skipped.

Storage: per-user sequences in CSR form (``user_ptr``, ``seq_item``, ``seq_ts``); an instance points at
its user and stores ``hist_len`` = number of the user's interactions strictly before t, so its history is
``seq_item[user_ptr[u] : user_ptr[u] + hist_len]``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.preprocess import output_dir
from cascade.data.split import PERIOD_NAMES
from cascade.utils.config import load_config
from cascade.utils.logging import get_logger

log = get_logger(__name__)

SPLIT_PERIODS = {"retriever_train": 0, "ranker_train": 1, "val": 2, "test": 3}


def build_sequences(inter: pd.DataFrame, n_users: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR user sequences from interactions sorted by (user, timestamp, parent_asin)."""
    counts = np.bincount(inter["user"].values, minlength=n_users + 1)
    user_ptr = np.zeros(n_users + 2, dtype=np.int64)
    user_ptr[1:] = np.cumsum(counts)
    return user_ptr, inter["item"].values.astype(np.int32), inter["timestamp"].values.astype(np.int64)


def history_lengths(users: np.ndarray, ts: np.ndarray, user_ptr: np.ndarray) -> np.ndarray:
    """#interactions of the same user strictly before each row's timestamp (rows sorted by user, ts)."""
    pos = np.arange(len(users)) - user_ptr[users]
    # rows sharing (user, ts): history stops at the first of them
    new_group = np.ones(len(users), dtype=bool)
    new_group[1:] = (users[1:] != users[:-1]) | (ts[1:] != ts[:-1])
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(users)), 0))
    return pos - (np.arange(len(users)) - group_start)


def build_instances(inter: pd.DataFrame, user_ptr: np.ndarray) -> tuple[pd.DataFrame, dict]:
    hl = history_lengths(inter["user"].values, inter["timestamp"].values, user_ptr)
    inst = pd.DataFrame(
        {
            "inst": np.arange(len(inter), dtype=np.int64),
            "user": inter["user"].values,
            "item": inter["item"].values,
            "ts": inter["timestamp"].values,
            "period": inter["period"].values,
            "hist_len": hl.astype(np.int32),
        }
    )
    log_info = {}
    for code, name in PERIOD_NAMES.items():
        m = inst["period"] == code
        log_info[name] = {"interactions": int(m.sum()), "skipped_empty_history": int((m & (inst["hist_len"] == 0)).sum())}
        log_info[name]["instances"] = log_info[name]["interactions"] - log_info[name]["skipped_empty_history"]
    return inst[inst["hist_len"] > 0].reset_index(drop=True), log_info


@dataclass
class ProcessedData:
    """Everything the harness and models need, loaded from ``data/processed[/sample]``."""

    dir: Path
    n_users: int
    n_items: int
    user_ptr: np.ndarray
    seq_item: np.ndarray
    seq_ts: np.ndarray
    instances: pd.DataFrame
    items: pd.DataFrame  # item metadata + first_ts + segment + cnt_before_t2 + top1pct
    split: dict
    max_hist_len: int

    @classmethod
    def load(cls, config_path: str = "configs/data.yaml", sample: bool = False) -> "ProcessedData":
        cfg = load_config(config_path)
        d = output_dir(cfg, sample)
        z = np.load(d / "sequences.npz")
        items = pd.read_parquet(d / "items.parquet").merge(pd.read_parquet(d / "item_segments.parquet"), on="item")
        items = items.sort_values("item").reset_index(drop=True)
        return cls(
            dir=d,
            n_users=int(z["n_users"]),
            n_items=int(z["n_items"]),
            user_ptr=z["user_ptr"],
            seq_item=z["seq_item"],
            seq_ts=z["seq_ts"],
            instances=pd.read_parquet(d / "instances.parquet"),
            items=items,
            split=json.loads((d / "split.json").read_text()),
            max_hist_len=cfg["max_hist_len"],
        )

    def split_instances(self, name: str) -> pd.DataFrame:
        return self.instances[self.instances["period"] == SPLIT_PERIODS[name]].reset_index(drop=True)

    def interactions(self, periods: list[str]) -> pd.DataFrame:
        """All interactions (user, item, ts) in the given periods, incl. first interactions (empty history)."""
        users = np.repeat(np.arange(len(self.user_ptr) - 1), np.diff(self.user_ptr))
        df = pd.DataFrame({"user": users, "item": self.seq_item, "ts": self.seq_ts})
        s = self.split
        bounds = {"retriever_train": (-np.inf, s["t_ranker_train"]), "ranker_train": (s["t_ranker_train"], s["t1"]), "val": (s["t1"], s["t2"]), "test": (s["t2"], np.inf)}
        m = np.zeros(len(df), dtype=bool)
        for p in periods:
            lo, hi = bounds[p]
            m |= (df["ts"].values >= lo) & (df["ts"].values < hi)
        return df[m].reset_index(drop=True)

    def histories(self, users: np.ndarray, hist_lens: np.ndarray, max_len: int | None) -> np.ndarray:
        """Left-padded [B, max_len] histories (most recent last). ``max_len=None`` → full history length."""
        if max_len is None:
            max_len = int(hist_lens.max()) if len(hist_lens) else 1
        out = np.zeros((len(users), max_len), dtype=np.int64)
        for r, (u, hl) in enumerate(zip(users, hist_lens)):
            start = self.user_ptr[u]
            h = self.seq_item[start + max(0, hl - max_len) : start + hl]
            out[r, max_len - len(h) :] = h
        return out


def main(config_path: str, sample: bool) -> None:
    cfg = load_config(config_path)
    d = output_dir(cfg, sample)
    inter = pd.read_parquet(d / "interactions.parquet")
    n_users, n_items = int(inter["user"].max()), int(inter["item"].max())
    user_ptr, seq_item, seq_ts = build_sequences(inter, n_users)
    np.savez(d / "sequences.npz", user_ptr=user_ptr, seq_item=seq_item, seq_ts=seq_ts, n_users=n_users, n_items=n_items)
    inst, info = build_instances(inter, user_ptr)
    inst.to_parquet(d / "instances.parquet", index=False)
    (d / "instances_log.json").write_text(json.dumps(info, indent=2))
    log.info("instances: %s", json.dumps(info))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    main(a.config, a.sample)
