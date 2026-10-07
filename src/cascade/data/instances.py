"""Chronological user histories and evaluation instances."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.preprocess import output_dir
from cascade.data.split import PERIOD_NAMES
from cascade.utils.config import load_config


def build_sequences(interactions: pd.DataFrame, n_users: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return user pointers, item IDs, and timestamps, sorted by user/time/item."""
    ordered = interactions.sort_values(["user", "ts", "item"], kind="stable")
    users = ordered["user"].to_numpy(dtype=np.int64)
    n_users = int(users.max()) if n_users is None else n_users
    counts = np.bincount(users, minlength=n_users + 1)
    ptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return ptr, ordered["item"].to_numpy(dtype=np.int32), ordered["ts"].to_numpy(dtype=np.int64)


def history_lengths(users: np.ndarray, timestamps: np.ndarray, user_ptr: np.ndarray) -> np.ndarray:
    """Count only interactions strictly before each target timestamp."""
    users = np.asarray(users)
    timestamps = np.asarray(timestamps)
    out = np.empty(len(users), dtype=np.int32)
    unique, starts, counts = np.unique(users, return_index=True, return_counts=True)
    for user, start, count in zip(unique, starts, counts):
        rows = slice(int(start), int(start + count))
        sequence_start, sequence_end = int(user_ptr[user]), int(user_ptr[user + 1])
        out[rows] = np.searchsorted(timestamps[sequence_start:sequence_end], timestamps[rows], side="left")
    return out


def build_instances(interactions: pd.DataFrame, user_ptr: np.ndarray) -> pd.DataFrame:
    """Construct cases with a nonempty, strictly earlier user history."""
    ordered = interactions.sort_values(["user", "ts", "item"], kind="stable").reset_index(drop=True)
    users = ordered["user"].to_numpy(dtype=np.int64)
    ts = ordered["ts"].to_numpy(dtype=np.int64)
    ordered["hist_len"] = history_lengths(users, ts, user_ptr)
    result = ordered.loc[ordered["hist_len"] > 0, ["user", "item", "ts", "period", "hist_len"]].copy()
    result.insert(0, "inst", np.flatnonzero(ordered["hist_len"].to_numpy() > 0).astype(np.int64))
    return result.reset_index(drop=True)


@dataclass
class ProcessedData:
    """Arrays and tables used by the baseline adapter and evaluation harness."""

    dir: Path
    items: pd.DataFrame
    instances: pd.DataFrame
    interaction_table: pd.DataFrame
    split: dict
    user_ptr: np.ndarray
    seq_item: np.ndarray
    seq_ts: np.ndarray
    max_hist_len: int

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_users(self) -> int:
        return len(self.user_ptr) - 2

    @classmethod
    def load(cls, config_path: str, toy: bool = False) -> "ProcessedData":
        cfg = load_config(config_path)
        directory = output_dir(cfg, toy)
        import json

        sequences = np.load(directory / "sequences.npz")
        return cls(directory, pd.read_parquet(directory / "items.parquet"),
                   pd.read_parquet(directory / "instances.parquet"),
                   pd.read_parquet(directory / "interactions.parquet"),
                   json.loads((directory / "split.json").read_text()),
                   sequences["user_ptr"], sequences["seq_item"], sequences["seq_ts"],
                   int(cfg["max_hist_len"]))

    def split_instances(self, split: str) -> pd.DataFrame:
        code = PERIOD_NAMES.index(split)
        return self.instances.loc[self.instances["period"] == code].reset_index(drop=True)

    def interactions(self, periods: list[str]) -> pd.DataFrame:
        codes = [PERIOD_NAMES.index(period) for period in periods]
        return self.interaction_table.loc[self.interaction_table["period"].isin(codes)].reset_index(drop=True)

    def histories(self, users: np.ndarray, lengths: np.ndarray, max_len: int) -> np.ndarray:
        """Last ``max_len`` earlier items, left padded with item ID zero."""
        out = np.zeros((len(users), max_len), dtype=np.int64)
        for row, (user, length) in enumerate(zip(users, lengths)):
            end = int(self.user_ptr[int(user)] + length)
            start = max(int(self.user_ptr[int(user)]), end - max_len)
            values = self.seq_item[start:end]
            if len(values):
                out[row, -len(values):] = values
        return out


def main(config_path: str, toy: bool = False) -> Path:
    cfg = load_config(config_path)
    directory = output_dir(cfg, toy)
    interactions = pd.read_parquet(directory / "interactions.parquet")
    n_users = int(interactions["user"].max())
    user_ptr, seq_item, seq_ts = build_sequences(interactions, n_users)
    np.savez_compressed(directory / "sequences.npz", user_ptr=user_ptr, seq_item=seq_item, seq_ts=seq_ts)
    instances = build_instances(interactions, user_ptr)
    instances.to_parquet(directory / "instances.parquet", index=False)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    main(args.config, args.toy)
