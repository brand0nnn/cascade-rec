"""Prepare the 5-core rating file and join metadata for its catalogue."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.utils.config import REPO_ROOT, load_config


def output_dir(config: dict, toy: bool = False) -> Path:
    """Directory holding one complete processed dataset."""
    base = REPO_ROOT / config["processed_dir"]
    return base / "toy" if toy else base


def _price(value: object) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d[\d,]*(?:\.\d+)?", value)
        if match:
            return float(match.group().replace(",", ""))
    return float("nan")


def _read_metadata(path: Path, item_ids: dict[str, int]) -> pd.DataFrame:
    records: dict[int, dict] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            item = item_ids.get(record.get("parent_asin"))
            if item is None or item in records:
                continue
            categories = record.get("categories") or []
            if not isinstance(categories, list):
                categories = []
            records[item] = {
                "title": str(record.get("title") or ""),
                "categories": [str(x) for x in categories],
                "leaf_category": " > ".join(map(str, categories)),
                "top_category": str(categories[0]) if categories else "",
                "store": str(record.get("store") or ""),
                "price": _price(record.get("price")),
                "has_meta": True,
            }
    return pd.DataFrame.from_dict(records, orient="index")


def main(config_path: str, toy: bool = False) -> Path:
    cfg = load_config(config_path)
    raw = REPO_ROOT / cfg["raw_dir"]
    directory = output_dir(cfg, toy)
    directory.mkdir(parents=True, exist_ok=True)
    source = raw / cfg["sources"]["interactions"]["file"]
    frame = pd.read_csv(source, usecols=["user_id", "parent_asin", "rating", "timestamp"])
    raw_rows = len(frame)
    frame = frame.drop_duplicates()
    exact_rows = len(frame)
    frame = frame.sort_values(["timestamp", "parent_asin"], kind="stable")
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="first")
    dedup_rows = len(frame)
    if toy:
        users = np.sort(frame["user_id"].unique())
        rng = np.random.default_rng(int(cfg["toy"]["seed"]))
        count = max(1, round(len(users) * float(cfg["toy"]["user_fraction"])))
        chosen = rng.choice(users, size=count, replace=False)
        frame = frame[frame["user_id"].isin(chosen)].copy()
    user_tokens = np.sort(frame["user_id"].unique())
    item_tokens = np.sort(frame["parent_asin"].unique())
    user_ids = {token: index + 1 for index, token in enumerate(user_tokens)}
    item_ids = {token: index + 1 for index, token in enumerate(item_tokens)}
    frame["user"] = frame["user_id"].map(user_ids).astype(np.int32)
    frame["item"] = frame["parent_asin"].map(item_ids).astype(np.int32)
    frame = frame.rename(columns={"timestamp": "ts"})
    frame["timestamp"] = frame["ts"]
    frame = frame[["user", "item", "ts", "timestamp", "rating"]].sort_values(
        ["user", "ts", "item"], kind="stable").reset_index(drop=True)
    frame.to_parquet(directory / "interactions.parquet", index=False)

    items = pd.DataFrame({"item": np.arange(1, len(item_tokens) + 1, dtype=np.int32),
                          "parent_asin": item_tokens})
    first = frame.groupby("item", sort=False)["ts"].min()
    items["first_ts"] = items["item"].map(first).astype(np.int64)
    meta = _read_metadata(raw / cfg["sources"]["metadata"]["file"], item_ids)
    if not meta.empty:
        meta.index.name = "item"
        items = items.join(meta, on="item")
    for column, default in (("title", ""), ("leaf_category", ""), ("top_category", ""), ("store", "")):
        if column not in items:
            items[column] = default
        items[column] = items[column].fillna(default)
    if "categories" not in items:
        items["categories"] = [[] for _ in range(len(items))]
    items["categories"] = items["categories"].map(lambda x: x if isinstance(x, list) else [])
    if "price" not in items:
        items["price"] = np.nan
    if "has_meta" not in items:
        items["has_meta"] = False
    items["has_meta"] = items["has_meta"].fillna(False).astype(bool)

    before_test = frame[frame["ts"] < int(cfg["split"]["t2"])].groupby("item").size()
    items["pretest_count"] = items["item"].map(before_test).fillna(0).astype(np.int32)
    warm = items[items["pretest_count"] > 0].sort_values(
        ["pretest_count", "item"], ascending=[False, True], kind="stable")
    head_count = int(np.ceil(len(warm) * 0.2))
    head = set(warm.head(head_count)["item"])
    top_count = max(1, int(np.ceil(len(items) * 0.01)))
    top_pop = set(items.sort_values(["pretest_count", "item"], ascending=[False, True]).head(top_count)["item"])
    items["segment"] = np.where(items["pretest_count"] == 0, "cold",
                                 np.where(items["item"].isin(head), "head", "tail"))
    items["top1pct"] = items["item"].isin(top_pop)
    items.to_parquet(directory / "items.parquet", index=False)
    items[["item", "segment", "top1pct", "pretest_count"]].to_parquet(directory / "item_segments.parquet", index=False)
    counts = {"toy": toy, "raw_rows": raw_rows, "after_exact_dedup": exact_rows,
              "after_user_item_dedup": dedup_rows, "n_users": len(user_tokens),
              "n_items": len(item_tokens), "n_interactions": len(frame),
              "items_without_metadata": int((~items["has_meta"]).sum())}
    (directory / "preprocess.json").write_text(json.dumps(counts, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    main(args.config, args.toy)
