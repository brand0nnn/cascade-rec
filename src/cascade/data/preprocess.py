"""Clean raw interactions + metadata into integer-ID tables (CLAUDE.md §4.2–4.3).

Outputs in ``<processed_dir>[/toy]``:
  interactions.parquet  user, item, rating, timestamp (sorted by user, timestamp, parent_asin)
  users.parquet         user (int, 1-based), user_id (raw)
  items.parquet         item (int, 1-based), parent_asin, title, categories, store, price_raw, price,
                        text, leaf_cat, top_cat, has_meta, first_ts
  preprocess_log.json   counts
Index 0 is reserved for padding in both user and item ID spaces.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.logging import get_logger

log = get_logger(__name__)

_PRICE_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def parse_price(x) -> float:
    """Parse a metadata price robustly. Returns NaN when missing/non-numeric.

    Handles numbers, '$12.99', '12.99 - 19.99' (takes the lower bound), 'None', '', '—'.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x) if np.isfinite(x) and x >= 0 else np.nan
    m = _PRICE_RE.search(str(x))
    if not m:
        return np.nan
    try:
        v = float(m.group(0).replace(",", ""))
    except ValueError:
        return np.nan
    return v if v >= 0 else np.nan


def item_text(title: str, categories: list[str], store: str) -> str:
    """Item-tower text: ``title | cat1 > cat2 > ... | store`` with empty parts skipped (§7.1)."""
    parts = [title or "", " > ".join(categories or []), store or ""]
    return " | ".join(p.strip() for p in parts if p and p.strip())


def output_dir(cfg: dict, toy: bool) -> Path:
    d = REPO_ROOT / cfg["processed_dir"]
    return d / "toy" if toy else d


def load_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user_id": str, "parent_asin": str, "rating": np.float32, "timestamp": np.int64})
    expected = {"user_id", "parent_asin", "rating", "timestamp"}
    if set(df.columns) != expected:
        raise ValueError(f"unexpected columns {df.columns.tolist()}")
    return df


def clean_interactions(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Drop exact duplicates, keep the earliest review per (user, item), sort deterministically."""
    info["raw_rows"] = len(df)
    df = df.drop_duplicates()
    info["after_exact_dedup"] = len(df)
    df = df.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")
    df = df.drop_duplicates(["user_id", "parent_asin"], keep="first")
    info["after_user_item_dedup"] = len(df)
    return df.reset_index(drop=True)


def load_metadata(path: Path, asins: set[str]) -> pd.DataFrame:
    """Stream the metadata jsonl, keeping only items present in the interaction file.

    ``average_rating`` / ``rating_number`` are deliberately NOT read (leakage, §4.2).
    """
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in tqdm(f, desc="metadata", mininterval=10):
            d = json.loads(line)
            a = d.get("parent_asin")
            if a not in asins:
                continue
            cats = [c for c in (d.get("categories") or []) if isinstance(c, str) and c.strip()]
            rows.append(
                {
                    "parent_asin": a,
                    "title": (d.get("title") or "").strip(),
                    "categories": cats,
                    "store": (d.get("store") or "").strip(),
                    "price_raw": None if d.get("price") is None else str(d.get("price")),
                }
            )
    meta = pd.DataFrame(rows, columns=["parent_asin", "title", "categories", "store", "price_raw"])
    return meta.drop_duplicates("parent_asin", keep="first")


def main(config_path: str, toy: bool) -> None:
    cfg = load_config(config_path)
    raw = REPO_ROOT / cfg["raw_dir"]
    out = output_dir(cfg, toy)
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"toy": toy}

    df = load_interactions(raw / cfg["sources"]["interactions"]["file"])
    df = clean_interactions(df, info)

    if toy:
        users = np.sort(df["user_id"].unique())
        rng = np.random.default_rng(cfg["toy"]["seed"])
        keep = rng.choice(users, size=int(len(users) * cfg["toy"]["user_fraction"]), replace=False)
        df = df[df["user_id"].isin(set(keep))].reset_index(drop=True)
        info["toy_users"] = len(keep)

    # contiguous IDs, 0 = padding; deterministic order by raw id
    user_ids = np.sort(df["user_id"].unique())
    item_asins = np.sort(df["parent_asin"].unique())
    umap = pd.Series(np.arange(1, len(user_ids) + 1, dtype=np.int32), index=user_ids)
    imap = pd.Series(np.arange(1, len(item_asins) + 1, dtype=np.int32), index=item_asins)

    inter = pd.DataFrame(
        {
            "user": umap.loc[df["user_id"].values].values,
            "item": imap.loc[df["parent_asin"].values].values,
            "rating": df["rating"].values.astype(np.float32),
            "timestamp": df["timestamp"].values.astype(np.int64),
        }
    )
    # df is already sorted by (user_id, timestamp, parent_asin) and IDs are order-preserving
    assert (np.diff(inter["user"].values) >= 0).all()
    info.update(n_users=len(user_ids), n_items=len(item_asins), n_interactions=len(inter))

    meta = load_metadata(raw / cfg["sources"]["metadata"]["file"], set(item_asins))
    items = pd.DataFrame({"item": imap.values, "parent_asin": imap.index.values})
    items = items.merge(meta, on="parent_asin", how="left")
    items["has_meta"] = items["title"].notna()
    items["title"] = items["title"].fillna("")
    items["store"] = items["store"].fillna("")
    items["categories"] = items["categories"].apply(lambda c: c if isinstance(c, list) else [])
    items["price"] = items["price_raw"].apply(parse_price).astype(np.float32)
    items["text"] = [item_text(t, c, s) for t, c, s in zip(items["title"], items["categories"], items["store"])]
    items["leaf_cat"] = items["categories"].apply(lambda c: " > ".join(c) if c else "")
    items["top_cat"] = items["categories"].apply(lambda c: c[0] if c else "")
    first_ts = inter.groupby("item")["timestamp"].min()
    items["first_ts"] = items["item"].map(first_ts).astype(np.int64)
    items = items.sort_values("item").reset_index(drop=True)
    info["items_without_metadata"] = int((~items["has_meta"]).sum())
    log.info("items without metadata: %d", info["items_without_metadata"])

    inter.to_parquet(out / "interactions.parquet", index=False)
    pd.DataFrame({"user": umap.values, "user_id": umap.index.values}).to_parquet(out / "users.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    (out / "preprocess_log.json").write_text(json.dumps(info, indent=2))
    log.info("preprocess done: %s", info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()
    main(a.config, a.toy)
