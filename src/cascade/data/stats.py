"""Dataset statistics for the critique (CLAUDE.md §4.8). Writes results/stats[_toy]/stats.json + figures."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash


def _desc(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": int(np.max(x))}


def main(config_path: str, toy: bool) -> None:
    data = ProcessedData.load(config_path, toy)
    out = REPO_ROOT / "results" / ("stats_toy" if toy else "stats")
    out.mkdir(parents=True, exist_ok=True)
    inst_log = json.loads((data.dir / "instances_log.json").read_text())
    pre_log = json.loads((data.dir / "preprocess_log.json").read_text())
    s: dict = {"toy": toy, "split": data.split, "preprocess": pre_log}

    all_inter = data.interactions(["A", "B", "val", "test"])
    s["overall"] = {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_interactions": len(all_inter),
        "density": len(all_inter) / (data.n_users * data.n_items),
    }
    s["per_period"] = {}
    for p in ["A", "B", "val", "test"]:
        df = data.interactions([p])
        nu, ni = df["user"].nunique(), df["item"].nunique()
        s["per_period"][p] = {
            "active_users": nu,
            "active_items": ni,
            "interactions": len(df),
            "density_active": len(df) / (nu * ni),
            **inst_log[p],
        }
    # items that first appear in each period (new releases)
    first_period = pd.cut(data.items["first_ts"], [-np.inf, data.split["t_ab"], data.split["t1"], data.split["t2"], np.inf], right=False, labels=["A", "B", "val", "test"])
    s["items_first_seen_in_period"] = first_period.value_counts().reindex(["A", "B", "val", "test"]).astype(int).to_dict()

    seq_len = np.diff(data.user_ptr)[1:]
    s["user_sequence_length"] = _desc(seq_len)
    s["history_length_at_instance"] = {}
    for p in ["B", "val", "test"]:
        hl = data.split_instances(p)["hist_len"].values
        s["history_length_at_instance"][p] = {**_desc(hl), "buckets": {
            "1-2": float(np.mean(hl <= 2)), "3-5": float(np.mean((hl >= 3) & (hl <= 5))),
            "6-10": float(np.mean((hl >= 6) & (hl <= 10))), "11+": float(np.mean(hl >= 11))}}

    item_cnt = np.bincount(all_inter["item"].values, minlength=data.n_items + 1)[1:]
    s["interactions_per_item"] = _desc(item_cnt)
    seg = data.items.set_index("item")["segment"]
    s["item_segments"] = data.items["segment"].value_counts().to_dict()
    s["target_segment_share"] = {}
    for p in ["val", "test"]:
        t = data.split_instances(p)["item"].map(seg).value_counts(normalize=True)
        s["target_segment_share"][p] = {k: float(t.get(k, 0.0)) for k in ["cold", "tail", "head"]}

    it = data.items
    s["metadata_coverage"] = {
        "has_meta": float(it["has_meta"].mean()),
        "title": float((it["title"].str.len() > 0).mean()),
        "categories": float((it["categories"].apply(len) > 0).mean()),
        "store": float((it["store"].str.len() > 0).mean()),
        "parseable_price": float(it["price"].notna().mean()),
        "n_distinct_leaf_cat": int(it.loc[it["leaf_cat"] != "", "leaf_cat"].nunique()),
        "n_distinct_top_cat": int(it.loc[it["top_cat"] != "", "top_cat"].nunique()),
        "n_distinct_store": int(it.loc[it["store"] != "", "store"].nunique()),
    }
    s["git_hash"] = git_hash()
    (out / "stats.json").write_text(json.dumps(s, indent=2, default=int))

    # figures
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals, cnts = np.unique(item_cnt, return_counts=True)
    ax.loglog(vals, cnts, ".", ms=3)
    ax.set_xlabel("interactions per item"); ax.set_ylabel("#items"); ax.set_title("Item popularity (log–log)")
    fig.tight_layout(); fig.savefig(out / "item_popularity_loglog.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    hl = data.split_instances("test")["hist_len"].values
    ax.hist(np.clip(hl, 1, 60), bins=np.arange(1, 62) - 0.5)
    ax.set_yscale("log"); ax.set_xlabel("history length at test instance (clipped at 60)"); ax.set_ylabel("#instances")
    fig.tight_layout(); fig.savefig(out / "history_length_test.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    v, c = np.unique(seq_len, return_counts=True)
    ax.loglog(v, c, ".", ms=3); ax.set_xlabel("interactions per user"); ax.set_ylabel("#users")
    fig.tight_layout(); fig.savefig(out / "user_activity_loglog.png", dpi=150); plt.close(fig)
    print(json.dumps({k: s[k] for k in ["overall", "per_period", "target_segment_share", "metadata_coverage", "history_length_at_instance", "items_first_seen_in_period"]}, indent=1, default=int))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()
    main(a.config, a.toy)
