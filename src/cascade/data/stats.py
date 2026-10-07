"""Dataset summary for the critique and later error analysis."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cascade.data.instances import ProcessedData
from cascade.data.split import PERIOD_NAMES, assign_periods
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash


def _date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main(config_path: str, toy: bool = False) -> Path:
    data = ProcessedData.load(config_path, toy)
    table = data.interaction_table
    items = data.items
    split = data.split
    n_users, n_items = data.n_users, data.n_items
    per_period = {}
    for code, name in enumerate(PERIOD_NAMES):
        rows = table.loc[table["period"] == code]
        cases = data.split_instances(name)
        active_users, active_items = rows["user"].nunique(), rows["item"].nunique()
        per_period[name] = {
            "active_users": int(active_users), "active_items": int(active_items),
            "interactions": len(rows),
            "density_active": float(len(rows) / max(1, active_users * active_items)),
            "skipped_empty_history": int(len(rows) - len(cases)),
            "instances": len(cases),
        }

    seq_counts = np.diff(data.user_ptr)[1:]
    out = {
        "toy": toy,
        "split": {**split, "dates": {**{key: _date(int(split[key])) for key in ("t_ab", "t1", "t2")},
                                     "min_ts": _date(int(table["ts"].min())),
                                     "max_ts": _date(int(table["ts"].max()))}},
        "preprocess": json.loads((data.dir / "preprocess.json").read_text()),
        "overall": {"n_users": n_users, "n_items": n_items,
                    "n_interactions": len(table), "density": float(len(table) / (n_users * n_items))},
        "per_period": per_period,
        "items_first_seen_in_period": {
            name: int((assign_periods(items["first_ts"].to_numpy(),
                                      split["t_ab"], split["t1"], split["t2"]) == code).sum())
            for code, name in enumerate(PERIOD_NAMES)
        },
        "user_sequence_length": {"mean": float(seq_counts.mean()), "median": float(np.median(seq_counts)),
                                 "p90": float(np.quantile(seq_counts, 0.9)), "max": int(seq_counts.max())},
        "history_length_at_instance": {},
        "interactions_per_item": {},
        "item_segments": {segment: int((items["segment"] == segment).sum())
                          for segment in ("cold", "tail", "head")},
        "target_segment_share": {},
        "metadata_coverage": {},
        "git_hash": git_hash(),
    }
    for name in ("B", "val", "test"):
        cases = data.split_instances(name)
        lengths = cases["hist_len"].to_numpy()
        out["history_length_at_instance"][name] = {
            "mean": float(lengths.mean()), "median": float(np.median(lengths)),
            "p90": float(np.quantile(lengths, 0.9)), "max": int(lengths.max()),
            "buckets": {"1-2": float(np.mean(lengths <= 2)),
                        "3-5": float(np.mean((lengths >= 3) & (lengths <= 5))),
                        "6-10": float(np.mean((lengths >= 6) & (lengths <= 10))),
                        "11+": float(np.mean(lengths >= 11))},
        }
    item_counts = table.groupby("item").size().to_numpy()
    out["interactions_per_item"] = {"mean": float(item_counts.mean()),
                                    "median": float(np.median(item_counts)),
                                    "p90": float(np.quantile(item_counts, 0.9)),
                                    "max": int(item_counts.max())}
    segments = np.concatenate((["pad"], items["segment"].to_numpy()))
    for name in ("val", "test"):
        target_segments = segments[data.split_instances(name)["item"].to_numpy()]
        out["target_segment_share"][name] = {
            segment: float(np.mean(target_segments == segment))
            for segment in ("cold", "tail", "head")
        }
    coverage = out["metadata_coverage"]
    coverage["has_meta"] = float(items["has_meta"].mean())
    for column in ("title", "leaf_category", "store"):
        coverage[column if column != "leaf_category" else "categories"] = float(items[column].ne("").mean())
    coverage["parseable_price"] = float(items["price"].notna().mean())
    coverage["n_distinct_leaf_cat"] = int(items.loc[items["leaf_category"].ne(""), "leaf_category"].nunique())
    coverage["n_distinct_top_cat"] = int(items.loc[items["top_category"].ne(""), "top_category"].nunique())
    coverage["n_distinct_store"] = int(items.loc[items["store"].ne(""), "store"].nunique())

    destination = REPO_ROOT / "results" / "stats"
    destination.mkdir(parents=True, exist_ok=True)
    suffix = "_toy" if toy else ""
    path = destination / f"stats{suffix}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for values, title, filename in (
        (seq_counts, "Reviews per user", "user_activity_loglog"),
        (item_counts, "Reviews per item", "item_popularity_loglog"),
        (data.split_instances("test")["hist_len"].to_numpy(), "Test history length", "history_length_test"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=np.logspace(0, np.log10(max(2, int(np.max(values)))), 40))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(destination / f"{filename}{suffix}.png", dpi=140)
        plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    main(args.config, args.toy)
