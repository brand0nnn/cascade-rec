"""Where the frozen retriever's Val targets rank, and how many Val targets have no ID-era history.

Validation only, exploratory; reads saved ranks, never touches Test, trains nothing.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.retrieval_misses --run-id 20261029-retrieval-misses

Writes results/analysis/<run-id>/{rank_buckets.csv, no_id_share.csv, results.json, config.yaml, git_hash.txt}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.eval.ranker_meso import group_labels
from cascade.utils.io import REPO_ROOT, make_run_dir, save_run

LABEL = "validation only, exploratory"
EDGES = (200, 500, 1_000, 2_000, 5_000, 20_000, np.inf)
NAMES = ("<=200", "201-500", "501-1K", "1K-2K", "2K-5K", "5K-20K", ">20K")


def rank_bucket_shares(ranks: np.ndarray) -> list[float]:
    """Share of instances per rank bucket (misses / unscorable count as rank = inf, i.e. the last bucket)."""
    lower = (0,) + EDGES[:-1]
    return [float(np.mean((ranks > lo) & (ranks <= hi))) for lo, hi in zip(lower, EDGES)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ranks", default="results/retriever/two_tower_text/seed0/val_ranks.npz")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()
    out = make_run_dir("analysis", args.run_id, exist_ok=False)
    data = ProcessedData.load(args.data_config)
    val = data.split_instances("val")
    with np.load(REPO_ROOT / args.ranks) as z:
        if not np.array_equal(z["inst"], val["inst"].to_numpy()):
            raise ValueError("saved ranks are not aligned to Val instances")
        ranks = z["rank"].astype(float)
        ranks[~z["scorable"]] = np.inf
    item_type = group_labels(data, val)["item_type"].to_numpy()
    # An ID vector exists only for products with >= 1 retriever-training interaction (CLAUDE.md §7.1, id_text).
    seen = np.bincount(data.interactions(["retriever_train"])["item"].to_numpy(), minlength=data.n_items + 1) > 0
    no_id = ~seen[val["item"].to_numpy()]

    rows, id_rows = [], []
    for name, mask in [("all", np.ones(len(val), dtype=bool))] + [(g, item_type == g) for g in sorted(set(item_type))]:
        rows.append({"group": name, "n_instances": int(mask.sum()), "median_rank": float(np.median(ranks[mask])),
                     **dict(zip(NAMES, rank_bucket_shares(ranks[mask])))})
        id_rows.append({"group": name, "n_instances": int(mask.sum()), "share_of_targets": float(mask.mean()),
                        "share_without_id_vector": float(no_id[mask].mean())})
    buckets, ids = pd.DataFrame(rows), pd.DataFrame(id_rows)
    buckets.to_csv(out / "rank_buckets.csv", index=False)
    ids.to_csv(out / "no_id_share.csv", index=False)
    result = {"label": LABEL, "n_instances": int(len(val)), "catalogue_items": int(data.n_items),
              "share_catalogue_with_id_vector": float(seen[1:].mean()),
              "rank_buckets": rows, "no_id_share": id_rows}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    save_run(out, {"label": LABEL, "n_instances": int(len(val))},
             {"label": LABEL, "split": "val", "ranks": args.ranks, "data_config": args.data_config,
              "rank_edges": [str(e) for e in EDGES]})
    print(buckets.round(3).to_string(index=False))
    print(ids.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
