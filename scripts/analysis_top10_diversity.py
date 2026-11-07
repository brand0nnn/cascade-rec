"""Beyond-accuracy metrics of the actual top-10 lists (validation, natural K200, seed 0).

For each system, rank every natural K200 validation list and keep the top 10. Report:
  coverage           distinct products in any top-10 / catalogue size N
  top1pct_share      share of top-10 slots filled by the 1% most-interacted products (counts before T2, CLAUDE.md §5)
  new_share          share of slots filled by products first seen at/after T1 (new in Val)
  median_popularity  median pre-T1 interaction count of recommended products
  leaf_diversity     mean distinct leaf categories within a list
  top_item_share     share of lists containing the single most-recommended product
Validation only, no Test. Retrieval order = no ranker (R1).

Usage: .venv/bin/python scripts/analysis_top10_diversity.py OUTPUT_DIR [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cascade.data.instances import ProcessedData
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.utils.config import load_config

import importlib.util
_spec = importlib.util.spec_from_file_location("cases", Path(__file__).with_name("analysis_micro_cases.py"))
cases = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cases)

ROOT = Path(__file__).resolve().parents[1]
K1, TOP = 200, 10
SYSTEMS = {"A_forced_mlp": ROOT / "results/ranker_policy/policy-20261106-pilot/arm_A_seed0",
           "C_natural_mlp": ROOT / "results/ranker_policy/policy-20261106-pilot/arm_C_seed0",
           "DIN_forced": ROOT / "results/ranker_din/20261024-din-matching-seed0"}


def top10(model, builder, lists, rows: np.ndarray, batch: int = 64) -> np.ndarray:
    out = np.zeros((len(rows), TOP), dtype=np.int64)
    with torch.inference_mode():
        for s in tqdm(range(0, len(rows), batch), mininterval=20):
            rr = rows[s:s + batch]
            cand = lists["cand"][rr, :K1]
            valid = cand > 0
            pr = np.repeat(rr, K1)[valid.ravel()]
            pc = np.tile(np.arange(K1), len(rr))[valid.ravel()]
            feats = builder.build(lists, pr, pc)
            vals = model({k: torch.as_tensor(v) for k, v in feats.items()}).float().numpy()
            scores = np.full(cand.shape, -np.inf, dtype=np.float32)
            scores[valid] = vals
            best = np.argsort(-scores, axis=1, kind="stable")[:, :TOP]
            out[s:s + len(rr)] = np.take_along_axis(cand, best, axis=1)
    return out


def describe(lists_top: np.ndarray, items: pd.DataFrame, pre_t1: np.ndarray, t1: int) -> dict:
    flat = lists_top[lists_top > 0]
    counts = np.bincount(flat, minlength=len(pre_t1))
    leaf = items["leaf_cat"].to_numpy()
    top_item = int(np.argmax(counts))
    return {"lists": int(len(lists_top)), "coverage": float((counts > 0).sum() / (len(items) - 1)),
            "distinct_products": int((counts > 0).sum()),
            "top1pct_share": float(items["top1pct"].to_numpy()[flat].mean()),
            "new_share": float((items["first_ts"].to_numpy()[flat] >= t1).mean()),
            "median_popularity_pre_t1": float(np.median(pre_t1[flat])),
            "leaf_diversity": float(np.mean([len(set(leaf[r[r > 0]])) for r in lists_top])),
            "top_item_share": float(np.mean((lists_top == top_item).any(axis=1))),
            "top_item": int(top_item), "top20_slot_share": float(np.sort(counts)[::-1][:20].sum() / len(flat))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = load_config(ROOT / "configs/ranker/policy_study.yaml")
    data = ProcessedData.load(str(ROOT / cfg["data_config"]))
    items = data.items.set_index("item").reindex(np.arange(data.n_items + 1))
    items["top1pct"] = items["top1pct"].fillna(False).astype(bool)
    items["first_ts"] = items["first_ts"].fillna(np.iinfo(np.int64).max)
    t1 = int(data.split["T1"]) if "T1" in data.split else int(data.split["t1"])
    pre = data.seq_item[data.seq_ts < t1]
    pre_t1 = np.bincount(pre, minlength=data.n_items + 1)
    lists = load_candidates(ROOT / cfg["candidate_dir"], "val", k1=500, force=False)
    rows = np.arange(len(lists["inst"]))
    if a.limit:
        rows = rows[:a.limit]
    results = {}
    retrieval = lists["cand"][rows, :TOP].astype(np.int64)
    results["R1_retrieval_order"] = describe(retrieval, items, pre_t1, t1)
    np.save(a.output / "top10_R1_retrieval_order.npy", retrieval)
    builder = frozen_feature_builder(data, load_candidates(ROOT / cfg["candidate_dir"], "ranker_train"),
                                     ROOT / cfg["reference_ranker_config"])
    for name, run in SYSTEMS.items():
        model = cases.load_model(run, builder)
        t = top10(model, builder, lists, rows)
        np.save(a.output / f"top10_{name}.npy", t)
        results[name] = describe(t, items, pre_t1, t1)
        print(name, json.dumps(results[name]), flush=True)
    (a.output / "diversity.json").write_text(json.dumps(
        {"systems": results, "inst": "natural K200 validation lists, seed-0 checkpoints",
         "n_items": int(data.n_items), "limit": a.limit, "test_predictions_loaded": False}, indent=2))
    pd.DataFrame(results).T.to_csv(a.output / "diversity.csv")
    print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
