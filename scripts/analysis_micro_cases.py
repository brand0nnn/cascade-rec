"""Micro case studies: 24 validation events in four prespecified categories (seed 0).

Categories (natural K200 lists, seed-0 checkpoints, 6 events each, RNG 0 among eligible):
  c_rescues_head   warm-head target: natural-only C ranks it top-10, forced-list A does not
  a_keeps_new      new-in-period target: A ranks it top-10, C does not
  din_rescues_new  new-in-period target: DIN top-10, matching MLP (R2) not
  retrieval_miss   target absent from the top-500 retrieved list (no ranker can recover it)
For each: last five prior purchases, target, retrieval rank, ranker target ranks, and the
top-3 products of A, C and DIN. Recomputed target ranks must equal the saved ones.
Unclicked candidates are unobserved, not known dislikes. Validation only; no Test.

Usage: .venv/bin/python scripts/analysis_micro_cases.py OUTPUT_DIR
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cascade.data.instances import ProcessedData
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "data/processed/candidates/two_tower_text_seed0"
PILOT = ROOT / "results/ranker_policy/policy-20261106-pilot"
RQ3 = ROOT / "results/ranker_comparison/20261025-rq3/validation_ranks.npz"
MODELS = {"A": PILOT / "arm_A_seed0", "C": PILOT / "arm_C_seed0",
          "DIN": ROOT / "results/ranker_din/20261024-din-matching-seed0"}
PER_CATEGORY, K1 = 6, 200


def load_model(run: Path, builder):
    cfg = load_config(run / "config.yaml")
    spec_cfg = cfg["run"]["feature_spec"]
    spec = RankerFeatureSpec(**{**spec_cfg, "feature_groups": tuple(spec_cfg["feature_groups"])})
    if dict(spec.cardinalities) != builder.item_cardinalities:
        raise ValueError(f"feature contract mismatch: {run}")
    model = create_ranker(spec, cfg)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
    return model.eval()


def score_list(model, builder, lists, row: int) -> np.ndarray:
    cand = lists["cand"][row, :K1]
    cols = np.flatnonzero(cand > 0)
    feats = builder.build(lists, np.full(len(cols), row), cols)
    with torch.inference_mode():
        values = model({k: torch.as_tensor(v) for k, v in feats.items()}).float().numpy()
    scores = np.full(K1, -np.inf, dtype=np.float32)
    scores[cols] = values
    return scores


def pessimistic_rank(scores: np.ndarray, pos: int) -> float:
    t = scores[pos]
    return float(1 + np.sum(scores > t) + np.sum(scores == t) - 1)


def describe(items: pd.DataFrame, item: int) -> dict:
    r = items.loc[item]
    first = datetime.fromtimestamp(r.first_ts / 1000, tz=timezone.utc).date().isoformat()
    return {"item": int(item), "asin": r.parent_asin, "title": (r.title or "")[:110],
            "store": r.store or "", "leaf": (r.leaf_cat or "").split(" > ")[-1],
            "segment": r.segment, "first_seen": first}


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(ROOT / "configs/ranker/policy_study.yaml")
    data = ProcessedData.load(str(ROOT / cfg["data_config"]))
    items = data.items.set_index("item")
    full_train = load_candidates(CAND, "ranker_train")
    lists = load_candidates(CAND, "val", k1=500, force=False)
    builder = frozen_feature_builder(data, full_train, ROOT / cfg["reference_ranker_config"])
    groups = pd.read_parquet(PILOT / "validation_groups.parquet").set_index("inst").loc[lists["inst"]]
    item_type = groups.item_type.to_numpy()

    saved = {}
    for arm in ("A", "C"):
        with np.load(MODELS[arm] / "prefix_ranks.npz") as z:
            assert np.array_equal(z["inst"], lists["inst"])
            saved[arm] = z["ranks"][2]
    with np.load(RQ3) as z:
        order = pd.Series(np.arange(len(z["inst"])), index=z["inst"]).loc[lists["inst"]].to_numpy()
        saved["MLP"], saved["DIN"] = z["R2"][0][order], z["R3"][0][order]

    def top(arm, k=10):
        return np.isfinite(saved[arm]) & (saved[arm] <= k)
    retrieved = np.isfinite(saved["C"])
    eligible = {
        "c_rescues_head": (item_type == "warm-head") & top("C") & ~top("A") & retrieved,
        "a_keeps_new": (item_type == "new-in-period") & top("A") & ~top("C") & retrieved,
        "din_rescues_new": (item_type == "new-in-period") & top("DIN") & ~top("MLP") & retrieved,
        "retrieval_miss": lists["target_pos"] < 0,
    }
    rng = np.random.default_rng(0)
    models = {name: load_model(run, builder) for name, run in MODELS.items()}
    inst_frame = data.instances.set_index("inst")
    cases, mismatches = [], 0
    for category, mask in eligible.items():
        pool = np.flatnonzero(mask)
        chosen = np.sort(rng.choice(pool, size=min(PER_CATEGORY, len(pool)), replace=False))
        for row in chosen:
            inst = int(lists["inst"][row])
            meta = inst_frame.loc[inst]
            user, ts = int(meta["user"]), int(meta["ts"])
            seq = slice(data.user_ptr[user], data.user_ptr[user + 1])
            hist = data.seq_item[seq][data.seq_ts[seq] < ts][-5:]
            target = int(meta["item"])
            pos = int(lists["target_pos"][row])
            case = {"category": category, "inst": inst, "user": user,
                    "date": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat(),
                    "eligible_pool": int(len(pool)), "history": [describe(items, i) for i in hist],
                    "target": describe(items, target), "retrieval_rank": pos + 1 if pos >= 0 else None,
                    "full_catalogue_rank": float(lists["target_rank"][row]),
                    "saved_ranks": {k: (None if not np.isfinite(v[row]) else int(v[row])) for k, v in saved.items()},
                    "top3": {}}
            for name, model in models.items():
                scores = score_list(model, builder, lists, row)
                if pos >= 0 and pos < K1:
                    key = "C" if name == "C" else ("A" if name == "A" else "DIN")
                    if pessimistic_rank(scores, pos) != saved[key][row]:
                        mismatches += 1
                best = np.argsort(-scores, kind="stable")[:3]
                case["top3"][name] = [{**describe(items, int(lists["cand"][row, c])), "retrieval_rank": int(c) + 1}
                                      for c in best]
            h_stores = {h["store"] for h in case["history"] if h["store"]}
            h_leaves = {h["leaf"] for h in case["history"]}
            t = case["target"]
            case["flags"] = {"target_store_in_history": t["store"] in h_stores,
                             "target_leaf_in_history": t["leaf"] in h_leaves}
            cases.append(case)
    if mismatches:
        raise ValueError(f"{mismatches} recomputed target ranks differ from saved ranks")
    (output / "cases.json").write_text(json.dumps(cases, indent=2, default=str))
    lines = ["# Micro case studies (validation, seed 0)", "",
             "Prespecified categories, 6 random events each (RNG 0). Recomputed target ranks match saved ranks.",
             "Unclicked candidates are unobserved, not known dislikes.", ""]
    for c in cases:
        r, t = c["saved_ranks"], c["target"]
        lines += [f"## {c['category']} · inst {c['inst']} · {c['date']} (pool {c['eligible_pool']})",
                  f"**Target:** {t['title']} — *{t['store']}*, {t['leaf']}, {t['segment']}, first seen {t['first_seen']}",
                  f"Retrieval rank {c['retrieval_rank']} (catalogue {c['full_catalogue_rank']:.0f}); "
                  f"A {r['A']} · C {r['C']} · MLP {r['MLP']} · DIN {r['DIN']}; flags {c['flags']}",
                  "History (oldest→newest): " + " ← ".join(f"{h['title'][:60]} ({h['store']})" for h in c["history"])]
        for name, top3 in c["top3"].items():
            lines.append(f"- {name} top-3: " + "; ".join(f"[{x['retrieval_rank']}] {x['title'][:55]} ({x['store']}, {x['segment']})" for x in top3))
        lines.append("")
    (output / "cases.md").write_text("\n".join(lines))
    print(f"wrote {len(cases)} cases, rank mismatches 0")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
