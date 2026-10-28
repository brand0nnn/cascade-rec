"""Build an explanation-only metadata/review graph without loading Test predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.analysis.explanation_graph import (
    PARSER_VERSION, assess_device_audit, make_audit, metadata_ego_html, parse_devices, sha256,
)
from cascade.data.split import compute_t_ranker_train
from cascade.utils.config import REPO_ROOT, load_config


def build_base(config_path: str, output: str) -> dict:
    cfg = load_config(config_path)
    data_cfg = load_config(cfg["data_config"])
    source = REPO_ROOT / data_cfg["processed_dir"]
    items_path, reviews_path = source / "items.parquet", source / "interactions.parquet"
    if not items_path.is_file() or not reviews_path.is_file():
        raise FileNotFoundError("Graph base requires original full items.parquet and interactions.parquet")
    out = Path(output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    items = pd.read_parquet(items_path).sort_values("item").reset_index(drop=True)
    reviews = pd.read_parquet(reviews_path, columns=["user", "item", "timestamp"])
    t1 = int(data_cfg["split"]["t1"])
    ta = compute_t_ranker_train(reviews["timestamp"].to_numpy(), t1, data_cfg["split"]["ranker_train_fraction"])
    parsed = [parse_devices(title) for title in items["title"]]
    products = items[["item", "parent_asin", "title", "leaf_cat", "store", "first_ts"]].copy()
    products["node_type"] = "product"
    products["parser_status"] = [p["status"] for p in parsed]
    products.to_parquet(out / "products.parquet", index=False)
    prior = reviews.loc[reviews["timestamp"] < t1].copy()
    prior["evidence_row"] = prior.index.to_numpy(dtype=np.int64)
    prior["relation"] = "reviewed"
    prior.to_parquet(out / "review_edges.parquet", index=False)
    reviewers = pd.DataFrame({"user": np.sort(prior["user"].unique()), "node_type": "reviewer"})
    reviewers.to_parquet(out / "reviewers.parquet", index=False)
    edges = []
    for item, p in zip(items.itertuples(index=False), parsed):
        for field, typ in (("leaf_cat", "category"), ("store", "store")):
            value = getattr(item, field)
            if isinstance(value, str) and value.strip():
                edges.append((item.item, typ, value, "has_" + typ, field, None, None))
        for m in p["mentions"]:
            edges.append((item.item, "device", m["device"], "excludes_device_mention" if m["excluded"] else "mentions_device", "title", m["start"], m["end"]))
    metadata = pd.DataFrame(edges, columns=["item", "target_type", "target_key", "relation", "source_field", "span_start", "span_end"])
    metadata["item"] = metadata["item"].astype("int32")
    metadata["span_start"] = metadata["span_start"].astype("Int32")
    metadata["span_end"] = metadata["span_end"].astype("Int32")
    metadata.to_parquet(out / "metadata_edges.parquet", index=False)
    metadata[["target_type", "target_key"]].drop_duplicates().sort_values(["target_type", "target_key"]).to_parquet(out / "entities.parquet", index=False)
    audit = make_audit(items, parsed, t1, int(cfg["seed"]))
    for subset in ("development", "locked"):
        audit[audit["audit_split"] == subset].to_csv(out / f"device_audit_{subset}.csv", index=False)
    gate = assess_device_audit(audit, float(cfg["device_minimum_precision"]))
    examples = []
    for status in ("parsed", "negated", "universal"):
        matches = products.index[products["parser_status"] == status]
        if len(matches):
            row = int(matches[0])
            name = f"metadata_example_{status}.html"
            (out / name).write_text(metadata_ego_html(items.iloc[row], parsed[row]))
            examples.append({"item": int(items.iloc[row]["item"]), "path": name,
                             "selection": "smallest item ID in parser status; metadata illustration only"})
    prerequisites = {"instances": source / "instances.parquet", "sequences": source / "sequences.npz",
                     "frozen_text_vectors": source / "item_text_emb.npy",
                     "natural_validation_candidates": source / "candidates/two_tower_text_seed0/val.npz"}
    missing = [key for key, path in prerequisites.items() if not path.is_file()]
    counts = {"products": len(products), "reviewers_before_t1": len(reviewers), "reviews_before_t1": len(prior),
              "reviews_before_retriever_cutoff": int((prior["timestamp"] < ta).sum()),
              "metadata_edges": len(metadata), "parser_status": products["parser_status"].value_counts().to_dict(),
              "audit_development": int((audit["audit_split"] == "development").sum()), "audit_locked": int((audit["audit_split"] == "locked").sum())}
    manifest = {"status": "base_complete_overlays_deferred", "counts": counts,
                "snapshots": {"retriever_train": {"strict_before_ms": ta}, "all_train": {"strict_before_ms": t1}},
                "review_snapshot_storage": "One pre-T1 edge table; filter timestamp strictly below the selected cutoff.",
                "parser_version": PARSER_VERSION, "device_claim_gate": gate, "seed": cfg["seed"],
                "sources": {"items": {"path": str(items_path), "sha256": sha256(items_path)},
                            "interactions": {"path": str(reviews_path), "sha256": sha256(reviews_path)}},
                "analysis_sha256": sha256(Path(__file__)),
                "parser_source_sha256": sha256(REPO_ROOT / "src/cascade/analysis/explanation_graph.py"),
                "config_sha256": sha256(REPO_ROOT / config_path),
                "candidate_overlay": {"status": "deferred", "missing_inputs": missing,
                                      "reason": "Base builder does not infer model outputs; original aligned candidates, histories, vectors and checkpoint provenance are required."},
                "examples": examples,
                "limits": ["Reviews are observed reviews, not purchases or known likes/dislikes.",
                           "Metadata is a crawl-time snapshot, not proof of historical compatibility.",
                           "Device mentions remain unaudited title claims; quantitative device profiles are gated.",
                           "No Test predictions, embeddings, candidate lists or trained models were loaded.",
                           "Evidence paths describe relations, not causal model explanations."]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Explanation graph base", "", "Status: base complete; model/candidate overlays deferred.", "",
             "## Observed counts", ""] + [f"- {key}: {value}" for key, value in counts.items()]
    lines += ["", "The title parser is not manually audited. Parser counts measure extraction coverage, not accuracy or compatibility.",
              f"The {counts['audit_development']} development and {counts['audit_locked']} locked audit titles have blank human labels; no audited precision is claimed.",
              "", "## Available evidence", "", "- Product/category/store/device-title relations and source spans.",
              "- Pseudonymous reviewer-product edges strictly before T1, reusable with the earlier retriever-training cutoff.",
              "- Small deterministic metadata-only HTML views; they do not show recommendation successes or failures.",
              "", "## Missing model overlays", "", "Missing inputs: " + (", ".join(missing) or "none detected by path check; identity/reproduction still required"),
              "", "Candidate-pool versus top-ten profiles, target funnels and causal explanations have not been computed by this base build.",
              "", "## Limits", ""] + ["- " + v for v in manifest["limits"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/explanation_graph.yaml")
    parser.add_argument("--output", required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    result = build_base(args.config, args.output)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "device_claim_gate": result["device_claim_gate"], "candidate_overlay": result["candidate_overlay"]}, indent=2))


if __name__ == "__main__":
    main()
