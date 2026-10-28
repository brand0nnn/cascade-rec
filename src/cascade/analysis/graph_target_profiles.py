"""Target/history graph profiles joined to original saved validation ranks.

These relationships describe target evidence, not candidate-slot amplification,
model attributions, purchases, or causal explanations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.analysis.explanation_graph import sha256
from cascade.data.instances import build_instances, build_sequences
from cascade.data.split import assign_periods, compute_t_ranker_train, item_segments
from cascade.eval.saved_results import validate_full_run, validate_ranks


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    except AssertionError as error:
        raise ValueError(f"{label} differs from the verified original structure") from error


def validate_run_identity(config: dict, metrics: dict, seed: int) -> None:
    """Historical metrics without a seed are bound to the accompanying config."""
    for value in (config.get("seed"), config.get("run", {}).get("seed"), metrics.get("seed", seed)):
        if type(value) is not int or value != seed:
            raise ValueError(f"saved config/run/metrics seed differs from advertised seed {seed}")
    validate_full_run(metrics)
    if not isinstance(metrics.get("best_epoch"), int) or metrics["best_epoch"] < 0 or "validation_comparison" not in metrics:
        raise ValueError("saved run lacks a completed selected-checkpoint evaluation")


def validate_rank_archive(archive: dict, length: int) -> None:
    for key in ("inst", "user", "ranks", "retrieval_ranks"):
        if key not in archive or np.asarray(archive[key]).shape != (length,):
            raise ValueError(f"saved {key} must be an aligned one-dimensional array")
    validate_ranks(archive["ranks"])
    validate_ranks(archive["retrieval_ranks"])


def verify_structure(data, data_config: dict, graph: Path, graph_manifest: dict) -> tuple[pd.DataFrame, dict]:
    """Bind restored history/targets/metadata to the original graph source tables.

    Validate bytes of the original mappings and reconstruct derived structure;
    matching Val IDs alone cannot detect a changed history or target item.
    """
    manifest_path = data.dir / "structure_manifest.json"
    record = json.loads(manifest_path.read_text())
    if Path(record["destination"]).resolve() != data.dir.resolve():
        raise ValueError("structure manifest destination differs from loaded data")
    source = Path(record["source"])
    hashes = record["source_hashes"]
    for filename in ("items.parquet", "users.parquet", "interactions.parquet"):
        if sha256(source / filename) != hashes[filename]:
            raise ValueError(f"original source identity changed: {filename}")
    for name, filename in (("items", "items.parquet"), ("interactions", "interactions.parquet")):
        if graph_manifest["sources"][name]["sha256"] != hashes[filename]:
            raise ValueError("graph source hashes differ from the structural manifest")
    for filename in ("items.parquet", "users.parquet"):
        if sha256(data.dir / filename) != hashes[filename]:
            raise ValueError(f"restored mapping or metadata changed: {filename}")
    original = pd.read_parquet(source / "interactions.parquet")
    originals_items = pd.read_parquet(source / "items.parquet")
    originals_users = pd.read_parquet(source / "users.parquet")
    n_items, n_users = len(originals_items), len(originals_users)
    if data.n_items != n_items or data.n_users != n_users:
        raise ValueError("restored user/item counts differ from original mappings")
    if not np.array_equal(originals_items["item"], np.arange(1, n_items + 1)) or not np.array_equal(originals_users["user"], np.arange(1, n_users + 1)):
        raise ValueError("original user/item mappings are not contiguous in source order")
    expected = original.copy()
    split = data_config["split"]
    ta = compute_t_ranker_train(original["timestamp"].to_numpy(), split["t1"], split["ranker_train_fraction"])
    for key, value in {"t_ranker_train": ta, "t1": split["t1"], "t2": split["t2"], "ranker_train_fraction": split["ranker_train_fraction"]}.items():
        if data.split[key] != value:
            raise ValueError(f"restored split disagrees with original data/config: {key}")
    if graph_manifest["snapshots"]["retriever_train"]["strict_before_ms"] != ta or graph_manifest["snapshots"]["all_train"]["strict_before_ms"] != split["t1"]:
        raise ValueError("graph snapshot cutoffs disagree with verified source/config")
    expected["period"] = assign_periods(original["timestamp"].to_numpy(), ta, split["t1"], split["t2"])
    _same_frame(pd.read_parquet(data.dir / "interactions.parquet"), expected, "restored interactions")
    pointer, seq_item, seq_ts = build_sequences(expected, n_users)
    if any(not np.array_equal(a, b) for a, b in ((data.user_ptr, pointer), (data.seq_item, seq_item), (data.seq_ts, seq_ts))):
        raise ValueError("restored sequences/history changed despite matching instance IDs")
    instances, _ = build_instances(expected, pointer)
    _same_frame(data.instances, instances, "restored instances/targets/history lengths")
    segments = item_segments(expected, n_items, split["t2"])
    _same_frame(pd.read_parquet(data.dir / "item_segments.parquet"), segments, "restored item segments")
    _same_frame(data.items[originals_items.columns], originals_items, "loaded item metadata")
    _same_frame(data.items[segments.columns], segments, "loaded item segments")
    columns = ["item", "parent_asin", "title", "leaf_cat", "store", "first_ts"]
    _same_frame(pd.read_parquet(graph / "products.parquet", columns=columns), originals_items[columns], "graph product metadata")
    reviews = pd.read_parquet(graph / "review_edges.parquet", columns=["user", "item", "timestamp"])
    prior = original.loc[original["timestamp"] < split["t1"], ["user", "item", "timestamp"]]
    order = ["user", "timestamp", "item"]
    _same_frame(reviews.sort_values(order, kind="stable"), prior.sort_values(order, kind="stable"), "graph pre-T1 review edges")
    filenames = ("items.parquet", "users.parquet", "interactions.parquet", "instances.parquet", "sequences.npz", "item_segments.parquet", "split.json", "structure_manifest.json")
    provenance = {"structure_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
                  "original_source": {"directory": str(source), "sha256": hashes},
                  "processed_inputs": [{"path": str(data.dir / name), "sha256": sha256(data.dir / name)} for name in filenames],
                  "validation": "source hashes and mapping bytes verified; split, interactions, sequences, instances, metadata and segments reconstructed/compared; graph pre-T1 review content exactly matched"}
    return reviews, provenance


class ReviewIndex:
    """Sparse item-to-reviewer index; no item-pair graph is materialised."""

    def __init__(self, reviews: pd.DataFrame):
        required = {"user", "item", "timestamp"}
        if not required <= set(reviews):
            raise ValueError("review index requires user/item/timestamp")
        if reviews.duplicated(["user", "item"]).any():
            raise ValueError("review source must retain only the earliest user-item interaction")
        self.times = {int(item): dict(zip(group["user"].astype(int), group["timestamp"].astype(int)))
                      for item, group in reviews.groupby("item", sort=False)}
        self.snapshots: dict[int, dict[int, set[int]]] = {}

    def snapshot(self, cutoff: int) -> dict[int, set[int]]:
        """Every edge is strictly before cutoff, including both witness legs."""
        if cutoff not in self.snapshots:
            self.snapshots[cutoff] = {item: {u for u, ts in users.items() if ts < cutoff}
                                      for item, users in self.times.items()}
        return self.snapshots[cutoff]

    def target_paths(self, history: np.ndarray, target: int, user: int, cutoff: int) -> dict:
        """Count distinct other reviewers and paths; keep one deterministic witness."""
        snapshot = self.snapshot(cutoff)
        target_users = snapshot.get(target, set()) - {user}
        witnesses: set[int] = set()
        pair_count = 0
        first = None
        for history_item in sorted(set(int(i) for i in history if i > 0) - {target}):
            common = target_users & snapshot.get(history_item, set())
            if common:
                witnesses.update(common)
                pair_count += len(common)
                witness = min(common)
                path = (witness, history_item)
                if first is None or path < first:
                    first = path
        result = {"other_reviewers": len(witnesses), "witness_paths": pair_count,
                  "witness_user": 0, "witness_history_item": 0,
                  "witness_history_ts": 0, "witness_target_ts": 0}
        if first:
            witness, history_item = first
            result.update(witness_user=witness, witness_history_item=history_item,
                          witness_history_ts=self.times[history_item][witness],
                          witness_target_ts=self.times[target][witness])
        return result


def align_validation(instances: pd.DataFrame, archive: dict, t1: int, t2: int) -> None:
    """Require the complete restored Val population, IDs/order/users and period."""
    if not np.array_equal(instances["inst"].to_numpy(), archive["inst"]):
        raise ValueError("restored validation instance IDs/order differ from saved ranks")
    if not np.array_equal(instances["user"].to_numpy(), archive["user"]):
        raise ValueError("restored validation users differ from saved ranks")
    if not ((instances["period"] == 2) & (instances["ts"] >= t1) & (instances["ts"] < t2)).all():
        raise ValueError("target-profile analysis is validation-only")


def target_profiles(data, instances: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Known metadata matches and co-review paths from the last 50 prior items."""
    items = data.items.set_index("item")
    target = instances["item"].to_numpy()
    users = instances["user"].to_numpy()
    histories = data.histories(users, instances["hist_len"].to_numpy(), data.max_hist_len)
    n = data.n_items
    leaf = np.full(n + 1, "", dtype=object)
    store = np.full(n + 1, "", dtype=object)
    leaf[items.index] = items["leaf_cat"].fillna("").astype(str)
    store[items.index] = items["store"].fillna("").astype(str)
    last_leaf = leaf[histories[:, -1]]
    target_leaf, target_store = leaf[target], store[target]
    leaf_known = (target_leaf != "") & (last_leaf != "")
    leaf_match = np.where(leaf_known, np.where(target_leaf == last_leaf, "true", "false"), "unknown")
    # Chunk store comparisons to avoid an extra full object history matrix.
    store_match = np.full(len(target), "unknown", dtype=object)
    for start in range(0, len(target), 4096):
        end = min(start + 4096, len(target))
        history_store = store[histories[start:end]]
        known = (target_store[start:end] != "") & (history_store != "").any(axis=1)
        matches = (history_store == target_store[start:end, None]).any(axis=1)
        store_match[start:end] = np.where(known, np.where(matches, "true", "false"), "unknown")
    first_ts = items.loc[target, "first_ts"].to_numpy()
    canonical = items.loc[target, "segment"].astype(str).to_numpy()
    new = first_ts >= data.split["t1"]
    result = pd.DataFrame({"inst": instances["inst"].to_numpy(), "user": users, "target": target,
                           "canonical_segment": canonical,
                           "item_type": np.where(new, "new-in-period", np.where(canonical == "head", "warm-head", "warm-tail")),
                           "seen_retriever_train": np.where(first_ts < data.split["t_ranker_train"], "true", "false"),
                           "seen_all_train": np.where(first_ts < data.split["t1"], "true", "false"),
                           "same_leaf_last": leaf_match, "same_store_history": store_match})
    index = ReviewIndex(reviews)
    for name, cutoff in (("retriever_train", int(data.split["t_ranker_train"])), ("all_train", int(data.split["t1"]))):
        paths = []
        for row, (history, item, user) in enumerate(zip(histories, target, users)):
            paths.append(index.target_paths(history, int(item), int(user), cutoff))
            if row and row % 50000 == 0:
                print(f"{name}: {row}/{len(target)} target/history queries", flush=True)
        evidence = pd.DataFrame(paths)
        result["co_review_" + name] = np.where(evidence["other_reviewers"] > 0, "true", "false")
        for column in evidence:
            result[name + "_" + column] = evidence[column].to_numpy()
    return result
