"""Shared pointwise MLP/DIN training on grouped candidate rows."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from cascade.models.ranker_mlp import MLPRanker
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.models.ranker_din import DINRanker
from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.pipelines.build_candidates import load_candidates
from cascade.eval import metrics as M
from cascade.eval.ranker import candidate_ranks, ranker_comparison
from cascade.utils.config import load_config
from cascade.utils.io import git_hash, make_run_dir, save_run
from cascade.utils.seed import set_seed


FeatureArrays = Mapping[str, np.ndarray]


def create_ranker(spec: RankerFeatureSpec, cfg: Mapping) -> MLPRanker:
    """Construct a named ranker without changing the common training recipe."""
    kind = cfg.get("model", "mlp")
    score_input = cfg.get("retrieval_score_input", "enabled")
    if score_input not in ("enabled", "masked"):
        raise ValueError("retrieval_score_input must be enabled or masked")
    if kind == "mlp":
        model = MLPRanker(spec, dropout=float(cfg.get("dropout", 0.1)))
    elif kind == "din":
        if cfg.get("attention_use_softmax", False):
            raise ValueError("this DIN recipe uses unnormalised attention; softmax must be false")
        model = DINRanker(spec, dropout=float(cfg.get("dropout", 0.1)),
                         attention_hidden_dims=tuple(cfg.get("attention_hidden_dims", (64, 32))))
    else:
        raise ValueError(f"unknown ranker model: {kind}")
    model.set_retrieval_score_enabled(score_input == "enabled")
    return model


@dataclass
class RankerTrainingResult:
    """Best validation checkpoint and its training history."""

    best_epoch: int
    best_val_gauc: float
    best_val_macro_ndcg10: float
    history: list[dict[str, float]]


def _validate_groups(inst: np.ndarray, labels: np.ndarray, split_name: str) -> None:
    """Require one observed positive per supervised candidate list."""
    if len(inst) != len(labels):
        raise ValueError(f"{split_name}: inst and labels have different lengths")
    if len(labels) == 0:
        raise ValueError(f"{split_name}: candidate table is empty")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split_name}: labels must be binary")
    unique, counts = np.unique(inst, return_counts=True)
    positive_counts = np.bincount(
        np.searchsorted(unique, inst), weights=labels, minlength=len(unique)
    )
    if not np.all(positive_counts == 1) or not np.all(counts > 1):
        raise ValueError(
            f"{split_name}: every supervised candidate list must have exactly one positive and "
            "at least one negative; "
            "force-insert targets for train/validation and keep end-to-end miss lists out of fitting"
        )


def _validate_features(features: FeatureArrays, n_rows: int, model: MLPRanker, split_name: str) -> None:
    required = set(model.active_categorical_names) | {"numeric"}
    required.update(getattr(model, "query_feature_names", ()))
    if model.use_matching:
        required.update(("history_item_ids", "history_leaf_ids"))
    if model.text_projection is not None:
        required.add("item_text")
    missing = required - set(features)
    if missing:
        raise ValueError(f"{split_name}: missing feature arrays {sorted(missing)}")
    bad_lengths = {key: len(value) for key, value in features.items() if len(value) != n_rows}
    if bad_lengths:
        raise ValueError(f"{split_name}: feature row counts must equal {n_rows}: {bad_lengths}")


def candidate_rows_from_lists(lists: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten one ``load_candidates`` result while preserving exact target retrieval rank.

    The candidate builder force-appends missed positives for supervised B/Val
    lists. Its per-instance ``target_rank`` records the target's actual rank in
    the full filtered catalogue, and ``score`` stores its actual retriever
    score, so neither feature needs a target-only sentinel.
    """
    required = {"inst", "user", "cand", "score", "label", "target_pos", "target_rank"}
    missing = required - set(lists)
    if missing:
        raise ValueError(f"candidate lists missing required arrays: {sorted(missing)}")
    cand = np.asarray(lists["cand"])
    score = np.asarray(lists["score"])
    label = np.asarray(lists["label"])
    if cand.ndim != 2 or score.shape != cand.shape or label.shape != cand.shape:
        raise ValueError("cand, score, and label must be aligned [instances, candidates] arrays")
    if not np.all(np.asarray(lists["target_pos"]) >= 0):
        raise ValueError("supervised candidate lists must contain a target; load them with force=True")
    valid = cand > 0  # exclude padding for instances with fewer than K eligible items
    rows, cols = np.nonzero(valid)
    item = cand[rows, cols].astype(np.int64)
    target = label[rows, cols].astype(np.int8)
    retrieval_rank = cols.astype(np.float32) + 1.0
    positive = target == 1
    retrieval_rank[positive] = np.asarray(lists["target_rank"], dtype=np.float32)[rows[positive]]
    return {
        "inst": np.asarray(lists["inst"], dtype=np.int64)[rows],
        "user": np.asarray(lists["user"], dtype=np.int64)[rows],
        "item": item,
        "label": target,
        "retrieval_score": score[rows, cols].astype(np.float32),
        "retrieval_rank": retrieval_rank,
        "list_row": rows.astype(np.int64),
    }


def sample_training_rows(
    inst: np.ndarray,
    labels: np.ndarray,
    n_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep each positive and uniformly sample up to N negatives in every list."""
    if n_negatives < 1:
        raise ValueError("n_negatives must be at least one")
    chosen: list[np.ndarray] = []
    order = np.argsort(inst, kind="stable")
    sorted_inst = inst[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_inst[1:] != sorted_inst[:-1]) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:end]
        positive = rows[labels[rows] == 1]
        negatives = rows[labels[rows] == 0]
        if len(positive) != 1:
            raise ValueError(f"instance {inst[rows[0]]} has {len(positive)} positives; expected one")
        n_take = min(n_negatives, len(negatives))
        sampled = rng.choice(negatives, size=n_take, replace=False) if n_take else np.empty(0, dtype=np.int64)
        chosen.extend((positive, sampled))
    result = np.concatenate(chosen).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


def _to_device_batch(features: FeatureArrays, rows: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value[rows], device=device) for key, value in features.items()}


def score_rows(
    model: MLPRanker,
    features: FeatureArrays,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Score candidate rows in order without changing the model's training mode."""
    device = torch.device(device)
    model = model.to(device)
    was_training = model.training
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(next(iter(features.values()))), batch_size):
            idx = np.arange(start, min(start + batch_size, len(next(iter(features.values())))))
            scores.append(model(_to_device_batch(features, idx, device)).float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def grouped_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    inst: np.ndarray,
    users: np.ndarray,
    cutoff: int = 10,
) -> dict[str, float]:
    """Compute per-user macro and per-list micro GAUC/nDCG for single-positive lists."""
    if not (len(scores) == len(labels) == len(inst) == len(users)):
        raise ValueError("scores, labels, inst, and users must have equal lengths")
    list_aucs: list[float] = []
    list_ndcgs: list[float] = []
    list_users: list[int] = []
    order = np.argsort(inst, kind="stable")
    sorted_inst = inst[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_inst[1:] != sorted_inst[:-1]) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:end]
        pos = rows[labels[rows] == 1]
        neg = rows[labels[rows] == 0]
        if len(pos) != 1:
            continue  # lists missing the target are retrieval misses, not conditional ranker cases
        if len(neg) == 0:
            continue
        positive_score = scores[pos[0]]
        auc = M.instance_auc(scores[rows], int(np.flatnonzero(labels[rows] == 1)[0]))
        # Pessimistic ties make candidate-list ranks deterministic, as in the full-ranking harness.
        rank = 1 + int((scores[neg] >= positive_score).sum())
        ndcg = float(M.ndcg_at(np.asarray([rank], dtype=np.float64), cutoff)[0])
        list_aucs.append(auc)
        list_ndcgs.append(ndcg)
        list_users.append(int(users[pos[0]]))

    if not list_users:
        return {"gauc_macro": float("nan"), "gauc_micro": float("nan"),
                f"ndcg@{cutoff}_macro": float("nan"), f"ndcg@{cutoff}_micro": float("nan")}
    aucs = np.asarray(list_aucs)
    ndcgs = np.asarray(list_ndcgs)
    list_users_array = np.asarray(list_users)
    ndcg_summary = M.macro_micro(ndcgs, list_users_array)
    return {
        "gauc_macro": M.gauc(aucs, list_users_array),
        "gauc_micro": float(aucs.mean()),
        f"ndcg@{cutoff}_macro": ndcg_summary["macro"],
        f"ndcg@{cutoff}_micro": ndcg_summary["micro"],
    }


def fit_ranker(
    model: MLPRanker,
    train_features: FeatureArrays,
    train_labels: np.ndarray,
    train_inst: np.ndarray,
    val_features: FeatureArrays,
    val_labels: np.ndarray,
    val_inst: np.ndarray,
    val_users: np.ndarray,
    *,
    n_negatives: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
) -> RankerTrainingResult:
    """Fit with per-list negative sampling and early stop on validation GAUC.

    Validation is never negative-subsampled. Labels 0 mean unobserved candidate,
    not confirmed dislikes, as specified in the project protocol.
    """
    _validate_groups(train_inst, train_labels, "train")
    _validate_groups(val_inst, val_labels, "validation")
    _validate_features(train_features, len(train_labels), model, "train")
    _validate_features(val_features, len(val_labels), model, "validation")
    if len(val_users) != len(val_labels):
        raise ValueError("val_users must contain one user id per candidate row")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    best_gauc = -np.inf
    best_ndcg = float("nan")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        sampled = sample_training_rows(train_inst, train_labels, n_negatives, rng)
        rng.shuffle(sampled)
        total_loss = 0.0
        total_rows = 0
        for start in range(0, len(sampled), batch_size):
            rows = sampled[start : start + batch_size]
            batch = _to_device_batch(train_features, rows, device)
            labels = torch.as_tensor(train_labels[rows], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            total_rows += len(rows)

        val_scores = score_rows(model, val_features, batch_size=batch_size * 2, device=device)
        metrics = grouped_metrics(val_scores, val_labels, val_inst, val_users)
        epoch_row = {"epoch": float(epoch), "train_bce": total_loss / max(1, total_rows), **metrics}
        history.append(epoch_row)
        val_gauc = metrics["gauc_macro"]
        if np.isfinite(val_gauc) and val_gauc > best_gauc:
            best_gauc = val_gauc
            best_ndcg = metrics["ndcg@10_macro"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "epoch": best_epoch,
                            "val_gauc_macro": best_gauc, "val_ndcg10_macro": best_ndcg,
                            "seed": seed}, path)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("validation GAUC was undefined on every epoch; check candidate lists")
    model.load_state_dict(best_state)
    return RankerTrainingResult(best_epoch, float(best_gauc), float(best_ndcg), history)


def _load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned feature and grouping arrays from one candidate-split archive."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"label", "inst", "user"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing grouping arrays {sorted(missing)}")
        labels = archive["label"].astype(np.float32)
        inst = archive["inst"].astype(np.int64)
        users = archive["user"].astype(np.int64)
        features = {
            key: archive[key]
            for key in archive.files
            if key not in {"label", "inst", "user"}
        }
    if not (len(labels) == len(inst) == len(users)):
        raise ValueError(f"{path}: label, inst, user arrays must have equal lengths")
    return features, labels, inst, users


def _cardinalities(train_features: FeatureArrays) -> dict[str, int]:
    """Infer ID-table sizes from training features only; zero remains padding."""
    sources = {
        "item_id": ("item_id", "history_item_ids"),
        "leaf_category": ("leaf_category", "history_leaf_ids"),
        "top_category": ("top_category",),
        "store": ("store",),
        "price_bucket": ("price_bucket",),
        "review_count_bucket": ("review_count_bucket",),
        "rating_bucket": ("rating_bucket",),
        "item_age_bucket": ("item_age_bucket",),
        "history_length_bucket": ("history_length_bucket",),
        "recency_bucket": ("recency_bucket",),
    }

    return {
        name: max(int(np.max(train_features[key])) for key in keys) + 1
        for name, keys in sources.items()
    }


def _sample_list_columns(lists: Mapping[str, np.ndarray], n_negatives: int, rng: np.random.Generator) -> np.ndarray:
    """Return positive + uniformly sampled negative candidate columns per list."""
    cand = lists["cand"]
    labels = lists["label"]
    selected = np.full((len(cand), n_negatives + 1), -1, dtype=np.int16)
    selected[:, 0] = np.asarray(lists["target_pos"], dtype=np.int16)
    for row in range(len(cand)):
        valid_neg = np.flatnonzero((labels[row] == 0) & (cand[row] > 0))
        if not len(valid_neg):
            continue
        take = min(n_negatives, len(valid_neg))
        selected[row, 1 : take + 1] = rng.choice(valid_neg, size=take, replace=False)
    return selected


def policy_training_columns(lists: Mapping[str, np.ndarray], n_negatives: int,
                            seed: int, epoch: int) -> np.ndarray:
    """Common negatives keyed by seed/epoch/instance, invariant to list filtering."""
    if n_negatives < 1 or seed < 0 or epoch < 1:
        raise ValueError("policy sampling requires positive epoch/negative count and nonnegative seed")
    selected = np.full((len(lists["inst"]), n_negatives + 1), -1, dtype=np.int16)
    for row, instance in enumerate(lists["inst"]):
        target = int(lists["target_pos"][row])
        if target < 0:
            raise ValueError("policy training requires a target in every retained list")
        selected[row, 0] = target
        alternatives = np.flatnonzero((lists["label"][row] == 0) & (lists["cand"][row] > 0))
        take = min(n_negatives, len(alternatives))
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, epoch, int(instance)], spawn_key=(0,)))
        if take:
            selected[row, 1:take + 1] = rng.choice(alternatives, take, replace=False)
    return selected


def policy_training_order(instances: np.ndarray, universe: np.ndarray,
                          seed: int, epoch: int) -> np.ndarray:
    """Filter a common full-list permutation, preserving shared relative order."""
    instances, universe = np.asarray(instances), np.asarray(universe)
    if len(np.unique(instances)) != len(instances) or len(np.unique(universe)) != len(universe):
        raise ValueError("policy sampling requires unique instance IDs")
    if not np.isin(instances, universe).all():
        raise ValueError("retained training instances must belong to the common universe")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch], spawn_key=(1,)))
    shuffled = universe[rng.permutation(len(universe))]
    local = {int(instance): row for row, instance in enumerate(instances)}
    return np.array([local[int(instance)] for instance in shuffled if int(instance) in local], dtype=np.int64)


def fit_candidate_lists(
    model: MLPRanker,
    builder: RankerFeatureBuilder,
    train_lists: Mapping[str, np.ndarray],
    val_lists: Mapping[str, np.ndarray],
    *,
    n_negatives: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: Mapping[str, object] | None = None,
    sampling_strategy: str = "historical",
    training_instance_universe: np.ndarray | None = None,
) -> RankerTrainingResult:
    """Train directly from grouped candidate files, expanding only each minibatch."""
    for name, lists in (("train", train_lists), ("validation", val_lists)):
        if not len(lists["inst"]):
            raise ValueError(f"{name} candidate population is empty")
        if np.any(lists["target_pos"] < 0):
            raise ValueError(f"{name} candidate lists must have force-inserted targets")
        if np.any(np.sum(lists["label"], axis=1) != 1):
            raise ValueError(f"{name} candidate lists must have exactly one target")
    if sampling_strategy not in ("historical", "instance_keyed"):
        raise ValueError("unknown training sampling strategy")
    universe = np.asarray(train_lists["inst"] if training_instance_universe is None else training_instance_universe)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    best_gauc, best_ndcg, best_epoch = -np.inf, -np.inf, 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    optimizer_steps = 0
    if resume_checkpoint is not None:
        best_gauc = float(resume_checkpoint["val_gauc_macro"])
        best_ndcg = float(resume_checkpoint["val_ndcg10_macro"])
        best_epoch = int(resume_checkpoint["epoch"])
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    list_batch_size = max(1, batch_size // (n_negatives + 1))
    n_lists = len(train_lists["cand"])
    val_n = len(val_lists["cand"])
    val_retrieved_rows = np.flatnonzero(~np.asarray(
        val_lists.get("forced", np.zeros(val_n, dtype=bool)), dtype=bool
    ))
    val_scores = np.zeros(val_lists["cand"].shape, dtype=np.float32)
    val_users = np.asarray(val_lists["user"], dtype=np.int64)
    val_labels = np.asarray(val_lists["label"], dtype=np.int8)
    val_inst = np.asarray(val_lists["inst"], dtype=np.int64)

    for epoch in range(best_epoch + 1, max_epochs + 1):
        model.train()
        if sampling_strategy == "instance_keyed":
            cols = policy_training_columns(train_lists, n_negatives, seed, epoch)
            order = policy_training_order(train_lists["inst"], universe, seed, epoch)
        else:
            cols = _sample_list_columns(train_lists, n_negatives, rng)
            order = rng.permutation(n_lists)
        total_loss, total_rows = 0.0, 0
        for start in tqdm(range(0, n_lists, list_batch_size), desc=f"Train epoch {epoch}/{max_epochs}", unit="batch", leave=False, mininterval=5):
            batch_lists = order[start : start + list_batch_size]
            batch_cols = cols[batch_lists]
            valid = batch_cols >= 0
            pair_rows = np.repeat(batch_lists, batch_cols.shape[1])[valid.ravel()]
            pair_cols = batch_cols.ravel()[valid.ravel()].astype(np.int64)
            features = builder.build(train_lists, pair_rows, pair_cols, score_only=model.spec.feature_groups == ("retrieval",))
            batch = {key: torch.as_tensor(value, device=device) for key, value in features.items()}
            labels = torch.as_tensor(train_lists["label"][pair_rows, pair_cols], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * len(pair_rows)
            total_rows += len(pair_rows)

        # Validation uses every non-padding candidate from the top-200 list, plus forced targets.
        model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(val_retrieved_rows), list_batch_size), desc="Validate retrieval hits", unit="batch", leave=False, mininterval=5):
                rows = val_retrieved_rows[start : start + list_batch_size]
                candidate_matrix = val_lists["cand"][rows]
                valid = candidate_matrix > 0
                pair_rows = np.repeat(rows, candidate_matrix.shape[1])[valid.ravel()]
                pair_cols = np.tile(np.arange(candidate_matrix.shape[1]), len(rows))[valid.ravel()]
                features = builder.build(val_lists, pair_rows, pair_cols, score_only=model.spec.feature_groups == ("retrieval",))
                batch = {key: torch.as_tensor(value, device=device) for key, value in features.items()}
                values = model(batch).float().cpu().numpy()
                val_scores[pair_rows, pair_cols] = values

        flat_valid = val_lists["cand"] > 0
        # Early stopping evaluates ranking only when retrieval actually returned
        # the logged target. Force-added targets are useful training positives,
        # but must not count as successful validation retrievals.
        retrieved_lists = ~np.asarray(val_lists.get("forced", np.zeros(val_n, dtype=bool)), dtype=bool)
        evaluated = flat_valid & retrieved_lists[:, None]
        metrics = grouped_metrics(
            val_scores[evaluated], val_labels[evaluated],
            np.broadcast_to(val_inst[:, None], flat_valid.shape)[evaluated],
            np.broadcast_to(val_users[:, None], flat_valid.shape)[evaluated],
        )
        epoch_row = {"epoch": float(epoch), "train_bce": total_loss / max(1, total_rows),
                     "optimizer_steps": optimizer_steps, "train_lists": n_lists, **metrics}
        ranks = candidate_ranks(val_scores, val_lists)
        comparison = ranker_comparison(ranks, val_lists)
        epoch_row.update({
            "end_to_end_ndcg@10_macro": comparison["ranker"]["ndcg@10"]["macro"],
            "rescued_count": comparison["outcomes"]["rescued"]["count"],
            "demoted_count": comparison["outcomes"]["demoted"]["count"],
        })
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        if checkpoint_path is not None:
            Path(checkpoint_path).with_name("training_history.json").write_text(json.dumps(history, indent=2))
        val_ndcg = metrics["ndcg@10_macro"]
        if np.isfinite(val_ndcg) and val_ndcg > best_ndcg:
            best_gauc, best_ndcg, best_epoch, stale_epochs = metrics["gauc_macro"], val_ndcg, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "epoch": best_epoch,
                            "val_gauc_macro": best_gauc, "val_ndcg10_macro": best_ndcg,
                            "seed": seed, "feature_groups": model.spec.feature_groups,
                            "model_type": type(model).__name__, "sampling_strategy": sampling_strategy,
                            "optimizer_steps": optimizer_steps,
                            "retrieval_score_input": "enabled" if bool(model._retrieval_score_enabled) else "masked"}, path)
                path.with_name("validation_comparison.json").write_text(json.dumps(comparison, indent=2))
                np.savez_compressed(path.with_name("val_ranks.npz"), inst=val_inst,
                                    user=val_users, ranks=ranks,
                                    retrieval_ranks=np.where(~val_lists["forced"], val_lists["target_pos"] + 1, np.inf))
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("validation macro nDCG@10 was undefined on every epoch; check candidate lists")
    model.load_state_dict(best_state)
    return RankerTrainingResult(best_epoch, float(best_gauc), float(best_ndcg), history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP or DIN ranker on frozen candidate lists.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", help="toy NPZ with period-B candidate rows and aligned features")
    parser.add_argument("--val", help="toy NPZ with validation candidate rows and aligned features")
    parser.add_argument("--candidate-dir", help="full grouped candidate directory from build_candidates")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sample", action="store_true", help="development only: fixed 2% full-data user IDs, one epoch")
    parser.add_argument("--resume", help="unsupported: exact training resumption is not implemented")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.resume is not None:
        parser.error(
            "--resume is unsupported: exact training resumption requires optimizer/RNG state "
            "and selected-artifact carryover, which are not implemented. Start a fresh run "
            "with an unused --run-id."
        )

    cfg = load_config(args.config)
    if args.candidate_dir:
        data = ProcessedData.load(args.data_config)
        train_lists = load_candidates(args.candidate_dir, "ranker_train")
        val_lists = load_candidates(args.candidate_dir, "val", k1=int(cfg.get("k1", 200)), force=True)
        if args.sample:
            sample_users = np.random.default_rng(0).choice(
                np.arange(1, data.n_users + 1), max(1, round(data.n_users * 0.02)), replace=False
            )
            train_mask = np.isin(train_lists["user"], sample_users)
            val_mask = np.isin(val_lists["user"], sample_users)
            train_lists = {k: v[train_mask] for k, v in train_lists.items()}
            val_lists = {k: v[val_mask] for k, v in val_lists.items()}
            cfg = {**cfg, "max_epochs": 1, "development_sample": True}
        if cfg.get("early_stopping", {}).get("metric", "ndcg@10_macro") != "ndcg@10_macro":
            raise ValueError("grouped training selects conditional macro ndcg@10 only")
        builder = RankerFeatureBuilder.create(data, train_lists)
        set_seed(int(cfg.get("seed", 0)))
        spec = RankerFeatureSpec(
            cardinalities=builder.item_cardinalities,
            history_length=int(cfg.get("history_length", 50)),
            item_text_dim=int(cfg.get("item_text_dim", 384)),
            feature_groups=tuple(cfg.get("feature_groups", ("retrieval", "matching", "attributes", "statistics"))),
        )
        model = create_ranker(spec, cfg)
        negatives = cfg.get("negative_sampling", {})
        run_dir = make_run_dir(f"ranker_{cfg.get('model', 'mlp')}", args.run_id, exist_ok=False)
        save_run(run_dir, {"status": "running", "development_sample": args.sample}, cfg)
        result = fit_candidate_lists(
            model, builder, train_lists, val_lists,
            n_negatives=int(negatives.get("negatives_per_list", 50)),
            batch_size=int(cfg.get("batch_size", 2048)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            max_epochs=int(cfg.get("max_epochs", 30)),
            patience=int(cfg.get("early_stopping", {}).get("patience", 3)),
            seed=int(cfg.get("seed", 0)), device=args.device,
            checkpoint_path=run_dir / "model.pt",
        )
        metrics = {
            "model": cfg.get("model", "mlp"),
            "best_epoch": result.best_epoch,
            "development_sample": args.sample,
            "feature_groups": list(spec.feature_groups),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "best_val_gauc_macro": result.best_val_gauc,
            "best_val_ndcg10_macro": result.best_val_macro_ndcg10,
            "history": result.history,
            "validation_comparison": (json.loads((run_dir / "validation_comparison.json").read_text())
                                      if (run_dir / "validation_comparison.json").exists() else None),
            "n_train_lists": int(len(train_lists["inst"])),
            "n_val_lists": int(len(val_lists["inst"])),
            "train_forced_fraction": float(train_lists["forced"].mean()),
            "val_forced_fraction": float(val_lists["forced"].mean()),
            "val_conditional_lists": int((~val_lists["forced"]).sum()),
        }
        cfg = {**cfg, "run": {"candidate_dir": str(args.candidate_dir),
                               "resume_checkpoint": args.resume,
                               "seed": int(cfg.get("seed", 0)), "git_hash": git_hash(),
                               "device": args.device, "feature_spec": {
                                   "cardinalities": dict(spec.cardinalities),
                                   "history_length": spec.history_length,
                                   "item_text_dim": spec.item_text_dim,
                                   "feature_groups": list(spec.feature_groups)},
                               "score_mean": builder.score_mean, "score_std": builder.score_std}}
        save_run(run_dir, metrics, cfg)
        print(json.dumps({"run_dir": str(run_dir), **{k: v for k, v in metrics.items() if k != "history"}}, indent=2))
        return
    if not args.train or not args.val:
        parser.error("provide --candidate-dir for full data, or both --train and --val for toy archives")
    train_features, train_labels, train_inst, _ = _load_npz(args.train)
    val_features, val_labels, val_inst, val_users = _load_npz(args.val)
    text_dim = int(cfg.get("item_text_dim", 384 if "item_text" in train_features else 0))
    if (text_dim > 0) != ("item_text" in train_features):
        raise ValueError("config item_text_dim and train archive item_text presence disagree")
    spec = RankerFeatureSpec(
        cardinalities=_cardinalities(train_features),
        history_length=int(cfg.get("history_length", 50)),
        item_text_dim=text_dim,
    )
    model = create_ranker(spec, cfg)
    training = cfg.get("training", {})
    negatives = cfg.get("negative_sampling", {})
    run_dir = make_run_dir(f"ranker_{cfg.get('model', 'mlp')}", args.run_id, exist_ok=False)
    result = fit_ranker(
        model, train_features, train_labels, train_inst,
        val_features, val_labels, val_inst, val_users,
        n_negatives=int(negatives.get("negatives_per_list", 50)),
        batch_size=int(training.get("batch_size", cfg.get("batch_size", 2048))),
        learning_rate=float(training.get("learning_rate", cfg.get("learning_rate", 1e-3))),
        max_epochs=int(training.get("max_epochs", cfg.get("max_epochs", 30))),
        patience=int(cfg.get("early_stopping", {}).get("patience", 3)),
        seed=int(cfg.get("seed", 0)),
        device=args.device,
        checkpoint_path=run_dir / "model.pt",
    )
    metrics = {
        "best_epoch": result.best_epoch,
        "best_val_gauc_macro": result.best_val_gauc,
        "best_val_ndcg10_macro": result.best_val_macro_ndcg10,
        "history": result.history,
        "n_train_rows": int(len(train_labels)),
        "n_val_rows": int(len(val_labels)),
        "n_train_instances": int(np.unique(train_inst).size),
        "n_val_instances": int(np.unique(val_inst).size),
    }
    cfg = {**cfg, "run": {"train_archive": str(args.train), "val_archive": str(args.val),
                           "seed": int(cfg.get("seed", 0)), "git_hash": git_hash(),
                           "device": args.device,
                           "feature_spec": {"cardinalities": dict(spec.cardinalities),
                                            "history_length": spec.history_length,
                                            "item_text_dim": spec.item_text_dim,
                                   "feature_groups": list(spec.feature_groups)}}}
    save_run(run_dir, metrics, cfg)
    print(json.dumps({"run_dir": str(run_dir), **{k: v for k, v in metrics.items() if k != "history"}}, indent=2))


if __name__ == "__main__":
    main()
