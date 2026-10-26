"""Feature-based MLP ranker for retrieve-then-rank experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

FEATURE_GROUPS = ("retrieval", "matching", "attributes", "statistics")


@dataclass(frozen=True)
class RankerFeatureSpec:
    """Vocabulary sizes and input dimensions for a ranker feature batch.

    Categorical IDs reserve zero for padding and a nonzero ID for OOV. ``item_text_dim`` is zero
    until frozen text embeddings are available.
    """

    cardinalities: Mapping[str, int]
    history_length: int = 50
    item_text_dim: int = 384
    feature_groups: tuple[str, ...] = FEATURE_GROUPS

    def __post_init__(self) -> None:
        if "retrieval" not in self.feature_groups or set(self.feature_groups) - set(FEATURE_GROUPS):
            raise ValueError("feature_groups must include retrieval and use known group names")
        if len(set(self.feature_groups)) != len(self.feature_groups):
            raise ValueError("feature_groups must not repeat")
        required = {
            "item_id", "leaf_category", "top_category", "store",
            "price_bucket", "review_count_bucket", "rating_bucket",
            "item_age_bucket", "history_length_bucket", "recency_bucket",
        }
        missing = required - set(self.cardinalities)
        if missing:
            raise ValueError(f"missing categorical cardinalities: {sorted(missing)}")
        invalid = {key: size for key, size in self.cardinalities.items() if size < 2}
        if invalid:
            raise ValueError(f"each categorical cardinality must include padding/OOV: {invalid}")
        if self.history_length < 1:
            raise ValueError("history_length must be positive")
        if self.item_text_dim < 0:
            raise ValueError("item_text_dim cannot be negative")


class MLPRanker(nn.Module):
    """Pointwise MLP that scores one user-candidate pair per input row.

    Expected batch keys are the ten categorical names in ``RankerFeatureSpec``,
    ``history_item_ids`` and ``history_leaf_ids`` with shape [batch, history],
    ``numeric`` with shape [batch, 4], and (when enabled) ``item_text`` with
    shape [batch, item_text_dim]. Numeric order is standardized retrieval
    score, category match, store-in-history, and text cosine. Candidate rank is
    deliberately excluded: the logged target's full-catalogue rank is known only
    for a force-inserted positive and would reveal its label.
    """

    categorical_names = (
        "item_id", "leaf_category", "top_category", "store", "price_bucket",
        "review_count_bucket", "rating_bucket", "item_age_bucket",
        "history_length_bucket", "recency_bucket",
    )
    embedding_dims = {
        "item_id": 32,
        "leaf_category": 16,
        "top_category": 16,
        "store": 16,
        "price_bucket": 8,
        "review_count_bucket": 8,
        "rating_bucket": 8,
        "item_age_bucket": 8,
        "history_length_bucket": 8,
        "recency_bucket": 8,
    }

    def __init__(self, spec: RankerFeatureSpec, dropout: float = 0.1) -> None:
        super().__init__()
        self.spec = spec
        groups = set(spec.feature_groups)
        self.use_matching = "matching" in groups
        active = set()
        if "attributes" in groups:
            active.update(("item_id", "leaf_category", "top_category", "store", "price_bucket"))
        if "statistics" in groups:
            active.update(("review_count_bucket", "rating_bucket", "item_age_bucket"))
        if self.use_matching:
            active.update(("history_length_bucket", "recency_bucket"))
        self.active_categorical_names = tuple(n for n in self.categorical_names if n in active)
        tables = active | ({"item_id", "leaf_category"} if self.use_matching else set())
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(spec.cardinalities[name], self.embedding_dims[name], padding_idx=0)
            for name in self.categorical_names if name in tables
        })
        self.text_projection = (
            nn.Linear(spec.item_text_dim, 32) if spec.item_text_dim and self.use_matching else None
        )
        self.numeric_dim = 4
        self.numeric_indices = (0, 1, 2, 3) if self.use_matching else (0,)
        input_dim = (
            sum(self.embedding_dims[n] for n in self.active_categorical_names)
            + (self.embedding_dims["item_id"] + self.embedding_dims["leaf_category"] if self.use_matching else 0)
            + len(self.numeric_indices)
        )
        if self.text_projection is not None:
            input_dim += 32
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.embeddings[name](batch[name].long()) for name in self.active_categorical_names]

        # Histories share the candidate item/category tables; padding ID 0 is excluded.
        history_tables = (
            ("history_item_ids", self.embeddings["item_id"]),
            ("history_leaf_ids", self.embeddings["leaf_category"]),
        ) if self.use_matching else ()
        for key, embedding in history_tables:
            ids = batch[key].long()
            vectors = embedding(ids)
            mask = (ids != 0).unsqueeze(-1)
            pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            parts.append(pooled)

        numeric = batch["numeric"].float()
        if numeric.ndim != 2 or numeric.shape[1] != self.numeric_dim:
            raise ValueError(
                f"numeric must have shape [batch, {self.numeric_dim}] in the documented feature order"
            )
        parts.append(numeric[:, self.numeric_indices])
        if self.text_projection is not None:
            text = batch["item_text"].float()
            if text.ndim != 2 or text.shape[1] != self.spec.item_text_dim:
                raise ValueError(
                    f"item_text must have shape [batch, {self.spec.item_text_dim}]"
                )
            parts.append(self.text_projection(text))
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)
