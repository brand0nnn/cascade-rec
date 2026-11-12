"""Construction contract for the November hybrid-study matching MLP/DIN fits.

This is distinct from R1's October natural-list checkpoint recipe. Score scalers,
candidate membership, RNG seeding and checkpoint loading stay with their callers.
"""
from __future__ import annotations

from collections.abc import Mapping

from cascade.models.ranker_mlp import RankerFeatureSpec


def matching_recipe(
    cardinalities: Mapping[str, int],
    model: str,
    *,
    extra_numeric: int = 0,
    residual_score: bool = False,
) -> tuple[RankerFeatureSpec, dict]:
    """Return the original feature spec/config, without constructing a model or drawing RNG values."""
    spec = RankerFeatureSpec(
        cardinalities=cardinalities, history_length=50, item_text_dim=384,
        feature_groups=("retrieval", "matching"), extra_numeric=extra_numeric,
    )
    config = {
        "model": model, "dropout": 0.1, "attention_hidden_dims": (64, 32),
        "attention_use_softmax": False, "retrieval_score_input": "enabled",
        "residual_score": residual_score,
    }
    return spec, config
