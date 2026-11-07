"""Small hand-computed checks for the 2026-11-05 Val-only analyses (secondary metrics, U-shape by popularity)."""
import numpy as np

from cascade.analysis.secondary_metrics_archived import mrr_at, per_instance
from cascade.analysis.ushape_popularity import K1, decile_table


def test_mrr_and_hr3():
    ranks = np.array([1, 3, 4, 10, 11, np.inf])
    assert np.allclose(mrr_at(ranks, 10), [1, 1 / 3, 1 / 4, 1 / 10, 0, 0])
    assert np.array_equal(per_instance(ranks)["hr@3"], [1, 1, 0, 0, 0, 0])


def test_decile_table_centres_within_list_and_shares():
    logits = np.tile(np.arange(K1, dtype=np.float32), (2, 1))
    logits[1] += 100.0  # a list-level offset must vanish after centring
    groups = np.where(np.arange(K1) % 2 == 0, "head", "tail")[None, :].repeat(2, axis=0)
    valid = np.ones_like(logits, dtype=bool)
    ranks = np.broadcast_to(np.arange(1, K1 + 1), logits.shape)
    t = decile_table(logits, groups, valid, ranks)
    first = t[(t["decile"] == 1) & (t["group"] == "all")].iloc[0]
    assert np.isclose(first["mean_centred_logit"], np.arange(20).mean() - np.arange(K1).mean())
    assert np.isclose(t[(t["decile"] == 1) & (t["group"] == "head")]["share"].iloc[0], 0.5)
    assert t[(t["group"] == "new")]["n_candidates"].sum() == 0
