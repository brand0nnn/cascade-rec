"""Unit checks for the regenerated-pipeline secondary metrics (synthetic arrays only)."""
import numpy as np

from cascade.analysis import secondary_metrics_current as S


def test_mrr_and_hr3():
    ranks = np.array([1, 3, 4, 10, 11, np.inf])
    assert np.allclose(S.mrr_at(ranks, 10), [1, 1 / 3, .25, .1, 0, 0])
    assert S.per_event(ranks)["hr3"].tolist() == [1, 1, 0, 0, 0, 0]


def test_contrast_union_has_no_duplicates():
    pairs = [(a, b) for _, a, b in S.contrasts()]
    assert len(pairs) == len(set(pairs)) == 12


def test_relative_advantage_point_estimate():
    users = np.array([0, 0, 1, 2]); mask = np.ones(4, bool)
    base = {"hr3": np.array([1., 0, 1, 0]), "ndcg10": np.array([.5, .5, .5, .5])}
    cand = {"hr3": np.array([1., 1, 1, 0]), "ndcg10": np.array([.5, .5, .5, .5])}
    out = S.relative_advantage_ci(users, mask, base, cand, "hr3", "ndcg10", n_boot=50)
    assert np.isclose(out["rel_metric"], (2 / 3) / (1.5 / 3) - 1) and np.isclose(out["rel_vs"], 0)
