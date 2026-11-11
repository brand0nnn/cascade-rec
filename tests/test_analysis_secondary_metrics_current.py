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
    # Some paired user resamples select only the zero-baseline user. Keep their
    # undefined ratios rather than silently conditioning the interval on hits.
    with np.errstate(divide="raise", invalid="raise"):
        out = S.relative_advantage_ci(users, mask, base, cand, "hr3", "ndcg10", n_boot=50)
    assert np.isclose(out["rel_metric"], (2 / 3) / (1.5 / 3) - 1) and np.isclose(out["rel_vs"], 0)
    assert np.isnan(out["ci_low"]) and np.isnan(out["ci_high"])


def test_relative_advantage_interval_matches_joint_user_resampling():
    users = np.array([0, 0, 1, 2])
    mask = np.ones(4, bool)
    base = {"hr3": np.array([.2, .4, .1, .5]), "ndcg10": np.array([.3, .5, .2, .6])}
    cand = {"hr3": np.array([.4, .4, .3, .6]), "ndcg10": np.array([.4, .5, .3, .7])}
    out = S.relative_advantage_ci(users, mask, base, cand, "hr3", "ndcg10", n_boot=50)
    # Aggregate users first, then draw the SAME users for both metrics/systems.
    per_user = np.array([[.3, .4, .4, .45], [.1, .3, .2, .3], [.5, .6, .6, .7]])
    draws = np.random.default_rng(0).integers(0, 3, size=(50, 3))
    means = per_user[draws].mean(1)
    ratios = (means[:, 1] / means[:, 0] - 1) - (means[:, 3] / means[:, 2] - 1)
    np.testing.assert_allclose([out["ci_low"], out["ci_high"]], np.percentile(ratios, [2.5, 97.5]))


def test_zero_baseline_keeps_relative_gain_undefined():
    users = np.array([0, 1])
    values = {"hr3": np.zeros(2), "ndcg10": np.full(2, .5)}
    with np.errstate(divide="raise", invalid="raise"):
        out = S.relative_advantage_ci(users, np.ones(2, bool), values, values, "hr3", "ndcg10", n_boot=50)
    assert np.isnan(out["rel_metric"]) and np.isnan(out["diff"])
    assert np.isnan(out["ci_low"]) and np.isnan(out["ci_high"])
