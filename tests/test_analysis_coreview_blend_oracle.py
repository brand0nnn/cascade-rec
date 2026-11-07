"""Unit checks for the BCE/pairwise blend and oracle helpers (synthetic arrays only)."""
import numpy as np

from cascade.analysis import coreview_blend_oracle as B
from cascade.pipelines.local_evidence_export import ranks_and_top10


def _lists(seed=0, n=50, k=12):
    rng = np.random.default_rng(seed)
    cand = np.stack([rng.permutation(np.arange(1, 40))[:k] for _ in range(n)]).astype(np.int32)
    cand[:5, -3:] = 0  # padding
    target = np.where(rng.random(n) < .7, cand[np.arange(n), rng.integers(0, k - 3, n)], 99)
    return rng, cand, target


def test_target_ranks_match_pipeline_without_padding_and_mask_padding():
    rng, cand, target = _lists()
    scores = rng.normal(size=cand.shape).astype(np.float32)
    ours = B.target_ranks(scores, cand, target)
    ref = ranks_and_top10(np.where(cand > 0, scores, -np.inf), cand, target)[0]
    assert np.array_equal(ours, ref)
    assert np.isinf(ours[target == 99]).all()
    scores[:5, -3:] = 1e6  # huge padding scores must never outrank
    assert np.array_equal(B.target_ranks(scores, cand, target), ours)


def test_ties_are_pessimistic():
    cand = np.array([[1, 2, 3]]); target = np.array([2])
    assert B.target_ranks(np.zeros((1, 3)), cand, target)[0] == 3


def test_zscore_and_blend_endpoints():
    rng, cand, target = _lists(1)
    lb = rng.normal(2, 3, size=cand.shape); lp = rng.normal(-1, .2, size=cand.shape)
    z = B.zscore(lb, cand > 0)
    valid = cand > 0
    for i in range(len(cand)):
        assert abs(z[i][valid[i]].mean()) < 1e-12 and abs(z[i][valid[i]].std() - 1) < 1e-12
    new = rng.random(cand.shape) < .3
    grid = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    r = B.blend_ranks(lb, lp, cand, target, new, grid)
    assert np.array_equal(r[0], B.target_ranks(lb, cand, target))
    assert np.array_equal(r[1], B.target_ranks(lp, cand, target))
    # (0, 1): warm candidates scored by z_BCE, new candidates by z_PW
    zb, zp = B.zscore(lb, valid), B.zscore(lp, valid)
    manual = B.target_ranks(np.where(new, zp, zb), cand, target)
    assert np.array_equal(r[2], manual)


def test_constant_list_zscore_is_zero():
    assert np.all(B.zscore(np.ones((1, 4)), np.ones((1, 4), bool)) == 0)


def test_oracles():
    eb = {"ndcg10": np.array([0.5, 0.0, 0.2]), "recall10": np.array([1., 0., 1.])}
    ep = {"ndcg10": np.array([0.3, 0.4, 0.2]), "recall10": np.array([1., 1., 1.])}
    o = B.oracle_events(eb, ep, np.array([True, False, False]))
    assert np.allclose(o["event_oracle"]["ndcg10"], [0.5, 0.4, 0.2])
    assert np.allclose(o["group_oracle"]["ndcg10"], [0.3, 0.0, 0.2])


def test_new_candidate_flag_uses_only_first_review_vs_period_start():
    first = np.array([np.iinfo(np.int64).max, 5, 10, 15])
    cand = np.array([[1, 2, 3, 0]])
    assert B.new_candidate_flag(first, cand, 10).tolist() == [[False, True, True, False]]
