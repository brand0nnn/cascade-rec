import numpy as np
import pytest

from cascade.eval.segments import segment_masks
from cascade.index.faiss_index import build_index, recall_vs_exact, search, set_search_params
from cascade.pipelines.index_sweep import compare_lists, effective_ef_search, overlap_fraction, summarise, target_positions


def test_overlap_fraction_hand_example():
    exact = np.array([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 10, 11]])
    approx = np.array([[4, 3, 9, 12], [6, 5, 7, 0], [0, 0, 0, 0]])
    assert np.allclose(overlap_fraction(approx, exact), [0.5, 1.0, 0.0])  # padding 0 never counts as a match


def test_overlap_fraction_matches_recall_vs_exact():
    rng = np.random.default_rng(0)
    exact = np.stack([rng.choice(np.arange(1, 60), 20, replace=False) for _ in range(30)])
    approx = np.stack([rng.choice(np.arange(1, 60), 20, replace=False) for _ in range(30)])
    approx[:3, -5:] = 0
    assert np.isclose(overlap_fraction(approx, exact).mean(), recall_vs_exact(approx, exact))


def test_target_positions():
    cand = np.array([[3, 1, 2], [4, 5, 6], [7, 0, 0]])
    assert target_positions(cand, np.array([1, 9, 7])).tolist() == [1, -1, 0]


def test_summarise_macro_micro_and_segments():
    # user 1: two instances (hit at col 0, miss); user 2: one hit at col 4 (rank 5)
    target_pos = np.array([0, -1, 4])
    users = np.array([1, 1, 2])
    masks = {"a": np.array([True, True, False]), "b": np.array([False, False, True])}
    out = summarise(target_pos, users, [1, 5], masks, rve=np.array([1.0, 0.5, 0.0]), short=np.array([False, True, False]))
    assert out["recall@1"] == {"macro": pytest.approx(0.25), "micro": pytest.approx(1 / 3)}
    assert out["recall@5"] == {"macro": pytest.approx(0.75), "micro": pytest.approx(2 / 3)}
    assert out["recall_vs_exact"] == pytest.approx(0.5) and out["short_of_k1"] == 1
    assert out["segments"]["a"]["recall@1"]["macro"] == pytest.approx(0.5) and out["segments"]["a"]["n_users"] == 1
    assert out["segments"]["b"]["recall@1"]["micro"] == 0 and out["segments"]["b"]["recall@5"]["micro"] == 1


def test_segment_masks():
    m = segment_masks(np.array(["cold", "tail", "head", "tail"]), np.array([True, False, False, True]), np.array([1, 3, 10, 11]))
    assert m["item_cold"].tolist() == [True, False, False, False]
    assert m["item_tail"].tolist() == [False, True, False, True]
    assert m["item_new_in_period"].tolist() == [True, False, False, True]
    assert m["item_not_new"].tolist() == [False, True, True, False]
    assert [m[f"hist_{h}"].nonzero()[0].tolist() for h in ("1-2", "3-5", "6-10", "11+")] == [[0], [1], [2], [3]]


def test_compare_lists():
    ours = np.array([[1, 2, 3], [4, 5, 6]])
    ref = np.array([[1, 2, 3], [4, 6, 9]])
    out = compare_lists(ours, ref, np.array([0, -1]), np.array([0, 2]))
    assert out["rows_same_order"] == 0.5 and out["rows_same_set"] == 0.5
    assert out["mean_overlap"] == pytest.approx((1 + 2 / 3) / 2)
    assert out["target_found_ref_only"] == 1 and out["target_found_both"] == 1


def test_effective_ef_search():
    assert effective_ef_search(200, 450, "at_least_requested") == 450
    assert effective_ef_search(800, 450, "at_least_requested") == 800
    with pytest.raises(ValueError):
        effective_ef_search(200, 450, "as_configured")


def test_faiss_hnsw_does_not_raise_ef_to_k():
    """Documents FAISS behaviour behind ef_search_policy: efSearch < k is NOT silently raised to k.

    Bypasses ``search`` (which asserts efSearch >= k) to call FAISS directly.
    """
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((50, 16)).astype(np.float32)
    x = centers[rng.integers(0, 50, 5001)] + 0.7 * rng.standard_normal((5001, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = x[1:41] + 0.1 * rng.standard_normal((40, 16)).astype(np.float32)
    h = build_index(x, "hnsw", hnsw_m=4, ef_construction=16)
    res = {}
    for ef in (20, 100):
        set_search_params(h, ef_search=ef)
        res[ef] = h.search(np.ascontiguousarray(q), 100)[1]
    assert not np.array_equal(res[20], res[100])
    _, e = search(build_index(x, "exact"), q, 100)
    assert recall_vs_exact(res[20] + 1, e) < recall_vs_exact(res[100] + 1, e)
