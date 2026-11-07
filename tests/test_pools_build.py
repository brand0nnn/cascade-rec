"""Step A0 pool rules: fixed interleave/RRF fusion and §4.6 filtering on toy data."""
import numpy as np
import pytest

from cascade.analysis.pools_build import filter_violations, interleave_fuse, interleave_reference, pool_ranks


def test_interleave_rrf_hand_example():
    rec = [1, 2, 3, 4]
    tt = [3, 5, 1, 6]
    # slots: r1=1, t1=3, r2=2, t2=5 -> four unique items; r3=3 and t3=1 would be duplicates
    pool, score, rr, tr = interleave_reference(rec, tt, k=4)
    assert pool == [1, 3, 2, 5]
    assert rr == [1, 3, 2, 0] and tr == [3, 1, 0, 2]
    assert score == pytest.approx([1 / 61 + 1 / 63, 1 / 63 + 1 / 61, 1 / 62, 1 / 62])
    f = interleave_fuse(np.array([rec]), np.array([tt]), k=4)
    assert f['cand'].tolist() == [pool]
    assert f['rec_rank'].tolist() == [rr] and f['tt_rank'].tolist() == [tr]
    assert np.allclose(f['score'][0], score)
    # RRF ties (single-source items at equal source rank) are pessimistic in retrieval order
    ranks, top = pool_ranks(f['score'], f['cand'], np.array([5]))
    assert ranks[0] == 4 and top[0, :4].tolist() == [1, 3, 2, 5]


def test_interleave_skips_duplicates_until_k_unique():
    rec = np.array([[10, 11, 12, 13, 14]])
    tt = np.array([[10, 11, 12, 13, 15]])      # identical prefix: every two-tower slot is a duplicate until 15
    f = interleave_fuse(rec, tt, k=5)
    assert f['cand'].tolist() == [[10, 11, 12, 13, 14]]
    assert np.allclose(f['score'][0], [2 / 61, 2 / 62, 2 / 63, 2 / 64, 1 / 65])


def test_vectorised_matches_reference_random():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m, k = 30, 20
        rec = rng.choice(np.arange(1, 60), m, replace=False)
        tt = rng.choice(np.arange(1, 60), m, replace=False)
        pool, score, rr, tr = interleave_reference(list(rec), list(tt), k=k)
        f = interleave_fuse(rec[None], tt[None], k=k)
        assert f['cand'][0].tolist() == pool
        assert f['rec_rank'][0].tolist() == rr and f['tt_rank'][0].tolist() == tr
        assert np.allclose(f['score'][0], score, rtol=1e-6)
        assert len(set(pool)) == k


def toy_filtered_source(rng, n_items, history, query_ts, first_ts, m):
    eligible = [i for i in range(1, n_items + 1) if i not in history and first_ts[i] <= query_ts]
    return list(rng.permutation(eligible)[:m])


def test_pool_never_contains_history_or_unreleased_items():
    rng = np.random.default_rng(1)
    n_items = 40
    first_ts = np.concatenate([[np.iinfo(np.int64).max], rng.integers(0, 100, n_items)])
    rows, rec, tt, ts, hist_rows, hist_items = 6, [], [], [], [], []
    for r in range(rows):
        t = int(rng.integers(60, 100))
        history = set(rng.choice(np.flatnonzero(first_ts[1:] < t) + 1, 5, replace=False).tolist())
        rec.append(toy_filtered_source(rng, n_items, history, t, first_ts, 12))
        tt.append(toy_filtered_source(rng, n_items, history, t, first_ts, 12))
        ts.append(t)
        hist_rows += [r] * len(history)
        hist_items += sorted(history)
    ts, hist_rows, hist_items = np.array(ts), np.array(hist_rows), np.array(hist_items)
    f = interleave_fuse(np.array(rec), np.array(tt), k=10)
    v = filter_violations(f['cand'], ts, hist_rows, hist_items, first_ts)
    assert v == {'padding': 0, 'duplicates': 0, 'history_items': 0, 'unreleased_items': 0}
    # the checker catches violations
    bad = f['cand'].copy()
    bad[0, 0] = hist_items[hist_rows == 0][0]
    unreleased = int(np.argmax(first_ts[1:] > ts[1])) + 1
    assert first_ts[unreleased] > ts[1]
    bad[1, 1] = unreleased
    v = filter_violations(bad, ts, hist_rows, hist_items, first_ts)
    assert v['history_items'] >= 1 and v['unreleased_items'] >= 1


def test_fuse_rejects_padding():
    with pytest.raises(ValueError):
        interleave_fuse(np.array([[1, 0, 3]]), np.array([[1, 2, 3]]), k=3)
