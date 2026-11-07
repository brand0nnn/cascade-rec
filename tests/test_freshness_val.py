"""Snapshot cut rule for the Val freshness study (toy data only)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.analysis.freshness_val import SnapshotFeatures, assert_cut_rule, month_start_ms
from cascade.data.instances import ProcessedData, build_sequences
from cascade.data.local_graph_features import FrozenGraphFeatures


def toy():
    # Users 1-4; item reviews at times 1..25. Cut points 10 (T0-like) and 20 (T1-like); Test starts at 30.
    edges = pd.DataFrame({'user': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                          'item': [1, 2, 4, 1, 3, 5, 2, 3, 5, 1, 2, 3],
                          'timestamp': [1, 2, 15, 1, 2, 18, 1, 12, 31, 1, 2, 3]})
    edges = edges.sort_values(['user', 'timestamp', 'item']).reset_index(drop=True)
    ptr, item, ts = build_sequences(edges, 4)
    items = pd.DataFrame({'item': np.arange(1, 7), 'leaf_cat': ['X', 'Y', 'X', '', 'X', ' '],
                          'store': ['A', 'B', 'A', '', 'A', ''], 'first_ts': [1, 1, 2, 15, 18, 40]})
    data = ProcessedData(Path('.'), 4, 6, ptr, item, ts, pd.DataFrame(), items,
                         {'t_ranker_train': 10, 't1': 20, 't2': 30}, 50)
    # Queries at t=22: user 1 history (1,2,4), user 2 history (1,3,5), user 3 history (2,3).
    queries = pd.DataFrame({'user': [1, 2, 3], 'hist_len': [3, 3, 2], 'ts': [22, 22, 22], 'item': [6, 6, 6]})
    candidates = np.array([[3, 5, 0], [2, 4, 0], [1, 4, 5]])
    return data, queries, candidates


def test_month_start_is_utc_calendar_month():
    ts = np.array([np.datetime64('2021-08-11T00:56:54.042', 'ms'), np.datetime64('2022-03-01T00:00:00.000', 'ms'),
                   np.datetime64('2021-12-31T23:59:59.999', 'ms')]).astype(np.int64)
    expected = np.array(['2021-08-01', '2022-03-01', '2021-12-01'], dtype='datetime64[ms]').astype(np.int64)
    assert np.array_equal(month_start_ms(ts), expected)
    assert (month_start_ms(ts) <= ts).all()


def test_snapshot_holds_only_reviews_strictly_before_cutoff():
    data, _, _ = toy()
    for cutoff in (10, 15, 20):
        snap = SnapshotFeatures(data, cutoff)
        assert snap.audit['max_review_timestamp'] < cutoff
        users = np.repeat(np.arange(5), np.diff(data.user_ptr))
        expected = int((data.seq_ts < cutoff).sum())
        assert snap.matrix.nnz == expected
        # A review exactly at the cutoff (user 1, item 4, t=15) is excluded at cutoff 15.
        if cutoff == 15: assert snap.by_user[1, 4] == 0
    with pytest.raises(ValueError, match='Test'):
        SnapshotFeatures(data, 31)


def test_cut_rule_rejects_snapshots_at_or_after_a_query():
    data, queries, candidates = toy()
    assert_cut_rule(SnapshotFeatures(data, 20), queries, candidates)
    late = queries.copy(); late['ts'] = 18
    with pytest.raises(ValueError, match='at or after a query|after a query'):
        assert_cut_rule(SnapshotFeatures(data, 20), late, candidates)


def test_query_user_excluded_and_matches_explicit_reference():
    data, queries, candidates = toy()
    snap = SnapshotFeatures(data, 20)
    assert_cut_rule(snap, queries, candidates)
    fast, ref = snap.build(queries, candidates), FrozenGraphFeatures.build(snap, queries, candidates)
    for a, b in zip(fast, ref):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
    # Candidate 5 for user 1: snapshot reviewers before 20 are user 2 (t=18) only -> degree 1.
    assert fast[0][0, 1, 0] == pytest.approx(np.log1p(1))
    # Candidate 4 for user 2: reviewed only by user 1 (t=15); user 2 is not a reviewer.
    assert fast[0][1, 1, 0] == pytest.approx(np.log1p(1))
    # User 3's own review of item 3 (t=12) must not support candidate 5 via user 3 (item 5 at t=31 is outside).
    common, graph = fast
    assert common[2, 2, 0] == pytest.approx(np.log1p(1))  # item 5: only user 2 before cutoff
    # A query user edge to a candidate violates the exclusion premise.
    bad = candidates.copy(); bad[0, 0] = 2
    with pytest.raises(ValueError, match='candidate'):
        assert_cut_rule(snap, queries, bad)


def test_snapshot_at_cutoff_drops_later_edges_from_features():
    data, queries, candidates = toy()
    early, late = SnapshotFeatures(data, 10), SnapshotFeatures(data, 20)
    q = queries.iloc[[0]]
    e = early.build(q, candidates[[0]])[0]; l = late.build(q, candidates[[0]])[0]
    # Item 5 is unreleased in the T0-like snapshot and known in the T1-like snapshot.
    assert e[0, 1, 1] == 0 and l[0, 1, 1] == 1


def test_mechanism_hybrids_swap_only_registered_columns():
    from cascade.analysis.freshness_mechanism import hybrid
    rng = np.random.default_rng(0)
    t1 = {'common': rng.normal(size=(2, 3, 9)).astype(np.float32), 'graph': rng.normal(size=(2, 3, 6)).astype(np.float32)}
    mo = {'common': rng.normal(size=(2, 3, 9)).astype(np.float32), 'graph': rng.normal(size=(2, 3, 6)).astype(np.float32)}
    h1, h2, h3 = hybrid('H1', t1, mo), hybrid('H2', t1, mo), hybrid('H3', t1, mo)
    assert np.array_equal(h1['common'][..., 1], t1['common'][..., 1]) and np.array_equal(h1['graph'], mo['graph'])
    assert np.array_equal(np.delete(h1['common'], 1, -1), np.delete(mo['common'], 1, -1))
    assert np.array_equal(h2['common'][..., :2], t1['common'][..., :2]) and np.array_equal(h2['common'][..., 2:], mo['common'][..., 2:])
    assert np.array_equal(h3['common'][..., 1], mo['common'][..., 1]) and np.array_equal(h3['graph'], t1['graph'])
    assert np.array_equal(np.delete(h3['common'], 1, -1), np.delete(t1['common'], 1, -1))


def test_unknown_indicator_auc():
    from cascade.analysis.freshness_mechanism import binary_auc, unknown_stats
    assert binary_auc(1, 0) == 1 and binary_auc(0, 1) == 0 and binary_auc(0.3, 0.3) == 0.5
    cand = np.array([[5, 6, 7, 0], [8, 9, 10, 11]]); target = np.array([5, 9])
    unknown = np.array([[True, False, False, False], [False, False, True, True]])
    s = unknown_stats(cand, target, unknown)
    assert s['share_positives_unknown'] == 0.5 and s['share_negatives_unknown'] == 2/5
    assert s['target_rate_among_unknown_slots'] == 1/3
