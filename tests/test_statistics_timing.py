"""Toy checks for the statistics-timing analysis (no data files)."""
import numpy as np

from cascade.analysis import statistics_timing as ST
from cascade.analysis import freshness_val as F


def test_shift_and_lag_cutoffs_are_calendar_months_and_past_only():
    t1 = F.T1  # 11 Aug 2021
    assert str(np.datetime64(int(ST.shift_months([t1], 3)[0]), 'ms'))[:10] == '2021-05-11'
    ts = np.array([F.T1 + 5 * 86400000, F.T1 + 60 * 86400000], np.int64)
    for lag in (0, 1, 12):
        cut = ST.lag_cutoffs(ts, lag)
        assert (cut <= ts).all()
    assert str(np.datetime64(int(ST.lag_cutoffs(ts, 12)[0]), 'ms'))[:10] == '2020-08-01'


def test_swap_by_column_and_by_product():
    t0 = {'common': np.zeros((1, 3, 9), np.float32), 'graph': np.zeros((1, 3, 6), np.float32)}
    t1 = {'common': np.ones((1, 3, 9), np.float32), 'graph': np.ones((1, 3, 6), np.float32)}
    a = ST.swap(t0, t1, 'common', [1], None)
    assert a['common'][..., 1].all() and not a['common'][..., 0].any() and not a['graph'].any() and not t0['common'].any()
    mask = np.array([[True, False, False]])
    b = ST.swap(t0, t1, 'both', None, mask)
    assert b['common'][0, 0].all() and b['graph'][0, 0].all() and not b['common'][0, 1:].any()


def test_unknown_stats_auc():
    scaler = {'common': {'mean': [0.0]*9, 'std': [1.0]*9}}
    common = np.zeros((2, 3, 9), np.float32); common[0, 1:, 1] = 1; common[1, :, 1] = 1  # row 0 target unknown
    cand = np.array([[5, 6, 7], [8, 9, 10]]); target = np.array([5, 8])
    s = ST.unknown_stats(common, cand, target, scaler)
    assert s['target_unknown'] == 0.5 and s['other_unknown'] == 0.0 and abs(s['auc_unknown'] - 0.75) < 1e-12
