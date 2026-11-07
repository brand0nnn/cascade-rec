"""Toy checks for the P3 x rolling wrapper (no data files)."""
import numpy as np

from cascade.analysis import freshness_val as F
from cascade.pipelines import pool_rolling as pr
from cascade.pipelines import rolling_ranking as rr


def test_masks_match_source_recipes():
    assert pr.mask_for('bce_coreview') == 'pairwise_coreview'
    assert pr.mask_for('bce_text') == 'bce_text'


def test_rolling_cutoffs_floor_partial_first_month_at_t0():
    ts = np.array([F.T0, F.T0+3*86400000, 1596240000000+5, F.T1-1], np.int64)  # 26 Jul, 29 Jul, 1 Aug 2020, last B ms
    cut = rr.rolling_cutoffs(ts, F.T0)
    assert cut[0] == F.T0 and cut[1] == F.T0
    assert cut[2] == 1596240000000  # 2020-08-01 UTC
    assert (cut <= ts).all() and cut[3] == F.month_start_ms(ts[3:])[0]


def test_interaction_terms_are_a_difference_in_differences():
    (pos_a, pos_b), (neg_a, neg_b) = pr.additive_terms()
    assert {pos_a, pos_b, neg_a, neg_b} == {'P3_rolling_bce_coreview_monthly', 'P1_bce_text_T0', 'P3_bce_text_T0',
                                            'P1_rolling_bce_coreview_monthly'}
    rng = np.random.default_rng(0)
    v = {k: rng.random(5) for k in (pos_a, pos_b, neg_a, neg_b)}
    did = (v['P3_rolling_bce_coreview_monthly']-v['P1_rolling_bce_coreview_monthly']) - (v['P3_bce_text_T0']-v['P1_bce_text_T0'])
    assert np.allclose((v[pos_a]+v[pos_b])-(v[neg_a]+v[neg_b]), did)


def test_contrast_plan_adds_text_control_only_when_run():
    base = {f for f, _, _ in pr.contrast_plan(False)}
    full = {f for f, _, _ in pr.contrast_plan(True)}
    assert 'coreview_minus_text_P3_rolling' not in base and 'coreview_minus_text_P3_rolling' in full
    assert base <= full
    refs = set(pr.reference_paths())
    for _, ref, cand in pr.contrast_plan(True):
        for name in (ref, cand):
            assert name in refs or name.startswith('P3_rolling_')


def test_natural_rows_serves_stored_rows_by_full_index_and_zeros_elsewhere():
    natural = np.array([False, True, False, True, True])
    stored = (np.arange(3*200*2, dtype=np.float32).reshape(3, 200, 2))
    view = pr.NaturalRows(stored, natural)
    got = view[np.array([4, 0, 1, 3, 1])]
    assert got.dtype == np.float32 and got.shape == (5, 200, 2)
    assert np.array_equal(got[0], stored[2]) and not got[1].any()
    assert np.array_equal(got[2], stored[0]) and np.array_equal(got[3], stored[1]) and np.array_equal(got[4], stored[0])
