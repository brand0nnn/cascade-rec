"""Frozen bucket rules and seeded draw of the Track N3 Val case study (synthetic arrays only)."""
import numpy as np

from cascade.analysis.case_study_current import BUCKETS, brand_change, bucket_masks, draw, price_jump

INF = np.inf


def test_bucket_rules_and_disjointness():
    r_ret = np.array([INF, INF, 3, 4, 50, 5, 9, 150])
    r_bce = np.array([INF, INF, 20, 2, 7, 1, 30, 11])
    r_pw = np.array([INF, INF, 4, 15, 40, 2, 31, 3])
    new = np.array([True, False, False, True, True, False, False, True])
    m = bucket_masks(r_ret, r_bce, r_pw, new)
    assert m[BUCKETS[0]].tolist() == [True] + [False] * 7          # new miss only
    assert np.flatnonzero(m[BUCKETS[1]]).tolist() == [2]           # top-10 hit, bce demotes, pairwise keeps
    assert np.flatnonzero(m[BUCKETS[2]]).tolist() == [4]           # bce rescue from rank 50
    assert np.flatnonzero(m[BUCKETS[3]]).tolist() == [3]           # pairwise demotes, bce keeps
    assert (np.stack(list(m.values())).sum(0) <= 1).all()


def test_draw_is_seeded_sorted_and_unique():
    rng = np.random.default_rng(5)
    inst = rng.permutation(1000) + 10
    masks = {b: rng.random(1000) < 0.3 for b in BUCKETS}
    a, b = draw(masks, inst, 7), draw(masks, inst, 7)
    assert a == b
    chosen = [i for v in a.values() for i in v['chosen']]
    assert len(chosen) == len(set(chosen)) == 28
    for name, v in a.items():
        assert v['chosen'] == sorted(v['chosen']) and set(v['chosen']) <= set(inst[masks[name]])
    small = {b: np.zeros(1000, bool) for b in BUCKETS}; small[BUCKETS[0]][:3] = True
    assert len(draw(small, inst, 7)[BUCKETS[0]]['chosen']) == 3


def test_computed_rubric_fields():
    assert brand_change('Anker', ' anker') == 'same' and brand_change('Anker', '') == 'unknown'
    assert brand_change(None, 'X') == 'unknown' and brand_change('A', 'B') == 'changed'
    assert price_jump(10.0, 25.0) == 'up' and price_jump(10.0, 4.0) == 'down'
    assert price_jump(10.0, 12.0) == 'similar' and price_jump(np.nan, 5.0) == 'unknown'
