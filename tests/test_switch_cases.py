import numpy as np
import pandas as pd

from cascade.analysis.switch_cases import BUCKETS, aggregate, bucket_masks, draw, price_jump, tri


def test_bucket_masks_disjoint_and_rules():
    dm = np.array([1, 1, 1, 1, 0, 1], bool)
    new = np.array([1, 0, 0, 0, 1, 0], bool)
    nb = np.array([0, 50, 49, 0, 0, 1])
    m = bucket_masks(dm, new, nb)
    assert m[BUCKETS[0]].tolist() == [1, 0, 0, 0, 0, 0]
    assert m[BUCKETS[1]].tolist() == [0, 1, 0, 0, 0, 0]
    assert m[BUCKETS[2]].tolist() == [0, 0, 1, 0, 0, 1]
    assert not (m[BUCKETS[0]] & m[BUCKETS[1]]).any()


def test_draw_deterministic_sorted_no_duplicates():
    inst = np.arange(100, 0, -1)
    masks = {b: np.zeros(100, bool) for b in BUCKETS}
    masks[BUCKETS[0]][:30] = True; masks[BUCKETS[1]][20:60] = True; masks[BUCKETS[2]][90:] = True
    a, b = draw(masks, inst, 8), draw(masks, inst, 8)
    assert a == b
    chosen = [i for k in BUCKETS for i in a[k]['chosen']]
    assert len(chosen) == len(set(chosen))
    assert all(a[k]['chosen'] == sorted(a[k]['chosen']) for k in BUCKETS)
    assert len(a[BUCKETS[2]]['chosen']) == 8 and a[BUCKETS[2]]['eligible'] == 10


def test_price_jump_strict_thresholds():
    assert price_jump(10, 20) == 'similar' and price_jump(10, 20.01) == 'up'
    assert price_jump(10, 5) == 'similar' and price_jump(10, 4.99) == 'down'
    assert price_jump(np.nan, 5) == 'unknown' and price_jump(0, 5) == 'unknown'


def test_tri_and_aggregate_shares():
    v = tri(np.array([1, 2, 0, 3]), np.array([1, 3, 1, 3]), np.array([1, 1, 0, 1], bool))
    assert v.tolist() == ['no', 'yes', 'unknown', 'no']
    meas = pd.DataFrame({m: v for m in ('leaf_category_jump', 'top_category_jump', 'second_level_jump',
                                        'store_change', 'leaf_unseen_in_history')})
    agg = aggregate(meas, {'all': np.ones(4, bool)})
    r = agg[agg.measure == 'leaf_category_jump'].iloc[0]
    assert (r.yes, r.no, r.unknown, r.share_yes_of_known) == (1, 2, 1, round(1 / 3, 4))
