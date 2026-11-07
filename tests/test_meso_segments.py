import numpy as np
import pytest

from cascade.analysis import meso_segments as ms


def test_history_labels_match_claude_buckets():
    out = ms.history_labels(np.array([1, 2, 3, 5, 6, 10, 11, 400]))
    assert list(out) == ['1-2', '1-2', '3-5', '3-5', '6-10', '6-10', '11+', '11+']
    with pytest.raises(ValueError):
        ms.history_labels(np.array([0]))


def test_gap_labels_boundaries():
    d = ms.DAY_MS
    out = ms.gap_labels(np.array([0, 7*d, 7*d + 1, 90*d, 90*d + 1]))
    assert list(out) == ['<=7d', '<=7d', '8-90d', '8-90d', '>90d']
    with pytest.raises(ValueError):
        ms.gap_labels(np.array([-1]))


def test_warm_terciles_ignore_new_and_cover_warm():
    counts = np.array([0, 1, 2, 3, 4, 5, 6, 0])
    warm = counts > 0
    lab, (q1, q2) = ms.warm_terciles(counts, warm)
    assert lab[0] is None and lab[-1] is None
    assert set(lab[warm]) == {'warm_low', 'warm_mid', 'warm_high'}
    assert all(c <= q1 for c in counts[lab == 'warm_low'])
    assert all(c > q2 for c in counts[lab == 'warm_high'])


def test_category_l2_and_top_n():
    cats = ms.category_l2([['A', 'x'], ['A'], [], None, ['B', 'y', 'z'], ['A', 'x']])
    assert list(cats) == ['x', ms.NONE_CAT, ms.NONE_CAT, ms.NONE_CAT, 'y', 'x']
    lab, keep = ms.top_n_labels(cats, 2)
    assert keep == [ms.NONE_CAT, 'x']
    assert list(lab) == ['x', ms.NONE_CAT, ms.NONE_CAT, ms.NONE_CAT, 'other', 'x']


def test_guard_refuses_test_paths(tmp_path):
    with pytest.raises(ValueError):
        ms.guard(tmp_path/'test'/'ranks.npz')
    with pytest.raises(ValueError):
        ms.guard(tmp_path/'test_ranks.npz')
