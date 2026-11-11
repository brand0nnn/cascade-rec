"""Regression: mixed-pool inference must use natural-only training scaling."""
import numpy as np
import pytest

from cascade.pipelines.hybrid_normalization import natural_score_stats


def test_excludes_forced_lists_padding_and_appended_target():
    lists = {"forced": np.array([False, True, False]),
             "cand": np.array([[1, 2, 0], [1, 2, 3], [1, 0, 0]]),
             "score": np.array([[1., 3., 999.], [100., 200., 300.], [5., -999., 999.]])}
    mean, std = natural_score_stats(lists)
    assert mean == 3.
    assert std == float(np.array([1., 3., 5.], np.float32).std())


@pytest.mark.parametrize("scores", [[[1., 1., 0.]], [[1., np.nan, 0.]]])
def test_rejects_invalid_statistics(scores):
    with pytest.raises(ValueError):
        natural_score_stats({"forced": np.array([False]), "cand": np.array([[1, 2, 0]]),
                             "score": np.array(scores)})


def test_rejects_no_natural_lists():
    with pytest.raises(ValueError):
        natural_score_stats({"forced": np.array([True]), "cand": np.array([[1, 2, 3]]),
                             "score": np.array([[1., 2., 3.]])})
