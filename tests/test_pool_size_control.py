"""Toy checks for the size-matched P3 control (no data files)."""
import numpy as np
import pytest

from cascade.pipelines import pool_ranking as pr
from cascade.pipelines import pool_size_control as sc
from cascade.utils.config import REPO_ROOT


def test_subset_matches_counts_per_part_and_only_natural_queries():
    rng = np.random.default_rng(0)
    prefix = np.arange(1000) < 800
    found = rng.random(1000) < 0.4
    keep = sc.size_matched_subset(prefix, found, 100, 30)
    assert (keep & prefix).sum() == 100 and (keep & ~prefix).sum() == 30
    assert not (keep & ~found).any()
    assert np.array_equal(keep, sc.size_matched_subset(prefix, found, 100, 30))  # fixed seed


def test_subset_refuses_more_than_available():
    prefix = np.array([True, True, False]); found = np.array([True, False, True])
    with pytest.raises(ValueError):
        sc.size_matched_subset(prefix, found, 2, 1)


def test_training_functions_reused_unchanged():
    names = pr.function_sources(REPO_ROOT/'src/cascade/pipelines/pool_ranking.py')
    assert {'fit', 'evaluate'} <= set(names)
