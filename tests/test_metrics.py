import numpy as np
import pytest

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap


def test_recall_ndcg_hand_computed():
    ranks = np.array([1, 2, 10, 11, np.inf])
    assert M.recall_at(ranks, 10).tolist() == [1, 1, 1, 0, 0]
    nd = M.ndcg_at(ranks, 10)
    assert nd[0] == 1.0  # target ranked first
    assert nd[1] == pytest.approx(1 / np.log2(3))
    assert nd[2] == pytest.approx(1 / np.log2(11))
    assert nd[3] == 0 and nd[4] == 0


def test_macro_vs_micro():
    vals = np.array([1.0, 0.0, 0.0, 1.0])
    users = np.array([1, 1, 1, 2])
    mm = M.macro_micro(vals, users)
    assert mm["micro"] == pytest.approx(0.5)
    assert mm["macro"] == pytest.approx((1 / 3 + 1) / 2)


def test_auc_and_gauc():
    assert M.instance_auc(np.array([0.9, 0.1, 0.2]), 0) == 1.0
    assert M.instance_auc(np.array([0.0, 0.1, 0.2]), 0) == 0.0
    assert M.instance_auc(np.array([0.5, 0.5, 0.1]), 0) == pytest.approx(0.75)
    aucs = np.array([1.0, 0.0, 0.5])
    assert M.gauc(aucs, np.array([1, 1, 2])) == pytest.approx((0.5 + 0.5) / 2)


def test_coverage_and_concentration():
    top = np.array([[1, 2], [2, 3]])
    assert M.coverage(top, 10) == pytest.approx(0.3)
    is_top = np.array([False, True, False, False])
    assert M.popularity_concentration(top, is_top) == pytest.approx(0.25)


def test_bootstrap_identical_systems_zero():
    rng = np.random.default_rng(0)
    a = rng.random(200)
    r = paired_bootstrap(np.arange(200) % 50, a, a, n_resamples=100)
    assert r["diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
