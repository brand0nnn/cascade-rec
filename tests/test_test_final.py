"""Toy checks for the locked Test pipeline helpers (no data files)."""
import numpy as np
import pytest

from cascade.pipelines import test_final as TF


def test_judge_verdicts():
    assert TF.judge(0.002, 0.001, 0.003, 'positive') == 'held'
    assert TF.judge(-0.002, -0.003, -0.001, 'positive') == 'not held'
    assert TF.judge(0.001, -0.001, 0.003, 'positive') == 'inconclusive'
    assert TF.judge(-0.002, -0.003, -0.001, 'negative') == 'held'
    assert TF.judge(0.0, -0.001, 0.001, 'nonpositive') == 'held'
    assert TF.judge(0.001, -0.001, 0.003, 'nonpositive') == 'not held'
    with pytest.raises(ValueError):
        TF.judge(0, 0, 0, 'other')


def test_cutoff_rules():
    ts = np.array([TF.T2, TF.T2 + 40 * 86400 * 1000], np.int64)
    assert (TF.cutoffs_for('T2', ts) == TF.T2).all()
    monthly = TF.cutoffs_for('monthly', ts)
    assert (monthly <= ts).all() and monthly[0] < TF.T2  # July 2022 queries use 1 Jul 2022


def test_unrestricted_view_keeps_data_and_lifts_only_the_t2_guard():
    class Data: pass
    d = Data(); d.split = {'t2': 5, 't1': 3}; d.arr = np.arange(3)
    v = TF.unrestricted(d)
    assert v.split['t1'] == 3 and v.split['t2'] > 10**15 and v.arr is d.arr and d.split['t2'] == 5


def test_every_system_uses_a_registered_model_and_arm():
    for model, arm in TF.SYSTEMS.values():
        assert model in TF.MODELS and arm in TF.ARMS and TF.MODELS[model][1] == TF.ARMS[arm][0]
