import numpy as np

from cascade.eval.segments import new_in_period


def test_new_in_period_boundaries():
    info = {"t1": 100, "t2": 200}
    ft = np.array([50, 100, 150, 199, 200, 250])
    assert new_in_period(ft, "val", info).tolist() == [False, True, True, True, True, True]
    assert new_in_period(ft, "test", info).tolist() == [False, False, False, False, True, True]


def test_new_in_period_equals_cold_on_test(sample_data):
    inst = sample_data.split_instances("test")
    items = sample_data.items.set_index("item")
    new = new_in_period(items.loc[inst["item"], "first_ts"].values, "test", sample_data.split)
    cold = items.loc[inst["item"], "segment"].values == "cold"
    assert (new == cold).all()
    val = sample_data.split_instances("val")
    assert new_in_period(items.loc[val["item"], "first_ts"].values, "val", sample_data.split).any()
