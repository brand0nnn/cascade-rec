import numpy as np
import pandas as pd

from cascade.data.split import PERIOD_A, PERIOD_B, PERIOD_TEST, PERIOD_VAL, assign_periods, compute_t_ab


def test_assign_periods_boundaries():
    ts = np.array([0, 9, 10, 19, 20, 29, 30, 100])
    p = assign_periods(ts, t_ab=10, t1=20, t2=30)
    assert p.tolist() == [PERIOD_A, PERIOD_A, PERIOD_B, PERIOD_B, PERIOD_VAL, PERIOD_VAL, PERIOD_TEST, PERIOD_TEST]


def test_t_ab_between_min_and_t1():
    ts = np.arange(1000)
    t_ab = compute_t_ab(ts, t1=500, period_b_fraction=0.15)
    assert ts.min() < t_ab < 500
    assert np.mean(ts[ts < 500] >= t_ab) == np.float64(0.15)


def test_toy_split_consistent(toy_data):
    s = toy_data.split
    inter = pd.read_parquet(toy_data.dir / "interactions.parquet")
    ts, p = inter["timestamp"].values, inter["period"].values
    assert (ts[p <= PERIOD_B] < s["t1"]).all(), "val/test timestamps leaked into training"
    assert (ts[p == PERIOD_A] < s["t_ab"]).all() and (ts[p == PERIOD_B] >= s["t_ab"]).all()
    assert ((ts[p == PERIOD_VAL] >= s["t1"]) & (ts[p == PERIOD_VAL] < s["t2"])).all()
    assert (ts[p == PERIOD_TEST] >= s["t2"]).all()
    assert ts.min() < s["t_ab"] < s["t1"]
    tr = toy_data.interactions(["A", "B"])
    assert (tr["ts"] < s["t1"]).all()
