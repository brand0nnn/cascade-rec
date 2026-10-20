import numpy as np
import pandas as pd

from cascade.data.split import PERIOD_RETRIEVER_TRAIN, PERIOD_RANKER_TRAIN, PERIOD_TEST, PERIOD_VAL, assign_periods, compute_t_ranker_train


def test_assign_periods_boundaries():
    ts = np.array([0, 9, 10, 19, 20, 29, 30, 100])
    p = assign_periods(ts, t_ranker_train=10, t1=20, t2=30)
    assert p.tolist() == [PERIOD_RETRIEVER_TRAIN, PERIOD_RETRIEVER_TRAIN, PERIOD_RANKER_TRAIN, PERIOD_RANKER_TRAIN, PERIOD_VAL, PERIOD_VAL, PERIOD_TEST, PERIOD_TEST]


def test_t_ab_between_min_and_t1():
    ts = np.arange(1000)
    t_ranker_train = compute_t_ranker_train(ts, t1=500, ranker_train_fraction=0.15)
    assert ts.min() < t_ranker_train < 500
    assert np.mean(ts[ts < 500] >= t_ranker_train) == np.float64(0.15)


def test_sample_split_consistent(sample_data):
    s = sample_data.split
    inter = pd.read_parquet(sample_data.dir / "interactions.parquet")
    ts, p = inter["timestamp"].values, inter["period"].values
    assert (ts[p <= PERIOD_RANKER_TRAIN] < s["t1"]).all(), "val/test timestamps leaked into training"
    assert (ts[p == PERIOD_RETRIEVER_TRAIN] < s["t_ranker_train"]).all() and (ts[p == PERIOD_RANKER_TRAIN] >= s["t_ranker_train"]).all()
    assert ((ts[p == PERIOD_VAL] >= s["t1"]) & (ts[p == PERIOD_VAL] < s["t2"])).all()
    assert (ts[p == PERIOD_TEST] >= s["t2"]).all()
    assert ts.min() < s["t_ranker_train"] < s["t1"]
    tr = sample_data.interactions(["retriever_train", "ranker_train"])
    assert (tr["ts"] < s["t1"]).all()
