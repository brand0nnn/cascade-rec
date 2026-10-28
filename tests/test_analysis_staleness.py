import numpy as np

from cascade.analysis.staleness import cluster_boot, product_era, val_quarter

SPLIT = {"t_ranker_train": 100, "t1": 200, "t2": 600}


def test_product_era_boundaries():
    era = product_era(np.array([0, 99, 100, 199, 200, 500]), SPLIT)
    assert era.tolist() == ["retriever era"] * 2 + ["ranker-training era"] * 2 + ["Val era"] * 2


def test_val_quarter_equal_duration():
    assert val_quarter(np.array([200, 299, 300, 499, 500, 599]), SPLIT).tolist() == ["Q1", "Q1", "Q2", "Q3", "Q4", "Q4"]


def test_cluster_boot_matches_macro_when_constant():
    users = np.array([1, 1, 2, 3, 3, 3])
    vals = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    groups = np.array(["a", "b", "a", "b", "b", "a"])
    b = cluster_boot(users, vals, groups, ["a", "b"], n_resamples=200)
    assert b.shape == (200, 2)
    assert np.nanmin(b) >= 0 and np.nanmax(b) <= 1
    # identical users => every resample gives the same macro
    b2 = cluster_boot(np.array([1, 2]), np.array([0.5, 0.5]), np.array(["a", "a"]), ["a"], n_resamples=20)
    assert np.allclose(b2, 0.5)


def test_count_bucket_and_item_boot():
    from cascade.analysis.staleness import count_bucket, item_boot_macro
    assert count_bucket(np.array([0, 1, 4, 5, 999, 1000])).tolist() == ["0:0", "1:1-4", "1:1-4", "2:5-19", "6:300-999", "7:1000+"]
    b = item_boot_macro(np.array([1, 2]), np.array([1, 2]), np.array([0.5, 0.5]), np.array(["a", "a"]), ["a"], 10)
    assert np.allclose(b, 0.5)
