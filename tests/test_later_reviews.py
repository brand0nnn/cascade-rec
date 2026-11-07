"""Opt-in exclusion of the user's later same-period reviews from sampled ranker negatives."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.data.instances import ProcessedData
from cascade.pipelines.ranker_training import _sample_list_columns, later_review_cells

T_RANKER_TRAIN, T1, T2 = 100, 200, 300


def _tiny_data() -> ProcessedData:
    """User 1: history item 5, target 6 at t=120, item 8 also at t=120, later item 7 (t=150), Val item 9.

    User 2: item 12 at t=90, then item 10 at t=160 (ranker-training period).
    """
    # CSR sequences sorted by (user, ts, item); user 0 is padding.
    seq = [  # (user, item, ts)
        (1, 5, 50), (1, 6, 120), (1, 8, 120), (1, 7, 150), (1, 9, 250),
        (2, 12, 90), (2, 10, 160),
    ]
    user_ptr = np.array([0, 0, 5, 7], dtype=np.int64)
    seq_item = np.array([s[1] for s in seq], dtype=np.int32)
    seq_ts = np.array([s[2] for s in seq], dtype=np.int64)
    period = lambda t: 0 if t < T_RANKER_TRAIN else 1 if t < T1 else 2 if t < T2 else 3  # noqa: E731
    hist = [0, 1, 1, 3, 4, 0, 1]
    inst = pd.DataFrame({
        "inst": np.arange(len(seq), dtype=np.int64),
        "user": [s[0] for s in seq], "item": seq_item, "ts": seq_ts,
        "period": [period(s[2]) for s in seq], "hist_len": np.array(hist, dtype=np.int32),
    })
    inst = inst[inst["hist_len"] > 0].reset_index(drop=True)
    return ProcessedData(
        dir=Path("."), n_users=2, n_items=13, user_ptr=user_ptr, seq_item=seq_item, seq_ts=seq_ts,
        instances=inst, items=pd.DataFrame({"item": np.arange(14)}),
        split={"t_ranker_train": T_RANKER_TRAIN, "t1": T1, "t2": T2}, max_hist_len=50,
    )


def _tiny_lists() -> dict[str, np.ndarray]:
    # Row 0: user 1's instance at t=120 (target 6 in column 4). Row 1: user 2's instance at t=160.
    cand = np.array([
        [7, 5, 9, 10, 6, 8, 11, 0],
        [7, 12, 13, 10, 0, 0, 0, 0],
    ], dtype=np.int32)
    target_pos = np.array([4, 3])
    label = np.zeros(cand.shape, dtype=np.int8)
    label[np.arange(2), target_pos] = 1
    return {"inst": np.array([1, 6]), "user": np.array([1, 2], dtype=np.int32), "cand": cand,
            "label": label, "target_pos": target_pos}


def test_later_same_period_review_is_the_only_exclusion():
    later = later_review_cells(_tiny_lists(), _tiny_data())
    expected = np.zeros((2, 8), dtype=bool)
    expected[0, 0] = True  # item 7: user 1 reviews it at t=150, after t=120 and before T1
    # Not excluded: history item 5 (earlier), Val item 9 (t >= T1), item 10 (another user's review),
    # the target 6, item 8 (same timestamp as the instance, not strictly later), padding.
    # Row 1: item 7 is user 1's later review, not user 2's.
    np.testing.assert_array_equal(later, expected)


def test_sampler_never_picks_excluded_and_keeps_positive():
    lists = _tiny_lists()
    eligible = ~later_review_cells(lists, _tiny_data())
    for seed in range(20):
        selected = _sample_list_columns(lists, 10, np.random.default_rng(seed), eligible)
        np.testing.assert_array_equal(selected[:, 0], lists["target_pos"])
        row0 = set(selected[0, 1:][selected[0, 1:] >= 0].tolist())
        assert row0 == {1, 2, 3, 5, 6}  # every eligible negative, never column 0 (later review) or padding
        row1 = set(selected[1, 1:][selected[1, 1:] >= 0].tolist())
        assert row1 == {0, 1, 2}
    # Without the mask the later review is an ordinary negative (labels are never changed).
    unmasked = _sample_list_columns(lists, 10, np.random.default_rng(0))
    assert 0 in set(unmasked[0, 1:].tolist())
    assert lists["label"][0, 0] == 0


def test_val_lists_are_rejected():
    lists = _tiny_lists()
    lists["inst"], lists["user"] = np.array([4, 6]), np.array([1, 2], dtype=np.int32)  # inst 4 is in Val
    with pytest.raises(ValueError, match="training lists only"):
        later_review_cells(lists, _tiny_data())


def _reference_sampler(lists, n_negatives, rng):
    """Verbatim copy of the sampler before the opt-in mask was added."""
    cand = lists["cand"]
    labels = lists["label"]
    selected = np.full((len(cand), n_negatives + 1), -1, dtype=np.int16)
    selected[:, 0] = np.asarray(lists["target_pos"], dtype=np.int16)
    for row in range(len(cand)):
        valid_neg = np.flatnonzero((labels[row] == 0) & (cand[row] > 0))
        if not len(valid_neg):
            continue
        take = min(n_negatives, len(valid_neg))
        selected[row, 1 : take + 1] = rng.choice(valid_neg, size=take, replace=False)
    return selected


def test_flag_off_matches_original_sampler_and_rng_stream():
    gen = np.random.default_rng(123)
    n_lists, n_cols = 64, 201
    cand = gen.integers(1, 5000, size=(n_lists, n_cols), dtype=np.int32)
    cand[gen.random((n_lists, n_cols)) < 0.05] = 0  # some padding
    target_pos = gen.integers(0, n_cols, size=n_lists)
    cand[np.arange(n_lists), target_pos] = gen.integers(1, 5000, size=n_lists)
    label = np.zeros(cand.shape, dtype=np.int8)
    label[np.arange(n_lists), target_pos] = 1
    lists = {"cand": cand, "label": label, "target_pos": target_pos}
    for n_negatives in (5, 50, 300):
        rng_ref, rng_new, rng_all = (np.random.default_rng(7) for _ in range(3))
        ref = _reference_sampler(lists, n_negatives, rng_ref)
        np.testing.assert_array_equal(_sample_list_columns(lists, n_negatives, rng_new), ref)
        np.testing.assert_array_equal(
            _sample_list_columns(lists, n_negatives, rng_all, np.ones(cand.shape, dtype=bool)), ref)
        # The generators are left in the same state, so later draws (list order, next epoch) match too.
        assert rng_new.integers(1 << 62) == rng_ref.integers(1 << 62) == rng_all.integers(1 << 62)
