"""Unit checks for the isolated random-mixing study helpers (synthetic data only)."""
import numpy as np
import pytest
import torch

from cascade.pipelines.graph_latent_study import negative_plan
from cascade.pipelines.random_mix import MixedView, head_inputs, insertion_slots, list_auc


def toy_archive(n_rows=6, width=60, seed=0):
    rng = np.random.default_rng(seed)
    cand = np.stack([rng.permutation(np.arange(1, 500))[:width] for _ in range(n_rows)]).astype(np.int32)
    target = np.where(np.arange(n_rows) % 2 == 0, cand[:, 3], 900+np.arange(n_rows))  # even rows natural hits
    return cand, target.astype(np.int64)


def test_mixed_view_passes_base_rows_and_overrides_slot():
    cand, target = toy_archive()
    src = np.array([1, 3, 5]); slot = np.array([0, 59, 17])
    view = MixedView(cand, src, slot, target[src])
    assert view.shape == (9, 60) and len(view) == 9
    np.testing.assert_array_equal(view[np.array([0, 2])], cand[[0, 2]])
    mixed = view[np.array([6, 7, 8])]
    for i, (s, j) in enumerate(zip(src, slot)):
        expected = cand[s].copy(); expected[j] = target[s]
        np.testing.assert_array_equal(mixed[i], expected)
    np.testing.assert_array_equal(view[np.int64(7)], mixed[1])  # scalar access returns one row
    assert cand[3, 59] != target[3]  # base untouched
    with pytest.raises(IndexError):
        view[np.array([9])]


def test_mixed_view_without_slot_copies_source_rows():
    score = np.arange(12, dtype=np.float32).reshape(4, 3)
    view = MixedView(score, np.array([2, 0]))
    np.testing.assert_array_equal(view[np.array([4, 5, 1])], score[[2, 0, 1]])


def test_insertion_slots_deterministic_in_range_and_seeded():
    ids = np.arange(5000)+123
    a = insertion_slots(ids, 0, np.full(5000, 200))
    np.testing.assert_array_equal(a, insertion_slots(ids, 0, np.full(5000, 200)))
    assert a.min() >= 0 and a.max() <= 199
    assert abs(a.mean()-99.5) < 4  # uniform over 0..199
    assert (a != insertion_slots(ids, 1, np.full(5000, 200))).mean() > .9
    short = insertion_slots(ids[:50], 0, np.full(50, 60))
    assert short.max() < 60
    assert a[0] == np.random.default_rng([0, 123]).integers(0, 200)


def test_negative_plan_on_mixed_rows_puts_target_first():
    cand, target = toy_archive()
    src = np.array([1, 3, 5]); slot = np.array([0, 59, 17])
    archive = {'cand': MixedView(cand, src, slot, target[src]), 'target': MixedView(target, src)}
    eligible = np.r_[[0, 2, 4], 6+np.arange(3)]
    rows, columns = negative_plan(archive, eligible, seed=0, epochs=2)
    for e in range(2):
        assert sorted(rows[e]) == sorted(eligible)
        lists = archive['cand'][rows[e]]
        picked = np.take_along_axis(lists, columns[e].astype(np.int64), axis=1)
        np.testing.assert_array_equal(picked[:, 0], archive['target'][rows[e]])
        assert not (picked[:, 1:] == picked[:, :1]).any()
        assert all(len(set(r)) == 51 for r in picked)


def test_list_auc_hand_computed():
    values = np.array([[3., 1., 2., 3., 0.], [0., 1., 2., 3., 4.]])
    valid = np.array([[True]*5, [True, True, True, True, False]])
    auc = list_auc(values, np.array([0, 0]), valid)
    # row 0: positive 3 vs {1,2,3,0}: 3 greater + one tie -> 3.5/4; row 1: positive 0 vs {1,2,3}: 0
    np.testing.assert_allclose(auc, [3.5/4, 0.])


def test_head_inputs_match_model_definitions():
    text = torch.zeros(6, 384); text[1, 0] = 1; text[2, 1] = 1; text[3, 0] = text[3, 1] = 2**-.5; text[4, 2] = 1; text[5, 0] = 1
    leaf = torch.tensor([0, 1, 2, 1, 0, 3]); store = torch.tensor([0, 7, 8, 7, 9, 0])
    cand = np.array([[3, 4, 5]]); history = np.array([[0, 1, 2]])
    out = head_inputs(text, leaf, store, cand, history)
    np.testing.assert_allclose(out['cosine_to_history'][0], [1., 0., 2**-.5], atol=1e-6)
    np.testing.assert_array_equal(out['leaf_match'][0], [False, False, False])  # last history leaf = 2
    np.testing.assert_array_equal(out['leaf_known'][0], [True, False, True])
    np.testing.assert_array_equal(out['store_match'][0], [True, False, False])
    np.testing.assert_array_equal(out['store_known'][0], [True, True, False])
