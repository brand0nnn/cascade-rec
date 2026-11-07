"""Permutation helpers for the exploratory Val permutation-importance analysis."""
import numpy as np
import torch

from cascade.analysis.permutation_importance import permutation_index, permuted_batch


def _lop():
    return np.repeat(np.arange(5), [3, 4, 1, 6, 2])


def test_candidate_permutation_stays_within_list_and_is_a_permutation():
    lop = _lop()
    src = permutation_index("candidate", lop, 5, np.random.default_rng(0))
    assert sorted(src.tolist()) == list(range(len(lop)))
    assert np.array_equal(lop[src], lop)


def test_user_permutation_takes_another_lists_first_pair():
    lop = _lop()
    src = permutation_index("user", lop, 5, np.random.default_rng(1))
    first = np.searchsorted(lop, np.arange(5))
    donors = lop[src]
    assert np.all(donors != lop)  # derangement: no list keeps its own values
    assert np.all(np.isin(src, first))
    for r in range(5):  # all pairs of a list share one donor
        assert len(set(donors[lop == r])) == 1


def test_permuted_batch_changes_only_the_group_columns():
    numeric = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    batch = {"numeric": numeric, "item_text": torch.eye(3)}
    src = torch.tensor([2, 0, 1])
    out = permuted_batch(batch, "retrieval_score", src)
    assert torch.equal(out["numeric"][:, 0], numeric[src, 0])
    assert torch.equal(out["numeric"][:, 1:], numeric[:, 1:])
    assert out["item_text"] is batch["item_text"]
    assert torch.equal(batch["numeric"], torch.arange(12, dtype=torch.float32).reshape(3, 4))  # input untouched
