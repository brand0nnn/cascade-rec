"""Unit checks for the exploratory last-item / mean-pool scorers (validation-only analysis)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade.analysis.last_item import LastItemScorer, MeanPoolScorer
from cascade.eval.harness import InstanceBatch


class _Fake(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self._V = V

    def all_item_vectors(self):
        return self._V


def _batch(hist):
    n = len(hist)
    z = np.zeros(n, dtype=np.int64)
    return InstanceBatch(inst=z, users=z, targets=z, ts=z, hist_len=z, hist=torch.tensor(hist), mask_rows=z[:0], mask_items=z[:0])


def test_last_item_and_mean_pool():
    V = F.normalize(torch.randn(6, 4), dim=-1)
    V[0] = 0
    b = _batch([[0, 0, 2, 5], [0, 0, 0, 3]])
    s = LastItemScorer(_Fake(V)).score_all(b)
    assert torch.allclose(s, torch.stack([V[5] @ V.T, V[3] @ V.T]))
    m = MeanPoolScorer(_Fake(V)).score_all(b)
    u0 = F.normalize((V[2] + V[5]) / 2, dim=0)
    assert torch.allclose(m[0], u0 @ V.T, atol=1e-6)
    assert torch.allclose(m[1], V[3] @ V.T, atol=1e-6)  # single-item history: mean-pool == last-item
