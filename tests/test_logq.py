import numpy as np
import torch
import torch.nn.functional as F

from cascade.models.retriever import TwoTowerRetriever, inbatch_softmax_loss


def test_logq_subtracts_target_log_q():
    torch.manual_seed(0)
    u, v = F.normalize(torch.randn(3, 4), dim=-1), F.normalize(torch.randn(3, 4), dim=-1)
    targets = torch.tensor([1, 2, 3])
    log_q = torch.log(torch.tensor([1.0, 0.5, 0.3, 0.2]))
    tau = 0.05
    expected_logits = u @ v.T / tau - log_q[targets].unsqueeze(0)  # column j corrected by log q(target_j)
    expected = F.cross_entropy(expected_logits, torch.arange(3))
    assert torch.allclose(inbatch_softmax_loss(u, v, targets, tau, log_q), expected)
    no_q = F.cross_entropy(u @ v.T / tau, torch.arange(3))
    assert torch.allclose(inbatch_softmax_loss(u, v, targets, tau, None), no_q)


def test_duplicate_targets_masked():
    torch.manual_seed(0)
    u, v = F.normalize(torch.randn(3, 4), dim=-1), F.normalize(torch.randn(3, 4), dim=-1)
    targets = torch.tensor([7, 7, 5])  # pairs 0 and 1 share a target
    tau = 1.0
    logits = u @ v.T / tau
    logits[0, 1] = logits[1, 0] = float("-inf")
    expected = F.cross_entropy(logits, torch.arange(3))
    assert torch.allclose(inbatch_softmax_loss(u, v, targets, tau, None), expected)


def _model(user="transformer", item="text"):
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((11, 384)).astype(np.float32)
    emb[0] = 0
    return TwoTowerRetriever(item, user, emb, 10, np.ones(11, dtype=bool), max_len=5).eval()


def test_user_tower_causal_and_padding():
    m = _model()
    h1 = torch.tensor([[0, 0, 3, 4, 5]])
    h2 = torch.tensor([[0, 0, 3, 4, 9]])  # differs only in the last position
    s1, s2 = m.user_sequence(h1), m.user_sequence(h2)
    assert torch.allclose(s1[:, :4], s2[:, :4], atol=1e-6), "earlier positions must not see later items"
    assert not torch.isnan(s1).any(), "padded positions must not produce NaN"
    assert torch.allclose(m.user_vectors(h1).norm(dim=-1), torch.ones(1))


def test_meanpool_ignores_padding():
    m = _model(user="meanpool")
    u = m.user_vectors(torch.tensor([[0, 0, 0, 2, 3]]))
    e = m.item_tower(torch.tensor([2, 3])).mean(0)
    assert torch.allclose(u[0], F.normalize(e, dim=0), atol=1e-6)


def test_id_only_unseen_items_zero():
    rng = np.random.default_rng(0)
    seen = np.ones(11, dtype=bool)
    seen[4] = False
    m = TwoTowerRetriever("id_only", "transformer", None, 10, seen, max_len=5)
    assert m.item_tower(torch.tensor([4])).abs().sum() == 0
    assert m.item_tower(torch.tensor([0])).abs().sum() == 0
