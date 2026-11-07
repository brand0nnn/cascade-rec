from types import SimpleNamespace

import numpy as np
import pytest
import torch
from recbole.config import Config

from cascade.models.local_sasrec import ContentItems, ContentSASRec, PriorItemSampler, backbone_digest
from cascade.models.recbole_baselines import SASRecBCE, recbole_config_dict


def dataset():
    return SimpleNamespace(uid_field='user_id', iid_field='item_id', user_num=3, item_num=7,
        inter_feat={'user_id': torch.tensor([1, 2]), 'item_id': torch.tensor([1, 2])},
        num=lambda field: 7 if field == 'item_id' else 3)


def test_restricted_sampler_never_returns_future_items_or_user_positives():
    sampler = PriorItemSampler(dataset(), [1, 2, 3]).set_phase('train')
    np.random.seed(0)
    negatives = sampler.sample_by_user_ids(np.array([1, 2]), np.array([1, 2]), 500).numpy()
    assert set(negatives) == {1, 2, 3}
    assert np.all(negatives[::2] != 1) and np.all(negatives[1::2] != 2)
    assert sampler.returned == 1000
    with pytest.raises(ValueError, match='exhausts'):
        PriorItemSampler(dataset(), [1])


def test_future_content_cannot_affect_training_on_prior_items():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(7, 384)).astype(np.float32)
    altered = features.copy()
    altered[4:] *= 100
    a, b = ContentItems(features, 64, 0), ContentItems(altered, 64, 0)
    items = torch.tensor([1, 2, 3])
    a(items).sum().backward()
    b(items).sum().backward()
    assert not a.features.requires_grad
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x.grad, y.grad)
    assert torch.equal(a(torch.tensor([0])), torch.zeros(1, 64))
    with torch.no_grad():
        a.eval()
        b.eval()
        assert not torch.equal(a(torch.tensor([5])), b(torch.tensor([5])))
    assert 'features' not in a.state_dict()


def test_content_id_masks_unseen_residuals_at_training_and_inference():
    features=np.ones((5,384),np.float32)
    residual=torch.nn.Embedding(5,64)
    seen=np.array([0,1,1,0,0])
    model=ContentItems(features,64,0,residual,seen)
    actual=model(torch.tensor([3]))
    expected=model.projection(model.features[torch.tensor([3])])
    assert torch.equal(actual,expected)
    actual.sum().backward()
    assert not residual.weight.grad.any()
    with torch.no_grad():
        model.eval()
        assert torch.allclose(model.weight[3],expected[0],atol=1e-6)


def test_content_replacement_preserves_sequence_initialisation_and_rng():
    cfg = recbole_config_dict('sequential', '.', 'local_test')
    cfg.update(n_layers=2, n_heads=1, hidden_size=64, inner_size=64,
        hidden_dropout_prob=.2, attn_dropout_prob=.2, hidden_act='relu',
        layer_norm_eps=1e-8, initializer_range=.02, loss_type='BPR', seed=0)
    cfg = Config(model=SASRecBCE, dataset='local_test', config_dict=cfg)
    cfg['device'] = torch.device('cpu')
    torch.manual_seed(0)
    a = SASRecBCE(cfg, dataset())
    after_id = torch.get_rng_state()
    torch.manual_seed(0)
    b = ContentSASRec(cfg, dataset(), np.ones((7, 384), np.float32))
    assert backbone_digest(a) == backbone_digest(b)
    assert torch.equal(after_id, torch.get_rng_state())
    b.eval()
    with torch.no_grad():
        output = b.forward(torch.tensor([[1, 2, 0]]), torch.tensor([2]))
        assert output.shape == (1, 64) and torch.isfinite(output).all()
        assert b.item_embedding.weight.shape == (7, 64)
