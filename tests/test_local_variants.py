from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from recbole.config import Config

from cascade.models.local_variants import MultiInterestSASRec, metadata_graph_features
from cascade.models.recbole_baselines import SASRecBCE, recbole_config_dict


def test_metadata_neighbours_exclude_future_items_self_and_unknown_relations():
    items = pd.DataFrame({'item': [1, 2, 3, 4], 'leaf_cat': ['Cases', 'Cases', 'Cases', ''],
        'store': ['A', 'A', 'B', ''], 'first_ts': [1, 2, 11, 12]})
    vectors = np.arange(20, dtype=np.float32).reshape(5, 4)
    features, neighbours = metadata_graph_features(items, vectors, cutoff=10, limit=2)
    assert set(neighbours[3, 0]) == {1, 2}
    assert set(neighbours[1, 0]) == {0, 2}
    assert not neighbours[4].any()
    assert not neighbours[3, 1].any()
    assert neighbours.max() == 2
    assert np.allclose(features[3, 4:8], vectors[[1, 2]].mean(0))
    altered = vectors.copy()
    altered[3:] += 100
    other, _ = metadata_graph_features(items, altered, cutoff=10, limit=2)
    assert np.array_equal(features[:3], other[:3])


def test_two_interest_target_scores_agree_with_full_catalogue_scores():
    cfg = recbole_config_dict('sequential', '.', 'local_test')
    cfg.update(n_layers=2, n_heads=1, hidden_size=64, inner_size=64,
        hidden_dropout_prob=.2, attn_dropout_prob=.2, hidden_act='relu',
        layer_norm_eps=1e-8, initializer_range=.02, loss_type='BPR', seed=0)
    cfg = Config(model=SASRecBCE, dataset='local_test', config_dict=cfg)
    cfg['device'] = torch.device('cpu')
    dataset = SimpleNamespace(num=lambda field: 7)
    features = np.random.default_rng(0).normal(size=(7, 384)).astype(np.float32)
    model = MultiInterestSASRec(cfg, dataset, features).eval()
    batch = {model.ITEM_SEQ: torch.tensor([[1, 2, 0], [2, 0, 0]]),
        model.ITEM_SEQ_LEN: torch.tensor([2, 1]), model.ITEM_ID: torch.tensor([3, 4])}
    with torch.no_grad():
        states = model.forward(batch[model.ITEM_SEQ], batch[model.ITEM_SEQ_LEN])
        assert states.shape == (2, 2, 64)
        all_scores = model.full_sort_predict(batch)
        assert torch.allclose(model.predict(batch), all_scores[torch.arange(2), batch[model.ITEM_ID]])
        expected = torch.stack([states[:, h] @ model.item_embedding.weight.T for h in range(2)]).amax(0)
        assert torch.allclose(all_scores, expected)
