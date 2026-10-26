"""Small deterministic checks for the project's DIN attention adaptation."""
import pytest
import torch
from torch import nn

from cascade.models.ranker_din import DINAttention, DINRanker
from cascade.models.ranker_mlp import FEATURE_GROUPS, MLPRanker, RankerFeatureSpec


def make_spec(groups=("retrieval", "matching")):
    return RankerFeatureSpec(
        {name: 8 for name in MLPRanker.categorical_names},
        history_length=4,
        item_text_dim=3,
        feature_groups=groups,
    )


def make_batch():
    batch = {name: torch.tensor([1, 2, 3]) for name in MLPRanker.categorical_names}
    batch.update(
        item_id=torch.tensor([5, 6, 7]),
        leaf_category=torch.tensor([2, 3, 4]),
        history_item_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 2, 4], [0, 0, 0, 0]]),
        history_leaf_ids=torch.tensor([[0, 1, 2, 3], [0, 0, 3, 4], [0, 0, 0, 0]]),
        numeric=torch.tensor([[0.2, 1.0, 0.0, 0.3], [-0.4, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 0.0]]),
        item_text=torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.4, 0.5], [0.6, -0.2, 0.1]]),
    )
    return batch


def test_local_activation_matches_hand_computed_signed_weighted_sum():
    attention = DINAttention(embedding_dim=1, hidden_dims=(1,))
    with torch.no_grad():
        attention.activation[0].weight.copy_(torch.tensor([[1.0, 2.0, -1.0, 0.5]]))
        attention.activation[0].bias.fill_(-1.0)
        attention.activation[2].weight.fill_(-2.0)
        attention.activation[2].bias.fill_(5.0)

    # q=2; [q,h,q-h,q*h] gives [2,1,1,2] and [2,3,-1,6].
    # Hidden activations are ReLU(3)=3 and ReLU(11)=11; coefficients
    # are -2*3+5=-1 and -2*11+5=-17. The sum is -1*1-17*3=-52.
    pooled, weights = attention(
        torch.tensor([[2.0]]),
        torch.tensor([[[1.0], [3.0]]]),
        torch.tensor([[True, True]]),
    )
    torch.testing.assert_close(weights, torch.tensor([[-1.0, -17.0]]), rtol=0, atol=0)
    torch.testing.assert_close(pooled, torch.tensor([[-52.0]]), rtol=0, atol=0)


def test_unnormalized_pool_doubles_when_a_history_item_is_duplicated():
    attention = DINAttention(embedding_dim=1, hidden_dims=())
    with torch.no_grad():
        attention.activation[0].weight.zero_()
        attention.activation[0].bias.fill_(1.0)
    query = torch.tensor([[2.0]])
    single, _ = attention(query, torch.tensor([[[3.0]]]), torch.ones(1, 1, dtype=torch.bool))
    doubled, weights = attention(query, torch.tensor([[[3.0], [3.0]]]), torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(weights, torch.ones(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(single, torch.tensor([[3.0]]), rtol=0, atol=0)
    torch.testing.assert_close(doubled, 2 * single, rtol=0, atol=0)


def test_padding_is_ignored_and_fully_masked_history_is_zero():
    torch.manual_seed(3)
    attention = DINAttention(embedding_dim=2, hidden_dims=(3,))
    query = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    history = torch.tensor([[[2.0, 4.0], [100.0, 200.0], [5.0, 6.0]],
                            [[-100.0, 200.0], [7.0, 8.0], [3.0, 4.0]]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    pooled, weights = attention(query, history, mask)
    changed = history.clone()
    changed[~mask] = -1000.0
    changed_pooled, changed_weights = attention(query, changed, mask)
    torch.testing.assert_close(changed_pooled, pooled, rtol=0, atol=0)
    torch.testing.assert_close(changed_weights, weights, rtol=0, atol=0)
    assert torch.count_nonzero(weights[~mask]) == 0
    assert torch.count_nonzero(pooled[1]) == 0
    assert torch.isfinite(pooled).all()


def test_zero_length_history_returns_zero_pool():
    attention = DINAttention(embedding_dim=2, hidden_dims=(3,))
    pooled, weights = attention(torch.zeros(2, 2), torch.empty(2, 0, 2), torch.empty(2, 0, dtype=torch.bool))
    assert pooled.shape == (2, 2)
    assert weights.shape == (2, 0)
    torch.testing.assert_close(pooled, torch.zeros(2, 2), rtol=0, atol=0)


@pytest.mark.parametrize("query_shape,history_shape,mask_shape", [
    ((2, 3), (2, 4, 2), (2, 4)),
    ((3, 2), (2, 4, 2), (2, 4)),
    ((2, 2), (2, 4), (2, 4)),
    ((2, 2), (2, 4, 3), (2, 4)),
    ((2, 2), (2, 4, 2), (2, 3)),
    ((2, 2), (2, 4, 2), (2, 4, 1)),
])
def test_attention_rejects_misaligned_shapes(query_shape, history_shape, mask_shape):
    attention = DINAttention(embedding_dim=2, hidden_dims=(3,))
    with pytest.raises(ValueError, match="must align"):
        attention(torch.zeros(query_shape), torch.zeros(history_shape), torch.ones(mask_shape, dtype=torch.bool))


@pytest.mark.parametrize("groups", [("retrieval", "matching"), FEATURE_GROUPS])
def test_din_forward_and_backward_are_finite(groups):
    torch.manual_seed(11)
    model = DINRanker(make_spec(groups), dropout=0, attention_hidden_dims=(5, 3))
    batch = make_batch()
    logits = model(batch)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    pooled, weights = model.history_attention(batch)
    assert torch.count_nonzero(pooled[2]) == 0
    assert torch.count_nonzero(weights[2]) == 0
    assert torch.count_nonzero(model.embeddings["item_id"].weight.grad[0]) == 0


@pytest.mark.parametrize("groups", [("retrieval", "matching"), FEATURE_GROUPS])
def test_shared_mlp_parameters_have_identical_initialization(groups):
    torch.manual_seed(9)
    mlp = MLPRanker(make_spec(groups), dropout=0)
    torch.manual_seed(9)
    din = DINRanker(make_spec(groups), dropout=0, attention_hidden_dims=(5, 3))
    shared = mlp.state_dict()
    din_state = din.state_dict()
    assert set(shared) < set(din_state)
    assert all(name.startswith("attention.") for name in set(din_state) - set(shared))
    for name, parameter in shared.items():
        torch.testing.assert_close(din_state[name], parameter, rtol=0, atol=0)


@pytest.mark.parametrize("groups", [("retrieval",), ("retrieval", "attributes", "statistics")])
def test_din_requires_matching_history_features(groups):
    with pytest.raises(ValueError, match="matching feature group"):
        DINRanker(make_spec(groups))
