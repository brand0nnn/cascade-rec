"""Feature exclusion and the top-ten boundary are experimental validity checks."""
import numpy as np
import pytest
import torch
from cascade.models.ranker_mlp import FEATURE_GROUPS, MLPRanker, RankerFeatureSpec
from cascade.eval.ranker import candidate_ranks, ranker_comparison


def make_spec(groups):
    return RankerFeatureSpec({n: 8 for n in MLPRanker.categorical_names}, item_text_dim=3, feature_groups=groups)


def batch():
    result = {n: torch.tensor([1, 2, 3]) for n in MLPRanker.categorical_names}
    result.update(numeric=torch.randn(3, 4), history_item_ids=torch.tensor([[0, 1], [1, 2], [2, 3]]),
                  history_leaf_ids=torch.tensor([[0, 1], [1, 2], [2, 3]]), item_text=torch.randn(3, 3))
    return result


def test_score_only_ignores_every_other_input():
    model = MLPRanker(make_spec(('retrieval',))).eval()
    a = batch(); b = {k: torch.zeros_like(v) for k, v in a.items()}
    b['numeric'][:, 0] = a['numeric'][:, 0]
    torch.testing.assert_close(model(a), model(b))
    torch.testing.assert_close(model(a), model({'numeric': a['numeric']}))
    assert not model.embeddings and model.text_projection is None


@pytest.mark.parametrize('groups', [FEATURE_GROUPS[:i] for i in (1, 2, 3, 4)])
def test_rung_gradient_and_disabled_inputs(groups):
    model = MLPRanker(make_spec(groups), dropout=0).eval()
    a = batch(); expected = model(a)
    b = {k: v.clone() for k, v in a.items()}
    if 'attributes' not in groups:
        for name in ('item_id', 'leaf_category', 'top_category', 'store', 'price_bucket'):
            b[name].fill_(7)
    if 'statistics' not in groups:
        for name in ('review_count_bucket', 'rating_bucket', 'item_age_bucket'):
            b[name].fill_(7)
    torch.testing.assert_close(model(b), expected)
    expected.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_full_default_preserves_checkpoint_layout():
    torch.manual_seed(2); a = MLPRanker(make_spec(FEATURE_GROUPS)).eval()
    default_spec = RankerFeatureSpec({n: 8 for n in MLPRanker.categorical_names}, item_text_dim=3)
    b = MLPRanker(default_spec).eval(); b.load_state_dict(a.state_dict(), strict=True)
    data = batch(); torch.testing.assert_close(a(data), b(data))


def test_rescue_demotion_and_miss_denominators():
    # One of every outcome. Last rank improves inside the top 10: retained, not rescued.
    lists = {'target_pos': np.array([0, 14, 3, 1, 29]), 'forced': np.array([True, False, False, False, False]),
             'user': np.array([1, 1, 2, 2, 3])}
    ranks = np.array([np.inf, 6, 28, 1, 35])
    result = ranker_comparison(ranks, lists)
    assert result['outcomes'] == {
        'retrieval_miss': {'count': 1, 'fraction_of_all_lists': 0.2},
        'unrescued': {'count': 1, 'fraction_of_all_lists': 0.2},
        'demoted': {'count': 1, 'fraction_of_all_lists': 0.2},
        'retained': {'count': 1, 'fraction_of_all_lists': 0.2},
        'rescued': {'count': 1, 'fraction_of_all_lists': 0.2},
    }
    assert result['ranker']['recall@10']['micro'] == 0.4
    assert result['ranker']['recall@10']['macro'] == pytest.approx(1/3)


def test_target_ties_padding_and_forced_misses():
    lists = {'target_pos': np.array([0, 1]), 'forced': np.array([False, True]),
             'cand': np.array([[1, 2, 0], [1, 2, 3]])}
    ranks = candidate_ranks(np.array([[0.4, 0.4, 999], [0.1, 999, 0.2]]), lists)
    np.testing.assert_equal(ranks, np.array([2, np.inf]))


def test_score_only_uses_float32_before_normalization():
    from types import SimpleNamespace
    from cascade.data.ranker_features import RankerFeatureBuilder
    builder = SimpleNamespace(score_mean=0.1234567, score_std=0.01234567)
    lists = {'cand': np.array([[1, 2]]), 'score': np.array([[0.231, 0.219]], dtype=np.float16)}
    r, c = np.array([0, 0]), np.array([0, 1])
    output = RankerFeatureBuilder.build(builder, lists, r, c, score_only=True)
    expected = (lists['score'][r, c].astype(np.float32) - builder.score_mean) / builder.score_std
    np.testing.assert_array_equal(output['numeric'][:, 0], expected)
