import numpy as np
import pandas as pd
import pytest
import torch

from cascade.pipelines.graph_encoder_diagnostic import (
    aligned_prediction, checkpoint_metrics, gradient_statistics, input_block_statistics,
    layer_weight_statistics, score_statistics, val_tables,
)
from cascade.pipelines.local_evidence_export import ranks_and_top10


def archive():
    ids = np.tile(np.r_[np.arange(1, 14), 0], (3, 1))
    score = np.tile(np.r_[np.arange(13, 0, -1, dtype=np.float32), -np.inf], (3, 1))
    target = np.array([1, 13, 99])
    rank, top = ranks_and_top10(score, ids, target)
    base = {'inst': np.arange(3), 'user': np.array([1, 1, 2]), 'target': target,
            'cand': ids, 'score': score, 'retrieval_rank': rank, 'top10': top}
    prediction = {key: base[key] for key in ('inst', 'user', 'target', 'top10')}
    prediction.update(logit=score, rank=rank)
    return base, prediction


def test_constant_score_offsets_leave_margins_and_ranking_but_centered_residual_zero():
    base, prediction = archive()
    original, arrays = score_statistics(base, prediction)
    shifted = {**prediction, 'logit': prediction['logit'] + np.array([[2.], [-3.], [4.]])}
    result, diagnostics = score_statistics(base, shifted)
    assert diagnostics['query_mean_residual'].tolist() == [2., -3., 4.]
    assert not diagnostics['centered_residual_rms'].any()
    for name in ('natural_target_mean_alternative_margin', 'natural_target_top10_boundary_margin'):
        assert np.allclose(arrays[name], diagnostics[name], equal_nan=True)
        assert np.isnan(diagnostics[name][-1])
    assert result['natural_target_events'] == 2 and result['all_query_events'] == 3
    assert result['natural_target_top10_boundary_margin_events'] == 2


def test_prefix_metrics_include_retrieval_misses_and_use_full_user_macro():
    base, prediction = archive()
    rows = checkpoint_metrics(base, prediction, np.array([False, True, True]))
    all_row = next(row for row in rows if row['group'] == 'all' and row['population'] == 'end_to_end')
    conditional = next(row for row in rows if row['group'] == 'all' and row['population'] == 'conditional')
    assert all_row['ndcg10'] == .25
    assert all_row['events'] == 3 and all_row['users'] == 2
    assert conditional['ndcg10'] == .5 and conditional['events'] == 2
    assert next(row for row in rows if row['group'] == 'unseen_before_T0' and row['population'] == 'end_to_end')['ndcg10'] == 0


def test_frozen_prediction_identity_and_padding_checks_fail_closed():
    base, prediction = archive()
    with pytest.raises(ValueError, match='inst identity'):
        aligned_prediction(base, {**prediction, 'inst': np.array([1, 2, 3])})
    invalid = prediction['logit'].copy(); invalid[:, -1] = 0
    with pytest.raises(ValueError, match='padding'):
        aligned_prediction(base, {**prediction, 'logit': invalid})
    invalid = prediction['logit'].copy(); invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        score_statistics(base, {**prediction, 'logit': invalid})


def test_input_blocks_use_unique_actual_candidates_and_missing_cosines_stay_nan():
    bank = np.zeros((4, 1154), np.float32)
    bank[1, 0] = bank[2, 0] = bank[3, 0] = 1
    bank[1, 384] = 1; bank[1, 1152] = 1
    bank[3, 385] = 1; bank[3, 1152] = 1
    first = np.array([0, 1, 4, 12])
    table = input_block_statistics(bank, np.array([0, 1, 1, 2, 3]), first, 10)
    all_leaf = table[(table.group == 'all') & (table.relation == 'leaf') & (table.population == 'all_candidates')].iloc[0]
    assert all_leaf.unique_items == 3 and all_leaf.neighbour_self_cosine_support == 2
    assert all_leaf.neighbour_self_cosine_mean == .5
    absent = table[(table.relation == 'store') & (table.population == 'neighbour_unavailable')]
    assert absent.neighbour_self_cosine_support.eq(0).all()
    assert absent.neighbour_self_cosine_mean.isna().all()
    new = table[(table.group == 'new_in_val') & (table.relation == 'leaf') & (table.population == 'neighbour_available')].iloc[0]
    assert new.unique_items == 1 and new.neighbour_self_cosine_mean == 0


def test_graph_column_norms_and_changes_keep_self_and_neighbour_blocks_distinct():
    initial = {
        'graph_encoder.0.weight': torch.ones(2, 1154),
        'graph_encoder.2.weight': torch.ones(2, 2),
        'graph_adapter.weight': torch.ones(2, 2),
    }
    state = {name: value.clone() for name, value in initial.items()}
    state['graph_encoder.0.weight'][:, 384:768] += 1
    rows = layer_weight_statistics(state, initial)
    table = pd.DataFrame(rows).set_index(['layer', 'input_block'])
    assert table.loc[('graph_encoder.0.weight', 'leaf'), 'change_from_initial_l2'] == pytest.approx(np.sqrt(768))
    assert table.loc[('graph_encoder.0.weight', 'self'), 'change_from_initial_l2'] == 0
    assert len(rows) == 6


def test_gradient_activity_treats_string_false_as_inactive_and_keeps_zero_active_steps():
    frame = pd.DataFrame({'stage': ['inner'] * 2, 'epoch': [1, 1],
        'graph_active': ['False', 'True'], 'graph_grad_l2': [0., 2.],
        'text_active': [True, True], 'text_grad_l2': [0., 3.],
        'head_active': [True, True], 'head_grad_l2': [.2, .4]})
    rows = gradient_statistics(frame)
    graph = next(row for row in rows if row['branch'] == 'graph')
    assert graph['steps'] == 2 and graph['active_steps'] == 1
    assert graph['nonzero_gradient_steps'] == 1 and graph['gradient_l2_mean'] == 1
    frame.loc[0, 'graph_grad_l2'] = 1
    with pytest.raises(ValueError, match='inactive'):
        gradient_statistics(frame)


def test_val_contrast_seed_averages_event_metrics_before_user_bootstrap():
    base, _ = archive()
    arrays = {'R6': np.array([[1., 13., np.inf]] * 3),
              'R7': np.array([[13., 1., np.inf], [13., 1., np.inf], [1., 1., np.inf]])}
    levels, differences, funnel = val_tables(arrays, base, {'all': np.ones(3, bool)}, n_resamples=30)
    row = differences[(differences.population == 'end_to_end') & (differences.metric == 'recall10')].iloc[0]
    assert row['diff'] == pytest.approx(1 / 12)
    assert row.n_users == 2
    assert levels[(levels.arm == 'R7') & (levels.population == 'end_to_end') & (levels.metric == 'recall10')].iloc[0].value == pytest.approx(1 / 3)
    assert funnel[funnel.outcome == 'retrieval_miss']['count'].eq(1).all()
    assert funnel[funnel.outcome == 'demotion']['count'].tolist() == [0, 0, 0, 1, 1, 0]
