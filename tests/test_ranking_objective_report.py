import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.eval.saved_results import digest
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.ranking_objective_report import (
    EXPANDED, PILOT, comparison_pairs, gate_values, replay_inner, replay_probes, summarise,
    validate_registration, verify_hashes,
)


def test_realised_methods_and_lock_follow_saved_gate_without_pruning_seeds():
    lock = {f'{method}/seed{seed}': {} for method in PILOT for seed in range(3)}
    validate_registration(list(PILOT), {'expanded': False}, lock)
    with pytest.raises(ValueError, match='method/seed'):
        validate_registration(list(PILOT), {'expanded': False}, {key: value for key, value in lock.items() if key != 'pairwise_R1/seed2'})
    with pytest.raises(ValueError, match='realised methods'):
        validate_registration(list(PILOT), {'expanded': True}, lock)
    expanded = {f'{method}/seed{seed}': {} for method in EXPANDED for seed in range(3)}
    validate_registration(list(EXPANDED), {'expanded': True}, expanded)
    assert ('pairwise_R6', 'pairwise_R7') in comparison_pairs(EXPANDED)
    assert ('pairwise_R2', 'pairwise_R4') in comparison_pairs(EXPANDED)
    assert len(comparison_pairs(PILOT)) == 3


def test_gate_uses_selected_seed_metrics_and_requires_cold_guard_and_two_wins():
    good = {0: (.11, .05), 1: (.12, .05), 2: (.09, .05)}
    assert gate_values(good, .1, .05)['expanded']
    assert not gate_values({0: (.4, .06), 1: (.09, .06), 2: (.09, .06)}, .1, .05)['expanded']
    assert not gate_values({0: (.11, .049), 1: (.12, .049), 2: (.09, .049)}, .1, .05)['expanded']
    assert not gate_values({0: (.1, .05), 1: (.1, .05), 2: (.1, .05)}, .1, .05)['expanded']


def test_generic_metric_report_averages_seed_events_before_user_intervals():
    base = np.array([[1, 20, np.inf, 2]], float)
    learned = np.array([[1, 11, np.inf, 11], [11, 1, np.inf, 11], [11, 11, np.inf, 1]], float)
    arrays = {'R0': {'rank': base}, 'bce_R1': {'rank': np.repeat(base, 3, axis=0)}, 'pairwise_R1': {'rank': learned}}
    users = np.array([1, 1, 2, 3])
    found = np.array([True, True, False, True])
    groups = {'all': np.ones(4, bool), 'empty': np.zeros(4, bool)}
    levels, differences, interactions = summarise(arrays, users, groups, found, n_resamples=20)
    row = levels[(levels.method == 'pairwise_R1') & (levels.group == 'all') &
                 (levels.population == 'end_to_end') & (levels.metric == 'recall10')].iloc[0]
    assert row.value == pytest.approx(2 / 9)
    assert row.recall200_ceiling_macro == pytest.approx(2 / 3)
    conditional = levels[(levels.method == 'pairwise_R1') & (levels.group == 'all') &
                        (levels.population == 'conditional') & (levels.metric == 'recall10')].iloc[0]
    assert conditional.value == pytest.approx(1 / 3)
    difference = differences[(differences.reference == 'bce_R1') & (differences.candidate == 'pairwise_R1') &
                             (differences.group == 'all') & (differences.population == 'end_to_end') &
                             (differences.metric == 'recall10')].iloc[0]
    assert difference['diff'] == pytest.approx(2 / 9 - .5)
    assert levels[levels.group == 'empty'].value.isna().all()
    assert levels[(levels.method == 'R0') & (levels.group == 'all')].training_seed_count.eq(0).all()
    assert interactions.empty


def test_expanded_report_includes_factorial_graph_contrast():
    base = np.array([[1, 20, np.inf]], float)
    arrays = {'R0': {'rank': base}, **{method: {'rank': np.repeat(base, 3, axis=0)} for method in EXPANDED}}
    levels, differences, interactions = summarise(arrays, np.arange(3), {'all': np.ones(3, bool)},
                                                  np.array([True, True, False]), n_resamples=5)
    assert len(interactions) == 4
    assert interactions['diff'].eq(0).all()
    assert differences[(differences.reference == 'pairwise_R6') & (differences.candidate == 'pairwise_R7')].shape[0] == 6


def test_retained_artifact_verification_rejects_changed_bytes_and_path_escape(tmp_path):
    path = tmp_path / 'epoch0.pth'
    path.write_bytes(b'fixed trained state')
    hashes = {path.name: digest(path)}
    assert verify_hashes(tmp_path, hashes, 'retained') == [path]
    path.write_bytes(b'mutated state')
    with pytest.raises(ValueError, match='identity changed'):
        verify_hashes(tmp_path, hashes, 'retained')
    with pytest.raises(ValueError, match='relative artifact path'):
        verify_hashes(tmp_path, {'../other': 'abc'}, 'retained')


def test_all_retained_inner_epochs_replay_and_earlier_equal_maximum_is_kept(tmp_path):
    ids = np.tile(np.arange(1, 13), (3, 1))
    targets = np.array([1, 12, 99])
    score = np.tile(np.arange(12, 0, -1, dtype=np.float32), (3, 1))
    rank, top = ranks_and_top10(score, ids, targets)
    base = {'inst': np.arange(3), 'user': np.arange(3), 'target': targets, 'cand': ids,
            'score': score, 'retrieval_rank': rank, 'top10': top}
    (tmp_path / 'inner').mkdir()
    for epoch in range(2):
        np.savez(tmp_path / f'inner/epoch{epoch}_tune.npz', inst=base['inst'], user=base['user'],
                 target=targets, logit=score, rank=rank, top10=top)
    history = [{'epoch': epoch, 'end_to_end_ndcg10': 1 / 3, 'unseen_before_T0_ndcg10': 0.} for epoch in range(2)]
    unseen = np.array([False, True, True])
    assert replay_inner(tmp_path, history, base, unseen, 0, 1 / 3) == (1 / 3, 0.)
    with pytest.raises(ValueError, match='earliest tuning maximum'):
        replay_inner(tmp_path, history, base, unseen, 1, 1 / 3)
    history[1]['unseen_before_T0_ndcg10'] = .1
    with pytest.raises(ValueError, match='unseen-target inner metric'):
        replay_inner(tmp_path, history, base, unseen, 0, 1 / 3)


def test_probe_replay_separates_common_offset_from_candidate_margins(tmp_path):
    ids = np.tile(np.arange(1, 13), (2, 1))
    targets = np.array([1, 12])
    score = np.tile(np.arange(12, 0, -1, dtype=np.float32), (2, 1))
    rank, top = ranks_and_top10(score, ids, targets)
    base = {'inst': np.arange(2), 'user': np.arange(2), 'target': targets, 'cand': ids, 'score': score}
    stages = np.array(['prefix', 'tune'])
    (tmp_path / 'probe').mkdir()
    np.savez(tmp_path / 'probe/epoch1.npz', **{key: value for key, value in base.items() if key != 'score'},
             base_score=score, logit=score + 2, rank=rank, top10=top, stage=stages,
             query_mean_residual=np.array([2., 2.]), centered_residual=np.zeros((2, 12)))
    summaries = [{'epoch': 1, 'split': split, 'logit_mean': 8.5, 'query_mean_residual_mean': 2.,
                  'centered_residual_std': 0., 'positive_margin_mean': margin, 'top10_boundary_margin_mean': boundary}
                 for split, margin, boundary in [('prefix', 6., 10.), ('tune', -6., -2.)]]
    (tmp_path / 'probe_history.json').write_text(json.dumps(summaries))
    assert replay_probes(tmp_path, [{'epoch': 1}], base, stages) == summaries
    summaries[0]['positive_margin_mean'] = 8.
    (tmp_path / 'probe_history.json').write_text(json.dumps(summaries))
    with pytest.raises(ValueError, match='summary differs from full scores'):
        replay_probes(tmp_path, [{'epoch': 1}], base, stages)
