import json

import numpy as np
import pandas as pd
import pytest

from cascade.eval.saved_results import digest
from cascade.pipelines.graph_family_report import (
    ARMS, COREVIEW, METADATA, PAIRS, compare_control, enabled_names,
    summarise, validate_protocol, validate_registration, verify_encoder_diagnostic,
)


NAMES = ['log_other_reviewers', 'log_witness_paths', 'normalised_path_strength',
         'same_leaf_history_fraction', 'same_store_history_fraction', 'supported_history_fraction']


def test_registered_masks_preserve_order_and_only_split_optional_six_slots():
    assert enabled_names('G0', NAMES) == []
    assert enabled_names('GM', NAMES) == list(METADATA)
    assert enabled_names('GC', NAMES) == list(COREVIEW)
    assert enabled_names('GMC', NAMES) == NAMES
    assert set(enabled_names('GM', NAMES)).isdisjoint(enabled_names('GC', NAMES))
    with pytest.raises(ValueError):
        enabled_names('GM', NAMES[:-1])
    with pytest.raises(ValueError):
        enabled_names('extra', NAMES)


def test_complete_registered_factorial_requires_every_seed_and_exact_arm_order():
    lock = {f'{arm}/seed{seed}': {} for arm in ARMS for seed in range(3)}
    meta = {'arms': list(ARMS), 'seeds': [0, 1, 2]}
    validate_registration(meta, lock)
    with pytest.raises(ValueError, match='twelve'):
        validate_registration(meta, {key: value for key, value in lock.items() if key != 'GC/seed2'})
    with pytest.raises(ValueError, match='arm/seed'):
        validate_registration({**meta, 'arms': ['G0', 'GC', 'GM', 'GMC']}, lock)
    assert len(PAIRS) == 6 and len(set(PAIRS)) == 6
    assert ('GM', 'GMC') in PAIRS and ('GC', 'GMC') in PAIRS


def test_protocol_checks_named_masks_standardisation_and_all_registered_contrasts():
    protocol = {'feature_families': {'metadata': list(METADATA), 'coreview': list(COREVIEW)},
                'enabled_features': {arm: enabled_names(arm, NAMES) for arm in ARMS},
                'mask_after_standardisation': True, 'common_features_unchanged': True,
                'pairwise_contrasts': [list(pair) for pair in PAIRS], 'interaction': 'GMC-GM-GC+G0'}
    validate_protocol(protocol, NAMES)
    with pytest.raises(ValueError, match='preprocessing/common'):
        validate_protocol({**protocol, 'mask_after_standardisation': False}, NAMES)
    changed = dict(protocol['enabled_features']); changed['GM'] = ['log_witness_paths']
    with pytest.raises(ValueError, match='masks'):
        validate_protocol({**protocol, 'enabled_features': changed}, NAMES)


def test_family_metrics_average_events_before_user_bootstrap_and_keep_empty_groups():
    baseline = np.array([[1., 20., np.inf, 2.]])
    learned = np.array([[1., 11., np.inf, 11.], [11., 1., np.inf, 11.], [11., 11., np.inf, 1.]])
    arrays = {'R0': {'rank': baseline}, **{arm: {'rank': learned.copy()} for arm in ARMS}}
    users = np.array([1, 1, 2, 3]); found = np.array([True, True, False, True])
    levels, differences, interactions = summarise(
        arrays, users, {'all': np.ones(4, bool), 'empty': np.zeros(4, bool)}, found, n_resamples=20)
    row = levels[(levels.method == 'GM') & (levels.group == 'all') &
                 (levels.population == 'end_to_end') & (levels.metric == 'recall10')].iloc[0]
    assert row.value == pytest.approx(2/9)
    assert row.recall200_ceiling_macro == pytest.approx(2/3)
    conditional = levels[(levels.method == 'GM') & (levels.group == 'all') &
                        (levels.population == 'conditional') & (levels.metric == 'recall10')].iloc[0]
    assert conditional.value == pytest.approx(1/3) and conditional.events == 3
    assert len(levels[(levels.group == 'empty') & levels.value.isna()]) > 0
    assert len(differences[(differences.group == 'all') & (differences.population == 'end_to_end') &
                           (differences.metric == 'ndcg10')]) == 6
    assert interactions['contrast'].eq('GMC-GM-GC+G0').all()
    assert len(interactions) == 12


def test_direct_factorial_contrast_is_not_a_combined_arm_winner_claim():
    rank = {'G0': [[11., 11.]], 'GM': [[1., 11.]], 'GC': [[11., 1.]], 'GMC': [[1., 1.]]}
    arrays = {'R0': {'rank': np.array([[11., 11.]])},
              **{arm: {'rank': np.repeat(np.array(rows), 3, axis=0)} for arm, rows in rank.items()}}
    _, _, interaction = summarise(arrays, np.array([1, 2]), {'all': np.ones(2, bool)}, np.ones(2, bool), n_resamples=20)
    row = interaction[(interaction.population == 'end_to_end') & (interaction.metric == 'ndcg10')].iloc[0]
    assert row['diff'] == pytest.approx(0)
    assert row.ci_low == row.ci_high == 0


def control_history():
    return [{'epoch': 0, 'loss': None, 'end_to_end_ndcg10': .1, 'unseen_before_T0_ndcg10': .2, 'conditional_ndcg10': .3},
            {'epoch': 1, 'loss': .7, 'end_to_end_ndcg10': .2, 'unseen_before_T0_ndcg10': .3, 'conditional_ndcg10': .4}]


def test_control_reproduction_is_exact_for_selector_and_ranking_but_tolerant_for_float_loss():
    old = control_history(); actual = [dict(row) for row in old]
    actual[1]['loss'] += 1e-7
    compare_control(actual, 1, old, 1)
    with pytest.raises(ValueError, match='selected epoch'):
        compare_control(actual, 0, old, 1)
    actual[1]['end_to_end_ndcg10'] += 1e-9
    with pytest.raises(ValueError, match='ranking curve'):
        compare_control(actual, 1, old, 1)
    actual = [dict(row) for row in old]; actual[1]['loss'] += 1e-3
    with pytest.raises(ValueError, match='loss'):
        compare_control(actual, 1, old, 1)


def test_frozen_encoder_provenance_checks_outputs_and_rejects_new_training(tmp_path):
    directory = tmp_path/'encoder_diagnostic'; directory.mkdir()
    source = directory/'source.py'; source.write_text('frozen encoder diagnostic\n')
    results = directory/'metrics.csv'; results.write_text('value\n1\n')
    proof = {'status': 'complete', 'training_performed': False, 'test_predictions_loaded': False,
             'source_sha256': digest(source), 'input_sha256': {str(source): digest(source)},
             'generated_sha256': {'source.py': digest(source), 'metrics.csv': digest(results)}}
    path = directory/'provenance.json'; path.write_text(json.dumps(proof))
    assert results in verify_encoder_diagnostic(tmp_path)
    results.write_text('value\n2\n')
    with pytest.raises(ValueError, match='output changed'):
        verify_encoder_diagnostic(tmp_path)
    proof['training_performed'] = True; path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='training/exposure'):
        verify_encoder_diagnostic(tmp_path)
