"""Protect study comparisons against incompatible cohorts and false promotion."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from cascade.eval.ranker_policy_report import run, summarize


def evidence(seeds=(0,)):
    groups = pd.DataFrame(dict(inst=[1, 2, 3, 4], user=[1, 1, 2, 3],
        item_type=['new-in-period', 'new-in-period', 'warm-head', 'warm-tail'],
        history_length=['3-5'] * 4, history_diversity=['3-5'] * 4,
        validation_window=['days 0-89'] * 4, new_at_val_entry=[True, True, False, False]))
    base = dict(inst=groups.inst.to_numpy(), user=groups.user.to_numpy(),
                candidate_counts=np.array([50, 100, 200, 500]), full_catalogue_ranks=np.array([2., 8., 5., 20.]),
                ranks=np.tile([2., 8., 5., 20.], (4, 1)),
                retrieval_ranks=np.tile([2., 8., 5., 20.], (4, 1)), target_pos=np.array([1, 7, 4, 19]))
    data = {arm: {s: copy.deepcopy(base) for s in seeds} for arm in 'ABCD'}
    return groups, data


def test_pilot_cannot_promote_and_population_mismatch_is_rejected():
    groups, data = evidence()
    data['B'][0]['ranks'][:, 0] = 1
    result = summarize(data, groups, resamples=20)
    assert result['qualified_fixes'] == []
    assert result['integration_policy'] == 'A'
    data['B'][0]['user'][0] = 99
    with pytest.raises(ValueError, match='alignment'):
        summarize(data, groups, resamples=20)


def test_three_seed_improvement_preserves_guard_and_interaction():
    groups, data = evidence((0, 1, 2))
    for seed in (0, 1, 2):
        data['B'][seed]['ranks'][:] = np.tile([1., 1., 1., 1.], (4, 1))
    result = summarize(data, groups, resamples=30)
    assert result['qualified_fixes'] == ['B']
    assert result['integration_policy'] == 'B'
    comparisons = result['comparisons'].query("axis == 'overall'").set_index('comparison')
    assert comparisons.loc['(D-C)-(B-A)', 'diff'] == pytest.approx(-comparisons.loc['B-A', 'diff'])
    assert np.allclose(result['expansion'].total_delta_ndcg10, 0)


def test_different_seed_sets_and_changed_membership_fail():
    groups, data = evidence()
    data['C'][1] = data['C'].pop(0)
    with pytest.raises(ValueError, match='same nonempty'):
        summarize(data, groups, resamples=10)
    groups, data = evidence()
    data['C'][0]['ranks'][0, 0] = np.inf
    with pytest.raises(ValueError, match='candidate membership'):
        summarize(data, groups, resamples=10)


def test_common_corrupt_membership_and_expansion_improvements_fail():
    groups, data = evidence()
    for arm in data:
        data[arm][0]['target_pos'][0] = -1
    with pytest.raises(ValueError, match='candidate membership'):
        summarize(data, groups, resamples=10)
    groups, data = evidence()
    data['A'][0]['ranks'][1, 0] = 1
    with pytest.raises(ValueError, match='improved after'):
        summarize(data, groups, resamples=10)


@pytest.mark.parametrize('changed_key,changed_path', [
    ('input_sha256', '/original/item_text_emb.npy'),
    ('source_sha256', '/original/ranker_training.py'),
])
def test_cross_stage_embeddings_or_code_change_fails_before_loading_second_jobs(
        tmp_path, changed_key, changed_path):
    groups, _ = evidence()
    roots = [tmp_path / name for name in ('pilot', 'replication')]
    provenance = {'input_sha256': {'/original/item_text_emb.npy': 'original-embedding'},
                  'source_sha256': {'/original/ranker_training.py': 'original-source'}}
    for root in roots:
        root.mkdir()
        # An invalid second job must never be loaded when provenance differs.
        jobs = [] if root == roots[0] else [{'directory': 'do-not-load'}]
        (root / 'study.json').write_text(json.dumps({
            'scope': 'validation only', 'test_predictions_loaded': False,
            'development_sample': False, 'candidate_source': {'val_sha256': 'same'}, 'jobs': jobs}))
        (root / 'status.json').write_text(json.dumps({'status': 'complete'}))
        groups.to_parquet(root / 'validation_groups.parquet', index=False)
        record = copy.deepcopy(provenance)
        if root == roots[1]:
            record[changed_key][changed_path] = 'changed'
        (root / 'provenance.json').write_text(json.dumps(record))
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='input/source hashes changed across stages'):
        run(roots, output)
    assert not output.exists()
