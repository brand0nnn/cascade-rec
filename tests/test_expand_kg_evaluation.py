import numpy as np
import pandas as pd
import pytest

from cascade.eval.bootstrap import paired_bootstrap
from cascade.pipelines.expand_kg_evaluation import (
    BOOTSTRAP_SEED, RESAMPLES, category_continuity, degree_bucket,
    ranker_slice, require_hash, run, witness_buckets,
)


def test_degree_boundaries_and_invalid_counts():
    assert degree_bucket(np.array([0, 1, 2, 5, 6, 200])).tolist() == ['0', '1', '2-5', '2-5', '6+', '6+']
    with pytest.raises(ValueError):
        degree_bucket(np.array([-1]))
    with pytest.raises(ValueError):
        degree_bucket(np.array([1.5]))


def test_history_category_padding_returns_and_unknowns():
    leaf = np.array(['', 'Cases', 'Chargers', '', 'Holders'], dtype=object)
    histories = np.array([[0, 1, 2], [0, 2, 1], [0, 2, 4], [0, 3, 4], [0, 0, 0], [0, 1, 3]])
    targets = np.array([1, 1, 1, 1, 3, 1])
    any_history, continuity, last = category_continuity(histories, targets, leaf)
    assert any_history.tolist() == ['true', 'true', 'false', 'unknown', 'unknown', 'true']
    assert continuity.tolist() == ['Returning category', 'Immediate continuity', 'Novel category for user',
                                  'Unknown category metadata', 'Unknown category metadata', 'Unknown category metadata']
    assert last.tolist() == ['false', 'true', 'false', 'false', 'unknown', 'unknown']


def test_witness_era_target_edge_and_strict_cutoff_on_both_legs():
    profiles = pd.DataFrame({'all_train_other_reviewers': [0, 1, 3],
                             'all_train_witness_target_ts': [0, 50, 150],
                             'all_train_witness_history_ts': [0, 150, 50]})
    _, era = witness_buckets(profiles, np.array([300, 300, 300]), 'all_train', 100, 200)
    assert era.tolist() == ['none', 'pre-T0', '[T0,T1)']
    for key in ('all_train_witness_target_ts', 'all_train_witness_history_ts'):
        changed = profiles.copy()
        changed.loc[1, key] = 200
        with pytest.raises(ValueError, match='strict temporal'):
            witness_buckets(changed, np.array([300, 300, 300]), 'all_train', 100, 200)


def test_cluster_weighting_seed_means_and_churn():
    # User 1 has two events; user 2 has one. Seed-average first, never ensemble ranks.
    users = np.array([1, 1, 2])
    base = np.array([1, 20, np.inf])
    mlp = np.array([[20, 1, np.inf], [1, 20, np.inf], [20, 1, np.inf]])
    din = np.array([[1, 20, np.inf]] * 3)
    row = ranker_slice(users, base, mlp, din, np.ones(3, bool))
    assert row['entered200'] == 2
    assert row['recall200'] == .5
    assert row['retrieval_ndcg10'] == .25
    assert row['mlp_rescued_into_top10'] == pytest.approx(2 / 3)
    assert row['mlp_knocked_out_of_top10'] == pytest.approx(2 / 3)
    assert row['mlp_rescued_seed1'] == row['mlp_demoted_seed1'] == 0
    assert row['mlp_net_rescue_ratio'] == 0
    assert np.isnan(row['din_net_rescue_ratio'])
    expected = paired_bootstrap(users, np.array([1, 0, 0]), np.array([1 / 3, 2 / 3, 0]),
                                n_resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
    assert row['mlp_change'] == expected['diff']
    assert row['mlp_ci_low'] == expected['ci_low']
    assert row['mlp_ci_high'] == expected['ci_high']


def test_source_identity_and_output_collision(tmp_path):
    source = tmp_path / 'source'
    source.write_text('changed')
    with pytest.raises(ValueError, match='source identity'):
        require_hash(source, 'invalid')
    with pytest.raises(FileExistsError):
        run(tmp_path)
