"""Saved-result invariants, user weighting, provenance and real-archive replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.eval import metrics as M
from cascade.eval.saved_results import (
    Sources, align, funnel, group_effect_contrast, rank_summary, selection,
    validate_candidate_source, validate_full_run, validate_ranks, validate_seeds,
)
from cascade.utils.io import REPO_ROOT


def test_funnel_preserves_user_denominators_and_retained_movements():
    # User 1 has three events, user 2 only one: macro shares differ from counts.
    users = np.array([1, 1, 1, 2, 3, 3])
    before = np.array([np.inf, 20, 2, 3, 15, 7])
    after = np.array([np.inf, 1, 20, 1, 30, 10])
    found = np.isfinite(before)
    result, codes = funnel(users, before, after, found)
    np.testing.assert_array_equal(codes, [0, 4, 2, 3, 1, 3])
    outcomes = result['outcomes']
    assert sum(v['count'] for v in outcomes.values()) == len(users)
    assert sum(v['macro_share'] for v in outcomes.values()) == pytest.approx(1)
    assert outcomes['retrieval_miss']['macro_share'] == pytest.approx(1 / 9)
    assert outcomes['retained']['ndcg_contribution'] != 0
    assert result['rescue_at_risk_instances'] == 2
    assert result['demotion_at_risk_instances'] == 3
    assert result['rescue_rate_macro'] == pytest.approx(.5)
    assert outcomes['rescued']['macro_share'] - outcomes['demoted']['macro_share'] == pytest.approx(result['delta_recall10_macro'])
    assert sum(v['ndcg_contribution'] for v in outcomes.values()) == pytest.approx(result['delta_ndcg10_macro'])


def test_empty_and_no_retrieved_groups_remain_undefined_conditionally():
    u = np.array([1, 2])
    ranks = np.array([np.inf, np.inf])
    result, _ = funnel(u, ranks, ranks, np.zeros(2, bool))
    assert result['outcomes']['retrieval_miss']['macro_share'] == 1
    assert np.isnan(result['rescue_rate_macro'])
    summary = rank_summary(u, ranks, np.zeros(2, bool))
    assert summary['ndcg10_macro'] == 0
    assert np.isnan(summary['conditional_ndcg10_macro'])
    empty, codes = funnel(np.array([], int), np.array([]), np.array([]), np.array([], bool))
    assert codes.size == 0
    assert np.isnan(empty['delta_ndcg10_macro'])


def test_ranker_must_not_rescue_a_retrieval_miss():
    with pytest.raises(ValueError, match='preserve natural retrieval misses'):
        funnel(np.array([1]), np.array([np.inf]), np.array([1.]), np.array([False]))


@pytest.mark.parametrize('ranks', [[0], [-1], [1.5], [np.nan], [-np.inf]])
def test_invalid_ranks_are_rejected(ranks):
    with pytest.raises(ValueError):
        validate_ranks(np.array(ranks))


def test_alignment_rejects_different_cases_users_and_order():
    a = dict(inst=np.array([1, 2]), user=np.array([3, 4]))
    align(a, a['inst'], a['user'])
    with pytest.raises(ValueError, match='instance alignment'):
        align(a, np.array([2, 1]), a['user'])
    with pytest.raises(ValueError, match='user alignment'):
        align(a, a['inst'], np.array([4, 3]))


@pytest.mark.parametrize('seeds', [[0, 0, 2], [0, 1], [0, 1, 3]])
def test_duplicate_and_incomplete_seeds_are_rejected(seeds):
    with pytest.raises(ValueError, match='seeds'):
        validate_seeds(seeds, [0, 1, 2])


@pytest.mark.parametrize('meta', [{'sample': True}, {'development_sample': True}, {'toy': True}, {},
                                 {'sample': False, 'status': 'running'}, {'sample': False, 'toy': True}])
def test_source_runs_must_be_completed_full_data(meta):
    with pytest.raises(ValueError):
        validate_full_run(meta)


def test_sample_selects_whole_users_reproducibly():
    users = np.repeat(np.arange(100), np.arange(1, 101))
    a, b = selection(users, .02, 0), selection(users, .02, 0)
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(users[a])) == 2
    assert all(np.all(a[users == u]) for u in np.unique(users[a]))


def test_candidate_hash_identity_and_explicit_historical_alias():
    source = dict(directory='data/processed/candidates/frozen', val_sha256='same', meta_sha256='same')
    validate_candidate_source(source, dict(source), '/original/data/processed/candidates/frozen',
                              ['/original/data/processed/candidates/frozen'])
    with pytest.raises(ValueError, match='source hashes'):
        validate_candidate_source(source, dict(source, val_sha256='changed'), source['directory'], [])
    with pytest.raises(ValueError, match='incompatible candidate source'):
        validate_candidate_source(source, source, '/other/data/processed/candidates/frozen', [])


def test_shared_user_contrast_matches_manual_cluster_bootstrap():
    u = np.array([1, 1, 1, 2, 2, 3, 3])
    effect = np.array([.1, .3, -.2, .5, .1, -.1, .4])
    mixed = np.array([True, True, False, True, False, False, True])
    homogeneous = ~mixed
    result = group_effect_contrast(u, effect, mixed, homogeneous, 103, 4)
    frame = pd.DataFrame(dict(u=u, effect=effect, mixed=mixed))
    means = [frame[frame.mixed == flag].groupby('u').effect.mean().reindex([1, 2, 3]) for flag in [True, False]]
    expected = means[0].mean() - means[1].mean()
    rng = np.random.default_rng(4)
    boots = []
    for _ in range(103):
        indices = rng.integers(0, 3, size=3)
        boots.append(means[0].iloc[indices].mean() - means[1].iloc[indices].mean())
    assert result['diff'] == pytest.approx(expected)
    np.testing.assert_allclose([result['ci_low'], result['ci_high']], np.percentile(boots, [2.5, 97.5]))


def test_manifest_detects_tampering_and_sources_block_test_data(tmp_path):
    results = tmp_path / 'results'
    results.mkdir()
    artifact = results / 'val_ranks.npz'
    np.savez(artifact, inst=[1], user=[2], ranks=[3])
    record = dict(path='results/val_ranks.npz', size_bytes=artifact.stat().st_size,
                  sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    manifest = results / 'manifest.json'
    manifest.write_text(json.dumps(dict(artifact_count=1, artifacts=[record])))
    sources = Sources(tmp_path)
    sources.verify_manifest('results/manifest.json')
    sources.archive('results/val_ranks.npz')
    with pytest.raises(ValueError, match='Test/sample'):
        sources.archive('results/test_ranks.npz')
    with pytest.raises(ValueError, match='existing results'):
        sources.archive('data/processed/val.npz')
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed during'):
        sources.track('results/val_ranks.npz')
    with pytest.raises(ValueError, match='hash mismatch'):
        Sources(tmp_path).verify_manifest('results/manifest.json')


def test_actual_saved_validation_funnel_and_rank_reconciliation():
    """Replay the project's real predictions, not just constructed examples."""
    with np.load(REPO_ROOT / 'results/ranker_comparison/20261025-rq3/validation_ranks.npz') as z:
        u, before, ranks, ids = z['user'], z['R1'][0], z['R2'], z['inst']
    with np.load(REPO_ROOT / 'results/retriever/two_tower_text/seed0/val_ranks.npz') as z:
        np.testing.assert_array_equal(z['inst'], ids)
        full = z['rank']
    found = np.isfinite(before)
    assert len(u) == 232904
    assert found.sum() == 16672
    assert (found & (before != full)).sum() == 77
    assert np.sum(M.ndcg_at(before, 10) != M.ndcg_at(full, 10)) == 2
    assert np.sum((before <= 10) != (full <= 10)) == 0
    assert np.sum(found != (full <= 200)) == 0
    stage = np.where(found, np.minimum(full, 200), np.inf)
    expected = [.00710484924666734, .006659718494920886, .0067637319628043955]
    for r, value in zip(ranks, expected):
        result, code = funnel(u, stage, r, found)
        assert rank_summary(u, r, found)['ndcg10_macro'] == pytest.approx(value, abs=1e-15)
        assert result['outcomes']['retrieval_miss']['macro_share'] == pytest.approx(.9312861410542652)
        assert len(code) == len(u)


def test_finite_pessimistic_boundary_rank_is_capped_to_candidate_prefix():
    # At a boundary tie the target can be returned even when full rank is K+1.
    found = np.array([True, False])
    full_rank = np.array([101., 102.])
    stage = np.where(found, np.minimum(full_rank, 100), np.inf)
    assert stage[0] == 100
    assert np.isinf(stage[1])
    detail, _ = funnel(np.array([1, 2]), stage, np.array([1., np.inf]), found)
    assert detail['outcomes']['rescued']['count'] == 1
