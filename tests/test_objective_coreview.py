import numpy as np
import pytest

from cascade.pipelines.objective_coreview_report import interaction, novelty_difference


def test_interaction_is_difference_of_graph_gains_and_cancels_objective_offsets():
    values = {'bce_text': np.array([2., 3.]), 'bce_coreview': np.array([3., 5.]),
              'pairwise_text': np.array([4., 7.]), 'pairwise_coreview': np.array([8., 8.])}
    np.testing.assert_array_equal(interaction(values), [3., -1.])
    changed = {name: v+(100 if name.startswith('bce') else -50) for name, v in values.items()}
    np.testing.assert_array_equal(interaction(changed), interaction(values))


def test_novelty_bootstrap_preserves_within_user_covariance():
    # Each user's new-minus-warm difference is exactly one, despite large between-user variation.
    users = np.repeat([1,2,3], 2)
    values = np.array([1.,0.,2.,1.,3.,2.])
    new = np.tile([True,False], 3)
    result = novelty_difference(users, values, new)
    assert result['diff'] == pytest.approx(1)
    assert result['ci_low'] == pytest.approx(1)
    assert result['ci_high'] == pytest.approx(1)
    assert result['n_users'] == 3 and result['n_resamples'] == 1000


def test_novelty_macro_is_invariant_to_repeating_a_users_identical_event():
    users = np.array([1,1,2,2,3,3])
    values = np.array([1.,0.,2.,1.,3.,2.]); new = np.tile([True,False], 3)
    a = novelty_difference(users, values, new)
    b = novelty_difference(np.r_[users, np.ones(20, int)], np.r_[values, np.ones(20)],
                           np.r_[new, np.ones(20, bool)])
    assert a == b


def test_novelty_comparison_requires_both_populations():
    with pytest.raises(ValueError, match='both novelty groups'):
        novelty_difference(np.arange(3), np.ones(3), np.ones(3, bool))
