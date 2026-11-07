import numpy as np
import pytest

from cascade.pipelines import refit_draws as R
from cascade.pipelines.graph_latent_study import negative_plan
from cascade.pipelines.local_experiments import study_path


def toy_archive(n=40, width=80, seed=3):
    rng = np.random.default_rng(seed)
    cand = np.stack([rng.permutation(np.arange(1, 500))[:width] for _ in range(n)]).astype(np.int32)
    cand[::7, -5:] = 0  # padding slots must never be sampled
    target = cand[np.arange(n), rng.integers(0, width-6, n)]
    rank = np.where(np.arange(n) % 5 == 0, np.inf, 1.0)
    return {'cand': cand, 'target': target, 'retrieval_rank': rank}


def test_draw_zero_is_the_registered_generator_and_plan():
    archive = toy_archive()
    eligible = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    for seed in (0, 1, 2):
        assert R.plan_rng(seed, 0).integers(0, 2**31, 8).tolist() == np.random.default_rng(seed).integers(0, 2**31, 8).tolist()
        rows, cols = R.negative_plan_draw(archive, eligible, seed, 0, epochs=3)
        ref_rows, ref_cols = negative_plan(archive, eligible, seed, epochs=3)
        np.testing.assert_array_equal(rows, ref_rows); np.testing.assert_array_equal(cols, ref_cols)
        assert rows.dtype == ref_rows.dtype and cols.dtype == ref_cols.dtype


def test_new_draws_change_only_order_and_negatives():
    archive = toy_archive()
    eligible = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    base = R.negative_plan_draw(archive, eligible, 0, 0, epochs=3)
    seen = [base]
    for draw in (1, 2, 3):
        rows, cols = R.negative_plan_draw(archive, eligible, 0, draw, epochs=3)
        again = R.negative_plan_draw(archive, eligible, 0, draw, epochs=3)
        np.testing.assert_array_equal(rows, again[0]); np.testing.assert_array_equal(cols, again[1])  # deterministic
        for r, c in seen:
            assert not np.array_equal(rows, r) and not np.array_equal(cols, c)
        seen.append((rows, cols))
        for e in range(3):
            np.testing.assert_array_equal(np.sort(rows[e]), np.sort(eligible))  # same eligible set every epoch
            ids = np.take_along_axis(archive['cand'][rows[e]], cols[e].astype(np.int64), axis=1)
            np.testing.assert_array_equal(ids[:, 0], archive['target'][rows[e]])  # positive first
            neg = ids[:, 1:]
            assert neg.shape[1] == 50 and (neg > 0).all() and (neg != ids[:, :1]).all()
            assert all(len(np.unique(x)) == 50 for x in cols[e][:, 1:])
    # different seeds give different draw-1 streams
    assert not np.array_equal(R.negative_plan_draw(archive, eligible, 1, 1, 2)[1], R.negative_plan_draw(archive, eligible, 0, 1, 2)[1])


def test_negative_draw_rejected():
    with pytest.raises(ValueError):
        R.plan_rng(0, -1)


@pytest.mark.skipif(not (study_path(R.PARENT)/'plans/seed0_inner.npz').exists(), reason='full parent study not present')
def test_draw_zero_reproduces_the_saved_full_data_inner_plan():
    from cascade.pipelines import full_data_ranking as P
    parent = study_path(R.PARENT)
    archive = P.arrays(parent/'candidates/ranker_train')
    sets = R.plan_sets(archive, np.load(parent/'split/prefix.npy'))
    rows, cols = R.negative_plan_draw(archive, sets['inner'], 0, 0, epochs=2)
    with np.load(parent/'plans/seed0_inner.npz') as z:
        np.testing.assert_array_equal(rows, z['rows'][:2]); np.testing.assert_array_equal(cols, z['columns'][:2])
