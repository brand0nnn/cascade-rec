"""Toy checks for the fixed-ranker pool wrapper (no data files, CPU only)."""
import numpy as np
import pandas as pd
import pytest
import torch

from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines import pool_ranking as pr
from cascade.pipelines.local_evidence_export import ranks_and_top10

N_ITEMS, ROWS = 320, 6


def toy_archive(seed=0):
    rng = np.random.default_rng(seed)
    cand = np.stack([rng.choice(np.arange(1, N_ITEMS+1), 200, replace=False) for _ in range(ROWS)]).astype(np.int32)
    # Unsorted columns with exact ties, as in the P3 interleave/RRF archive.
    score = rng.choice(np.array([.03, .02, .02, .015, .01], np.float32), size=(ROWS, 200)).astype(np.float32)
    target = cand[np.arange(ROWS), rng.integers(0, 200, ROWS)]
    target[-1] = N_ITEMS+5  # one retrieval miss
    history = np.zeros((ROWS, 50), np.int64); history[:, -1] = N_ITEMS+1
    ranks, top = ranks_and_top10(score, cand, target)
    return {'cand': cand, 'score': score, 'inst': np.arange(ROWS), 'user': np.array([0, 0, 1, 2, 2, 3]),
            'target': target, 'retrieval_rank': ranks, 'top10': top, 'history': history}


def toy_model():
    torch.manual_seed(0)
    n = N_ITEMS+6
    text = np.random.default_rng(1).normal(size=(n, 384)).astype(np.float32)
    return GraphLatentRanker(text, np.arange(n) % 7, np.arange(n) % 5, np.zeros((n, 1154), np.float32),
                             np.zeros((n, 64), np.float32), np.zeros(n, bool), 9, 6, 'R2')


def test_unsorted_columns_rank_pessimistically_and_top10_follows_score():
    arc = toy_archive()
    s, c, t = arc['score'], arc['cand'], arc['target']
    for i in range(ROWS-1):
        col = int(np.flatnonzero(c[i] == t[i])[0])
        assert arc['retrieval_rank'][i] == (s[i] >= s[i, col]).sum()  # ties counted against the target
        order = sorted(range(200), key=lambda j: (-s[i, j], j))[:10]  # score desc, then column order
        assert arc['top10'][i].tolist() == c[i, order].tolist()
    assert np.isinf(arc['retrieval_rank'][-1])
    assert (np.diff(s, axis=1) > 0).any()  # columns really are unsorted


def test_epoch_zero_equals_retrieval_order_on_unsorted_archive():
    arc = toy_archive()
    feature = {'common': np.random.default_rng(2).normal(size=(ROWS, 200, 9)).astype(np.float32), 'graph': pr.ZeroGraph()}
    ranks, top, logits = fdr.predict(toy_model(), arc, feature, np.arange(ROWS), pr.METHOD, 'cpu')
    assert np.array_equal(logits, arc['score'])
    assert np.array_equal(ranks, arc['retrieval_rank']) and np.array_equal(top, arc['top10'])


def test_zero_graph_matches_disabled_columns():
    rows = np.array([3, 1, 4])
    assert pr.ZeroGraph()[rows].shape == (3, 200, 6) and not pr.ZeroGraph()[rows].any()
    cols = np.zeros((3, 51), np.int16)
    arc = toy_archive()
    feature = {'common': np.ones((ROWS, 200, 9), np.float32), 'graph': pr.ZeroGraph()}
    args = fdr.batch_args(arc, feature, rows, pr.METHOD, 'cpu', cols)
    assert args[4].shape == (3, 51, 6) and not args[4].any()


def test_check_archive_accepts_unsorted_and_rejects_bad_rank():
    arc = toy_archive()
    arc['target'][-1] = arc['cand'][-1, 0]
    arc['retrieval_rank'], arc['top10'] = ranks_and_top10(arc['score'], arc['cand'], arc['target'])
    inst = pd.DataFrame({'inst': arc['inst'], 'user': arc['user'], 'item': arc['target'], 'ts': np.full(ROWS, 10)})
    first = np.zeros(N_ITEMS+6, np.int64)
    out = pr.check_archive(arc, inst, first)
    assert out['unsorted_rows'] == ROWS and out['natural'] == ROWS and out['padding_slots'] == 0
    bad = {**arc, 'retrieval_rank': arc['retrieval_rank']+1}
    with pytest.raises(ValueError):
        pr.check_archive(bad, inst, first)
    late = first.copy(); late[arc['cand'][0, 5]] = 11
    with pytest.raises(ValueError):
        pr.check_archive(arc, inst, late)


def test_tiebreak_is_not_a_pure_tiebreak_when_gaps_are_small():
    # Column 0 has a slightly larger RRF score than column 150; the 1e-6 x position offset reverses them.
    score = np.full((1, 200), .001, np.float32)
    score[0, 0], score[0, 150] = .0160, .0161
    cand = np.arange(1, 201)[None].astype(np.int32)
    tb = pr.tiebreak_scores(score)
    assert tb[0, 0] > tb[0, 150]  # reordered distinct values
    lex = pr.lexicographic_ranks(score, cand, np.array([1]))
    assert lex[0] == 2  # a pure tie-break keeps column 150 first
    tied = np.full((1, 200), .001, np.float32)
    assert pr.lexicographic_ranks(tied, cand, np.array([4]))[0] == 4  # ties follow column order


def test_funnel_partitions_population():
    retrieval = np.array([1., 5, 30, np.inf, 12, 3])
    final = np.array([2., 15, 4, np.inf, 40, 1])
    masks = pr.funnel_masks(retrieval, final)
    assert [k for k in pr.FUNNEL] == list(masks)
    got = {k: np.flatnonzero(v).tolist() for k, v in masks.items()}
    assert got == {'retrieval_miss': [3], 'unrescued': [4], 'rescued': [2], 'demoted': [1], 'retained': [0, 5]}


def test_overlap_index_maps_shared_items():
    a = np.array([[5, 3, 9, 1]]); b = np.array([[1, 7, 5, 2]])
    assert pr.overlap_index(a, b).tolist() == [[2, -1, -1, 0]]


def test_frozen_function_sources_are_extracted():
    src = pr.function_sources(pr.__file__)
    assert set(src) == set(pr.FROZEN_FUNCTIONS)


def test_combo_rowview_and_additivity():
    from cascade.pipelines import pool_combo as pc
    view = pc.RowView(np.arange(12).reshape(3, 4), np.array([2, 5, 7]), 9)
    assert view[np.array([7, 2])].tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]
    with pytest.raises(ValueError):
        view[np.array([3])]
    a = pc.additivity({'pool': .003, 'fresh': .001}, .0035)
    assert abs(a['sum_individual']-.004) < 1e-12 and abs(a['interaction']+.0005) < 1e-12
