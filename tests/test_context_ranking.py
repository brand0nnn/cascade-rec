import numpy as np
import pandas as pd
import pytest
import torch

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines import context_ranking as C
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines.local_experiments import study_path

DAY = C.DAY_MS
T0, T1 = 100*DAY, 200*DAY


def synthetic(tmp_path, extra=()):
    """Five items; item 5 first appears at T1 (Val), item 4 between T0 and T1."""
    rows = [(1, 1, 5., 10*DAY), (1, 2, 4., 50*DAY), (2, 1, 3., 60*DAY), (2, 3, 1., 99*DAY),
            (1, 4, 2., 150*DAY), (2, 2, 5., 180*DAY), (3, 1, 4., 70*DAY), (3, 5, 5., 250*DAY), *extra]
    inter = pd.DataFrame(rows, columns=['user', 'item', 'rating', 'timestamp']).sort_values(['user', 'timestamp', 'item'])
    tmp_path.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(tmp_path/'interactions.parquet')
    np.save(tmp_path/'item_text_emb.npy', np.zeros((6, 384), np.float32))
    first = inter.groupby('item').timestamp.min()
    items = pd.DataFrame({'item': np.arange(1, 6), 'leaf_cat': list('aabbc'), 'top_cat': 'x', 'store': 's',
                          'price': [1., 2., np.nan, 30., 4.], 'first_ts': first.loc[np.arange(1, 6)].to_numpy()})
    counts = inter.groupby('user').size().reindex(range(4), fill_value=0).to_numpy()
    ptr = np.r_[0, np.cumsum(counts)]
    data = ProcessedData(tmp_path, 3, 5, ptr, inter.item.to_numpy(), inter.timestamp.to_numpy(),
                         pd.DataFrame({'inst': [0], 'user': [1], 'item': [4], 'ts': [150*DAY], 'period': [1], 'hist_len': [2]}),
                         items, {'t_ranker_train': T0, 't1': T1, 't2': 400*DAY}, 50)
    return data


def item_codes(data, period):
    builder = RankerFeatureBuilder.create(data, {'score': np.zeros((1, 2), np.float32)}, data.dir/'item_text_emb.npy')
    cutoff = T0 if period == 'ranker_train' else T1
    return C.period_item_codes(builder, data.items, data.n_items, period, cutoff), cutoff


def test_period_statistics_use_only_interactions_strictly_before_the_period_start(tmp_path):
    base = synthetic(tmp_path/'base')
    # Extra rows at/after each period start (one exactly at it), with extreme ratings.
    after_t0 = synthetic(tmp_path/'after_t0', extra=[(3, 3, 5., T0), (3, 4, 1., T0+5)])
    after_t1 = synthetic(tmp_path/'after_t1', extra=[(3, 2, 1., T1), (3, 4, 1., T1+1)])
    for future, periods in ((after_t0, ('ranker_train',)), (after_t1, ('ranker_train', 'val'))):
        for period in periods:
            a, cutoff = item_codes(base, period)
            b, _ = item_codes(future, period)
            np.testing.assert_array_equal(a, b)
            proof = C.check_period_codes(b, future.dir, future.n_items, cutoff)
            assert proof['latest_interaction_used_ms'] < cutoff
    codes, _ = item_codes(base, 'ranker_train')
    # item 1: 3 reviews before T0 (5,3,4 -> mean 4), first at day 10 -> age log2(91)=6
    assert codes[1].tolist() == [2, 4, 6]
    assert codes[3].tolist() == [1, 1, 1]          # 1 review (rating 1), first one day before T0
    assert codes[4].tolist() == [0, 0, C.NEW_AGE]  # first review after T0
    assert codes[5].tolist() == [0, 0, C.NEW_AGE]
    val, _ = item_codes(base, 'val')
    assert val[4].tolist()[:2] == [1, 2] and val[4, 2] != C.NEW_AGE
    assert val[5].tolist() == [0, 0, C.NEW_AGE]


def test_recount_detects_a_table_built_with_future_rows(tmp_path):
    data = synthetic(tmp_path)
    codes, cutoff = item_codes(data, 'val')
    leaked = codes.copy(); leaked[5] = [0, 5, 0]
    with pytest.raises(ValueError, match='differ'):
        C.check_period_codes(leaked, data.dir, data.n_items, cutoff)


def test_price_deciles_fit_only_on_pre_t0_items():
    items = pd.DataFrame({'item': np.arange(1, 21), 'price': np.r_[np.arange(1., 13.), np.nan, 1000., np.arange(6.)],
                          'first_ts': np.r_[np.zeros(13), T0, np.full(6, T0)]})
    codes, cuts = C.price_codes(items, 20, T0)
    np.testing.assert_allclose(cuts, np.quantile(np.arange(1., 13.), np.linspace(.1, .9, 9)))
    assert codes[0] == -1 and codes[13] == 0 and codes[14] == 10 and codes[1] == 1 and codes[12] == 10


def test_user_buckets(tmp_path):
    data = synthetic(tmp_path)
    inst = pd.DataFrame({'user': [1, 1, 2], 'hist_len': [1, 2, 3], 'ts': [11*DAY, 50*DAY+1, 181*DAY]})
    codes = C.user_codes(data, inst)
    # gaps: 1 day -> log2(2)=1; 1 ms -> 0; 1 day -> 1
    assert codes.tolist() == [[0, 1], [0, 0], [1, 1]]


def test_same_timestamp_new_targets_are_indistinguishable(tmp_path):
    data = synthetic(tmp_path)
    codes, _ = item_codes(data, 'val')
    price = np.array([-1, 1, 2, 0, 3, 4])
    table_codes = np.concatenate([price[:, None], codes], 1)
    users = np.zeros((2, 2), np.int64)
    item_table, _ = C.scaled_tables(table_codes, users, np.zeros(C.CONTEXT_DIM), np.ones(C.CONTEXT_DIM))
    first = np.full(6, np.iinfo(np.int64).max); first[1:] = data.items.first_ts.to_numpy()
    # query 0: target 5 first reviewed at its query time; query 1: item 5 is another new candidate.
    cand = np.array([[5, 1, 2, 0], [5, 1, 3, 2]])
    report = C.new_candidate_check(cand, np.array([5, 1]), np.array([250*DAY, 260*DAY]), first,
                                   table_codes, item_table, T1)
    assert report['same_ts_candidates'] == 1 and report['other_new_candidates'] == 1
    assert report['same_ts_time_codes'] == report['other_new_time_codes'] == [(0, 0, C.NEW_AGE)]
    with pytest.raises(ValueError, match='not the target'):
        C.new_candidate_check(cand, np.array([1, 1]), np.array([250*DAY, 260*DAY]), first, table_codes, item_table, T1)
    leaked = table_codes.copy(); leaked[5, 1] = 5   # a rating that could only come from the target review
    with pytest.raises(ValueError, match='non-new'):
        C.new_candidate_check(cand, np.array([5, 1]), np.array([250*DAY, 260*DAY]), first, leaked, item_table, T1)


def test_context_model_reduces_to_base_when_context_inputs_are_zero():
    n = 7
    rng = np.random.default_rng(0)
    args = (rng.normal(size=(n, 384)).astype(np.float32), np.r_[0, rng.integers(1, 3, n-1)], np.r_[0, rng.integers(1, 3, n-1)],
            np.zeros((n, 1154), np.float32), np.zeros((n, 64), np.float32), np.zeros(n, bool))
    torch.manual_seed(0)
    base = GraphLatentRanker(*args, C.BASE_COMMON, C.GRAPH_DIM, 'R2')
    torch.nn.init.normal_(base.head[-1].weight)
    model = GraphLatentRanker(*args, C.BASE_COMMON+C.CONTEXT_DIM, C.GRAPH_DIM, 'R2')
    C.copy_with_context(base, model, seed=0)
    extra_before = model.head[0].weight[:, C.COMMON_OFFSET:C.COMMON_OFFSET+C.CONTEXT_DIM].detach().clone()
    assert extra_before.abs().max() > 0
    cand = torch.tensor([[1, 2, 3], [4, 5, 6]]); hist = torch.tensor([[0, 1], [2, 3]])
    score = torch.randn(2, 3); common = torch.randn(2, 3, C.BASE_COMMON); graph = torch.zeros(2, 3, C.GRAPH_DIM)
    padded = torch.cat([common, torch.zeros(2, 3, C.CONTEXT_DIM)], -1)
    base.eval(); model.eval()
    torch.testing.assert_close(model(cand, hist, score, padded, graph), base(cand, hist, score, common, graph))
    optimisers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in (base, model)]
    for m, opt, c in ((base, optimisers[0], common), (model, optimisers[1], padded)):
        loss = m(cand, hist, score, c, graph).logsumexp(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    w = model.head[0].weight.detach()
    torch.testing.assert_close(w[:, C.COMMON_OFFSET:C.COMMON_OFFSET+C.CONTEXT_DIM], extra_before)
    torch.testing.assert_close(torch.cat([w[:, :C.COMMON_OFFSET], w[:, C.COMMON_OFFSET+C.CONTEXT_DIM:]], 1),
                               base.head[0].weight.detach())
    assert len(C.column_names()) == C.CONTEXT_DIM == 63


def test_batch_args_append_context_without_changing_shared_inputs():
    archive = {'cand': np.array([[3, 1, 2, 0]]), 'score': np.ones((1, 4), np.float32), 'history': np.array([[0, 2]]),
               'target': np.array([1])}
    feature = {'common': np.arange(36, dtype=np.float32).reshape(1, 4, 9)}
    context = {'item': torch.arange(4*C.ITEM_DIM, dtype=torch.float32).reshape(4, C.ITEM_DIM),
               'user': torch.full((1, C.USER_DIM), 7.)}
    columns = np.array([[1, 0, 3]])
    args = C.batch_args(archive, feature, context, np.array([0]), 'cpu', columns)
    common = args[3].numpy()
    assert common.shape == (1, 3, C.BASE_COMMON+C.CONTEXT_DIM)
    np.testing.assert_array_equal(common[..., :9], np.take_along_axis(feature['common'], columns[:, :, None], axis=1))
    np.testing.assert_array_equal(common[0, :, 9:9+C.ITEM_DIM], context['item'][[1, 3, 0]].numpy())
    assert (common[..., 9+C.ITEM_DIM:] == 7).all() and (args[4].numpy() == 0).all()


def real_source(name):
    root = study_path(name)
    if not (root/'study.json').exists():
        pytest.skip(f'{name} not available locally')
    meta = fdr.read(root/'study.json')
    return meta, root, fdr.data_for(meta)


def test_sample_context_leak_checks():
    meta, root, data = real_source(C.SAMPLE_SOURCE)
    built = C.build_context(data, meta, root)
    for split in ('ranker_train', 'val'):
        leak = built['leak'][split]
        assert leak['same_ts_all_targets'] and leak['time_dependent_columns_identical']
        assert built['cutoff_proof'][split]['latest_interaction_used_ms'] < data.split[C.PERIOD_KEY[split]]


def test_full_val_656_same_timestamp_targets_are_indistinguishable_from_other_new_candidates():
    meta, root, data = real_source(C.FULL_SOURCE)
    assert data.split['t_ranker_train'] == C.FULL_T0 and data.split['t1'] == C.FULL_T1
    built = C.build_context(data, meta, root)
    leak = built['leak']['val']
    assert leak['same_ts_candidates'] == 656 and leak['same_ts_all_targets']
    assert leak['same_ts_time_codes'] == leak['other_new_time_codes'] == [(0, 0, C.NEW_AGE)]
    assert leak['same_ts_distinct_scaled_time_rows'] == leak['other_new_distinct_scaled_time_rows'] == 1
    for split in ('ranker_train', 'val'):
        assert built['cutoff_proof'][split]['latest_interaction_used_ms'] < data.split[C.PERIOD_KEY[split]]


def test_t0_snapshot_context_swaps_only_the_candidate_table(tmp_path):
    for split, value in (('ranker_train', 1.), ('val', 2.)):
        d = tmp_path/'context'/split; d.mkdir(parents=True)
        np.save(d/'item_table.npy', np.full((4, C.ITEM_DIM), value, np.float32))
        np.save(d/'user_table.npy', np.full((3, C.USER_DIM), value+10, np.float32))
    snap = C.load_context(tmp_path, 'val', 'cpu', item_split='ranker_train')
    fresh = C.load_context(tmp_path, 'val', 'cpu')
    assert (snap['item'] == 1).all() and (fresh['item'] == 2).all()
    assert (snap['user'] == 12).all() and (fresh['user'] == 12).all()
    assert C.METRICS['mrr10'](np.array([1., 4., 11., np.inf])).tolist() == [1., .25, 0., 0.]
    assert C.METRICS['hr3'](np.array([3., 4.])).tolist() == [1., 0.]
