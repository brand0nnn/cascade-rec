from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.data.instances import ProcessedData, build_sequences
from cascade.data.local_graph_features import FrozenGraphFeatures


def example():
    edges = pd.DataFrame({'user': [1,1,1,2,2,3,3,4,4,4],
                          'item': [1,2,4,1,3,2,3,1,2,3],
                          'timestamp': [1,2,11,1,2,1,2,1,2,3]})
    ptr, item, ts = build_sequences(edges, 4)
    items = pd.DataFrame({'item': np.arange(1,7), 'leaf_cat': ['X','Y','X','','X',' '],
                          'store': ['A','B','A','','A',''], 'first_ts': [1,1,1,11,12,12]})
    data = ProcessedData(Path('.'), 4, 6, ptr, item, ts, pd.DataFrame(), items,
                         {'t_ranker_train': 10, 't1': 20, 't2': 30}, 50)
    queries = pd.DataFrame({'user': [1,2,3,4], 'hist_len': [3,2,2,3], 'ts': [12]*4, 'item': [3,2,1,5]})
    candidates = np.array([[3,5,6,0],[2,4,5,0],[1,4,6,0],[4,5,6,0]])
    return data, queries, candidates


def test_sparse_batch_matches_reference_with_own_edges_missing_metadata_and_future_history():
    data, queries, candidates = example()
    fast, slow = BatchedGraphFeatures(data), FrozenGraphFeatures(data)
    actual, expected = fast.build(queries, candidates), slow.build(queries, candidates)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
        assert not a[:, -1].any()
    relabelled = queries.copy(); relabelled['item'] = 6
    for index in range(len(queries)):
        single = fast.build(relabelled.iloc[[index]], candidates[[index]])
        for a, b in zip(single, actual): np.testing.assert_allclose(a[0], b[index], rtol=2e-6, atol=2e-6)


def test_sparse_batch_rejects_candidates_in_full_history():
    data, queries, candidates = example()
    candidates[0, 0] = 1
    with pytest.raises(ValueError, match='full-history-filtered'):
        BatchedGraphFeatures(data).build(queries, candidates)
