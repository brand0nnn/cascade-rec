"""Explicit T0/T1 snapshot adapters; canonical splits and frozen builders stay unchanged."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from cascade.data.batched_graph_features import BatchedGraphFeatures


class SnapshotGraphFeatures(BatchedGraphFeatures):
    def __init__(self, data, cutoff):
        cutoff = int(cutoff)
        if cutoff not in (data.split['t_ranker_train'], data.split['t1']):
            raise ValueError('only registered T0/T1 graph snapshots are supported')
        self.data = data
        self.strict_before_ms = cutoff
        reviews = data.interactions(['retriever_train', 'ranker_train'])
        reviews = reviews.loc[reviews.ts < cutoff]
        if reviews.duplicated(['user', 'item']).any(): raise ValueError('duplicate graph edge')
        users, items = reviews.user.to_numpy(), reviews.item.to_numpy()
        self.matrix = sparse.csc_matrix((np.ones(len(reviews), np.float32), (users, items)),
                                        shape=(data.n_users+1, data.n_items+1))
        self.by_user = self.matrix.tocsr()
        self.degree = np.asarray(self.matrix.sum(axis=0)).ravel().astype(np.float32)
        self.leaf, self.store = self._metadata(data)
        self.leaf_available = self._neighbour_available(self.leaf)
        self.store_available = self._neighbour_available(self.store)
        self.cooccurrence = (self.matrix.T @ self.matrix).tocsr()
        self.support = self.cooccurrence.copy(); self.support.data.fill(1)
        self.audit = {'strict_before_ms': cutoff, 'review_edges': int(self.matrix.nnz),
            'max_review_timestamp': int(reviews.ts.max()) if len(reviews) else None,
            'query_user_excluded': True, 'self_history_paths_excluded': True,
            'labels_read': False, 'history_limit': 50, 'canonical_split_unchanged': True,
            'common_names': list(self.common_names), 'graph_names': list(self.graph_names)}
