"""Sparse-batch equivalent of FrozenGraphFeatures for filtered natural lists.

Candidates must exclude the user's complete history. Consequently no candidate
has a query-user edge and no candidate equals a history item; this permits exact
algebraic batching of the reference's query-user/self exclusions. No labels enter
the calculations. All review edges remain strictly pre-T0.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from cascade.data.local_graph_features import FrozenGraphFeatures


class BatchedGraphFeatures(FrozenGraphFeatures):
    def __init__(self, data):
        super().__init__(data)
        self.cooccurrence = (self.matrix.T @ self.matrix).tocsr()
        self.support = self.cooccurrence.copy()
        self.support.data.fill(1)

    @staticmethod
    def gather(matrix, candidates):
        selected = matrix[np.arange(len(candidates))[:, None], candidates]
        return np.asarray(selected.toarray() if sparse.issparse(selected) else selected, dtype=np.float32)

    def build(self, instances, candidates):
        candidates = np.asarray(candidates)
        if candidates.ndim != 2 or len(candidates) != len(instances) or not np.issubdtype(candidates.dtype, np.integer):
            raise ValueError('expected integer candidates aligned to queries')
        if (candidates < 0).any() or (candidates > self.data.n_items).any():
            raise ValueError('candidate outside catalogue')
        if (instances.ts.to_numpy() < self.strict_before_ms).any():
            raise ValueError('query precedes frozen graph cutoff')
        users = instances.user.to_numpy(dtype=np.int64)
        lengths = instances.hist_len.to_numpy(dtype=np.int64)
        histories = self.data.histories(users, lengths, 50)
        own_history = np.zeros_like(histories, dtype=bool)
        for r, (user, length, cand) in enumerate(zip(users, lengths, candidates)):
            begin = self.data.user_ptr[user]
            full = self.data.seq_item[begin:begin+length]
            if np.isin(cand[cand > 0], full).any():
                raise ValueError('batched features require full-history-filtered candidates')
            valid = histories[r] > 0
            if len(np.unique(histories[r, valid])) != valid.sum():
                raise ValueError('histories must contain unique products')
            times = self.data.seq_ts[begin+max(0, length-50):begin+length]
            own_history[r, valid] = times < self.strict_before_ms
        valid = histories > 0
        row, col = np.nonzero(valid)
        h = sparse.csr_matrix((np.ones(len(row), np.float32), (row, histories[row, col])),
                              shape=(len(instances), self.data.n_items+1))
        degree_h = np.maximum(self.degree[histories]-own_history, 0)
        weight = np.divide(1., np.sqrt(degree_h), out=np.zeros_like(degree_h), where=(degree_h > 0) & valid)
        weighted_h = sparse.csr_matrix((weight[row, col], (row, histories[row, col])), shape=h.shape)
        reviewers = h @ self.matrix.T
        reviewers.data.fill(1)
        common = np.zeros((*candidates.shape, len(self.common_names)), np.float32)
        graph = np.zeros((*candidates.shape, len(self.graph_names)), np.float32)
        degree_c = self.degree[candidates]
        common[..., 0] = np.log1p(degree_c)
        common[..., 1] = degree_c > 0
        common[..., 2] = ((self.degree[histories] > 0) & valid).sum(1)[:, None]/np.maximum(valid.sum(1)[:, None], 1)
        for labels, cindex, hindex, gindex in ((self.leaf, 3, 4, 3), (self.store, 5, 6, 4)):
            known = (labels[histories] > 0) & valid
            common[..., cindex] = labels[candidates] > 0
            common[..., hindex] = known.sum(1)[:, None]/np.maximum(valid.sum(1)[:, None], 1)
            matching = ((labels[candidates][:, None, :] == labels[histories][:, :, None]) &
                        known[:, :, None] & (labels[candidates][:, None, :] > 0))
            graph[..., gindex] = matching.sum(1)/np.maximum(known.sum(1)[:, None], 1)
        common[..., 7] = self.leaf_available[candidates]
        common[..., 8] = self.store_available[candidates]
        graph[..., 0] = np.log1p(self.gather(reviewers @ self.matrix, candidates))
        graph[..., 1] = np.log1p(self.gather(h @ self.cooccurrence, candidates))
        strength = self.gather(weighted_h @ self.cooccurrence, candidates)
        strength = np.divide(strength, np.sqrt(degree_c), out=np.zeros_like(strength), where=degree_c > 0)
        graph[..., 2] = strength/np.maximum(((degree_h > 0) & valid).sum(1)[:, None], 1)
        graph[..., 5] = self.gather(h @ self.support, candidates)/np.maximum(valid.sum(1)[:, None], 1)
        common[candidates == 0] = 0
        graph[candidates == 0] = 0
        if not np.isfinite(common).all() or not np.isfinite(graph).all():
            raise ValueError('nonfinite batched graph features')
        return common, graph
