"""Candidate-wise graph inputs for the isolated graph/latent ranker study.

The review matrix contains only edges strictly before T0. Relational features
remove the query user's row. Let R(i) be the remaining reviewers of item i and H the distinct
nonpadding last-50 history items. Optional relational features exclude i itself
from H. Reviewer/path counts are respectively |union_h R(h) intersect R(i)| and
sum_h |R(h) intersect R(i)|. Normalised strength averages the pair cosine
|R(h) intersect R(i)| / sqrt(deg(h)*deg(i)) over graph-supported nonself H.
Supported-history fraction counts nonself H with any shared reviewer divided by
all nonself H. The common history graph fraction instead records global pre-T0
item availability, including the query user's own prior history. This makes
factor-knownness available to all arms without revealing candidate adjacency.

Leaf/store fractions exclude self and use known-metadata history denominators.
Unknown relations contribute zero, with separate knownness/support controls.
No true-target or label field is read by the feature builder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class FrozenGraphFeatures:
    common_names = (
        'candidate_log_degree', 'candidate_graph_known', 'history_graph_fraction',
        'candidate_leaf_known', 'history_leaf_fraction_known',
        'candidate_store_known', 'history_store_fraction_known',
        'leaf_neighbor_available', 'store_neighbor_available',
    )
    graph_names = (
        'log_other_reviewers', 'log_witness_paths', 'normalised_path_strength',
        'same_leaf_history_fraction', 'same_store_history_fraction',
        'supported_history_fraction',
    )

    def __init__(self, data):
        self.data = data
        self.strict_before_ms = int(data.split['t_ranker_train'])
        reviews = data.interactions(['retriever_train'])
        time = 'ts' if 'ts' in reviews else 'timestamp'
        reviews = reviews.loc[reviews[time] < self.strict_before_ms]
        if reviews.duplicated(['user', 'item']).any():
            raise ValueError('pre-T0 review graph has duplicate user-item pairs')
        users = reviews['user'].to_numpy(dtype=np.int64)
        items = reviews['item'].to_numpy(dtype=np.int64)
        if np.any(users < 1) or np.any(users > data.n_users) or np.any(items < 1) or np.any(items > data.n_items):
            raise ValueError('review graph contains invalid IDs')
        self.matrix = sparse.csc_matrix((np.ones(len(reviews), np.float32), (users, items)),
                                       shape=(data.n_users+1, data.n_items+1))
        self.by_user = self.matrix.tocsr()
        self.degree = np.asarray(self.matrix.sum(axis=0)).ravel().astype(np.float32)
        self.leaf, self.store = self._metadata(data)
        self.leaf_available = self._neighbour_available(self.leaf)
        self.store_available = self._neighbour_available(self.store)
        self.audit = {
            'strict_before_ms': self.strict_before_ms,
            'review_edges': int(self.matrix.nnz),
            'max_review_timestamp': int(reviews[time].max()) if len(reviews) else None,
            'common_names': list(self.common_names), 'graph_names': list(self.graph_names),
            'query_user_excluded': True, 'self_history_paths_excluded': True,
            'labels_read': False, 'history_limit': 50,
            'degree_definition': 'number of distinct pre-T0 reviewers after excluding query user',
            'common_history_graph_fraction': 'fraction of distinct valid last-50 history items with GLOBAL pre-T0 degree>0, including own prior reviews; candidate-independent factor-knownness control',
            'normalised_path_strength': 'mean |R(h) intersection R(c)|/sqrt(deg(h)*deg(c)) over graph-supported nonself history items; zero when unsupported',
            'supported_history_fraction': 'fraction of all nonself valid history items with at least one shared other reviewer with candidate',
            'metadata_fractions': 'matching nonself known history labels / nonself known history labels; missing candidate or empty denominator gives zero',
            'metadata_neighbour_availability': 'at least one other pre-T0 product with the same known leaf/store, regardless of current query user',
        }

    @staticmethod
    def _metadata(data):
        result = []
        for column in ('leaf_cat', 'store'):
            values = data.items[column].fillna('').astype(str)
            values = values.where(values.str.strip().ne(''), '')
            vocabulary = {value:i+1 for i,value in enumerate(sorted(set(values)-{''}))}
            codes = np.zeros(data.n_items+1, np.int64)
            codes[data.items['item'].to_numpy(dtype=np.int64)] = [vocabulary.get(value, 0) for value in values]
            result.append(codes)
        return result

    def _neighbour_available(self, labels):
        prior = self.degree > 0
        counts = np.bincount(labels[prior], minlength=int(labels.max())+1)
        available = (labels > 0) & ((counts[labels]-prior.astype(np.int64)) > 0)
        available[0] = False
        return available

    def build(self, instances, candidates):
        candidates = np.asarray(candidates)
        if candidates.ndim != 2 or candidates.shape[0] != len(instances) or not np.issubdtype(candidates.dtype, np.integer):
            raise ValueError('candidates must be an integer [queries, candidates] array')
        if np.any(candidates < 0) or np.any(candidates > self.data.n_items):
            raise ValueError('candidate ID outside graph vocabulary')
        if 'ts' in instances and (instances['ts'].to_numpy() < self.strict_before_ms).any():
            raise ValueError('queries precede the frozen graph cutoff')
        users = instances['user'].to_numpy(dtype=np.int64)
        lengths = instances['hist_len'].to_numpy(dtype=np.int64)
        if np.any(users < 1) or np.any(users > self.data.n_users):
            raise ValueError('query user outside graph vocabulary')
        histories = self.data.histories(users, lengths, 50)
        common = np.zeros((*candidates.shape, len(self.common_names)), np.float32)
        graph = np.zeros((*candidates.shape, len(self.graph_names)), np.float32)
        for row,(user,history,cand) in enumerate(zip(users,histories,candidates)):
            history = np.unique(history[history > 0]).astype(np.int64)
            own = self.by_user.getrow(user).indices
            degree = self.degree.copy()
            degree[own] -= 1
            degree = np.maximum(degree, 0)
            candidate_degree = degree[cand]
            common[row,:,0] = np.log1p(candidate_degree)
            common[row,:,1] = candidate_degree > 0
            common[row,:,2] = float((self.degree[history] > 0).mean()) if len(history) else 0
            common[row,:,3] = self.leaf[cand] > 0
            common[row,:,4] = float((self.leaf[history] > 0).mean()) if len(history) else 0
            common[row,:,5] = self.store[cand] > 0
            common[row,:,6] = float((self.store[history] > 0).mean()) if len(history) else 0
            common[row,:,7] = self.leaf_available[cand]
            common[row,:,8] = self.store_available[cand]
            if len(history):
                hmat = self.matrix[:,history]
                cmat = self.matrix[:,cand]
                own_h = np.asarray(hmat.getrow(user).toarray()).ravel()
                own_c = np.asarray(cmat.getrow(user).toarray()).ravel()
                overlap = (hmat.T @ cmat).toarray() - own_h[:,None]*own_c[None,:]
                overlap = np.maximum(overlap, 0)
                nonself = history[:,None] != cand[None,:]
                overlap *= nonself
                history_count = np.asarray(hmat.sum(axis=1)).ravel()
                history_count[user] = 0
                reviewer_count = np.asarray(cmat.T @ (history_count > 0).astype(np.float32)).ravel()
                # A self item must not serve as the sole history witness.
                for column in np.flatnonzero(np.isin(cand, history)):
                    reviewers = self.matrix[:,int(cand[column])].indices
                    reviewers = reviewers[reviewers != user]
                    reviewer_count[column] = np.sum(history_count[reviewers] > 1)
                graph[row,:,0] = np.log1p(reviewer_count)
                graph[row,:,1] = np.log1p(overlap.sum(axis=0))
                supported = (degree[history] > 0)[:,None] & nonself
                denominator = np.sqrt(degree[history,None]*candidate_degree[None,:])
                cosine = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
                graph[row,:,2] = cosine.sum(axis=0)/np.maximum(supported.sum(axis=0),1)
                for column,labels in ((3,self.leaf),(4,self.store)):
                    known = (labels[history,None] > 0) & nonself
                    matching = known & (labels[history,None] == labels[cand][None,:]) & (labels[cand][None,:] > 0)
                    graph[row,:,column] = matching.sum(axis=0)/np.maximum(known.sum(axis=0),1)
                graph[row,:,5] = (overlap > 0).sum(axis=0)/np.maximum(nonself.sum(axis=0),1)
            common[row,cand == 0] = 0
            graph[row,cand == 0] = 0
        if not np.isfinite(common).all() or not np.isfinite(graph).all():
            raise ValueError('graph feature builder produced nonfinite inputs')
        return common,graph
