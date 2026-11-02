"""Additional local variants: inductive metadata neighbours and two interests."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cascade.models.local_sasrec import ContentSASRec
from cascade.models.recbole_baselines import SASRecScorer


def metadata_graph_features(items, vectors, cutoff: int, limit: int = 16):
    """Mean content of seeded pre-T0 neighbours; later items never supply messages."""
    items = items.sort_values('item')
    prior = items.first_ts.to_numpy() < cutoff
    output = np.zeros((len(vectors), vectors.shape[1] * 3 + 2), np.float32)
    output[:, :vectors.shape[1]] = vectors
    neighbours = np.zeros((len(vectors), 2, limit), np.int32)
    rng = np.random.default_rng(0)
    width = vectors.shape[1]
    for axis, column in enumerate(('leaf_cat', 'store')):
        groups = {}
        for label, group in items.loc[prior].groupby(column, sort=True):
            if isinstance(label, str) and label.strip():
                groups[label] = rng.permutation(group.item.to_numpy())[:limit+1]
        for item, label in zip(items.item, items[column]):
            ids = groups.get(label, np.array([], dtype=int))
            ids = ids[ids != item][:limit]
            if len(ids):
                output[item, (axis+1)*width:(axis+2)*width] = vectors[ids].mean(0)
                output[item, 3*width+axis] = 1
                neighbours[item, axis, :len(ids)] = ids
    return output, neighbours


class MultiInterestSASRec(ContentSASRec):
    """Two masked pools of causal states; inference and BCE both use max-head scores."""

    def __init__(self, config, dataset, features):
        super().__init__(config, dataset, features)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config['seed']) + 130363)
            self.interest_queries = nn.Parameter(torch.randn(2, self.hidden_size) * .02)

    def forward(self, item_seq, item_seq_len):
        positions = torch.arange(item_seq.shape[1], device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        x = self.item_embedding(item_seq) + self.position_embedding(positions)
        x = self.dropout(self.LayerNorm(x))
        states = self.trm_encoder(x, self.get_attention_mask(item_seq), output_all_encoded_layers=True)[-1]
        valid = item_seq.ne(0)
        logits = torch.einsum('bld,hd->bhl', states, self.interest_queries) / math.sqrt(self.hidden_size)
        weights = logits.masked_fill(~valid[:, None, :], -10000).softmax(-1) * valid[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum('bhl,bld->bhd', weights, states)

    def calculate_loss(self, interaction):
        states = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        positive = self.item_embedding(interaction[self.POS_ITEM_ID])
        negative = self.item_embedding(interaction[self.NEG_ITEM_ID])
        pos = (states * positive[:, None, :]).sum(-1).amax(1)
        neg = (states * negative[:, None, :]).sum(-1).amax(1)
        return F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))

    def full_sort_predict(self, interaction):
        states = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        return (states @ self.item_embedding.weight.T).amax(1)

    def predict(self, interaction):
        states = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        item = self.item_embedding(interaction[self.ITEM_ID])
        return (states * item[:, None, :]).sum(-1).amax(1)


class LocalSASRecScorer(SASRecScorer):
    def score_all(self, batch):
        from cascade.models.recbole_baselines import left_to_right_pad
        sequence, length = left_to_right_pad(self.ours2rb[batch.hist])
        states = self.model.forward(sequence.to(self.device), length.to(self.device))
        scores = states @ self.model.item_embedding.weight.T
        if scores.ndim == 3:
            scores = scores.amax(1)
        return self._to_ours(scores)
