"""Project DIN adaptation: candidate-conditioned, unnormalised history pooling.

Recipe: Zhou et al. (2018), https://arxiv.org/abs/1706.06978;
checked against FuxiCTR DIN_Attention (use_softmax=False). The project retains
ReLU, BCE, and the shared MLP head rather than reproducing Dice/MB-aware L2.
"""
from __future__ import annotations
from typing import Mapping
import torch
from torch import nn
from cascade.models.ranker_mlp import MLPRanker, RankerFeatureSpec


class DINAttention(nn.Module):
    """Local activation over [query, history, query-history, query*history].

    Scores are signed, unnormalised coefficients, not attention probabilities.
    Padding receives exactly zero weight; an empty history yields a zero vector.
    """
    def __init__(self, embedding_dim: int = 48, hidden_dims: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        dims = (4 * embedding_dim,) + tuple(hidden_dims)
        layers: list[nn.Module] = []
        for before, after in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(before, after), nn.ReLU()))
        layers.append(nn.Linear(dims[-1], 1))
        self.activation = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or query.shape != (history.shape[0], self.embedding_dim) or history.shape[-1] != self.embedding_dim or mask.shape != history.shape[:2]:
            raise ValueError('query [batch, dim], history [batch, length, dim], and mask [batch, length] must align')
        q = query[:, None, :].expand_as(history)
        interactions = torch.cat((q, history, q-history, q*history), dim=-1)
        weights = self.activation(interactions).squeeze(-1)
        weights = weights.masked_fill(~mask.bool(), 0)
        pooled = torch.bmm(weights.unsqueeze(1), history).squeeze(1)
        return pooled, weights


class DINRanker(MLPRanker):
    """Same feature groups/head as MLP, with candidate-specific history weights.

    The item/category query is used only inside attention when attributes are
    disabled. This explicitly adds candidate identity interactions absent from
    matching-only mean pooling; conclusions concern this combined mechanism.
    """
    query_feature_names = ('item_id', 'leaf_category')

    def __init__(self, spec: RankerFeatureSpec, dropout: float = 0.1,
                 attention_hidden_dims: tuple[int, ...] = (64, 32)) -> None:
        super().__init__(spec, dropout)
        if not self.use_matching:
            raise ValueError('DIN needs the matching feature group/history embeddings')
        self.attention = DINAttention(self.embedding_dims['item_id'] + self.embedding_dims['leaf_category'], attention_hidden_dims)

    def history_attention(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.cat((self.embeddings['item_id'](batch['history_item_ids'].long()),
                             self.embeddings['leaf_category'](batch['history_leaf_ids'].long())), dim=-1)
        query = torch.cat((self.embeddings['item_id'](batch['item_id'].long()),
                           self.embeddings['leaf_category'](batch['leaf_category'].long())), dim=-1)
        return self.attention(query, history, batch['history_item_ids'] != 0)

    def pool_history(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.history_attention(batch)[0]
