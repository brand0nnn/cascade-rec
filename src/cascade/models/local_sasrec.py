"""Opt-in local content experiments; historical RecBole models stay unchanged."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn
from recbole.sampler import Sampler

from cascade.models.recbole_baselines import SASRecBCE


class PriorItemSampler(Sampler):
    """Uniform pre-cutoff items, retaining RecBole's train-user exclusions."""

    def __init__(self, dataset, allowed_ids):
        self.allowed_ids = np.unique(np.asarray(allowed_ids, dtype=np.int64))
        if not len(self.allowed_ids) or np.any(self.allowed_ids <= 0) or np.any(self.allowed_ids >= dataset.item_num):
            raise ValueError('negative pool must contain valid non-padding item IDs')
        super().__init__(['train'], [dataset], distribution='uniform')
        allowed = set(self.allowed_ids.tolist())
        for used in self.used_ids['train']:
            if allowed <= used:
                raise ValueError('a training user exhausts the restricted negative pool')
        self.returned = 0
        self.first_negatives: list[int] = []

    def _uni_sampling(self, sample_num):
        return self.allowed_ids[np.random.randint(len(self.allowed_ids), size=sample_num)]

    def sample_by_user_ids(self, user_ids, item_ids, num):
        result = super().sample_by_user_ids(user_ids, item_ids, num)
        if not np.isin(result.numpy(), self.allowed_ids).all():
            raise ValueError('negative escaped the pre-cutoff pool')
        self.returned += result.numel()
        remaining = max(0, 1024 - len(self.first_negatives))
        self.first_negatives.extend(result.tolist()[:remaining])
        return result


class ContentItems(nn.Module):
    """Frozen metadata vectors with a learned shared, ID-independent projection."""

    def __init__(self, features, dim: int, seed: int, residual=None, seen=None):
        super().__init__()
        self.register_buffer('features', torch.as_tensor(features, dtype=torch.float32), persistent=False)
        self.residual = residual
        self.register_buffer('seen', None if seen is None else torch.as_tensor(seen, dtype=torch.float32), persistent=False)
        # Preserve the ordinary dropout/negative-sampling RNG stream of the ID control.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 104729)
            self.projection = nn.Sequential(nn.Linear(features.shape[1], 256), nn.GELU(), nn.Linear(256, dim))
            for layer in self.projection:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=.02)
                    nn.init.zeros_(layer.bias)
        self._cache = None

    def train(self, mode=True):
        self._cache = None
        return super().train(mode)

    @property
    def weight(self):
        if not self.training and not torch.is_grad_enabled() and self._cache is not None:
            return self._cache
        out = self.projection(self.features)
        if self.residual is not None:
            out = out + self.residual.weight * self.seen[:, None]
        out = torch.cat([torch.zeros_like(out[:1]), out[1:]], dim=0)
        if not self.training and not torch.is_grad_enabled():
            self._cache = out
        return out

    def forward(self, items):
        if not self.training and not torch.is_grad_enabled():
            return self.weight[items]
        out = self.projection(self.features[items])
        if self.residual is not None:
            out = out + self.residual(items) * self.seen[items].unsqueeze(-1)
        return out * (items > 0).unsqueeze(-1)


class ContentSASRec(SASRecBCE):
    """Same RecBole sequence backbone and BCE, with content replacing item IDs."""

    def __init__(self, config, dataset, features, seen=None):
        super().__init__(config, dataset)
        residual = self.item_embedding if seen is not None else None
        self.item_embedding = ContentItems(features, self.hidden_size, int(config['seed']), residual, seen)


def backbone_digest(model) -> str:
    value = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        if not key.startswith('item_embedding.') and key != 'interest_queries':
            value.update(key.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()
