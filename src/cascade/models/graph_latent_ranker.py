"""Matched residual ranker with optional frozen factors and inductive graph input.

Every arm constructs the same modules in the same order. Ablation masks affect
only forward inputs; seed-specific parameter keys/shapes/initial values match.
Input banks are nonpersistent buffers and must be restored from their manifest.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GraphLatentRanker(nn.Module):
    ARMS = ('R1','R2','R3','R4','R5','R6','R7')

    def __init__(self,text,leaf,store,graph_bank,latent,latent_known,common_dim,graph_dim,arm):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError('unknown graph/latent ranker arm')
        self.arm = arm
        self.common_dim,self.graph_dim = int(common_dim),int(graph_dim)
        text = np.asarray(text,dtype=np.float32)
        graph_bank = np.asarray(graph_bank,dtype=np.float32)
        if text.ndim != 2 or text.shape[1] != 384 or graph_bank.shape != (len(text),1154):
            raise ValueError('expected padded text [N+1,384] and graph bank [N+1,1154]')
        if len(leaf) != len(text) or len(store) != len(text):
            raise ValueError('metadata lookup sizes differ')
        self.register_buffer('text',torch.as_tensor(text),persistent=False)
        self.register_buffer('leaf',torch.as_tensor(leaf,dtype=torch.long),persistent=False)
        self.register_buffer('store',torch.as_tensor(store,dtype=torch.long),persistent=False)
        self.register_buffer('graph_bank',torch.as_tensor(graph_bank),persistent=False)
        tables = latent if isinstance(latent,dict) else {'bpr':latent,'pca':latent}
        masks = latent_known if isinstance(latent_known,dict) else {'bpr':latent_known,'pca':latent_known}
        for name in ('bpr','pca'):
            table = np.asarray(tables[name],dtype=np.float32)
            known = np.asarray(masks[name],dtype=bool)
            if table.shape != (len(text),64) or known.shape != (len(text),):
                raise ValueError('factor bank/mask shape differs')
            # Padding remains unavailable even if the caller accidentally marks it known.
            known = known.copy();known[0] = False
            self.register_buffer(name+'_factors',torch.as_tensor(table),persistent=False)
            self.register_buffer(name+'_known',torch.as_tensor(known),persistent=False)
        self.text_projection = nn.Linear(384,32)
        self.latent_adapter = nn.Linear(64,16)
        self.graph_encoder = nn.Sequential(nn.Linear(1154,128),nn.GELU(),nn.Linear(128,64))
        self.graph_adapter = nn.Linear(64,16)
        self.head = nn.Sequential(nn.Linear(133+self.common_dim+self.graph_dim+128,128),nn.ReLU(),nn.Dropout(.1),
                                  nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _interactions(candidate,pooled):
        return torch.cat([candidate,pooled,candidate*pooled,(candidate-pooled).abs()],dim=-1)

    def _factor_block(self,candidates,history,name):
        table = getattr(self,name+'_factors')
        known = getattr(self,name+'_known')
        c_mask = known[candidates] & candidates.ne(0)
        h_mask = known[history] & history.ne(0)
        # Mask after the adapter: its bias may not manufacture an unseen factor.
        c = self.latent_adapter(table[candidates])*c_mask.unsqueeze(-1)
        h = self.latent_adapter(table[history])*h_mask.unsqueeze(-1)
        pooled = h.sum(1)/h_mask.sum(1,keepdim=True).clamp_min(1)
        return self._interactions(c,pooled[:,None,:].expand_as(c))

    def _graph_vectors(self,items):
        values = self.graph_bank[items]
        if self.arm == 'R6':
            values = torch.cat([values[...,:384],torch.zeros_like(values[...,384:])],dim=-1)
        encoded = self.graph_encoder(values)
        return self.graph_adapter(encoded)*items.ne(0).unsqueeze(-1)

    def _graph_block(self,candidates,history):
        c = self._graph_vectors(candidates)
        h = self._graph_vectors(history)
        valid = history.ne(0)
        pooled = h.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        return self._interactions(c,pooled[:,None,:].expand_as(c))

    def forward(self,candidates,history,scores,common,graph):
        if candidates.ndim != 2 or history.ndim != 2 or len(history) != len(candidates):
            raise ValueError('candidate/history tensors must be matched batches')
        if scores.shape != candidates.shape or common.shape != (*candidates.shape,self.common_dim) or graph.shape != (*candidates.shape,self.graph_dim):
            raise ValueError('score or feature tensor dimensions differ')
        valid = history.ne(0)
        c = self.text_projection(self.text[candidates])*candidates.ne(0).unsqueeze(-1)
        h = self.text_projection(self.text[history])*valid.unsqueeze(-1)
        pooled = h.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        text_features = self._interactions(c,pooled[:,None,:].expand_as(c))
        candidate_leaf,last_leaf = self.leaf[candidates],self.leaf[history[:,-1]]
        leaf_known = candidate_leaf.gt(0) & last_leaf[:,None].gt(0)
        leaf_match = leaf_known & candidate_leaf.eq(last_leaf[:,None])
        candidate_store,history_store = self.store[candidates],self.store[history]
        store_known = candidate_store.gt(0) & (history_store.gt(0)&valid).any(1)[:,None]
        store_match = store_known & ((candidate_store[:,:,None] == history_store[:,None,:])&valid[:,None,:]).any(-1)
        raw_hist = (self.text[history]*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        cosine = F.cosine_similarity(self.text[candidates],raw_hist[:,None,:],dim=-1)
        numeric = torch.stack([leaf_match,leaf_known,store_match,store_known,cosine],dim=-1).float()
        factor_name = 'bpr' if self.arm in ('R3','R4') else 'pca' if self.arm == 'R5' else None
        factors = self._factor_block(candidates,history,factor_name) if factor_name else c.new_zeros((*candidates.shape,64))
        encoded_graph = self._graph_block(candidates,history) if self.arm in ('R6','R7') else c.new_zeros((*candidates.shape,64))
        graph_features = graph if self.arm in ('R2','R4') else torch.zeros_like(graph)
        features = torch.cat([text_features,numeric,common,graph_features,factors,encoded_graph],dim=-1)
        # Frozen retrieval scores enter only through this skip, never as head inputs.
        result = scores+self.head(features).squeeze(-1)
        return result.masked_fill(candidates.eq(0),-torch.inf)
