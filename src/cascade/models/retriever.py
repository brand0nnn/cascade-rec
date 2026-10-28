"""Two-tower retriever (CLAUDE.md §7): text item tower + SASRec-style user tower,
trained with in-batch softmax and logQ correction (Yi et al., 2019).

Item tower:  e_i = MLP(frozen MiniLM text emb)            (variant ``text``)
             e_i = MLP(text) + ID emb (items seen in retriever-train)   (variant ``id_text``)
             e_i = ID emb (items seen in retriever-train, else 0)       (variant ``id_only``)
             v_i = normalize(e_i)                          (scoring vector)
User tower:  causal transformer over the last L item vectors e_i (shared item tower), left-padded,
             u = normalize(hidden state at the last position)  (variant ``transformer``)
             u = normalize(mean of non-padded e_i)             (variant ``meanpool``)
Score:       s(u, i) = u · v_i / τ
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cascade.eval.harness import InstanceBatch


class ItemTower(nn.Module):
    def __init__(self, variant: str, text_emb: np.ndarray | None, n_items: int, seen_in_retriever_train: np.ndarray, dim: int = 64, hidden: int = 256, dropout: float = 0.1):
        """``text_emb``: [N+1, 384] frozen embeddings (row 0 = padding). ``seen_in_retriever_train``: bool [N+1], items with ≥1 interaction in the retriever-training period."""
        super().__init__()
        assert variant in ("text", "id_text", "id_only")
        self.variant = variant
        if variant in ("text", "id_text"):
            self.register_buffer("text_emb", torch.as_tensor(text_emb, dtype=torch.float32), persistent=False)
            self.mlp = nn.Sequential(nn.Linear(self.text_emb.shape[1], hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        if variant in ("id_text", "id_only"):
            self.id_emb = nn.Embedding(n_items + 1, dim)
            nn.init.normal_(self.id_emb.weight, std=0.02)
            # items never seen in the retriever-training period get no ID signal (cold items fall back to text, or 0 for id_only)
            self.register_buffer("id_mask", torch.as_tensor(seen_in_retriever_train, dtype=torch.float32).unsqueeze(1), persistent=False)

    def forward(self, items: torch.Tensor, *, use_id: bool = True, id_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Unnormalised item vectors e_i for an id tensor of any shape (padding id 0 → 0 vector)."""
        out = 0
        if self.variant in ("text", "id_text"):
            out = self.mlp(self.text_emb[items])
        if self.variant in ("id_text", "id_only"):
            residual = self.id_emb(items) * self.id_mask[items]
            if not use_id:
                residual = residual * 0
            elif id_keep_mask is not None:
                residual = residual * id_keep_mask[items].unsqueeze(-1)
            out = out + residual
        return out * (items > 0).unsqueeze(-1).float()


class UserTower(nn.Module):
    def __init__(self, variant: str, dim: int = 64, max_len: int = 50, n_blocks: int = 2, n_heads: int = 1, ffn_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        assert variant in ("transformer", "meanpool")
        self.variant = variant
        if variant == "transformer":
            self.pos_emb = nn.Embedding(max_len, dim)
            self.in_drop = nn.Dropout(dropout)
            layer = nn.TransformerEncoderLayer(dim, n_heads, ffn_dim, dropout, activation="relu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(dim)
            self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1), persistent=False)

    def sequence(self, e: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Hidden state at every position: e [B, L, d] item vectors, pad [B, L] True at padding → [B, L, d].

        Position p only sees positions ≤ p (causal), so its output summarises the history up to p.
        """
        B, L, _ = e.shape
        if self.variant == "meanpool":
            m = (~pad).float().unsqueeze(-1)
            return torch.cumsum(e * m, 1) / torch.cumsum(m, 1).clamp(min=1)
        x = self.in_drop(e + self.pos_emb.weight[-L:].unsqueeze(0))
        x = x * (~pad).unsqueeze(-1).float()
        # causal + key-padding mask; every query may attend to itself so padded rows never become all-masked (NaN)
        mask = self.causal[:L, :L].unsqueeze(0) | pad.unsqueeze(1)
        mask = mask & ~torch.eye(L, dtype=torch.bool, device=e.device).unsqueeze(0)
        x = self.blocks(x, mask=mask.repeat_interleave(self.blocks.layers[0].self_attn.num_heads, 0))
        return self.final_norm(x)


class TwoTowerRetriever(nn.Module):
    def __init__(self, item_variant: str, user_variant: str, text_emb: np.ndarray | None, n_items: int, seen_in_retriever_train: np.ndarray, max_len: int = 50, temperature: float = 0.05, id_dropout: float = 0.0, **kw):
        super().__init__()
        if not 0 <= id_dropout < 1:
            raise ValueError("id_dropout must be in [0, 1)")
        if id_dropout and item_variant != "id_text":
            raise ValueError("ID residual dropout requires the text + ID item tower")
        self.item_tower = ItemTower(item_variant, text_emb, n_items, seen_in_retriever_train, **kw.get("item", {}))
        self.user_tower = UserTower(user_variant, max_len=max_len, **kw.get("user", {}))
        self.temperature = temperature
        self.n_items = n_items
        self.id_dropout = float(id_dropout)

    def user_sequence(self, hist: torch.Tensor, *, use_id: bool = True, id_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalised user vectors at every position of left-padded histories [B, L] → [B, L, d]."""
        return F.normalize(self.user_tower.sequence(self.item_tower(hist, use_id=use_id, id_keep_mask=id_keep_mask), hist == 0), dim=-1)

    def user_vectors(self, hist: torch.Tensor, *, use_id: bool = True) -> torch.Tensor:
        """u for each history (last position; histories are left-padded so the last position is the most recent)."""
        return self.user_sequence(hist, use_id=use_id)[:, -1]

    def item_vectors(self, items: torch.Tensor, *, use_id: bool = True, id_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.item_tower(items, use_id=use_id, id_keep_mask=id_keep_mask), dim=-1)

    def batch_id_keep_mask(self, seq_in: torch.Tensor, seq_tgt: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """One inverted-dropout scalar per item, shared by history and target calls.

        A dedicated CPU generator keeps ordinary dropout and negative-position RNG
        streams unchanged between controlled training arms.
        """
        ids = torch.unique(torch.cat((seq_in.reshape(-1), seq_tgt.reshape(-1))))
        ids = ids[ids > 0]
        mask = torch.ones(self.n_items + 1, device=seq_in.device)
        if self.training and self.id_dropout:
            keep = (torch.rand(len(ids), generator=generator, device="cpu") >= self.id_dropout).float()
            mask[ids] = (keep / (1 - self.id_dropout)).to(seq_in.device)
        mask[0] = 0
        return mask

    @torch.no_grad()
    def all_item_vectors(self, chunk: int = 32768, *, use_id: bool = True) -> torch.Tensor:
        """v_i for every item id 0..N (row 0 = padding = 0 vector)."""
        dev = next(self.parameters()).device
        ids = torch.arange(self.n_items + 1, device=dev)
        return torch.cat([self.item_vectors(ids[s : s + chunk], use_id=use_id) for s in range(0, len(ids), chunk)])


def inbatch_softmax_loss(u: torch.Tensor, v: torch.Tensor, targets: torch.Tensor, temperature: float, log_q: torch.Tensor | None) -> torch.Tensor:
    """In-batch sampled softmax with logQ correction and duplicate masking (§7.3).

    u [P, d] user vectors, v [P, d] vectors of each pair's target, targets [P] item ids.
    logits[i, j] = u_i · v_j / τ − log q(target_j); the positive is the diagonal. If target_j == target_i for j ≠ i
    the logit is masked to −inf (the same item must not act as its own negative).
    """
    logits = u @ v.T / temperature
    if log_q is not None:
        logits = logits - log_q[targets].unsqueeze(0)
    dup = (targets.unsqueeze(0) == targets.unsqueeze(1)) & ~torch.eye(len(targets), dtype=torch.bool, device=u.device)
    logits = logits.masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(targets), device=u.device))


def training_loss(model: TwoTowerRetriever, seq_in: torch.Tensor, seq_tgt: torch.Tensor, log_q: torch.Tensor | None, max_inbatch: int, generator: torch.Generator | None = None, id_dropout_generator: torch.Generator | None = None) -> torch.Tensor:
    """SASRec-style training: every non-padded position p of ``seq_in`` predicts ``seq_tgt[:, p]`` (one forward pass)."""
    id_keep_mask = None
    if model.training and model.id_dropout:
        if id_dropout_generator is None:
            raise ValueError("ID dropout requires its dedicated CPU generator")
        id_keep_mask = model.batch_id_keep_mask(seq_in, seq_tgt, id_dropout_generator)
    U = model.user_sequence(seq_in, id_keep_mask=id_keep_mask)  # [B, L, d]
    valid = (seq_in > 0) & (seq_tgt > 0)
    u, tgt = U[valid], seq_tgt[valid]
    if len(tgt) > max_inbatch:
        keep = torch.randperm(len(tgt), generator=generator, device="cpu")[:max_inbatch].to(u.device)
        u, tgt = u[keep], tgt[keep]
    return inbatch_softmax_loss(u, model.item_vectors(tgt, id_keep_mask=id_keep_mask), tgt, model.temperature, log_q)


class RetrieverScorer:
    """Exact-search scoring of the full catalogue for the evaluation harness (u · v_i for every item)."""

    def __init__(self, model: TwoTowerRetriever, *, history_use_id: bool = True, candidate_use_id: bool = True):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.history_use_id = history_use_id
        self.V = model.all_item_vectors(use_id=candidate_use_id)

    @torch.no_grad()
    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        u = self.model.user_vectors(batch.hist.to(self.device), use_id=self.history_use_id)
        return (u @ self.V.T).float().cpu()
