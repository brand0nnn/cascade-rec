"""RecBole-backed baselines: Pop, ItemKNN, BPR-MF, SASRec (CLAUDE.md §6; see docs/decisions.md 2026-09-28).

RecBole handles model definitions, training loops and early stopping (on a fixed Val subset). Final
Val/Test numbers are *not* taken from RecBole's evaluator: each trained model is wrapped in a scorer that
implements ``score_all`` and is evaluated by ``cascade.eval.harness`` like every other model.

Our split is exported as RecBole benchmark atomic files (train / valid / test). The RecBole "test" file is a
copy of the valid file, so RecBole never sees Test data.

Item tokens are our integer item ids as strings; a ``.item`` file lists every item so all items get a
RecBole id (including items that never appear in the training file).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.general_recommender.itemknn import ComputeSimilarity
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.utils import InputType, ModelType

from cascade.data.instances import ProcessedData
from cascade.eval.harness import InstanceBatch
from cascade.utils.logging import get_logger

log = get_logger(__name__)

TRAIN_PERIODS = {"AB": ["A", "B"], "A": ["A"]}


# --------------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------------
def _hist_str(data: ProcessedData, inst: pd.DataFrame) -> list[str]:
    H = data.histories(inst["user"].values, inst["hist_len"].values, data.max_hist_len)
    return [" ".join(map(str, row[row > 0])) for row in H]


def export_atomic(data: ProcessedData, root: Path, kind: str, train_on: str, val_subset: int, subset_seed: int = 0) -> str:
    """Write RecBole benchmark files for ``kind`` in {general, sequential}. Returns the dataset name.

    general:    train = all interactions in the training periods (incl. users' first interactions);
    sequential: train = every training-period instance (history = last L items before t, target = item),
                i.e. the same (prefix -> next item) pairs RecBole's own augmentation would build.
    valid = a fixed random subset of Val instances (early stopping only); test = copy of valid.
    """
    name = f"{kind}_{train_on}_v{val_subset}"
    d = root / name
    if (d / f"{name}.test.inter").exists():
        return name
    d.mkdir(parents=True, exist_ok=True)
    val = data.split_instances("val")
    if val_subset and val_subset < len(val):
        rng = np.random.default_rng(subset_seed)
        val = val.iloc[np.sort(rng.choice(len(val), val_subset, replace=False))].reset_index(drop=True)

    if kind == "general":
        tr = data.interactions(TRAIN_PERIODS[train_on])
        train = pd.DataFrame({"user_id:token": tr["user"], "item_id:token": tr["item"], "timestamp:float": tr["ts"]})
        valid = pd.DataFrame({"user_id:token": val["user"], "item_id:token": val["item"], "timestamp:float": val["ts"]})
    elif kind == "sequential":
        tr = data.instances[data.instances["period"].isin([{"A": 0, "B": 1}[p] for p in TRAIN_PERIODS[train_on]])]
        train = pd.DataFrame({"user_id:token": tr["user"], "item_id_list:token_seq": _hist_str(data, tr), "item_id:token": tr["item"], "timestamp:float": tr["ts"]})
        valid = pd.DataFrame({"user_id:token": val["user"], "item_id_list:token_seq": _hist_str(data, val), "item_id:token": val["item"], "timestamp:float": val["ts"]})
    else:
        raise ValueError(kind)
    train.to_csv(d / f"{name}.train.inter", sep="\t", index=False)
    valid.to_csv(d / f"{name}.valid.inter", sep="\t", index=False)
    valid.to_csv(d / f"{name}.test.inter", sep="\t", index=False)  # RecBole never sees Test
    pd.DataFrame({"item_id:token": np.arange(1, data.n_items + 1)}).to_csv(d / f"{name}.item", sep="\t", index=False)
    log.info("exported %s: train=%d valid=%d", name, len(train), len(valid))
    return name


def recbole_config_dict(kind: str, data_root: Path, dataset: str) -> dict:
    """Dataset-related RecBole settings shared by all baselines."""
    cfg = {
        "data_path": str(data_root),
        "dataset": dataset,
        "benchmark_filename": ["train", "valid", "test"],
        "field_separator": "\t",
        "seq_separator": " ",
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"], "item": ["item_id"]},
        "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO", "group_by": "user", "mode": "full"},
        "metrics": ["Recall", "NDCG"],
        "topk": [10, 100],
        "valid_metric": "Recall@100",
        "reproducibility": True,
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "log_wandb": False,
    }
    if kind == "sequential":
        cfg["load_col"]["inter"].append("item_id_list")
        cfg["alias_of_item_id"] = ["item_id_list"]
        cfg["MAX_ITEM_LIST_LENGTH"] = 50
    return cfg


# --------------------------------------------------------------------------------------------------
# Model tweaks
# --------------------------------------------------------------------------------------------------
class SASRecBCE(SASRec):
    """RecBole SASRec with the original paper's loss: binary cross-entropy with 1 sampled negative.

    RecBole only ships 'CE' and 'BPR'; set ``loss_type: BPR`` in the config so the dataloader samples one
    negative per row (NEG_ITEM_ID), and this override replaces the BPR loss by BCE (Kang & McAuley, 2018).
    """

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        pos = (seq_output * self.item_embedding(interaction[self.POS_ITEM_ID])).sum(-1)
        neg = (seq_output * self.item_embedding(interaction[self.NEG_ITEM_ID])).sum(-1)
        return F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))


class PopExact(GeneralRecommender):
    """Popularity = #training interactions per item, read from RecBole's training interaction matrix.

    RecBole's ``Pop`` counts inside ``calculate_loss`` with ``item_cnt[item] += 1``, which (a) also counts the
    sampled negatives when negative sampling is on (the default) and (b) does not accumulate duplicate
    items within a batch. Both bias the counts, so we count directly.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        X = dataset.inter_matrix(form="csr")
        self.item_cnt = torch.from_numpy(np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32))
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["item_cnt"]

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def full_sort_predict(self, interaction):
        raise NotImplementedError("PopExact is scored through cascade.eval.harness")


class ItemKNNLite(GeneralRecommender):
    """RecBole ItemKNN without the precomputed all-users prediction matrix (too large at 381K users).

    Uses RecBole's ``ComputeSimilarity`` (cosine, top-k neighbours per item) unchanged; scoring is done by
    ``ItemKNNScorer`` from each instance's history. RecBole validation is skipped for this model.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        X = dataset.inter_matrix(form="csr").astype(np.float32)
        X.data[:] = 1.0
        _, self.w = ComputeSimilarity(X, topk=config["k"], shrink=config["shrink"], method="item").compute_similarity()
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w"]

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def full_sort_predict(self, interaction):
        raise NotImplementedError("ItemKNNLite is scored through cascade.eval.harness")


# --------------------------------------------------------------------------------------------------
# Scorers for our harness
# --------------------------------------------------------------------------------------------------
def left_to_right_pad(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, L] left-padded sequences -> (right-padded sequences, lengths), RecBole's layout."""
    lens = (H > 0).sum(1)
    L = H.shape[1]
    idx = (torch.arange(L).unsqueeze(0) + (L - lens).unsqueeze(1)) % L
    R = torch.gather(H, 1, idx)
    R[torch.arange(L).unsqueeze(0) >= lens.unsqueeze(1)] = 0
    return R, lens


class _RecBoleScorer:
    """Maps between our item ids (0 = pad) and RecBole internal ids."""

    def __init__(self, model, dataset, n_items: int):
        self.model = model.eval()
        self.dataset = dataset
        self.device = next(model.parameters()).device
        tokens = np.arange(1, n_items + 1).astype(str)
        rb = dataset.token2id("item_id", tokens)
        self.ours2rb = torch.zeros(n_items + 1, dtype=torch.long)
        self.ours2rb[1:] = torch.from_numpy(np.asarray(rb, dtype=np.int64))
        assert len(set(rb)) == n_items, "every item must have a RecBole id"

    def _to_ours(self, rb_scores: torch.Tensor) -> torch.Tensor:
        return rb_scores[:, self.ours2rb].cpu()


class PopScorer(_RecBoleScorer):
    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        cnt = self.model.item_cnt.view(1, -1).float()
        return self._to_ours(cnt).expand(len(batch.users), -1).clone()


class ItemKNNScorer(_RecBoleScorer):
    """score_j = sum over the last L history items h of W[h, j] (same as RecBole's X · W)."""

    def __init__(self, model, dataset, n_items):
        super().__init__(model, dataset, n_items)
        self.w = sp.csr_matrix(model.w)

    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        H = self.ours2rb[batch.hist]  # pad (0) -> 0 = RecBole pad
        rows = np.repeat(np.arange(H.shape[0]), H.shape[1])
        cols = H.numpy().ravel()
        m = cols > 0
        X = sp.csr_matrix((np.ones(m.sum(), np.float32), (rows[m], cols[m])), shape=(H.shape[0], self.w.shape[0]))
        return self._to_ours(torch.from_numpy((X @ self.w).toarray()))


class BPRScorer(_RecBoleScorer):
    """User embedding · item embedding. Users without training interactions are unscorable (skipped + logged)."""

    def __init__(self, model, dataset, n_items, train_users: np.ndarray):
        super().__init__(model, dataset, n_items)
        self.train_users = train_users

    def _uid(self, users):
        """RecBole user ids; users unknown to RecBole (only in Test) map to the padding id 0."""
        tok2id = self.dataset.field2token_id["user_id"]
        ids = [tok2id.get(str(u), 0) for u in users]
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def scorable(self, batch: InstanceBatch) -> np.ndarray:
        return np.isin(batch.users, self.train_users)

    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        u = self.model.user_embedding(self._uid(batch.users))
        return self._to_ours(u @ self.model.item_embedding.weight.T)


class SASRecScorer(_RecBoleScorer):
    """RecBole SASRec on the last L history items (converted to RecBole's right-padded layout)."""

    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        R, lens = left_to_right_pad(self.ours2rb[batch.hist])
        seq_out = self.model.forward(R.to(self.device), lens.to(self.device))
        return self._to_ours(seq_out @ self.model.item_embedding.weight.T)
