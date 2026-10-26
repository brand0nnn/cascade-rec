"""Period-causal, batch-built ranker features for stored retrieval lists.

Candidate arrays stay grouped on disk. This builder only expands feature rows for
the current training/scoring batch, avoiding a many-gigabyte repeated history and
text-feature archive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData


@dataclass
class RankerFeatureBuilder:
    data: ProcessedData
    text: np.ndarray
    item_id_lut: np.ndarray
    leaf_lut: np.ndarray
    top_lut: np.ndarray
    store_lut: np.ndarray
    price_lut: np.ndarray
    review_lut: dict[str, np.ndarray]
    rating_lut: dict[str, np.ndarray]
    age_lut: dict[str, np.ndarray]
    item_cardinalities: dict[str, int]
    instance_table: pd.DataFrame
    score_mean: float
    score_std: float

    @classmethod
    def create(cls, data: ProcessedData, train_lists: Mapping[str, np.ndarray], text_path: str | Path | None = None) -> "RankerFeatureBuilder":
        """Prepare vocabularies and causal item tables using only ranker-train/Val cutoffs."""
        items = data.items.sort_values("item").reset_index(drop=True)
        n = data.n_items
        text_path = Path(text_path) if text_path else data.dir / "item_text_emb.npy"
        text = np.load(text_path, mmap_mode="r")
        if text.shape[0] != n + 1 or text.shape[1] != 384:
            raise ValueError(f"expected frozen text embeddings [{n + 1}, 384], got {text.shape}")

        # Item OOV is based only on the retriever-training interactions. Padding=0, OOV=1.
        before_ranker = np.bincount(
            data.interactions(["retriever_train"]) ["item"].to_numpy(), minlength=n + 1
        )
        item_id_lut = np.ones(n + 1, dtype=np.int64)
        warm = np.flatnonzero(before_ranker >= 2)
        item_id_lut[warm] = warm + 1
        item_id_lut[0] = 0

        def vocab_lut(values: pd.Series) -> tuple[np.ndarray, int]:
            vals = values.fillna("").astype(str).to_numpy()
            vocab = {v: i + 2 for i, v in enumerate(sorted(set(vals) - {""}))}
            lut = np.ones(n + 1, dtype=np.int64)
            lut[0] = 0
            for iid, value in enumerate(vals, start=1):
                lut[iid] = vocab.get(value, 1)
            return lut, len(vocab) + 2

        leaf_lut, leaf_card = vocab_lut(items["leaf_cat"])
        top_lut, top_card = vocab_lut(items["top_cat"])
        store_counts = items.groupby("store")["item"].nunique()
        frequent_stores = set(store_counts[store_counts >= 5].index.astype(str))
        store_values = items["store"].fillna("").astype(str).where(items["store"].fillna("").astype(str).isin(frequent_stores), "")
        store_lut, store_card = vocab_lut(store_values)

        prices = items["price"].to_numpy(dtype=np.float64)
        valid_price = prices[np.isfinite(prices)]
        quantiles = np.quantile(valid_price, np.linspace(0.1, 0.9, 9)) if len(valid_price) else np.zeros(9)
        price_lut = np.ones(n + 1, dtype=np.int64)  # 1 = missing
        price_lut[1:] = np.where(np.isfinite(prices), 2 + np.searchsorted(quantiles, np.nan_to_num(prices, nan=0.0), side="right"), 1)
        price_lut[0] = 0

        # Build statistics using interactions strictly before each period start.
        all_inter = pd.read_parquet(data.dir / "interactions.parquet")
        all_inter = all_inter.rename(columns={"timestamp": "ts"})
        t0 = int(data.split["t_ranker_train"])
        t1 = int(data.split["t1"])
        cutoffs = {"ranker_train": t0, "val": t1}
        review_lut: dict[str, np.ndarray] = {}
        rating_lut: dict[str, np.ndarray] = {}
        age_lut: dict[str, np.ndarray] = {}
        for period, cutoff in cutoffs.items():
            prior = all_inter[all_inter["ts"] < cutoff]
            prior_items = prior["item"].to_numpy()
            prior_ts = prior["ts"].to_numpy()
            counts = np.bincount(prior_items, minlength=n + 1).astype(np.int64)
            sums = np.bincount(prior_items, weights=prior["rating"].to_numpy(dtype=np.float64), minlength=n + 1)
            means = np.divide(sums, counts, out=np.zeros(n + 1, dtype=np.float64), where=counts > 0)
            review_lut[period] = np.minimum(15, 2 + np.floor(np.log2(counts + 1))).astype(np.int64)
            # 1 means no ratings; observed means are bucketed to half-star steps 2..11.
            rating_lut[period] = np.where(counts > 0, 1 + np.clip(np.rint(means), 1, 5), 1).astype(np.int64)
            # Only look at item appearances before this period. Using a global
            # first-seen timestamp would reveal whether an item appears later.
            first_ts_prior = np.full(n + 1, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(first_ts_prior, prior_items, prior_ts)
            age_days = np.maximum(0.0, (cutoff - first_ts_prior).astype(np.float64) / 86_400_000.0)
            age_bucket = np.where(
                counts > 0,
                2 + np.minimum(15, np.floor(np.log2(age_days + 1)).astype(np.int64)),
                0,
            )
            age_lut[period] = age_bucket.astype(np.int64)
        review_lut["ranker_train"][0] = rating_lut["ranker_train"][0] = age_lut["ranker_train"][0] = 0
        review_lut["val"][0] = rating_lut["val"][0] = age_lut["val"][0] = 0

        score_values = np.asarray(train_lists["score"][:, :-1], dtype=np.float32)
        score_values = score_values[np.isfinite(score_values)]
        score_mean = float(score_values.mean()) if score_values.size else 0.0
        score_std = float(score_values.std()) if score_values.size else 1.0
        if score_std < 1e-6:
            score_std = 1.0
        return cls(
            data, text, item_id_lut, leaf_lut, top_lut, store_lut, price_lut,
            review_lut, rating_lut, age_lut,
            {"item_id": n + 2, "leaf_category": leaf_card, "top_category": top_card,
             "store": store_card, "price_bucket": 12, "review_count_bucket": 16,
             "rating_bucket": 12, "item_age_bucket": 18, "history_length_bucket": 6,
             "recency_bucket": 8},
            data.instances.set_index("inst"), score_mean, score_std,
        )

    def build(self, lists: Mapping[str, np.ndarray], list_rows: np.ndarray, candidate_cols: np.ndarray, *, score_only: bool = False) -> dict[str, np.ndarray]:
        """Build aligned features for selected (list-row, candidate-column) pairs."""
        list_rows = np.asarray(list_rows, dtype=np.int64)
        candidate_cols = np.asarray(candidate_cols, dtype=np.int64)
        if list_rows.shape != candidate_cols.shape:
            raise ValueError("list_rows and candidate_cols must be aligned vectors")
        item = np.asarray(lists["cand"][list_rows, candidate_cols], dtype=np.int64)
        if np.any(item <= 0):
            raise ValueError("padding candidates cannot be passed to the ranker")
        if score_only:
            numeric = np.zeros((len(item), 4), dtype=np.float32)
            scores = np.asarray(lists["score"][list_rows, candidate_cols], dtype=np.float32)
            numeric[:, 0] = (scores - self.score_mean) / self.score_std
            return {"numeric": numeric}
        unique_rows, inverse = np.unique(list_rows, return_inverse=True)
        inst_ids = np.asarray(lists["inst"], dtype=np.int64)[unique_rows]
        inst_unique = self.instance_table.loc[inst_ids]
        users_unique = inst_unique["user"].to_numpy(dtype=np.int64)
        hist_len_unique = inst_unique["hist_len"].to_numpy(dtype=np.int64)
        histories_unique = self.data.histories(users_unique, hist_len_unique, 50)
        mapped_hist_unique = self.item_id_lut[histories_unique]
        hist_leaf_unique = self.leaf_lut[histories_unique]
        periods_unique = np.where(inst_unique["period"].to_numpy() == 1, "ranker_train", "val")
        inst = inst_unique.iloc[inverse]
        users = users_unique[inverse]
        hist_len = hist_len_unique[inverse]
        histories = histories_unique[inverse]
        mapped_hist = mapped_hist_unique[inverse]
        hist_leaf = hist_leaf_unique[inverse]
        periods = periods_unique[inverse]

        # Context features computed from the exact pre-target history.
        last_items = histories[:, -1]
        last_leaf = self.leaf_lut[last_items]
        last_store = self.store_lut[last_items]
        # Store-in-history depends on the candidate; compare against the unique history context.
        candidate_stores = self.store_lut[item]
        history_stores = self.store_lut[histories_unique]
        hist_store_match = ((candidate_stores > 1) & np.any(
            history_stores[inverse] == candidate_stores[:, None], axis=1
        )).astype(np.float32)
        history_text = np.asarray(self.text[histories_unique], dtype=np.float32)
        hist_mask = histories_unique > 0
        hist_mean_unique = (history_text * hist_mask[:, :, None]).sum(axis=1) / np.maximum(1, hist_mask.sum(axis=1))[:, None]
        hist_mean = hist_mean_unique[inverse]
        hist_norm = np.linalg.norm(hist_mean, axis=1)
        cand_text = np.asarray(self.text[item], dtype=np.float32)
        cand_norm = np.linalg.norm(cand_text, axis=1)
        cosine = np.divide((cand_text * hist_mean).sum(axis=1), cand_norm * hist_norm,
                           out=np.zeros(len(item), dtype=np.float32), where=(cand_norm * hist_norm) > 0)

        scores = np.asarray(lists["score"][list_rows, candidate_cols], dtype=np.float32)
        ts_unique = inst_unique["ts"].to_numpy(dtype=np.int64)
        recency_unique = np.zeros(len(unique_rows), dtype=np.float64)
        for r, (u, hl) in enumerate(zip(users_unique, hist_len_unique)):
            end = self.data.user_ptr[u] + hl
            recency_unique[r] = ts_unique[r] - self.data.seq_ts[end - 1]
        recency_ms = recency_unique[inverse]
        recency_days = recency_ms / 86_400_000.0
        recency_bucket = 2 + np.digitize(recency_days, [1, 7, 30, 90, 365])
        history_length_bucket = np.select([hist_len <= 2, hist_len <= 5, hist_len <= 10], [2, 3, 4], default=5)

        categorical = {
            "item_id": self.item_id_lut[item], "leaf_category": self.leaf_lut[item],
            "top_category": self.top_lut[item], "store": self.store_lut[item],
            "price_bucket": self.price_lut[item],
            "review_count_bucket": np.fromiter((self.review_lut[p][i] for p, i in zip(periods, item)), dtype=np.int64, count=len(item)),
            "rating_bucket": np.fromiter((self.rating_lut[p][i] for p, i in zip(periods, item)), dtype=np.int64, count=len(item)),
            "item_age_bucket": np.fromiter((self.age_lut[p][i] for p, i in zip(periods, item)), dtype=np.int64, count=len(item)),
            "history_length_bucket": history_length_bucket.astype(np.int64),
            "recency_bucket": recency_bucket.astype(np.int64),
            "history_item_ids": mapped_hist,
            "history_leaf_ids": hist_leaf,
            "numeric": np.column_stack(((scores - self.score_mean) / self.score_std,
                                         (self.leaf_lut[item] == last_leaf).astype(np.float32),
                                         hist_store_match, cosine)).astype(np.float32),
            "item_text": cand_text,
        }
        return categorical
