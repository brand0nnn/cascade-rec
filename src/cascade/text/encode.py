"""Encode every item's text once with a frozen sentence encoder (CLAUDE.md §7.1).

    python -m cascade.text.encode [--toy]

Writes ``<processed_dir>/item_text_emb.npy``: float32 [N+1, 384], L2-normalised, row = item id
(row 0 = padding, all zeros) plus ``item_text_emb.json`` with the model name and a hash of the texts.
Toy mode reuses the full-data embeddings (matched on parent_asin) when they exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from cascade.data.preprocess import output_dir
from cascade.utils.config import load_config
from cascade.utils.logging import get_logger

log = get_logger(__name__)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def text_hash(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]


def encode_texts(texts: list[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def main(config_path: str, toy: bool, batch_size: int, device: str) -> None:
    cfg = load_config(config_path)
    d = output_dir(cfg, toy)
    items = pd.read_parquet(d / "items.parquet").sort_values("item")
    assert (items["item"].values == np.arange(1, len(items) + 1)).all()
    texts = items["text"].tolist()

    full_dir = output_dir(cfg, False)
    if toy and (full_dir / "item_text_emb.npy").exists():
        full_items = pd.read_parquet(full_dir / "items.parquet")
        full_emb = np.load(full_dir / "item_text_emb.npy")
        row = pd.Series(full_items["item"].values, index=full_items["parent_asin"].values).loc[items["parent_asin"].values].values
        emb = full_emb[row]
        log.info("toy: reused full-data embeddings for %d items", len(items))
    else:
        emb = encode_texts(texts, MODEL, batch_size, device)
    out = np.zeros((len(items) + 1, emb.shape[1]), dtype=np.float32)
    out[1:] = emb
    np.save(d / "item_text_emb.npy", out)
    (d / "item_text_emb.json").write_text(json.dumps({"model": MODEL, "n_items": len(items), "dim": int(emb.shape[1]), "text_hash": text_hash(texts), "normalized": True}, indent=2))
    log.info("saved %s %s", d / "item_text_emb.npy", out.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    main(a.config, a.toy, a.batch_size, a.device)
