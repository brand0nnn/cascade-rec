"""Candidate lists from the frozen retriever with exact search (CLAUDE.md §9.1).

    python -m cascade.pipelines.build_candidates --run-dir results/retriever/two_tower_text/seed0 [--sample] [--device cuda]

Writes ``data/processed[/sample]/candidates/<name>_seed<k>/``:
  ranker_train.npz  ranker-training lists: top-``k_train`` candidates, target force-inserted if missing (label 1)
  val.npz   top-``k_eval`` candidates, never forced (ranker-val lists are derived from it by ``load_candidates``)
  test.npz  top-``k_eval`` candidates, never forced
  meta.json source run, git hash, list sizes, retrieval Recall@K per split
Each npz: inst, cand [n, k] int32 (sorted by retrieval score, 0 = padding), score [n, k] float16,
target_pos [n] (column of the target in ``cand``, -1 if absent), target_rank [n] (target's rank in the full
filtered catalogue). Ranker-training lists have one extra column holding the forced-in target when it was missed (else 0).
Top-K₁ lists for any K₁ <= k_eval are prefixes of the stored lists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cascade.data.instances import ProcessedData
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.models.retriever import RetrieverScorer
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.io import git_hash
from cascade.utils.logging import get_logger

log = get_logger(__name__)

RECALL_CUTOFFS = (10, 50, 100, 200, 500)


def retrieve(scorer, data: ProcessedData, inst, k: int, batch_size: int = 512, show_progress: bool = True) -> dict:
    """Exact top-k after §4.6 filtering, plus the target's position and full filtered rank."""
    first_ts = first_ts_tensor(data)
    n = len(inst)
    cand = np.zeros((n, k), dtype=np.int32)
    score = np.zeros((n, k), dtype=np.float16)
    target_rank = np.zeros(n, dtype=np.float64)
    for s in tqdm(range(0, n, batch_size), disable=not show_progress, mininterval=10):
        b = make_batch(data, inst.iloc[s : s + batch_size], data.max_hist_len)
        sc = apply_filters(scorer.score_all(b), b, first_ts)
        top = torch.topk(sc, k, dim=1)
        vals = top.values.numpy()
        idx = top.indices.numpy()
        idx[~np.isfinite(vals)] = 0  # fewer than k eligible items: pad with 0
        cand[s : s + len(b.targets)] = idx
        score[s : s + len(b.targets)] = np.where(np.isfinite(vals), vals, 0)
        tgt = sc[torch.arange(len(b.targets)), torch.from_numpy(b.targets.astype(np.int64))]
        target_rank[s : s + len(b.targets)] = (sc >= tgt.unsqueeze(1)).sum(1).numpy()
    hit = cand == inst["item"].values[:, None].astype(np.int32)
    target_pos = np.where(hit.any(1), hit.argmax(1), -1)
    return {"inst": inst["inst"].values, "user": inst["user"].values, "cand": cand, "score": score, "target_pos": target_pos, "target_rank": target_rank}


def force_insert(lists: dict, targets: np.ndarray, target_scores: np.ndarray) -> dict:
    """Append the target as an extra column where it was missed (ranker training lists, §9.1)."""
    n, k = lists["cand"].shape
    miss = lists["target_pos"] < 0
    cand = np.zeros((n, k + 1), dtype=np.int32)
    score = np.zeros((n, k + 1), dtype=np.float16)
    cand[:, :k], score[:, :k] = lists["cand"], lists["score"]
    cand[miss, k] = targets[miss]
    score[miss, k] = target_scores[miss]
    out = dict(lists, cand=cand, score=score)
    out["target_pos"] = np.where(miss, k, lists["target_pos"])
    out["forced"] = miss
    return out


def target_scores(scorer, data: ProcessedData, inst, batch_size: int = 512) -> np.ndarray:
    out = np.zeros(len(inst), dtype=np.float32)
    for s in range(0, len(inst), batch_size):
        b = make_batch(data, inst.iloc[s : s + batch_size], data.max_hist_len)
        sc = scorer.score_all(b)
        out[s : s + len(b.targets)] = sc[torch.arange(len(b.targets)), torch.from_numpy(b.targets.astype(np.int64))].numpy()
    return out


def load_candidates(cand_dir: str | Path, split: str, k1: int | None = None, force: bool = False) -> dict:
    """Load lists for ``split`` in {ranker_train, val, test}, cut to the top ``k1``.

    ``force=True`` (ranker-val) appends the target where it is not in the top ``k1``. Ranker-training lists are already forced.
    ``label`` [n, cols] is 1 at the target, 0 elsewhere (0s are unobserved, not confirmed dislikes).
    """
    z = dict(np.load(Path(cand_dir) / f"{split}.npz"))
    if k1 is not None and split != "ranker_train":
        z["cand"], z["score"] = z["cand"][:, :k1], z["score"][:, :k1]
        z["target_pos"] = np.where(z["target_pos"] < k1, z["target_pos"], -1)
    if force and split != "ranker_train":
        meta = json.loads((Path(cand_dir) / "meta.json").read_text())
        tgt = np.load(Path(cand_dir) / f"{split}_target_score.npz")
        z = force_insert(z, tgt["item"], tgt["score"])
        assert meta["source_run"]
    label = np.zeros(z["cand"].shape, dtype=np.int8)
    rows = np.flatnonzero(z["target_pos"] >= 0)
    label[rows, z["target_pos"][rows]] = 1
    z["label"] = label
    return z


def main(run_dir: str, sample: bool, device: str, k_train: int, k_eval: int, batch_size: int, splits: tuple[str, ...] = ("ranker_train", "val", "test")) -> None:
    data = ProcessedData.load(sample=sample)
    model, cfg = load_retriever(run_dir, data, device)
    scorer = RetrieverScorer(model)
    name = Path(run_dir).parent.name.removesuffix("_sample")  # e.g. two_tower_text (folder name, not the run's old config name)
    out = data.dir / "candidates" / f"{name}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source_run": str(run_dir), "retriever": name, "seed": cfg["seed"], "search": "exact",
            "k_train": k_train, "k_eval": k_eval, "git_hash": git_hash(), "splits": {}}
    split_config = {"ranker_train": (k_train, True), "val": (k_eval, False), "test": (k_eval, False)}
    for split in splits:
        k, force = split_config[split]
        inst = data.split_instances(split)
        log.info("%s: %d instances, top-%d", split, len(inst), k)
        lists = retrieve(scorer, data, inst, k, batch_size)
        tscore = target_scores(scorer, data, inst, batch_size)
        if force:
            lists = force_insert(lists, inst["item"].values.astype(np.int32), tscore)
        else:
            np.savez(out / f"{split}_target_score.npz", item=inst["item"].values.astype(np.int32), score=tscore)
        np.savez(out / f"{split}.npz", **lists)
        r = lists["target_rank"]
        meta["splits"][split] = {
            "n_lists": len(inst),
            "list_size": int(lists["cand"].shape[1]),
            "forced_fraction": float(lists["forced"].mean()) if force else 0.0,
            "lists_short_of_k": int((lists["cand"][:, :k] == 0).any(1).sum()),
            "recall_micro": {f"@{c}": float((r <= c).mean()) for c in RECALL_CUTOFFS if c <= k},
        }
        log.info("%s: %s", split, meta["splits"][split])
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("saved candidates to %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="trained retriever run, e.g. results/retriever/two_tower_text/seed0")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k-train", type=int, default=200)
    ap.add_argument("--k-eval", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--splits", nargs="+", choices=("ranker_train", "val", "test"), default=("ranker_train", "val"),
                    help="splits to build; defaults to ranker training and validation, leaving test for final evaluation")
    a = ap.parse_args()
    main(a.run_dir, a.sample, a.device, a.k_train, a.k_eval, a.batch_size, tuple(a.splits))
