"""Hybrid candidate lists: top-A two-tower + top-B last-item (deduplicated), same file format as build_candidates.

Every candidate keeps its two-tower retrieval score (the ranker's score input), and each list is sorted by that
score. Last-item score = cosine of the frozen item vectors to the user's most recent product (Brandon's
untrained scorer). Both sources use the same §4.6 filtering. ranker_train lists are force-inserted as usual
(natural-only training drops forced lists); val lists are never forced, plus val_target_score.npz so
load_candidates(force=True) works for checkpoint selection. Validation/ranker-train only; Test is not built.

    python scripts/build_hybrid_candidates.py --run-dir results/retriever/two_tower_text/seed0 \
        --two-tower 180 --last-item 20 --output-name hybrid_tt180_li20_seed0 --device cuda [--limit N]
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
from cascade.pipelines.build_candidates import force_insert, target_scores
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.io import git_hash


def hybrid(scorer: RetrieverScorer, data: ProcessedData, inst, a: int, b: int, batch_size: int) -> dict:
    first_ts = first_ts_tensor(data)
    k = a + b
    n = len(inst)
    cand = np.zeros((n, k), dtype=np.int32)
    score = np.zeros((n, k), dtype=np.float16)
    source_li = np.zeros((n, k), dtype=bool)
    target_rank = np.zeros(n, dtype=np.float64)
    V = scorer.V
    Vn = torch.nn.functional.normalize(V, dim=1)
    for s in tqdm(range(0, n, batch_size), mininterval=20):
        bt = make_batch(data, inst.iloc[s:s + batch_size], data.max_hist_len)
        tt = apply_filters(scorer.score_all(bt), bt, first_ts)
        last = bt.hist[:, -1].to(V.device)
        if (last == 0).any():
            raise ValueError("empty history")
        li = apply_filters((Vn[last] @ Vn.T).float().cpu(), bt, first_ts)
        top_tt = torch.topk(tt, a, dim=1).indices
        li_masked = li.clone()
        li_masked.scatter_(1, top_tt, -torch.inf)
        top_li = torch.topk(li_masked, b, dim=1)
        li_idx = top_li.indices.clone()
        li_idx[~torch.isfinite(top_li.values)] = 0
        both = torch.cat([top_tt, li_idx], dim=1)
        both_scores = torch.gather(tt, 1, both)
        both_scores[both == 0] = -torch.inf
        order = torch.argsort(both_scores, dim=1, descending=True, stable=True)
        both = torch.gather(both, 1, order)
        sc = torch.gather(both_scores, 1, order)
        is_li = torch.gather(torch.cat([torch.zeros_like(top_tt, dtype=torch.bool),
                                        torch.ones_like(li_idx, dtype=torch.bool)], 1), 1, order)
        rows = slice(s, s + len(bt.targets))
        cand[rows] = both.numpy()
        score[rows] = np.where(np.isfinite(sc.numpy()), sc.numpy(), 0)
        source_li[rows] = is_li.numpy()
        tgt = tt[torch.arange(len(bt.targets)), torch.from_numpy(bt.targets.astype(np.int64))]
        target_rank[rows] = (tt >= tgt.unsqueeze(1)).sum(1).numpy()
    hit = cand == inst["item"].values[:, None].astype(np.int32)
    target_pos = np.where(hit.any(1), hit.argmax(1), -1)
    return {"inst": inst["inst"].values, "user": inst["user"].values, "cand": cand, "score": score,
            "target_pos": target_pos, "target_rank": target_rank, "source_last_item": source_li}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--two-tower", type=int, default=180)
    ap.add_argument("--last-item", type=int, default=20)
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, help="first N instances per split, for a quick check")
    a = ap.parse_args()
    data = ProcessedData.load("configs/data.yaml")
    model, cfg = load_retriever(a.run_dir, data, a.device)
    scorer = RetrieverScorer(model)
    out = data.dir / "candidates" / a.output_name
    out.mkdir(parents=True, exist_ok=False)
    meta = {"source_run": a.run_dir, "kind": "hybrid two-tower + last-item", "two_tower": a.two_tower,
            "last_item": a.last_item, "seed": cfg["seed"], "git_hash": git_hash(), "limit": a.limit, "splits": {}}
    for split, force in (("ranker_train", True), ("val", False)):
        inst = data.split_instances(split)
        if a.limit:
            inst = inst.iloc[:a.limit]
        lists = hybrid(scorer, data, inst, a.two_tower, a.last_item, a.batch_size)
        tscore = target_scores(scorer, data, inst, a.batch_size)
        if force:
            lists = force_insert(lists, inst["item"].values.astype(np.int32), tscore)
            lists["source_last_item"] = np.pad(lists["source_last_item"], ((0, 0), (0, 1)))
        else:
            np.savez(out / f"{split}_target_score.npz", item=inst["item"].values.astype(np.int32), score=tscore)
        np.savez(out / f"{split}.npz", **lists)
        found = lists["target_pos"] >= 0
        if force:
            found = ~lists["forced"]
        meta["splits"][split] = {"n_lists": len(inst), "list_size": int(lists["cand"].shape[1]),
                                 "recall_micro": float(found.mean()),
                                 "hits_from_last_item": int(lists["source_last_item"][np.flatnonzero(found), lists["target_pos"][found]].sum()),
                                 "lists_short": int((lists["cand"][:, :a.two_tower + a.last_item] == 0).any(1).sum())}
        print(split, meta["splits"][split], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
