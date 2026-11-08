"""Rebuild Brandon's P3 mixed pool (recency + two-tower round-robin) in our candidate format, plus a +last-item variant.

Recency retriever (as src/cascade/pipelines/frozen_text_study.py): frozen MiniLM item vectors; user query = weighted
mean of the last 50 prior products, weight 2^(-age_days/90), L2-normalised; score = vectors @ query. Two-tower =
frozen Two-tower (text) seed 0. Both after §4.6 filtering. P3 = round-robin recency #1, two-tower #1, recency #2, ...
skipping duplicates until 200 unique products (docs/experiment_overview.md). Variant p3li replaces the last 20
P3 slots by the top-20 last-item products not already in the list (cosine of two-tower item vectors to the most
recent product).

Every candidate carries its TWO-TOWER score (our ranker's frozen score contract); lists keep interleave order.
`--rrf` instead writes only `p3rrf_seed0{suffix}`: the same P3 lists, with Brandon's reciprocal-rank-fusion score
sum over both retrievers of 1/(60 + full-catalogue rank), stored as 1000*rrf - 1000/61 (affine, order-preserving,
keeps float16 precision); the target score files use the same formula.
`--splits test --rrf` (final Test, user-authorised 8 Nov 2026) writes natural Test lists only, to
`p3rrf_test_seed0{suffix}` plus `tt_test_seed0{suffix}` (two-tower top 200 with two-tower scores, the original pool).

    python scripts/build_p3_candidates.py --run-dir results/retriever/two_tower_text/seed0 --device cuda [--limit N]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from cascade.data.instances import ProcessedData
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.models.retriever import RetrieverScorer
from cascade.pipelines.build_candidates import force_insert, target_scores
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.io import git_hash

K, LI = 200, 20
LOCAL = False
DAY = 86_400_000.0


def interleave(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Round-robin a[0], b[0], a[1], b[1], ... skipping duplicates and zeros until k unique items."""
    out = np.zeros(k, dtype=np.int64)
    seen, n, i = set(), 0, 0
    while n < k and i < max(len(a), len(b)):
        for src in (a, b):
            if i < len(src):
                x = int(src[i])
                if x > 0 and x not in seen:
                    seen.add(x); out[n] = x; n += 1
                    if n == k:
                        break
        i += 1
    return out


def full_rank(scores: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    """1-based full-catalogue rank of each item (ties count against it), filtered (-inf) rows included."""
    asc = torch.sort(scores, dim=1).values
    val = torch.gather(scores, 1, items)
    return (scores.shape[1] - torch.searchsorted(asc.contiguous(), val.contiguous(), right=False)).float()


def rrf(rec: torch.Tensor, tt: torch.Tensor, items: torch.Tensor) -> np.ndarray:
    r = 1 / (60 + full_rank(rec, items)) + 1 / (60 + full_rank(tt, items))
    return (1000 * r - 1000 / 61).numpy()


def local_rrf(top_rec: np.ndarray, top_tt: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Fast combined score: position within each retriever's own top-200 list (absent = 201), same formula/scale as rrf()."""
    pr = {int(x): j + 1 for j, x in enumerate(top_rec)}; pt = {int(x): j + 1 for j, x in enumerate(top_tt)}
    r1 = np.array([pr.get(int(x), K + 1) for x in items], float); r2 = np.array([pt.get(int(x), K + 1) for x in items], float)
    return 1000 * (1 / (60 + r1) + 1 / (60 + r2)) - 1000 / 61


def build(scorer, text, data, inst, batch_size):
    first_ts = first_ts_tensor(data)
    dev = scorer.V.device
    E = torch.as_tensor(text, dtype=torch.float32, device=dev)
    Vn = torch.nn.functional.normalize(scorer.V, dim=1)
    n = len(inst)
    out = {name: np.zeros((n, K), dtype=np.int32) for name in ("p3", "p3li")}
    score = {name: np.zeros((n, K), dtype=np.float16) for name in out}
    rec_rank = np.zeros(n, dtype=np.float64)
    rrf_score = np.zeros((n, K), dtype=np.float16)
    rrf_target = np.zeros(n, dtype=np.float32)
    rrfl_score = np.zeros((n, K), dtype=np.float16)
    rrfl_target = np.zeros(n, dtype=np.float32)
    top_tt_all = np.zeros((n, K), dtype=np.int32)
    tt_score = np.zeros((n, K), dtype=np.float16)
    users, hl, ts = inst["user"].to_numpy(), inst["hist_len"].to_numpy(), inst["ts"].to_numpy()
    for s in tqdm(range(0, n, batch_size), mininterval=30):
        sl = slice(s, s + batch_size)
        bt = make_batch(data, inst.iloc[sl], data.max_hist_len)
        b = len(bt.targets)
        H = np.zeros((b, 50), dtype=np.int64); HT = np.zeros((b, 50), dtype=np.float64)
        for r, (u, h) in enumerate(zip(users[sl], hl[sl])):
            st = data.user_ptr[u]; end = st + h; lo = max(st, end - 50)
            H[r, 50 - (end - lo):] = data.seq_item[lo:end]; HT[r, 50 - (end - lo):] = data.seq_ts[lo:end]
        w = np.where(H > 0, np.exp2(-(ts[sl][:, None] - HT) / DAY / 90), 0.0)
        Wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
        q = (Wt[:, :, None] * E[torch.as_tensor(H, device=dev)]).sum(1) / Wt.sum(1, keepdim=True).clamp_min(1e-12)
        q = torch.nn.functional.normalize(q, dim=1)
        rec = apply_filters((q @ E.T).float().cpu(), bt, first_ts)
        tt = apply_filters(scorer.score_all(bt), bt, first_ts)
        last = bt.hist[:, -1].to(dev)
        li = apply_filters((Vn[last] @ Vn.T).float().cpu(), bt, first_ts)
        top_rec = torch.topk(rec, K, dim=1).indices.numpy()
        top_tt = torch.topk(tt, K, dim=1).indices.numpy()
        top_li = torch.topk(li, K, dim=1).indices.numpy()
        tgt = torch.from_numpy(bt.targets.astype(np.int64))
        rec_rank[sl] = (rec >= rec[torch.arange(b), tgt].unsqueeze(1)).sum(1).numpy()
        ttn = tt.numpy()
        top_tt_all[sl] = top_tt
        tsc = np.take_along_axis(ttn, top_tt, 1)
        tt_score[sl] = np.where(np.isfinite(tsc), tsc, 0)
        rrf_target[sl] = rrf(rec, tt, tgt[:, None])[:, 0]
        for r in range(b):
            p3 = interleave(top_rec[r], top_tt[r], K)
            have = set(p3[:K - LI].tolist())
            extra = [x for x in top_li[r] if x > 0 and x not in have][:LI]
            p3li = np.concatenate([p3[:K - LI], extra]) if len(extra) == LI else p3
            rrfl_score[s + r] = local_rrf(top_rec[r], top_tt[r], p3)
            rrfl_target[s + r] = local_rrf(top_rec[r], top_tt[r], np.array([bt.targets[r]]))[0]
            for name, lst in (("p3", p3), ("p3li", p3li)):
                out[name][s + r] = lst
                sc = ttn[r, lst]
                score[name][s + r] = np.where(np.isfinite(sc), sc, 0)
        rrf_score[sl] = rrf(rec, tt, torch.from_numpy(out["p3"][sl].astype(np.int64)))
    out["p3rrf"], score["p3rrf"] = out["p3"], rrf_score
    out["tt"], score["tt"] = top_tt_all, tt_score
    out["p3rrfl"], score["p3rrfl"] = out["p3"], rrfl_score
    return out, score, rec_rank, (rrfl_target if LOCAL else rrf_target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rrf", action="store_true", help="write only the P3 lists with the RRF score")
    ap.add_argument("--rrf-local", action="store_true", help="write only P3 lists with the fast (within-list) combined score, all given splits incl. test")
    ap.add_argument("--splits", nargs="+", default=["ranker_train", "val"], choices=["ranker_train", "val", "test"])
    a = ap.parse_args()
    data = ProcessedData.load("configs/data.yaml")
    model, cfg = load_retriever(a.run_dir, data, a.device)
    scorer = RetrieverScorer(model)
    text = np.load(data.dir / "item_text_emb.npy")
    global LOCAL
    LOCAL = a.rrf_local
    test_only = a.splits == ["test"]
    if "test" in a.splits and not ((test_only and a.rrf) or a.rrf_local):
        raise ValueError("Test lists are built only with --splits test --rrf")
    names = ("p3rrfl",) if a.rrf_local else (("p3rrf", "tt") if test_only else ("p3rrf",)) if a.rrf else ("p3", "p3li")
    dirs = {name: data.dir / "candidates" / f"{name}{'_test' if test_only and not a.rrf_local else ''}_seed0{a.suffix}" for name in names}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=False)
    meta = {name: {"source_run": a.run_dir, "pool": name, "git_hash": git_hash(), "limit": a.limit, "splits": {}} for name in dirs}
    for split, force in [(x, x == "ranker_train") for x in a.splits]:
        inst = data.split_instances(split)
        if a.limit:
            inst = inst.iloc[:a.limit]
        cands, scores, rec_rank, rrf_target = build(scorer, text, data, inst, a.batch_size)
        tscore = rrf_target if (a.rrf or a.rrf_local) else target_scores(scorer, data, inst, a.batch_size)
        items = inst["item"].to_numpy().astype(np.int32)
        for name, d in dirs.items():
            hit = cands[name] == items[:, None]
            lists = {"inst": inst["inst"].values, "user": inst["user"].values, "cand": cands[name],
                     "score": scores[name], "target_pos": np.where(hit.any(1), hit.argmax(1), -1),
                     "target_rank": rec_rank}
            found = lists["target_pos"] >= 0
            if force:
                lists = force_insert(lists, items, tscore)
            else:
                np.savez(d / f"{split}_target_score.npz", item=items, score=tscore)
            np.savez(d / f"{split}.npz", **lists)
            meta[name]["splits"][split] = {"n_lists": len(inst), "recall_at_200": float(found.mean()),
                                           "recency_recall_at_200": float((rec_rank <= K).mean()),
                                           "lists_short": int((cands[name] == 0).any(1).sum())}
            print(name, split, meta[name]["splits"][split], flush=True)
    for name, d in dirs.items():
        (d / "meta.json").write_text(json.dumps(meta[name], indent=2))


if __name__ == "__main__":
    main()
