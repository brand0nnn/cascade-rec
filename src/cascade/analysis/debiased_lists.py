"""Validation only, exploratory: §4.6-filtered top-K₁ Val lists selected by a serving-time logQ-debiased score.

Candidates are SELECTED and ORDERED by s'_j = u.v_j - beta * tau * log q_j (see ``logq_serving.py``; unseen-item
convention ``minseen``), but the stored per-candidate ``score`` is the ORIGINAL retrieval score u.v_j (float16), the
same definition as the index sweep's exact lists, because the frozen rankers were trained on it. Output format matches
``index_sweep.py``: <variant>/val.npz with inst, user, cand [n, K₁] int32, score [n, K₁] float16, target_pos int64 (-1 = miss).
beta = 0 is built as a check against the reference exact lists.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from cascade.analysis.logq_serving import log_q_variants
from cascade.data.instances import ProcessedData
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash, save_run

LABEL = "validation only, exploratory"
VARIANTS = {"beta0": (0.0, "train"), "beta0.25_minseen": (0.25, "minseen"), "beta0.5_minseen": (0.5, "minseen")}


def main(run_dir: str, reference: str, device: str, out_name: str, k1: int, batch_size: int) -> None:
    out = REPO_ROOT / "results" / "analysis" / out_name
    out.mkdir(parents=True, exist_ok=False)
    (out / "LABEL.txt").write_text(LABEL + "\n")
    t0 = time.time()
    data = ProcessedData.load()
    model, _ = load_retriever(REPO_ROOT / run_dir, data, device)
    dev = next(model.parameters()).device
    V = model.all_item_vectors()
    lq, lq_info = log_q_variants(data)
    offsets = {n: torch.as_tensor(b * model.temperature * lq[v], dtype=torch.float32) for n, (b, v) in VARIANTS.items()}
    inst = data.split_instances("val")
    n = len(inst)
    first_ts = first_ts_tensor(data)
    cand = {v: np.zeros((n, k1), dtype=np.int32) for v in VARIANTS}
    score = {v: np.zeros((n, k1), dtype=np.float16) for v in VARIANTS}
    for s in range(0, n, batch_size):
        b = make_batch(data, inst.iloc[s : s + batch_size], data.max_hist_len)
        with torch.no_grad():
            raw = (model.user_vectors(b.hist.to(dev)) @ V.T).float().cpu()
        for v, off in offsets.items():
            deb = apply_filters(raw - off.unsqueeze(0) if VARIANTS[v][0] else raw.clone(), b, first_ts)
            val, idx = torch.topk(deb, k1, dim=1)
            ok = torch.isfinite(val)
            ids = torch.where(ok, idx, torch.zeros_like(idx))
            sc = torch.where(ok, raw.gather(1, idx), torch.zeros_like(val))
            cand[v][s : s + len(ids)] = ids.numpy()
            score[v][s : s + len(ids)] = sc.numpy().astype(np.float16)
    targets = inst["item"].to_numpy()
    ref = np.load(REPO_ROOT / reference / "val.npz")
    if not np.array_equal(ref["inst"], inst["inst"].to_numpy()):
        raise ValueError("reference exact lists are not aligned with Val instances")
    summary = {"label": LABEL, "n_instances": n, "k1": k1, "log_q_info": lq_info, "variants": {}}
    for v in VARIANTS:
        hit = cand[v] == targets[:, None]
        pos = np.where(hit.any(1), hit.argmax(1), -1).astype(np.int64)
        (out / v).mkdir()
        np.savez(out / v / "val.npz", inst=inst["inst"].to_numpy(), user=inst["user"].to_numpy().astype(np.int32), cand=cand[v], score=score[v], target_pos=pos)
        info = {"beta": VARIANTS[v][0], "unseen": VARIANTS[v][1], "short_rows": int((cand[v] == 0).any(1).sum()),
                "recall@k1_micro": float((pos >= 0).mean()), "mean_raw_score": float(score[v].astype(np.float32).mean()),
                "mean_raw_score_last_slot": float(score[v][:, -1].astype(np.float32).mean()),
                "overlap_with_reference": float(np.mean([len(np.intersect1d(a, r)) / k1 for a, r in zip(cand[v][::50], ref["cand"][::50])]))}
        if v == "beta0":
            info["check_vs_reference"] = {
                "identical_ordered_lists_share": float(np.mean((cand[v] == ref["cand"]).all(1))),
                "identical_candidate_sets_share": float(np.mean((np.sort(cand[v], 1) == np.sort(ref["cand"], 1)).all(1))),
                "slot_agreement": float(np.mean(cand[v] == ref["cand"])),
                "score_equal_where_slot_agrees": float(np.mean(score[v][cand[v] == ref["cand"]] == ref["score"][cand[v] == ref["cand"]])),
                "score_max_abs_diff_where_slot_agrees": float(np.max(np.abs(score[v].astype(np.float32) - ref["score"].astype(np.float32))[cand[v] == ref["cand"]])),
                "target_pos_equal_share": float(np.mean(pos == ref["target_pos"])),
            }
        summary["variants"][v] = info
        print(v, json.dumps(info), flush=True)
    summary["elapsed_s"] = time.time() - t0
    summary["git_hash"] = git_hash()
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    save_run(out, summary, {"label": LABEL, "split": "val", "retriever_run": run_dir, "reference": reference, "k1": k1,
                            "variants": {k: {"beta": b, "unseen": u} for k, (b, u) in VARIANTS.items()},
                            "selection_score": "u.v - beta * tau * log q", "stored_score": "u.v (original retrieval score, float16)",
                            "temperature": float(model.temperature), "device": device, "batch_size": batch_size})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/retriever/two_tower_text/seed0")
    ap.add_argument("--reference", default="data/processed/candidates/index_sweep/20261028-index-sweep/exact")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="20261028-debiased-lists")
    ap.add_argument("--k1", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    main(a.run_dir, a.reference, a.device, a.out, a.k1, a.batch_size)
