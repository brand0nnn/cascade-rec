"""Validation only, exploratory: does subtracting beta * log q_j at serving time undo the frozen retriever's head skew?

Frozen Two-tower (text) seed 0, no training. Serving score (ranking-equivalent to u.v/tau - beta * log q_j):
    s_j = u.v_j - beta * tau * log q_j        (tau = model.temperature = 0.05)
Written in the u.v scale so that beta = 0 is bit-identical to RetrieverScorer (u.v). Multiplying every score of an
instance by 1/tau > 0 does not change the ranking, apart from float rounding.

log q_j comes from ``retriever_train_log_q`` (retriever-training counts / total). There, items never seen in
retriever-training get log q = 0, which is *larger* than every seen item's log q (< 0), so they get the smallest
correction, as if they were the most popular items. Two unseen-item conventions are reported:
  * ``train``  : log q = 0 for unseen, exactly as in training;
  * ``minseen``: unseen items get the minimum seen log q (the largest correction, same as the rarest seen item).
Also a near-duplicate check of the last-item scorer's new-in-period hits@10 (no new scoring).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from cascade.analysis.last_item import align, load_ranks, macro
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import InstanceBatch, evaluate
from cascade.eval.ranker_meso import group_labels
from cascade.eval.segments import HIST_BUCKETS, hist_bucket
from cascade.models.retriever import TwoTowerRetriever
from cascade.pipelines.train_retriever import load_retriever, retriever_train_log_q
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash, save_run

LABEL = "validation only, exploratory"
SPLIT = "val"  # hard-coded: this analysis never touches Test
CUTOFFS = (10, 50, 100, 200, 500)
COMPARE_CUTOFFS = (10, 200, 500)
BETAS = (0.0, 0.25, 0.5, 1.0)


class LogQServingScorer:
    """u from the frozen user tower; s_j = u.v_j - beta * tau * log q_j (column 0 = padding, filtered by the harness)."""

    def __init__(self, model: TwoTowerRetriever, log_q: np.ndarray, beta: float):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.V = model.all_item_vectors()
        self.beta = float(beta)
        self.offset = torch.as_tensor(self.beta * model.temperature * log_q, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        u = self.model.user_vectors(batch.hist.to(self.device))
        s = u @ self.V.T
        if self.beta:
            s = s - self.offset.unsqueeze(0)
        return s.float().cpu()


def log_q_variants(data: ProcessedData) -> tuple[dict[str, np.ndarray], dict]:
    log_q, seen = retriever_train_log_q(data)
    seen[0] = False
    minseen = log_q.copy()
    lo = float(log_q[seen].min())
    minseen[~seen] = lo
    minseen[0] = 0.0
    info = {"n_items": int(data.n_items), "n_seen_in_retriever_train": int(seen[1:].sum()), "n_unseen": int((~seen[1:]).sum()),
            "log_q_seen_min": lo, "log_q_seen_max": float(log_q[seen].max()), "log_q_seen_median": float(np.median(log_q[seen]))}
    return {"train": log_q, "minseen": minseen.astype(np.float32)}, info


def near_duplicate_check(data: ProcessedData, inst_df: pd.DataFrame, labels: pd.DataFrame, last_ranks: np.ndarray, n_examples: int = 5) -> dict:
    """Last-item hits@10 on new-in-period targets: MiniLM cosine to the last item, same store, example titles."""
    emb = np.load(data.dir / "item_text_emb.npy", mmap_mode="r")
    last = data.histories(inst_df["user"].to_numpy(), inst_df["hist_len"].to_numpy(), 1)[:, -1]
    tgt = inst_df["item"].to_numpy()
    new = labels["item_type"].to_numpy() == "new-in-period"
    items = data.items.set_index("item")
    store = items["store"].fillna("").astype(str)
    title = items["title"].fillna("").astype(str)

    def stats(m: np.ndarray) -> dict:
        a = np.asarray(emb[last[m]], dtype=np.float64)
        b = np.asarray(emb[tgt[m]], dtype=np.float64)
        cos = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        sl, st = store.loc[last[m]].to_numpy(), store.loc[tgt[m]].to_numpy()
        tl, tt = title.loc[last[m]].to_numpy(), title.loc[tgt[m]].to_numpy()
        return {"n": int(m.sum()), "cosine_quantiles": {str(q): float(np.quantile(cos, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                "cosine_mean": float(cos.mean()), "share_cosine_ge_0.9": float(np.mean(cos >= 0.9)), "share_cosine_ge_0.95": float(np.mean(cos >= 0.95)),
                "share_same_store": float(np.mean((sl == st) & (sl != ""))), "share_identical_title": float(np.mean((tl == tt) & (tl != "")))}

    hit = new & (last_ranks <= 10)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.flatnonzero(hit), min(n_examples, int(hit.sum())), replace=False))
    examples = [{"inst": int(inst_df["inst"].iat[i]), "rank": float(last_ranks[i]), "last_title": title.loc[last[i]], "last_store": store.loc[last[i]],
                 "target_title": title.loc[tgt[i]], "target_store": store.loc[tgt[i]]} for i in pick]
    return {"last_item_hits10_new_in_period": stats(hit), "reference_all_new_in_period": stats(new),
            "reference_new_in_period_misses_at_200": stats(new & (last_ranks > 200)), "examples_seed0": examples}


def main(run_dir: str, last_item_dir: str, device: str, out_name: str, batch_size: int, n_resamples: int) -> dict:
    out = REPO_ROOT / "results" / "analysis" / out_name
    out.mkdir(parents=True, exist_ok=False)
    (out / "LABEL.txt").write_text(LABEL + "\n")
    t0 = time.time()
    data = ProcessedData.load()
    model, _ = load_retriever(REPO_ROOT / run_dir, data, device)
    lq, lq_info = log_q_variants(data)
    settings = {"label": LABEL, "split": SPLIT, "retriever_run": run_dir, "device": device, "betas": list(BETAS),
                "temperature": float(model.temperature), "score": "u.v - beta * tau * log q (ranking-equivalent to u.v/tau - beta * log q)",
                "log_q": "retriever_train_log_q (retriever-training counts / total)",
                "unseen_conventions": {"train": "log q = 0 for unseen (as in training)", "minseen": "unseen = min seen log q"},
                "cutoffs": list(CUTOFFS), "batch_size": batch_size,
                "bootstrap": {"n_resamples": n_resamples, "seed": 0, "unit": "user", "diff": "beta - beta0"},
                "near_duplicate_source": last_item_dir, "hypothesis": "subtracting beta*log q at serving shifts recall toward new/tail items; head falls; pop concentration falls"}

    inst_df = data.split_instances(SPLIT)
    ref_inst, users = inst_df["inst"].to_numpy(), inst_df["user"].to_numpy()
    labels = group_labels(data, inst_df)
    if not np.array_equal(labels["inst"].to_numpy(), ref_inst):
        raise ValueError("group labels misaligned")
    hb = hist_bucket(inst_df["hist_len"].to_numpy())
    groups = {"overall": np.ones(len(ref_inst), dtype=bool)}
    for t in ("new-in-period", "warm-tail", "warm-head"):
        groups[f"item_{t}"] = labels["item_type"].to_numpy() == t
    for h in HIST_BUCKETS:
        groups[f"hist_{h}"] = hb == h

    # near-duplicate check (from the earlier last-item run; no new scoring)
    li = load_ranks(REPO_ROOT / last_item_dir / "aligned_val_ranks.npz")
    if not np.array_equal(li["inst"], ref_inst):
        raise ValueError("last-item ranks misaligned")
    near_dup = near_duplicate_check(data, inst_df, labels, li["last_item"])
    print("near_dup", json.dumps(near_dup["last_item_hits10_new_in_period"]), flush=True)

    runs = [("beta0", 0.0, "train")] + [(f"beta{b}_{v}", b, v) for v in ("train", "minseen") for b in BETAS if b]
    harness, ranks = {}, {}
    for name, beta, var in runs:
        res = evaluate(LogQServingScorer(model, lq[var], beta), data, SPLIT, cutoffs=CUTOFFS, batch_size=batch_size, save_dir=out / name, show_progress=False)
        harness[name] = res
        ranks[name] = align(ref_inst, load_ranks(out / name / f"{SPLIT}_ranks.npz"))
        print(name, "R@200", round(res["recall@200"]["macro"], 5), "pop", round(res["pop_concentration@10"], 4), f"{time.time() - t0:.0f}s", flush=True)

    saved = align(ref_inst, load_ranks(REPO_ROOT / run_dir / f"{SPLIT}_ranks.npz"))
    r0 = ranks["beta0"]
    beta0_check = {"n": int(len(r0)), "exact_rank_agreement_vs_saved": float(np.mean(r0 == saved)),
                   "abs_rank_diff_max": float(np.max(np.abs(r0 - saved))),
                   **{f"hit@{c}_agreement_vs_saved": float(np.mean((r0 <= c) == (saved <= c))) for c in CUTOFFS},
                   "recall@200_macro_beta0": macro(M.recall_at(r0, 200), users), "recall@200_macro_saved": macro(M.recall_at(saved, 200), users)}
    print("beta0_check", json.dumps(beta0_check), flush=True)

    rows = []
    for name, beta, var in runs:
        for g, m in groups.items():
            u = users[m]
            row = {"run": name, "beta": beta, "unseen": var, "group": g, "n_instances": int(m.sum())}
            for c in COMPARE_CUTOFFS:
                h0, h = M.recall_at(r0[m], c), M.recall_at(ranks[name][m], c)
                row[f"recall@{c}"] = macro(h, u)
                if beta:
                    b = paired_bootstrap(u, h0, h, n_resamples=n_resamples, seed=0)
                    row[f"diff@{c}"], row[f"ci_low@{c}"], row[f"ci_high@{c}"] = b["diff"], b["ci_low"], b["ci_high"]
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "logq_serving_table.csv", index=False)
    beyond = pd.DataFrame([{"run": n, "beta": b, "unseen": v, "coverage@10": harness[n]["coverage@10"],
                            "pop_concentration@10": harness[n]["pop_concentration@10"]} for n, b, v in runs])
    beyond.to_csv(out / "beyond_accuracy.csv", index=False)
    np.savez_compressed(out / "aligned_val_ranks.npz", inst=ref_inst, user=users, **{n: ranks[n] for n, _, _ in runs})

    results = {"label": LABEL, "split": SPLIT, "log_q_info": lq_info, "beta0_check": beta0_check, "harness": harness,
               "table": rows, "beyond_accuracy": beyond.to_dict("records"), "near_duplicate": near_dup,
               "elapsed_s": time.time() - t0, "git_hash": git_hash()}
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    save_run(out, {"label": LABEL, "beta0_check": beta0_check, "log_q_info": lq_info}, settings)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/retriever/two_tower_text/seed0")
    ap.add_argument("--last-item-dir", default="results/analysis/20261028-last-item")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="20261028-logq-serving")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-resamples", type=int, default=1000)
    a = ap.parse_args()
    main(a.run_dir, a.last_item_dir, a.device, a.out, a.batch_size, a.n_resamples)
