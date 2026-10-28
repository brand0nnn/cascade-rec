"""Validation only, exploratory: is the frozen two-tower retriever mostly a last-product model?

Two untrained scorers reuse the frozen Two-tower (text) item vectors v_i = normalize(item_tower(i)):
  * last-item: u = v of the most recent history item;
  * mean-pool: u = normalize(masked mean of the history item vectors v).
Both go through the shared harness (§4.6 filtering, pessimistic ties) on Val only and are compared with the
frozen retriever's saved Val ranks, aligned by instance id. No training, no Test data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import InstanceBatch, evaluate
from cascade.eval.ranker_meso import group_labels
from cascade.eval.segments import HIST_BUCKETS, hist_bucket
from cascade.models.retriever import RetrieverScorer, TwoTowerRetriever
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash, save_run

LABEL = "validation only, exploratory"
SPLIT = "val"  # hard-coded: this analysis never touches Test
CUTOFFS = (10, 50, 100, 200, 500)
COMPARE_CUTOFFS = (10, 200, 500)


class _FrozenItemVectors:
    """Shared item-vector table V [N+1, d] from the frozen item tower (row 0 = padding = 0)."""

    def __init__(self, model: TwoTowerRetriever):
        model.eval()
        self.device = next(model.parameters()).device
        with torch.no_grad():
            self.V = model.all_item_vectors()

    def scorable(self, batch: InstanceBatch) -> np.ndarray:
        return np.ones(len(batch.targets), dtype=bool)


class LastItemScorer(_FrozenItemVectors):
    """u = v of the most recent history item (histories are left-padded; last column = most recent)."""

    @torch.no_grad()
    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        last = batch.hist[:, -1].to(self.device)
        if (last == 0).any():
            raise ValueError("empty history reached the last-item scorer")
        return (self.V[last] @ self.V.T).float().cpu()


class MeanPoolScorer(_FrozenItemVectors):
    """u = normalize(mean of non-padded history vectors v); padding rows of V are 0 so a sum suffices."""

    @torch.no_grad()
    def score_all(self, batch: InstanceBatch) -> torch.Tensor:
        h = batch.hist.to(self.device)
        u = F.normalize(self.V[h].sum(1), dim=-1)
        return (u @ self.V.T).float().cpu()


def load_ranks(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def align(ref_inst: np.ndarray, other: dict) -> np.ndarray:
    """Ranks of ``other`` reordered to ``ref_inst`` (fails if any instance is missing)."""
    pos = pd.Series(np.arange(len(other["inst"])), index=other["inst"])
    idx = pos.reindex(ref_inst).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise ValueError("instance alignment failed")
    return other["rank"][idx.astype(np.int64)]


def macro(values: np.ndarray, users: np.ndarray) -> float:
    return M.macro_micro(values, users)["macro"] if len(values) else float("nan")


def compare(users: np.ndarray, frozen: np.ndarray, scorer: np.ndarray, groups: dict[str, np.ndarray], n_resamples: int) -> list[dict]:
    rows = []
    for gname, m in groups.items():
        u, rf, rs = users[m], frozen[m], scorer[m]
        row = {"group": gname, "n_instances": int(m.sum()), "n_users": int(len(np.unique(u)))}
        for c in COMPARE_CUTOFFS:
            hf, hs = M.recall_at(rf, c), M.recall_at(rs, c)
            row[f"frozen_recall@{c}"] = macro(hf, u)
            row[f"scorer_recall@{c}"] = macro(hs, u)
            row[f"ratio@{c}"] = row[f"scorer_recall@{c}"] / row[f"frozen_recall@{c}"] if row[f"frozen_recall@{c}"] else float("nan")
            b = paired_bootstrap(u, hf, hs, n_resamples=n_resamples, seed=0)
            row[f"diff@{c}"], row[f"ci_low@{c}"], row[f"ci_high@{c}"] = b["diff"], b["ci_low"], b["ci_high"]
        hf, hs = rf <= 200, rs <= 200
        row["frozen_hits@200"] = int(hf.sum())
        row["scorer_hits@200"] = int(hs.sum())
        row["both_hits@200"] = int((hf & hs).sum())
        row["share_frozen_hits_found_by_scorer@200"] = float((hf & hs).sum() / hf.sum()) if hf.sum() else float("nan")
        row["share_scorer_hits_not_found_by_frozen@200"] = float((hs & ~hf).sum() / hs.sum()) if hs.sum() else float("nan")
        row["jaccard_hits@200"] = float((hf & hs).sum() / (hf | hs).sum()) if (hf | hs).sum() else float("nan")
        rows.append(row)
    return rows


def main(run_dir: str, device: str, out_name: str, sanity_n: int, batch_size: int, n_resamples: int) -> dict:
    out = REPO_ROOT / "results" / "analysis" / out_name
    out.mkdir(parents=True, exist_ok=False)
    settings = {
        "label": LABEL, "split": SPLIT, "retriever_run": run_dir, "device": device, "cutoffs": list(CUTOFFS),
        "compare_cutoffs": list(COMPARE_CUTOFFS), "sanity_max_instances": sanity_n, "sanity_subset_seed": 0,
        "batch_size": batch_size, "bootstrap": {"n_resamples": n_resamples, "seed": 0, "unit": "user", "diff": "scorer - frozen"},
        "scorers": {"last_item": "u = v_last (frozen item tower, no training)",
                    "mean_pool": "u = normalize(masked mean of history v), untrained"},
        "hypothesis": "last-item reaches >= 80% of frozen Val Recall@200 and the gap does not grow much with history length",
    }
    (out / "LABEL.txt").write_text(LABEL + "\n")
    t0 = time.time()
    data = ProcessedData.load()
    model, _ = load_retriever(REPO_ROOT / run_dir, data, device)
    saved = load_ranks(REPO_ROOT / run_dir / f"{SPLIT}_ranks.npz")

    # 1. sanity: re-score a fixed Val subset with the frozen RetrieverScorer and compare with the saved ranks
    frozen_scorer = RetrieverScorer(model)
    evaluate(frozen_scorer, data, SPLIT, cutoffs=CUTOFFS, batch_size=batch_size, max_instances=sanity_n, subset_seed=0, save_dir=out / "sanity_frozen", show_progress=False)
    sub = load_ranks(out / "sanity_frozen" / f"{SPLIT}_ranks.npz")
    saved_sub = align(sub["inst"], saved)
    r_new, r_old = sub["rank"], saved_sub
    sanity = {
        "n": int(len(r_new)),
        "exact_rank_agreement": float(np.mean(r_new == r_old)),
        "abs_rank_diff_max": float(np.max(np.abs(r_new - r_old))),
        "abs_rank_diff_median": float(np.median(np.abs(r_new - r_old))),
        **{f"hit@{c}_agreement": float(np.mean((r_new <= c) == (r_old <= c))) for c in CUTOFFS},
    }
    del frozen_scorer
    print("sanity", json.dumps(sanity), flush=True)

    # 2. evaluate the two untrained scorers on all Val instances through the shared harness
    harness = {}
    ranks = {}
    for name, cls in (("last_item", LastItemScorer), ("mean_pool", MeanPoolScorer)):
        res = evaluate(cls(model), data, SPLIT, cutoffs=CUTOFFS, batch_size=batch_size, save_dir=out / name, show_progress=False)
        harness[name] = res
        ranks[name] = load_ranks(out / name / f"{SPLIT}_ranks.npz")
        print(name, "recall@200", res["recall@200"], f"{time.time() - t0:.0f}s", flush=True)

    # 3. aligned comparison with the frozen retriever's saved Val ranks
    inst_df = data.split_instances(SPLIT)
    ref_inst = inst_df["inst"].to_numpy()
    users = inst_df["user"].to_numpy()
    frozen = align(ref_inst, saved)
    labels = group_labels(data, inst_df)
    if not np.array_equal(labels["inst"].to_numpy(), ref_inst):
        raise ValueError("group labels misaligned")
    hb = hist_bucket(inst_df["hist_len"].to_numpy())
    groups = {"overall": np.ones(len(ref_inst), dtype=bool)}
    for t in ("new-in-period", "warm-tail", "warm-head"):
        groups[f"item_{t}"] = labels["item_type"].to_numpy() == t
    for h in HIST_BUCKETS:
        groups[f"hist_{h}"] = hb == h
    comparison = {}
    for name in ranks:
        rows = compare(users, frozen, align(ref_inst, ranks[name]), groups, n_resamples)
        comparison[name] = rows
        pd.DataFrame(rows).to_csv(out / f"comparison_{name}.csv", index=False)
    # frozen vs mean-pool vs last-item overlap of hits at 200 on the whole set (three-way)
    rl, rm = align(ref_inst, ranks["last_item"]), align(ref_inst, ranks["mean_pool"])
    hf, hl, hm = frozen <= 200, rl <= 200, rm <= 200
    venn = {"frozen_only": int((hf & ~hl & ~hm).sum()), "last_only": int((~hf & hl & ~hm).sum()), "mean_only": int((~hf & ~hl & hm).sum()),
            "frozen_last": int((hf & hl & ~hm).sum()), "frozen_mean": int((hf & ~hl & hm).sum()), "last_mean": int((~hf & hl & hm).sum()),
            "all_three": int((hf & hl & hm).sum()), "none": int((~hf & ~hl & ~hm).sum())}
    np.savez_compressed(out / "aligned_val_ranks.npz", inst=ref_inst, user=users, frozen=frozen, last_item=rl, mean_pool=rm)

    results = {"label": LABEL, "split": SPLIT, "sanity": sanity, "harness": harness, "comparison": comparison,
               "hits200_venn": venn, "elapsed_s": time.time() - t0, "git_hash": git_hash()}
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    save_run(out, {"label": LABEL, "sanity": sanity, "hits200_venn": venn}, settings)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/retriever/two_tower_text/seed0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="20261028-last-item")
    ap.add_argument("--sanity-n", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-resamples", type=int, default=1000)
    a = ap.parse_args()
    main(a.run_dir, a.device, a.out, a.sanity_n, a.batch_size, a.n_resamples)
