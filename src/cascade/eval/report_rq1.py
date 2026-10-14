"""RQ1 table: the two-tower retrievers vs the four baselines used as retrievers, all trained on period A (CLAUDE.md §7.4).

    python -m cascade.eval.report_rq1 [--toy] [--n-resamples 1000]

Every number is recomputed from the runs' saved per-case ranks ({val,test}_ranks.npz), so all systems are scored the
same way: over every Val/Test case, with cases a model cannot score (BPR users unseen in A) counted as misses, since a
retriever has to return candidates for every user. Also reports per-segment Recall (new-in-period, cold/tail/head,
history length) and the frozen-retriever check: v2 vs v4 on Val Recall@200, overall and on new-in-period products,
with a paired bootstrap CI over users (per-case recall averaged over seeds).

Writes results/rq1/summary[_toy].{md,json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.segments import HIST_BUCKETS, ITEM_SEGMENTS, hist_bucket, new_in_period
from cascade.utils.config import REPO_ROOT

# (label, run directory under results/); all trained on period A only
SYSTEMS = [
    ("Popularity", "baselines/Pop_A"),
    ("ItemKNN", "baselines/ItemKNN_A"),
    ("BPR-MF", "baselines/BPR_A"),
    ("SASRec", "baselines/SASRec_A"),
    ("Two-tower v2 (text)", "retriever/v2_transformer"),
    ("Two-tower v4 (ID + text)", "retriever/v4_id_text"),
]
CUTOFFS = (10, 50, 100, 200, 500)
SEGMENT_CUTOFFS = (10, 200)
FROZEN_CUTOFF = 200  # the frozen retriever is chosen on Val Recall@200 (§7.4)
V2, V4 = "retriever/v2_transformer", "retriever/v4_id_text"
DETERMINISTIC = {"baselines/Pop_A", "baselines/ItemKNN_A"}  # run for seed 0 only; std over seeds is 0 by construction


def segment_masks(data: ProcessedData, inst, split: str) -> dict[str, np.ndarray]:
    """Boolean masks over the split's cases, in the order of ``inst``."""
    items = inst["item"].values
    seg_map = np.concatenate([["pad"], data.items["segment"].values])
    first_ts = np.concatenate([[np.iinfo(np.int64).max], data.items["first_ts"].values])
    is_new = new_in_period(first_ts[items], split, data.split)
    hb = hist_bucket(inst["hist_len"].values)
    return {
        "new-in-period": is_new,
        "not new": ~is_new,
        **{s: seg_map[items] == s for s in ITEM_SEGMENTS},
        **{f"hist {h}": hb == h for h in HIST_BUCKETS},
    }


def load_runs(run_root: Path, inst_ids: np.ndarray, split: str) -> list[dict]:
    """Ranks (aligned to ``inst_ids``) and run info for every seed of one system that has saved ranks."""
    runs = []
    for d in sorted(run_root.glob("seed*")):
        f = d / f"{split}_ranks.npz"
        if not f.exists() or not (d / "metrics.json").exists():
            continue
        z = np.load(f)
        if not np.array_equal(z["inst"], inst_ids):
            raise ValueError(f"{f}: cases differ from the {split} instances (a subset run or stale data?)")
        meta = json.loads((d / "metrics.json").read_text())
        git = (d / "git_hash.txt").read_text().strip() if (d / "git_hash.txt").exists() else "?"
        runs.append({"seed": meta["seed"], "device": meta.get("device", "?"), "git": git, "rank": z["rank"]})
    return runs


def _fmt(vals: list[float]) -> str:
    if not vals:
        return "–"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"


def frozen_check(runs: dict[str, list[dict]], users: np.ndarray, masks: dict[str, np.ndarray], n_resamples: int) -> dict | None:
    """v4 − v2 on Val Recall@200, overall and on new-in-period cases (per-case recall averaged over seeds)."""
    if not runs.get(V2) or not runs.get(V4):
        return None
    hit = {k: np.mean([M.recall_at(r["rank"], FROZEN_CUTOFF) for r in runs[k]], axis=0) for k in (V2, V4)}
    out = {}
    for name, m in (("all", np.ones(len(users), dtype=bool)), ("new-in-period", masks["new-in-period"])):
        bs = paired_bootstrap(users[m], hit[V2][m], hit[V4][m], n_resamples=n_resamples)
        better = "tie (CI includes 0)" if bs["ci_low"] <= 0 <= bs["ci_high"] else ("v4" if bs["diff"] > 0 else "v2")
        out[name] = {**bs, "better": better}
    picks = {v["better"] for v in out.values()}
    if len(picks) == 1:
        rec = picks.pop()
        out["recommendation"] = f"{rec} on both" if not rec.startswith("tie") else "no significant difference: keep v2 (simpler, main retriever)"
    elif "v2" in picks and "v4" in picks:
        out["recommendation"] = "v2 and v4 disagree (overall vs new-in-period): discuss with the team before freezing (§7.4)"
    else:
        winner = (picks - {"tie (CI includes 0)"}).pop()
        out["recommendation"] = f"{winner} (better on one, no significant difference on the other)"
    return out


def main(toy: bool, n_resamples: int) -> None:
    data = ProcessedData.load(toy=toy)
    suffix = "_toy" if toy else ""
    summary, lines = {}, []
    for split in ("val", "test"):
        inst = data.split_instances(split)
        users = inst["user"].values
        masks = segment_masks(data, inst, split)
        runs = {path: load_runs(REPO_ROOT / "results" / (path + suffix), inst["inst"].values, split) for _, path in SYSTEMS}
        block = summary.setdefault(split, {"n_instances": len(inst), "n_users": int(len(np.unique(users))),
                                           "segment_sizes": {s: int(m.sum()) for s, m in masks.items()}, "systems": {}})

        cols = [f"R@{c}" for c in CUTOFFS] + ["nDCG@10"]
        lines += [f"\n## {split.capitalize()} ({len(inst):,} cases; per-user macro, mean ± std over seeds)\n",
                  "| Retriever | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        warnings = []
        for label, path in SYSTEMS:
            rs = runs[path]
            per_seed = {f"recall@{c}": [M.macro_micro(M.recall_at(r["rank"], c), users)["macro"] for r in rs] for c in CUTOFFS}
            per_seed["ndcg@10"] = [M.macro_micro(M.ndcg_at(r["rank"], 10), users)["macro"] for r in rs]
            segs = {f"recall@{c}": {s: [M.macro_micro(M.recall_at(r["rank"][m], c), users[m])["macro"] for r in rs] for s, m in masks.items()}
                    for c in SEGMENT_CUTOFFS}
            block["systems"][label] = {
                "path": path, "seeds": [r["seed"] for r in rs], "devices": [r["device"] for r in rs], "git": [r["git"] for r in rs],
                "unscorable": [int(np.isinf(r["rank"]).sum()) for r in rs],
                **{k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v} if v else None for k, v in per_seed.items()},
                "segments": {k: {s: (float(np.mean(v)) if v and not np.isnan(v).all() else None) for s, v in d.items()} for k, d in segs.items()},
            }
            block["systems"][label]["_segs"] = segs  # for the markdown below; dropped before saving
            seeds = "missing" if not rs else ("1 (deterministic)" if path in DETERMINISTIC else len(rs))
            lines.append(f"| {label} | {seeds} | " + " | ".join(_fmt(per_seed[k]) for k in [*(f"recall@{c}" for c in CUTOFFS), "ndcg@10"]) + " |")
            if len(set(r["device"] for r in rs)) > 1:
                warnings.append(f"{label}: seeds come from different devices {sorted(set(r['device'] for r in rs))}")
            if rs and len(rs) < 3 and path not in DETERMINISTIC:
                warnings.append(f"{label}: only {len(rs)} seed(s)")

        for c in SEGMENT_CUTOFFS:
            names = list(masks)
            lines += [f"\n### {split.capitalize()} Recall@{c} by segment\n",
                      "| Retriever | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
                      "| *cases* | " + " | ".join(f"*{int(masks[s].sum()):,}*" for s in names) + " |"]
            for label, _ in SYSTEMS:
                segs = block["systems"][label]["_segs"][f"recall@{c}"]
                lines.append(f"| {label} | " + " | ".join(_fmt([v for v in segs[s] if not np.isnan(v)]) for s in names) + " |")
        for label, _ in SYSTEMS:
            del block["systems"][label]["_segs"]
        if warnings:
            lines += ["", *(f"- ⚠️ {w}" for w in warnings)]
        block["warnings"] = warnings

        if split == "val":
            fc = frozen_check(runs, users, masks, n_resamples)
            block["frozen_retriever_check"] = fc
            lines.append(f"\n### Frozen retriever check (Val Recall@{FROZEN_CUTOFF}, v4 − v2, paired bootstrap over users, 95% CI)\n")
            if fc is None:
                lines.append("Needs at least one seed of both v2 and v4.")
            else:
                lines += ["| Cases | v4 − v2 | 95% CI | better |", "|---|---|---|---|"]
                lines += [f"| {k} | {fc[k]['diff']:+.4f} | [{fc[k]['ci_low']:+.4f}, {fc[k]['ci_high']:+.4f}] | {fc[k]['better']} |" for k in ("all", "new-in-period")]
                lines.append(f"\n**Recommendation:** {fc['recommendation']}. Record the choice in `docs/decisions.md`.")

    out_dir = REPO_ROOT / "results" / "rq1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    md = ("# RQ1: two-tower retriever vs baseline retrieval (all trained on period A, exact search)\n\n"
          "Full ranking over the filtered catalogue; every Val/Test case counts, so cases a model cannot score "
          "(BPR users with no period-A interactions) are misses. Recomputed from each run's saved ranks.\n"
          + "\n".join(lines) + "\n")
    (out_dir / f"summary{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--n-resamples", type=int, default=1000)
    a = ap.parse_args()
    main(a.toy, a.n_resamples)
