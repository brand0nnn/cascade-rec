"""Validation-only comparison of two ranker run groups from saved per-instance ranks (exploratory).

Each group is a list of ranker run dirs (one per seed) with ``val_ranks.npz``. Every group is compared with its
own retrieval order (R1, from the same archives) and the second group with the first, overall and per meso group.
Seed-mean per-instance nDCG@10 is used for paired user-bootstrap CIs; seed SD is reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker_meso import aligned_archive, group_labels
from cascade.utils.io import REPO_ROOT, git_hash

GROUP_COLUMNS = ("item_type", "history_length", "history_diversity")


def load_group(runs: list[str]) -> dict:
    """Stack seed ranks; all seeds must share instances, users and retrieval ranks."""
    ranks, ref = [], None
    for run in runs:
        a = aligned_archive(Path(run) / "val_ranks.npz", None if ref is None else ref["inst"], None if ref is None else ref["user"])
        if ref is None:
            ref = a
        elif not np.array_equal(a["retrieval_ranks"], ref["retrieval_ranks"]):
            raise ValueError(f"retrieval baseline differs within group: {run}")
        ranks.append(a["ranks"])
    return {"inst": ref["inst"], "user": ref["user"], "r1": ref["retrieval_ranks"], "ranks": np.stack(ranks)}


def summarize(users: np.ndarray, r1: np.ndarray, ranks: np.ndarray, n_resamples: int) -> dict:
    """Seed-mean metrics of one group against its own retrieval order."""
    found = np.isfinite(r1)
    ndcg = np.stack([M.ndcg_at(r, 10) for r in ranks])
    macro = np.array([M.macro_micro(v, users)["macro"] for v in ndcg])
    cond = np.array([M.macro_micro(v[found], users[found])["macro"] for v in ndcg]) if found.any() else np.full(len(ranks), np.nan)
    recall = np.array([M.macro_micro(M.recall_at(r, 10), users)["macro"] for r in ranks])
    hit1 = r1 <= 10
    return {
        "n_instances": int(len(users)), "n_users": int(np.unique(users).size), "n_retrieved": int(found.sum()),
        "r1_ndcg10_macro": M.macro_micro(M.ndcg_at(r1, 10), users)["macro"],
        "ndcg10_macro_mean": float(macro.mean()), "ndcg10_macro_seed_sd": float(macro.std(ddof=1)) if len(macro) > 1 else None,
        "conditional_ndcg10_macro_mean": float(np.nanmean(cond)),
        "recall10_macro_mean": float(recall.mean()),
        "rescued_mean": float(np.mean([np.sum(found & ~hit1 & (r <= 10)) for r in ranks])),
        "demoted_mean": float(np.mean([np.sum(found & hit1 & (r > 10)) for r in ranks])),
        "vs_r1": paired_bootstrap(users, M.ndcg_at(r1, 10), ndcg.mean(axis=0), n_resamples=n_resamples, seed=0),
        "per_seed_ndcg10_macro": macro.tolist(),
    }


def compare(groups: dict[str, list[str]], n_resamples: int = 1000) -> dict:
    """Overall and per-group summaries; the second group is compared with the first on the same instances."""
    (name_a, runs_a), (name_b, runs_b) = list(groups.items())
    a, b = load_group(runs_a), load_group(runs_b)
    if not (np.array_equal(a["inst"], b["inst"]) and np.array_equal(a["user"], b["user"])):
        raise ValueError("groups are not aligned on the same validation instances")
    data = ProcessedData.load()
    val = data.split_instances("val").set_index("inst").loc[a["inst"]].reset_index()
    labels = group_labels(data, val)
    users = a["user"]
    nd_a = np.stack([M.ndcg_at(r, 10) for r in a["ranks"]]).mean(axis=0)
    nd_b = np.stack([M.ndcg_at(r, 10) for r in b["ranks"]]).mean(axis=0)

    def block(mask: np.ndarray) -> dict:
        return {
            name_a: summarize(users[mask], a["r1"][mask], a["ranks"][:, mask], n_resamples),
            name_b: summarize(users[mask], b["r1"][mask], b["ranks"][:, mask], n_resamples),
            f"{name_b}_minus_{name_a}": paired_bootstrap(users[mask], nd_a[mask], nd_b[mask], n_resamples=n_resamples, seed=0),
        }

    out = {"scope": "validation only, exploratory", "groups": groups, "git_hash": git_hash(),
           "retrieval_rank_differences_between_groups": int(np.sum(a["r1"] != b["r1"])),
           "overall": block(np.ones(len(users), dtype=bool)), "by_group": {}}
    for col in GROUP_COLUMNS:
        for value in sorted(labels[col].astype(str).unique()):
            out["by_group"][f"{col}={value}"] = block(labels[col].astype(str).to_numpy() == value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML/JSON with run_id and two named groups of run dirs")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = REPO_ROOT / "results" / "ranker_compare" / cfg["run_id"]
    out_dir.mkdir(parents=True, exist_ok=False)
    result = compare(cfg["groups"], args.bootstrap)
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.yaml").write_text(Path(args.config).read_text())
    o = result["overall"]
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_seed_ndcg10_macro"}) for k, v in o.items()}, indent=2))


if __name__ == "__main__":
    main()
