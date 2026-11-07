"""Meso metrics (nDCG@10, Recall@10, Hit@3, MRR) by item group and per-instance oracles.

Validation only, saved predictions only (no training, no Test). Oracle = for each
event, keep the better target rank of two systems trained with the same seed; it
is an upper bound for a perfect per-event selector, not an achievable system.

Usage: .venv/bin/python scripts/analysis_meso_oracle.py OUTPUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average

ROOT = Path(__file__).resolve().parents[1]
POLICY = [ROOT / "results/ranker_policy/policy-20261106-pilot",
          ROOT / "results/ranker_policy/policy-20261107-replication"]
RQ3 = ROOT / "results/ranker_comparison/20261025-rq3/validation_ranks.npz"
K200 = 2  # index of K1=200 in candidate_counts [50, 100, 200, 500]


def per_instance(ranks: np.ndarray) -> dict[str, np.ndarray]:
    finite = np.isfinite(ranks)
    return {"ndcg10": M.ndcg_at(ranks, 10), "recall10": M.recall_at(ranks, 10),
            "hit3": M.recall_at(ranks, 3),
            "mrr": np.where(finite, 1.0 / np.where(finite, ranks, 1.0), 0.0)}


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    groups = pd.read_parquet(POLICY[0] / "validation_groups.parquet")
    ids, users = groups.inst.to_numpy(), groups.user.to_numpy()
    masks = {"all": np.ones(len(ids), bool)}
    masks.update({g: groups.item_type.to_numpy() == g for g in ("warm-head", "warm-tail", "new-in-period")})

    systems: dict[str, list[np.ndarray]] = {}
    for root in POLICY:
        for arm in "ABCD":
            for path in sorted(root.glob(f"arm_{arm}_seed*/prefix_ranks.npz")):
                with np.load(path) as z:
                    if not np.array_equal(z["inst"], ids):
                        raise ValueError(f"population mismatch: {path}")
                    systems.setdefault(f"policy_{arm}", []).append(z["ranks"][K200])
    with np.load(RQ3) as z:
        order = pd.Series(np.arange(len(z["inst"])), index=z["inst"]).loc[ids].to_numpy()
        if not np.array_equal(z["user"][order], users):
            raise ValueError("RQ3 user alignment mismatch")
        systems["R1_retrieval_order"] = [z["R1"][0][order]]
        systems["R2_mlp"] = [r[order] for r in z["R2"]]
        systems["R3_din"] = [r[order] for r in z["R3"]]
    for name in ("policy_A", "policy_C"):
        if len(systems.get(name, [])) != 3:
            raise ValueError(f"{name} needs seeds 0/1/2 locally")
    oracles = {"oracle_A_C": ("policy_A", "policy_C"), "oracle_mlp_din": ("R2_mlp", "R3_din")}
    for name, (a, b) in oracles.items():
        systems[name] = [np.fmin(x, y) for x, y in zip(systems[a], systems[b])]

    rows, seed_mean = [], {}
    for name, seeds in systems.items():
        values = [per_instance(r) for r in seeds]
        seed_mean[name] = {m: np.mean([v[m] for v in values], axis=0) for m in values[0]}
        for group, mask in masks.items():
            row = dict(system=name, seeds=len(seeds), group=group, n_events=int(mask.sum()))
            for metric, value in seed_mean[name].items():
                row[metric] = average(value[mask], users[mask])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "meso_metrics.csv", index=False)

    comps = []
    pairs = [("oracle_A_C", "policy_C"), ("oracle_A_C", "policy_A"), ("policy_C", "policy_A"),
             ("oracle_mlp_din", "R2_mlp"), ("R3_din", "R2_mlp")]
    for treat, ctrl in pairs:
        for group, mask in masks.items():
            for metric in ("ndcg10", "hit3", "mrr"):
                ci = paired_bootstrap(users[mask], seed_mean[ctrl][metric][mask],
                                      seed_mean[treat][metric][mask], n_resamples=1000, seed=0)
                comps.append(dict(comparison=f"{treat}-{ctrl}", group=group, metric=metric, **ci))
    pd.DataFrame(comps).to_csv(output / "comparisons.csv", index=False)

    # Who wins, event by event, when the two systems disagree on a top-10 hit (seed 0).
    wins = []
    for a, b in (("policy_A", "policy_C"), ("R2_mlp", "R3_din")):
        ra, rb = systems[a][0], systems[b][0]
        for group, mask in masks.items():
            ha, hb = (ra <= 10) & mask, (rb <= 10) & mask
            wins.append(dict(pair=f"{a} vs {b}", group=group, both=int((ha & hb).sum()),
                             only_first=int((ha & ~hb).sum()), only_second=int((~ha & hb).sum()),
                             either=int((ha | hb).sum())))
    pd.DataFrame(wins).to_csv(output / "disagreement_seed0.csv", index=False)
    print(table.round(6).to_string())
    print(pd.DataFrame(wins).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
