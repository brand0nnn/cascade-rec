"""HR@3 and MRR@10 for archived Val ranker pipelines, from saved ranks (Val only, exploratory; historical).

Reads only saved Val rank archives (no scoring, no training, never opens Test files). For every system the saved
end-to-end per-user macro nDCG@10 is first recomputed from the same ranks and must match the stored value to
1e-9; otherwise the run stops. Then per-user macro HR@3 (= Recall@3) and MRR@10 (1/rank if rank <= 10 else 0) are
computed over the full Val population (misses = 0), overall and by target item group (the project's
new-in-period / warm-tail / warm-head labels from ``cascade.eval.ranker_meso.group_labels``, as saved by the policy
study), with seed mean +- SD and paired user bootstraps of two seed-averaged differences.

Forced-list systems (20261024 runs, policy arms A/B) are historical: CLAUDE.md / plan.md §15.1 retire them as primary
answers. These numbers are secondary metrics for archived pipelines only.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.secondary_metrics_archived \
        --out results/analysis/20261105-secondary-metrics-archived
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.segments import new_in_period
from cascade.utils.io import REPO_ROOT, git_hash

LABEL = "Val only, exploratory (historical archived pipelines; not primary)"
RQ3 = "results/ranker_comparison/20261025-rq3"
GROUPS_FILE = "results/ranker_policy/policy-pilot-20261029/validation_groups.parquet"
ITEM_TYPES = ("new-in-period", "warm-tail", "warm-head")
N_EVENTS, N_USERS = 232_904, 124_416
TOL = 1e-9


def _runs(prefix: str, seeds=(0, 1, 2)) -> list[str]:
    return [prefix.format(s=s) for s in seeds]


POLICY = {0: "results/ranker_policy/policy-pilot-20261029",
          1: "results/ranker_policy/policy-replication-20261029",
          2: "results/ranker_policy/policy-replication-20261029"}
# name -> list of run directories (each with val_ranks.npz + metrics.json[validation_comparison])
SYSTEMS: dict[str, list[str]] = {
    "forced_mlp": ["results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching",
                   "results/ranker_mlp/20261024-shortlist-repeat-seed1-rung1_matching",
                   "results/ranker_mlp/20261024-shortlist-repeat-seed2-rung1_matching"],
    "forced_din": _runs("results/ranker_din/20261024-din-matching-seed{s}"),
    "natural_mlp": _runs("results/ranker_mlp/20261028-natural-matching-seed{s}"),
    "natural_mlp_ktrain500": _runs("results/ranker_mlp/20261028-natural-ktrain500-seed{s}"),
    "natural_din": _runs("results/ranker_din/20261028-din-natural-seed{s}"),
    "natural_din_ktrain500": _runs("results/ranker_din/20261028-din-natural-ktrain500-seed{s}"),
    **{f"policy_arm_{a}": [f"{POLICY[s]}/arm_{a}_seed{s}" for s in range(3)] for a in "ABCD"},
}
COMPARISONS = {"natural_din_ktrain500_minus_forced_mlp": ("forced_mlp", "natural_din_ktrain500"),
               "natural_mlp_minus_forced_mlp": ("forced_mlp", "natural_mlp")}


def mrr_at(ranks: np.ndarray, cutoff: int) -> np.ndarray:
    """MRR@K per instance: 1/rank if rank <= K else 0 (single relevant item)."""
    out = np.zeros(len(ranks), dtype=np.float64)
    hit = ranks <= cutoff
    out[hit] = 1.0 / ranks[hit]
    return out


def per_instance(ranks: np.ndarray) -> dict[str, np.ndarray]:
    return {"ndcg@10": M.ndcg_at(ranks, 10), "recall@10": M.recall_at(ranks, 10),
            "hr@3": M.recall_at(ranks, 3), "mrr@10": mrr_at(ranks, 10)}


def load_ranks(path: Path, key: str, inst: np.ndarray, users: np.ndarray, row: int | None = None) -> np.ndarray:
    """Saved ranks aligned to the reference instance order (fails on population/user mismatch)."""
    with np.load(path, allow_pickle=False) as z:
        r = z[key] if row is None else z[key][row]
        s_inst, s_user = z["inst"], z["user"]
    if len(s_inst) != len(inst) or set(s_inst.tolist()) != set(inst.tolist()):
        raise ValueError(f"instance population differs: {path}")
    pos = pd.Series(np.arange(len(s_inst)), index=s_inst).loc[inst].to_numpy()
    if not np.array_equal(s_user[pos].astype(np.int64), users):
        raise ValueError(f"user alignment mismatch: {path}")
    r = np.asarray(r, dtype=np.float64)[pos]
    if np.any(r < 1) or np.any(np.isnan(r)):
        raise ValueError(f"invalid ranks in {path}")
    return r


def check_groups(groups: pd.DataFrame) -> dict:
    """Recompute the saved target item type from processed tables (same rule as group_labels)."""
    inst = pd.read_parquet(REPO_ROOT / "data/processed/instances.parquet", columns=["inst", "item"]).set_index("inst")
    items = pd.read_parquet(REPO_ROOT / "data/processed/items.parquet", columns=["item", "first_ts"]).set_index("item")
    seg = pd.read_parquet(REPO_ROOT / "data/processed/item_segments.parquet", columns=["item", "segment"]).set_index("item")
    split = json.loads((REPO_ROOT / "data/processed/split.json").read_text())
    item = inst.loc[groups["inst"].to_numpy(), "item"].to_numpy()
    new = new_in_period(items.loc[item, "first_ts"].to_numpy(), "val", split)
    s = seg.loc[item, "segment"].astype(str).to_numpy()
    recomputed = np.where(new, "new-in-period", np.where(s == "head", "warm-head", "warm-tail"))
    if not np.array_equal(recomputed, groups["item_type"].to_numpy()):
        raise ValueError("saved validation_groups item_type does not match the segment rule")
    return {"source": GROUPS_FILE, "recomputed_match": True,
            "rule": "new-in-period = target first interaction >= T1; otherwise warm-head/warm-tail by the project "
                    "segment (top 20% / rest of items by interaction count before T2, data/processed/item_segments.parquet)",
            "counts": groups["item_type"].value_counts().to_dict()}


def summarise(values: dict[str, list[np.ndarray]], users: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for group, mask in masks.items():
        for metric, per_seed in values.items():
            macro = np.array([M.macro_micro(v[mask], users[mask])["macro"] for v in per_seed])
            micro = np.array([float(v[mask].mean()) for v in per_seed])
            rows.append({"group": group, "metric": metric, "n_seeds": len(per_seed),
                         "macro_mean": float(macro.mean()),
                         "macro_seed_sd": float(macro.std(ddof=1)) if len(macro) > 1 else None,
                         "micro_mean": float(micro.mean()),
                         "macro_per_seed": json.dumps([float(x) for x in macro]),
                         "n_instances": int(mask.sum()), "n_users": int(np.unique(users[mask]).size)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    cfg = {"scope": LABEL, "out": args.out, "n_boot": args.n_boot, "bootstrap_seed": 0, "seed": 0,
           "systems": SYSTEMS, "baselines": {"R0_popularity_order": f"{RQ3}/validation_ranks.npz[R0]",
                                             "R1_retrieval_order": f"{RQ3}/validation_ranks.npz[R1]",
                                             "R1_retrieval_order_rebuilt": "policy arm_A_seed0 val_ranks.npz[retrieval_ranks]"},
           "groups_file": GROUPS_FILE, "comparisons": COMPARISONS,
           "metrics": {"hr@3": "Recall@3 = 1[rank <= 3]", "mrr@10": "1/rank if rank <= 10 else 0",
                       "aggregation": "per-user macro over all 232,904 Val events (misses = 0); seed mean +- SD"},
           "reproduction_tolerance": TOL, "test_files_opened": False}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_hash.txt").write_text(git_hash() + "\n")
    (out / "seed").write_text("0\n")

    groups = pd.read_parquet(REPO_ROOT / GROUPS_FILE)
    inst, users = groups["inst"].to_numpy(), groups["user"].to_numpy().astype(np.int64)
    if len(inst) != N_EVENTS or np.unique(users).size != N_USERS:
        raise ValueError("unexpected Val population")
    group_check = check_groups(groups)
    masks = {"all": np.ones(len(inst), bool), **{t: groups["item_type"].to_numpy() == t for t in ITEM_TYPES}}

    # (system -> list of (source, ranks, saved ndcg macro, saved recall@10 macro))
    sources: dict[str, list[tuple[str, np.ndarray, float, float]]] = {}
    rq3 = json.loads((REPO_ROOT / RQ3 / "comparison.json").read_text())["models"]
    rq3_summary = pd.read_csv(REPO_ROOT / RQ3 / "summary.csv", index_col=0)
    for key, name in (("R0", "R0_popularity_order"), ("R1", "R1_retrieval_order")):
        r = load_ranks(REPO_ROOT / RQ3 / "validation_ranks.npz", key, inst, users, row=0)
        sources[name] = [(f"{RQ3}/validation_ranks.npz[{key}]", r, float(rq3_summary.loc[key, "ndcg10_macro_mean"]),
                          float(rq3_summary.loc[key, "recall10_macro_mean"]))]
    ref = "results/ranker_policy/policy-pilot-20261029/arm_A_seed0"
    vc = json.loads((REPO_ROOT / ref / "metrics.json").read_text())["validation_comparison"]["retrieval_order"]
    sources["R1_retrieval_order_rebuilt"] = [(f"{ref}/val_ranks.npz[retrieval_ranks]",
        load_ranks(REPO_ROOT / ref / "val_ranks.npz", "retrieval_ranks", inst, users),
        vc["ndcg@10"]["macro"], vc["recall@10"]["macro"])]
    for name, runs in SYSTEMS.items():
        sources[name] = []
        for run in runs:
            vc = json.loads((REPO_ROOT / run / "metrics.json").read_text())["validation_comparison"]["ranker"]
            sources[name].append((run, load_ranks(REPO_ROOT / run / "val_ranks.npz", "ranks", inst, users),
                                  vc["ndcg@10"]["macro"], vc["recall@10"]["macro"]))

    # 1) Reproduce the saved nDCG@10 (and Recall@10) from the same ranks before anything else.
    repro = []
    for name, items in sources.items():
        for src, r, nd, rc in items:
            got_nd = M.macro_micro(M.ndcg_at(r, 10), users)["macro"]
            got_rc = M.macro_micro(M.recall_at(r, 10), users)["macro"]
            repro.append({"system": name, "source": src, "saved_ndcg10_macro": nd, "recomputed_ndcg10_macro": got_nd,
                          "abs_diff_ndcg": abs(got_nd - nd), "saved_recall10_macro": rc,
                          "recomputed_recall10_macro": got_rc, "abs_diff_recall": abs(got_rc - rc),
                          "n_finite_ranks": int(np.isfinite(r).sum())})
    repro_df = pd.DataFrame(repro)
    repro_df.to_csv(out / "reproduction_check.csv", index=False)
    bad = repro_df[(repro_df["abs_diff_ndcg"] > TOL) | (repro_df["abs_diff_recall"] > TOL)]
    if len(bad):
        (out / "STOPPED.txt").write_text("saved nDCG@10/Recall@10 not reproduced to 1e-9:\n" + bad.to_string() + "\n")
        raise SystemExit(f"reproduction failed for {len(bad)} sources; see {out / 'STOPPED.txt'}")
    print(f"reproduced {len(repro_df)} saved nDCG@10 values (max abs diff {repro_df['abs_diff_ndcg'].max():.2e})")

    # 2) Secondary metrics.
    rows, per_inst = [], {}
    for name, items in sources.items():
        vals = [per_instance(r) for _, r, _, _ in items]
        per_inst[name] = {m: np.mean([v[m] for v in vals], axis=0) for m in vals[0]}  # seed-averaged per instance
        for row in summarise({m: [v[m] for v in vals] for m in vals[0]}, users, masks):
            rows.append({"system": name, **row})
    table = pd.DataFrame(rows)
    table.to_csv(out / "secondary_metrics.csv", index=False)
    wide = table.pivot_table(index="system", columns=["metric", "group"], values="macro_mean")
    wide.columns = [f"{m}|{g}" for m, g in wide.columns]
    sd = table.pivot_table(index="system", columns=["metric", "group"], values="macro_seed_sd")
    sd.columns = [f"{m}|{g}|sd" for m, g in sd.columns]
    order = list(sources)
    pd.concat([wide, sd], axis=1).loc[order].to_csv(out / "secondary_metrics_wide.csv")

    # 3) Paired user bootstraps of seed-averaged differences (same 1,000 draws for every metric: seed 0).
    boots = []
    for cname, (a, b) in COMPARISONS.items():
        for group, mask in masks.items():
            for metric in ("hr@3", "mrr@10", "ndcg@10"):
                bs = paired_bootstrap(users[mask], per_inst[a][metric][mask], per_inst[b][metric][mask],
                                      n_resamples=args.n_boot, seed=0)
                boots.append({"comparison": cname, "a": a, "b": b, "group": group, "metric": metric, **bs})
    pd.DataFrame(boots).to_csv(out / "bootstrap.csv", index=False)
    results = {"scope": LABEL, "git_hash": git_hash(), "n_events": len(inst), "n_users": int(np.unique(users).size),
               "group_check": group_check, "reproduction_max_abs_diff_ndcg10": float(repro_df["abs_diff_ndcg"].max()),
               "reproduction_max_abs_diff_recall10": float(repro_df["abs_diff_recall"].max()),
               "n_sources_reproduced": len(repro_df), "test_files_opened": False}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    print("done")


if __name__ == "__main__":
    main()
