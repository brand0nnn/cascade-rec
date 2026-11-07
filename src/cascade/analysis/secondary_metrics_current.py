"""HR@3 and MRR@10 for the regenerated full-data pipeline (Val only, exploratory; regenerated pipeline).

Systems on the identical natural frozen-MiniLM/90-day-recency K200 Val lists: popularity order (strictly pre-T0
review counts, pessimistic target ties; plan.md §15.3 Phase 1 / ``steps_final_analysis.popularity``), retrieval
order, the four regenerated rankers (bce_text, pairwise_text, pairwise_coreview, pairwise_combined) and, when its
evaluation is complete, bce_coreview. Ranks come from saved scores only (no scoring/training, Test never opened);
pessimistic ties, padding excluded. HR@3 = Recall@3; MRR@10 = 1/rank if rank <= 10 else 0. nDCG@10/Recall@10 are
recomputed from the same ranks for ordering comparison. Per-user macro over the full population (misses = 0),
overall and new-in-Val / warm targets; seed mean +- SD (ddof 0); paired user bootstrap (1,000 draws, seed 0) of
seed-mean event differences for the full_promising.yaml primary contrasts and the plan.md §15 STePS contrasts.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.secondary_metrics_current --phase register \
        --out results/analysis/20261105-secondary-metrics-current
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.secondary_metrics_current --phase run \
        --out results/analysis/20261105-secondary-metrics-current
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.analysis import coreview_blend_oracle as B
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.utils.io import REPO_ROOT, git_hash

LABEL = B.LABEL
SYSTEMS = ("popularity", "retrieval", "bce_text", "pairwise_text", "pairwise_coreview", "pairwise_combined",
           "bce_coreview")
METRICS = ("ndcg10", "recall10", "hr3", "mrr10")
PRIMARY = (("retrieval", "pairwise_text"), ("bce_text", "pairwise_text"), ("pairwise_text", "pairwise_coreview"),
           ("pairwise_coreview", "pairwise_combined"), ("retrieval", "pairwise_coreview"),
           ("retrieval", "pairwise_combined"))
STEPS = (("retrieval", "popularity"), ("popularity", "bce_coreview"), ("retrieval", "bce_text"),
         ("retrieval", "bce_coreview"), ("bce_text", "pairwise_text"), ("bce_text", "bce_coreview"),
         ("pairwise_text", "pairwise_coreview"), ("pairwise_coreview", "bce_coreview"))
HYPOTHESIS = ("Pre-registered: HR@3 and MRR@10 give the same system ordering as nDCG@10; pairwise's new-item "
              "advantage over BCE is larger (relative) at HR@3 than at nDCG@10.")
N_BOOT, BOOT_SEED = 1000, 0


def mrr_at(ranks: np.ndarray, cutoff: int) -> np.ndarray:
    """1/rank if rank <= cutoff else 0 (single relevant item)."""
    out = np.zeros(len(ranks))
    hit = ranks <= cutoff
    out[hit] = 1.0 / ranks[hit]
    return out


def per_event(ranks: np.ndarray) -> dict[str, np.ndarray]:
    return {"ndcg10": M.ndcg_at(ranks, 10), "recall10": M.recall_at(ranks, 10), "hr3": M.recall_at(ranks, 3),
            "mrr10": mrr_at(ranks, 10)}


def contrasts() -> list[tuple[str, str, str]]:
    """Union of the two registered contrast families; shared pairs carry both family labels."""
    families: dict[tuple[str, str], list[str]] = {}
    for family, pairs in (("full_promising_primary", PRIMARY), ("plan15_steps", STEPS)):
        for pair in pairs:
            families.setdefault(pair, []).append(family)
    return [("+".join(f), a, b) for (a, b), f in families.items()]


def relative_advantage_ci(users, mask, base, cand, m1, m2, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Joint user bootstrap of rel(m1) - rel(m2), rel(m) = macro(cand_m)/macro(base_m) - 1, on one group."""
    u = users[mask]
    frame = pd.DataFrame({"u": u, "b1": base[m1][mask], "c1": cand[m1][mask], "b2": base[m2][mask], "c2": cand[m2][mask]})
    per = frame.groupby("u").mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(0, n_boot, 32):
        idx = rng.integers(0, len(per), size=(min(32, n_boot - i), len(per)))
        mean = per[idx].mean(1)
        boots[i:i + len(idx)] = (mean[:, 1] / mean[:, 0] - 1) - (mean[:, 3] / mean[:, 2] - 1)
    est = per.mean(0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"base_metric": float(est[0]), "cand_metric": float(est[1]), "base_vs": float(est[2]), "cand_vs": float(est[3]),
            "rel_metric": float(est[1] / est[0] - 1), "rel_vs": float(est[3] / est[2] - 1),
            "diff": float((est[1] / est[0] - 1) - (est[3] / est[2] - 1)), "ci_low": float(lo), "ci_high": float(hi),
            "n_users": int(len(per)), "n_resamples": n_boot}


def register(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg = {"scope": LABEL, "registered_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "hypothesis": HYPOTHESIS, "parent_study": str(B.PARENT.relative_to(REPO_ROOT)),
           "factorial_study": str(B.FACTORIAL.relative_to(REPO_ROOT)), "systems": list(SYSTEMS),
           "metrics": {"hr3": "Recall@3 (target rank <= 3)", "mrr10": "1/rank if rank <= 10 else 0",
                       "ndcg10": "unchanged harness definition", "recall10": "unchanged harness definition"},
           "popularity": "strictly pre-T0 (retriever_train) review counts of the K200 candidates; pessimistic target ties",
           "ranks": "saved scores; pessimistic ties; padding excluded", "groups": ["all", "new", "warm"],
           "new_definition": "target first review >= T1 (pipeline groups_for new_in_val)",
           "aggregation": "per-user macro over the full population (misses 0); seed mean +- SD ddof0",
           "contrasts": [list(c) for c in contrasts()],
           "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "user, paired, after seed-mean event metrics"},
           "relative_advantage_test": "joint user bootstrap of rel_hr3 - rel_ndcg10 for pairwise_text vs bce_text on new targets",
           "rules": "Val only; no training; no Test file opened; exploratory metrics, no definition changes"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_hash.txt").write_text(git_hash() + "\n")
    (out / "seed").write_text(f"{BOOT_SEED}\n")
    print(f"registered {out}")


def run(out: Path, parent: Path, factorial: Path, flags: dict[str, str] | None = None) -> None:
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    if cfg.get("hypothesis") != HYPOTHESIS:
        raise ValueError("registration mismatch")
    if (out / "secondary_metrics.csv").exists():
        raise FileExistsError("outputs exist")
    from cascade.pipelines.steps_final_analysis import popularity
    ctx = B.load_context(parent)
    val, users = ctx["val"], ctx["users"]
    masks = B.group_masks(ctx["new_target"])
    counts, _ = popularity(ctx["data"])
    ranks, status, checks = {}, {}, []
    for system in SYSTEMS:
        if system == "popularity":
            ranks[system] = B.target_ranks(counts[np.asarray(val["cand"])].astype(np.float64), val["cand"], val["target"])[None]
        elif system == "retrieval":
            r = B.target_ranks(val["score"], val["cand"], val["target"])
            if not np.array_equal(r, np.asarray(val["retrieval_rank"])):
                raise ValueError("retrieval replay")
            ranks[system] = r[None]
        elif B.available(system, parent, factorial):
            ranks[system], proof = B.replay_check(system, parent, factorial, val)
            checks += proof
        else:
            status[system] = "not_available"
            continue
        status[system] = "computed"
        if not np.all(np.isfinite(ranks[system]) == np.isfinite(np.asarray(val["retrieval_rank"]))[None]):
            raise ValueError(f"{system} changed the candidate ceiling")
    ev_seed = {s: [per_event(r) for r in rs] for s, rs in ranks.items()}
    ev = {s: {m: np.mean([e[m] for e in es], 0) for m in METRICS} for s, es in ev_seed.items()}
    rows = []
    for system in ranks:
        for g, mask in masks.items():
            row = {"system": system, "group": g, "events": int(mask.sum()), "users": int(np.unique(users[mask]).size),
                   "seeds": len(ev_seed[system])}
            for m in METRICS:
                vals = [B.macro(e[m][mask], users[mask]) for e in ev_seed[system]]
                row[m] = float(np.mean(vals)); row[m + "_seed_sd"] = float(np.std(vals))
                for i, v in enumerate(vals):
                    row[f"{m}_seed{i}"] = v
            rows.append(row)
    flags = flags or {}
    frame = pd.DataFrame(rows)
    frame["note"] = frame.system.map(flags).fillna("")
    frame.to_csv(out / "secondary_metrics.csv", index=False)
    ci_rows = []
    for family, ref, cand in contrasts():
        if ref not in ev or cand not in ev:
            ci_rows.append({"family": family, "reference": ref, "candidate": cand, "status": "not_available"})
            continue
        for g, mask in masks.items():
            for m in METRICS:
                ci_rows.append({"family": family, "reference": ref, "candidate": cand, "group": g, "metric": m,
                                "status": "computed", **paired_bootstrap(users[mask], ev[ref][m][mask], ev[cand][m][mask],
                                                                         n_resamples=N_BOOT, seed=BOOT_SEED)})
    ci = pd.DataFrame(ci_rows)
    ci["note"] = [("; ".join(f"{x}: {flags[x]}" for x in (r, c) if x in flags)) for r, c in zip(ci.reference, ci.candidate)]
    ci.to_csv(out / "paired_differences.csv", index=False)
    rel = []
    for base, cand in (("bce_text", "pairwise_text"), ("bce_coreview", "pairwise_text")):
        if base in ev and cand in ev:
            for g in ("new", "warm", "all"):
                for m1 in ("hr3", "mrr10"):
                    rel.append({"base": base, "candidate": cand, "group": g, "metric": m1, "vs": "ndcg10",
                                **relative_advantage_ci(users, masks[g], ev[base], ev[cand], m1, "ndcg10")})
    pd.DataFrame(rel).to_csv(out / "relative_advantage.csv", index=False)
    order = {}
    for g in masks:
        sub = frame[frame.group == g]
        order[g] = {m: sub.sort_values(m, ascending=False).system.tolist() for m in METRICS}
    (out / "orderings.json").write_text(json.dumps(order, indent=1))
    (out / "checks.json").write_text(json.dumps({"label": LABEL, "status": status, "replay": checks, "flags": flags,
                                                 "popularity_counts_period": "retriever_train (< T0)"}, indent=1))
    print(frame[["system", "group", *METRICS]].to_string())
    print(json.dumps(order, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("register", "run"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--parent", type=Path, default=B.PARENT)
    ap.add_argument("--factorial", type=Path, default=B.FACTORIAL)
    ap.add_argument("--flag", action="append", default=[], help="system=note, e.g. a reproduction caveat")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if args.phase == "register":
        return register(out)
    return run(out, args.parent, args.factorial, dict(f.split("=", 1) for f in args.flag))


if __name__ == "__main__":
    main()
