"""Exploratory validation-only report for policy-study extension arms (E/F/G/H).

Also runs E3: the candidate-count (K1) curve for every arm, with paired user
bootstrap differences against each arm's best observed K. Compares arms
against fresh A (reference) and C (natural-only) on identical natural lists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average

COUNTS = [50, 100, 200, 500]


def load(studies: list[Path]) -> tuple[dict, pd.DataFrame]:
    archives: dict[str, dict[int, dict]] = {}
    groups = None
    for root in studies:
        g = pd.read_parquet(root / "validation_groups.parquet")
        if groups is None:
            groups = g
        elif not g[["inst", "user"]].equals(groups[["inst", "user"]]):
            raise ValueError(f"validation population differs in {root}")
        for directory in sorted(root.glob("arm_*_seed*")):
            if not (directory / "prefix_ranks.npz").is_file():
                continue
            arm, seed = directory.name[4:].split("_seed")
            if int(seed) in archives.get(arm, {}):
                raise ValueError(f"duplicate {arm} seed {seed}")
            with np.load(directory / "prefix_ranks.npz") as z:
                archives.setdefault(arm, {})[int(seed)] = {k: z[k] for k in z.files}
    return archives, groups


def summarize(archives: dict, groups: pd.DataFrame, resamples: int = 1000) -> dict:
    ids, users = groups.inst.to_numpy(), groups.user.to_numpy()
    masks = [("overall", "all", np.ones(len(groups), bool))]
    for v in sorted(groups["item_type"].unique()):
        masks.append(("item_type", str(v), groups["item_type"].to_numpy() == v))
    masks.append(("new_in_period", "new", groups.new_at_val_entry.to_numpy(dtype=bool)))
    reference_found = None
    means, rows, expansion = {}, [], []
    for arm, by_seed in sorted(archives.items()):
        per_seed = []
        for seed, z in sorted(by_seed.items()):
            if not np.array_equal(z["inst"], ids) or z["candidate_counts"].tolist() != COUNTS:
                raise ValueError("archive alignment mismatch")
            found = np.isfinite(z["ranks"])
            if reference_found is None:
                reference_found = found
            if not np.array_equal(found, reference_found):
                raise ValueError("arms must share identical retrieval opportunities")
            ndcg = np.stack([M.ndcg_at(r, 10) for r in z["ranks"]])
            per_seed.append(ndcg)
            for ki, k in enumerate(COUNTS):
                for axis, label, mask in masks:
                    rows.append(dict(arm=arm, seed=seed, k1=k, axis=axis, group=label,
                                     ndcg10=average(ndcg[ki, mask], users[mask])))
            for ki in range(3):
                newly = ~found[ki] & found[ki + 1]
                delta = ndcg[ki + 1] - ndcg[ki]
                expansion.append(dict(arm=arm, seed=seed, k_low=COUNTS[ki], k_high=COUNTS[ki + 1],
                    new_target_contribution=average(delta * newly, users),
                    existing_target_contribution=average(delta * found[ki], users),
                    total_delta_ndcg10=average(delta, users)))
        means[arm] = (sorted(by_seed), np.mean(per_seed, axis=0))
    metrics = pd.DataFrame(rows)
    comparisons = []
    for control in ("A", "C"):
        if control not in means:
            continue
        for arm, (seeds, value) in means.items():
            if arm == control or seeds != means[control][0]:
                continue
            for ki in (2, 3):
                for axis, label, mask in masks:
                    ci = paired_bootstrap(users[mask], means[control][1][ki, mask], value[ki, mask],
                                          n_resamples=resamples, seed=0)
                    comparisons.append(dict(comparison=f"{arm}-{control}", k1=COUNTS[ki],
                                            axis=axis, group=label, seeds=len(seeds), **ci))
    k_curve = []
    for arm, (seeds, value) in means.items():
        overall = [average(value[ki], users) for ki in range(4)]
        best = int(np.argmax(overall))
        for ki, k in enumerate(COUNTS):
            row = dict(arm=arm, seeds=len(seeds), k1=k, ndcg10=overall[ki], best_k1=COUNTS[best])
            if ki != best:
                ci = paired_bootstrap(users, value[best], value[ki], n_resamples=resamples, seed=0)
                row.update({f"vs_best_{key}": val for key, val in ci.items()})
            k_curve.append(row)
    qualified = []
    for arm, (seeds, value) in means.items():
        if arm == "A" or "A" not in means or seeds != [0, 1, 2] or means["A"][0] != seeds:
            continue
        overall = paired_bootstrap(users, means["A"][1][2], value[2], n_resamples=resamples, seed=0)
        new = masks[-1][2]
        delta_new = average(value[2, new] - means["A"][1][2, new], users[new])
        if overall["ci_low"] > 0 and delta_new >= 0:
            qualified.append(arm)
    return dict(metrics=metrics, comparisons=pd.DataFrame(comparisons),
                expansion=pd.DataFrame(expansion), k_curve=pd.DataFrame(k_curve),
                qualified=qualified, seeds={a: s for a, (s, _) in means.items()})


def write(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name in ("metrics", "comparisons", "expansion", "k_curve"):
        result[name].to_csv(output / f"{name}.csv", index=False)
    m = result["metrics"]
    seed_mean = m.groupby(["arm", "k1", "axis", "group"]).ndcg10.agg(["mean", "std"]).reset_index()
    lines = ["# Policy-study extension (exploratory, validation only)", "",
             f"Seeds per arm: {result['seeds']}. No Test predictions loaded.", "",
             "## nDCG@10 by arm and K1 (seed mean)", "",
             "| Arm | K50 | K100 | K200 | K500 | New-item K200 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in sorted(result["seeds"]):
        o = seed_mean[(seed_mean.arm == arm) & (seed_mean.axis == "overall")].set_index("k1")["mean"]
        n = seed_mean[(seed_mean.arm == arm) & (seed_mean.axis == "new_in_period") & (seed_mean.k1 == 200)]["mean"]
        lines.append(f"| {arm} | " + " | ".join(f"{o[k]:.6f}" for k in COUNTS) + f" | {n.iloc[0]:.6f} |")
    lines += ["", "## K200 paired differences (seed-mean, user bootstrap 95% CI)", "",
              "| Comparison | Group | Delta | CI |", "|---|---|---:|---|"]
    c = result["comparisons"]
    for _, r in c[(c.k1 == 200) & c.axis.isin(["overall", "new_in_period"])].iterrows():
        lines.append(f"| {r.comparison} | {r.group} | {r['diff']:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] |")
    lines += ["", f"Qualified vs A (3 seeds, overall CI>0, no new-item loss): {result['qualified']}",
              "Extension arms were added after the pilot was seen; treat any qualification as exploratory.",
              "See k_curve.csv for E3 (best K per arm and paired gaps), expansion.csv for gain/competition."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(
        {"qualified": result["qualified"], "seeds": result["seeds"], "test_predictions_loaded": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    archives, groups = load(args.studies)
    write(summarize(archives, groups, args.resamples), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
