"""Analyse target/history evidence with saved Val ranks; no model execution."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.analysis.explanation_graph import sha256
from cascade.analysis.graph_target_profiles import (
    align_validation, target_profiles, validate_rank_archive, validate_run_identity, verify_structure,
)
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker_meso import aligned_archive
from cascade.eval.saved_results import funnel, rank_summary, write_json
from cascade.utils.config import REPO_ROOT, load_config

AXES = ("canonical_segment", "item_type", "seen_retriever_train", "seen_all_train", "same_leaf_last",
        "same_store_history", "co_review_retriever_train", "co_review_all_train")


def case_html(case: dict, titles: dict[int, str]) -> str:
    """Source-backed target/history ego view, without inventing candidate slots."""
    witness = case["witness"]
    history_item = witness["history_item"]
    target = case["target"]
    path = f'history item {history_item} ← other reviewer {witness["user"]} → target {target}'
    return '<!doctype html><meta charset="utf-8"><title>Target evidence path</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px}svg{width:100%}li{margin:8px 0}</style>' + f'<h1>{html.escape(case["bucket"])}</h1><p>Instance {case["inst"]}; target: {html.escape(titles[target])}</p><p>Retrieval rank: {case["retrieval_rank"]}; MLP seed-0 rank: {case["mlp_rank"]}.</p><svg viewBox="0 0 1000 140" role="img" aria-label="Co-review witness path"><line x1="140" y1="70" x2="860" y2="70" stroke="#555"/><rect x="10" y="40" width="260" height="60" rx="8" fill="#ddeaff"/><rect x="370" y="40" width="260" height="60" rx="8" fill="#eef4ed"/><rect x="730" y="40" width="260" height="60" rx="8" fill="#ddeaff"/>' + f'<text x="25" y="75">history item {history_item}</text><text x="385" y="75">other reviewer {witness["user"]}</text><text x="745" y="75">target {target}</text></svg><p>{html.escape(path)}</p><p>Witness review timestamps: {witness["history_ts"]} and {witness["target_ts"]}; both strictly before {case["cutoff"]}. Query user excluded.</p><h2>Last prior products</h2><ul>' + ''.join('<li>'+html.escape(titles[item])+'</li>' for item in case["last_history_items"]) + '</ul><p>Metadata titles are crawl-time snapshots. Reviews are not verified purchases. This path is associated evidence, not model attribution or a causal explanation. Other candidates and their slot amplification remain unavailable.</p>'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="results/targeted_study/data.yaml")
    ap.add_argument("--graph-dir", default="results/explanation_graph/20261027-base")
    ap.add_argument("--saved-config", default="configs/evaluation/saved_results.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = ap.parse_args()
    data = ProcessedData.load(args.data_config)
    instances = data.split_instances("val")
    cfg = load_config(args.saved_config)
    graph = REPO_ROOT / args.graph_dir
    manifest = json.loads((graph / "manifest.json").read_text())
    if int(manifest["snapshots"]["all_train"]["strict_before_ms"]) != int(data.split["t1"]) or int(manifest["snapshots"]["retriever_train"]["strict_before_ms"]) != int(data.split["t_ranker_train"]):
        raise ValueError("graph and restored data cutoffs differ")
    reviews, structure_provenance = verify_structure(data, load_config(args.data_config), graph, manifest)
    archives, paths, source_runs = {}, [], []
    reference = None
    for system in ("matching", "din"):
        entry = cfg["ranker_systems"][system]
        if entry["seeds"] != [0, 1, 2] or len(entry["runs"]) != 3:
            raise ValueError("target analysis requires completed ranker seeds 0/1/2")
        values = []
        if len({str((REPO_ROOT / run).resolve()) for run in entry["runs"]}) != 3:
            raise ValueError("saved seed run paths must be unique")
        for expected_seed, run in zip(entry["seeds"], entry["runs"]):
            path = REPO_ROOT / run / "val_ranks.npz"
            run_config_path, metrics_path = path.parent / "config.yaml", path.parent / "metrics.json"
            run_config = load_config(run_config_path)
            metrics = json.loads(metrics_path.read_text())
            validate_run_identity(run_config, metrics, expected_seed)
            archive = aligned_archive(path)
            validate_rank_archive(archive, len(instances))
            align_validation(instances, archive, int(data.split["t1"]), int(data.split["t2"]))
            if reference is None:
                reference = archive["retrieval_ranks"]
            if not np.array_equal(archive["retrieval_ranks"], reference) or not np.array_equal(np.isfinite(archive["ranks"]), np.isfinite(reference)):
                raise ValueError("saved rankers changed baseline ranks or natural retrieval membership")
            values.append(archive["ranks"])
            paths.append(path)
            actual_ndcg = M.macro_micro(M.ndcg_at(archive["ranks"], 10), archive["user"])["macro"]
            np.testing.assert_allclose(actual_ndcg, metrics["validation_comparison"]["ranker"]["ndcg@10"]["macro"], atol=1e-12, rtol=0)
            source_runs.append({"system": system, "seed": expected_seed,
                                "config": {"path": str(run_config_path.relative_to(REPO_ROOT)), "sha256": sha256(run_config_path)},
                                "metrics": {"path": str(metrics_path.relative_to(REPO_ROOT)), "sha256": sha256(metrics_path)},
                                "ranks": {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)},
                                "reproduced_ndcg10_macro": actual_ndcg})
        archives[system] = np.stack(values)
    out = REPO_ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    profiles = target_profiles(data, instances, reviews)
    profiles.to_parquet(out / "target_profiles.parquet", index=False)
    users = instances["user"].to_numpy()
    found = np.isfinite(reference)
    masks = [("overall", "all", np.ones(len(profiles), dtype=bool))]
    for axis in AXES:
        masks.extend((axis, str(group), profiles[axis].to_numpy() == group) for group in sorted(profiles[axis].unique()))
    rows, funnels, intervals = [], [], []
    for axis, group, mask in masks:
        print(f"Summarise {axis}/{group}: {mask.sum()} events", flush=True)
        u, stage, present = users[mask], reference[mask], found[mask]
        base = rank_summary(u, stage, present)
        rows.append({"system": "retrieval_order", "seed": -1, "axis": axis, "group": group,
                     "n_instances": int(mask.sum()), "n_users": int(np.unique(u).size), **base})
        for system, ranks in archives.items():
            for seed, rank in enumerate(ranks):
                r = rank[mask]
                summary = rank_summary(u, r, present)
                breakdown, _ = funnel(u, stage, r, present)
                outcomes = breakdown.pop("outcomes")
                rows.append({"system": system, "seed": seed, "axis": axis, "group": group,
                             "n_instances": int(mask.sum()), "n_users": int(np.unique(u).size), **summary, **breakdown})
                for outcome, stats in outcomes.items():
                    funnels.append({"system": system, "seed": seed, "axis": axis, "group": group, "outcome": outcome, **stats})
        base_values = M.ndcg_at(stage, 10)
        mlp = np.stack([M.ndcg_at(r[mask], 10) for r in archives["matching"]]).mean(axis=0)
        din = np.stack([M.ndcg_at(r[mask], 10) for r in archives["din"]]).mean(axis=0)
        for comparison, a, b in (("matching_minus_retrieval", base_values, mlp), ("din_minus_matching", mlp, din)):
            intervals.append({"axis": axis, "group": group, "comparison": comparison,
                              **paired_bootstrap(u, a, b, n_resamples=args.bootstrap_resamples, seed=0)})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    pd.DataFrame(funnels).to_csv(out / "funnel.csv", index=False)
    interval_table = pd.DataFrame(intervals)
    interval_table.to_csv(out / "uncertainty.csv", index=False)
    numeric = [c for c in per_seed.select_dtypes(include="number").columns if c != "seed"]
    aggregate = per_seed.groupby(["axis", "group", "system"], sort=True)[numeric].agg(["mean", "std"])
    aggregate.columns = [a + "_" + b for a, b in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(out / "groups.csv", index=False)
    examples = []
    seed_rank = archives["matching"][0]
    buckets = [("rescued_with_co_review", found & (reference > 10) & (seed_rank <= 10)),
               ("demoted_despite_co_review", found & (reference <= 10) & (seed_rank > 10)),
               ("retrieval_miss_despite_co_review", ~found)]
    rng = np.random.default_rng(0)
    used = set()
    titles = data.items.set_index("item")["title"].fillna("").astype(str).to_dict()
    for bucket, mask in buckets:
        eligible = np.flatnonzero(mask & (profiles["co_review_all_train"].to_numpy() == "true"))
        eligible = np.array([i for i in eligible if int(profiles.iloc[i]["inst"]) not in used], dtype=int)
        if not len(eligible):
            continue
        selected = int(rng.choice(eligible))
        row = profiles.iloc[selected]
        instance = instances.iloc[selected]
        history = data.histories(np.array([instance["user"]]), np.array([instance["hist_len"]]), data.max_hist_len)[0]
        case = {"bucket": bucket, "eligible_events": len(eligible), "inst": int(row["inst"]),
                "target": int(row["target"]), "query_user": int(row["user"]),
                "retrieval_rank": str(reference[selected]), "mlp_rank": str(seed_rank[selected]),
                "last_history_items": [int(i) for i in history[history > 0][-5:]], "cutoff": int(data.split["t1"]),
                "witness": {"user": int(row["all_train_witness_user"]), "history_item": int(row["all_train_witness_history_item"]),
                            "history_ts": int(row["all_train_witness_history_ts"]), "target_ts": int(row["all_train_witness_target_ts"])}}
        used.add(case["inst"])
        examples.append(case)
        (out / (bucket + ".html")).write_text(case_html(case, titles))
    write_json(out / "examples.json", {"seed": 0, "selection": "seeded draw from stable eligible Val order in listed bucket order, without replacement", "cases": examples})
    provenance = {"status": "target_profiles_complete_candidate_profiles_deferred", "population": {"events": len(instances), "users": len(np.unique(users))},
                  "data_config": {"path": args.data_config, "sha256": sha256(REPO_ROOT / args.data_config)},
                  "graph_manifest_sha256": sha256(graph / "manifest.json"),
                  "review_edges_sha256": sha256(graph / "review_edges.parquet"),
                  "graph_review_integrity": "Base manifest has no output-edge hash; review content exactly matches the restored pre-T1 interaction projection.",
                  "restored_structure": structure_provenance,
                  "saved_ranks": [{"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256(p)} for p in paths],
                  "source_runs": source_runs,
                  "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in (
                      Path(__file__), REPO_ROOT / "src/cascade/analysis/graph_target_profiles.py",
                      REPO_ROOT / "src/cascade/analysis/explanation_graph.py",
                      REPO_ROOT / "src/cascade/eval/metrics.py", REPO_ROOT / "src/cascade/eval/bootstrap.py",
                      REPO_ROOT / "src/cascade/eval/saved_results.py", REPO_ROOT / "src/cascade/eval/ranker_meso.py",
                      REPO_ROOT / "src/cascade/data/instances.py", REPO_ROOT / "src/cascade/data/split.py",
                      REPO_ROOT / "src/cascade/utils/config.py")},
                  "saved_config": {"path": args.saved_config, "sha256": sha256(REPO_ROOT / args.saved_config)},
                  "cutoffs": {"retriever_train": int(data.split["t_ranker_train"]), "all_train": int(data.split["t1"])},
                  "bootstrap": {"resamples": args.bootstrap_resamples, "seed": 0, "scope": "exploratory paired users conditional on archived training runs"},
                  "alignment": "complete restored Val IDs/order/users checked against all six saved rank archives",
                  "stage1_rank_convention": "Historical returned positions from the ranker archives, not reconciled pessimistic full-catalogue ranks; original report notes two nDCG event differences with no hit change.",
                  "device_profiles": "not computed: locked manual audit incomplete", "candidate_slot_profiles": "not computed: original full candidates/vectors unavailable",
                  "test_predictions_loaded": False, "models_executed": False}
    write_json(out / "provenance.json", provenance)
    lines = ["# Target/history graph profiles from saved validation ranks", "",
             f"{len(instances):,} aligned Val events / {len(np.unique(users)):,} users. No model execution or Test predictions.",
             "", "Target/history relations only. Candidate slot-amplification and device profiles are not computed.",
             "Co-review requires another reviewer, with both review edges strictly before the named cutoff. Histories use the last 50 strictly prior items.",
             "Stage-1 ranks retain historical returned positions from the ranker archives. They are not the reconciled pessimistic full-catalogue companion: the original report identifies two nDCG event differences and no hit-count change.",
             "", "## Profile results", "", "User-macro metrics; matching/DIN average archived seeds 0/1/2. Each slice has its own user denominator.",
             "", "| Axis | Group | Events | Users | R@200 | Retrieval nDCG@10 | MLP nDCG@10 | DIN nDCG@10 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for axis, group, mask in masks:
        group_rows = aggregate[(aggregate["axis"] == axis) & (aggregate["group"] == group)].set_index("system")
        r = group_rows.loc["retrieval_order"]
        lines.append(f"| {axis} | {group} | {int(mask.sum())} | {int(np.unique(users[mask]).size)} | {r['retrieval_coverage_macro_mean']:.6f} | {r['ndcg10_macro_mean']:.6f} | {group_rows.loc['matching','ndcg10_macro_mean']:.6f} | {group_rows.loc['din','ndcg10_macro_mean']:.6f} |")
    lines += ["", "## Selected exploratory differences", "",
              "These overlapping profiles were highlighted after the first profile report; intervals are unadjusted, not independent confirmations.",
              "", "| Profile | Group | MLP minus retrieval nDCG@10 | Paired-user 95% CI |", "|---|---|---:|---|"]
    selected_axes = {"co_review_retriever_train", "co_review_all_train", "seen_retriever_train", "same_store_history", "same_leaf_last"}
    for row in interval_table.itertuples(index=False):
        if row.axis in selected_axes and row.comparison == "matching_minus_retrieval":
            lines.append(f"| {row.axis} | {row.group} | {row.diff:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] |")
    lines += ["", "## Interpretation limits and next evidence", "",
              "- Profile associations do not establish causal model use, item relevance, purchases, or user preferences.",
              "- New items have no pre-entry collaborative evidence by construction; absence of co-review is not an item-quality judgement.",
              "- Canonical cold/tail/head labels use pre-T2 counts for descriptive continuity, never model features; they overlap with new-in-Val labels.",
              "- Missing category/store values remain unknown; they do not match each other. These analytic rules may differ from historical model feature handling.",
              "- Seen-in-training flags mean at least one prior interaction; they do not imply an ID met a model's vocabulary threshold.",
              "- Per-seed five-way funnel counts, full-denominator contributions, at-risk rescue/demotion rates and support are in funnel.csv/per_seed.csv; paired-user intervals in uncertainty.csv are exploratory and unadjusted.",
              "- Small source-backed case graphs include successes and counterexamples; cases illustrate profiles, not independent confirmations.",
              "- To measure candidate redistribution or device mismatches, recover original candidates/vectors and complete the locked manual parser audit first."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": provenance["status"], "population": provenance["population"], "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
