"""Permutation importance of ranker input groups on natural Val lists (validation only, exploratory).

Frozen MLP/DIN checkpoints are scored on every Val list whose target was naturally retrieved in the top 200
(no forced insertions). For each input group consumed by a model, that input is permuted and conditional
per-user macro nDCG@10 is recomputed; importance = baseline - permuted.

* Candidate-level inputs (retrieval score, flags, text cosine, item text, candidate id/leaf) are shuffled across
  the candidate rows of the same list, keeping the list's value distribution but breaking the candidate link.
* User-level inputs (history ids + leaf ids; history-length / recency buckets) are constant within a list, so
  each list instead receives another list's values (a derangement within a randomly ordered batch of lists).
* Derived features (cosine, leaf/store match flags) are NOT recomputed when history is permuted: each group is
  a model-input intervention.

Inference only; never touches Test; trains nothing.

Run from the repo root:
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.permutation_importance \
        --out results/analysis/20261028-permutation-importance
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker import candidate_ranks
from cascade.models.ranker_din import DINRanker
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, git_hash

LABEL = "validation only, exploratory"
CAND_DIR = "data/processed/candidates/two_tower_text_seed0"
MODEL_RUNS = {
    "forced_mlp": ["results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching",
                   "results/ranker_mlp/20261024-shortlist-repeat-seed1-rung1_matching",
                   "results/ranker_mlp/20261024-shortlist-repeat-seed2-rung1_matching"],
    "natural_mlp": [f"results/ranker_mlp/20261028-natural-matching-seed{s}" for s in range(3)],
    "natural_din": [f"results/ranker_din/20261028-din-natural-seed{s}" for s in range(3)],
    "forced_din": [f"results/ranker_din/20261024-din-matching-seed{s}" for s in range(3)],
}
# name -> (level, [(batch key, numeric column or None)])
GROUPS = {
    "retrieval_score": ("candidate", [("numeric", 0)]),
    "text_cosine": ("candidate", [("numeric", 3)]),
    "leaf_eq_last_leaf": ("candidate", [("numeric", 1)]),
    "store_in_history": ("candidate", [("numeric", 2)]),
    "item_text_embedding": ("candidate", [("item_text", None)]),
    "candidate_item_id": ("candidate", [("item_id", None)]),
    "candidate_attributes": ("candidate", [("leaf_category", None), ("top_category", None), ("store", None),
                                            ("price_bucket", None), ("review_count_bucket", None),
                                            ("rating_bucket", None), ("item_age_bucket", None)]),
    "history_items": ("user", [("history_item_ids", None), ("history_leaf_ids", None)]),
    "user_context": ("user", [("history_length_bucket", None), ("recency_bucket", None)]),
}


def consumed_groups(model) -> list[str]:
    """Groups whose inputs the model's forward pass actually reads (from its active-input attributes)."""
    used = set(model.active_categorical_names)
    if model.use_matching:
        used |= {"history_item_ids", "history_leaf_ids"}
    if isinstance(model, DINRanker):
        used |= set(DINRanker.query_feature_names)
    if model.text_projection is not None:
        used.add("item_text")
    cols = set(model.numeric_indices) if bool(model._retrieval_score_enabled) else set(model.numeric_indices) - {0}
    out = []
    for name, (_, fields) in GROUPS.items():
        if any((key == "numeric" and col in cols) or (key != "numeric" and key in used) for key, col in fields):
            out.append(name)
    return out


def load_ranker(run: Path, builder: RankerFeatureBuilder):
    """Frozen checkpoint with the same feature-contract checks as index_rank_eval.load_ranker."""
    rcfg = yaml.safe_load((run / "config.yaml").read_text())
    fs = rcfg["run"]["feature_spec"]
    spec = RankerFeatureSpec(**{**fs, "feature_groups": tuple(fs["feature_groups"])})
    if dict(spec.cardinalities) != builder.item_cardinalities or not np.allclose(
            [rcfg["run"]["score_mean"], rcfg["run"]["score_std"]], [builder.score_mean, builder.score_std], rtol=0, atol=1e-6):
        # atol relaxed from 1e-8: the forced 20261024 MLP checkpoints store score_std 0.08012501895 vs the
        # current builder's 0.08012503385 (float32 accumulation difference, ~2e-7 relative; recorded in results).
        raise ValueError(f"frozen feature contract differs from the current builder: {run}")
    model = create_ranker(spec, rcfg)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
    return model.eval(), int(rcfg["seed"])


def permutation_index(level: str, list_of_pair: np.ndarray, n_lists: int, rng: np.random.Generator) -> np.ndarray:
    """Source pair index for each pair. Candidate level: random permutation within the list's valid pairs.
    User level: every pair takes the first pair of another list (derangement over the batch's lists)."""
    n = len(list_of_pair)
    if level == "candidate":
        # Sort pairs by (list, random key): positions within each list become a random permutation.
        order = np.lexsort((rng.random(n), list_of_pair))
        base = np.argsort(list_of_pair, kind="stable")
        src = np.empty(n, dtype=np.int64)
        src[base] = order
        return src
    first = np.searchsorted(list_of_pair, np.arange(n_lists))
    if n_lists < 2:
        return np.arange(n)
    perm = rng.permutation(n_lists)
    donor = np.empty(n_lists, dtype=np.int64)
    donor[perm] = np.roll(perm, 1)  # cyclic shift of a random order: no list keeps its own values
    return first[donor[list_of_pair]]


def permuted_batch(batch: dict, group: str, src: torch.Tensor) -> dict:
    out = dict(batch)
    for key, col in GROUPS[group][1]:
        if col is None:
            out[key] = batch[key][src]
        else:
            num = out[key].clone() if out[key] is batch[key] else out[key]
            num[:, col] = batch[key][src, col]
            out[key] = num
    return out


def run_pass(models: dict, builder, lists: dict, rows: np.ndarray, groups_by_model: dict, *, repeats: int,
             device: str, batch_pairs: int, check_unused: bool, perm_seed: int = 0) -> dict:
    """Ranks [n_rows] per (model, condition); features built once per batch of lists."""
    width = lists["cand"].shape[1]
    batch_lists = max(2, batch_pairs // width)
    conds = {m: ["baseline"] + [f"{g}#r{r}" for g in groups_by_model[m] for r in range(repeats)] for m in models}
    ranks = {m: {c: np.full(len(rows), np.inf) for c in conds[m]} for m in models}
    t0 = time.time()
    with torch.inference_mode():
        for bi, start in enumerate(range(0, len(rows), batch_lists)):
            rr = rows[start:start + batch_lists]
            cand = lists["cand"][rr]
            valid = cand > 0
            lop = np.repeat(np.arange(len(rr)), width)[valid.ravel()]
            pair_rows = rr[lop]
            pair_cols = np.tile(np.arange(width), len(rr))[valid.ravel()]
            feats = builder.build(lists, pair_rows, pair_cols)
            batch = {k: torch.as_tensor(v, device=device) for k, v in feats.items()}
            sub = {"cand": cand, "target_pos": lists["target_pos"][rr]}
            srcs = {}
            for gi, g in enumerate(GROUPS):
                for r in range(repeats):
                    rng = np.random.default_rng([perm_seed + r, gi, bi])
                    srcs[(g, r)] = torch.as_tensor(permutation_index(GROUPS[g][0], lop, len(rr), rng), device=device)

            def rank_of(model, b):
                s = np.full(cand.shape, -np.inf, dtype=np.float32)
                s[valid] = model(b).float().cpu().numpy()
                return candidate_ranks(s, sub)

            for m, model in models.items():
                ranks[m]["baseline"][start:start + len(rr)] = base = rank_of(model, batch)
                if check_unused and bi == 0:
                    for g in set(GROUPS) - set(groups_by_model[m]):
                        if not np.array_equal(rank_of(model, permuted_batch(batch, g, srcs[(g, 0)])), base):
                            raise AssertionError(f"{m}: permuting supposedly unused group {g} changed ranks")
                for g in groups_by_model[m]:
                    for r in range(repeats):
                        ranks[m][f"{g}#r{r}"][start:start + len(rr)] = rank_of(model, permuted_batch(batch, g, srcs[(g, r)]))
            if bi % 25 == 0:
                print(f"  batch {bi}/{-(-len(rows) // batch_lists)}  {time.time() - t0:.0f}s", flush=True)
    return ranks


def summarise(model_ranks: dict, users: np.ndarray, n_boot: int) -> dict:
    nd = {c: M.ndcg_at(r, 10) for c, r in model_ranks.items()}
    base = M.macro_micro(nd["baseline"], users)
    out = {"baseline_conditional_ndcg10_macro": base["macro"], "baseline_conditional_ndcg10_micro": base["micro"],
           "groups": {}}
    groups = sorted({c.split("#")[0] for c in nd if c != "baseline"})
    for g in groups:
        reps = [c for c in nd if c.startswith(g + "#")]
        perm = [M.macro_micro(nd[c], users)["macro"] for c in reps]
        rec = {"permuted_ndcg10_macro": perm, "drop_mean_over_repeats": float(base["macro"] - np.mean(perm))}
        if n_boot:
            # metric_b - metric_a = baseline - permuted = drop (per-user macro)
            bs = paired_bootstrap(users, nd[reps[0]], nd["baseline"], n_resamples=n_boot, seed=0)
            rec["drop_repeat0_user_bootstrap"] = bs
        out["groups"][g] = rec
    return out


def score_response(model, builder, lists: dict, rows: np.ndarray, *, device: str, batch_pairs: int) -> dict:
    """Within-list link between the model logit and the raw retrieval score on natural-hit lists:
    mean within-list Spearman, and mean logit (centred per list) by retrieval-rank decile (0 = top 20)."""
    width = lists["cand"].shape[1]
    batch_lists = max(1, batch_pairs // width)
    rhos, dec_sum, dec_n = [], np.zeros(10), np.zeros(10)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_lists):
            rr = rows[start:start + batch_lists]
            cand = lists["cand"][rr]
            valid = cand > 0
            pr = np.repeat(rr, width)[valid.ravel()]
            pc = np.tile(np.arange(width), len(rr))[valid.ravel()]
            feats = builder.build(lists, pr, pc)
            s = np.full(cand.shape, np.nan, dtype=np.float64)
            s[valid] = model({k: torch.as_tensor(v, device=device) for k, v in feats.items()}).float().cpu().numpy()
            for i in range(len(rr)):
                v = valid[i]
                logit, ret = s[i, v], lists["score"][rr[i], v].astype(np.float64)
                rhos.append(pd.Series(logit).corr(pd.Series(ret), method="spearman"))
                dec = np.minimum(9, np.arange(v.sum()) * 10 // max(1, v.sum()))  # columns are in retrieval order
                np.add.at(dec_sum, dec, logit - logit.mean())
                np.add.at(dec_n, dec, 1)
    return {"n_lists": int(len(rows)), "mean_within_list_spearman_logit_vs_retrieval_score": float(np.nanmean(rhos)),
            "share_lists_negative_spearman": float(np.nanmean(np.array(rhos) < 0)),
            "mean_centred_logit_by_retrieval_rank_decile": (dec_sum / np.maximum(1, dec_n)).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-types", nargs="+", default=["forced_mlp", "natural_mlp", "natural_din"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="checkpoint seed indices, run in order")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--perm-seed", type=int, default=0, help="offset for permutation RNG (repeat r uses perm_seed + r)")
    ap.add_argument("--batch-pairs", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--score-response-lists", type=int, default=0,
                    help="if >0, also record logit-vs-retrieval-score response on this many natural-hit lists")
    ap.add_argument("--max-lists", type=int, default=0, help="debug: cap on natural-hit lists (0 = all)")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    cfg = {"scope": LABEL, **vars(args), "candidate_dir": CAND_DIR, "k1": 200, "model_runs": MODEL_RUNS,
           "groups": {g: {"level": lv, "inputs": [f"{k}[{c}]" if c is not None else k for k, c in f]}
                      for g, (lv, f) in GROUPS.items()},
           "metric": "conditional per-user macro nDCG@10 over natural-hit Val lists, pessimistic ties",
           "list_order_seed": 12345}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_hash.txt").write_text(git_hash() + "\n")

    data = ProcessedData.load("configs/data.yaml")
    builder = RankerFeatureBuilder.create(data, load_candidates(REPO_ROOT / CAND_DIR, "ranker_train"))
    lists = load_candidates(REPO_ROOT / CAND_DIR, "val", k1=200, force=False)
    if "forced" in lists and np.any(lists["forced"]):
        raise ValueError("val lists loaded with force=False must not contain forced targets")
    hits = np.flatnonzero(lists["target_pos"] >= 0)
    n_all = len(lists["inst"])
    lists = {k: v[hits] for k, v in lists.items()}
    order = np.random.default_rng(12345).permutation(len(hits))  # random users per batch (user-level shuffles)
    if args.max_lists:
        order = order[:args.max_lists]
    users = lists["user"][order].astype(np.int64)
    print(f"natural-hit Val lists: {len(hits)} of {n_all}; using {len(order)}", flush=True)

    results = {"scope": LABEL, "git_hash": git_hash(), "n_val_lists": int(n_all), "n_natural_hit_lists": int(len(hits)),
               "n_lists_used": int(len(order)), "n_users": int(len(np.unique(users))), "models": {}}
    rank_store = {"inst": lists["inst"][order], "user": users}
    for si in args.seeds:
        models, groups_by_model = {}, {}
        for mt in args.model_types:
            run = REPO_ROOT / MODEL_RUNS[mt][si]
            model, seed = load_ranker(run, builder)
            name = f"{mt}/seed{seed}"
            models[name] = model.to(args.device)
            groups_by_model[name] = consumed_groups(model)
            results["models"][name] = {"run": MODEL_RUNS[mt][si], "type": mt, "seed": seed,
                                       "consumed_groups": groups_by_model[name]}
        print(f"pass seed index {si}: {list(models)}", flush=True)
        ranks = run_pass(models, builder, lists, order, groups_by_model, repeats=args.repeats, device=args.device,
                         batch_pairs=args.batch_pairs, check_unused=True, perm_seed=args.perm_seed)
        for name, mr in ranks.items():
            results["models"][name].update(summarise(mr, users, args.n_boot))
            # Sanity: compare with the run's saved Val ranks (all lists, natural hits only).
            run = REPO_ROOT / results["models"][name]["run"]
            vr = run / "val_ranks.npz"
            if vr.exists():
                z = np.load(vr)
                saved = pd.Series(z["ranks"], index=z["inst"]).loc[rank_store["inst"]].to_numpy()
                results["models"][name]["saved_val_ranks_exact_match_fraction"] = float(np.mean(saved == mr["baseline"]))
                results["models"][name]["saved_conditional_ndcg10_macro_from_val_ranks"] = M.macro_micro(
                    M.ndcg_at(saved, 10), users)["macro"]
            vc = run / "validation_comparison.json"
            if vc.exists():
                results["models"][name]["saved_validation_comparison_conditional_ndcg10_macro"] = json.loads(
                    vc.read_text())["ranker"]["conditional_ndcg@10"]["macro"]
            for c, r in mr.items():
                rank_store[f"{name}|{c}"] = r.astype(np.float32)
        if args.score_response_lists:
            for name, model in models.items():
                results["models"][name]["score_response"] = score_response(
                    model, builder, lists, order[:args.score_response_lists], device=args.device,
                    batch_pairs=args.batch_pairs)
        write_outputs(out, results)
        np.savez_compressed(out / "ranks.npz", **rank_store)
        del models
    print("done", flush=True)


def write_outputs(out: Path, results: dict) -> None:
    rows = []
    for name, m in results["models"].items():
        if "groups" not in m:
            continue
        for g, rec in m["groups"].items():
            bs = rec.get("drop_repeat0_user_bootstrap", {})
            rows.append({"model": name, "type": m["type"], "seed": m["seed"], "group": g,
                         "baseline_ndcg10": m["baseline_conditional_ndcg10_macro"],
                         "permuted_ndcg10_mean": float(np.mean(rec["permuted_ndcg10_macro"])),
                         "drop": rec["drop_mean_over_repeats"],
                         "drop_r0_ci_low": bs["ci_low"] if bs else None,
                         "drop_r0_ci_high": bs["ci_high"] if bs else None})
    per_ckpt = pd.DataFrame(rows)
    per_ckpt.to_csv(out / "importance_per_checkpoint.csv", index=False)
    summary = {}
    if len(per_ckpt):
        agg = per_ckpt.groupby(["type", "group"]).agg(drop_mean=("drop", "mean"), drop_sd=("drop", "std"),
                                                      n_seeds=("seed", "nunique"),
                                                      baseline_mean=("baseline_ndcg10", "mean")).reset_index()
        agg["rank_within_type"] = agg.groupby("type")["drop_mean"].rank(ascending=False, method="min").astype(int)
        agg = agg.sort_values(["type", "rank_within_type"])
        agg.to_csv(out / "importance_by_type.csv", index=False)
        for t, d in agg.groupby("type"):
            summary[t] = d.drop(columns="type").to_dict(orient="records")
    results["summary_by_type"] = summary
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
