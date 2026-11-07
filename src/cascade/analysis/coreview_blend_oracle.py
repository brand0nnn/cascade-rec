"""BCE/pairwise oracle and serving-time candidate-level blend (Val only, exploratory; regenerated pipeline).

Question: how much of the new-vs-warm trade-off between the BCE and pairwise rankers of the regenerated
full-data pipeline (``results/local_experiments/20261103-full-promising-ranking`` and its BCE+co-review cell
``20261104-objective-coreview``) can be recovered by combining them?

(a) Target-conditioned oracles (upper bounds only; the choice uses the true target):
    * event oracle: per Val event pick the system with the higher seed-mean nDCG@10 (ties -> BCE) and report that
      system's seed-mean nDCG@10 / Recall@10;
    * group oracle: pairwise for new-in-Val targets, BCE for warm targets (the target's group is unknown at query time).
(b) Candidate-level blend usable at serving time: within each natural K200 list z-standardise each model's logits
    over the valid (non-padding) candidates, then score = (1-a) z_BCE + a z_PW for warm candidates and
    (1-b) z_BCE + b z_PW for new candidates. "New candidate" = first review at/after the start of the query's period
    (T1 for Val, T0 = t_ranker_train for the ranker-training tuning queries), the pipeline's own new-in-period rule
    applied to the candidate, not the target. It equals "the candidate has zero reviews before the period start",
    so it needs only pre-period interactions plus the pipeline's existing availability-filtered pool.
    (a, b) is chosen on the chronological inner TUNING queries from the inner (prefix-trained) selected-epoch logits of
    both models, so Val does not influence the choice. Blends are applied per seed; event metrics are averaged over
    seeds before user-paired bootstraps (1,000 draws, seed 0), as the pipeline does.

Reads saved scores only. No training, no selection for the final pipeline, never opens Test files.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.coreview_blend_oracle --phase register \
        --out results/analysis/20261105-coreview-blend-oracle
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.coreview_blend_oracle --phase reproduce \
        --out results/analysis/20261105-coreview-blend-oracle
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.coreview_blend_oracle --phase run \
        --out results/analysis/20261105-coreview-blend-oracle
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.utils.io import REPO_ROOT, git_hash

LABEL = "Val only, exploratory; regenerated pipeline"
PARENT = REPO_ROOT / "results/local_experiments/20261103-full-promising-ranking"
FACTORIAL = REPO_ROOT / "results/local_experiments/20261104-objective-coreview"
PUBLISHED = {"parent": REPO_ROOT / "results/full_data_replication/20261103-promising/metrics.csv",
             "factorial": REPO_ROOT / "results/full_data_replication/20261104-objective-coreview/metrics.csv"}
PARENT_METHODS = ("bce_text", "pairwise_text", "pairwise_coreview", "pairwise_combined")
SEEDS = (0, 1, 2)
GRID = tuple(round(i / 6, 10) for i in range(7))  # 7 x 7 = 49 (a, b) points
PAIRS = (("bce_text", "pairwise_text"), ("bce_coreview", "pairwise_text"))
N_BOOT, BOOT_SEED = 1000, 0
CHUNK = 4096
HYPOTHESIS = ("Pre-registered: the candidate-level blend keeps BCE's warm quality and recovers part of pairwise's "
              "new-item advantage, beating each component overall by a small margin; the target-conditioned oracle "
              "gap is much larger than the blend gain.")
SELECTION_RULE = ("Shared (a, b) over seeds maximising the seed-mean all-tuning-query user-macro nDCG@10 (the "
                  "pipeline's inner selector criterion) on the chronological inner tuning queries, from each model's "
                  "inner selected-epoch logits; exact ties -> earliest grid point in (a ascending, b ascending) order.")


# ----------------------------------------------------------------------------------------------- core functions
def target_ranks(scores: np.ndarray, cand: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pessimistic 1-based target rank among valid (id > 0) candidates; inf when the target is absent.

    Identical to ``steps_final.rank_top`` ranks (padding never outranks)."""
    scores, cand = np.asarray(scores), np.asarray(cand)
    out = np.full(len(cand), np.inf)
    for start in range(0, len(cand), CHUNK):
        c = cand[start:start + CHUNK]
        valid = c > 0
        s = np.where(valid, scores[start:start + CHUNK], -np.inf)
        match = (c == np.asarray(target[start:start + CHUNK])[:, None]) & valid
        found = match.any(1)
        t = s[np.arange(len(c)), match.argmax(1)]
        out[start:start + len(c)] = np.where(found, ((s >= t[:, None]) & valid).sum(1), np.inf)
    return out


def zscore(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-list z-standardisation over valid candidates (float64, ddof 0); constant lists -> 0; padding -> 0."""
    x = np.where(valid, np.asarray(logits, np.float64), 0.0)
    n = np.maximum(valid.sum(1, keepdims=True), 1)
    mean = x.sum(1, keepdims=True) / n
    var = np.where(valid, (x - mean) ** 2, 0.0).sum(1, keepdims=True) / n
    std = np.sqrt(var)
    ok = std > 1e-12
    z = np.where(ok, (x - mean) / np.where(ok, std, 1.0), 0.0)
    return np.where(valid, z, 0.0)


def blend_ranks(logit_b, logit_p, cand, target, new_cand, grid_points) -> np.ndarray:
    """Target ranks [len(grid_points), n] for score = (1-w) z_B + w z_P, w = b on new candidates, a otherwise."""
    cand = np.asarray(cand)
    out = np.full((len(grid_points), len(cand)), np.inf)
    for start in range(0, len(cand), CHUNK):
        sl = slice(start, start + CHUNK)
        c = cand[sl]
        valid = c > 0
        zb, zp = zscore(logit_b[sl], valid), zscore(logit_p[sl], valid)
        d = zp - zb
        nc = np.asarray(new_cand[sl])
        match = (c == np.asarray(target[sl])[:, None]) & valid
        found = match.any(1)
        col = match.argmax(1)
        rows = np.arange(len(c))
        for g, (a, b) in enumerate(grid_points):
            w = np.where(nc, b, a)
            s = np.where(valid, zb + w * d, -np.inf)
            t = s[rows, col]
            out[g, sl] = np.where(found, ((s >= t[:, None]) & valid).sum(1), np.inf)
    return out


def macro(values: np.ndarray, users: np.ndarray) -> float:
    return M.macro_micro(np.asarray(values, float), users)["macro"]


def event_metrics(rank_seeds: np.ndarray) -> dict[str, np.ndarray]:
    """Seed-mean per-event nDCG@10 / Recall@10 from ranks [seeds, n]."""
    r = np.atleast_2d(rank_seeds)
    return {"ndcg10": np.mean([M.ndcg_at(x, 10) for x in r], 0), "recall10": np.mean([M.recall_at(x, 10) for x in r], 0)}


def oracle_events(ev_b: dict, ev_p: dict, new_target: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Target-conditioned oracles: event choice by seed-mean nDCG@10 (ties -> BCE) and the new/warm group oracle."""
    pick_p = ev_p["ndcg10"] > ev_b["ndcg10"]
    return {"event_oracle": {m: np.where(pick_p, ev_p[m], ev_b[m]) for m in ev_b},
            "group_oracle": {m: np.where(new_target, ev_p[m], ev_b[m]) for m in ev_b}}


# ----------------------------------------------------------------------------------------------- data access
def owner(method: str, parent: Path, factorial: Path) -> Path:
    return factorial if method == "bce_coreview" else parent


def val_dir(method: str, seed: int, parent: Path, factorial: Path) -> Path:
    if method == "bce_coreview":
        return factorial / f"evaluation/seed{seed}"
    return parent / f"evaluation/{method}/seed{seed}"


def available(method: str, parent: Path, factorial: Path) -> bool:
    root = factorial if method == "bce_coreview" else parent
    path = root / "evaluation/completion.json"
    if not path.exists() or json.loads(path.read_text()).get("status") != "complete":
        return False
    return all((val_dir(method, s, parent, factorial) / "logits.npy").exists() for s in SEEDS)


def val_logits(method, seed, parent, factorial):
    return np.load(val_dir(method, seed, parent, factorial) / "logits.npy", mmap_mode="r")


def inner_logits(method, seed, parent, factorial):
    run = owner(method, parent, factorial) / f"runs/{method}/seed{seed}"
    epoch = int(json.loads((run / "metrics.json").read_text())["selected_epoch"])
    return np.load(run / f"inner/epoch{epoch}_logits.npy", mmap_mode="r"), run / f"inner/epoch{epoch}_ranks.npz", epoch


def load_context(parent: Path):
    """Val/tuning archives, item first-review times, split boundaries and target groups (no Test file is opened)."""
    from cascade.pipelines import full_data_ranking as P
    meta = json.loads((parent / "study.json").read_text())
    if meta["config"].get("test_predictions_loaded") or meta.get("test_predictions_loaded"):
        raise ValueError("study records Test exposure")
    data = P.data_for(meta)
    val = P.arrays(parent / "candidates/val")
    inst = data.split_instances("val")
    if not (np.array_equal(val["inst"], inst.inst) and np.array_equal(val["user"], inst.user)
            and np.array_equal(val["target"], inst.item)):
        raise ValueError("Val archive misaligned with instances")
    first = np.full(data.n_items + 1, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    split = data.split
    rt = P.arrays(parent / "candidates/ranker_train")
    tune = np.flatnonzero(np.load(parent / "split/tune.npy"))
    b_inst = data.split_instances("ranker_train")
    if not np.array_equal(rt["inst"], b_inst.inst):
        raise ValueError("ranker-training archive misaligned")
    ctx = {"meta": meta, "data": data, "val": val, "val_ts": inst.ts.to_numpy(), "first": first, "split": split,
           "users": np.asarray(val["user"]), "new_target": first[np.asarray(val["target"])] >= split["t1"],
           "tune": {"rows": tune, "cand": rt["cand"], "inst": np.asarray(rt["inst"])[tune], "target": np.asarray(rt["target"])[tune],
                    "users": np.asarray(rt["user"])[tune], "ts": b_inst.ts.to_numpy()[tune],
                    "retrieval_rank": np.asarray(rt["retrieval_rank"])[tune]}}
    ctx["tune"]["new_target"] = first[ctx["tune"]["target"]] >= split["t_ranker_train"]
    return ctx


def new_candidate_flag(first: np.ndarray, cand: np.ndarray, period_start: int) -> np.ndarray:
    """Serving-time-valid candidate flag: zero reviews before the period start (first review >= start).

    Padding (id 0) maps to False. Uses only each item's first-review time relative to a fixed past boundary."""
    c = np.asarray(cand)
    return (first[c] >= period_start) & (c > 0)


def group_masks(new_target: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": np.ones(len(new_target), bool), "new": new_target, "warm": ~new_target}


def replay_check(method, parent, factorial, val) -> tuple[np.ndarray, list[dict]]:
    """Recompute target ranks from saved Val logits and compare exactly with the saved val_ranks.npz."""
    ranks, proof = [], []
    for s in SEEDS:
        logits = val_logits(method, s, parent, factorial)
        r = target_ranks(logits, val["cand"], val["target"])
        with np.load(val_dir(method, s, parent, factorial) / "val_ranks.npz") as z:
            saved = z["rank"]
            ok = (np.array_equal(z["inst"], val["inst"]) and np.array_equal(z["user"], val["user"])
                  and np.array_equal(saved, pipeline_ranks(logits, val["cand"], val["target"])))
        if not ok:
            raise ValueError(f"saved Val logits do not replay saved ranks: {method} seed{s}")
        proof.append({"method": method, "seed": s, "rank_replay_exact": True,
                      "padding_convention_differing_events": int((saved != r).sum())})
        ranks.append(r)
    return np.stack(ranks), proof


def pipeline_ranks(scores, cand, target) -> np.ndarray:
    """The pipeline's own ``ranks_and_top10`` ranks (padding not masked), chunked."""
    from cascade.pipelines.local_evidence_export import ranks_and_top10
    out = np.empty(len(cand))
    for start in range(0, len(cand), CHUNK):
        sl = slice(start, start + CHUNK)
        out[sl] = ranks_and_top10(np.asarray(scores[sl]), np.asarray(cand[sl]), np.asarray(target[sl]))[0]
    return out


# ----------------------------------------------------------------------------------------------- phases
def register(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"{out} exists; registrations are never overwritten")
    out.mkdir(parents=True)
    cfg = {"scope": LABEL, "registered_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "hypothesis": HYPOTHESIS, "parent_study": str(PARENT.relative_to(REPO_ROOT)),
           "factorial_study": str(FACTORIAL.relative_to(REPO_ROOT)), "pairs": [list(p) for p in PAIRS],
           "seeds": list(SEEDS), "grid_a": list(GRID), "grid_b": list(GRID), "grid_points": len(GRID) ** 2,
           "selection_rule": SELECTION_RULE,
           "selection_population": "chronological inner tuning queries (split/tune.npy) of the ranker-training "
                                   "period; inner selected-epoch logits of both models (prefix-trained); Val untouched",
           "new_candidate_flag": "first review >= period start (T1 for Val queries, T0=t_ranker_train for tuning "
                                 "queries); equals zero reviews before the period start; padding excluded",
           "target_groups": "new_in_val = target first review >= T1 (pipeline groups_for); warm otherwise",
           "standardisation": "per list, per model, per seed over valid candidates; float64; ddof 0; constant -> 0",
           "ranks": "pessimistic ties, padding excluded (steps_final.rank_top convention)",
           "oracle": "event oracle by seed-mean nDCG@10 (ties BCE) and new->PW / warm->BCE group oracle; "
                     "target-conditioned upper bounds only",
           "metrics": ["ndcg10", "recall10"], "groups": ["all", "new", "warm"],
           "aggregation": "seed-mean event metrics, then user macro over the full population (misses = 0)",
           "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "user, paired"},
           "val_grid_surface": "computed for description only after the tuning choice; never used for selection",
           "rules": "Val only; no training; no Test file opened; selects nothing for the final pipeline"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_hash.txt").write_text(git_hash() + "\n")
    (out / "seed").write_text(f"{BOOT_SEED}\n")
    print(f"registered {out}")


def _published(kind: str) -> pd.DataFrame:
    return pd.read_csv(PUBLISHED[kind])


def reproduce(out: Path, parent: Path, factorial: Path, ctx=None, only=None, name="reproduction_check") -> pd.DataFrame:
    """Recompute the regenerated pipeline's Val nDCG@10/Recall@10 from saved scores; compare with publications."""
    if not (out / "config.yaml").exists():
        raise FileNotFoundError("register first")
    dest = out / f"{name}.csv"
    if dest.exists():
        raise FileExistsError(dest)
    ctx = ctx or load_context(parent)
    val, users, masks = ctx["val"], ctx["users"], group_masks(ctx["new_target"])
    pub = {"parent": _published("parent"), "factorial": _published("factorial")}
    gname = {"all": "all", "new": "new_in_val", "warm": "warm_in_val"}
    rows, proof = [], []
    methods = [("retrieval", None)] + [(m, "parent") for m in PARENT_METHODS] + [("bce_coreview", "factorial")]
    for method, kind in methods:
        if only is not None and method not in only:
            continue
        if method == "retrieval":
            r = target_ranks(val["score"], val["cand"], val["target"])
            if not np.array_equal(r, np.asarray(val["retrieval_rank"])):
                raise ValueError("retrieval replay differs")
            seeds = r[None]
        else:
            if not available(method, parent, factorial):
                rows.append({"method": method, "status": "not_available"})
                continue
            seeds, p = replay_check(method, parent, factorial, val)
            proof += p
        for g, mask in masks.items():
            for metric, fn in (("ndcg10", lambda x: M.ndcg_at(x, 10)), ("recall10", lambda x: M.recall_at(x, 10))):
                vals = [macro(fn(x)[mask], users[mask]) for x in seeds]
                ref = pub["factorial" if method == "bce_coreview" else "parent"]
                sel = ref[(ref.method == method) & (ref.group == gname[g]) & (ref.metric == metric)]
                published = float(sel.value.iloc[0]) if len(sel) else np.nan
                row = {"method": method, "status": "computed", "group": g, "metric": metric,
                       "events": int(mask.sum()), "users": int(np.unique(users[mask]).size),
                       "regenerated": float(np.mean(vals)), "regenerated_seed_sd": float(np.std(vals)),
                       "published": published, "difference": float(np.mean(vals)) - published}
                for s, v in enumerate(vals):
                    row[f"seed{s}"] = v
                    col = f"seed{s}"
                    row[f"published_seed{s}"] = float(sel[col].iloc[0]) if len(sel) and col in sel and pd.notna(sel[col].iloc[0]) else np.nan
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(dest, index=False)
    (out / f"{name}_replay_proof.json").write_text(json.dumps(proof, indent=1))
    print(frame.to_string())
    return frame


def _ci(users, mask, a, b) -> dict:
    return paired_bootstrap(users[mask], a[mask], b[mask], n_resamples=N_BOOT, seed=BOOT_SEED)


def tuning_grid(ctx, bce, pw, parent, factorial):
    """Seed-mean tuning-query user-macro metrics over the 49-point grid and the registered choice (no Val access)."""
    pair = f"{bce}+{pw}"
    first, split, tune = ctx["first"], ctx["split"], ctx["tune"]
    grid_points = [(a, b) for a in GRID for b in GRID]
    tune_cand = np.asarray(tune["cand"][tune["rows"]])
    new_tune = new_candidate_flag(first, tune_cand, split["t_ranker_train"])
    surf_rows, check_rows, tb, tp = [], [], [], []
    for s in SEEDS:
        lb, rb_path, eb = inner_logits(bce, s, parent, factorial)
        lp, rp_path, ep = inner_logits(pw, s, parent, factorial)
        for lg, path, name, e in ((lb, rb_path, bce, eb), (lp, rp_path, pw, ep)):
            r = target_ranks(lg, tune_cand, tune["target"])
            with np.load(path) as z:
                if (not np.array_equal(z["rank"], pipeline_ranks(lg, tune_cand, tune["target"]))
                        or not np.array_equal(z["inst"], tune["inst"])):
                    raise ValueError(f"inner logits do not replay saved inner ranks: {name} seed{s}")
                pad = int((z["rank"] != r).sum())
            check_rows.append({"pair": pair, "check": f"inner_replay_exact_{name}_seed{s}_epoch{e}", "value": 1})
            check_rows.append({"pair": pair, "check": f"inner_padding_convention_diff_{name}_seed{s}", "value": pad})
        tb.append(blend_ranks(lb, lp, tune_cand, tune["target"], new_tune, grid_points))
        tp.append((eb, ep))
    tb = np.stack(tb)  # [seeds, grid, n]
    for s in SEEDS:
        check_rows.append({"pair": pair, "check": f"tune_endpoint00_vs_{bce}_seed{s}_differing_events",
                           "value": int((tb[s, 0] != target_ranks(inner_logits(bce, s, parent, factorial)[0], tune_cand, tune["target"])).sum())})
    nt, tu = tune["new_target"], tune["users"]
    for g_idx, (a, b) in enumerate(grid_points):
        e = event_metrics(tb[:, g_idx])
        row = {"pair": pair, "population": "tuning", "a": a, "b": b, "tune_ndcg10": macro(e["ndcg10"], tu),
               "tune_recall10": macro(e["recall10"], tu), "tune_new_ndcg10": macro(e["ndcg10"][nt], tu[nt]),
               "tune_warm_ndcg10": macro(e["ndcg10"][~nt], tu[~nt])}
        for s in SEEDS:
            row[f"tune_ndcg10_seed{s}"] = macro(M.ndcg_at(tb[s, g_idx], 10), tu)
        surf_rows.append(row)
    score = np.asarray([r["tune_ndcg10"] for r in surf_rows])
    chosen = int(np.argmax(score))  # first maximum = earliest grid point
    per_seed = {s: grid_points[int(np.argmax([r[f"tune_ndcg10_seed{s}"] for r in surf_rows]))] for s in SEEDS}
    choice = {"pair": pair, "status": "chosen_on_tuning", "a": grid_points[chosen][0], "b": grid_points[chosen][1],
              "tune_ndcg10": float(score[chosen]), "tune_ndcg10_bce_endpoint": float(score[0]),
              "tune_ndcg10_pw_endpoint": float(score[-1]), "tune_queries": int(len(tu)),
              "tune_new_target_events": int(nt.sum()), "tune_new_candidate_slots": int(new_tune.sum()),
              "inner_epochs_bce": [t[0] for t in tp], "inner_epochs_pw": [t[1] for t in tp],
              **{f"per_seed_optimum_seed{s}": list(per_seed[s]) for s in SEEDS}}
    return surf_rows, choice, check_rows


def choose(out: Path, parent: Path, factorial: Path, pair_index: int) -> None:
    """Freeze (a, b) from tuning queries into config.yaml while no Val score of either model exists."""
    cfg_path = out / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    bce, pw = PAIRS[pair_index]
    tag = f"{bce}__{pw}"
    if tag in cfg.get("frozen_choice", {}):
        raise FileExistsError(f"choice for {tag} already frozen")
    val_exists = {m: any(val_dir(m, s, parent, factorial).joinpath("logits.npy").exists() for s in SEEDS) for m in (bce, pw)}
    for m in (bce, pw):
        for s in SEEDS:
            run_dir = owner(m, parent, factorial) / f"runs/{m}/seed{s}"
            rec = json.loads((run_dir / "metrics.json").read_text())
            if rec.get("status") != "fitted":
                raise ValueError(f"{m} seed{s} not fitted")
    ctx = load_context(parent)
    surf, choice, checks = tuning_grid(ctx, bce, pw, parent, factorial)
    pd.DataFrame(surf).to_csv(out / f"tuning_surface_at_freeze__{tag}.csv", index=False)
    pd.DataFrame(checks).to_csv(out / f"tuning_checks_at_freeze__{tag}.csv", index=False)
    cfg.setdefault("frozen_choice", {})[tag] = {
        "a": choice["a"], "b": choice["b"], "tune_ndcg10": choice["tune_ndcg10"],
        "tune_ndcg10_bce_endpoint": choice["tune_ndcg10_bce_endpoint"],
        "tune_ndcg10_pw_endpoint": choice["tune_ndcg10_pw_endpoint"],
        "inner_epochs_bce": choice["inner_epochs_bce"], "inner_epochs_pw": choice["inner_epochs_pw"],
        "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "val_scores_existed_at_freeze": val_exists}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(cfg["frozen_choice"][tag], indent=1))


def new_slot_diagnostic(logit_b, logit_p, cand, new_cand, points) -> list[dict]:
    """Per point: mean share of new candidates in the top 10 and mean within-list score SD of new / warm candidates."""
    cand = np.asarray(cand)
    acc = {p: np.zeros(3) for p in points}
    lists = 0
    for start in range(0, len(cand), CHUNK):
        sl = slice(start, start + CHUNK)
        c = cand[sl]; valid = c > 0; nc = np.asarray(new_cand[sl])
        zb, zp = zscore(logit_b[sl], valid), zscore(logit_p[sl], valid)
        for p in points:
            w = np.where(nc, p[1], p[0])
            sc = np.where(valid, zb + w * (zp - zb), -np.inf)
            top = np.argsort(-sc, axis=1, kind="stable")[:, :10]
            share = np.take_along_axis(nc, top, axis=1).mean(1)
            def sd(m):
                n = m.sum(1); x = np.where(m, sc, 0.0); mu = x.sum(1) / np.maximum(n, 1)
                v = np.where(m, (x - mu[:, None]) ** 2, 0).sum(1) / np.maximum(n, 1)
                return np.where(n > 1, np.sqrt(v), np.nan)
            acc[p] += [share.sum(), np.nansum(sd(nc & valid)), np.nansum(sd(~nc & valid))]
        lists += len(c)
    return [{"a": p[0], "b": p[1], "new_share_top10": v[0] / lists, "mean_sd_new_candidates": v[1] / lists,
             "mean_sd_warm_candidates": v[2] / lists} for p, v in acc.items()]


def diagnose(out: Path, parent: Path, factorial: Path, pair_index: int) -> None:
    """Mechanism diagnostic after the frozen run (descriptive; seed 0 only): new-candidate top-10 share and spread."""
    bce, pw = PAIRS[pair_index]; tag = f"{bce}__{pw}"
    dest = out / f"mechanism__{tag}.csv"
    if dest.exists():
        raise FileExistsError(dest)
    cfg = yaml.safe_load((out / "config.yaml").read_text()); fz = cfg["frozen_choice"][tag]
    surf = pd.read_csv(out / f"blend_grid_surface__{tag}.csv")
    v = surf[surf.population == "val_descriptive"].sort_values("val_ndcg10").iloc[-1]
    labelled = [("bce_endpoint", (0.0, 0.0)), ("pw_endpoint", (1.0, 1.0)), ("frozen_choice", (fz["a"], fz["b"])),
                ("val_descriptive_best", (float(v.a), float(v.b)))]
    points = list(dict.fromkeys(p for _, p in labelled))
    ctx = load_context(parent); first, split, tune = ctx["first"], ctx["split"], ctx["tune"]
    rows = []
    tc = np.asarray(tune["cand"][tune["rows"]])
    for r in new_slot_diagnostic(inner_logits(bce, 0, parent, factorial)[0], inner_logits(pw, 0, parent, factorial)[0],
                                 tc, new_candidate_flag(first, tc, split["t_ranker_train"]), points):
        rows.append({"population": "tuning_seed0", **r})
    val = ctx["val"]
    for r in new_slot_diagnostic(val_logits(bce, 0, parent, factorial), val_logits(pw, 0, parent, factorial),
                                 val["cand"], new_candidate_flag(first, val["cand"], split["t1"]), points):
        rows.append({"population": "val_seed0", **r})
    names = {}
    for name, p in labelled:
        names[p] = names[p] + "+" + name if p in names else name
    frame = pd.DataFrame(rows); frame["point"] = [names[(a, b)] for a, b in zip(frame.a, frame.b)]
    frame.to_csv(dest, index=False); print(frame.to_string())


def run(out: Path, parent: Path, factorial: Path, pair_index: int) -> None:
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    if cfg.get("hypothesis") != HYPOTHESIS:
        raise ValueError("registration does not match this module")
    pairs = (PAIRS[pair_index],)
    tag = "__".join(PAIRS[pair_index])
    if (out / f"blend_val__{tag}.csv").exists():
        raise FileExistsError("run outputs exist")
    ctx = load_context(parent)
    if not (out / "reproduction_check.csv").exists():
        reproduce(out, parent, factorial, ctx)
    if (available("bce_coreview", parent, factorial) and not (out / "reproduction_check_bce_coreview.csv").exists()
            and "bce_coreview" in PAIRS[pair_index]):
        frame = pd.read_csv(out / "reproduction_check.csv")
        if (frame.loc[frame.method == "bce_coreview", "status"] == "not_available").any():
            reproduce(out, parent, factorial, ctx, only={"bce_coreview"}, name="reproduction_check_bce_coreview")
    val, users, first, split = ctx["val"], ctx["users"], ctx["first"], ctx["split"]
    masks = group_masks(ctx["new_target"])
    tune = ctx["tune"]
    grid_points = [(a, b) for a in GRID for b in GRID]
    new_val = new_candidate_flag(first, val["cand"], split["t1"])
    disclosure = {"val_candidate_slots": int((np.asarray(val["cand"]) > 0).sum()),
                  "val_new_candidate_slots": int(new_val.sum()),
                  "val_lists_with_any_new_candidate": int(new_val.any(1).sum()),
                  "val_candidates_first_review_equal_query_ts": int(
                      ((first[np.asarray(val["cand"])] == ctx["val_ts"][:, None]) & (np.asarray(val["cand"]) > 0)).sum()),
                  "val_candidates_first_review_after_query_ts": int(
                      ((first[np.asarray(val["cand"])] > ctx["val_ts"][:, None]) & (np.asarray(val["cand"]) > 0)).sum()),
                  "val_candidates_first_review_equal_query_ts_that_are_target": int(
                      ((first[np.asarray(val["cand"])] == ctx["val_ts"][:, None])
                       & (np.asarray(val["cand"]) == np.asarray(val["target"])[:, None])).sum()),
                  "val_padding_slots": int((np.asarray(val["cand"]) == 0).sum()),
                  "flag_note": "new = zero reviews before T1 (decidable at T1 for every item); pool admission "
                               "(first review <= query ts, full-history exclusion) is the pipeline's unchanged rule"}
    tune_cand = np.asarray(tune["cand"][tune["rows"]])
    new_tune = new_candidate_flag(first, tune_cand, split["t_ranker_train"])
    disclosure.update({"tune_queries": int(len(tune["rows"])), "tune_new_candidate_slots": int(new_tune.sum()),
                       "tune_candidates_first_review_after_query_ts": int(
                           ((first[tune_cand] > tune["ts"][:, None]) & (tune_cand > 0)).sum())})
    if disclosure["val_candidates_first_review_after_query_ts"] or disclosure["tune_candidates_first_review_after_query_ts"]:
        raise ValueError("candidate pool contains unreleased items")
    cache: dict[str, np.ndarray] = {}
    oracle_rows, oracle_ci, surf_rows, val_rows, ci_rows, choice_rows, check_rows = [], [], [], [], [], [], []
    for method in sorted({m for p in pairs for m in p}):
        if available(method, parent, factorial):
            cache[method], proof = replay_check(method, parent, factorial, val)
            for p in proof:
                check_rows.append({"pair": "", "check": f"val_replay_exact_{method}_seed{p['seed']}", "value": 1})
                check_rows.append({"pair": "", "check": f"val_padding_convention_diff_{method}_seed{p['seed']}",
                                   "value": p["padding_convention_differing_events"]})
    ev = {m: event_metrics(r) for m, r in cache.items()}
    saved_ranks = {}
    for bce, pw in pairs:
        pair = f"{bce}+{pw}"
        if bce not in cache or pw not in cache:
            choice_rows.append({"pair": pair, "status": "not_available"})
            continue
        # ---------------- (a) oracles
        orc = oracle_events(ev[bce], ev[pw], ctx["new_target"])
        systems = {bce: ev[bce], pw: ev[pw], **orc}
        for g, mask in masks.items():
            for name, e in systems.items():
                oracle_rows.append({"pair": pair, "system": name, "group": g, "events": int(mask.sum()),
                                    "users": int(np.unique(users[mask]).size),
                                    **{m: macro(e[m][mask], users[mask]) for m in ("ndcg10", "recall10")}})
            for name in orc:
                for ref in (bce, pw):
                    for m in ("ndcg10", "recall10"):
                        oracle_ci.append({"pair": pair, "group": g, "metric": m, "reference": ref, "candidate": name,
                                          **_ci(users, mask, systems[ref][m], orc[name][m])})
        pick = ev[pw]["ndcg10"] > ev[bce]["ndcg10"]
        check_rows.append({"pair": pair, "check": "event_oracle_pw_events", "value": int(pick.sum())})
        check_rows.append({"pair": pair, "check": "event_oracle_pw_events_new",
                           "value": int((pick & ctx["new_target"]).sum())})
        # ---------------- (b) tuning-query choice, frozen before Val scores existed
        t_surf, t_choice, t_checks = tuning_grid(ctx, bce, pw, parent, factorial)
        frozen = cfg.get("frozen_choice", {}).get(tag)
        if frozen is None:
            raise ValueError("freeze (a, b) with --phase choose before Val scoring")
        if (t_choice["a"], t_choice["b"]) != (frozen["a"], frozen["b"]):
            raise ValueError("recomputed tuning choice differs from the frozen choice")
        surf_rows += t_surf; check_rows += t_checks
        choice_rows.append({**t_choice, "frozen_at": frozen["frozen_at"],
                            "val_scores_existed_at_freeze": frozen["val_scores_existed_at_freeze"]})
        a_star, b_star = frozen["a"], frozen["b"]
        chosen = grid_points.index((a_star, b_star))
        # ---------------- Val: full surface (descriptive) and chosen point
        vb = np.stack([blend_ranks(val_logits(bce, s, parent, factorial), val_logits(pw, s, parent, factorial),
                                   val["cand"], val["target"], new_val, grid_points) for s in SEEDS])
        for s in SEEDS:  # endpoint alignment: (0,0) must equal BCE ranks, (1,1) PW ranks
            check_rows.append({"pair": pair, "check": f"val_endpoint00_vs_{bce}_seed{s}_differing_events",
                               "value": int((vb[s, 0] != cache[bce][s]).sum())})
            check_rows.append({"pair": pair, "check": f"val_endpoint11_vs_{pw}_seed{s}_differing_events",
                               "value": int((vb[s, -1] != cache[pw][s]).sum())})
        for g_idx, (a, b) in enumerate(grid_points):
            e = event_metrics(vb[:, g_idx])
            surf_rows.append({"pair": pair, "a": a, "b": b, "population": "val_descriptive",
                              "val_ndcg10": macro(e["ndcg10"], users), "val_recall10": macro(e["recall10"], users),
                              "val_new_ndcg10": macro(e["ndcg10"][masks["new"]], users[masks["new"]]),
                              "val_warm_ndcg10": macro(e["ndcg10"][masks["warm"]], users[masks["warm"]])})
        blend = event_metrics(vb[:, chosen])
        saved_ranks[pair.replace("+", "__")] = vb[:, chosen].astype(np.float32)
        for g, mask in masks.items():
            for name, e in ((bce, ev[bce]), (pw, ev[pw]), ("blend", blend), ("event_oracle", orc["event_oracle"]),
                            ("group_oracle", orc["group_oracle"])):
                row = {"pair": pair, "system": name, "group": g, "a": a_star, "b": b_star, "events": int(mask.sum()),
                       "users": int(np.unique(users[mask]).size)}
                for m in ("ndcg10", "recall10"):
                    row[m] = macro(e[m][mask], users[mask])
                    if name == "blend":
                        sv = [macro((M.ndcg_at if m == "ndcg10" else M.recall_at)(vb[s, chosen], 10)[mask], users[mask]) for s in SEEDS]
                    elif name in (bce, pw):
                        sv = [macro((M.ndcg_at if m == "ndcg10" else M.recall_at)(cache[name][s], 10)[mask], users[mask]) for s in SEEDS]
                    else:
                        sv = []
                    if sv:
                        row[m + "_seed_sd"] = float(np.std(sv))
                val_rows.append(row)
            for ref in (bce, pw):
                for m in ("ndcg10", "recall10"):
                    ci_rows.append({"pair": pair, "group": g, "metric": m, "reference": ref, "candidate": "blend",
                                    "a": a_star, "b": b_star, **_ci(users, mask, ev[ref][m], blend[m])})
            for m in ("ndcg10", "recall10"):
                ci_rows.append({"pair": pair, "group": g, "metric": m, "reference": "blend", "candidate": "event_oracle",
                                **_ci(users, mask, blend[m], orc["event_oracle"][m])})
                ci_rows.append({"pair": pair, "group": g, "metric": m, "reference": "blend", "candidate": "group_oracle",
                                **_ci(users, mask, blend[m], orc["group_oracle"][m])})
        print(f"{pair}: chosen a={a_star:.4f} b={b_star:.4f}", flush=True)
    pd.DataFrame(oracle_rows).to_csv(out / f"oracle_metrics__{tag}.csv", index=False)
    pd.DataFrame(oracle_ci).to_csv(out / f"oracle_paired_differences__{tag}.csv", index=False)
    pd.DataFrame(surf_rows).to_csv(out / f"blend_grid_surface__{tag}.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(out / f"blend_choice__{tag}.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / f"blend_val__{tag}.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(out / f"blend_paired_differences__{tag}.csv", index=False)
    pd.DataFrame(check_rows).to_csv(out / f"checks__{tag}.csv", index=False)
    if saved_ranks:
        np.savez_compressed(out / f"blend_val_ranks__{tag}.npz", inst=np.asarray(val["inst"]), user=users, **saved_ranks)
    (out / f"disclosure__{tag}.json").write_text(json.dumps({"label": LABEL, **disclosure}, indent=1))
    print("done", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("register", "choose", "reproduce", "run", "diagnose"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--parent", type=Path, default=PARENT)
    ap.add_argument("--factorial", type=Path, default=FACTORIAL)
    ap.add_argument("--pair", type=int, choices=range(len(PAIRS)), default=0,
                    help="0: bce_text+pairwise_text, 1: bce_coreview+pairwise_text")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if args.phase == "register":
        return register(out)
    if args.phase == "diagnose":
        return diagnose(out, args.parent, args.factorial, args.pair)
    if args.phase == "choose":
        return choose(out, args.parent, args.factorial, args.pair)
    if args.phase == "reproduce":
        reproduce(out, args.parent, args.factorial)
        return None
    return run(out, args.parent, args.factorial, args.pair)


if __name__ == "__main__":
    main()
