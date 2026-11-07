"""U-shaped logit-vs-retrieval-rank response split by candidate popularity group (Val only, exploratory).

Question: is the forced-list ranker's U-shaped within-list logit over retrieval rank explained by a preference for
unpopular products, by the retrieval-score shortcut, or both?

Hypothesis (written before running): policy arm A (all lists, forced positives appended, score on) shows both
effects -- within each candidate popularity group its centred logit rises at low retrieval ranks (bottom decile above
the middle), and tail/new candidates score higher than head candidates at the same rank decile; arm C (natural lists
only, score on) shows neither.

Frozen policy-study checkpoints (arms A and C, seeds 0-2) score a fixed uniform random subsample of natural Val
top-200 lists (no forced insertions; the retrieved list is the same whether or not the target was retrieved). The
feature builder is the policy study's frozen full-training contract, checked against
``configs/ranker/policy_candidate_reference_rebuilt.json`` (candidate hashes and score statistics). Pairwise
features do not depend on the other candidates, so each candidate is scored once per checkpoint.

Candidate popularity group (Val-valid; uses only interactions before T1):
* new-in-period: candidate has no interaction before T1 (equivalently first interaction >= T1, released during Val
  before the event; checked);
* head / tail: top 20% / remaining 80% of items with >= 1 interaction before T1, ranked by that count, via the
  project's ``cascade.data.split.item_segments`` with the cutoff set to T1 instead of T2.
A sensitivity with the project's standard segments (counts before T2, ``item_segments.parquet``) is also written.

Inference only; trains nothing; never opens Test files.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.ushape_popularity \
        --out results/analysis/20261105-ushape-popularity --n-lists 30000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from cascade.analysis.plot_findings import AQUA, BLUE, GREY, INK_2, ORANGE, _save, _style
from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.data.split import item_segments
from cascade.eval import metrics as M
from cascade.eval.ranker import candidate_ranks
from cascade.eval.segments import new_in_period
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_policy_study import digest, frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, git_hash

LABEL = "Val only, exploratory"
CAND_DIR = "data/processed/candidates/two_tower_text_seed0"
CAND_REF = "configs/ranker/policy_candidate_reference_rebuilt.json"
REF_RANKER_CFG = "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml"
POLICY = {0: "results/ranker_policy/policy-pilot-20261029",
          1: "results/ranker_policy/policy-replication-20261029",
          2: "results/ranker_policy/policy-replication-20261029"}
ARMS = {"A": "A: all lists (forced positives), score on", "C": "C: natural lists only, score on"}
GROUPS = ("head", "tail", "new")
K1, N_DECILES = 200, 10
HYPOTHESIS = {
    "statement": "Arm A shows both effects: within each candidate popularity group its centred logit rises at low "
                 "retrieval ranks, and tail/new candidates score higher than head at the same rank decile; arm C "
                 "shows neither.",
    "operationalisation": {
        "low_rank_rise": "per group, mean centred logit in decile 10 (ranks 181-200) minus decile 6 (ranks 101-120) "
                         "> 0 in every seed",
        "unpopular_preference": "per non-head group, mean over deciles of (group - head) centred logit > 0 in every "
                                "seed, and positive in >= 8 of 10 deciles on the seed mean",
    },
}


def candidate_groups(data: ProcessedData) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-item labels (index = item id; 0 = padding) for the Val-valid and the project (pre-T2) definitions."""
    t1 = int(data.split["t1"])
    pre = data.interactions(["retriever_train", "ranker_train"]).rename(columns={"ts": "timestamp"})
    if len(pre) and pre["timestamp"].max() >= t1:
        raise ValueError("pre-T1 interactions include Val events")
    seg = item_segments(pre, data.n_items, t1).sort_values("item")  # cutoff T1; 'cold' = no pre-T1 interaction
    items = data.items.set_index("item").loc[seg["item"].to_numpy()]
    new = new_in_period(items["first_ts"].to_numpy(), "val", data.split)
    if not np.array_equal(new, seg["segment"].to_numpy() == "cold"):
        raise ValueError("no pre-T1 interaction must equal first interaction >= T1")
    lab = np.full(data.n_items + 1, "", dtype=object)
    lab[seg["item"].to_numpy()] = np.where(new, "new", seg["segment"].to_numpy())
    # Project segments (counts before T2) + new-in-period, as in cascade.eval.ranker_meso.group_labels.
    proj = np.full(data.n_items + 1, "", dtype=object)
    s2 = items["segment"].astype(str).to_numpy()
    proj[seg["item"].to_numpy()] = np.where(new, "new", np.where(s2 == "head", "head", "tail"))
    info = {"n_items": int(data.n_items), "val_valid_counts": pd.Series(lab[1:]).value_counts().to_dict(),
            "project_t2_counts": pd.Series(proj[1:]).value_counts().to_dict(),
            "head_min_pre_t1_count": int(seg.loc[seg["segment"] == "head", "cnt_before_t2"].min()),
            "note": "item_segments column 'cnt_before_t2' holds the count before the cutoff passed (here T1)"}
    return lab, proj, info


def load_arm(arm: str, seed: int, builder: RankerFeatureBuilder):
    run = REPO_ROOT / POLICY[seed] / f"arm_{arm}_seed{seed}"
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if cfg["policy_arm"] != arm or int(cfg["seed"]) != seed or cfg["retrieval_score_input"] != "enabled":
        raise ValueError(f"unexpected checkpoint config: {run}")
    fs = cfg["run"]["feature_spec"]
    spec = RankerFeatureSpec(**{**fs, "feature_groups": tuple(fs["feature_groups"])})
    if dict(spec.cardinalities) != builder.item_cardinalities or not np.allclose(
            [cfg["run"]["score_mean"], cfg["run"]["score_std"]], [builder.score_mean, builder.score_std], rtol=0, atol=1e-8):
        raise ValueError(f"frozen feature contract differs: {run}")
    model = create_ranker(spec, cfg)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
    return model.eval(), run


def score_lists(models: dict, builder, lists: dict, batch_pairs: int) -> dict[str, np.ndarray]:
    """Logits [n_lists, K1] per checkpoint (NaN at padding); features built once per batch."""
    n, width = lists["cand"].shape
    out = {k: np.full((n, width), np.nan, dtype=np.float32) for k in models}
    batch_lists = max(1, batch_pairs // width)
    t0 = time.time()
    with torch.inference_mode():
        for bi, start in enumerate(range(0, n, batch_lists)):
            rr = np.arange(start, min(n, start + batch_lists))
            valid = lists["cand"][rr] > 0
            pr = np.repeat(rr, width)[valid.ravel()]
            pc = np.tile(np.arange(width), len(rr))[valid.ravel()]
            batch = {k: torch.as_tensor(v) for k, v in builder.build(lists, pr, pc).items()}
            for name, model in models.items():
                out[name][pr, pc] = model(batch).float().numpy()
            if bi % 100 == 0:
                print(f"  batch {bi}/{-(-n // batch_lists)}  {time.time() - t0:.0f}s", flush=True)
    return out


def decile_table(logits: np.ndarray, groups: np.ndarray, valid: np.ndarray, ret_rank: np.ndarray) -> pd.DataFrame:
    """Per (decile, group): candidate share within decile, mean within-list-centred logit, mean retrieval rank."""
    centred = logits - np.nanmean(np.where(valid, logits, np.nan), axis=1, keepdims=True)
    dec = (ret_rank - 1) // (K1 // N_DECILES)
    rows = []
    for d in range(N_DECILES):
        in_d = valid & (dec == d)
        total = in_d.sum()
        for g in ("all",) + GROUPS:
            m = in_d if g == "all" else in_d & (groups == g)
            rows.append({"decile": d + 1, "ranks": f"{d * 20 + 1}-{d * 20 + 20}", "group": g,
                         "n_candidates": int(m.sum()), "share": float(m.sum() / total) if total else np.nan,
                         "mean_centred_logit": float(centred[m].mean()) if m.any() else np.nan,
                         "mean_retrieval_rank": float(ret_rank[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def hypothesis_checks(per_seed: pd.DataFrame) -> dict:
    out = {}
    for arm in ARMS:
        d = per_seed[per_seed["arm"] == arm]
        piv = d.pivot_table(index=["seed", "group"], columns="decile", values="mean_centred_logit")
        rise = {g: (piv.xs(g, level="group")[10] - piv.xs(g, level="group")[6]).tolist() for g in GROUPS}
        contrast = {}
        for g in ("tail", "new"):
            diff = piv.xs(g, level="group") - piv.xs("head", level="group")  # seeds x deciles
            contrast[g] = {"mean_over_deciles_per_seed": diff.mean(axis=1).tolist(),
                           "deciles_positive_on_seed_mean": int((diff.mean(axis=0) > 0).sum()),
                           "per_decile_seed_mean": diff.mean(axis=0).tolist()}
        out[arm] = {
            "low_rank_rise_d10_minus_d6_per_seed": rise,
            "low_rank_rise_holds_all_groups": bool(all(min(v) > 0 for v in rise.values())),
            "unpopular_minus_head": contrast,
            "unpopular_preference_holds": bool(all(min(c["mean_over_deciles_per_seed"]) > 0 and
                                                   c["deciles_positive_on_seed_mean"] >= 8 for c in contrast.values())),
        }
    a, c = out["A"], out["C"]
    out["hypothesis_held"] = bool(a["low_rank_rise_holds_all_groups"] and a["unpopular_preference_holds"] and
                                  not c["low_rank_rise_holds_all_groups"] and not c["unpopular_preference_holds"])
    return out


def composition_decomposition(seed_mean: pd.DataFrame) -> pd.DataFrame:
    """Overall decile curve vs (i) group shares fixed at their overall values (within-group rank effect only) and
    (ii) group means fixed at their all-decile values (composition effect only)."""
    rows = []
    for arm, d in seed_mean.groupby("arm"):
        g = d[d["group"].isin(GROUPS)]
        share = g.pivot(index="decile", columns="group", values="share_mean")[list(GROUPS)]
        mean = g.pivot(index="decile", columns="group", values="mean_centred_logit_mean")[list(GROUPS)]
        overall_share = share.mean(axis=0)  # every decile has the same number of candidates
        overall_mean = (share * mean).sum(axis=0) / share.sum(axis=0)
        for dec in share.index:
            rows.append({"arm": arm, "decile": dec,
                         "observed": float((share.loc[dec] * mean.loc[dec]).sum()),
                         "within_group_rank_effect_only": float((overall_share * mean.loc[dec]).sum()),
                         "composition_effect_only": float((share.loc[dec] * overall_mean).sum())})
    return pd.DataFrame(rows)


def plot(seed_mean: pd.DataFrame, out: Path, note: str) -> None:
    _style()
    colours = {"head": BLUE, "tail": ORANGE, "new": AQUA}
    names = {"head": "Popular (head)", "tail": "Tail", "new": "New in Val"}
    deciles = np.arange(1, N_DECILES + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, arm in zip(axes, ARMS):
        d = seed_mean[seed_mean["arm"] == arm]
        for g in GROUPS:
            s = d[d["group"] == g].sort_values("decile")
            ax.errorbar(deciles, s["mean_centred_logit_mean"], yerr=s["mean_centred_logit_sd"], color=colours[g],
                        marker="o", markersize=4, capsize=2, label=names[g])
        s = d[d["group"] == "all"].sort_values("decile")
        ax.plot(deciles, s["mean_centred_logit_mean"], color=GREY, linestyle="--", linewidth=1.2, label="All candidates")
        ax.axhline(0, color=GREY, linewidth=0.8)
        ax.set(xticks=deciles, xlabel="Retrieval-rank decile in the top-200 list (1 = ranks 1-20)")
        ax.set_title(ARMS[arm])
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Mean ranker logit, centred within list")
    fig.suptitle("Ranker logit by retrieval rank and candidate popularity (Val only, exploratory)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold", y=1.03)
    table = seed_mean[["arm", "decile", "ranks", "group", "share_mean", "share_sd", "mean_centred_logit_mean",
                       "mean_centred_logit_sd", "n_seeds"]]
    _save(fig, out, "fig_ushape_popularity", table, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-lists", type=int, default=30000, help="uniform random Val lists (seed 0); 0 = all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batch-pairs", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    checkpoints = {f"{a}/seed{s}": f"{POLICY[s]}/arm_{a}_seed{s}/model.pt" for a in ARMS for s in args.seeds}
    cfg = {"scope": LABEL, **vars(args), "list_sample_seed": 0, "seed": 0, "candidate_dir": CAND_DIR, "k1": K1,
           "candidate_reference": CAND_REF, "reference_ranker_config": REF_RANKER_CFG, "checkpoints": checkpoints,
           "lists": "natural Val top-200 (load_candidates(force=False)); uniform over all Val events, incl. lists "
                    "whose target was not retrieved",
           "centring": "logit minus the mean logit of the list's valid candidates",
           "candidate_group_definition": {
               "new": "no interaction before T1 (= first interaction >= T1)",
               "head": "top 20% of items with >=1 interaction before T1, by pre-T1 count (item_segments, cutoff T1)",
               "tail": "remaining 80% of items with >=1 pre-T1 interaction",
               "sensitivity": "project segments: counts before T2 (item_segments.parquet) + new-in-period"},
           "hypothesis": HYPOTHESIS, "test_files_opened": False}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_hash.txt").write_text(git_hash() + "\n")
    (out / "seed").write_text("0\n")

    cand_dir = REPO_ROOT / CAND_DIR
    ref = json.loads((REPO_ROOT / CAND_REF).read_text())
    for name, key in (("ranker_train.npz", "ranker_train_sha256"), ("val.npz", "val_sha256"), ("meta.json", "meta_sha256")):
        if digest(cand_dir / name) != ref[key]:
            raise ValueError(f"candidate file differs from the policy-study reference: {name}")
    data = ProcessedData.load("configs/data.yaml")
    builder = frozen_feature_builder(data, load_candidates(cand_dir, "ranker_train"), REPO_ROOT / REF_RANKER_CFG,
                                     (ref["score_mean"], ref["score_std"]))
    full = load_candidates(cand_dir, "val", k1=K1, force=False)
    n_all = len(full["inst"])
    rows = np.sort(np.random.default_rng(0).choice(n_all, args.n_lists, replace=False)) if args.n_lists else np.arange(n_all)
    lists = {k: np.ascontiguousarray(v[rows]) for k, v in full.items()}
    del full
    table = data.instances.set_index("inst").loc[lists["inst"]]
    if not np.array_equal(table["user"].to_numpy(), lists["user"]) or not (table["period"] == 2).all():
        raise ValueError("Val list alignment failed")
    lab, proj, group_info = candidate_groups(data)
    valid = lists["cand"] > 0
    ret_rank = np.broadcast_to(np.arange(1, K1 + 1), valid.shape)
    print(f"Val lists: {len(rows)} of {n_all}; valid candidates {valid.mean():.4f}", flush=True)

    models, runs = {}, {}
    for a in ARMS:
        for s in args.seeds:
            models[f"{a}/seed{s}"], runs[f"{a}/seed{s}"] = load_arm(a, s, builder)
    logits = score_lists(models, builder, lists, args.batch_pairs)

    # Alignment: on naturally retrieved targets in the sample, ranks must reproduce the saved selected-checkpoint
    # K200 prefix ranks of the same checkpoint.
    hits = lists["target_pos"] >= 0
    checks = {}
    for name, lg in logits.items():
        ranks = candidate_ranks(np.where(valid, lg, -np.inf).astype(np.float32), lists)
        with np.load(runs[name] / "prefix_ranks.npz", allow_pickle=False) as z:
            k = list(z["candidate_counts"]).index(K1)
            saved = pd.Series(z["ranks"][k], index=z["inst"]).loc[lists["inst"]].to_numpy()
        checks[name] = {"n_hit_lists": int(hits.sum()),
                        "exact_rank_match_fraction": float(np.mean(saved[hits] == ranks[hits])),
                        "max_abs_rank_diff": float(np.max(np.abs(saved[hits] - ranks[hits]))) if hits.any() else 0.0,
                        "conditional_ndcg10_macro_recomputed": M.macro_micro(M.ndcg_at(ranks[hits], 10), lists["user"][hits])["macro"],
                        "conditional_ndcg10_macro_saved": M.macro_micro(M.ndcg_at(saved[hits], 10), lists["user"][hits])["macro"]}
        print(name, checks[name], flush=True)

    frames = []
    for definition, labels in (("val_valid_pre_t1", lab), ("project_pre_t2", proj)):
        groups = labels[lists["cand"]]
        for name, lg in logits.items():
            arm, seed = name.split("/seed")
            for subset, mask in (("all_lists", np.ones(len(rows), bool)), ("natural_hit_lists", hits)):
                t = decile_table(lg[mask], groups[mask], valid[mask], ret_rank[mask])
                t.insert(0, "subset", subset); t.insert(0, "definition", definition)
                t.insert(0, "seed", int(seed)); t.insert(0, "arm", arm)
                frames.append(t)
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed.to_csv(out / "deciles_per_seed.csv", index=False)
    seed_mean = (per_seed.groupby(["definition", "subset", "arm", "decile", "ranks", "group"], sort=False)
                 .agg(share_mean=("share", "mean"), share_sd=("share", "std"),
                      mean_centred_logit_mean=("mean_centred_logit", "mean"),
                      mean_centred_logit_sd=("mean_centred_logit", "std"),
                      mean_retrieval_rank=("mean_retrieval_rank", "mean"), n_seeds=("seed", "nunique"))
                 .reset_index())
    seed_mean.to_csv(out / "deciles_seed_mean.csv", index=False)
    primary = seed_mean[(seed_mean["definition"] == "val_valid_pre_t1") & (seed_mean["subset"] == "all_lists")]
    decomp = composition_decomposition(primary)
    decomp.to_csv(out / "composition_decomposition.csv", index=False)
    hyp = {f"{d}|{s}": hypothesis_checks(per_seed[(per_seed["definition"] == d) & (per_seed["subset"] == s)])
           for d in ("val_valid_pre_t1", "project_pre_t2") for s in ("all_lists", "natural_hit_lists")}
    results = {"scope": LABEL, "git_hash": git_hash(), "n_val_lists_total": int(n_all), "n_lists_used": int(len(rows)),
               "n_users": int(np.unique(lists["user"]).size), "n_natural_hit_lists": int(hits.sum()),
               "candidate_groups": group_info, "alignment_checks": checks, "hypothesis": HYPOTHESIS,
               "hypothesis_checks": hyp, "primary": "val_valid_pre_t1|all_lists", "test_files_opened": False}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    plot(primary, out, f"Val only, exploratory. {len(rows):,} uniform random natural Val top-200 lists (seed 0), "
         "policy-study MLP checkpoints, mean ± SD over seeds 0-2. Candidate group from interactions before T1: new = "
         f"none before T1; head/tail = top 20%/rest by pre-T1 count. Source: {args.out}.")
    print("done", flush=True)


if __name__ == "__main__":
    main()
