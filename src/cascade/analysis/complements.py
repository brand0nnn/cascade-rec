"""Similar vs complementary Val targets for the frozen retriever (validation only, exploratory).

Pre-registered hypothesis: the text-based retriever finds products that resemble the user's last product
but misses complements. Each Val instance is classified by its target against the LAST prior history item
(leaf category x store) and, separately, by whether the target's leaf category occurs anywhere in the last
50 prior items. Recall, rank-depth shares and user-bootstrap CIs are reported per class, split by item type
so popularity cannot explain the difference. Never touches Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.metrics import macro_micro, recall_at
from cascade.eval.ranker_meso import group_labels
from cascade.utils.io import git_hash, make_run_dir

LABEL = "validation only, exploratory"
RANK_BUCKETS = [(1, 200, "<=200"), (201, 1000, "201-1K"), (1001, 5000, "1K-5K"), (5001, 20000, "5K-20K"),
                (20001, np.inf, ">20K")]
DEEP = 5000


def _lut(data: ProcessedData, column: str) -> np.ndarray:
    """Item id -> metadata string ('' = missing); index 0 = padding."""
    out = np.full(data.n_items + 1, "", dtype=object)
    out[data.items["item"].to_numpy()] = data.items[column].fillna("").astype(str).to_numpy()
    return out


def classify(target: np.ndarray, histories: np.ndarray, leaf: np.ndarray, store: np.ndarray) -> pd.DataFrame:
    """Target vs last history item (leaf x store) and target leaf anywhere in the last-50 history."""
    last = histories[:, -1]
    if (last == 0).any():
        raise ValueError("empty history in Val instances")
    t_leaf, l_leaf, t_store, l_store = leaf[target], leaf[last], store[target], store[last]
    leaf_known = (t_leaf != "") & (l_leaf != "")
    store_known = (t_store != "") & (l_store != "")
    same_leaf, same_store = t_leaf == l_leaf, t_store == l_store
    last_class = np.select(
        [~leaf_known,
         ~store_known & same_leaf, ~store_known & ~same_leaf,
         same_leaf & same_store, same_leaf & ~same_store, ~same_leaf & same_store],
        ["leaf unknown", "same leaf / store unknown", "diff leaf / store unknown",
         "same leaf & same store", "same leaf, diff store", "diff leaf, same store"],
        default="diff leaf & diff store")
    in_hist = np.full(len(target), "unknown", dtype=object)
    for start in range(0, len(target), 8192):
        end = min(start + 8192, len(target))
        h_leaf = leaf[histories[start:end]]
        known = (t_leaf[start:end] != "") & (h_leaf != "").any(axis=1)
        match = (h_leaf == t_leaf[start:end, None]).any(axis=1) & (t_leaf[start:end] != "")
        in_hist[start:end] = np.where(known, np.where(match, "leaf in history", "leaf not in history"), "unknown")
    return pd.DataFrame({"last_class": last_class, "leaf_in_history": in_hist,
                         "same_leaf_last": np.where(leaf_known, np.where(same_leaf, "true", "false"), "unknown")})


def ci(values: np.ndarray, users: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    """95% user-bootstrap CI of the per-user macro mean (paired_bootstrap against zeros)."""
    r = paired_bootstrap(users, np.zeros(len(values)), values, n_resamples=resamples, seed=seed)
    return r["ci_low"], r["ci_high"]


def summarise(frame: pd.DataFrame, group_cols: list[str], n_total: int, resamples: int, seed: int,
              with_ci: bool = True) -> pd.DataFrame:
    """Per-group support, recall (macro/micro), rank-bucket shares, median rank and macro R@200 CI."""
    rows = []
    for key, g in frame.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ranks, users = g["rank"].to_numpy(), g["user"].to_numpy()
        row = dict(zip(group_cols, key))
        row.update(n=len(g), users=int(g["user"].nunique()), share=len(g) / n_total)
        for cutoff in (10, 200, 500):
            mm = macro_micro(recall_at(ranks, cutoff), users)
            row[f"R@{cutoff}_macro"], row[f"R@{cutoff}_micro"] = mm["macro"], mm["micro"]
        if with_ci:
            row["R@200_macro_ci_low"], row["R@200_macro_ci_high"] = ci(recall_at(ranks, 200), users, resamples, seed)
        for lo, hi, name in RANK_BUCKETS:
            row[f"share_rank_{name}"] = float(((ranks >= lo) & (ranks <= hi)).mean())
        row["median_rank"] = float(np.median(ranks))
        row["median_pre_t1_count"] = float(g["pre_t1_count"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def miss_composition(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Share of all instances, retrieval misses (rank > 200) and deep misses (rank > 5,000) per class."""
    miss, deep = frame["rank"] > 200, frame["rank"] > DEEP
    out = pd.DataFrame({"all": frame[column].value_counts(normalize=True),
                        "miss_gt200": frame.loc[miss, column].value_counts(normalize=True),
                        "deep_miss_gt5000": frame.loc[deep, column].value_counts(normalize=True)}).fillna(0.0)
    out["deep_over_all"] = out["deep_miss_gt5000"] / out["all"]
    return out.rename_axis(column).reset_index()


def standardised(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Macro R@200 per class re-weighted to the overall item-type mix (direct standardisation)."""
    weights = frame["item_type"].value_counts(normalize=True)
    rows = []
    for cls, g in frame.groupby(column):
        stratum = {t: macro_micro(recall_at(s["rank"].to_numpy(), 200), s["user"].to_numpy())["macro"]
                   for t, s in g.groupby("item_type")}
        if set(stratum) != set(weights.index):
            continue
        rows.append({column: cls, "R@200_macro_item_type_standardised": float(sum(weights[t] * stratum[t] for t in weights.index))})
    return pd.DataFrame(rows)


def ranking_given_retrieval(frame: pd.DataFrame, archive: dict, column: str) -> pd.DataFrame:
    """Among natural K1=200 hits: top-10 hit rate in retrieval order vs natural-matching MLP order."""
    if not np.array_equal(archive["inst"], frame["inst"].to_numpy()):
        raise ValueError("MLP archive is not aligned to the Val instances")
    hit = np.isfinite(archive["retrieval_ranks"])
    rows = []
    sub = frame.assign(mlp=archive["ranks"], ret=archive["retrieval_ranks"])[hit]
    pieces = [(("all", cls), g) for cls, g in sub.groupby(column)]
    pieces += [((itype, cls), g) for (cls, itype), g in sub.groupby([column, "item_type"])]
    for keys, g in pieces:
        users = g["user"].to_numpy()
        ret, mlp = recall_at(g["ret"].to_numpy(), 10), recall_at(g["mlp"].to_numpy(), 10)
        d = paired_bootstrap(users, ret, mlp, n_resamples=1000, seed=0)
        rows.append({"item_type": keys[0], column: keys[1], "n_retrieved": len(g),
                     "top10_retrieval_order_macro": macro_micro(ret, users)["macro"],
                     "top10_mlp_macro": macro_micro(mlp, users)["macro"],
                     "top10_mlp_micro": float(mlp.mean()),
                     "mlp_minus_retrieval": d["diff"], "diff_ci_low": d["ci_low"], "diff_ci_high": d["ci_high"]})
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    settings = {"label": LABEL, "split": "val", "data_config": args.config,
                "retriever_ranks": args.ranks, "mlp_ranks": args.mlp_ranks, "bootstrap_resamples": args.resamples,
                "bootstrap_seed": 0, "deep_miss_rank": DEEP, "rank_buckets": [b[2] for b in RANK_BUCKETS],
                "history": "last 50 prior items (ProcessedData.histories, most recent last)",
                "metadata": "items.leaf_cat (full category path; '' = missing), items.store",
                "pre_t1_count": "target interactions in retriever_train + ranker_train (descriptive only)"}
    out = make_run_dir("analysis", args.run_id, exist_ok=False)
    data = ProcessedData.load(args.config)
    val = data.split_instances("val")
    s = data.split
    if not ((val["ts"] >= s["t1"]) & (val["ts"] < s["t2"])).all():
        raise ValueError("non-validation instances")
    with np.load(args.ranks) as z:
        archive = {k: z[k] for k in z.files}
    if not (np.array_equal(archive["inst"], val["inst"].to_numpy()) and np.array_equal(archive["user"], val["user"].to_numpy())):
        raise ValueError("retriever ranks are not aligned to Val instances")
    histories = data.histories(val["user"].to_numpy(), val["hist_len"].to_numpy(), data.max_hist_len)
    target = val["item"].to_numpy()
    if (histories == target[:, None]).any():
        raise ValueError("target found in its own history")
    leaf, store = _lut(data, "leaf_cat"), _lut(data, "store")
    labels = classify(target, histories, leaf, store)
    groups = group_labels(data, val)
    if not np.array_equal(groups["inst"].to_numpy(), val["inst"].to_numpy()):
        raise ValueError("group labels misaligned")
    train = data.interactions(["retriever_train", "ranker_train"])
    pre_t1 = np.bincount(train["item"].to_numpy(), minlength=data.n_items + 1)
    frame = pd.concat([val[["inst", "user", "item"]], labels], axis=1).assign(
        item_type=groups["item_type"].to_numpy(), rank=archive["rank"], pre_t1_count=pre_t1[target])
    n = len(frame)

    tables = {
        "overall": summarise(frame.assign(all="all"), ["all"], n, args.resamples, 0),
        "last_class": summarise(frame, ["last_class"], n, args.resamples, 0),
        "same_leaf_last": summarise(frame, ["same_leaf_last"], n, args.resamples, 0),
        "leaf_in_history": summarise(frame, ["leaf_in_history"], n, args.resamples, 0),
        "last_class_by_item_type": summarise(frame, ["item_type", "last_class"], n, args.resamples, 0),
        "leaf_in_history_by_item_type": summarise(frame, ["item_type", "leaf_in_history"], n, args.resamples, 0),
        "joint_last_class_leaf_in_history": summarise(frame, ["leaf_in_history", "last_class"], n, args.resamples, 0, with_ci=False),
        "miss_composition_last_class": miss_composition(frame, "last_class"),
        "miss_composition_same_leaf_last": miss_composition(frame, "same_leaf_last"),
        "miss_composition_leaf_in_history": miss_composition(frame, "leaf_in_history"),
        "standardised_last_class": standardised(frame, "last_class"),
        "standardised_leaf_in_history": standardised(frame, "leaf_in_history"),
    }
    for itype, g in frame.groupby("item_type"):
        tables[f"miss_composition_same_leaf_last__{itype}"] = miss_composition(g, "same_leaf_last")
    if args.mlp_ranks and Path(args.mlp_ranks).exists():
        with np.load(args.mlp_ranks) as z:
            mlp = {k: z[k] for k in z.files}
        consistent = float(np.mean(np.isfinite(mlp["retrieval_ranks"]) == (archive["rank"] <= 200)))
        settings["mlp_retrieval_membership_agreement"] = consistent
        tables["ranking_given_retrieval_last_class"] = ranking_given_retrieval(frame, mlp, "last_class")
        tables["ranking_given_retrieval_leaf_in_history"] = ranking_given_retrieval(frame, mlp, "leaf_in_history")

    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    results = {"label": LABEL, "n_instances": n, "n_users": int(frame["user"].nunique()),
               "tables": {k: v.to_dict(orient="records") for k, v in tables.items()}}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    (out / "git_hash.txt").write_text(git_hash() + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for name in tables:
        print(f"\n== {name} ({LABEL})\n{tables[name].round(4).to_string(index=False)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--ranks", default="results/retriever/two_tower_text/seed0/val_ranks.npz")
    ap.add_argument("--mlp-ranks", default="results/ranker_mlp/20261028-natural-matching-seed0/val_ranks.npz")
    ap.add_argument("--run-id", default="20261028-complements")
    ap.add_argument("--resamples", type=int, default=1000)
    main(ap.parse_args())
