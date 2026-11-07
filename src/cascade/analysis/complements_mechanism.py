"""Does target-to-last-item similarity explain the same-store recall lift? (validation only, exploratory)

Pre-registered hypothesis: the same-store Recall@200 lift of the frozen Two-tower (text) retriever comes from
store/brand tokens in the item text, not behaviour. For each Val instance we compute the cosine between the
target and the LAST prior history item in (i) the frozen MiniLM text space and (ii) the frozen retriever's
own item-vector space, then compare classes (from ``complements.classify``), retrieved vs missed targets,
Recall@200 by cosine decile, and class differences within a cosine decile. No training; never touches Test.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from cascade.analysis.complements import _lut, classify
from cascade.data.instances import ProcessedData
from cascade.eval.harness import filtered_ranks, first_ts_tensor, make_batch
from cascade.eval.metrics import macro_micro, recall_at
from cascade.eval.ranker_meso import group_labels
from cascade.models.retriever import RetrieverScorer
from cascade.pipelines.train_retriever import load_retriever
from cascade.utils.io import git_hash, make_run_dir

LABEL = "validation only, exploratory"
MAIN = ["same leaf & same store", "same leaf, diff store", "diff leaf, same store", "diff leaf & diff store"]


def pair_cosine(vectors: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 32768) -> np.ndarray:
    """Row-wise dot product of (already L2-normalised) vectors[a] and vectors[b]."""
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", vectors[a[s:s + chunk]], vectors[b[s:s + chunk]])
    return out


def rank_check(model, data, val: pd.DataFrame, saved: np.ndarray, n: int = 256, seed: int = 0) -> float:
    """Recompute full-catalogue Val ranks for a random subset; return the fraction matching the archive."""
    idx = np.sort(np.random.default_rng(seed).choice(len(val), n, replace=False))
    batch = make_batch(data, val.iloc[idx], data.max_hist_len)
    with torch.no_grad():
        ranks, _ = filtered_ranks(RetrieverScorer(model).score_all(batch), batch, first_ts_tensor(data))
    return float(np.mean(ranks == saved[idx]))


def cos_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        hit = g["rank"] <= 200
        row = dict(zip(keys, key), n=len(g),
                   R200_macro=macro_micro(recall_at(g["rank"].to_numpy(), 200), g["user"].to_numpy())["macro"])
        for space in ("text_cos", "ret_cos"):
            row[f"{space}_mean"], row[f"{space}_median"] = g[space].mean(), g[space].median()
            row[f"{space}_mean_retrieved"] = g.loc[hit, space].mean() if hit.any() else np.nan
            row[f"{space}_mean_missed"] = g.loc[~hit, space].mean()
            row[f"{space}_retrieved_minus_missed"] = row[f"{space}_mean_retrieved"] - row[f"{space}_mean_missed"]
        rows.append(row)
    return pd.DataFrame(rows)


def decile_table(frame: pd.DataFrame, decile_col: str, class_col: str | None) -> pd.DataFrame:
    keys = [decile_col] + ([class_col] if class_col else [])
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        r = recall_at(g["rank"].to_numpy(), 200)
        mm = macro_micro(r, g["user"].to_numpy())
        rows.append(dict(zip(keys, key), n=len(g), R200_macro=mm["macro"], R200_micro=mm["micro"]))
    return pd.DataFrame(rows)


def standardised_contrast(frame: pd.DataFrame, bin_col: str, group_col: str, a: str, b: str,
                          resamples: int, seed: int, min_cell: int = 30) -> dict:
    """Bin-standardised micro R@200 of groups a and b and a - b with a 95% Poisson user-bootstrap CI.

    Common support only: bins where both groups have >= ``min_cell`` instances; weights = the passed
    population's mix over those bins (renormalised). Each user's instances share one Poisson(1) weight.
    """
    sub = frame[frame[group_col].isin([a, b])]
    bins = sub[bin_col].to_numpy().astype(int)
    n_bins = int(max(frame[bin_col].max(), bins.max())) + 1
    grp = (sub[group_col] == a).to_numpy().astype(int)
    cell = bins * 2 + grp
    hit = (sub["rank"].to_numpy() <= 200).astype(float)
    counts = np.bincount(cell, minlength=2 * n_bins).reshape(n_bins, 2)
    support = (counts >= min_cell).all(axis=1)
    weights = np.bincount(frame[bin_col].to_numpy().astype(int), minlength=n_bins).astype(float)
    weights = np.where(support, weights, 0.0)
    weights /= weights.sum()
    _, user_idx = np.unique(sub["user"].to_numpy(), return_inverse=True)

    def estimate(w: np.ndarray) -> tuple[float, float]:
        n = np.bincount(cell, weights=w, minlength=2 * n_bins).reshape(n_bins, 2)[support]
        h = np.bincount(cell, weights=w * hit, minlength=2 * n_bins).reshape(n_bins, 2)[support]
        rate = h / np.maximum(n, 1e-12)
        return float(weights[support] @ rate[:, 1]), float(weights[support] @ rate[:, 0])

    est_a, est_b = estimate(np.ones(len(sub)))
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for i in range(resamples):
        uw = rng.poisson(1.0, size=user_idx.max() + 1).astype(float)
        ea, eb = estimate(uw[user_idx])
        diffs[i] = ea - eb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    in_support = support[bins]
    raw_a, raw_b = hit[grp == 1].mean(), hit[grp == 0].mean()
    return {"bin_space": bin_col, "group_col": group_col, "a": a, "b": b, "n_a": int(grp.sum()),
            "n_b": int((grp == 0).sum()), "support_bins": int(support.sum()),
            "share_a_in_support": float(in_support[grp == 1].mean()), "share_b_in_support": float(in_support[grp == 0].mean()),
            "raw_micro_a": raw_a, "raw_micro_b": raw_b, "raw_diff": raw_a - raw_b,
            "standardised_a": est_a, "standardised_b": est_b, "standardised_diff": est_a - est_b,
            "ci_low": float(lo), "ci_high": float(hi), "share_of_raw_gap_remaining": (est_a - est_b) / (raw_a - raw_b)}


def popularity_by_bin(frame: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Pre-T1 interaction counts of target and last item, and item-type mix, per similarity bin."""
    return frame.groupby(bin_col).agg(n=("rank", "size"), R200_micro=("rank", lambda r: float((r <= 200).mean())),
                                      target_pre_t1_median=("target_cnt", "median"),
                                      last_pre_t1_median=("last_cnt", "median"),
                                      share_warm_head=("item_type", lambda s: float((s == "warm-head").mean())),
                                      share_new=("item_type", lambda s: float((s == "new-in-period").mean()))).reset_index()


def main(args: argparse.Namespace) -> None:
    settings = {"label": LABEL, "split": "val", "data_config": args.config, "retriever_run": args.retriever,
                "retriever_ranks": args.ranks, "text_embeddings": "data/processed/item_text_emb.npy (row = item id)",
                "pair": "target vs LAST prior history item", "deciles": "computed over all Val instances; bin05 = fixed 0.05-wide cosine bins",
                "contrast_support": "bins with >= 30 instances in both groups",
                "bootstrap": f"Poisson user bootstrap, {args.resamples} resamples, seed 0", "rank_check_n": 256}
    out = make_run_dir("analysis", args.run_id, exist_ok=False)
    torch.set_num_threads(2)
    data = ProcessedData.load(args.config)
    val = data.split_instances("val")
    if not ((val["ts"] >= data.split["t1"]) & (val["ts"] < data.split["t2"])).all():
        raise ValueError("non-validation instances")
    with np.load(args.ranks) as z:
        archive = {k: z[k] for k in z.files}
    if not np.array_equal(archive["inst"], val["inst"].to_numpy()):
        raise ValueError("ranks not aligned to Val instances")
    histories = data.histories(val["user"].to_numpy(), val["hist_len"].to_numpy(), data.max_hist_len)
    target, last = val["item"].to_numpy(), histories[:, -1].copy()
    labels = classify(target, histories, _lut(data, "leaf_cat"), _lut(data, "store"))
    del histories

    text = np.load(data.dir / "item_text_emb.npy", mmap_mode="r")
    if text.shape[0] != data.n_items + 1:
        raise ValueError("text embedding rows do not match item ids")
    norms = np.linalg.norm(np.asarray(text[1:]), axis=1)
    settings["text_norm_min_max"] = [float(norms.min()), float(norms.max())]
    text_cos = pair_cosine(np.asarray(text), target, last)
    model, _ = load_retriever(args.retriever, data, device="cpu")
    settings["rank_check_match_fraction"] = rank_check(model, data, val, archive["rank"])
    with torch.no_grad():
        item_vec = model.all_item_vectors().cpu().numpy()
    del model
    ret_cos = pair_cosine(item_vec, target, last)
    settings["text_vs_retriever_cos_spearman"] = float(pd.Series(text_cos).corr(pd.Series(ret_cos), method="spearman"))

    groups = group_labels(data, val)
    frame = pd.DataFrame({"inst": val["inst"].to_numpy(), "user": val["user"].to_numpy(),
                          "last_class": labels["last_class"].to_numpy(), "item_type": groups["item_type"].to_numpy(),
                          "rank": archive["rank"], "text_cos": text_cos, "ret_cos": ret_cos})
    for space in ("text_cos", "ret_cos"):
        frame[space.replace("cos", "decile")] = pd.qcut(frame[space], 10, labels=False)
        frame[space.replace("cos", "bin05")] = np.clip(np.floor((frame[space] + 1.0) / 0.05), 0, 39).astype(int)
    train = data.interactions(["retriever_train", "ranker_train"])
    counts = np.bincount(train["item"].to_numpy(), minlength=data.n_items + 1)
    frame["target_cnt"], frame["last_cnt"] = counts[target], counts[last]
    frame["store_match"] = np.where(frame["last_class"].isin(["same leaf & same store", "diff leaf, same store"]),
                                    "same store", np.where(frame["last_class"].isin(MAIN), "diff store", "unknown"))
    main4 = frame[frame["last_class"].isin(MAIN)]

    tables = {
        "cosine_by_class": cos_summary(frame, ["last_class"]),
        "cosine_by_item_type_class": cos_summary(main4, ["item_type", "last_class"]),
        "decile_edges": pd.DataFrame({space: frame[space].quantile(np.linspace(0, 1, 11)).to_numpy()
                                      for space in ("text_cos", "ret_cos")}).rename_axis("quantile_step").reset_index(),
    }
    for space in ("text_decile", "ret_decile"):
        tables[f"recall_by_{space}"] = decile_table(frame, space, None)
        tables[f"recall_by_{space}_class"] = decile_table(main4, space, "last_class")
        tables[f"recall_by_{space}_store"] = decile_table(main4, space, "store_match")
    tables["popularity_by_ret_decile"] = popularity_by_bin(frame, "ret_decile")
    tables["popularity_by_text_decile"] = popularity_by_bin(frame, "text_decile")
    contrasts = []
    for space in ("text_decile", "text_bin05", "ret_decile", "ret_bin05"):
        contrasts.append(standardised_contrast(main4, space, "store_match", "same store", "diff store", args.resamples, 0))
        contrasts.append(standardised_contrast(main4, space, "last_class", "same leaf & same store", "same leaf, diff store", args.resamples, 0))
        contrasts.append(standardised_contrast(main4, space, "last_class", "same leaf, diff store", "diff leaf & diff store", args.resamples, 0))
        for itype, g in main4.groupby("item_type"):
            row = standardised_contrast(g, space, "store_match", "same store", "diff store", args.resamples, 0)
            contrasts.append({**row, "item_type": itype})
    tables["standardised_contrasts"] = pd.DataFrame(contrasts)

    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "results.json").write_text(json.dumps({"label": LABEL, "n_instances": len(frame),
                                                  "tables": {k: v.to_dict(orient="records") for k, v in tables.items()}},
                                                 indent=2, default=float))
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    (out / "git_hash.txt").write_text(git_hash() + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(json.dumps(settings, indent=2))
    for name, table in tables.items():
        print(f"\n== {name} ({LABEL})\n{table.round(4).to_string(index=False)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--retriever", default="results/retriever/two_tower_text/seed0")
    ap.add_argument("--ranks", default="results/retriever/two_tower_text/seed0/val_ranks.npz")
    ap.add_argument("--run-id", default="20261028-complements-mechanism")
    ap.add_argument("--resamples", type=int, default=1000)
    main(ap.parse_args())
