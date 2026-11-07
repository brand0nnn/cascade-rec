"""Retriever staleness on Val (validation only, exploratory).

Question: does the frozen Two-tower (text) retriever (trained before t_ranker_train) retrieve worse later in
Val, and worse for products first reviewed after its training data ended, once product age, item type and user
history length are held fixed? Reads saved per-instance Val ranks only; never touches Test; trains nothing.

Run from the repo root:
    PYTHONPATH=src python -m cascade.analysis.staleness --out results/analysis/20261028-staleness
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.ranker_meso import group_labels
from cascade.utils.io import REPO_ROOT, git_hash

LABEL = "validation only, exploratory"
AGE_ORDER = ["retriever era", "ranker-training era", "Val era"]


def product_era(first_ts: np.ndarray, split: dict) -> np.ndarray:
    """Target product's first-appearance period relative to the retriever's training cut-off."""
    return np.select([first_ts < split["t_ranker_train"], first_ts < split["t1"]], AGE_ORDER[:2], default=AGE_ORDER[2])


def val_quarter(ts: np.ndarray, split: dict) -> np.ndarray:
    """Equal-duration quarters of the Val period [T1, T2): 'Q1'..'Q4'."""
    frac = (ts - split["t1"]) / (split["t2"] - split["t1"])
    return np.array(["Q1", "Q2", "Q3", "Q4"])[np.clip((frac * 4).astype(int), 0, 3)]


def cluster_boot(users: np.ndarray, values: np.ndarray, groups: np.ndarray, levels: list[str],
                 n_resamples: int = 1000, seed: int = 0) -> np.ndarray:
    """User-cluster bootstrap of per-user macro means for each group level.

    Users are resampled with replacement from all users present (in any level), so a user appearing in two
    levels keeps both contributions together. Returns [n_resamples, len(levels)].
    """
    codes, _ = pd.factorize(users)
    n_users = codes.max() + 1
    means = np.zeros((len(levels), n_users))
    has = np.zeros((len(levels), n_users), dtype=bool)
    for j, lv in enumerate(levels):
        m = groups == lv
        s = np.bincount(codes[m], weights=values[m], minlength=n_users)
        c = np.bincount(codes[m], minlength=n_users)
        has[j] = c > 0
        means[j, has[j]] = s[has[j]] / c[has[j]]
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, len(levels)))
    for b in range(n_resamples):
        w = np.bincount(rng.integers(0, n_users, n_users), minlength=n_users).astype(float)
        wh = w * has
        with np.errstate(invalid="ignore"):
            out[b] = (wh * means).sum(1) / wh.sum(1)
    return out


def ci(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2


def bucket_table(df: pd.DataFrame, by: list[str], n_resamples: int) -> pd.DataFrame:
    """n, users, macro/micro Recall@200/500, median rank, user-bootstrap CI of macro Recall@200 per bucket."""
    rows = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        u = g["user"].to_numpy()
        r200, r500 = g["r200"].to_numpy(), g["r500"].to_numpy()
        boot = cluster_boot(u, r200, np.zeros(len(g), dtype=object).astype(str), ["0"], n_resamples)[:, 0]
        mm200, mm500 = M.macro_micro(r200, u), M.macro_micro(r500, u)
        rows.append(dict(zip(by, key)) | {
            "n_instances": len(g), "n_users": int(pd.Series(u).nunique()),
            "recall200_macro": mm200["macro"], "recall200_macro_ci_lo": ci(boot)[0], "recall200_macro_ci_hi": ci(boot)[1],
            "recall200_micro": mm200["micro"], "recall500_macro": mm500["macro"], "recall500_micro": mm500["micro"],
            "median_rank": float(np.median(g["rank"].to_numpy())),
        })
    return pd.DataFrame(rows)


def contrast(df: pd.DataFrame, col: str, a: str, b: str, strata: list[str], n_resamples: int) -> pd.DataFrame:
    """Macro Recall@200 of level b minus level a of ``col``, within each stratum value, with user-cluster CI."""
    rows = []
    for name in strata:
        pieces = [("all", df)] if name == "all" else list(df.groupby(name, sort=True))
        for sv, g in pieces:
            g = g[g[col].isin([a, b])]
            na, nb = int((g[col] == a).sum()), int((g[col] == b).sum())
            if na < 50 or nb < 50:
                continue
            u, v, grp = g["user"].to_numpy(), g["r200"].to_numpy(), g[col].to_numpy().astype(str)
            pa = M.macro_micro(v[grp == a], u[grp == a])["macro"]
            pb = M.macro_micro(v[grp == b], u[grp == b])["macro"]
            boot = cluster_boot(u, v, grp, [a, b], n_resamples)
            d = boot[:, 1] - boot[:, 0]
            rows.append({"stratifier": name, "stratum": sv, f"n_{a}": na, f"n_{b}": nb,
                         f"recall200_{a}": pa, f"recall200_{b}": pb, "diff": pb - pa,
                         "diff_ci_lo": ci(d)[0], "diff_ci_hi": ci(d)[1]})
    return pd.DataFrame(rows)


def standardised_month(df: pd.DataFrame) -> pd.DataFrame:
    """Direct standardisation (micro Recall@200) of each month to the overall Val mix of
    product era × item type × history bucket. Cells empty in a month are dropped and the weights renormalised."""
    cell = ["era", "item_type", "history_length"]
    w = df.groupby(cell).size() / len(df)
    rows = []
    for mo, g in df.groupby("month"):
        rate = g.groupby(cell)["r200"].mean()
        ww = w.reindex(rate.index)
        rows.append({"month": mo, "crude_micro": float(g["r200"].mean()),
                     "standardised_micro": float((rate * ww).sum() / ww.sum()), "weight_covered": float(ww.sum())})
    return pd.DataFrame(rows)


def rescore_ranks(data: ProcessedData, inst: pd.DataFrame, retriever_dir: str, batch_size: int = 512) -> dict:
    """Recompute Val ranks with the frozen retriever (inference only) and a counterfactual 'frozen-pool' rank:
    the target's rank when every competitor first released at/after T1 is removed (target always kept).
    Separates catalogue growth during Val from a change in how well the retriever scores older products."""
    import torch
    from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
    from cascade.models.retriever import RetrieverScorer
    from cascade.pipelines.train_retriever import load_retriever

    torch.set_num_threads(4)
    model, _ = load_retriever(REPO_ROOT / retriever_dir, data, "cpu")
    scorer = RetrieverScorer(model)
    fts = first_ts_tensor(data)
    pre = fts < data.split["t1"]
    full, frozen = np.empty(len(inst)), np.empty(len(inst))
    for lo in range(0, len(inst), batch_size):
        b = make_batch(data, inst.iloc[lo:lo + batch_size], data.max_hist_len)
        sc = apply_filters(scorer.score_all(b), b, fts)
        tg = torch.from_numpy(b.targets.astype(np.int64))
        t = sc[torch.arange(len(tg)), tg].unsqueeze(1)
        ge = sc >= t
        full[lo:lo + len(tg)] = ge.sum(1).numpy()
        frozen[lo:lo + len(tg)] = ((ge & pre.unsqueeze(0)).sum(1) + (~pre[tg]).long()).numpy()
    return {"full": full, "frozen_pool": frozen}


def main(out: str, ranks_path: str, n_resamples: int, rescore: str | None = None) -> None:
    out_dir = REPO_ROOT / out
    out_dir.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load("configs/data.yaml")
    split = data.split
    inst = data.split_instances("val")
    with np.load(REPO_ROOT / ranks_path, allow_pickle=False) as z:
        z_inst, z_user, z_rank, z_ok, z_top = z["inst"], z["user"], z["rank"], z["scorable"], z["top10"]
    if not (np.array_equal(z_inst, inst["inst"].to_numpy()) and np.array_equal(z_user, inst["user"].to_numpy())):
        raise ValueError("val_ranks.npz is not aligned with Val instances")
    if inst["ts"].min() < split["t1"] or inst["ts"].max() >= split["t2"]:
        raise ValueError("Non-Val timestamps in Val instances")
    rank = np.where(z_ok, z_rank, np.inf)
    labels = group_labels(data, inst)
    first_ts = data.items.set_index("item").loc[inst["item"].to_numpy(), "first_ts"].to_numpy()
    if np.any(first_ts > inst["ts"].to_numpy()):
        raise ValueError("Target first_ts after its own instance timestamp")
    # warm-head/warm-tail popularity as of Val start (counts before T1), as a finer popularity control
    pre = data.interactions(["retriever_train", "ranker_train"])
    cnt_t1 = np.bincount(pre["item"].to_numpy(), minlength=data.n_items + 1)[inst["item"].to_numpy()]
    rt = data.interactions(["ranker_train"])  # "recent" popularity: the ~12.5 months before Val
    cnt_rt = np.bincount(rt["item"].to_numpy(), minlength=data.n_items + 1)[inst["item"].to_numpy()]
    df = pd.DataFrame({
        "inst": inst["inst"].to_numpy(), "user": inst["user"].to_numpy(), "rank": rank,
        "r200": (rank <= 200).astype(float), "r500": (rank <= 500).astype(float),
        "month": pd.to_datetime(inst["ts"].to_numpy(), unit="ms", utc=True).strftime("%Y-%m"),
        "quarter": val_quarter(inst["ts"].to_numpy(), split),
        "era": product_era(first_ts, split), "item_type": labels["item_type"].to_numpy(),
        "history_length": labels["history_length"].to_numpy(),
        "pop_t1_bucket": np.select([cnt_t1 == 0, cnt_t1 < 5, cnt_t1 < 20, cnt_t1 < 100], ["0", "1-4", "5-19", "20-99"], "100+"),
        "pop_t1_fine": np.select([cnt_t1 < 5, cnt_t1 < 20, cnt_t1 < 50, cnt_t1 < 100, cnt_t1 < 300, cnt_t1 < 1000],
                                 ["a:0-4", "b:5-19", "c:20-49", "d:50-99", "e:100-299", "f:300-999"], "g:1000+"),
        "pop_recent_bucket": np.select([cnt_rt < 5, cnt_rt < 20, cnt_rt < 50, cnt_rt < 100, cnt_rt < 300],
                                       ["a:0-4", "b:5-19", "c:20-49", "d:50-99", "e:100-299"], "f:300+"),
        "cnt_t1": cnt_t1, "cnt_recent": cnt_rt,
    })
    # top-10 intrusion of products released during Val (per month)
    top_ft = data.items.set_index("item")["first_ts"].reindex(z_top.ravel()).to_numpy().reshape(z_top.shape)
    df["top10_share_val_era"] = (top_ft >= split["t1"]).mean(1)
    assert ((df["era"] == "Val era") == (df["item_type"] == "new-in-period")).all(), "Val era must equal new-in-period"

    rescore_check = None
    if rescore:
        rr = rescore_ranks(data, inst, rescore)
        agree = rr["full"] == z_rank
        rescore_check = {"retriever_dir": rescore, "exact_rank_agreement": float(agree.mean()),
                         "max_abs_rank_diff": float(np.max(np.abs(rr["full"] - z_rank))),
                         "recall200_agreement": float(np.mean((rr["full"] <= 200) == (z_rank <= 200)))}
        df["rank_fp"] = rr["frozen_pool"]
        df["r200_fp"] = (rr["frozen_pool"] <= 200).astype(float)
    # catalogue growth: released items (first_ts <= month start) — the filtered pool grows during Val
    ft = np.sort(data.items["first_ts"].to_numpy())
    months = sorted(df["month"].unique())
    released = {m: int(np.searchsorted(ft, pd.Timestamp(m + "-01", tz="UTC").value // 10**6, side="right"))
                for m in months}
    released[months[0]] = int(np.searchsorted(ft, split["t1"], side="right"))

    tables = {
        "by_month": bucket_table(df, ["month"], n_resamples),
        "by_era": bucket_table(df, ["era"], n_resamples),
        "by_item_type": bucket_table(df, ["item_type"], n_resamples),
        "by_quarter": bucket_table(df, ["quarter"], n_resamples),
        "by_quarter_era": bucket_table(df, ["era", "quarter"], n_resamples),
        "by_quarter_item_type": bucket_table(df, ["item_type", "quarter"], n_resamples),
        "by_quarter_history": bucket_table(df, ["history_length", "quarter"], n_resamples),
        "by_month_item_type": bucket_table(df, ["item_type", "month"], n_resamples),
        "mix_by_quarter": (df.groupby(["quarter", "era"]).size().unstack(fill_value=0)
                           .pipe(lambda t: t.div(t.sum(1), axis=0)).reset_index()),
        "q4_minus_q1": contrast(df, "quarter", "Q1", "Q4", ["all", "era", "item_type", "history_length"], n_resamples),
        "warm_head_era": bucket_table(df[df["item_type"] == "warm-head"], ["era"], n_resamples),
        "warm_head_era_contrast": contrast(df[df["item_type"] == "warm-head"], "era", "retriever era",
                                           "ranker-training era", ["all", "pop_t1_bucket", "history_length", "quarter"],
                                           n_resamples),
        "warm_tail_era_contrast": contrast(df[df["item_type"] == "warm-tail"], "era", "retriever era",
                                           "ranker-training era", ["all", "pop_t1_bucket"], n_resamples),
        "standardised_month": standardised_month(df),
        "warm_head_era_by_pop_fine": bucket_table(df[df["item_type"] == "warm-head"], ["pop_t1_fine", "era"], n_resamples),
        "warm_head_era_contrast_fine": contrast(df[df["item_type"] == "warm-head"], "era", "retriever era",
                                                "ranker-training era", ["pop_t1_fine", "pop_recent_bucket"], n_resamples),
        "pop_by_era_quarter": df[df["era"] != "Val era"].groupby(["era", "quarter"])[["cnt_t1", "cnt_recent"]]
                              .median().reset_index(),
        "top10_val_era_share_by_month": df.groupby("month")["top10_share_val_era"].mean().reset_index(),
    }
    if rescore:
        fp = df.assign(r200=df["r200_fp"], rank=df["rank_fp"], r500=np.nan)
        tables["frozen_pool_q4_minus_q1"] = contrast(fp, "quarter", "Q1", "Q4", ["all", "era", "item_type"], n_resamples)
        tables["frozen_pool_by_month"] = bucket_table(fp, ["month"], n_resamples).drop(
            columns=["recall500_macro", "recall500_micro"])
        tables["frozen_pool_standardised_month"] = standardised_month(fp)
    tables["standardised_month"]["released_items_at_month_start"] = tables["standardised_month"]["month"].map(released)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    config = {"label": LABEL, "split": "val", "ranks": ranks_path, "data_config": "configs/data.yaml",
              "bootstrap": {"n_resamples": n_resamples, "seed": 0, "unit": "user (cluster, within bucket/stratum)"},
              "recall_cutoffs": [200, 500], "unscorable": "treated as miss",
              "quarters": "equal-duration quarters of [T1, T2)",
              "eras": {"retriever era": "first_ts < t_ranker_train", "ranker-training era": "t_ranker_train <= first_ts < T1",
                       "Val era": "first_ts >= T1 (= new-in-period)"},
              "pop_t1_bucket": "target review count before T1",
              "pop_recent_bucket": "target review count in ranker_train period [t_ranker_train, T1)",
              "rescore": rescore,
              "frozen_pool": "rank among items first released before T1 (target always kept); inference only"}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    (out_dir / "git_hash.txt").write_text(git_hash() + "\n")
    results = {"label": LABEL, "n_instances": len(df), "n_users": int(df["user"].nunique()),
               "overall": {"recall200": M.macro_micro(df["r200"].to_numpy(), df["user"].to_numpy()),
                           "recall500": M.macro_micro(df["r500"].to_numpy(), df["user"].to_numpy())},
               "released_items_at_month_start": released, "rescore_check": rescore_check,
               "tables": {k: t.to_dict(orient="records") for k, t in tables.items()}}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("rescore_check:", rescore_check)
    for name in ["by_month", "by_era", "by_item_type", "mix_by_quarter", "q4_minus_q1", "warm_head_era",
                 "warm_head_era_contrast", "warm_tail_era_contrast", "standardised_month",
                 "warm_head_era_by_pop_fine", "warm_head_era_contrast_fine", "pop_by_era_quarter",
                 "top10_val_era_share_by_month"] + [k for k in tables if k.startswith("frozen_pool")]:
        print(f"\n== {name} ({LABEL})\n", tables[name].round(4).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------
# Follow-up: popularity in the retriever-training period (saved ranks only, no re-scoring)
# ---------------------------------------------------------------------------------------------------------------
COUNT_EDGES = [0, 1, 5, 20, 50, 100, 300, 1000]
COUNT_LABELS = ["0", "1-4", "5-19", "20-49", "50-99", "100-299", "300-999", "1000+"]
MIN_ITEMS = 20


def count_bucket(c: np.ndarray) -> np.ndarray:
    """Log-ish count buckets; labels are prefixed with their order (e.g. '2:5-19') so they sort correctly."""
    idx = np.searchsorted(COUNT_EDGES, c, side="right") - 1
    return np.array([f"{i}:{COUNT_LABELS[i]}" for i in range(len(COUNT_LABELS))])[idx]


def item_boot_macro(items: np.ndarray, users: np.ndarray, values: np.ndarray, groups: np.ndarray,
                    levels: list[str], n_resamples: int = 1000, seed: int = 0) -> np.ndarray:
    """Item-cluster bootstrap of per-user macro means per group level.

    Target items are resampled with replacement (from all items present in any level); each instance gets weight =
    its item's multiplicity; macro = weighted mean within user, then plain mean over users with positive weight.
    Returns [n_resamples, len(levels)].
    """
    icode, _ = pd.factorize(items)
    ucode, _ = pd.factorize(users)
    n_items, n_users = icode.max() + 1, ucode.max() + 1
    lv_idx = [np.flatnonzero(groups == lv) for lv in levels]
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, len(levels)))
    for b in range(n_resamples):
        w_item = np.bincount(rng.integers(0, n_items, n_items), minlength=n_items).astype(float)
        for j, ix in enumerate(lv_idx):
            w = w_item[icode[ix]]
            sw = np.bincount(ucode[ix], weights=w, minlength=n_users)
            sv = np.bincount(ucode[ix], weights=w * values[ix], minlength=n_users)
            ok = sw > 0
            out[b, j] = (sv[ok] / sw[ok]).mean() if ok.any() else np.nan
    return out


def pop_cell_table(df: pd.DataFrame, by: list[str], n_resamples: int) -> pd.DataFrame:
    """n instances, n users, n distinct items, macro Recall@10/@200 (+ item-cluster CI for @200) per cell."""
    rows = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        u, it = g["user"].to_numpy(), g["item"].to_numpy()
        n_it = int(pd.Series(it).nunique())
        boot = item_boot_macro(it, u, g["r200"].to_numpy(), np.full(len(g), "x"), ["x"], n_resamples)[:, 0]
        lo, hi = ci(boot)
        rows.append(dict(zip(by, key)) | {
            "n_instances": len(g), "n_users": int(pd.Series(u).nunique()), "n_items": n_it,
            "few_items_flag": n_it < MIN_ITEMS,
            "recall10_macro": M.macro_micro(g["r10"].to_numpy(), u)["macro"],
            "recall200_macro": M.macro_micro(g["r200"].to_numpy(), u)["macro"],
            "recall200_item_ci_lo": lo, "recall200_item_ci_hi": hi,
            "recall200_micro": float(g["r200"].mean()),
            "median_cnt_retriever_train": float(g["cnt_rtrain"].median()),
            "median_cnt_pre_t1": float(g["cnt_t1"].median()),
        })
    return pd.DataFrame(rows)


def pop_contrast(df: pd.DataFrame, col: str, a: str, b: str, label: str, n_resamples: int) -> dict:
    """Macro Recall@200 (and @10) of level b minus level a of ``col`` with an item-cluster bootstrap CI."""
    g = df[df[col].isin([a, b])]
    grp = g[col].to_numpy().astype(str)
    u, it = g["user"].to_numpy(), g["item"].to_numpy()
    out = {"contrast": label, "a": a, "b": b}
    for lv, tag in [(a, "a"), (b, "b")]:
        m = grp == lv
        out[f"n_inst_{tag}"] = int(m.sum())
        out[f"n_items_{tag}"] = int(pd.Series(it[m]).nunique())
        out[f"recall200_{tag}"] = M.macro_micro(g["r200"].to_numpy()[m], u[m])["macro"]
        out[f"recall10_{tag}"] = M.macro_micro(g["r10"].to_numpy()[m], u[m])["macro"]
    out["few_items_flag"] = min(out["n_items_a"], out["n_items_b"]) < MIN_ITEMS
    out["diff200"] = out["recall200_b"] - out["recall200_a"]
    out["diff10"] = out["recall10_b"] - out["recall10_a"]
    if out["n_inst_a"] and out["n_inst_b"]:
        boot = item_boot_macro(it, u, g["r200"].to_numpy(), grp, [a, b], n_resamples)
        out["diff200_item_ci_lo"], out["diff200_item_ci_hi"] = ci(boot[:, 1] - boot[:, 0])
    return out


def popularity_main(out: str, ranks_path: str, n_resamples: int) -> None:
    """Warm Val targets: is recall explained by the target's count in the retriever-training period?"""
    out_dir = REPO_ROOT / out
    out_dir.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load("configs/data.yaml")
    split = data.split
    inst = data.split_instances("val")
    with np.load(REPO_ROOT / ranks_path, allow_pickle=False) as z:
        z_inst, z_user, z_rank, z_ok = z["inst"], z["user"], z["rank"], z["scorable"]
    if not (np.array_equal(z_inst, inst["inst"].to_numpy()) and np.array_equal(z_user, inst["user"].to_numpy())):
        raise ValueError("val_ranks.npz is not aligned with Val instances")
    if inst["ts"].min() < split["t1"] or inst["ts"].max() >= split["t2"]:
        raise ValueError("Non-Val timestamps in Val instances")
    rank = np.where(z_ok, z_rank, np.inf)
    labels = group_labels(data, inst)
    items = inst["item"].to_numpy()
    first_ts = data.items.set_index("item").loc[items, "first_ts"].to_numpy()
    n = data.n_items + 1
    c_rtrain = np.bincount(data.interactions(["retriever_train"])["item"].to_numpy(), minlength=n)
    c_rk = np.bincount(data.interactions(["ranker_train"])["item"].to_numpy(), minlength=n)
    df = pd.DataFrame({
        "user": inst["user"].to_numpy(), "item": items, "rank": rank,
        "r10": (rank <= 10).astype(float), "r200": (rank <= 200).astype(float),
        "era": product_era(first_ts, split), "item_type": labels["item_type"].to_numpy(),
        "cnt_rtrain": c_rtrain[items], "cnt_ranker": c_rk[items], "cnt_t1": c_rtrain[items] + c_rk[items],
    })
    df = df[df["item_type"] != "new-in-period"].reset_index(drop=True)
    # sanity: era is a deterministic function of the retriever-training count for warm targets
    if not (((df["era"] == "retriever era") == (df["cnt_rtrain"] > 0)).all() and (df["cnt_t1"] > 0).all()):
        raise ValueError("Era / retriever-training count mismatch")
    df["rtrain_bucket"] = count_bucket(df["cnt_rtrain"].to_numpy())
    df["pre_t1_bucket"] = count_bucket(df["cnt_t1"].to_numpy())
    df["ranker_period_bucket"] = count_bucket(df["cnt_ranker"].to_numpy())

    tables = {
        "a_by_rtrain_bucket": pop_cell_table(df, ["rtrain_bucket"], n_resamples),
        "a_by_pre_t1_bucket": pop_cell_table(df, ["pre_t1_bucket"], n_resamples),
        "b_rtrain_x_pre_t1": pop_cell_table(df, ["rtrain_bucket", "pre_t1_bucket"], n_resamples),
        "b_rtrain_x_ranker_period": pop_cell_table(df, ["rtrain_bucket", "ranker_period_bucket"], n_resamples),
        "c_era_x_pre_t1": pop_cell_table(df, ["era", "pre_t1_bucket"], n_resamples),
    }
    # contrasts (item-cluster bootstrap)
    rows = []
    rb = sorted(df["rtrain_bucket"].unique())
    for lo_b, hi_b in zip(rb[:-1], rb[1:]):  # (a) adjacent retriever-training buckets
        rows.append(pop_contrast(df, "rtrain_bucket", lo_b, hi_b, "a: next retriever-train bucket", n_resamples))
    # (b) within a retriever-training bucket: high vs low recent (ranker-period) popularity
    for b_ in rb:
        g = df[df["rtrain_bucket"] == b_]
        med = np.median(g["cnt_ranker"])
        g = g.assign(recent=np.where(g["cnt_ranker"] > med, "above median", "at/below median"))
        if (g["recent"] == "above median").sum() and (g["recent"] == "at/below median").sum():
            r = pop_contrast(g, "recent", "at/below median", "above median",
                             f"b: recent pop within rtrain {b_} (median {med:g})", n_resamples)
            rows.append(r | {"stratum": b_})
    # (c) ranker-era (rtrain 0) vs retriever-era at matched pre-T1 count; and vs retriever-era rtrain 1-4
    for pb in sorted(df["pre_t1_bucket"].unique()):
        g = df[df["pre_t1_bucket"] == pb]
        if g["era"].nunique() == 2:
            rows.append(pop_contrast(g, "era", "retriever era", "ranker-training era",
                                     f"c: era at matched pre-T1 {pb}", n_resamples) | {"stratum": pb})
    low = df[(df["rtrain_bucket"] == "1:1-4") | (df["era"] == "ranker-training era")]
    for pb in sorted(df.loc[df["era"] == "ranker-training era", "pre_t1_bucket"].unique()):
        g = low[(low["era"] == "retriever era") | (low["pre_t1_bucket"] == pb)]
        rows.append(pop_contrast(g.assign(grp=np.where(g["era"] == "retriever era", "rtrain 1-4", f"ranker-era pre-T1 {pb}")),
                                 "grp", "rtrain 1-4", f"ranker-era pre-T1 {pb}",
                                 "c: ranker-era (rtrain 0) by pre-T1 vs retriever-era rtrain 1-4", n_resamples)
                    | {"stratum": pb})
    tables["contrasts_item_bootstrap"] = pd.DataFrame(rows)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    config = {"label": LABEL, "split": "val", "ranks": ranks_path, "data_config": "configs/data.yaml",
              "population": "warm Val targets (item_type != new-in-period); unscorable treated as miss",
              "counts": {"cnt_rtrain": "target reviews before t_ranker_train",
                         "cnt_ranker": "target reviews in [t_ranker_train, T1)", "cnt_t1": "target reviews before T1"},
              "buckets": COUNT_LABELS, "few_items_flag": f"n distinct items < {MIN_ITEMS}",
              "bootstrap": {"n_resamples": n_resamples, "seed": 0, "unit": "target item (cluster)"},
              "rescoring": "none (saved ranks only)"}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    (out_dir / "git_hash.txt").write_text(git_hash() + "\n")
    results = {"label": LABEL, "n_instances": len(df), "n_users": int(df["user"].nunique()),
               "n_items": int(df["item"].nunique()),
               "tables": {k: t.to_dict(orient="records") for k, t in tables.items()}}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 200)
    for name, t in tables.items():
        print(f"\n== {name} ({LABEL})\n", t.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/analysis/20261028-staleness")
    p.add_argument("--ranks", default="results/retriever/two_tower_text/seed0/val_ranks.npz")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--mode", choices=["staleness", "popularity"], default="staleness")
    p.add_argument("--rescore", default=None,
                   help="retriever run dir: recompute Val ranks (inference only) for the frozen-pool counterfactual")
    a = p.parse_args()
    if a.mode == "popularity":
        popularity_main(a.out, a.ranks, a.n_resamples)
    else:
        main(a.out, a.ranks, a.n_resamples, a.rescore)
