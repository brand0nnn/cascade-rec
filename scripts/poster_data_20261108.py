"""Data for the 8 Nov poster diagrams (Validation unless stated; Test only for the funnel's Test column).

Writes results/targeted_study/poster-data-20261108/:
  funnel.csv          retrieval miss / ranking miss / hit@10 per system, Val and Test (seed means)
  overlap.json        20,000 random Val purchases (RNG 0): recency top 200 vs two-tower top 200 overlap and
                      which retriever found the purchase, overall and by item group
  heatmap.csv         user behaviour x (store vs history) x (product age) : purchases, hit@10 original / best
  launches.csv        monthly nDCG@10 original / best for iPhone 13, Galaxy S22, Pixel 6 accessory purchases
  upgrader_example.json   one upgrade purchase (fixed RNG among qualifying cases) with the recency retriever's
                      10 closest candidates, and the share statistics from upgrader_candidates.csv
  future_title.json   example purchases dated before the iPhone 13 announcement whose title names iPhone 13

    .venv/bin/python scripts/poster_data_20261108.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.saved_results import average
from cascade.pipelines.build_candidates import load_candidates
sys.path.insert(0, str(Path(__file__).parent))
from analysis_behaviour_failures import device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/targeted_study/poster-data-20261108"
DAY = 86_400_000.0
VAL = {"original": ("mlp_tt", "hybrid-20261107-s{}"), "best": ("din_rrf", "rrf-20261108-s{}")}


def val_ranks():
    out, ref = {}, None
    for k, (arm, run) in VAL.items():
        out[k] = np.stack([np.load(ROOT / "results/hybrid_study" / run.format(s) / f"{arm}_seed{s}" / "ranks.npz")["ranks"] for s in (0, 1, 2)])
        ref = np.load(ROOT / "results/hybrid_study" / run.format(0) / f"{arm}_seed0" / "ranks.npz")
    return out, ref


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    ranks, ref = val_ranks()
    rows = []
    for k, r in ranks.items():
        rows.append(dict(split="Validation", system=k, retrieval_miss=np.isinf(r).mean(), ranking_miss=(np.isfinite(r) & (r > 10)).mean(), hit10=(r <= 10).mean()))
    T = ROOT / "results/test_final_ranker/20261108"
    for k, arm in (("original", "mlp_tt"), ("best", "din_rrf")):
        r = np.stack([np.load(T / f"{arm}_seed{s}_test_ranks.npz")["ranks"] for s in (0, 1, 2)])
        rows.append(dict(split="Test", system=k, retrieval_miss=np.isinf(r).mean(), ranking_miss=(np.isfinite(r) & (r > 10)).mean(), hit10=(r <= 10).mean()))
    pd.DataFrame(rows).to_csv(OUT / "funnel.csv", index=False)

    inst = data.instances.set_index("inst").loc[ref["inst"]].reset_index()
    users = ref["user"]
    it = data.items.set_index("item")
    titles = it.title.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    first_ts = np.full(data.n_items + 1, np.iinfo(np.int64).max); first_ts[data.items["item"].to_numpy()] = data.items.first_ts.to_numpy()
    E = np.load(data.dir / "item_text_emb.npy").astype(np.float32)
    tt = load_candidates(ROOT / "data/processed/candidates/two_tower_text_seed0", "val", k1=200)
    assert np.array_equal(tt["inst"], ref["inst"])
    seg = it.segment.astype(str).reindex(inst["item"].to_numpy()).to_numpy()
    newp = first_ts[inst["item"].to_numpy()] >= data.split["t1"]
    grp = np.where(newp, "new", np.where(seg == "head", "popular", "niche"))

    def recency_top(i, k=200):
        u, hl, t = int(inst.user[i]), int(inst.hist_len[i]), int(inst.ts[i])
        st = data.user_ptr[u]; full = data.seq_item[st: st + hl]; fts = data.seq_ts[st: st + hl]
        h, ht = full[-50:], fts[-50:]
        w = np.exp2(-(t - ht) / DAY / 90).astype(np.float32)
        q = (w[:, None] * E[h]).sum(0); q /= max(np.linalg.norm(q), 1e-12)
        s = E @ q; s[0] = -np.inf; s[first_ts > t] = -np.inf; s[full] = -np.inf
        top = np.argpartition(-s, k)[:k]
        return top[np.argsort(-s[top])], h

    sample = np.random.default_rng(0).choice(len(inst), 20000, replace=False)
    ov, found = [], []
    for i in sample:
        rec, _ = recency_top(i)
        ttc = tt["cand"][i]; ttc = ttc[ttc > 0]
        ov.append(len(set(rec.tolist()) & set(ttc.tolist())))
        tgt = int(inst["item"][i])
        found.append((tgt in set(rec.tolist()), tgt in set(ttc.tolist())))
    found = np.array(found)
    o = {"n": 20000, "mean_overlap_of_200": float(np.mean(ov)), "groups": {}}
    for g in ("all", "popular", "niche", "new"):
        m = np.ones(len(sample), bool) if g == "all" else grp[sample] == g
        f = found[m]
        o["groups"][g] = {"n": int(m.sum()), "recency_only": float((f[:, 0] & ~f[:, 1]).mean()), "two_tower_only": float((~f[:, 0] & f[:, 1]).mean()),
                          "both": float((f[:, 0] & f[:, 1]).mean()), "neither": float((~f[:, 0] & ~f[:, 1]).mean())}
    (OUT / "overlap.json").write_text(json.dumps(o, indent=2))
    print("overlap", json.dumps(o)[:400])

    lab = pd.read_parquet(ROOT / "results/targeted_study/behaviour-failures-20261108/labels.parquet").set_index("inst").loc[ref["inst"]]
    store = it.store.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    hist = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    ts_store = store[inst["item"].to_numpy()]
    same = np.array([bool(s) and s in set(store[h[h > 0]]) for s, h in zip(ts_store, hist)])
    age = (inst.ts.to_numpy() - first_ts[inst["item"].to_numpy()]) / DAY
    agel = np.where(age < 180, "fresh (<6 months)", "older (6 months+)")
    beh = lab.behaviour.replace({"other model, same family": "other phone", "other device, same platform": "other phone",
                                 "older model": "other phone", "unknown history": "no phone in history"}).to_numpy()
    h_o, h_b = (ranks["original"] <= 10).mean(0), (ranks["best"] <= 10).mean(0)
    hm = pd.DataFrame({"behaviour": beh, "store": np.where(same, "store bought from before", "new store"), "age": agel,
                       "hit_o": h_o, "hit_b": h_b}).groupby(["behaviour", "store", "age"]).agg(purchases=("hit_o", "size"), hit_o=("hit_o", "mean"), hit_b=("hit_b", "mean")).reset_index()
    hm.to_csv(OUT / "heatmap.csv", index=False)

    when = pd.to_datetime(inst.ts.to_numpy(), unit="ms")
    tl = pd.Series(titles[inst["item"].to_numpy()]).str.lower()
    nd_o, nd_b = np.stack([M.ndcg_at(x, 10) for x in ranks["original"]]).mean(0), np.stack([M.ndcg_at(x, 10) for x in ranks["best"]]).mean(0)
    lrows = []
    for fam, pat in (("iPhone 13", r"iphone\s*13"), ("Galaxy S22", r"galaxy\s*s22"), ("Pixel 6", r"pixel\s*6")):
        m = tl.str.contains(pat, regex=True).to_numpy()
        for mo in pd.period_range("2021-08", "2022-07", freq="M"):
            k = m & (when.to_period("M") == mo)
            if k.sum() >= 150:
                lrows.append(dict(family=fam, month=str(mo), n=int(k.sum()), orig=average(nd_o[k], users[k]), best=average(nd_b[k], users[k])))
    pd.DataFrame(lrows).to_csv(OUT / "launches.csv", index=False)

    up = np.flatnonzero(lab.behaviour.to_numpy() == "upgrade")
    rng = np.random.default_rng(1)
    ex = None
    for i in rng.permutation(up)[:400]:
        rec, h = recency_top(i, 10)
        td = device(titles[int(inst["item"][i])])
        hd = set().union(*[device(titles[x]) for x in h if device(titles[x])])
        old = {(f, n) for f, _, n in hd} - {(f, n) for f, _, n in td}
        cls = []
        for x in rec:
            d = device(titles[x]); mm = {(f, n) for f, _, n in d} if d else set()
            cls.append("old phone" if mm & old else "new phone" if mm & {(f, n) for f, _, n in td} else "other")
        hfam = {f for f, _, _ in hd}; tfam = {f for f, _, _ in td}
        last4 = [device(titles[x]) for x in h[-4:]]
        clean = tfam == {"iphone"} and hfam == {"iphone"} and all(d is not None for d in last4)  # an iPhone-only user upgrading iPhones
        if clean and sum(c == "old phone" for c in cls) >= 6 and np.isinf(ranks["best"][0, i]):
            ex = {"inst": int(inst.inst[i]), "date": str(when[i].date()), "history": [titles[x][:80] for x in h[-4:]],
                  "bought": titles[int(inst["item"][i])][:80], "top10": [{"title": titles[x][:70], "kind": c} for x, c in zip(rec, cls)]}
            break
    (OUT / "upgrader_example.json").write_text(json.dumps(ex, indent=2))
    print("example", ex and ex["date"], ex and ex["bought"])

    pre = np.flatnonzero(tl.str.contains(r"iphone\s*13", regex=True).to_numpy() & (when < "2021-09-14"))
    early = [i for i in pre if (m := re.search(r"iphone\s*13", tl[i])) and m.start() < 45]  # model named up front, not in a long compatibility list
    fut = [{"date": str(when[i].date()), "title": titles[int(inst["item"][i])][:110]} for i in np.random.default_rng(0).choice(early, 5, replace=False)]
    (OUT / "future_title.json").write_text(json.dumps({"count": int(len(pre)), "named_up_front": int(len(early)), "examples": fut}, indent=2))
    print("future", fut[:2])


if __name__ == "__main__":
    main()
