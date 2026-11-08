"""Product-side archetypes and failure breakdown (Validation), the mirror of the user-behaviour analysis.

Labels are descriptive only (never ranker inputs):
  product age at purchase   days since the product's first review: <30, 30-180, 180-365, 1-2 y, >2 y
  compatibility breadth     phone models named in the title: 0 (device-agnostic), 1, 2-3, 4+
  price                     parsed price: <$10, $10-25, $25-50, >$50, missing
  store size                products listed by the same store in the catalogue: 1-9, 10-99, 100-499, 500+
  same store as history     the store appears among the user's last 50 purchases
Failure split per system as before: retrieval miss / ranking miss / hit@10; original vs best, seeds averaged.

    .venv/bin/python scripts/analysis_product_archetypes.py results/targeted_study/product-archetypes-20261108
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average
sys.path.insert(0, str(Path(__file__).parent))
from analysis_behaviour_failures import SYS, device, nd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    ranks, ref = {}, None
    for k, (arm, run) in SYS.items():
        r = []
        for s in (0, 1, 2):
            z = np.load(ROOT / "results/hybrid_study" / run.format(s) / f"{arm}_seed{s}" / "ranks.npz")
            ref = z if ref is None else ref
            r.append(z["ranks"])
        ranks[k] = np.stack(r)
    inst = data.instances.set_index("inst").loc[ref["inst"]].reset_index()
    users = ref["user"]
    it = data.items.set_index("item")
    tgt = it.loc[inst["item"].to_numpy()]
    age = (inst.ts.to_numpy() - tgt.first_ts.to_numpy()) / 86_400_000
    age_l = pd.cut(age, [-1, 30, 180, 365, 730, 1e6], labels=["<30 days", "30-180 days", "180-365 days", "1-2 years", ">2 years"]).astype(str)
    nmod = np.array([len({(f, n) for f, _, n in d}) if (d := device(t)) else 0 for t in tgt.title.fillna("")])
    breadth = np.select([nmod == 0, nmod == 1, nmod <= 3], ["0 (device-agnostic)", "1 model", "2-3 models"], "4+ models")
    price = tgt.price.to_numpy(dtype=float)
    price_l = np.where(np.isnan(price), "missing", pd.cut(np.nan_to_num(price, nan=-1), [-2, 10, 25, 50, 1e9], labels=["<$10", "$10-25", "$25-50", ">$50"]).astype(str))
    store_n = it.store.fillna("").map(it.store.fillna("").value_counts())
    sn = store_n.reindex(inst["item"].to_numpy()).to_numpy()
    sn = np.where(tgt.store.fillna("").to_numpy() == "", 0, sn)
    store_l = pd.cut(sn, [-1, 0, 9, 99, 499, 1e9], labels=["no store", "1-9", "10-99", "100-499", "500+"]).astype(str)
    store_arr = it.store.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    hist = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    ts_store = store_arr[inst["item"].to_numpy()]
    same = np.array([bool(s) and s in set(store_arr[h[h > 0]]) for s, h in zip(ts_store, hist)])
    cuts = {"product age at purchase": age_l, "compatibility breadth": breadth, "price": price_l,
            "store size": store_l, "store vs history": np.where(same, "same store as history", "new store")}
    rows = []
    for cut, col in cuts.items():
        for g in pd.unique(col):
            m = col == g
            if m.sum() < 300:
                continue
            row = dict(cut=cut, group=g, purchases=int(m.sum()), share=float(m.mean()))
            for k in SYS:
                rk = ranks[k][:, m]
                row[f"{k}_retrieval_miss"] = float(np.isinf(rk).mean())
                row[f"{k}_ranking_miss"] = float((np.isfinite(rk) & (rk > 10)).mean())
                row[f"{k}_hit10"] = float((rk <= 10).mean())
                row[f"{k}_ndcg"] = average(nd(rk), users[m])
            ci = paired_bootstrap(users[m], nd(ranks["orig"][:, m]), nd(ranks["best"][:, m]), n_resamples=1000, seed=0)
            row.update(lo=ci["ci_low"], hi=ci["ci_high"])
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "product_archetypes.csv", index=False)
    pd.set_option("display.width", 260)
    show = res[["cut", "group", "share", "orig_retrieval_miss", "best_retrieval_miss", "orig_hit10", "best_hit10", "orig_ndcg", "best_ndcg", "lo", "hi"]].copy()
    for c in show.columns[2:]:
        show[c] = show[c] * (1000 if c in ("orig_ndcg", "best_ndcg", "lo", "hi") else 100)
    print(show.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
