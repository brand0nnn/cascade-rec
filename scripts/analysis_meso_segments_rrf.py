"""New meso cuts for the final systems: product type, phone platform, user-vs-product platform, model upgrades.

Systems (seed-averaged Val per-purchase nDCG@10, per-user macro, paired user bootstrap 1,000):
  mlp_tt   original: MLP on two-tower lists        din_tt   DIN on two-tower lists
  mlp_rrf  MLP on mixed lists + combined score     din_rrf  DIN on mixed lists + combined score (best)
Also coverage = purchase is in the 200 candidates (two-tower list vs mixed list).

Labels come from titles/categories only (descriptive, never ranker inputs):
  product type   from leaf category (case, screen protector, charger/cable, watch band, mount, other)
  platform       title mentions iPhone/iPad/Apple/AirPods -> apple; Galaxy/Samsung/Pixel/Moto/LG/OnePlus/Android -> android
  user platform  majority platform of the last 50 prior products (ties/none -> unknown)
  iPhone model   number in "iPhone 11/12/13/..." titles; upgrade = target's model newer than every model in history
Validation only; no Test.

    .venv/bin/python scripts/analysis_meso_segments_rrf.py results/targeted_study/meso-segments-rrf-20261108
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average

ROOT = Path(__file__).resolve().parents[1]
SYS = {"mlp_tt": "hybrid-20261107-s{}", "din_tt": "hybrid-20261107-s{}",
       "mlp_rrf": "rrf-20261108-s{}", "din_rrf": "rrf-20261108-s{}"}
APPLE = re.compile(r"\b(iphone|ipad|apple|airpods?|magsafe)\b", re.I)
ANDROID = re.compile(r"\b(galaxy|samsung|pixel|moto|motorola|lg|oneplus|android|xiaomi|redmi|huawei)\b", re.I)
IPHONE = re.compile(r"iphone\s*(1[1-5]|[6-9]|x[sr]?|se)\b", re.I)
ORDER = {"6": 6, "7": 7, "8": 8, "x": 10, "xr": 10.5, "xs": 10.6, "se": 0, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15}


def ptype(leaf: str) -> str:
    s = leaf.lower()
    for key, name in (("screen protector", "screen protector"), ("cases, holsters", "case"), ("charg", "charger/cable"),
                      ("cable", "charger/cable"), ("smartwatch", "watch accessory"), ("automobile", "car mount/charger"),
                      ("cell phones & accessories > cell phones", "phone")):
        if key in s:
            return name
    return "other"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    vals, cov = {}, {}
    ref = None
    for name, run in SYS.items():
        seeds = []
        for s in (0, 1, 2):
            z = np.load(ROOT / "results/hybrid_study" / run.format(s) / f"{name}_seed{s}" / "ranks.npz")
            ref = ref if ref is not None else z
            assert np.array_equal(z["inst"], ref["inst"])
            seeds.append(M.ndcg_at(z["ranks"], 10))
            cov[name.split("_")[1]] = z["target_pos"] >= 0
        vals[name] = np.mean(seeds, 0)
    inst = data.instances.set_index("inst").loc[ref["inst"]].reset_index()
    users = ref["user"]
    it = data.items.set_index("item")
    titles = it.title.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    leaf = it.leaf_cat.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    plat = np.array(["apple" if APPLE.search(t) and not ANDROID.search(t) else "android" if ANDROID.search(t) and not APPLE.search(t)
                     else "none" for t in titles])
    model = np.array([max([ORDER.get(m.lower(), -1) for m in IPHONE.findall(t)], default=-1) for t in titles], float)
    types = np.array([ptype(x) for x in leaf])
    hist = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    tgt = inst["item"].to_numpy()
    hp = plat[hist]; hp[hist == 0] = "none"
    na, nd = (hp == "apple").sum(1), (hp == "android").sum(1)
    uplat = np.where(na > nd, "apple user", np.where(nd > na, "android user", "unknown"))
    tplat = plat[tgt]
    match = np.where(uplat == "unknown", "unknown user", np.where(tplat == "none", "no-platform product",
                     np.where(np.char.add(tplat.astype(str), " user") == uplat, "same platform", "switch platform")))
    hm = np.where(hist > 0, model[hist], -1).max(1)
    upgrade = np.where((model[tgt] > 0) & (hm > 0), np.where(model[tgt] > hm, "newer iPhone model", np.where(model[tgt] == hm, "same iPhone model", "older iPhone model")), "n/a")
    week = pd.to_datetime(inst.ts.to_numpy(), unit="ms")
    launch = np.where((week >= "2021-09-14") & (week < "2021-10-15"), "iPhone 13 launch month", "other weeks")
    cuts = {"product type": types[tgt], "product platform": tplat, "user platform": uplat, "user vs product": match,
            "iPhone upgrade": upgrade, "period": launch}
    rows = []
    for cut, lab in cuts.items():
        for g in pd.unique(lab):
            m = lab == g
            if m.sum() < 500:
                continue
            row = dict(cut=cut, group=g, events=int(m.sum()), share=float(m.mean()),
                       cover_tt=average(cov["tt"][m].astype(float), users[m]), cover_mixed=average(cov["rrf"][m].astype(float), users[m]))
            for n, v in vals.items():
                row[n] = average(v[m], users[m])
            ci = paired_bootstrap(users[m], vals["mlp_tt"][m], vals["din_rrf"][m], n_resamples=1000, seed=0)
            row.update(best_vs_orig=ci["diff"], lo=ci["ci_low"], hi=ci["ci_high"])
            ci = paired_bootstrap(users[m], vals["mlp_rrf"][m], vals["din_rrf"][m], n_resamples=1000, seed=0)
            row.update(din_vs_mlp=ci["diff"], dlo=ci["ci_low"], dhi=ci["ci_high"])
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "segments.csv", index=False)
    pd.DataFrame({"inst": ref["inst"], **cuts}).to_parquet(out / "labels.parquet", index=False)
    pd.set_option("display.width", 250)
    show = res.copy()
    for c in ["mlp_tt", "din_tt", "mlp_rrf", "din_rrf", "best_vs_orig", "lo", "hi", "din_vs_mlp", "dlo", "dhi"]:
        show[c] = show[c] * 1000
    print(show.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
