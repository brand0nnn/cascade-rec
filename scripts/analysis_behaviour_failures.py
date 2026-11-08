"""User-behaviour taxonomy and failure breakdown (Validation).

Each purchase is labelled from product titles (current metadata; approximate) by which phone it is for,
compared with the phones in the user's last 50 prior purchases:
  device       family + model parsed from title: iPhone N / X / XR / XS / SE, Galaxy S/Note/A/Z N, Pixel N
  behaviour    same device       target device appears in history (more accessories for a phone they have)
               upgrade           same family, newer model than any in history
               older model       same family, older model than any in history (second phone / family member?)
               other device, same platform   a different family on the same platform
               switch platform   target device's platform absent from history
               device-agnostic   target names no device (cables, chargers, mounts...)
               unknown history   history names no device
  multi-device history  history names >= 2 distinct devices (households / several phones)
Failure breakdown per group and system: retrieval miss (not in 200), ranking miss (in 200, rank > 10), hit (top 10).
Systems: original (MLP, two-tower lists) and best (DIN, mixed lists + combined score), seed 0..2 averaged.
Also: how many prior purchases the user made for the target device, and days since last purchase.

    .venv/bin/python scripts/analysis_behaviour_failures.py results/targeted_study/behaviour-failures-20261108
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


def nd(r):
    return np.stack([M.ndcg_at(x, 10) for x in r]).mean(0)

ROOT = Path(__file__).resolve().parents[1]
SYS = {"orig": ("mlp_tt", "hybrid-20261107-s{}"), "best": ("din_rrf", "rrf-20261108-s{}")}
PAT = [("iphone", "apple", re.compile(r"iphone\s*(1[0-5]|[4-9]|xr|xs|x|se)\b")),
       ("galaxy s", "android", re.compile(r"galaxy\s*s\s?(\d{1,2})\b")),
       ("galaxy note", "android", re.compile(r"note\s?(\d{1,2})\b")),
       ("galaxy a", "android", re.compile(r"galaxy\s*a\s?(\d{1,3})\b")),
       ("galaxy z", "android", re.compile(r"z\s?(?:fold|flip)\s?(\d)\b")),
       ("pixel", "android", re.compile(r"pixel\s*(\d)a?\b"))]
NUM = {"x": 10, "xr": 10.2, "xs": 10.5, "se": -1}


def device(title: str):
    """All (family, platform, model number) a title mentions; None if none. Titles often list several models."""
    t = title.lower()
    found = set()
    for fam, plat, p in PAT:
        for g in p.findall(t):
            n = NUM.get(g, None)
            found.add((fam, plat, float(g) if n is None else n))
    return frozenset(found) or None


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    ranks, cov, ref = {}, {}, None
    for k, (arm, run) in SYS.items():
        r = []
        for s in (0, 1, 2):
            z = np.load(ROOT / "results/hybrid_study" / run.format(s) / f"{arm}_seed{s}" / "ranks.npz")
            ref = z if ref is None else ref
            assert np.array_equal(z["inst"], ref["inst"])
            r.append(z["ranks"])
        ranks[k] = np.stack(r)
        cov[k] = z["target_pos"] >= 0
    inst = data.instances.set_index("inst").loc[ref["inst"]].reset_index()
    users = ref["user"]
    titles = data.items.set_index("item").title.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    dev = [device(t) for t in titles]
    hist = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    beh, multi, nsame = [], [], []
    for i in range(len(inst)):
        hitems = [dev[x] for x in hist[i] if x > 0 and dev[x] is not None]
        hd = set().union(*hitems) if hitems else set()
        td = dev[int(inst["item"][i])]
        # multi-device: two history items that share no model (a title listing several models counts once)
        multi.append(any(not (a & b) for j, a in enumerate(hitems) for b in hitems[j + 1:]))
        nsame.append(sum(bool(td & h) for h in hitems) if td else 0)
        if td is None:
            beh.append("device-agnostic"); continue
        if not hd:
            beh.append("unknown history"); continue
        if td & hd:
            beh.append("same device"); continue
        tplat, hplat = {p for _, p, _ in td}, {p for _, p, _ in hd}
        if not (tplat & hplat):
            beh.append("switch platform"); continue
        fams = {f for f, _, _ in td} & {f for f, _, _ in hd}
        if fams:
            tn = [n for f, _, n in td if f in fams]; hn = [n for f, _, n in hd if f in fams]
            beh.append("upgrade" if min(tn) > max(hn) else "older model" if max(tn) < min(hn) else "other model, same family")
        else:
            beh.append("other device, same platform")
    lab = pd.DataFrame({"inst": ref["inst"], "behaviour": beh, "multi_device_history": multi, "prior_same_device": nsame})
    lab.to_parquet(out / "labels.parquet", index=False)
    lab["same_device_count"] = pd.cut(lab.prior_same_device, [-1, 0, 1, 2, 4, 100], labels=["0", "1", "2", "3-4", "5+"])
    rows = []
    for cut, col in (("behaviour", lab.behaviour.to_numpy()), ("multi-device history", lab.multi_device_history.map({True: "2+ devices", False: "0-1 device"}).to_numpy()),
                     ("prior purchases for this device", lab.same_device_count.astype(str).to_numpy())):
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
            row.update(diff=ci["diff"], lo=ci["ci_low"], hi=ci["ci_high"])
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "behaviour_failures.csv", index=False)
    pd.set_option("display.width", 260)
    show = res[["cut", "group", "purchases", "share", "orig_retrieval_miss", "best_retrieval_miss", "orig_ranking_miss", "best_ranking_miss",
                "orig_hit10", "best_hit10", "orig_ndcg", "best_ndcg", "lo", "hi"]].copy()
    for c in show.columns[3:]:
        show[c] = show[c] * (1000 if c in ("orig_ndcg", "best_ndcg", "lo", "hi") else 100)
    print(show.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
