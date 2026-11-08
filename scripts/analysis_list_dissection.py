"""Dissect what the retriever list (200) and the ranker's top 10 contain, by purchase type (Validation sample).

Input: results/hybrid_study/top10-export-20261108/{original,best}.npz (same 20,000 random Val purchases).
Each candidate is classified against the purchase and the user's last 50 purchases, by the phone models titles name:
  bought phone   names a model of the purchased product
  owned phone    names a model in the user's history (and not the bought one)
  other phone    names some other phone model
  no phone       names no phone model (cables, chargers, mounts...)
and by product type (leaf category: case, screen protector, charger/cable, ...) vs the purchase's type.
Purchase types = behaviour labels from analysis_behaviour_failures.py.
Per system x purchase type: list and top-10 composition, share of lists whose top 10 holds >= 1 bought-phone
product / >= 1 same-type product, and purchase found (in 200) / in top 10.
Labels approximate (titles are current metadata). Validation only.

    .venv/bin/python scripts/analysis_list_dissection.py results/targeted_study/list-dissection-20261108
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
sys.path.insert(0, str(Path(__file__).parent))
from analysis_behaviour_failures import device  # noqa: E402
from analysis_meso_segments_rrf import ptype  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results/hybrid_study/top10-export-20261108"
CLS = ["bought phone", "owned phone", "other phone", "no phone"]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    titles = data.items.set_index("item").title.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    leaf = data.items.set_index("item").leaf_cat.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    models = [({(f, n) for f, _, n in d} if (d := device(t)) else set()) for t in titles]
    types = np.array([ptype(x) for x in leaf])
    lab = pd.read_parquet(ROOT / "results/targeted_study/behaviour-failures-20261108/labels.parquet").set_index("inst")
    rows, rec = [], []
    for sysname in ("original", "best"):
        z = np.load(EXP / f"{sysname}.npz")
        inst = data.instances.set_index("inst").loc[z["inst"]]
        hist = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
        tgt = inst["item"].to_numpy()
        beh = lab.loc[z["inst"]].behaviour.to_numpy()
        for i in range(len(tgt)):
            tm = models[tgt[i]]; hm = set().union(*[models[x] for x in hist[i] if x > 0]) - tm
            def cls(x):
                m = models[x]
                return "no phone" if not m else "bought phone" if m & tm else "owned phone" if m & hm else "other phone"
            cand = z["cand"][i]; cand = cand[cand > 0]; top = z["top10"][i]
            lc = pd.Series([cls(x) for x in cand]).value_counts(normalize=True)
            tc = [cls(x) for x in top]
            rec.append(dict(system=sysname, behaviour=beh[i], target_specific=bool(tm),
                            **{f"list_{c}": lc.get(c, 0.0) for c in CLS},
                            **{f"top_{c}": tc.count(c) / 10 for c in CLS},
                            top_has_bought_phone=("bought phone" in tc), top_same_type=float(np.mean(types[top] == types[tgt[i]])),
                            list_same_type=float(np.mean(types[cand] == types[tgt[i]])),
                            found=bool(z["target_pos"][i] >= 0), hit10=bool(tgt[i] in set(top.tolist()))))
    df = pd.DataFrame(rec)
    df.to_parquet(out / "per_purchase.parquet", index=False)
    agg = df.groupby(["behaviour", "system"]).mean(numeric_only=True)
    agg["n"] = df.groupby(["behaviour", "system"]).size()
    agg = agg.reset_index()
    agg.to_csv(out / "dissection.csv", index=False)
    pd.set_option("display.width", 250)
    show = agg.copy(); num = [c for c in show.columns if c not in ("behaviour", "system", "n", "target_specific")]
    show[num] = (show[num] * 100).round(1)
    print(show[["behaviour", "system", "n"] + [f"list_{c}" for c in CLS] + [f"top_{c}" for c in CLS] +
               ["top_has_bought_phone", "list_same_type", "top_same_type", "found", "hit10"]].to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
