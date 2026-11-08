"""Test the upgrade hypothesis: for purchases labelled 'upgrade', what phones are the 200 candidates for?

For a fixed random sample of upgrade purchases (RNG 0), rebuild the recency retriever's top 200 (same recipe as
build_p3_candidates.py) and load the two-tower top 200 (saved Val lists). Each candidate is classified by the
models its title mentions: 'old phone' (shares a model with the user's history), 'target's new model' (shares a
model with the purchased product), both, other device, or device-agnostic. Also: is the bought product's model
present at all, and the share of the catalogue (released by then) that is for the new model.
Validation only; labels approximate (titles are current metadata).

    .venv/bin/python scripts/analysis_upgrader_candidates.py results/targeted_study/behaviour-failures-20261108
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cascade.data.instances import ProcessedData
from cascade.pipelines.build_candidates import load_candidates
sys.path.insert(0, str(Path(__file__).parent))
from analysis_behaviour_failures import device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = 86_400_000.0


def main(d: Path) -> None:
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    lab = pd.read_parquet(d / "labels.parquet")
    tt = load_candidates(ROOT / "data/processed/candidates/two_tower_text_seed0", "val", k1=200)
    pos = pd.Series(np.arange(len(tt["inst"])), index=tt["inst"])
    E = np.load(data.dir / "item_text_emb.npy").astype(np.float32)
    titles = data.items.set_index("item").title.fillna("").reindex(np.arange(data.n_items + 1)).fillna("").to_numpy()
    dev = [device(t) for t in titles]
    first_ts = np.full(data.n_items + 1, np.iinfo(np.int64).max); first_ts[data.items["item"].to_numpy()] = data.items.first_ts.to_numpy()
    inst = data.instances.set_index("inst")
    rows = []
    for group in ("upgrade", "same device"):
        sample = np.random.default_rng(0).choice(lab.inst[lab.behaviour == group].to_numpy(), 2000, replace=False)
        for i in sample:
            r = inst.loc[i]; u, hl, t, tgt = int(r.user), int(r.hist_len), int(r.ts), int(r["item"])
            st = data.user_ptr[u]; full = data.seq_item[st: st + hl]; fts = data.seq_ts[st: st + hl]
            h, ht = full[-50:], fts[-50:]
            w = np.exp2(-(t - ht) / DAY / 90).astype(np.float32)
            q = (w[:, None] * E[h]).sum(0); q /= max(np.linalg.norm(q), 1e-12)
            s = E @ q; s[0] = -np.inf; s[first_ts > t] = -np.inf; s[full] = -np.inf
            top = np.argpartition(-s, 200)[:200]
            hd = set().union(*[dev[x] for x in h if dev[x]]) if any(dev[x] for x in h) else set()
            td = dev[tgt]
            new_models = {(f, n) for f, _, n in td}; old_models = {(f, n) for f, _, n in hd} - new_models

            def cls(x):
                m = {(f, n) for f, _, n in dev[x]} if dev[x] else set()
                if not m: return "device-agnostic"
                a, b = bool(m & old_models), bool(m & new_models)
                return "both" if a and b else "old phone" if a else "new model" if b else "other device"
            for src, cand in (("recency", top), ("two-tower", tt["cand"][pos[i]])):
                c = pd.Series([cls(x) for x in cand if x > 0]).value_counts(normalize=True)
                rows.append(dict(group=group, source=src, inst=int(i), **c.to_dict()))
            released = np.flatnonzero(first_ts <= t)
            # share of released catalogue mentioning the new model vs old phone (sampled for speed)
            sub = np.random.default_rng(int(i)).choice(released, 3000, replace=False)
            cc = pd.Series([cls(x) for x in sub]).value_counts(normalize=True)
            rows.append(dict(group=group, source="catalogue (released)", inst=int(i), **cc.to_dict()))
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(d / "upgrader_candidates.csv", index=False)
    cols = ["old phone", "new model", "both", "other device", "device-agnostic"]
    print((df.groupby(["group", "source"])[cols].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
