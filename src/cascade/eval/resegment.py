"""Compute the new-in-period segment for runs saved before it existed, from their per-instance ranks.

    python -m cascade.eval.resegment results/retriever/two_tower_text/seed0 [more run dirs...]

Writes ``extra_segments.json`` next to each run's metrics.json (metrics.json itself is left untouched).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.segments import new_in_period

CUTOFFS = (10, 50, 100, 200, 500)


def resegment(run_dir: Path, data: ProcessedData) -> dict:
    first_ts = data.items.set_index("item")["first_ts"]
    inst_items = data.instances.set_index("inst")["item"]
    out = {}
    for split in ("val", "test"):
        f = run_dir / f"{split}_ranks.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ok = z["scorable"]
        is_new = new_in_period(first_ts.loc[inst_items.loc[z["inst"]].values].values, split, data.split)
        out[split] = {}
        for name, m in (("item_new_in_period", ok & is_new), ("item_not_new", ok & ~is_new)):
            block = {"n_instances": int(m.sum())}
            for c in CUTOFFS:
                block[f"recall@{c}"] = M.macro_micro(M.recall_at(z["rank"][m], c), z["user"][m])
                block[f"ndcg@{c}"] = M.macro_micro(M.ndcg_at(z["rank"][m], c), z["user"][m])
            out[split][name] = block
    (run_dir / "extra_segments.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    data = ProcessedData.load()
    for d in a.run_dirs:
        r = resegment(Path(d), data)
        print(d, {s: {k: round(v["recall@100"]["macro"], 4) for k, v in r[s].items()} for s in r})
