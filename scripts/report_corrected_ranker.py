"""Report the four matching-family systems from saved Test ranks; no model execution.

Uses original two-tower controls and corrected mixed-pool ranks. Requires the
original processed item/instance tables for subgroup labels. Outputs are fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.utils.config import REPO_ROOT

ORIGINAL = REPO_ROOT / "results/test_final_ranker/20261108"
CORRECTED = REPO_ROOT / "results/test_final_ranker/20261108-normalization-corrected"
SYSTEMS = ("mlp_tt", "din_tt", "mlp_rrf", "din_rrf")
PAIRS = (("mlp_tt", "din_tt"), ("mlp_rrf", "din_rrf"), ("mlp_tt", "mlp_rrf"),
         ("din_tt", "din_rrf"), ("mlp_tt", "din_rrf"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="fresh directory")
    args = ap.parse_args()
    out = REPO_ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    sources: dict[str, str] = {}

    def track(path: Path) -> Path:
        sources[str(path.relative_to(REPO_ROOT))] = digest(path)
        return path

    audit = json.loads(track(CORRECTED / "audit.json").read_text())
    if audit["status"] != "complete" or len(audit["replays"]) != 12:
        raise ValueError("completed correction audit required")
    if not all(r["val_ranks_exact"] and r["old_test_ranks_exact"] for r in audit["replays"]):
        raise ValueError("correction replay checks did not pass")
    values, ids, users = {}, None, None
    for system in SYSTEMS:
        root = ORIGINAL if system.endswith("_tt") else CORRECTED
        seeds = []
        for seed in (0, 1, 2):
            with np.load(track(root / f"{system}_seed{seed}_test_ranks.npz")) as z:
                if ids is None:
                    ids, users = z["inst"].copy(), z["user"].copy()
                np.testing.assert_array_equal(ids, z["inst"])
                np.testing.assert_array_equal(users, z["user"])
                seeds.append(M.ndcg_at(z["ranks"], 10))
        values[system] = np.mean(seeds, axis=0)
    if len(ids) != 297390 or len(np.unique(users)) != 112081:
        raise ValueError("unexpected Test cohort")

    data = REPO_ROOT / "data/processed"
    inst = pd.read_parquet(track(data / "instances.parquet")).set_index("inst").loc[ids]
    items = pd.read_parquet(track(data / "items.parquet")).set_index("item").loc[inst.item.to_numpy()]
    segment = pd.read_parquet(track(data / "item_segments.parquet")).set_index("item").loc[inst.item.to_numpy()]
    split = json.loads(track(data / "split.json").read_text())
    np.testing.assert_array_equal(users, inst.user.to_numpy())
    if not (inst.period.to_numpy() == 3).all():
        raise ValueError("non-Test instance")
    new = items.first_ts.to_numpy() >= split["t2"]
    head = segment.segment.astype(str).to_numpy() == "head"
    masks = {"all": np.ones(len(ids), bool), "new-in-period": new,
             "warm-head": ~new & head, "warm-tail": ~new & ~head}
    previous = {r.system: r for r in pd.read_csv(track(ORIGINAL / "levels.csv")).itertuples()
                if r.group == "all"}
    corrected = {r.system: r for r in pd.read_csv(track(CORRECTED / "levels.csv")).itertuples()
                 if r.group == "all"}
    levels, contrasts = [], []
    for group, mask in masks.items():
        for system, value in values.items():
            level = M.macro_micro(value[mask], users[mask])["macro"]
            if group == "all":
                expected = (previous if system.endswith("_tt") else corrected)[system].ndcg10
                if abs(level - expected) > 1e-12:
                    raise ValueError(f"saved level not reproduced: {system}")
            levels.append({"system": system, "group": group, "ndcg10": level})
        for a, b in PAIRS:
            ci = paired_bootstrap(users[mask], values[a][mask], values[b][mask], n_resamples=1000, seed=0)
            baseline = M.macro_micro(values[a][mask], users[mask])["macro"]
            contrasts.append({"a": a, "b": b, "group": group, **ci, "relative": ci["diff"] / baseline})
    pd.DataFrame(levels).to_csv(out / "levels.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "comparisons.csv", index=False)
    manifest = {"status": "complete", "evaluation_split": "test", "events": len(ids),
                "users": len(np.unique(users)), "model_execution": False,
                "source_sha256": digest(Path(__file__)), "inputs": sources,
                "outputs": {name: digest(out / name) for name in ("levels.csv", "comparisons.csv")}}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(levels)} levels and {len(contrasts)} paired contrasts to {out}")


if __name__ == "__main__":
    main()
