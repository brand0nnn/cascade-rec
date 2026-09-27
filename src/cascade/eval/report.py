"""Aggregate baseline runs into a table: mean ± std over seeds, per-user macro (CLAUDE.md §6 acceptance).

    python -m cascade.eval.report [--toy]
Writes results/baselines/summary[_toy].{json,md}.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from cascade.utils.config import REPO_ROOT

METRICS = ["recall@10", "ndcg@10", "recall@100"]
SEGMENT_METRIC = "recall@100"


def _fmt(vals: list[float]) -> str:
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"


def main(toy: bool) -> None:
    root = REPO_ROOT / "results" / "baselines"
    runs = defaultdict(list)
    for f in sorted(root.glob("*/seed*/metrics.json")):
        name = f.parent.parent.name
        if name.endswith("_toy") != toy:
            continue
        runs[name.removesuffix("_toy")].append(json.loads(f.read_text()))
    summary, lines = {}, []
    for split in ("val", "test"):
        lines += [f"\n### {split.capitalize()} (per-user macro; mean ± std over seeds)\n",
                  "| Model | train | seeds | " + " | ".join(METRICS) + f" | {SEGMENT_METRIC} cold | tail | head | coverage@10 | top-1% share |",
                  "|---|---|---|" + "---|" * (len(METRICS) + 5)]
        for name, rs in sorted(runs.items(), key=lambda kv: (kv[0].split("_")[-1] != "AB", kv[0])):
            model, train_on = name.rsplit("_", 1)
            row = {m: [r[split][m]["macro"] for r in rs] for m in METRICS}
            segs = {s: [r[split]["segments"][f"item_{s}"][SEGMENT_METRIC]["macro"] for r in rs] for s in ("cold", "tail", "head")}
            cov = [r[split]["coverage@10"] for r in rs]
            pop = [r[split]["pop_concentration@10"] for r in rs]
            summary.setdefault(split, {})[name] = {"seeds": [r["seed"] for r in rs], **{m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in row.items()},
                                                   "segments_" + SEGMENT_METRIC: {s: (float(np.nanmean(v)) if not np.isnan(v).all() else None) for s, v in segs.items()},
                                                   "n_unscorable": rs[0][split]["n_unscorable"], "n_instances": rs[0][split]["n_instances"]}
            lines.append(f"| {model} | {train_on} | {len(rs)} | " + " | ".join(_fmt(row[m]) for m in METRICS)
                         + " | " + " | ".join(_fmt(segs[s]) if not np.isnan(segs[s]).all() else "–" for s in ("cold", "tail", "head"))
                         + f" | {_fmt(cov)} | {_fmt(pop)} |")
    suffix = "_toy" if toy else ""
    (root / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    md = "## Baselines (full ranking, filtered catalogue, our harness)\n" + "\n".join(lines) + "\n"
    (root / f"summary{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    main(ap.parse_args().toy)
