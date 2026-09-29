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
                  "| Model | train | seeds | cases | " + " | ".join(METRICS) + f" | {SEGMENT_METRIC} cold | tail | head | coverage@10 | top-1% share |",
                  "|---|---|---|---|" + "---|" * (len(METRICS) + 5)]
        for name, rs in sorted(runs.items(), key=lambda kv: (kv[0].split("_")[-1] != "AB", kv[0])):
            model, train_on = name.rsplit("_", 1)
            variants = [(model, name, [r[split] for r in rs])]
            # models that skip some cases (BPR: users unseen in training) also get a row on the full case set
            if rs[0][split]["n_unscorable"] > 0:
                variants.append((f"{model} (skipped = miss)", f"{name}_skipped_as_miss", [r[split]["with_unscorable_as_miss"] for r in rs]))
            for label, key, blocks in variants:
                row = {m: [b[m]["macro"] for b in blocks] for m in METRICS}
                has_extra = "segments" in blocks[0]
                entry = {"seeds": [r["seed"] for r in rs], "n_instances": blocks[0]["n_instances"],
                         **{m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in row.items()}}
                cells = ["–"] * 5
                if has_extra:
                    segs = {g: [b["segments"][f"item_{g}"][SEGMENT_METRIC]["macro"] for b in blocks] for g in ("cold", "tail", "head")}
                    cov = [b["coverage@10"] for b in blocks]
                    pop = [b["pop_concentration@10"] for b in blocks]
                    entry["segments_" + SEGMENT_METRIC] = {g: (float(np.nanmean(v)) if not np.isnan(v).all() else None) for g, v in segs.items()}
                    entry["n_unscorable"] = blocks[0]["n_unscorable"]
                    cells = [_fmt(segs[g]) if not np.isnan(segs[g]).all() else "–" for g in ("cold", "tail", "head")] + [_fmt(cov), _fmt(pop)]
                summary.setdefault(split, {})[key] = entry
                lines.append(f"| {label} | {train_on} | {len(rs)} | {blocks[0]['n_instances']:,} | " + " | ".join(_fmt(row[m]) for m in METRICS)
                             + " | " + " | ".join(cells) + " |")
    suffix = "_toy" if toy else ""
    (root / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    md = ("## Baselines (full ranking, filtered catalogue, our harness)\n\n"
          "`cases` = number of evaluated instances. BPR cannot score users with no training interactions; its first row "
          "skips those cases (CLAUDE.md §6), its `(skipped = miss)` row counts them as misses and is the one comparable "
          "to the other models.\n" + "\n".join(lines) + "\n")
    (root / f"summary{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    main(ap.parse_args().toy)
