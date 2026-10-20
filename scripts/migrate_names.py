"""One-off migration for the 2026-10-20 renaming (see docs/decisions.md). Run once on every machine after `git pull`:

    .venv/bin/python scripts/migrate_names.py

Old → new names:
  period A / B / A∪B            → retriever_train / ranker_train / all_train
  v1..v5 retriever variants      → two_tower_text_meanpool, two_tower_text, two_tower_text_no_logq,
                                   two_tower_text_id, two_tower_id_only
  toy (2% user debug sample)     → sample
  results/baselines/SASRec_A     → results/baselines/sasrec/retriever_train   (same for pop/itemknn/bpr, AB → all_train)

What it does (safe to re-run; only moves or renames, never deletes data):
  1. data/processed/toy → data/processed/sample
  2. renames keys inside the processed data's split.json, instances_log.json and preprocess_log.json
  3. renames candidate-list folders and B.npz → ranker_train.npz
  4. moves files git does not track (model.pth, RecBole checkpoints) from old result folders to the new ones
     (git pull already moved the tracked files)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVERS = {"v1_meanpool": "two_tower_text_meanpool", "v2_transformer": "two_tower_text", "v3_no_logq": "two_tower_text_no_logq",
              "v4_id_text": "two_tower_text_id", "v5_id_only": "two_tower_id_only"}
BASELINES = {"Pop": "popularity", "ItemKNN": "itemknn", "BPR": "bpr", "SASRec": "sasrec"}
TRAIN_ON = {"AB": "all_train", "A": "retriever_train"}
PERIODS = {"A": "retriever_train", "B": "ranker_train"}


def merge_move(old: Path, new: Path) -> None:
    """Move every file under ``old`` into ``new`` (keeping relative paths, never overwriting), then drop empty dirs."""
    if not old.exists():
        return
    for f in sorted(p for p in old.rglob("*") if p.is_file()):
        target = new / f.relative_to(old)
        if target.exists():
            print(f"  keep existing {target.relative_to(ROOT)} (left {f.relative_to(ROOT)} in place)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(target))
    for d in sorted((p for p in old.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [old]:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    print(f"moved {old.relative_to(ROOT)} → {new.relative_to(ROOT)}")


def rename_keys(path: Path, mapping: dict[str, str]) -> None:
    """Rename top-level and nested dict keys of a JSON file."""
    if not path.exists():
        return
    def fix(o):
        if isinstance(o, dict):
            return {mapping.get(k, k): fix(v) for k, v in o.items()}
        return o
    old = json.loads(path.read_text())
    new = fix(old)
    if new != old:
        path.write_text(json.dumps(new, indent=2))
        print(f"updated keys in {path.relative_to(ROOT)}")


def main() -> None:
    processed = ROOT / "data" / "processed"
    # 1. debug sample folder
    if (processed / "toy").exists():
        if (processed / "sample").exists():
            sys.exit("both data/processed/toy and data/processed/sample exist; resolve by hand")
        (processed / "toy").rename(processed / "sample")
        print("renamed data/processed/toy → data/processed/sample")
    for d in (processed, processed / "sample"):
        # 2. keys inside the processed data
        rename_keys(d / "split.json", {"t_ab": "t_ranker_train", "period_b_fraction": "ranker_train_fraction", **PERIODS})
        rename_keys(d / "instances_log.json", PERIODS)
        rename_keys(d / "preprocess_log.json", {"toy": "sample", "toy_users": "sample_users"})
        # 3. candidate lists
        for old, new in RETRIEVERS.items():
            for c in sorted((d / "candidates").glob(f"{old}_seed*")):
                merge_move(c, d / "candidates" / c.name.replace(old, new, 1))
        for c in sorted((d / "candidates").glob("*_seed*")):
            if (c / "B.npz").exists() and not (c / "ranker_train.npz").exists():
                (c / "B.npz").rename(c / "ranker_train.npz")
                print(f"renamed {c.relative_to(ROOT)}/B.npz → ranker_train.npz")
            meta = c / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                if m.get("retriever") in RETRIEVERS:
                    m["retriever"] = RETRIEVERS[m["retriever"]]
                    meta.write_text(json.dumps(m, indent=2))
    # 4. result folders (untracked leftovers after git pull; everything when run before committing the rename)
    res = ROOT / "results"
    for old, new in RETRIEVERS.items():
        merge_move(res / "retriever" / old, res / "retriever" / new)
        merge_move(res / "retriever" / f"{old}_toy", res / "retriever" / f"{new}_sample")
    for model, new_model in BASELINES.items():
        for tr, new_tr in TRAIN_ON.items():
            merge_move(res / "baselines" / f"{model}_{tr}", res / "baselines" / new_model / new_tr)
            merge_move(res / "baselines" / f"{model}_{tr}_toy", res / "baselines" / new_model / f"{new_tr}_sample")
    merge_move(res / "stats_toy", res / "stats_sample")
    print("done")


if __name__ == "__main__":
    main()
