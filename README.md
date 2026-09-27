# Retrieve, Then Rank — cascade recommender study (CP4285)

See `CLAUDE.md` for the full plan and `docs/decisions.md` for every design decision / deviation.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
```
RecBole 1.2.1 needs numpy < 2 and pandas < 3 (pinned in `requirements.txt`).

## Phase 0 — data + baselines

```bash
scripts/prepare_data.sh --toy       # ~2% of users, for debugging (outputs in data/processed/toy)
scripts/prepare_data.sh             # download (+MANIFEST with SHA-256), preprocess, split, instances, stats
.venv/bin/python -m pytest -q       # must be green before any experiment
scripts/run_baselines.sh --toy      # Pop, ItemKNN, BPR, SASRec (RecBole) evaluated by our harness
scripts/run_baselines.sh            # full data, sequential   (add --device cuda on a GPU machine)
scripts/launch_full_baselines.sh    # full data, 4 parallel CPU queues (what was used locally)
.venv/bin/python -m cascade.eval.report   # -> results/baselines/summary.{md,json}
```

Outputs:
- `data/processed/`: `interactions.parquet`, `items.parquet`, `item_segments.parquet`, `instances.parquet`,
  `sequences.npz`, `split.json`, logs; `recbole/` holds the RecBole atomic-file export of our split.
- `results/stats/stats.json` + figures (dataset statistics, §4.8).
- `results/baselines/<Model>_<AB|A>/seed<k>/`: `metrics.json`, `config.yaml`, `git_hash.txt`, `recbole.log`,
  per-instance ranks (`val_ranks.npz`, `test_ranks.npz`) for bootstrap tests.

Single-run example:
```bash
.venv/bin/python -m cascade.pipelines.train_baseline --config configs/baselines/sasrec.yaml --seed 0 --train-on AB [--toy] [--device cuda]
```
