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

## Baseline results (Phase 0)

Test set, trained on periods A+B, per-user macro average, mean ± std over seeds 0–2 (Pop / ItemKNN are deterministic).
Full table incl. Val and segments: `results/baselines/summary.md`.

| Model | nDCG@10 | Recall@10 | Recall@100 |
|---|---|---|---|
| Popularity | 0.0010 | 0.0022 | 0.0166 |
| ItemKNN | 0.0008 | 0.0014 | 0.0052 |
| BPR-MF | 0.0013 ± 0.0001 | 0.0027 ± 0.0002 | 0.0154 ± 0.0003 |
| **SASRec** | **0.0026 ± 0.0003** | **0.0053 ± 0.0006** | **0.0248 ± 0.0011** |

- Evaluated on 297,390 Test cases (Test reviews with ≥ 1 earlier review by the same user); full ranking over all
  111,480 products, excluding products the user already reviewed and products not yet released at that time.
- BPR counts the 74,437 cases from users with no training interactions as misses, so all rows use the same cases.
- 44% of Test targets are products with no reviews before the Test period; all four ID-based baselines score ≈ 0 on them.
