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

## RQ1 — retriever vs baseline retrieval

After the retriever (v2, v4) and A-only baseline runs (seeds 0–2), with their `{val,test}_ranks.npz` in `results/`:
```bash
.venv/bin/python -m cascade.eval.report_rq1   # -> results/rq1/summary.{md,json}
```
Recall@{10, 50, 100, 200, 500} and nDCG@10 (mean ± std over seeds), Recall@10/@200 by segment, and the
frozen-retriever check (v4 − v2 on Val Recall@200, overall and new-in-period, paired bootstrap CI).

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

## Running on the NUS SoC compute cluster (Slurm)

Submit every job **from the repository root** (job logs go to `results/logs/slurm-*.out`).

```bash
# once, on the login node
git clone https://github.com/brand0nnn/cascade-rec.git && cd cascade-rec
bash scripts/slurm/setup_env.sh                      # uv + Python 3.11 venv + requirements
# torch cannot be imported on the login node (memory cap), so test the venv on a GPU node:
srun -p gpu --gres=gpu:1 -c 2 --mem=8G -t 00:10:00 .venv/bin/python -c "import torch; print(torch.cuda.is_available())"

# 1. data (download ~0.9 GB, preprocess, encode item texts, run tests)
sbatch scripts/slurm/prepare_data.sbatch

# 2. RQ1 runs (after job 1 finished): retriever v2/v4 x seeds 0-2, and A-only SASRec/BPR x seeds 0-2
sbatch scripts/slurm/train_retriever.sbatch
sbatch scripts/slurm/train_baselines_a.sbatch
squeue -u $USER                                      # monitor; one array task per run

# 3. after choosing the frozen retriever (record it in docs/decisions.md)
sbatch --export=ALL,RUN_DIR=results/retriever/v2_transformer/seed0 scripts/slurm/build_candidates.sbatch
```

Or chain them: `jid=$(sbatch --parsable scripts/slurm/prepare_data.sbatch); sbatch --dependency=afterok:$jid scripts/slurm/train_retriever.sbatch`.

**Partitions** (checked 2026-10-15 with `sinfo`): the default `normal` partition has no GPUs, so every script names
one. Short jobs (data, candidates; < 3 h) use `gpu` with any GPU type. Training jobs use `gpu-long` (3-day limit) and
pin `a100-40`, so all seeds of a model run on the same GPU type. The account (`allusers`) and QoS (`normal`) are
defaults and need no flag.

**Getting results back:** metrics, configs, git hashes and per-case ranks under `results/` are tracked by git, so
commit and push them from the cluster. Model weights (`model.pth`) and `data/` are not in git; build the candidate
lists on the cluster, where the weights are.
