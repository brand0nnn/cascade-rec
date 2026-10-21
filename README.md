# Retrieve, Then Rank — cascade recommender study (CP4285)

See `CLAUDE.md` for the full plan and `docs/decisions.md` for every design decision / deviation.

## Names used everywhere

Data: Amazon Reviews 2023, Cell Phones & Accessories (5-core), split by time:

| Name (in code) | Time range | Used for |
|---|---|---|
| retriever-training period (`retriever_train`) | before 26 Jul 2020 | training the retriever and the baselines |
| ranker-training period (`ranker_train`) | 26 Jul 2020 – 11 Aug 2021 | training the rankers (on the retriever's candidate lists) |
| all training data (`all_train`) | both of the above | the baselines' main setting |
| validation (`val`) | 11 Aug 2021 – 16 Jul 2022 | early stopping and every choice we make |
| test (`test`) | from 16 Jul 2022 | final numbers only |

| Name | Meaning |
|---|---|
| Two-tower (text) (`two_tower_text`) | Main retriever: each product is described by its text (title, categories, store) |
| Two-tower (text + ID) (`two_tower_text_id`) | Same, plus a learned vector per product seen in the retriever-training period |
| sample (`--sample`) | A fixed ~2% of users from the same data, for quick debugging; never reported |

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
```
`requirements.txt` pins exact versions: RecBole 1.2.1 needs numpy < 2 and pandas < 3, and pyarrow > 25 needs numpy >= 2.

## Phase 0 — data + baselines

```bash
scripts/prepare_data.sh --sample    # ~2% of users of the same data, for debugging (outputs in data/processed/sample)
scripts/prepare_data.sh             # download (+MANIFEST with SHA-256), preprocess, split, instances, stats
.venv/bin/python -m pytest -q       # must be green before any experiment
scripts/run_baselines.sh --sample   # Pop, ItemKNN, BPR, SASRec (RecBole) evaluated by our harness
scripts/run_baselines.sh            # full data, sequential   (add --device cuda on a GPU machine)
scripts/launch_full_baselines.sh    # full data, 4 parallel CPU queues (what was used locally)
.venv/bin/python -m cascade.eval.report   # -> results/baselines/summary.{md,json}
```

Outputs:
- `data/processed/`: `interactions.parquet`, `items.parquet`, `item_segments.parquet`, `instances.parquet`,
  `sequences.npz`, `split.json`, logs; `recbole/` holds the RecBole atomic-file export of our split.
- `results/stats/stats.json` + figures (dataset statistics, §4.8).
- `results/baselines/<model>/<all_train|retriever_train>/seed<k>/`: `metrics.json`, `config.yaml`, `git_hash.txt`, `recbole.log`,
  per-instance ranks (`val_ranks.npz`, `test_ranks.npz`) for bootstrap tests.

Single-run example:
```bash
.venv/bin/python -m cascade.pipelines.train_baseline --config configs/baselines/sasrec.yaml --seed 0 --train-on all_train [--sample] [--device cuda]
```

## RQ1 — retriever vs baseline retrieval

After the two-tower runs (text, text + ID) and the baseline runs on the retriever-training period (seeds 0–2), with their `{val,test}_ranks.npz` in `results/`:
```bash
.venv/bin/python -m cascade.eval.report_rq1   # -> results/rq1/summary.{md,json}
```
Recall@{10, 50, 100, 200, 500} and nDCG@10 (mean ± std over seeds), Recall@10/@200 by segment, and the
frozen-retriever check ((text + ID) − (text) on Val Recall@200, overall and new-in-period, paired bootstrap CI).

Val, all trained on the retriever-training period, per-user macro, mean over seeds 0–2 (full table: `results/rq1/summary.md`):

| Retriever | Recall@10 | Recall@200 | Recall@200, new-in-period products |
|---|---|---|---|
| Popularity | 0.0042 | 0.0270 | 0 |
| ItemKNN | 0.0012 | 0.0083 | 0 |
| BPR-MF | 0.0036 | 0.0273 | 0.0001 |
| SASRec | 0.0062 | 0.0402 | 0 |
| **Two-tower (text)** | 0.0105 | 0.0686 | **0.0263** |
| Two-tower (text + ID) | 0.0121 | 0.0724 | 0.0190 |

**Frozen retriever (21 Oct): Two-tower (text), seed 0**, `results/retriever/two_tower_text/seed0` (weights in git).
Text + ID is 5% better overall but 28% worse on new products; reasons in `docs/decisions.md`.

## Candidate lists for the rankers

Built once on the cluster from the frozen retriever (step 3 below), then shared from there: about 2 GB, too big
for git. Output: `data/processed/candidates/two_tower_text_seed0/`

| File | Contents | Used for |
|---|---|---|
| `ranker_train.npz` | 315,114 ranker-training cases × top 200 (+ the target as column 201 when missed) | ranker training |
| `val.npz`, `val_target_score.npz` | 232,904 Val cases × top 500, target never added | early stopping (target added back at load time) and Val results |
| `test.npz`, `test_target_score.npz` | 297,390 Test cases × top 500 | final evaluation only |
| `meta.json` | source run, git hash, recall per split, share of lists with an added target | checks |

```python
from cascade.pipelines.build_candidates import load_candidates
train = load_candidates(cand_dir, "ranker_train")              # labels: 1 = target, 0 = other candidates
val   = load_candidates(cand_dir, "val", k1=200, force=True)   # ranker-val lists (loss / GAUC / early stopping)
```

For debugging, the sample lists (~40 MB) are built with
`.venv/bin/python -m cascade.pipelines.build_candidates --run-dir results/retriever/two_tower_text_sample/seed0 --sample`
(needs the sample retriever, which is not in git). Two questions about the training labels are open; see
`docs/decisions.md` (2026-10-21) before training a ranker.

## Baseline results (Phase 0)

Test set, trained on all training data, per-user macro average, mean ± std over seeds 0–2 (Pop / ItemKNN are deterministic).
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

# 2. RQ1 runs (after job 1 finished): two-tower (text, text + ID) x seeds 0-2, and SASRec/BPR on the retriever-training period x seeds 0-2
sbatch scripts/slurm/train_retriever.sbatch
sbatch scripts/slurm/train_baselines_retriever_train.sbatch
squeue -u $USER                                      # monitor; one array task per run

# 3. candidate lists from the frozen retriever (Two-tower (text), seed 0)
sbatch --export=ALL,RUN_DIR=results/retriever/two_tower_text/seed0 scripts/slurm/build_candidates.sbatch
```

Or chain them: `jid=$(sbatch --parsable scripts/slurm/prepare_data.sbatch); sbatch --dependency=afterok:$jid scripts/slurm/train_retriever.sbatch`.

**Partitions** (checked 2026-10-15 with `sinfo`): the default `normal` partition has no GPUs, so every script names
one. Short jobs (data, candidates; < 3 h) use `gpu` with any GPU type. Training jobs use `gpu-long` (3-day limit) and
pin `a100-40`, so all seeds of a model run on the same GPU type. The account (`allusers`) and QoS (`normal`) are
defaults and need no flag.

**Getting results back:** metrics, configs, git hashes and per-case ranks under `results/` are tracked by git;
copy them to your machine with `rsync -avz --exclude 'checkpoint/' '<user>@xlogin.comp.nus.edu.sg:~/cascade-rec/results/' results/` and
commit them there. The trained two-tower weights (`results/retriever/*/seed*/model.pth`, ~87 MB) are in git; the
RecBole baseline checkpoints (BPR is 352 MB), job logs (`results/logs/`) and `data/` are not. Candidate lists stay on
the cluster.
