# Fresh Two-tower (text) seeds 1/2 — Val only, exploratory robustness check

Written before any seed 1/2 training or metric computation (7 November 2026).

## Question

Tonight's fresh Two-tower (text) seed 0, retrained on all interactions before T1
(`results/retriever/local/20261106-two-tower-text-all-train-seed0`), reached full-Val macro
Recall@200 0.1113 versus 0.0687 for the frozen pre-T0 seed 0 (about +62% relative). Is that gain a
single-seed accident, or does it hold across seeds?

## Recipe (identical to seed 0; only the seed and run ID change)

- `src/cascade/pipelines/fresh_two_tower_seeds.py` reuses `fresh_two_tower.AllTrainView` and the
  same monkeypatching of `train_retriever.main`'s three `retriever_train` reads (training windows,
  static logQ counts, q audit) so that they cover retriever_train + ranker_train (all before T1).
- Config: `results/analysis/20261106-fresh-two-tower/two_tower_text_all_train.yaml` (unchanged).
  Text item tower, transformer user tower, tau 0.05, lr 1e-3, batch 256, max in-batch 4096, logQ,
  up to 50 epochs, patience 5, selection on macro Recall@100 over the fixed 20K Val subset
  (subset seed 0, the same subset for every training seed). MPS.
- Seeds 1 and 2, run IDs `local/20261106-two-tower-text-all-train-seed1` and `-seed2` (the seed 0
  naming series), i.e. `results/retriever/local/20261106-two-tower-text-all-train-seed{1,2}`.
  `train_retriever.main` refuses existing run directories; the two seeds run back-to-back on MPS
  from one detached chain (`run_chain.sh`), which writes `tables.md` when both finish.
- Evaluation: Val only (the training pipeline's own full-Val evaluation and `val_ranks.npz`).
  No pools, features or rankers. No Test file is opened.

## Comparison

Full-Val (232,904 events / 124,416 users) per-user macro Recall@50/100/200/500 and nDCG@10,
recomputed from each run's `val_ranks.npz` (full filtered catalogue), for groups overall,
new-in-Val (target first_ts >= T1 = 1628643414042) and warm. Fresh seeds 0/1/2 versus frozen
pre-T0 `results/retriever/two_tower_text/seed{0,1,2}` (their `val_ranks.npz` only; never
`test_ranks`). Mean +/- SD (ddof 1) across seeds; per-seed fresh - frozen differences (seed pairing
is nominal: different devices, code revisions and training data); paired user bootstrap (1,000 draws,
seed 0) on seed-mean event metrics for fresh-mean minus frozen-mean. Selected epochs.

## Pre-stated hypothesis

The ~+60% relative Recall@200 gain holds across seeds. Operationalised: every fresh seed's overall
full-Val Recall@200 exceeds every frozen seed's by at least +50% relative to the frozen seed mean,
and the SD of fresh-seed Recall@200 is < 0.003.

## Caveats (stated in advance)

- Val selected each fresh retriever's epoch, so Val is a reused, not untouched, population.
- Frozen seeds were trained on CUDA with an older code revision; fresh seeds on MPS with the current
  code. Device/code differences are confounded with the training-period change.
- Nothing is selected from this study.

## Outcome (appended automatically after computing; see tables.md)

- Per-fresh-seed R@200 gain relative to the frozen seed mean (0.068630): s0 +62.2%, s1 +61.5%.
- Fresh-seed R@200 SD: 0.000344 (threshold < 0.003).
- Pre-stated hypothesis (every seed >= +50% relative, SD < 0.003): SUPPORTED.

## Deviation: seed 2 incomplete (added after the chain)

The lead stopped fresh seed 2 at epoch 8 at 02:14 (disk at the 20 GB floor; seed 1 had already
replicated seed 0), so the chain's automatic compare never ran. The comparison was run manually with
`--fresh-seeds 0,1`: fresh statistics are over **two** seeds (SD ddof 1 with n = 2), frozen over three.
The tables.md title still says "seeds 0/1/2" (generic header). The partial directory
`results/retriever/local/20261106-two-tower-text-all-train-seed2` (no val_ranks, stub metrics.json,
epoch-8 checkpoint) is not a completed run and is not used. The hypothesis verdict above therefore
covers seeds 0 and 1 only.
