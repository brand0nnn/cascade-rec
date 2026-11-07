# Fresh two-tower retriever (all data before T1) — Val only, exploratory

Written before any training or metric computation (6 November 2026).

## Question

Graph/support freshness helped every ranker. Does **retriever** freshness help too? The frozen
Two-tower (text) seed 0 (`results/retriever/two_tower_text/seed0`) was trained on the
retriever-training period only (before T0 = 26 Jul 2020). Here the same recipe is retrained on all
interactions before T1 (`all_train` = retriever_train + ranker_train) and used only to **serve**
fresher Val candidates to the frozen P2/P3 rankers.

## Registered recipe (unchanged except the training period)

- `cascade.pipelines.train_retriever.main` unchanged; the wrapper
  `src/cascade/pipelines/fresh_two_tower.py` maps the three internal `retriever_train` interaction
  reads (training windows, static logQ counts, q audit) to all pre-T1 interactions.
  logQ counts therefore come from the same (all pre-T1) period.
- Config `two_tower_text_all_train.yaml` (this folder): identical values to
  `configs/retriever/base.yaml` + `two_tower_text.yaml`: text item tower, transformer user tower,
  tau 0.05, lr 1e-3, batch 256, max in-batch 4096, logQ, up to 50 epochs, patience 5, selection on
  macro Recall@100 over the fixed 20K Val subset (exact search). Seed 0. MPS.
- Evaluation splits: Val only. No Test file is opened.

## Pools and rankers

- P2' = fresh two-tower natural top 200 (exact search, §4.6 filters, float32 scores).
- P3' = round-robin interleave of P1 (90-day recency) and P2', duplicates skipped, RRF k=60 skip
  score — identical rule (`pools_build.interleave_fuse`) to P3.
- Val features: same `BatchedGraphFeatures` builder (strictly pre-T0 graph), standardised with the
  **frozen P2 / P3 prefix scalers**. Rankers stay frozen:
  `20261106-pool-ranking-P3` (seeds 0/1/2) on P3', `20261106-pool-ranking-P2` on P2'.
- Check first: the same scoring path on the stored P3/P2 Val lists reproduces the stored ranks exactly.

## Metrics

Per-user macro over 232,904 Val events / 124,416 users, misses = 0; ranker values are seed means
(0/1/2); paired user bootstrap 1,000 draws, seed 0, on seed-mean event metrics (unadjusted).
Pool Recall@200 and retrieval-order nDCG@10; ranker nDCG@10 / Recall@10 / HR@3 / MRR@10;
groups overall / new-in-Val / warm, plus items first reviewed in [T0, T1).
Contrasts: P2'−P2 and P3'−P3 at pool and ranker level.

## Pre-stated hypothesis

The fresh retriever raises Val Recall@200 mainly for products first reviewed between T0 and T1
and for new-in-Val items, via a better text projection of the recent catalogue. The frozen P3
ranker on P3' gains overall over P3 (positive CI), with the largest relative gain on new items.

## Caveats (stated in advance)

- The P2/P3 rankers were trained on lists from the stale (pre-T0) retriever: P2'/P3' are a
  train/serve mismatch, not a retrained cascade.
- Val also selected the fresh retriever's epoch (the registered recipe), so Val is not untouched for
  the retriever; this is follow-up evidence on a reused population.
- The frozen seed 0 was trained on CUDA with an older code revision; the fresh fit is on MPS with
  the current code. Device/code differences are confounded with the period change. A single seed.
- Nothing is selected from this study.

## Outcome (added after computing; see tables.md)

- Fresh retriever (epoch 39/45): full-Val R@200 0.1113 vs frozen 0.0687 (new-in-Val 0.0315 vs 0.0261;
  warm 0.1478 vs 0.0874; first in [T0,T1) 0.0686 vs 0.0283; first before T0 0.2227 vs 0.1434).
- Hypothesis partly supported: the largest *relative* R@200 gain is for [T0,T1) products (x2.4), but the
  largest absolute gain is for products first seen before T0; new-in-Val gains are small (+0.0054).
- P3'−P3 ranker nDCG@10 +0.000365 [+0.000160, +0.000569] (positive CI), but new-in-Val −0.000835
  [−0.001014, −0.000630]: the opposite of the predicted largest gain on new items. The gain is warm/[T0,T1).
- P2'−P2 ranker nDCG@10 −0.000149 [−0.000369, +0.000081] (inconclusive); the frozen P2 ranker on P2' is no better
  than P2' retrieval order alone (−0.000103 [−0.000448, +0.000213]).
- Most of the pool-level gain (P3' interleave/RRF order +0.0022/+0.0024) is not carried through by the
  stale-trained rankers (train/serve mismatch). Nothing selected.
- Process note: a first feature-build attempt crashed on a logging bug after writing a partial file; that
  partial output was moved (not deleted) to `results/local_experiments/20261106-fresh-two-tower/features_failed_attempt1_P3f/`.
