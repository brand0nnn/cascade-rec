# Pool x freshness x context additivity (Val only, exploratory)

**Label: Val only, exploratory. This is an additivity test, NOT the selection of a final pipeline.** Reused Val
population (232,904 events / 124,416 users). No Test file is opened. Nothing is selected.

## Pre-stated hypothesis (written 2026-11-06 13:52 +0800, before any combo scoring or fitting)

Each factor keeps most of its gain when combined. The combined P3 + context + T1 ranker beats the P3 ranker (T0)
overall, and new-item quality recovers part of P3's new-item loss (context/newness helps new items).

## Registered arms
1. **P3 + T1 (no training):** the three locked P3 bce_text checkpoints
   (`results/local_experiments/20261106-pool-ranking-P3`) score Val with the nine common columns rebuilt from all
   reviews strictly before T1. This uses `freshness_val.SnapshotFeatures` with query-user exclusion, cut-rule and
   reference checks, and the P3 prefix scaler unchanged. Only natural-hit rows are rescored, on CPU; a CPU T0 rescore
   checks against the saved MPS ranks.
2. **P3 + context (training):** the `context_ranking` columns (63; period-start statistics T0 for B and T1 for Val;
   same leak checks) on the P3 pool with the P3 plans, split and common features. bce recipe, seeds 0/1/2. The shared
   initial state must equal the P3 fits, and the full initial state must equal P1 bce_context. Evaluated with
   (a) Val common columns at T0 (as trained) and (b) at T1 (all inputs at the period start).

Contrasts: paired user bootstrap (1,000, seed 0) on seed-mean event metrics. Metrics: nDCG@10, Recall@10, HR@3,
MRR@10; groups overall/new/warm. Each arm is compared with P3 bce_text (T0) and with P1 bce_text (T0); P3 + context
(T1) is also compared with P1 bce_context at consistent T1 (`results/analysis/20261106-context/followup/`).
Additivity table: the individual gains (pool, freshness, context-at-T1, each measured from P1 bce_text T0, or from
P1 bce_text T1 for context) are summed and set against the observed combined gain.

Code: `src/cascade/pipelines/pool_combo.py`. Bulk: `results/local_experiments/20261106-pool-combo-{T1,context}/`.
