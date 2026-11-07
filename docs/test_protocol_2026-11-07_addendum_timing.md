# Addendum to the locked Test protocol: statistics-timing replication (7 Nov 2026)

**Approved by Brandon on 7 Nov 2026, before this file was hashed.** It extends `docs/test_protocol_2026-11-07.md`. It changes no system, model or result already reported. Its only purpose is to replicate on Test the Val timing diagnostics in `results/analysis/20261107-statistics-timing/`.

## Setup

- **Models:** the four frozen P1 rankers trained with T0 statistics (B1 bce_text, B4 pairwise_text, B5 pairwise_coreview, B6 bce_coreview), seeds 0/1/2. No training.
- **Lists:** the Test P1 recency lists already built (`results/local_experiments/20261107-test-final/candidates/P1/test`), natural-hit rows only. Misses keep rank infinity.
- **Statistics:** same builder, cut rule and parent T0 scaler as Val. T2 = 16 Jul 2022 is the Test period start.
- **Arms:**
  - **Column swap T0 -> T2:**
    - X1: known flag only.
    - X2: known flag + log degree.
    - X3: all graph-derived common columns.
    - X6: all T2 values, only for candidates first reviewed before T0.
    - X7: all T2 values, only for candidates first reviewed in [T0, T2).
  - **Fixed cut date:** T2 − k months, k ∈ {3, 6, 9, 12, 18}, plus T0 and T2.
  - **Fixed lag:** reviews before (query month start − L months), L ∈ {0, 1, 3, 6, 9, 12}.
- **Metrics:** nDCG@10, per-user macro, seed means. Groups: all, new (target first review ≥ T2), warm. Intervals: 1,000 paired-user draws, seed 0.

## Predictions (written before any Test arm is built)

- **R1a.** X1 alone recovers at least a third of the T0 -> T2 gain on new targets, for all four models.
- **R1b.** X7 (products released in [T0, T2)) recovers at least 75% of the overall T0 -> T2 gain, and X6 recovers less than a third, for all four models.
- **R2a.** Overall nDCG@10 rises monotonically from T0 through T2 − 18, −12, −9, −6, −3 months to T2, for all four models.
- **R2b.**
  - For every model, new-target nDCG@10 is lowest at lag 0 among the six lags.
  - It peaks at a lag in {3, 6, 9}.
  - Lag 12 is below that peak.
- **R2c.** Warm-target nDCG@10 is higher at lag 0 than at lag 12, for every model.

## Rules

- Each arm is scored once.
- Every prediction is reported as held, not held or inconclusive.
- Nothing is selected, and no earlier result is changed.
