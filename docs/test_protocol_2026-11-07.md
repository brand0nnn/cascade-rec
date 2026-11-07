# Locked Test protocol (DRAFT, 7 Nov 2026). Needs team approval before any Test file is opened

**Status:** draft. Nothing in this protocol has been run. Test (events at or after T2 = 16 Jul 2022) has not been
opened for any system trained on 5–7 Nov. Once approved, this file is frozen; its hash is recorded before scoring.

## Purpose

Confirm, once and on untouched data, the Val findings of 5–7 Nov (`docs/overnight_results_2026-11-06.md`,
`docs/experiment_overview.md`). Test is used only to report. No model, setting or system is chosen or changed
because of a Test result.

## Frozen models (no training on Test; 3 seeds each, weights already saved)

| ID | Val row | Lists | Inputs | Loss | Training statistics | Study folder (`results/local_experiments/`) |
|---|---|---|---|---|---|---|
| M1 | B1 | P1 recency | text | BCE | T0 | `20261103-full-promising-ranking` |
| M2 | B9 | P1 recency | text | BCE | monthly | `20261106-rolling-ranking` |
| M3 | B10 | P1 recency | text + co-review | BCE | monthly | `20261106-rolling-ranking` |
| M4 | D1 | P3 mixed | text | BCE | T0 | `20261106-pool-ranking-P3` |
| M5 | D6 | P3 mixed | text | BCE | monthly | `20261107-pool-rolling` |
| M6 | D7 | P3 mixed | text + co-review | BCE | monthly | `20261107-pool-rolling` |

Weight hashes are checked against each study's `metrics.json` before scoring.

## Test inputs to build (frozen retrievers, no training)

- **Candidate lists for all Test events:** P1 (recency, training-free), P2 (old two-tower, trained on A only, same
  checkpoint as the Val lists), P3 (interleave of P1 and P2, RRF score). The builders are the same as for Val.
- **Product statistics** (graph and support features), all computed only from reviews strictly before each query:
  - **T0:** the team's original setting, for comparability.
  - **T1:** as in the team's 4 Nov Test run, for comparability.
  - **T2:** the period-start rule (§4.5), the Test equivalent of T1 on Val. **Needs approval (decision 2).**
  - **Monthly:** reviews before the first day of each Test query's month. **Needs approval (decision 3).**
- Model inputs keep each study's own scaler, which is fitted on period B. Nothing is refitted on Test.

## Systems scored (each once, 3 seeds, seed-mean)

| System | Model | Test lists | Test statistics |
|---|---|---|---|
| S0 | none (P1 recency order) | P1 | — |
| S1 | M1 | P1 | T0 (current pipeline as the team ran it originally) |
| S2 | M1 | P1 | T1 (as in the team's 4 Nov Test run) |
| S3 | M1 | P1 | T2 (period-start rule) |
| S4 | M2 | P1 | monthly |
| S5 | M3 | P1 | monthly |
| S6 | M4 | P3 | T0 |
| S7 | M5 | P3 | monthly |
| S8 | M6 | P3 | monthly (best Val system) |

## Metrics and uncertainty (unchanged from Val)

- nDCG@10, per-user macro, over the full Test population with misses counted as 0. Secondary: Recall@10, HR@3,
  MRR@10, Recall@200.
- Groups: all, new-in-period (target first review ≥ T2, which equals cold on Test), warm.
- Paired user bootstrap on seed-mean event metrics: 1,000 draws, seed 0, 95% percentile interval.

## Primary contrasts (fixed in advance; all others are secondary)

| # | Contrast | Question |
|---|---|---|
| C1 | S3 − S1 | Do period-start statistics help frozen models on Test? |
| C2 | S6 − S1 | Does the mixed pool help when everything else is equal? |
| C3 | S5 − S4 | Does co-review help when statistics are monthly in both training and scoring? |
| C4 | S8 − S1 | Combined system vs the current pipeline |

## Predictions (written before Test is opened)

1. **C1 > 0** overall. A smaller version was already seen on Test with T1 (+0.0009 to +0.0014), and T2 should be at
   least as good as T1 (S3 ≥ S2).
2. **C2:** positive on warm targets and negative on new targets. Overall it is **smaller than on Val (+0.0029)** and
   may be near zero or negative, because 44% of Test targets are new against 18% on Val.
3. **C3 > 0** overall, concentrated on warm targets.
4. **C4 > 0** on warm targets. Overall smaller than on Val (+0.0049); new-target change ≤ 0.
5. **Not predicted:** whether S8 is the best system overall on Test.

## Rules

- Each system is scored exactly once. Results are reported whatever their direction, including failed predictions.
- No retraining, tuning, threshold, blend or system choice uses Test.
- A bug found after scoring is fixed only if it is a technical error (for example misaligned events). The fix, the
  reason and both results are reported.
- Leak guards: every statistic comes from reviews strictly before its query; candidate filtering (§4.6) is unchanged;
  `test_predictions_loaded` is recorded in every output.
- Outputs: `results/test_final/20261107/` (tables, provenance, hashes); bulk data under
  `results/local_experiments/20261107-test-final/`.

## Known limits that Test cannot remove

- Product titles name devices released after the review date (≥ 8.5–21% of top-10 slots on Val).
- The 5-core filter was computed over all time, which makes "no reviews yet" look more predictive.
- Monthly statistics depart from the agreed §4.5 period-start rule (they never use future data). They are reported
  as an exploratory variant.

## Decisions needed from the team

1. Approve a one-time Test run of the systems above.
2. Approve **T2 statistics**. The 4 Nov authorisation ruled out a T2 graph. Without approval, drop S3 and use S2 for C1.
3. Approve **monthly statistics on Test** (S4, S5, S7, S8). Without approval, C3 and C4 cannot be tested.

## Estimated effort

About 4 hours after approval: code and checks about 2 h, input building about 1 h, scoring and report about 1 h.
