# Post-hoc Test check: frozen rankers on better candidate lists (9 Nov 2026)

**Status:** written and hash-locked before any Test file of this check is opened. Requested by the user on 9 Nov
2026 to address the limitation that the "better list, frozen ranker" evidence was Validation-only. This is a
**third, post-hoc look at Test** (after the locked runs of 7 and 8 Nov). It is reported as a follow-up check, not
as part of the locked runs. Nothing is trained, tuned or selected; every system is scored once; all results are
reported whatever they show.

Code: `src/cascade/pipelines/frozen_ranker_followup.py`. Bulk outputs:
`results/local_experiments/20261109-frozen-ranker-followup/`; tables: `results/analysis/20261109-frozen-ranker-followup/`.

## Question

Does a ranker that is **not retrained** turn a different, higher-recall candidate list into a better top 10?

## Frozen rankers (3 seeds each, weights already saved)

| Family | Ranker | Trained on | Inputs |
|---|---|---|---|
| A | `armC_mlp`: policy-study arm C MLP, 29 Oct (`results/ranker_policy/policy-{pilot,replication}-20261029/arm_C_seed*`) | natural two-tower top-200 lists | retrieval score + history/text matching |
| A | `din`: DIN natural, 28 Oct (`results/ranker_din/20261028-din-natural-seed*`) | natural two-tower top-200 lists | retrieval score + history/text matching |
| B | `p2ranker`: pool-ranking-P2 `bce_text` (score plus correction) | two-tower top-200 lists | text + period-start (T0) common inputs |
| B | `m1`: full-promising `bce_text` (B1) | recency top-200 lists | text + period-start (T0) common inputs |

## Lists (frozen retrievers; same builders as Val, each reproduced exactly on Val first)

- **tt200 / tt500:** frozen two-tower (text) seed 0, top 200 / top 500 (nested prefixes of one list).
- **fresh:** two-tower (text) retrained on all data before T1 (`results/retriever/local/20261106-two-tower-text-all-train-seed0`), top 200.
- **mixed:** the locked Test P3 lists (recency + two-tower round-robin). Family A receives each candidate's two-tower
  score (the score it was trained on); family B (m1) receives the recency score, as in the Val P1onP3 study.
- Family B inputs: common features from reviews before T0, each study's own scaler, exactly as in its Val study.

## Comparisons (frozen ranker on the better list minus the same ranker on the list it was trained on)

| Cell | Ranker | Better list | Reference |
|---|---|---|---|
| A-K | armC_mlp, din | tt500 | tt200 |
| A-F | armC_mlp, din | fresh | tt200 |
| A-M | armC_mlp, din | mixed | tt200 |
| B-F | p2ranker | fresh (P2f) | P2 |
| B-M | m1 | mixed (P3) | P1 (locked S1 ranks) |

Also reported: retrieval-order nDCG on P2 vs P2f; m1 on P3 vs the ranker retrained on P3 (locked S6 ranks).

Metrics: nDCG@10 per-user macro over all 297,390 Test events (misses = 0), seed means; groups all / new (target
first review ≥ T2) / warm. Paired user bootstrap on seed-mean event values, 1,000 draws, seed 0, unadjusted.
Purchases in list: share of events whose target is in the list.

## Validation results these predictions are based on (nDCG@10 ×10⁻³, seed mean)

| Ranker | Own list | Better list | Change [95% CI] | Purchases in list |
|---|---|---|---|---|
| armC_mlp | tt200 8.63 | tt500 8.07 | −0.56 [−0.69, −0.42] | 7.2% → 12.3% |
| din | tt200 9.82 | tt500 9.73 | −0.09 [−0.22, +0.07] | 7.2% → 12.3% |
| armC_mlp | tt200 8.63 | fresh 8.00 | −0.63 [−0.87, −0.39] | 7.2% → 11.8% |
| din | tt200 9.82 | fresh 9.18 | −0.64 [−0.91, −0.37] | 7.2% → 11.8% |
| armC_mlp | tt200 8.63 | mixed 8.82 | +0.19 [+0.00, +0.38] | 7.2% → 8.8% |
| din | tt200 9.82 | mixed 9.62 | −0.20 [−0.39, −0.02] | 7.2% → 8.8% |
| p2ranker | P2 10.08 | P2f 9.93 | −0.15 [−0.37, +0.08]; new −0.88 [−1.17, −0.57] | 6.9% → 11.1% (macro) |
| m1 | P1 8.14 | P3 8.81 | +0.67; new 8.67 → 5.35 | 5.5% → 8.0% (macro) |

## Written predictions for Test (verdicts are mechanical)

**U. Little or no gain (all eight cells):** the relative change in overall nDCG@10 is less than one third of the
relative change in purchases in the list (point estimates; the one-third threshold is fixed here in advance).

**Directional (only where the Val interval excluded zero):**

| # | Prediction | Rule |
|---|---|---|
| D1 | armC_mlp: tt500 − tt200 < 0 (all) | CI upper < 0 |
| D2 | armC_mlp: fresh − tt200 < 0 (all) | CI upper < 0 |
| D3 | din: fresh − tt200 < 0 (all) | CI upper < 0 |
| D4 | din: mixed − tt200 < 0 (all) | CI upper < 0 |
| D5 | p2ranker: P2f − P2 < 0 (new) | CI upper < 0 |
| D6 | p2ranker gain from P2 to P2f is smaller than the retrieval-order gain (all) | CI upper < 0 |
| D7 | m1: P3 − P1 < 0 (new) | CI upper < 0 |
| D8 | the ranker retrained on P3 (S6) beats m1 frozen on P3 (all) | CI lower > 0 |

Verdicts: **held** if the rule is met; **not held** if the interval lies entirely on the other side of zero;
otherwise **inconclusive**. No prediction is changed after any Test result is seen.

## Known limits of this check

- Post-hoc third look at Test; unadjusted intervals; eight directional predictions.
- The fresher retriever's epoch was selected on Val; on Test it is untuned.
- Family A cells D1–D4 and the U criterion for family A are new on Val today (9 Nov) and were written with those
  Val results in view; family B cells reproduce the poster's Val tests.
- S1 ranks (locked 7 Nov) were scored on CPU; m1 on P3 is scored on MPS, the device of its Val study (CPU flips
  one near-tied Val row out of ~18,600). This affects only exact ties.
