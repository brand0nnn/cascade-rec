# Graph-feature freshness on Val (Val only, exploratory)

## Pre-stated hypothesis (written 2026-11-05T17:51:42Z, before any feature was rebuilt or scored)

T1 refresh +0.0008 to +0.0014 overall on Val, largest on new; monthly adds a further, smaller gain.

(Source of the expectation: the team's post-Test-exposure T1-vs-T0 graph refresh on Test, results/steps_restoration/20261104.
Only its protocol and source code were read here; none of its Test outputs.)

Note on what was read: besides the protocol and source files of results/steps_restoration/20261104, the file listing
and the event count/timing fields of its graph_completion.json were seen while locating code. No Test metric output was opened.

## Design (Val only, exploratory; nothing selected)

- Frozen checkpoints: bce_text, pairwise_text, pairwise_coreview (results/local_experiments/20261103-full-promising-ranking)
  and bce_coreview (results/local_experiments/20261104-objective-coreview), seeds 0/1/2: twelve checkpoints, CPU scoring.
- Fixed: K200 Val candidates, histories, prefix scalers (fitted on T0-feature ranker-train prefix slots), post-standardisation masks.
- Arms rebuild all nine common and six graph inputs. The builder is `SnapshotFeatures` in
  src/cascade/analysis/freshness_val.py. It uses the same batched algebra as the T0 builder, so the query user is excluded the same way.
  - **T0**: the saved parent features.
  - **T1**: all reviews strictly before T1. That is 2,208,880 edges, versus 1,877,548 for T0.
  - **monthly**: all reviews strictly before the query's UTC month start, giving twelve snapshots from 2021-08-01 to 2022-07-01.
    The August 2021 snapshot (1 Aug) precedes T1 (11 Aug).
- Only the 14,916 natural K200 hit rows are re-featurised and rescored. Every miss keeps rank infinity in every arm.
- Metrics:
  - Population: all 232,904 events / 124,416 users.
  - Aggregation: seed-mean event metric, then per-user macro.
  - Intervals: paired user bootstrap, 1,000 draws, seed 0.
  - Groups: new = target first_ts ≥ T1; warm = otherwise.

## Checks

- **Reproduction (bce_text seed 0, existing T0 features, CPU):** all 232,904 ranks equal the saved evaluation exactly.
  The new builder at the T0 cutoff also reproduces the saved standardised T0 features exactly (256 rows, max abs error 0).
- **T0 CPU controls for all twelve checkpoints:** 7 of 2.79M event ranks differ from the saved MPS ranks. Each is a ±1 shift at ranks 19–137,
  so no @10 metric changes. All contrasts use the CPU-scored T0 arm.
- **Per snapshot:**
  - Latest review is before the earliest query in its group.
  - No query user has a snapshot edge to any of its candidates or its target.
  - Every query-user snapshot edge lies in that user's strict history.
  - The batched build matches the explicit-exclusion reference builder (max abs error ≤3e-8).
  - The non-graph common columns 3–6 are unchanged.
- **Tests:** tests/test_freshness_val.py (5 toy tests) passes.

## nDCG@10 (seed-mean, user macro)

| Method | T0 all | T1 all | monthly all | T0 new | T1 new | monthly new | T0 warm | T1 warm | monthly warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bce_text | 0.008141 | 0.009432 | 0.008477 | 0.008666 | 0.011721 | 0.006181 | 0.007887 | 0.008231 | 0.009570 |
| pairwise_text | 0.007787 | 0.009684 | 0.008735 | 0.008754 | 0.013146 | 0.007061 | 0.007298 | 0.007841 | 0.009475 |
| pairwise_coreview | 0.007735 | 0.010221 | 0.009804 | 0.008575 | 0.014365 | 0.009033 | 0.007293 | 0.008041 | 0.010022 |
| bce_coreview | 0.008167 | 0.009801 | 0.009640 | 0.008250 | 0.011132 | 0.007294 | 0.008105 | 0.009055 | 0.010693 |

## nDCG@10 contrasts [95% paired-user CI]

| Method | Group | T1 − T0 | monthly − T1 | monthly − T0 |
|---|---|---:|---:|---:|
| bce_text | all | +0.001291 [0.001124, 0.001459] | −0.000955 [−0.001097, −0.000825] | +0.000336 [0.000133, 0.000539] |
| bce_text | new | +0.003056 [0.002810, 0.003287] | −0.005540 [−0.005925, −0.005159] | −0.002484 [−0.002845, −0.002143] |
| bce_text | warm | +0.000345 [0.000125, 0.000581] | +0.001339 [0.001241, 0.001439] | +0.001683 [0.001449, 0.001946] |
| pairwise_text | all | +0.001897 [0.001683, 0.002095] | −0.000949 [−0.001119, −0.000795] | +0.000948 [0.000727, 0.001172] |
| pairwise_text | new | +0.004392 [0.004120, 0.004695] | −0.006085 [−0.006539, −0.005651] | −0.001693 [−0.002072, −0.001329] |
| pairwise_text | warm | +0.000543 [0.000309, 0.000814] | +0.001634 [0.001514, 0.001754] | +0.002177 [0.001920, 0.002457] |
| pairwise_coreview | all | +0.002486 [0.002288, 0.002698] | −0.000416 [−0.000586, −0.000248] | +0.002070 [0.001835, 0.002310] |
| pairwise_coreview | new | +0.005790 [0.005454, 0.006114] | −0.005332 [−0.005798, −0.004901] | +0.000458 [0.000082, 0.000842] |
| pairwise_coreview | warm | +0.000748 [0.000498, 0.001014] | +0.001981 [0.001837, 0.002116] | +0.002729 [0.002450, 0.003015] |
| bce_coreview | all | +0.001634 [0.001432, 0.001822] | −0.000161 [−0.000319, −0.000024] | +0.001473 [0.001251, 0.001679] |
| bce_coreview | new | +0.002882 [0.002653, 0.003105] | −0.003838 [−0.004214, −0.003458] | −0.000956 [−0.001306, −0.000592] |
| bce_coreview | warm | +0.000950 [0.000712, 0.001215] | +0.001639 [0.001513, 0.001769] | +0.002589 [0.002325, 0.002878] |

Co-review − text, nDCG@10, all events:

| Objective | T0 | T1 | monthly |
|---|---:|---:|---:|
| BCE | +0.000026 [−0.000093, 0.000151] | +0.000369 [0.000229, 0.000520] | +0.001164 [0.001004, 0.001321] |
| Pairwise | −0.000052 [−0.000180, 0.000075] | +0.000537 [0.000384, 0.000702] | +0.001070 [0.000895, 0.001229] |

Under T1, the co-review new-item contrast is −0.000589 for BCE and +0.001219 for pairwise. Under monthly it is +0.001113 for BCE and +0.001972 for pairwise.
Recall@10, HR@3 and MRR@10 follow the same signs. One exception: monthly − T1 for bce_coreview is inconclusive on HR@3 and MRR@10.
See contrasts.csv.

Sensitivity check: dropping August 2021 (sensitivity_excluding_august.csv) leaves monthly − T1 essentially unchanged, for example bce_text −0.001023 overall and −0.005610 on new.
The pre-T1 August snapshot does not explain the monthly loss.

## Target graph support and feature shifts

Share of targets with candidate_graph_known, over all Val events:

| Snapshot | All | New | Warm |
|---|---:|---:|---:|
| T0 | 0.347 | 0.000 | 0.517 |
| T1 | 0.670 | 0.000 | 1.000 |
| monthly | 0.934 | 0.802 | 0.999 |

Standardised shifts versus T0, on natural-hit rows (feature_shifts.csv):
- **T1:** the candidate_graph_known z-score rises by +0.87 for non-target candidates. It rises by +1.55 for warm targets and by 0 for new targets.
  History_graph_fraction rises by +1.03. Co-review path features rise by +0.1 for non-targets and by +0.4–0.67 for warm targets.
- **monthly:** new targets become known (+2.30 z) and acquire co-review paths (+0.65 to +1.35 z).
  Many inputs move more than one prefix-scaler standard deviation away from the T0 training distribution.

## Plain-words finding

- **T1 refresh:** freshness holds on Val with correct sequencing. T1 refresh improves every method and every metric.
  - Gains are largest on new targets (+0.0029 to +0.0058 nDCG@10).
  - Overall gains are +0.0013 to +0.0025, so the hypothesis range of +0.0008 to +0.0014 is met only by bce_text; the others exceed it.
- **Monthly refresh:** hypothesis rejected. It does not add a further gain: it is worse than T1 overall for all four methods.
  - On new targets it loses heavily (−0.0038 to −0.0061).
  - On warm targets it gains clearly (+0.0013 to +0.0020).
- **Interpretation:** this pattern is consistent with the frozen rankers using "no graph history" (degree 0) as a recency cue learned on T0 features.
  - Under T1, new Val products stay unknown while competitors become known, so the cue sharpens.
  - Monthly makes most new products known, so the cue disappears, while fresher co-occurrence helps warm targets.
  - This is an inference from the feature/support pattern. No attribution experiment tested it.
- **Co-review:** the co-review advantage over text grows with snapshot freshness under both objectives.

## Caveats

- Scalers and checkpoints were fitted on T0-era features. Fresher snapshots move inputs off that training distribution (shifts of +1 to +2 z), so these are frozen-model inference substitutions, not retrained freshness effects.
- The new/warm effects partly reflect how the knownness cue interacts with the snapshot; they are not a clean measure of information gain.
- Val has been used before (the reused population includes pilot users), so this is not untouched holdout evidence. The intervals are unadjusted, conditional on these checkpoints and candidates, and treat seeds as fixed.
- Monthly snapshots use Val reviews from earlier months as features for later Val queries. That is legitimate sequencing, but it is a different information regime from a single frozen T1 snapshot.
- Nothing is selected or promoted. No Test file was opened, and no training was performed.

## Files

- metrics.csv, contrasts.csv, feature_shifts.csv, target_support.csv, sensitivity_excluding_august.csv
- reproduce.json, t0_controls.json, snapshots.json, config.yaml, git_hash.txt
- Bulk data (ranks and standardised features for the hit rows, 366 MB): results/local_experiments/20261106-freshness-val/
- Logs: results/logs/20261106/freshness-val-*.log
- Code: src/cascade/analysis/freshness_val.py. Tests: tests/test_freshness_val.py.
