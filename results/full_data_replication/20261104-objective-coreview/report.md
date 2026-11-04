# Training objective x additional co-review evidence

Exploratory completion of a previously inspected validation factorial. Three new BCE+co-review fits reuse the exact parent candidates/features/plans and nine existing locked control fits. All cells retain basic metadata matching and nine common support/degree inputs. Only four additional co-review columns are switched. No Test predictions or graph retrieval experiment.

| Objective | Additional co-review | Overall nDCG@10 | Recall@10 | New | Warm |
|---|---|---:|---:|---:|---:|
| BCE | Off | 0.008115 | 0.015399 | 0.008697 | 0.007846 |
| BCE | On | 0.008409 | 0.015845 | 0.008342 | 0.008384 |
| Pairwise | Off | 0.007767 | 0.014865 | 0.009152 | 0.007103 |
| Pairwise | On | 0.008065 | 0.015290 | 0.008873 | 0.007571 |

Interaction = graph gain under pairwise minus graph gain under BCE. A positive value favours the incremental graph effect under pairwise. Inference uses the direct paired contrast, not comparison of significance across cells.

| Group | nDCG@10 interaction [95% CI] |
|---|---:|
| all | +0.000003 [-0.000159, 0.000163] |
| new_in_val | +0.000076 [-0.000201, 0.000374] |
| warm_in_val | -0.000070 [-0.000268, 0.000132] |

New-minus-warm interaction: **+0.000146 [-0.000162, 0.000511]**.

All intervals use 1,000 user-cluster resamples, seed 0, after event-metric averaging over training seeds. They are exploratory/unadjusted on reused Val and conditional on these trained runs/candidates. Selection follows the same rule but selected epochs/update counts can differ; this tests the whole objective/selection/refit recipe. Novelty-group means retain their own qualifying-user denominators. Monthly tables are descriptive slices, not independent temporal replications.
