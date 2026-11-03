# Full-data promising-ranker replication

232,904 validation events / 124,416 users; full catalogue unless labelled technical sample. Identical natural frozen-text/90-day-recency K200 lists, strictly pre-T0 graph, chronological B tuning, prefix-only preprocessing, paired negatives/initial states, epoch zero eligible and fixed-epoch all-B refits. All selected weights locked before outer evaluation. No held-out Test predictions.

This is exploratory replication on reused validation data, including pilot users, not an independent holdout. Intervals use 1,000 paired-user draws, seed 0, after averaging event metrics across training seeds. They are unadjusted and conditional on these runs/candidates. No final production system or causal mechanism is selected.

| Method | nDCG@10 | Recall@10 | Recall@200 | New nDCG@10 | Warm nDCG@10 |
|---|---:|---:|---:|---:|---:|
| retrieval | 0.006284 | 0.011506 | 0.054746 | 0.007611 | 0.005606 |
| bce_text | 0.008115 | 0.015399 | 0.054746 | 0.008697 | 0.007846 |
| pairwise_text | 0.007767 | 0.014865 | 0.054746 | 0.009152 | 0.007103 |
| pairwise_coreview | 0.008065 | 0.015290 | 0.054746 | 0.008873 | 0.007571 |
| pairwise_combined | 0.007930 | 0.015129 | 0.054746 | 0.008261 | 0.007707 |

| Group | Contrast | nDCG@10 change [95% CI] |
|---|---|---:|
| all | pairwise_text minus retrieval | +0.001484 [0.001174, 0.001780] |
| all | pairwise_text minus bce_text | -0.000347 [-0.000497, -0.000188] |
| all | pairwise_coreview minus pairwise_text | +0.000298 [0.000161, 0.000439] |
| all | pairwise_combined minus pairwise_coreview | -0.000135 [-0.000239, -0.000026] |
| all | pairwise_coreview minus retrieval | +0.001782 [0.001463, 0.002068] |
| all | pairwise_combined minus retrieval | +0.001647 [0.001331, 0.001942] |
| new_in_val | pairwise_text minus retrieval | +0.001541 [0.000994, 0.002074] |
| new_in_val | pairwise_text minus bce_text | +0.000455 [0.000191, 0.000704] |
| new_in_val | pairwise_coreview minus pairwise_text | -0.000279 [-0.000498, -0.000039] |
| new_in_val | pairwise_combined minus pairwise_coreview | -0.000612 [-0.000820, -0.000408] |
| new_in_val | pairwise_coreview minus retrieval | +0.001262 [0.000741, 0.001776] |
| new_in_val | pairwise_combined minus retrieval | +0.000650 [0.000127, 0.001161] |
| warm_in_val | pairwise_text minus retrieval | +0.001497 [0.001153, 0.001846] |
| warm_in_val | pairwise_text minus bce_text | -0.000743 [-0.000936, -0.000558] |
| warm_in_val | pairwise_coreview minus pairwise_text | +0.000468 [0.000308, 0.000623] |
| warm_in_val | pairwise_combined minus pairwise_coreview | +0.000136 [0.000002, 0.000261] |
| warm_in_val | pairwise_coreview minus retrieval | +0.001965 [0.001614, 0.002336] |
| warm_in_val | pairwise_combined minus retrieval | +0.002100 [0.001746, 0.002457] |

All methods retain existing text/basic metadata matching and nine common support controls. Optional co-review features are four additional relational statistics; combined adds category/store-history fractions. New targets have no pre-T0 review edges; their gains do not imply otherwise. Natural-only training changes the supervised population versus historical force-insertion runs. The historical MLP is not a matched objective control for these models.

Monthly/history/new-warm tables and failure funnels retain supports and seed outcomes. Months are descriptive slices, not temporal retraining replications. Offline preparation/fit/scoring costs are not serving latency. Bulk candidates, features, logits and weights stay local; compact tables and provenance can be shared.
