# Policy-study extension (exploratory, validation only)

Seeds per arm: {'A': [0, 1, 2], 'B': [0, 1, 2], 'C': [0, 1, 2], 'D': [0, 1, 2]}. No Test predictions loaded.

## nDCG@10 by arm and K1 (seed mean)

| Arm | K50 | K100 | K200 | K500 | New-item K200 |
|---|---:|---:|---:|---:|---:|
| A | 0.008077 | 0.008272 | 0.006887 | 0.002301 | 0.003147 |
| B | 0.006579 | 0.005968 | 0.004200 | 0.001653 | 0.002549 |
| C | 0.008333 | 0.008609 | 0.008572 | 0.007923 | 0.002390 |
| D | 0.007658 | 0.007826 | 0.007625 | 0.006689 | 0.002514 |

## K200 paired differences (seed-mean, user bootstrap 95% CI)

| Comparison | Group | Delta | CI |
|---|---|---:|---|
| B-A | all | -0.002687 | [-0.002896, -0.002478] |
| B-A | new | -0.000599 | [-0.000863, -0.000364] |
| C-A | all | +0.001685 | [+0.001445, +0.001920] |
| C-A | new | -0.000757 | [-0.001030, -0.000489] |
| D-A | all | +0.000739 | [+0.000502, +0.000947] |
| D-A | new | -0.000633 | [-0.000902, -0.000362] |
| A-C | all | -0.001685 | [-0.001920, -0.001445] |
| A-C | new | +0.000757 | [+0.000489, +0.001030] |
| B-C | all | -0.004372 | [-0.004656, -0.004047] |
| B-C | new | +0.000158 | [-0.000144, +0.000461] |
| D-C | all | -0.000946 | [-0.001106, -0.000787] |
| D-C | new | +0.000124 | [-0.000016, +0.000262] |

Qualified vs A (3 seeds, overall CI>0, no new-item loss): []
Extension arms were added after the pilot was seen; treat any qualification as exploratory.
See k_curve.csv for E3 (best K per arm and paired gaps), expansion.csv for gain/competition.
