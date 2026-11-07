# Policy-study extension (exploratory, validation only)

Seeds per arm: {'A': [0], 'B': [0], 'C': [0], 'D': [0]}. No Test predictions loaded.

## nDCG@10 by arm and K1 (seed mean)

| Arm | K50 | K100 | K200 | K500 | New-item K200 |
|---|---:|---:|---:|---:|---:|
| A | 0.008026 | 0.008370 | 0.007076 | 0.002400 | 0.003541 |
| B | 0.006598 | 0.006023 | 0.004216 | 0.001711 | 0.002404 |
| C | 0.008406 | 0.008684 | 0.008617 | 0.007863 | 0.002104 |
| D | 0.007677 | 0.007932 | 0.007780 | 0.006937 | 0.002496 |

## K200 paired differences (seed-mean, user bootstrap 95% CI)

| Comparison | Group | Delta | CI |
|---|---|---:|---|
| B-A | all | -0.002860 | [-0.003147, -0.002591] |
| B-A | new | -0.001136 | [-0.001494, -0.000780] |
| C-A | all | +0.001541 | [+0.001244, +0.001819] |
| C-A | new | -0.001436 | [-0.001769, -0.001106] |
| D-A | all | +0.000704 | [+0.000400, +0.000998] |
| D-A | new | -0.001045 | [-0.001384, -0.000712] |
| A-C | all | -0.001541 | [-0.001819, -0.001244] |
| A-C | new | +0.001436 | [+0.001106, +0.001769] |
| B-C | all | -0.004401 | [-0.004714, -0.004045] |
| B-C | new | +0.000300 | [-0.000035, +0.000653] |
| D-C | all | -0.000837 | [-0.001053, -0.000609] |
| D-C | new | +0.000391 | [+0.000180, +0.000585] |

Qualified vs A (3 seeds, overall CI>0, no new-item loss): []
Extension arms were added after the pilot was seen; treat any qualification as exploratory.
See k_curve.csv for E3 (best K per arm and paired gaps), expansion.csv for gain/competition.
