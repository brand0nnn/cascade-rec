# Size-matched P3 ranker control (Val only, exploratory)

Pre-stated (7 Nov 2026, before fitting): if the P3 gain is mainly the pool, the size-matched P3 ranker keeps most of +0.002878 over the P1 ranker; if it is mainly training-set size, the gain shrinks towards the frozen-P1-ranker-on-P3 value (+0.000670). Either outcome is reported.

B natural positives: P1 20,676; P3 38,861; size-matched 20,676 (prefix 16,819 / tune 3,857). Selected epochs: [10, 10, 10].

| System | all | new_in_val | warm_in_val |
|---|---:|---:|---:|
| P1_ranker | 0.008141 | 0.008666 | 0.007887 |
| P3_ranker | 0.011019 | 0.006965 | 0.012684 |
| P3_sizematched_ranker | 0.010387 | 0.006300 | 0.012056 |
| P1ranker_on_P3 | 0.008811 | 0.005347 | 0.010308 |

## Paired differences, nDCG@10 (candidate − reference) [95% CI]

| Contrast | all | new_in_val | warm_in_val |
|---|---|---|---|
| P3_sizematched_ranker − P1_ranker | +0.002246 [+0.001901, +0.002602] | -0.002366 [-0.002719, -0.001969] | +0.004170 [+0.003729, +0.004592] |
| P3_ranker − P3_sizematched_ranker | +0.000632 [+0.000464, +0.000790] | +0.000665 [+0.000441, +0.000895] | +0.000627 [+0.000436, +0.000832] |
| P3_sizematched_ranker − P1ranker_on_P3 | +0.001576 [+0.001338, +0.001805] | +0.000953 [+0.000677, +0.001215] | +0.001748 [+0.001443, +0.002048] |
| P3_ranker − P1_ranker | +0.002878 [+0.002544, +0.003211] | -0.001701 [-0.002071, -0.001314] | +0.004797 [+0.004367, +0.005231] |

1,000 paired-user draws (seed 0) on seed-mean event metrics, unadjusted. No Test file was opened.
