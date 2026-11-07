# P3 fused pool x rolling monthly co-review features: results (Val only, exploratory)

Pre-stated (README, before computing): Yes on both (beats P3 bce_text T1 0.011451 and P1 rolling bce_coreview 0.010613), sub-additively; co-review - text on P3 under rolling features is positive.

232,904 Val events / 124,416 users; per-user macro, misses = 0; seed means (0/1/2). Intervals: 1,000 paired-user draws, seed 0, on seed-mean event values; unadjusted. Fits on MPS, scoring on CPU. No Test file opened; nothing selected. These are reused-Val exploratory comparisons.

Rolling P3 features: 14 B snapshots, 20,441 Val natural-hit rows (P3 recall@200 ceiling). Build 12.9 min.

## Systems

| System | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 | nDCG@10 seed sd |
|---|---|---:|---:|---:|---:|---:|
| P3_bce_text_T0 | all | 0.011019 | 0.020941 | 0.009226 | 0.008018 | 0.000120 |
| P3_bce_text_T0 | new | 0.006965 | 0.013617 | 0.005557 | 0.004965 | 0.000463 |
| P3_bce_text_T0 | warm | 0.012684 | 0.023958 | 0.010714 | 0.009270 | 0.000394 |
| P3_bce_text_T1 | all | 0.011451 | 0.021455 | 0.009748 | 0.008425 | 0.000190 |
| P3_bce_text_T1 | new | 0.006751 | 0.013106 | 0.005608 | 0.004832 | 0.000781 |
| P3_bce_text_T1 | warm | 0.013221 | 0.024609 | 0.011308 | 0.009776 | 0.000122 |
| P3_context_T1 | all | 0.012167 | 0.023040 | 0.010170 | 0.008884 | 0.000199 |
| P3_context_T1 | new | 0.010815 | 0.021250 | 0.008691 | 0.007671 | 0.000682 |
| P3_context_T1 | warm | 0.012332 | 0.023111 | 0.010390 | 0.009079 | 0.000007 |
| P1_rolling_bce_coreview_monthly | all | 0.010613 | 0.019116 | 0.009390 | 0.008027 | 0.000156 |
| P1_rolling_bce_coreview_monthly | new | 0.010639 | 0.019191 | 0.009267 | 0.008055 | 0.000586 |
| P1_rolling_bce_coreview_monthly | warm | 0.010508 | 0.018935 | 0.009344 | 0.007941 | 0.000198 |
| P1_rolling_bce_text_monthly | all | 0.009304 | 0.017048 | 0.007909 | 0.006961 | 0.000222 |
| P1_rolling_bce_text_monthly | new | 0.008699 | 0.015784 | 0.007229 | 0.006571 | 0.000867 |
| P1_rolling_bce_text_monthly | warm | 0.009579 | 0.017612 | 0.008206 | 0.007142 | 0.000096 |
| P1_bce_text_T0 | all | 0.008141 | 0.015447 | 0.006787 | 0.005932 | 0.000021 |
| P1_bce_text_T0 | new | 0.008666 | 0.017292 | 0.006758 | 0.006076 | 0.000168 |
| P1_bce_text_T0 | warm | 0.007887 | 0.014567 | 0.006793 | 0.005860 | 0.000134 |
| P3_rolling_bce_coreview_monthly | all | 0.013084 | 0.023607 | 0.011348 | 0.009887 | 0.000188 |
| P3_rolling_bce_coreview_monthly | new | 0.006890 | 0.012296 | 0.006043 | 0.005248 | 0.000433 |
| P3_rolling_bce_coreview_monthly | warm | 0.015571 | 0.028250 | 0.013459 | 0.011721 | 0.000098 |
| P3_rolling_bce_coreview_T1 | all | 0.013568 | 0.025135 | 0.011546 | 0.010062 | 0.000139 |
| P3_rolling_bce_coreview_T1 | new | 0.013485 | 0.024754 | 0.011703 | 0.010055 | 0.000782 |
| P3_rolling_bce_coreview_T1 | warm | 0.012926 | 0.024144 | 0.010850 | 0.009535 | 0.000159 |
| P3_rolling_bce_text_monthly | all | 0.011545 | 0.021153 | 0.009963 | 0.008636 | 0.000047 |
| P3_rolling_bce_text_monthly | new | 0.006092 | 0.010561 | 0.005428 | 0.004738 | 0.000426 |
| P3_rolling_bce_text_monthly | warm | 0.013791 | 0.025633 | 0.011784 | 0.010206 | 0.000174 |
| P3_rolling_bce_text_T1 | all | 0.012724 | 0.023648 | 0.010935 | 0.009414 | 0.000075 |
| P3_rolling_bce_text_T1 | new | 0.014261 | 0.025133 | 0.012871 | 0.010939 | 0.000808 |
| P3_rolling_bce_text_T1 | warm | 0.011390 | 0.021938 | 0.009411 | 0.008208 | 0.000479 |

## Paired differences (candidate − reference) [95% CI]

| Contrast | Candidate − reference | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |
|---|---|---|---:|---:|---:|---:|
| vs_P3_text_T1 | P3_rolling_bce_coreview_monthly − P3_bce_text_T1 | all | +0.001633 [+0.001391, +0.001870] | +0.002151 [+0.001775, +0.002535] | +0.001600 [+0.001275, +0.001925] | +0.001463 [+0.001240, +0.001681] |
| vs_P3_text_T1 | P3_rolling_bce_coreview_monthly − P3_bce_text_T1 | new | +0.000138 [-0.000269, +0.000522] | -0.000809 [-0.001496, -0.000178] | +0.000436 [-0.000042, +0.000907] | +0.000417 [+0.000072, +0.000752] |
| vs_P3_text_T1 | P3_rolling_bce_coreview_monthly − P3_bce_text_T1 | warm | +0.002350 [+0.002064, +0.002629] | +0.003641 [+0.003177, +0.004083] | +0.002151 [+0.001777, +0.002548] | +0.001944 [+0.001679, +0.002218] |
| vs_P1_rolling_coreview | P3_rolling_bce_coreview_monthly − P1_rolling_bce_coreview_monthly | all | +0.002471 [+0.002095, +0.002847] | +0.004490 [+0.003867, +0.005115] | +0.001958 [+0.001508, +0.002393] | +0.001860 [+0.001510, +0.002193] |
| vs_P1_rolling_coreview | P3_rolling_bce_coreview_monthly − P1_rolling_bce_coreview_monthly | new | -0.003750 [-0.004110, -0.003403] | -0.006895 [-0.007542, -0.006267] | -0.003223 [-0.003689, -0.002756] | -0.002806 [-0.003135, -0.002496] |
| vs_P1_rolling_coreview | P3_rolling_bce_coreview_monthly − P1_rolling_bce_coreview_monthly | warm | +0.005063 [+0.004535, +0.005549] | +0.009315 [+0.008458, +0.010102] | +0.004114 [+0.003530, +0.004691] | +0.003779 [+0.003326, +0.004210] |
| vs_P3_text_T0 | P3_rolling_bce_coreview_monthly − P3_bce_text_T0 | all | +0.002065 [+0.001822, +0.002310] | +0.002666 [+0.002223, +0.003072] | +0.002122 [+0.001813, +0.002463] | +0.001869 [+0.001648, +0.002081] |
| vs_P3_text_T0 | P3_rolling_bce_coreview_monthly − P3_bce_text_T0 | new | -0.000075 [-0.000487, +0.000341] | -0.001321 [-0.002056, -0.000620] | +0.000486 [-0.000028, +0.000960] | +0.000283 [-0.000061, +0.000661] |
| vs_P3_text_T0 | P3_rolling_bce_coreview_monthly − P3_bce_text_T0 | warm | +0.002888 [+0.002574, +0.003198] | +0.004292 [+0.003788, +0.004804] | +0.002745 [+0.002341, +0.003160] | +0.002451 [+0.002171, +0.002738] |
| vs_P3_context_T1 | P3_rolling_bce_coreview_monthly − P3_context_T1 | all | +0.000917 [+0.000602, +0.001217] | +0.000567 [+0.000046, +0.001086] | +0.001178 [+0.000784, +0.001596] | +0.001004 [+0.000732, +0.001283] |
| vs_P3_context_T1 | P3_rolling_bce_coreview_monthly − P3_context_T1 | new | -0.003925 [-0.004377, -0.003427] | -0.008954 [-0.009757, -0.008110] | -0.002647 [-0.003191, -0.002075] | -0.002422 [-0.002828, -0.002010] |
| vs_P3_context_T1 | P3_rolling_bce_coreview_monthly − P3_context_T1 | warm | +0.003239 [+0.002879, +0.003596] | +0.005138 [+0.004547, +0.005754] | +0.003069 [+0.002604, +0.003513] | +0.002642 [+0.002324, +0.002974] |
| serving_monthly_minus_T1 | P3_rolling_bce_coreview_monthly − P3_rolling_bce_coreview_T1 | all | -0.000484 [-0.000679, -0.000311] | -0.001528 [-0.001882, -0.001209] | -0.000198 [-0.000421, +0.000038] | -0.000174 [-0.000362, -0.000012] |
| serving_monthly_minus_T1 | P3_rolling_bce_coreview_monthly − P3_rolling_bce_coreview_T1 | new | -0.006596 [-0.007058, -0.006105] | -0.012458 [-0.013274, -0.011604] | -0.005660 [-0.006221, -0.005154] | -0.004807 [-0.005202, -0.004383] |
| serving_monthly_minus_T1 | P3_rolling_bce_coreview_monthly − P3_rolling_bce_coreview_T1 | warm | +0.002645 [+0.002484, +0.002810] | +0.004106 [+0.003808, +0.004363] | +0.002608 [+0.002387, +0.002840] | +0.002185 [+0.002031, +0.002347] |
| coreview_minus_text_P3_rolling | P3_rolling_bce_coreview_monthly − P3_rolling_bce_text_monthly | all | +0.001539 [+0.001337, +0.001748] | +0.002454 [+0.002130, +0.002775] | +0.001385 [+0.001086, +0.001678] | +0.001251 [+0.001053, +0.001452] |
| coreview_minus_text_P3_rolling | P3_rolling_bce_coreview_monthly − P3_rolling_bce_text_monthly | new | +0.000798 [+0.000557, +0.001070] | +0.001736 [+0.001342, +0.002231] | +0.000615 [+0.000311, +0.000971] | +0.000510 [+0.000287, +0.000752] |
| coreview_minus_text_P3_rolling | P3_rolling_bce_coreview_monthly − P3_rolling_bce_text_monthly | warm | +0.001781 [+0.001504, +0.002037] | +0.002617 [+0.002175, +0.003028] | +0.001675 [+0.001297, +0.002038] | +0.001514 [+0.001262, +0.001772] |
| coreview_minus_text_P3_rolling_T1 | P3_rolling_bce_coreview_T1 − P3_rolling_bce_text_T1 | all | +0.000843 [+0.000652, +0.001035] | +0.001487 [+0.001172, +0.001805] | +0.000611 [+0.000322, +0.000889] | +0.000648 [+0.000464, +0.000841] |
| coreview_minus_text_P3_rolling_T1 | P3_rolling_bce_coreview_T1 − P3_rolling_bce_text_T1 | new | -0.000775 [-0.000999, -0.000550] | -0.000379 [-0.000759, +0.000001] | -0.001168 [-0.001534, -0.000800] | -0.000884 [-0.001119, -0.000647] |
| coreview_minus_text_P3_rolling_T1 | P3_rolling_bce_coreview_T1 − P3_rolling_bce_text_T1 | warm | +0.001536 [+0.001264, +0.001783] | +0.002206 [+0.001766, +0.002638] | +0.001440 [+0.001065, +0.001806] | +0.001328 [+0.001086, +0.001574] |
| text_rolling_vs_P3_text_T1 | P3_rolling_bce_text_monthly − P3_bce_text_T1 | all | +0.000094 [-0.000095, +0.000268] | -0.000303 [-0.000642, +0.000015] | +0.000215 [-0.000019, +0.000454] | +0.000212 [+0.000038, +0.000371] |
| text_rolling_vs_P3_text_T1 | P3_rolling_bce_text_monthly − P3_bce_text_T1 | new | -0.000660 [-0.001047, -0.000317] | -0.002545 [-0.003229, -0.001962] | -0.000179 [-0.000640, +0.000243] | -0.000093 [-0.000420, +0.000223] |
| text_rolling_vs_P3_text_T1 | P3_rolling_bce_text_monthly − P3_bce_text_T1 | warm | +0.000569 [+0.000359, +0.000753] | +0.001024 [+0.000680, +0.001387] | +0.000476 [+0.000181, +0.000752] | +0.000430 [+0.000233, +0.000609] |
| text_rolling_vs_P3_text_T0 | P3_rolling_bce_text_monthly − P3_bce_text_T0 | all | +0.000526 [+0.000324, +0.000727] | +0.000212 [-0.000171, +0.000578] | +0.000737 [+0.000493, +0.001001] | +0.000618 [+0.000436, +0.000802] |
| text_rolling_vs_P3_text_T0 | P3_rolling_bce_text_monthly − P3_bce_text_T0 | new | -0.000873 [-0.001233, -0.000506] | -0.003057 [-0.003729, -0.002422] | -0.000129 [-0.000588, +0.000318] | -0.000227 [-0.000563, +0.000117] |
| text_rolling_vs_P3_text_T0 | P3_rolling_bce_text_monthly − P3_bce_text_T0 | warm | +0.001107 [+0.000872, +0.001337] | +0.001675 [+0.001287, +0.002108] | +0.001070 [+0.000772, +0.001376] | +0.000936 [+0.000725, +0.001143] |
| interaction_pool_x_rolling | P3_rolling_bce_coreview_monthly + P1_bce_text_T0 − (P3_bce_text_T0 + P1_rolling_bce_coreview_monthly) | all | -0.000407 [-0.000688, -0.000142] | -0.001003 [-0.001459, -0.000536] | -0.000481 [-0.000880, -0.000106] | -0.000225 [-0.000494, +0.000023] |
| interaction_pool_x_rolling | P3_rolling_bce_coreview_monthly + P1_bce_text_T0 − (P3_bce_text_T0 + P1_rolling_bce_coreview_monthly) | new | -0.002049 [-0.002469, -0.001645] | -0.003220 [-0.003987, -0.002471] | -0.002022 [-0.002605, -0.001440] | -0.001695 [-0.002090, -0.001330] |
| interaction_pool_x_rolling | P3_rolling_bce_coreview_monthly + P1_bce_text_T0 − (P3_bce_text_T0 + P1_rolling_bce_coreview_monthly) | warm | +0.000266 [-0.000067, +0.000593] | -0.000076 [-0.000687, +0.000523] | +0.000193 [-0.000272, +0.000649] | +0.000370 [+0.000056, +0.000688] |

## Selected epochs

| method       |   seed |   selected_epoch |   p3_bce_text_selected_epoch |   inner_ndcg10 |   p3_bce_text_inner_ndcg10 |   fit_seconds |   refit_seconds |
|:-------------|-------:|-----------------:|-----------------------------:|---------------:|---------------------------:|--------------:|----------------:|
| bce_coreview |      0 |                7 |                            7 |      0.0181909 |                  0.0147787 |       317.935 |        132.387  |
| bce_coreview |      1 |                7 |                            8 |      0.0179487 |                  0.0141613 |       316.653 |        131.975  |
| bce_coreview |      2 |                4 |                            9 |      0.0172953 |                  0.0142353 |       220.198 |         77.4953 |
| bce_text     |      0 |                7 |                            7 |      0.0154512 |                  0.0147787 |       317.993 |        133.226  |
| bce_text     |      1 |                7 |                            8 |      0.01515   |                  0.0141613 |       318.357 |        133.036  |
| bce_text     |      2 |                7 |                            9 |      0.0152447 |                  0.0142353 |       317.678 |        135.813  |

## Pre-stated expectation versus outcome (nDCG@10, all events)

- Beats P3 bce_text T1 (0.011451): +0.001633 [+0.001391, +0.001870] → positive.
- Beats P1 rolling bce_coreview (0.010613): +0.002471 [+0.002095, +0.002847] → positive.
- Additivity: interaction (observed − [P3 text T0 + P1 rolling co-review − P1 text T0]) -0.000407 [-0.000688, -0.000142] → negative (negative = sub-additive). The pool gain is measured with bce_text at T0 and the rolling gain with co-review on P1, so this interaction also absorbs any method-by-pool difference.
- Co-review − text within P3 rolling (monthly): +0.001539 [+0.001337, +0.001748] → positive.

Groups: new = target first review at/after T1; warm = otherwise. New/warm rows are shown in the tables above.
