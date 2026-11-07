# Expanded explanation-graph validation evaluation

232,904 validation events / 124,416 users. Exploratory post-hoc target slices; no causal model attribution. Each slice averages its own events within user and then users. Metrics average training seeds 0/1/2 before paired user bootstrap: B=1000, seed=20261027, unadjusted 95% percentile CIs. These are conditional on archived trained runs.

The task lists T0=1598016867874, but the verified graph, split and source hashes use **1595800947930**; this earlier cutoff is retained. T1=1628643414042. Histories are reconstructed from the verified sequences/instance pointers; metadata comes from products.parquet. The requested val.npz and product_attributes.parquet names do not exist here.

Rescue/demotion columns in each CSV include exact per-seed event counts and their seed means, not ensemble outcomes. Net rescue ratio is (rescued−demoted)/(rescued+demoted), undefined with no churn. Raw event counts do not explain a user-macro metric by themselves.

## Selected findings

- Warm targets, 1 witness reviewer(s): 17,836 events; MLP minus retrieval +0.002741 [0.001636, 0.003813]; DIN minus retrieval +0.002412 [0.001269, 0.003570].
- Warm targets, 6+ witness reviewer(s): 8,084 events; MLP minus retrieval -0.022474 [-0.027760, -0.017029]; DIN minus retrieval -0.041376 [-0.046950, -0.035703].

The problem cell is partitioned below into returning, novel-in-window and unknown metadata groups. Do not compare significance across groups as an interaction test. The store table describes conditional associations, not causal protection; witness multiplicity and history relations are external graph evidence, not established internal ranker features.

## all_other_reviewers

| all_train_other_reviewers_bucket | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0 | 192341 | 110112 | 6415 | 0.000947 | 0.002521 | 0.003006 | +0.001574 [0.001382, 0.001765] | +0.002059 [0.001865, 0.002263] |
| 1 | 17836 | 16343 | 1932 | 0.005509 | 0.008250 | 0.007921 | +0.002741 [0.001636, 0.003813] | +0.002412 [0.001269, 0.003570] |
| 2-5 | 14643 | 13075 | 3488 | 0.019081 | 0.021691 | 0.019272 | +0.002610 [0.000474, 0.004583] | +0.000191 [-0.001923, 0.002207] |
| 6+ | 8084 | 7072 | 4837 | 0.106920 | 0.084446 | 0.065543 | -0.022474 [-0.027760, -0.017029] | -0.041376 [-0.046950, -0.035703] |

## warm_other_reviewers

| all_train_other_reviewers_bucket | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0 | 115573 | 78858 | 4191 | 0.001378 | 0.002355 | 0.002313 | +0.000977 [0.000748, 0.001227] | +0.000935 [0.000692, 0.001176] |
| 1 | 17836 | 16343 | 1932 | 0.005509 | 0.008250 | 0.007921 | +0.002741 [0.001636, 0.003813] | +0.002412 [0.001269, 0.003570] |
| 2-5 | 14643 | 13075 | 3488 | 0.019081 | 0.021691 | 0.019272 | +0.002610 [0.000474, 0.004583] | +0.000191 [-0.001923, 0.002207] |
| 6+ | 8084 | 7072 | 4837 | 0.106920 | 0.084446 | 0.065543 | -0.022474 [-0.027760, -0.017029] | -0.041376 [-0.046950, -0.035703] |

## warm_joint_other_reviewers

| all_train_other_reviewers_bucket | training_head | same_leaf_last | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | False | false | 51248 | 42444 | 661 | 0.000187 | 0.000740 | 0.000830 | +0.000553 [0.000387, 0.000747] | +0.000643 [0.000462, 0.000840] |
| 0 | False | true | 11950 | 10863 | 413 | 0.000428 | 0.002762 | 0.003182 | +0.002334 [0.001653, 0.003037] | +0.002753 [0.002076, 0.003506] |
| 0 | False | unknown | 3257 | 2990 | 30 | 0.000167 | 0.000789 | 0.000951 | +0.000622 [-0.000132, 0.001552] | +0.000784 [0.000043, 0.001650] |
| 0 | True | false | 39974 | 34182 | 2266 | 0.002902 | 0.003879 | 0.003571 | +0.000976 [0.000431, 0.001501] | +0.000669 [0.000139, 0.001179] |
| 0 | True | true | 7295 | 6891 | 719 | 0.004196 | 0.006465 | 0.006069 | +0.002269 [0.000692, 0.003793] | +0.001873 [0.000286, 0.003436] |
| 0 | True | unknown | 1849 | 1769 | 102 | 0.003261 | 0.005022 | 0.005457 | +0.001760 [-0.000428, 0.004007] | +0.002195 [0.000138, 0.004487] |
| 1 | False | false | 2785 | 2715 | 75 | 0.000381 | 0.002088 | 0.002420 | +0.001707 [0.000585, 0.003039] | +0.002040 [0.000979, 0.003332] |
| 1 | False | true | 972 | 958 | 59 | 0.000933 | 0.009196 | 0.009987 | +0.008263 [0.004352, 0.013303] | +0.009054 [0.004933, 0.014034] |
| 1 | False | unknown | 171 | 167 | 3 | 0.001731 | 0.004990 | 0.003117 | +0.003259 [-0.005193, 0.014970] | +0.001386 [-0.005193, 0.009351] |
| 1 | True | false | 11134 | 10592 | 1303 | 0.006092 | 0.008444 | 0.007420 | +0.002352 [0.001089, 0.003689] | +0.001328 [-0.000025, 0.002735] |
| 1 | True | true | 2366 | 2314 | 434 | 0.009541 | 0.014830 | 0.015050 | +0.005289 [0.000977, 0.009397] | +0.005509 [0.001244, 0.009581] |
| 1 | True | unknown | 408 | 401 | 58 | 0.014138 | 0.006079 | 0.013449 | -0.008059 [-0.016798, -0.000127] | -0.000689 [-0.011263, 0.009890] |
| 2-5 | False | false | 618 | 599 | 34 | 0.002204 | 0.006686 | 0.007410 | +0.004481 [0.001036, 0.008442] | +0.005206 [0.000931, 0.010563] |
| 2-5 | False | true | 231 | 226 | 31 | 0.001332 | 0.019560 | 0.021612 | +0.018228 [0.007407, 0.031019] | +0.020280 [0.007657, 0.034265] |
| 2-5 | False | unknown | 35 | 34 | 2 | 0.000000 | 0.000000 | 0.002951 | +0.000000 [0.000000, 0.000000] | +0.002951 [0.000000, 0.008854] |
| 2-5 | True | false | 10820 | 9948 | 2565 | 0.019822 | 0.021270 | 0.018205 | +0.001449 [-0.000810, 0.003786] | -0.001617 [-0.003921, 0.000606] |
| 2-5 | True | true | 2623 | 2538 | 789 | 0.020397 | 0.027656 | 0.025562 | +0.007259 [0.001705, 0.012936] | +0.005165 [-0.000208, 0.010808] |
| 2-5 | True | unknown | 316 | 310 | 67 | 0.018390 | 0.012363 | 0.013900 | -0.006028 [-0.019988, 0.006263] | -0.004490 [-0.020351, 0.009999] |
| 6+ | False | false | 28 | 28 | 5 | 0.000000 | 0.049101 | 0.026441 | +0.049101 [0.000000, 0.115593] | +0.026441 [0.000000, 0.072297] |
| 6+ | False | true | 12 | 12 | 2 | 0.000000 | 0.000000 | 0.000000 | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |
| 6+ | True | false | 6195 | 5582 | 3615 | 0.100146 | 0.078015 | 0.059986 | -0.022132 [-0.028932, -0.015792] | -0.040161 [-0.047145, -0.033665] |
| 6+ | True | true | 1636 | 1560 | 1103 | 0.137966 | 0.106254 | 0.083747 | -0.031712 [-0.045433, -0.017675] | -0.054220 [-0.068765, -0.039815] |
| 6+ | True | unknown | 213 | 204 | 112 | 0.094500 | 0.082808 | 0.069478 | -0.011692 [-0.044019, 0.021315] | -0.025022 [-0.060847, 0.010879] |

## all_witness_paths

| all_train_witness_paths_bucket | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0 | 192341 | 110112 | 6415 | 0.000947 | 0.002521 | 0.003006 | +0.001574 [0.001382, 0.001765] | +0.002059 [0.001865, 0.002263] |
| 1 | 17681 | 16228 | 1920 | 0.005571 | 0.008225 | 0.007902 | +0.002654 [0.001483, 0.003813] | +0.002331 [0.001129, 0.003549] |
| 2-5 | 14698 | 13134 | 3483 | 0.018983 | 0.021573 | 0.019223 | +0.002590 [0.000545, 0.004502] | +0.000240 [-0.001866, 0.002244] |
| 6+ | 8184 | 7143 | 4854 | 0.105874 | 0.083894 | 0.065007 | -0.021981 [-0.027377, -0.015657] | -0.040868 [-0.046545, -0.034650] |

## warm_witness_paths

| all_train_witness_paths_bucket | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 0 | 115573 | 78858 | 4191 | 0.001378 | 0.002355 | 0.002313 | +0.000977 [0.000748, 0.001227] | +0.000935 [0.000692, 0.001176] |
| 1 | 17681 | 16228 | 1920 | 0.005571 | 0.008225 | 0.007902 | +0.002654 [0.001483, 0.003813] | +0.002331 [0.001129, 0.003549] |
| 2-5 | 14698 | 13134 | 3483 | 0.018983 | 0.021573 | 0.019223 | +0.002590 [0.000545, 0.004502] | +0.000240 [-0.001866, 0.002244] |
| 6+ | 8184 | 7143 | 4854 | 0.105874 | 0.083894 | 0.065007 | -0.021981 [-0.027377, -0.015657] | -0.040868 [-0.046545, -0.034650] |

## warm_joint_witness_paths

| all_train_witness_paths_bucket | training_head | same_leaf_last | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | False | false | 51248 | 42444 | 661 | 0.000187 | 0.000740 | 0.000830 | +0.000553 [0.000387, 0.000747] | +0.000643 [0.000462, 0.000840] |
| 0 | False | true | 11950 | 10863 | 413 | 0.000428 | 0.002762 | 0.003182 | +0.002334 [0.001653, 0.003037] | +0.002753 [0.002076, 0.003506] |
| 0 | False | unknown | 3257 | 2990 | 30 | 0.000167 | 0.000789 | 0.000951 | +0.000622 [-0.000132, 0.001552] | +0.000784 [0.000043, 0.001650] |
| 0 | True | false | 39974 | 34182 | 2266 | 0.002902 | 0.003879 | 0.003571 | +0.000976 [0.000431, 0.001501] | +0.000669 [0.000139, 0.001179] |
| 0 | True | true | 7295 | 6891 | 719 | 0.004196 | 0.006465 | 0.006069 | +0.002269 [0.000692, 0.003793] | +0.001873 [0.000286, 0.003436] |
| 0 | True | unknown | 1849 | 1769 | 102 | 0.003261 | 0.005022 | 0.005457 | +0.001760 [-0.000428, 0.004007] | +0.002195 [0.000138, 0.004487] |
| 1 | False | false | 2744 | 2676 | 72 | 0.000387 | 0.002080 | 0.002418 | +0.001693 [0.000559, 0.003144] | +0.002032 [0.000838, 0.003357] |
| 1 | False | true | 956 | 943 | 57 | 0.000948 | 0.008686 | 0.009652 | +0.007738 [0.003542, 0.012612] | +0.008705 [0.004335, 0.013572] |
| 1 | False | unknown | 168 | 165 | 2 | 0.001752 | 0.000000 | 0.000000 | -0.001752 [-0.005256, 0.000000] | -0.001752 [-0.005256, 0.000000] |
| 1 | True | false | 11061 | 10533 | 1298 | 0.006156 | 0.008507 | 0.007452 | +0.002350 [0.000922, 0.003820] | +0.001296 [-0.000153, 0.002737] |
| 1 | True | true | 2352 | 2302 | 434 | 0.009591 | 0.014907 | 0.015128 | +0.005317 [0.001014, 0.009612] | +0.005538 [0.001478, 0.009900] |
| 1 | True | unknown | 400 | 393 | 57 | 0.014426 | 0.006203 | 0.013456 | -0.008223 [-0.017241, -0.000622] | -0.000971 [-0.012185, 0.008870] |
| 2-5 | False | false | 653 | 631 | 37 | 0.002093 | 0.006513 | 0.007080 | +0.004421 [0.001320, 0.008324] | +0.004987 [0.001113, 0.010010] |
| 2-5 | False | true | 242 | 240 | 33 | 0.001254 | 0.020997 | 0.022288 | +0.019743 [0.008258, 0.032297] | +0.021034 [0.008329, 0.034657] |
| 2-5 | False | unknown | 38 | 37 | 3 | 0.000000 | 0.022523 | 0.016781 | +0.022523 [0.000000, 0.067568] | +0.016781 [0.000000, 0.047630] |
| 2-5 | True | false | 10829 | 9962 | 2560 | 0.019767 | 0.021046 | 0.018110 | +0.001279 [-0.001005, 0.003612] | -0.001657 [-0.004015, 0.000587] |
| 2-5 | True | true | 2614 | 2529 | 782 | 0.020469 | 0.027754 | 0.025653 | +0.007285 [0.002243, 0.012812] | +0.005183 [0.000069, 0.010874] |
| 2-5 | True | unknown | 322 | 316 | 68 | 0.018041 | 0.012128 | 0.013969 | -0.005913 [-0.018727, 0.006530] | -0.004072 [-0.018587, 0.009650] |
| 6+ | False | false | 34 | 34 | 5 | 0.000000 | 0.040436 | 0.021775 | +0.040436 [0.000000, 0.101059] | +0.021775 [0.000000, 0.060982] |
| 6+ | False | true | 17 | 17 | 2 | 0.000000 | 0.000000 | 0.000000 | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |
| 6+ | True | false | 6259 | 5630 | 3625 | 0.099314 | 0.077714 | 0.059633 | -0.021600 [-0.028277, -0.015606] | -0.039680 [-0.047013, -0.033376] |
| 6+ | True | true | 1659 | 1580 | 1110 | 0.136220 | 0.104909 | 0.082654 | -0.031311 [-0.044673, -0.016383] | -0.053566 [-0.067480, -0.039206] |
| 6+ | True | unknown | 215 | 206 | 112 | 0.093582 | 0.082004 | 0.068804 | -0.011578 [-0.042604, 0.021253] | -0.024779 [-0.059897, 0.011121] |

## all_train_witness_age

| all_train_witness_age | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 365-729 days | 13660 | 12375 | 3415 | 0.031685 | 0.030965 | 0.026661 | -0.000720 [-0.003145, 0.001767] | -0.005024 [-0.007484, -0.002445] |
| 730+ days | 12406 | 10999 | 4680 | 0.048659 | 0.026099 | 0.019108 | -0.022559 [-0.025321, -0.019642] | -0.029551 [-0.032602, -0.026512] |
| <365 days | 14497 | 12942 | 2162 | 0.010769 | 0.025518 | 0.022749 | +0.014749 [0.012809, 0.016844] | +0.011980 [0.009991, 0.013899] |
| none | 192341 | 110112 | 6415 | 0.000947 | 0.002521 | 0.003006 | +0.001574 [0.001382, 0.001765] | +0.002059 [0.001865, 0.002263] |

## all_train_witness_era

| all_train_witness_era | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| [T0,T1) | 22265 | 19179 | 3610 | 0.013728 | 0.025192 | 0.022514 | +0.011463 [0.009951, 0.013232] | +0.008785 [0.007265, 0.010413] |
| none | 192341 | 110112 | 6415 | 0.000947 | 0.002521 | 0.003006 | +0.001574 [0.001382, 0.001765] | +0.002059 [0.001865, 0.002263] |
| pre-T0 | 18298 | 15836 | 6647 | 0.048108 | 0.030059 | 0.023318 | -0.018049 [-0.020463, -0.015714] | -0.024790 [-0.027396, -0.022337] |

## all_category_continuity

| category_continuity | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Immediate continuity | 44157 | 34704 | 4364 | 0.008143 | 0.010895 | 0.010655 | +0.002751 [0.001816, 0.003579] | +0.002511 [0.001603, 0.003343] |
| Novel category for user | 95793 | 64905 | 6028 | 0.005215 | 0.005591 | 0.004945 | +0.000377 [-0.000012, 0.000832] | -0.000270 [-0.000681, 0.000187] |
| Returning category | 61068 | 47209 | 4580 | 0.006427 | 0.007638 | 0.007590 | +0.001211 [0.000638, 0.001803] | +0.001164 [0.000540, 0.001757] |
| Unknown category metadata | 31886 | 22755 | 1700 | 0.003854 | 0.004721 | 0.004156 | +0.000867 [0.000211, 0.001579] | +0.000302 [-0.000353, 0.001000] |

## warm_category_continuity

| category_continuity | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Immediate continuity | 27085 | 22778 | 3550 | 0.012518 | 0.014247 | 0.012638 | +0.001729 [0.000456, 0.002975] | +0.000120 [-0.001180, 0.001409] |
| Novel category for user | 68536 | 50901 | 5519 | 0.007002 | 0.006923 | 0.005784 | -0.000078 [-0.000620, 0.000469] | -0.001218 [-0.001748, -0.000639] |
| Returning category | 37981 | 31684 | 3858 | 0.009877 | 0.010232 | 0.009274 | +0.000355 [-0.000470, 0.001275] | -0.000603 [-0.001443, 0.000375] |
| Unknown category metadata | 22534 | 17399 | 1521 | 0.005186 | 0.005963 | 0.004734 | +0.000777 [-0.000092, 0.001660] | -0.000453 [-0.001333, 0.000415] |

## problem_cell_category_continuity

| category_continuity | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Novel category for user | 13980 | 12607 | 3722 | 0.030794 | 0.027112 | 0.021430 | -0.003681 [-0.006172, -0.001140] | -0.009363 [-0.011696, -0.006875] |
| Returning category | 10701 | 9640 | 2952 | 0.033141 | 0.030913 | 0.026531 | -0.002228 [-0.004934, 0.000616] | -0.006610 [-0.009533, -0.003774] |
| Unknown category metadata | 3468 | 3199 | 809 | 0.022887 | 0.021958 | 0.015552 | -0.000929 [-0.005147, 0.003078] | -0.007335 [-0.011427, -0.003593] |

## all_store_joint

| same_store_history | co_review_all_train | training_head | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| false | false | False | 130529 | 85033 | 2185 | 0.000193 | 0.000949 | 0.001230 | +0.000756 [0.000630, 0.000892] | +0.001036 [0.000895, 0.001178] |
| false | false | True | 46829 | 39008 | 2803 | 0.003098 | 0.003598 | 0.003248 | +0.000500 [0.000018, 0.000974] | +0.000150 [-0.000319, 0.000643] |
| false | true | False | 3921 | 3754 | 98 | 0.000364 | 0.000912 | 0.001423 | +0.000547 [-0.000033, 0.001225] | +0.001059 [0.000358, 0.001874] |
| false | true | True | 30525 | 25518 | 7989 | 0.029973 | 0.023520 | 0.018519 | -0.006453 [-0.008211, -0.004790] | -0.011454 [-0.013217, -0.009806] |
| true | false | False | 12644 | 10624 | 1141 | 0.001225 | 0.013237 | 0.018917 | +0.012012 [0.010524, 0.013574] | +0.017692 [0.015889, 0.019598] |
| true | false | True | 2276 | 2228 | 284 | 0.004291 | 0.019575 | 0.020823 | +0.015284 [0.011004, 0.019629] | +0.016532 [0.011909, 0.021033] |
| true | true | False | 931 | 892 | 113 | 0.002970 | 0.023726 | 0.023474 | +0.020756 [0.014136, 0.027618] | +0.020504 [0.013438, 0.027217] |
| true | true | True | 5186 | 4720 | 2057 | 0.050207 | 0.072233 | 0.065689 | +0.022026 [0.016451, 0.027591] | +0.015481 [0.009524, 0.021227] |
| unknown | false | False | 50 | 50 | 2 | 0.000000 | 0.000000 | 0.000000 | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |
| unknown | false | True | 13 | 13 | 0 | 0.000000 | 0.000000 | 0.000000 | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |

## warm_store_joint

| same_store_history | co_review_all_train | training_head | Events | Users | Entered@200 | Ret nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| false | false | False | 62750 | 49837 | 823 | 0.000162 | 0.000586 | 0.000663 | +0.000424 [0.000276, 0.000571] | +0.000501 [0.000348, 0.000656] |
| false | false | True | 46829 | 39008 | 2803 | 0.003098 | 0.003598 | 0.003248 | +0.000500 [0.000018, 0.000974] | +0.000150 [-0.000319, 0.000643] |
| false | true | False | 3921 | 3754 | 98 | 0.000364 | 0.000912 | 0.001423 | +0.000547 [-0.000033, 0.001225] | +0.001059 [0.000358, 0.001874] |
| false | true | True | 30525 | 25518 | 7989 | 0.029973 | 0.023520 | 0.018519 | -0.006453 [-0.008211, -0.004790] | -0.011454 [-0.013217, -0.009806] |
| true | false | False | 3680 | 3481 | 279 | 0.001125 | 0.010175 | 0.011545 | +0.009050 [0.006753, 0.011467] | +0.010420 [0.007866, 0.013171] |
| true | false | True | 2276 | 2228 | 284 | 0.004291 | 0.019575 | 0.020823 | +0.015284 [0.011004, 0.019629] | +0.016532 [0.011909, 0.021033] |
| true | true | False | 931 | 892 | 113 | 0.002970 | 0.023726 | 0.023474 | +0.020756 [0.014136, 0.027618] | +0.020504 [0.013438, 0.027217] |
| true | true | True | 5186 | 4720 | 2057 | 0.050207 | 0.072233 | 0.065689 | +0.022026 [0.016451, 0.027591] | +0.015481 [0.009524, 0.021227] |
| unknown | false | False | 25 | 25 | 2 | 0.000000 | 0.000000 | 0.000000 | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |
| unknown | false | True | 13 | 13 | 0 | 0.000000 | 0.000000 | 0.000000 | +0.000000 [0.000000, 0.000000] | +0.000000 [0.000000, 0.000000] |

## Retriever Text versus Text+ID

| Axis | Bucket | Metric | Events | Users | Text | Text+ID | Difference [95% CI] |
|---|---|---|---:|---:|---:|---:|---|
| retriever_train_other_reviewers_bucket | 0 | recall200 | 212022 | 117079 | 0.037265 | 0.037273 | +0.000008 [-0.000869, 0.000849] |
| retriever_train_other_reviewers_bucket | 0 | ndcg10 | 212022 | 117079 | 0.001343 | 0.001214 | -0.000129 [-0.000244, -0.000015] |
| retriever_train_other_reviewers_bucket | 1 | recall200 | 8663 | 8197 | 0.207662 | 0.255254 | +0.047592 [0.040887, 0.055124] |
| retriever_train_other_reviewers_bucket | 1 | ndcg10 | 8663 | 8197 | 0.014007 | 0.019692 | +0.005685 [0.003862, 0.007273] |
| retriever_train_other_reviewers_bucket | 2-5 | recall200 | 7446 | 6897 | 0.396569 | 0.451804 | +0.055235 [0.047109, 0.063359] |
| retriever_train_other_reviewers_bucket | 2-5 | ndcg10 | 7446 | 6897 | 0.040842 | 0.053371 | +0.012529 [0.009778, 0.015286] |
| retriever_train_other_reviewers_bucket | 6+ | recall200 | 4773 | 4280 | 0.718867 | 0.729307 | +0.010441 [0.000601, 0.019824] |
| retriever_train_other_reviewers_bucket | 6+ | ndcg10 | 4773 | 4280 | 0.148279 | 0.164287 | +0.016008 [0.010421, 0.020868] |
| retriever_train_witness_paths_bucket | 0 | recall200 | 212022 | 117079 | 0.037265 | 0.037273 | +0.000008 [-0.000869, 0.000849] |
| retriever_train_witness_paths_bucket | 0 | ndcg10 | 212022 | 117079 | 0.001343 | 0.001214 | -0.000129 [-0.000244, -0.000015] |
| retriever_train_witness_paths_bucket | 1 | recall200 | 8594 | 8144 | 0.208372 | 0.255892 | +0.047520 [0.040179, 0.054733] |
| retriever_train_witness_paths_bucket | 1 | ndcg10 | 8594 | 8144 | 0.014115 | 0.019814 | +0.005699 [0.004013, 0.007395] |
| retriever_train_witness_paths_bucket | 2-5 | recall200 | 7471 | 6918 | 0.394193 | 0.449264 | +0.055071 [0.046586, 0.063528] |
| retriever_train_witness_paths_bucket | 2-5 | ndcg10 | 7471 | 6918 | 0.040619 | 0.052852 | +0.012232 [0.009466, 0.015053] |
| retriever_train_witness_paths_bucket | 6+ | recall200 | 4817 | 4312 | 0.716875 | 0.727768 | +0.010893 [0.001320, 0.020105] |
| retriever_train_witness_paths_bucket | 6+ | ndcg10 | 4817 | 4312 | 0.147313 | 0.163628 | +0.016314 [0.010710, 0.021838] |

## Scope and remaining gates

Witness age uses the saved deterministic smallest-reviewer/history-item witness, not the most recent path or the whole witness distribution. Era refers to that witness target edge; both legs satisfy the strict graph cutoff. Age bins: <365, 365–729 and 730+ days before the validation event. Degree/path bins were fixed to 0, 1, 2–5 and 6+ before this run.

Immediate continuity matches the last known category; returning matches an earlier item with a known different last category. Novel means no match anywhere in a fully known observed history window, not a lifetime-new category. Missing values remain a separate unknown group. No raw retrieval scores or causal score-weight tests are available in these target archives.

Device gate: **blocked**, 0/100 locked titles reviewed. Locked human device audit is incomplete; no quantitative device claims. Development parser statuses: {'unknown': 30, 'parsed': 20, 'ambiguous': 19, 'universal': 18, 'negated': 13}. Gold labels are needed to establish false positives; ambiguous shorthand is a review issue, not a measured false positive. No regex was tuned and no Same-device evaluation column was enabled.

Development review issues include slash-separated model/variant lists (iPhone SE/5/5S, iPhone 7/8 Plus) and mixtures of explicit names with shorthand continuations. Universal claims and excluded models require separate human review; parser status alone cannot establish correctness. Three smallest-ID ambiguous development titles:

- Item 4795: SwitchEasy Melt Ice Cream Hard Case for iPhone 4/4S - 1 Pack - Retail Packaging - Sweet Candy
- Item 6912: iPhone SE case, elago® [Slim Fit][Extreme Hot Red] - [Light][Minimalistic][True Fit] - for iPhone SE/5/5S
- Item 23113: iPhone 6/6S Belt Clip Case - Encased (SlimShield Series) Ultra Thin Grip Case with Holster for Apple iPhone 6/6s 4.7" (Smooth Black)

Steps 4A/4B candidate-pool versus top-10 amplification and Step 5 ranker top-three overtakers remain blocked by missing item_text_emb.npy and original natural candidate matrices. Target-rank tables cannot reconstruct those outputs. No Test predictions, model inference, retraining or cluster jobs were used.
