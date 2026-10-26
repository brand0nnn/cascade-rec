# Ranker replication and group analysis

Validation only. Checkpoint rule and candidate lists are fixed. No Test evaluation.

Training seeds vary the ranker; the retriever remains the same frozen seed-0 retriever.

| Setup | Completed seeds | Macro nDCG@10 mean ± seed SD | R1 baseline | Difference, user-bootstrap 95% CI |
|---|---|---:|---:|---|
| matching | [0, 1, 2] | 0.006843 ± 0.000233 | 0.005642 | 0.001201 [0.000902, 0.001494] |
| attributes | [0, 1, 2] | 0.006551 ± 0.000290 | 0.005642 | 0.000908 [0.000581, 0.001256] |
| din | [0, 1, 2] | 0.006436 ± 0.000499 | 0.005642 | 0.000794 [0.000470, 0.001104] |

## Who benefits?

Each row averages within user inside that group, then across users. Group means do not add up to the overall mean. Differences compare identical cases against retriever order. Rescue/demotion columns are mean raw counts across completed seeds.

### item_type

| Setup / group | Cases / users | Retrieved % | R1 nDCG@10 | Ranker nDCG@10 | Difference [95% CI] | Positive seeds | Rescued / demoted |
|---|---:|---:|---:|---:|---|---:|---:|
| matching / new-in-period | 76,768 / 52,420 | 2.90 | 0.000361 | 0.002820 | 0.002460 [0.002165, 0.002787] | 3/3 | 475.3 / 51.0 |
| matching / warm-head | 109,765 / 75,877 | 12.47 | 0.011290 | 0.011538 | 0.000249 [-0.000303, 0.000839] | 2/3 | 1881.3 / 1680.0 |
| matching / warm-tail | 46,371 / 37,944 | 1.64 | 0.000166 | 0.001175 | 0.001009 [0.000805, 0.001224] | 3/3 | 119.7 / 16.0 |
| attributes / new-in-period | 76,768 / 52,420 | 2.90 | 0.000361 | 0.004126 | 0.003765 [0.003426, 0.004164] | 3/3 | 718.0 / 58.0 |
| attributes / warm-head | 109,765 / 75,877 | 12.47 | 0.011290 | 0.010150 | -0.001139 [-0.001714, -0.000521] | 0/3 | 1998.3 / 1967.0 |
| attributes / warm-tail | 46,371 / 37,944 | 1.64 | 0.000166 | 0.001218 | 0.001052 [0.000808, 0.001302] | 3/3 | 127.3 / 17.0 |
| din / new-in-period | 76,768 / 52,420 | 2.90 | 0.000361 | 0.004074 | 0.003713 [0.003369, 0.004072] | 3/3 | 708.3 / 51.3 |
| din / warm-head | 109,765 / 75,877 | 12.47 | 0.011290 | 0.009865 | -0.001425 [-0.001964, -0.000815] | 0/3 | 1812.3 / 1909.0 |
| din / warm-tail | 46,371 / 37,944 | 1.64 | 0.000166 | 0.001398 | 0.001232 [0.000993, 0.001483] | 3/3 | 152.0 / 16.7 |

### history_length

| Setup / group | Cases / users | Retrieved % | R1 nDCG@10 | Ranker nDCG@10 | Difference [95% CI] | Positive seeds | Rescued / demoted |
|---|---:|---:|---:|---:|---|---:|---:|
| matching / 1-2 | 31,556 / 22,413 | 7.09 | 0.005783 | 0.006629 | 0.000846 [0.000137, 0.001608] | 3/3 | 314.0 / 231.7 |
| matching / 11+ | 25,271 / 11,043 | 6.67 | 0.005226 | 0.005488 | 0.000262 [-0.000721, 0.001265] | 3/3 | 240.7 / 204.7 |
| matching / 3-5 | 116,282 / 79,720 | 7.23 | 0.005698 | 0.006979 | 0.001282 [0.000898, 0.001676] | 3/3 | 1249.0 / 835.3 |
| matching / 6-10 | 59,795 / 39,178 | 7.27 | 0.005956 | 0.007208 | 0.001252 [0.000624, 0.001883] | 3/3 | 672.7 / 475.3 |
| attributes / 1-2 | 31,556 / 22,413 | 7.09 | 0.005783 | 0.006633 | 0.000850 [0.000064, 0.001701] | 3/3 | 377.7 / 276.0 |
| attributes / 11+ | 25,271 / 11,043 | 6.67 | 0.005226 | 0.005675 | 0.000450 [-0.000570, 0.001621] | 3/3 | 290.0 / 230.3 |
| attributes / 3-5 | 116,282 / 79,720 | 7.23 | 0.005698 | 0.006624 | 0.000927 [0.000496, 0.001355] | 3/3 | 1420.0 / 988.7 |
| attributes / 6-10 | 59,795 / 39,178 | 7.27 | 0.005956 | 0.006766 | 0.000811 [0.000143, 0.001465] | 3/3 | 756.0 / 547.0 |
| din / 1-2 | 31,556 / 22,413 | 7.09 | 0.005783 | 0.006639 | 0.000856 [0.000102, 0.001605] | 3/3 | 344.7 / 247.0 |
| din / 11+ | 25,271 / 11,043 | 6.67 | 0.005226 | 0.005247 | 0.000022 [-0.001012, 0.001045] | 1/3 | 252.0 / 224.3 |
| din / 3-5 | 116,282 / 79,720 | 7.23 | 0.005698 | 0.006501 | 0.000804 [0.000393, 0.001223] | 3/3 | 1340.3 / 962.3 |
| din / 6-10 | 59,795 / 39,178 | 7.27 | 0.005956 | 0.006870 | 0.000914 [0.000210, 0.001572] | 3/3 | 735.7 / 543.3 |

### history_diversity

| Setup / group | Cases / users | Retrieved % | R1 nDCG@10 | Ranker nDCG@10 | Difference [95% CI] | Positive seeds | Rescued / demoted |
|---|---:|---:|---:|---:|---|---:|---:|
| matching / 1-2 | 31,556 / 22,413 | 7.09 | 0.005783 | 0.006629 | 0.000846 [0.000137, 0.001608] | 3/3 | 314.0 / 231.7 |
| matching / 11+ / homogeneous | 248 / 138 | 8.06 | 0.006634 | 0.001969 | -0.004664 [-0.011866, 0.000618] | 0/3 | 1.3 / 3.7 |
| matching / 11+ / mixed | 25,023 / 10,936 | 6.65 | 0.005193 | 0.005517 | 0.000324 [-0.000700, 0.001214] | 3/3 | 239.3 / 201.0 |
| matching / 3-5 | 116,282 / 79,720 | 7.23 | 0.005698 | 0.006979 | 0.001282 [0.000898, 0.001676] | 3/3 | 1249.0 / 835.3 |
| matching / 6-10 | 59,795 / 39,178 | 7.27 | 0.005956 | 0.007208 | 0.001252 [0.000624, 0.001883] | 3/3 | 672.7 / 475.3 |
| attributes / 1-2 | 31,556 / 22,413 | 7.09 | 0.005783 | 0.006633 | 0.000850 [0.000064, 0.001701] | 3/3 | 377.7 / 276.0 |
| attributes / 11+ / homogeneous | 248 / 138 | 8.06 | 0.006634 | 0.003958 | -0.002675 [-0.009635, 0.002768] | 0/3 | 3.0 / 3.0 |
| attributes / 11+ / mixed | 25,023 / 10,936 | 6.65 | 0.005193 | 0.005685 | 0.000492 [-0.000659, 0.001552] | 3/3 | 287.0 / 227.3 |
| attributes / 3-5 | 116,282 / 79,720 | 7.23 | 0.005698 | 0.006624 | 0.000927 [0.000496, 0.001355] | 3/3 | 1420.0 / 988.7 |
| attributes / 6-10 | 59,795 / 39,178 | 7.27 | 0.005956 | 0.006766 | 0.000811 [0.000143, 0.001465] | 3/3 | 756.0 / 547.0 |
| din / 1-2 | 31,556 / 22,413 | 7.09 | 0.005783 | 0.006639 | 0.000856 [0.000102, 0.001605] | 3/3 | 344.7 / 247.0 |
| din / 11+ / homogeneous | 248 / 138 | 8.06 | 0.006634 | 0.002825 | -0.003808 [-0.011125, 0.001880] | 0/3 | 2.7 / 3.7 |
| din / 11+ / mixed | 25,023 / 10,936 | 6.65 | 0.005193 | 0.005268 | 0.000075 [-0.000930, 0.001006] | 1/3 | 249.3 / 220.7 |
| din / 3-5 | 116,282 / 79,720 | 7.23 | 0.005698 | 0.006501 | 0.000804 [0.000393, 0.001223] | 3/3 | 1340.3 / 962.3 |
| din / 6-10 | 59,795 / 39,178 | 7.27 | 0.005956 | 0.006870 | 0.000914 [0.000210, 0.001572] | 3/3 | 735.7 / 543.3 |

### validation_window

| Setup / group | Cases / users | Retrieved % | R1 nDCG@10 | Ranker nDCG@10 | Difference [95% CI] | Positive seeds | Rescued / demoted |
|---|---:|---:|---:|---:|---|---:|---:|
| matching / days 0-89 | 61,523 / 43,362 | 8.61 | 0.007170 | 0.008360 | 0.001190 [0.000569, 0.001840] | 3/3 | 789.7 / 592.3 |
| matching / days 180-269 | 63,759 / 42,556 | 6.30 | 0.004585 | 0.005642 | 0.001057 [0.000565, 0.001622] | 3/3 | 587.7 / 398.3 |
| matching / days 270+ | 44,863 / 31,365 | 5.80 | 0.004273 | 0.005304 | 0.001031 [0.000418, 0.001680] | 3/3 | 395.3 / 284.0 |
| matching / days 90-179 | 62,759 / 42,934 | 7.58 | 0.005806 | 0.007143 | 0.001338 [0.000796, 0.001938] | 3/3 | 703.7 / 472.3 |
| attributes / days 0-89 | 61,523 / 43,362 | 8.61 | 0.007170 | 0.009147 | 0.001977 [0.001272, 0.002737] | 3/3 | 1016.7 / 687.7 |
| attributes / days 180-269 | 63,759 / 42,556 | 6.30 | 0.004585 | 0.004944 | 0.000359 [-0.000200, 0.000919] | 3/3 | 615.3 / 475.7 |
| attributes / days 270+ | 44,863 / 31,365 | 5.80 | 0.004273 | 0.004474 | 0.000201 [-0.000468, 0.000842] | 2/3 | 400.3 / 319.7 |
| attributes / days 90-179 | 62,759 / 42,934 | 7.58 | 0.005806 | 0.006771 | 0.000965 [0.000396, 0.001639] | 3/3 | 811.3 / 559.0 |
| din / days 0-89 | 61,523 / 43,362 | 8.61 | 0.007170 | 0.008426 | 0.001256 [0.000599, 0.001991] | 3/3 | 923.3 / 670.0 |
| din / days 180-269 | 63,759 / 42,556 | 6.30 | 0.004585 | 0.005195 | 0.000610 [0.000037, 0.001159] | 3/3 | 613.7 / 457.3 |
| din / days 270+ | 44,863 / 31,365 | 5.80 | 0.004273 | 0.004600 | 0.000326 [-0.000301, 0.000957] | 3/3 | 386.0 / 315.0 |
| din / days 90-179 | 62,759 / 42,934 | 7.58 | 0.005806 | 0.006575 | 0.000770 [0.000205, 0.001372] | 3/3 | 749.7 / 534.7 |

## attributes versus matching

Common seeds: [0, 1, 2]. Paired end-to-end macro nDCG@10; positive favours attributes.

| Group | Difference [95% user CI] |
|---|---|
| all | -0.000292 [-0.000493, -0.000092] |
| new-in-period | 0.001306 [0.000993, 0.001614] |
| warm-head | -0.001388 [-0.001732, -0.001035] |
| warm-tail | 0.000044 [-0.000167, 0.000243] |
| 1-2 | 0.000004 [-0.000504, 0.000491] |
| 11+ / homogeneous | 0.001989 [-0.001090, 0.005827] |
| 11+ / mixed | 0.000168 [-0.000481, 0.000775] |
| 3-5 | -0.000355 [-0.000642, -0.000075] |
| 6-10 | -0.000442 [-0.000872, -0.000033] |

## din versus matching

Common seeds: [0, 1, 2]. Paired end-to-end macro nDCG@10; positive favours din.

| Group | Difference [95% user CI] |
|---|---|
| all | -0.000406 [-0.000578, -0.000227] |
| new-in-period | 0.001254 [0.001013, 0.001501] |
| warm-head | -0.001673 [-0.001961, -0.001345] |
| warm-tail | 0.000223 [0.000048, 0.000409] |
| 1-2 | 0.000010 [-0.000461, 0.000441] |
| 11+ / homogeneous | 0.000856 [-0.001919, 0.003708] |
| 11+ / mixed | -0.000250 [-0.000823, 0.000300] |
| 3-5 | -0.000478 [-0.000725, -0.000250] |
| 6-10 | -0.000338 [-0.000707, 0.000016] |

DIN uses item/category candidate embeddings inside unnormalised target attention. In matching-only MLP these embeddings are used for history pooling, so this comparison includes the candidate identity interactions introduced by attention. It does not isolate attention independently of that information. Full recipe and reference links are in docs/ranker_experiment.md.


## Definitions and limits

- New-in-period means target first interaction at/after validation start: cold at window entry. Warm tail/head retain the project item labels based on counts before T2, after separating new-in-period targets. These are descriptive evaluation groups; the before-T2 counts are never passed to the ranker.
- History length is prior interactions at target time. Diversity counts known leaf categories in the last 50 prior items; long histories are homogeneous at 1–2 distinct categories or mixed at 3+. Missing categories have an explicit unknown group when none is known.
- Validation windows are elapsed days since T1: 0–89, 90–179, 180–269, 270+. Entry novelty remains defined relative to T1 throughout.
- Conditional metrics use retrieved-target cases only and are saved in meso.json alongside end-to-end macro/micro metrics and support counts.
- Seed SD describes training variability. CIs resample paired users on mean per-instance metrics across completed seeds; they do not capture training-seed uncertainty. Averaging metrics across runs is not an ensemble.
- CIs are exploratory and unadjusted across groups. Group-specific gains do not establish causes or prove that gains differ between groups. Test replication and the planned three seeds are still needed.
- Any report with fewer than three completed seeds is provisional. Thresholds and definitions were specified before examining these group results.
