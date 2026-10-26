# RQ3: rankers on identical validation candidates

Full validation: frozen two-tower-text seed-0 retriever, exact top-200 candidates. Test remains untouched.

| System | End-to-end macro nDCG@10 | Macro Recall@10 | Conditional macro nDCG@10 | Conditional macro GAUC | CPU total median / p95, ms |
|---|---:|---:|---:|---:|---:|
| R0: Popularity order | 0.003281 | 0.007270 | 0.048350 | 0.586998 | 0.010 / 0.011 |
| R1: Retrieval order | 0.005642 | 0.010446 | 0.082816 | 0.625756 | 0.003 / 0.003 |
| R2: MLP matching | 0.006843 ± 0.000233 | 0.013868 ± 0.000252 | 0.099056 ± 0.003558 | 0.711995 ± 0.001403 | 1.591 / 2.014 |
| R3: DIN matching | 0.006436 ± 0.000499 | 0.013629 ± 0.000582 | 0.094151 ± 0.007327 | 0.743015 ± 0.005554 | 7.324 / 8.363 |

Accuracy ± values are sample SD across three training seeds. R0/R1 are deterministic baselines. Timing values for learned models are means of the three seed-specific query medians/p95s; individual runs and all per-query timings are retained.

## CPU timing phases

| System / seed | Prepare median ms | Score median ms | Order median ms | Total median ms | Total p95 ms |
|---|---:|---:|---:|---:|---:|
| R0 | 0.003 | 0.001 | 0.006 | 0.010 | 0.011 |
| R1 | 0.003 | 0.000 | 0.000 | 0.003 | 0.003 |
| R2_seed0 | 0.821 | 0.740 | 0.013 | 1.587 | 2.016 |
| R2_seed1 | 0.830 | 0.742 | 0.013 | 1.594 | 2.017 |
| R2_seed2 | 0.827 | 0.743 | 0.013 | 1.590 | 2.011 |
| R3_seed0 | 1.224 | 6.084 | 0.026 | 7.402 | 8.930 |
| R3_seed1 | 1.065 | 6.243 | 0.026 | 7.328 | 8.116 |
| R3_seed2 | 1.038 | 6.142 | 0.025 | 7.242 | 8.042 |

Component medians need not sum to the total median. Total is timed directly for each query. Preparation includes valid-candidate extraction and, for learned models, the current feature builder and CPU tensor conversion. Score-only timing starts with prepared inputs. Disk I/O, checkpoint loading, one-time table preparation and retrieval are excluded.

Hardware: Apple M1 Max; one PyTorch/BLAS thread; 1000 identical measured validation queries after 50 warm-up queries per run.

## Paired comparisons

| Comparison | Macro nDCG@10 difference | User-bootstrap 95% CI |
|---|---:|---|
| R2_minus_R0 | 0.003562 | [0.003278, 0.003878] |
| R2_minus_R1 | 0.001201 | [0.000902, 0.001494] |
| R3_minus_R2 | -0.000406 | [-0.000578, -0.000227] |

Intervals resample users on seed-mean per-instance metric values. They describe user uncertainty conditional on the runs; training-seed uncertainty is shown separately. Group intervals are exploratory and unadjusted.

## Group results

### item_type

| System / group | Cases / users | Macro nDCG@10 | Conditional macro nDCG@10 | Macro Recall@10 |
|---|---:|---:|---:|---:|
| R0 / new-in-period | 76,768 / 52,420 | 0.000000 | 0.000000 | 0.000000 |
| R0 / warm-head | 109,765 / 75,877 | 0.006740 | 0.058811 | 0.014976 |
| R0 / warm-tail | 46,371 / 37,944 | 0.000000 | 0.000000 | 0.000000 |
| R1 / new-in-period | 76,768 / 52,420 | 0.000361 | 0.012432 | 0.000882 |
| R1 / warm-head | 109,765 / 75,877 | 0.011290 | 0.098293 | 0.020729 |
| R1 / warm-tail | 46,371 / 37,944 | 0.000166 | 0.010495 | 0.000376 |
| R2 / new-in-period | 76,768 / 52,420 | 0.002820 | 0.107199 | 0.005755 |
| R2 / warm-head | 109,765 / 75,877 | 0.011538 | 0.098482 | 0.023407 |
| R2 / warm-tail | 46,371 / 37,944 | 0.001175 | 0.078677 | 0.002371 |
| R3 / new-in-period | 76,768 / 52,420 | 0.004074 | 0.155016 | 0.008548 |
| R3 / warm-head | 109,765 / 75,877 | 0.009865 | 0.084187 | 0.020932 |
| R3 / warm-tail | 46,371 / 37,944 | 0.001398 | 0.091517 | 0.003042 |
### history_length

| System / group | Cases / users | Macro nDCG@10 | Conditional macro nDCG@10 | Macro Recall@10 |
|---|---:|---:|---:|---:|
| R0 / 1-2 | 31,556 / 22,413 | 0.003492 | 0.051977 | 0.007897 |
| R0 / 11+ | 25,271 / 11,043 | 0.002946 | 0.045088 | 0.006354 |
| R0 / 3-5 | 116,282 / 79,720 | 0.003133 | 0.046747 | 0.007033 |
| R0 / 6-10 | 59,795 / 39,178 | 0.003801 | 0.052125 | 0.008183 |
| R1 / 1-2 | 31,556 / 22,413 | 0.005783 | 0.086261 | 0.010663 |
| R1 / 11+ | 25,271 / 11,043 | 0.005226 | 0.079655 | 0.010363 |
| R1 / 3-5 | 116,282 / 79,720 | 0.005698 | 0.084184 | 0.010462 |
| R1 / 6-10 | 59,795 / 39,178 | 0.005956 | 0.083878 | 0.011233 |
| R2 / 1-2 | 31,556 / 22,413 | 0.006629 | 0.094304 | 0.013660 |
| R2 / 11+ | 25,271 / 11,043 | 0.005488 | 0.081587 | 0.011929 |
| R2 / 3-5 | 116,282 / 79,720 | 0.006979 | 0.102640 | 0.014078 |
| R2 / 6-10 | 59,795 / 39,178 | 0.007208 | 0.101188 | 0.014593 |
| R3 / 1-2 | 31,556 / 22,413 | 0.006639 | 0.096445 | 0.013891 |
| R3 / 11+ | 25,271 / 11,043 | 0.005247 | 0.078905 | 0.011388 |
| R3 / 3-5 | 116,282 / 79,720 | 0.006501 | 0.095255 | 0.013773 |
| R3 / 6-10 | 59,795 / 39,178 | 0.006870 | 0.095654 | 0.014590 |
### history_diversity

| System / group | Cases / users | Macro nDCG@10 | Conditional macro nDCG@10 | Macro Recall@10 |
|---|---:|---:|---:|---:|
| R0 / 1-2 | 31,556 / 22,413 | 0.003492 | 0.051977 | 0.007897 |
| R0 / 11+ / homogeneous | 248 / 138 | 0.008107 | 0.065450 | 0.009662 |
| R0 / 11+ / mixed | 25,023 / 10,936 | 0.002873 | 0.044832 | 0.006294 |
| R0 / 3-5 | 116,282 / 79,720 | 0.003133 | 0.046747 | 0.007033 |
| R0 / 6-10 | 59,795 / 39,178 | 0.003801 | 0.052125 | 0.008183 |
| R1 / 1-2 | 31,556 / 22,413 | 0.005783 | 0.086261 | 0.010663 |
| R1 / 11+ / homogeneous | 248 / 138 | 0.006634 | 0.078585 | 0.014090 |
| R1 / 11+ / mixed | 25,023 / 10,936 | 0.005193 | 0.079669 | 0.010287 |
| R1 / 3-5 | 116,282 / 79,720 | 0.005698 | 0.084184 | 0.010462 |
| R1 / 6-10 | 59,795 / 39,178 | 0.005956 | 0.083878 | 0.011233 |
| R2 / 1-2 | 31,556 / 22,413 | 0.006629 | 0.094304 | 0.013660 |
| R2 / 11+ / homogeneous | 248 / 138 | 0.001969 | 0.019350 | 0.005636 |
| R2 / 11+ / mixed | 25,023 / 10,936 | 0.005517 | 0.082369 | 0.011976 |
| R2 / 3-5 | 116,282 / 79,720 | 0.006979 | 0.102640 | 0.014078 |
| R2 / 6-10 | 59,795 / 39,178 | 0.007208 | 0.101188 | 0.014593 |
| R3 / 1-2 | 31,556 / 22,413 | 0.006639 | 0.096445 | 0.013891 |
| R3 / 11+ / homogeneous | 248 / 138 | 0.002825 | 0.050405 | 0.007649 |
| R3 / 11+ / mixed | 25,023 / 10,936 | 0.005268 | 0.079263 | 0.011417 |
| R3 / 3-5 | 116,282 / 79,720 | 0.006501 | 0.095255 | 0.013773 |
| R3 / 6-10 | 59,795 / 39,178 | 0.006870 | 0.095654 | 0.014590 |
### validation_window

| System / group | Cases / users | Macro nDCG@10 | Conditional macro nDCG@10 | Macro Recall@10 |
|---|---:|---:|---:|---:|
| R0 / days 0-89 | 61,523 / 43,362 | 0.003704 | 0.046244 | 0.008175 |
| R0 / days 180-269 | 63,759 / 42,556 | 0.002968 | 0.050240 | 0.006533 |
| R0 / days 270+ | 44,863 / 31,365 | 0.002508 | 0.045905 | 0.005628 |
| R0 / days 90-179 | 62,759 / 42,934 | 0.003628 | 0.049718 | 0.008178 |
| R1 / days 0-89 | 61,523 / 43,362 | 0.007170 | 0.090877 | 0.013463 |
| R1 / days 180-269 | 63,759 / 42,556 | 0.004585 | 0.077875 | 0.008510 |
| R1 / days 270+ | 44,863 / 31,365 | 0.004273 | 0.078874 | 0.008028 |
| R1 / days 90-179 | 62,759 / 42,934 | 0.005806 | 0.080525 | 0.010585 |
| R2 / days 0-89 | 61,523 / 43,362 | 0.008360 | 0.105018 | 0.016613 |
| R2 / days 180-269 | 63,759 / 42,556 | 0.005642 | 0.094517 | 0.011653 |
| R2 / days 270+ | 44,863 / 31,365 | 0.005304 | 0.093746 | 0.010930 |
| R2 / days 90-179 | 62,759 / 42,934 | 0.007143 | 0.098499 | 0.014520 |
| R3 / days 0-89 | 61,523 / 43,362 | 0.008426 | 0.106558 | 0.017194 |
| R3 / days 180-269 | 63,759 / 42,556 | 0.005195 | 0.085867 | 0.011325 |
| R3 / days 270+ | 44,863 / 31,365 | 0.004600 | 0.082741 | 0.009923 |
| R3 / days 90-179 | 62,759 / 42,934 | 0.006575 | 0.092623 | 0.014064 |

## Protocol notes

- R0 uses raw interaction counts strictly before validation start T1, sorting only the same retrieved candidates. These are not bucketed counts or metadata rating_number.
- R1 retains the stored retrieval order. Its GAUC uses cached float16 retrieval scores with half credit for ties; quantization ties can therefore affect GAUC without changing the stored R1 ordering.
- R0 target ranks use the shared pessimistic-tie policy. Candidate ties get no target-dependent tie break. The stable ordering performed during timing does not redefine accuracy.
- MLP/DIN accuracy and global GAUC come from their saved selected checkpoints, with case/user/retrieval-rank alignment enforced. Latency executes those checkpoints on CPU. GAUC is conditional on target retrieval; group tables report nDCG/Recall because per-instance learned score ties were not archived.
- Retrieval misses receive zero end-to-end credit; conditional metrics use retrieved-target cases only. No target is force-inserted into latency lists.
- Item groups separate new-in-period first. Existing warm head/tail labels use the project counts before T2 for descriptive analysis only. These labels are never model inputs. History groups use strictly prior events; mixed means at least three known leaf categories among the last 50.
- The benchmark measures the current Python feature builder, which prepares a common batch including some disabled feature fields. Cached text vectors are reused. These are local ranker-stage times, not full service or retrieval latency.
- DIN includes candidate item/category interactions inside attention and uses an unnormalised weighted sum. The matching-only MLP comparison does not isolate attention independently of that added query information or pooling scale.
