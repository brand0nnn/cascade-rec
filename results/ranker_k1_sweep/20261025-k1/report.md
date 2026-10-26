# Candidate-count sweep with frozen MLPs

> **Interpretation update, 2026-10-26:** No final K₁ is selected. The current evaluation plan requires an acceptable-loss margin δ and cost/tie-break rule recorded before inspecting outcomes; none was recorded for this sweep. K100 is the observed quality maximum. The K50 “recommendation” below, in saved JSON, and in the figure's dashed line is the historical exploratory heuristic only; it establishes neither equivalence nor a cost-optimal choice. Measurements are unchanged. The [original report](report_historical.md) is preserved byte-for-byte.

Full validation, frozen MLP seeds 0/1/2 and frozen exact two-tower-text seed-0 retriever. No retraining or Test evaluation.

| K1 | Retrieval Recall@K1 macro | End-to-end nDCG@10 macro ± seed SD | Recall@10 macro | Conditional nDCG@10 macro | Conditional GAUC macro | CPU ranker median / p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.029751 | 0.008315 ± 0.000119 | 0.014825 | 0.284566 | 0.726847 | 0.662 / 0.986 |
| 100 | 0.045741 | 0.008363 ± 0.000109 | 0.015519 | 0.184401 | 0.727526 | 0.941 / 1.288 |
| 200 | 0.068714 | 0.006843 ± 0.000233 | 0.013868 | 0.099056 | 0.711995 | 1.594 / 2.288 |
| 500 | 0.117544 | 0.002210 ± 0.000095 | 0.004894 | 0.019133 | 0.644834 | 3.730 / 4.767 |

## Recommendation by the prespecified rule

Observed best mean: K1=100. Its user-bootstrap 95% CI is [0.008019, 0.008687].
Eligible counts whose observed means lie within that interval: [50, 100]. Smallest eligible count: **50**.

This is the project's descriptive selection heuristic. Inclusion within a confidence interval does not prove practical equivalence. The best configuration was chosen on these validation results; intervals are exploratory and do not correct for selecting the maximum. Paired differences below provide a complementary view.

| K1 | Best minus K1, macro nDCG@10 | Paired-user 95% CI |
|---:|---:|---|
| 50 | 0.000047 | [-0.000145, 0.000241] |
| 100 | 0.000000 | [0.000000, 0.000000] |
| 200 | 0.001520 | [0.001302, 0.001734] |
| 500 | 0.006153 | [0.005804, 0.006469] |

## What adding candidates changes

Contributions below preserve the same overall per-user weights, so new-target gain plus existing-target competition equals the overall change. These are not subgroup macro means. Because scores come from one shared pointwise pass, adding competitors cannot improve the rank of a target already retrieved.

| Expansion | Newly retrieved targets | New-target nDCG contribution | Existing-target competition contribution | Overall nDCG change | New-target top-10 hits, mean | Existing top-10 demotions, mean |
|---|---:|---:|---:|---:|---:|---:|
| 50 → 100 | 3,851 | 0.002250 | -0.002203 | 0.000047 | 1069.0 | 941.7 |
| 100 → 200 | 5,514 | 0.002251 | -0.003771 | -0.001520 | 1133.0 | 1602.3 |
| 200 → 500 | 11,903 | 0.001757 | -0.006389 | -0.004633 | 929.3 | 3089.7 |

## Who benefits?

Each group uses the same cases/users across candidate counts. Macro averaging is performed within each group. Group means use different user populations and do not add to the overall mean. Full macro/micro metrics, conditional GAUC, support, rescue/demotion counts and paired differences versus the observed best are retained in sweep.json.

### item_type

| Group | Cases / users | K50 nDCG | K100 nDCG | K200 nDCG | K500 nDCG |
|---|---:|---:|---:|---:|---:|
| new-in-period | 76,768 / 52,420 | 0.001875 | 0.002497 | 0.002820 | 0.001915 |
| warm-head | 109,765 / 75,877 | 0.015532 | 0.015044 | 0.011538 | 0.002863 |
| warm-tail | 46,371 / 37,944 | 0.000623 | 0.000992 | 0.001175 | 0.001030 |
### history_length

| Group | Cases / users | K50 nDCG | K100 nDCG | K200 nDCG | K500 nDCG |
|---|---:|---:|---:|---:|---:|
| 1-2 | 31,556 / 22,413 | 0.008505 | 0.008552 | 0.006629 | 0.001852 |
| 11+ | 25,271 / 11,043 | 0.007532 | 0.007202 | 0.005488 | 0.002378 |
| 3-5 | 116,282 / 79,720 | 0.008665 | 0.008611 | 0.006979 | 0.002164 |
| 6-10 | 59,795 / 39,178 | 0.008418 | 0.008567 | 0.007208 | 0.002568 |
### history_diversity

| Group | Cases / users | K50 nDCG | K100 nDCG | K200 nDCG | K500 nDCG |
|---|---:|---:|---:|---:|---:|
| 1-2 | 31,556 / 22,413 | 0.008505 | 0.008552 | 0.006629 | 0.001852 |
| 11+ / homogeneous | 248 / 138 | 0.008988 | 0.003746 | 0.001969 | 0.000254 |
| 11+ / mixed | 25,023 / 10,936 | 0.007492 | 0.007227 | 0.005517 | 0.002398 |
| 3-5 | 116,282 / 79,720 | 0.008665 | 0.008611 | 0.006979 | 0.002164 |
| 6-10 | 59,795 / 39,178 | 0.008418 | 0.008567 | 0.007208 | 0.002568 |
### validation_window

| Group | Cases / users | K50 nDCG | K100 nDCG | K200 nDCG | K500 nDCG |
|---|---:|---:|---:|---:|---:|
| days 0-89 | 61,523 / 43,362 | 0.010187 | 0.010031 | 0.008360 | 0.002741 |
| days 180-269 | 63,759 / 42,556 | 0.007141 | 0.007164 | 0.005642 | 0.001915 |
| days 270+ | 44,863 / 31,365 | 0.006379 | 0.006426 | 0.005304 | 0.001657 |
| days 90-179 | 62,759 / 42,934 | 0.008736 | 0.008977 | 0.007143 | 0.002209 |

## K1=200 reproduction checks

| Seed | Original nDCG | Rescored nDCG | Cases with rank differences |
|---:|---:|---:|---:|
| 0 | 0.007104849 | 0.007104849 | 0 |
| 1 | 0.006659718 | 0.006659718 | 0 |
| 2 | 0.006763732 | 0.006763732 | 0 |

## Scope and interpretation

- Candidate sets are natural nested prefixes of the saved exact top-500 lists. No positive is forced into evaluation. All target absences receive zero end-to-end credit.
- Frozen weights and training feature normalization are retained. Each pair is scored once at K500 and its score is reused for prefixes: this is valid for the current pointwise MLP, whose pair features do not depend on other candidates. K200 must reproduce the saved macro nDCG within 0.000001 per seed; rank-level differences are reported above.
- At K500 the model receives more candidates than the K200 pool used during training. It still scores each pair independently, but the extra candidates change its evaluation distribution.
- Conditional metrics use different retrieved-target populations at different K. A decline in conditional nDCG alone does not establish worse ranking of the same cases.
- Cold-at-entry means new-in-period at T1. Warm tail/head use existing before-T2 descriptive labels after separating new items; those labels are never model inputs. Histories are strictly prior events.
- Seed SD describes three frozen ranker runs. User-bootstrap intervals condition on their mean per-instance metric values, are exploratory and are unadjusted across groups/configurations.
- CPU ranking timing: Apple M1 Max, one PyTorch/BLAS thread, 1000 identical measured validation queries after 50 warm-ups for every K/seed. Times include feature preparation, CPU tensor conversion, scoring and ordering. Retrieval, disk I/O, model loading and one-time table construction are excluded. Learned entries average seed-specific medians/p95s. These are ranker-stage timings, not total cascade latency.
- Cached exact candidate prefixes measure ranking quality without rerunning retrieval. Full retrieval latency and index sweeps remain separate work.
