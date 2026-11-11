# Numbers and sources for poster brief v2

All paths are relative to the repository root. nDCG@10 values are ×10⁻³, per-user macro averages.

## Section 2

| Number | Value | Source |
|---|---|---|
| RQ1 Recall@200 (Val) | Pop 0.027, ItemKNN 0.008, BPR 0.027, SASRec 0.040, two-tower 0.069, recency 0.055, mixed 0.080 | `results/rq1/summary.md`; `results/analysis/20261106-fresh-two-tower/tables.md` (P1/P3 pool R@200 0.054746 / 0.079521) |
| Shared candidates | 12 of 200 | `docs/research_story.md` §5 |
| RQ2 recall at K (two-tower seed 0, Val) | K50 0.0298, K100 0.0457, K200 0.0687, K500 0.1175 | `results/analysis/20261107-fresh-tt-seeds/tables.md` (frozen s0) and `results/analysis/20261106-fresh-two-tower/tables.md` |
| RQ2 ranker nDCG@10 at K (Val, natural-list MLP, 3 seeds) | 8.33 / 8.61 / 8.57 / 7.92 | `results/targeted_study/e3-20261107-all-seeds/report.md`, arm C |
| K500 vs best K (K100) | −0.00069 [−0.00087, −0.00048] | `results/targeted_study/e3-20261107-all-seeds/k_curve.csv` |
| K200 → K500 decomposition (3-seed mean) | newly found purchases +1.33; existing purchases pushed down −1.98 | `results/targeted_study/e3-20261107-all-seeds/expansion.csv`, arm C |
| Index (K200, Val) | Flat 100% / 3.4 ms; HNSW 99.4% / 1.5 ms; IVF 94.6% / 0.9 ms | `results/steps_final/20261104/benchmark_*.csv` (as in the 8 Nov brief) |
| RQ3 popularity / retriever order (Test, recency lists) | 1.51 / 6.04 | `results/steps_final/20261104/test_comparison.csv` |
| RQ3 MLP on recency lists (Test) | 6.44 | `results/test_final/20261107/tables.md`, S1 |
| RQ3 two-tower / mixed lists (Test) | MLP 5.19 / 7.30; DIN 6.37 / 8.27 | `results/test_final_ranker/20261108/levels.csv` |

## Section 3 (Validation)

| Test | Purchases in list | Same ranker, top 10 | Source |
|---|---|---|---|
| 500 vs 200 candidates | 0.0687 → 0.1175 (+71%) | 8.57 → 7.92 (−8%) | e3 all-seeds report (arm C); fresh-two-tower retriever table |
| Fresher retriever | 0.0687 → 0.1113 (+62%; 3-seed +0.0425 [0.0413, 0.0436]) | 10.08 → 9.93, −0.15 [−0.37, +0.08]; new products 3.70 → 2.82, −0.88 [−1.17, −0.57] | `results/analysis/20261106-fresh-two-tower/tables.md`; `results/analysis/20261107-fresh-tt-seeds/tables.md` |
| Mixed lists, recency ranker not retrained | 0.0547 → 0.0795 (+45%) | 8.14 → 8.81 (+8%) | `results/analysis/20261107-pool-size-control/tables.md` |

## Section 4

| Number | Value | Source |
|---|---|---|
| Retrained on mixed lists (Val) | same data 10.39 (+28%); all data 11.02 (+35%), +0.00288 [0.00254, 0.00321] | pool-size-control tables |
| Retrained vs not, same data (Val) | +0.00158 [0.00134, 0.00181] | pool-size-control tables |
| Mixed lists on Test | 6.44 → 7.86 (+22%), +0.00142 [0.00115, 0.00170] | `results/test_final/20261107/tables.md`, C2 |
| Statistics timing (Test, same frozen ranker) | Jul 2020 6.44, Aug 2021 7.38, Jul 2022 7.82; +0.00138 [0.00122, 0.00153] (+21%) | `results/test_final/20261107/tables.md`, S1–S3, C1 |
| "No reviews yet" share in training | 43% of bought products vs 20% of other candidates | `docs/research_story.md` §6 |
| New products peak at ~6-month-old statistics (Test) | figure 07 | `results/figures/20261107-story*/07_statistics_age_curve.png` source data |
| 5/5 statistics predictions held on Test | — | `docs/research_story.md` §6 |

## Sections 5–6

| Number | Value | Source |
|---|---|---|
| Final ladder (Test) | SASRec 2.59; two-tower→MLP 5.19; two-tower→DIN 6.37; mixed→DIN 8.27; mixed + timed statistics + co-review → MLP 10.38 | `results/test_final_ranker/20261108/levels.csv`; `results/test_final/20261107/tables.md` (S8); SASRec Test baseline |
| Funnel (Test, best DIN system) | 93.6 never reach ranker; 4.8 below 10; 1.5 in top 10 | `fig15_funnel_flow_orange.png` |
| Latency | 17.6 ms (faster combined score, Test 8.23 vs 8.27, −0.00004 [−0.00012, +0.00005]); planned 4.0 ms | `docs/evaluations.md`, faster combined score section |
| Heatmap | 21.8% (was 3.0%); switcher at new store 0.0% | `fig17_user_x_product_orange.png` |
| Upgraders (2,000 Val upgrade purchases, recency retriever top 200) | 37.2% old phone vs 3.1% purchased model | `docs/evaluations.md`, "Upgrade hypothesis check"; `results/targeted_study/behaviour-failures-20261108/upgrader_candidates.csv` |
