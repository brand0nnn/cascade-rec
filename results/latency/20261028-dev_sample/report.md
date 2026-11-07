# Cascade latency (Val) — DEVELOPMENT SAMPLE, timings invalid, do not report

20 queries one at a time after 5 warm-up queries (seed 0); Apple M2 Pro, macOS-26.6.2-arm64-arm-64bit; torch 2.5.1 (1 thread), FAISS 1.15.1 (1 thread). Load average before / after: [3.48, 3.68, 3.16] / [3.48, 3.68, 3.16].

Median / p95 in ms. User tower = history lookup + forward; retrieval = FAISS search (K1 + buffer) + §4.6 filtering; ranker = features + forward + top-10 ordering; total is timed end to end (sum of phases per query, so the median total is not the sum of component medians).

| job | efSearch used | user fwd | user tower | FAISS search | filter | retrieval | rank features | rank forward | ranker | total | overlap w/ exact ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_k200__20261028-natural-matching-seed0 | — | 0.72 / 0.90 | 0.75 / 0.92 | 1.21 / 1.33 | 0.07 / 0.10 | 1.28 / 1.40 | 1.30 / 1.46 | 0.93 / 1.09 | 2.26 / 2.54 | 4.34 / 4.80 | 0.9998 |
| hnsw_ef200_k200__20261028-natural-matching-seed0 | 450 | 0.73 / 0.94 | 0.74 / 0.96 | 0.79 / 1.07 | 0.06 / 0.11 | 0.85 / 1.15 | 1.26 / 1.48 | 0.98 / 1.25 | 2.25 / 2.72 | 3.84 / 4.51 | 0.9315 |
| exact_k500__20261028-natural-matching-seed0 | — | 0.73 / 1.06 | 0.74 / 1.08 | 1.46 / 1.55 | 0.08 / 0.11 | 1.54 / 1.63 | 2.47 / 2.75 | 2.28 / 2.73 | 4.85 / 5.63 | 7.18 / 8.19 | 0.9971 |
| ivf_nprobe4_k200__20261028-natural-matching-seed0 | — | 0.63 / 0.71 | 0.63 / 0.72 | 0.08 / 0.10 | 0.05 / 0.07 | 0.13 / 0.17 | 1.05 / 1.32 | 0.86 / 1.07 | 1.96 / 2.34 | 2.73 / 3.17 | 0.3563 |
| hnsw_ef800_k200__20261028-natural-matching-seed0 | 800 | 0.69 / 0.79 | 0.69 / 0.80 | 1.38 / 1.58 | 0.05 / 0.07 | 1.43 / 1.69 | 1.09 / 1.28 | 0.92 / 1.06 | 2.07 / 2.32 | 4.14 / 4.67 | 0.9677 |
| exact_k50__20261028-natural-matching-seed0 | — | 0.79 / 0.99 | 0.80 / 1.00 | 1.10 / 1.31 | 0.07 / 0.14 | 1.18 / 1.45 | 0.79 / 0.99 | 0.51 / 0.62 | 1.32 / 1.61 | 3.28 / 3.97 | 1.0000 |
| exact_k100__20261028-natural-matching-seed0 | — | 0.68 / 0.86 | 0.69 / 0.87 | 1.14 / 1.21 | 0.08 / 0.12 | 1.23 / 1.29 | 0.98 / 1.23 | 0.60 / 0.75 | 1.60 / 1.92 | 3.54 / 3.88 | 1.0000 |
| ivf_nprobe32_k200__20261028-natural-matching-seed0 | — | 0.77 / 0.97 | 0.78 / 0.98 | 0.19 / 0.24 | 0.07 / 0.10 | 0.28 / 0.33 | 1.27 / 1.53 | 1.00 / 1.32 | 2.30 / 2.81 | 3.34 / 3.93 | 0.7790 |

Index build (single thread, not part of query latency): exact 0.0s, hnsw 21.0s, ivf 2.3s.
Rows are listed in the (shuffled) order they were timed. Sanity fields in metrics.json are on the timed queries only; they are not accuracy results.
