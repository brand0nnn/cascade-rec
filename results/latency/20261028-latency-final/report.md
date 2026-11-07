# Cascade latency (Val)

1000 queries one at a time after 50 warm-up queries (seed 0); Apple M2 Pro, macOS-26.6.2-arm64-arm-64bit; torch 2.5.1 (1 thread), FAISS 1.15.1 (1 thread). Load average before / after: [1.3, 1.41, 1.65] / [2.7, 1.87, 1.8].

Median / p95 in ms. User tower = history lookup + forward; retrieval = FAISS search (K1 + buffer) + §4.6 filtering; ranker = features + forward + top-10 ordering; total is timed end to end (sum of phases per query, so the median total is not the sum of component medians).

| job | efSearch used | user fwd | user tower | FAISS search | filter | retrieval | rank features | rank forward | ranker | total | overlap w/ exact ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_k100__20261028-natural-matching-seed0 | — | 0.64 / 0.68 | 0.65 / 0.69 | 1.12 / 1.25 | 0.07 / 0.09 | 1.19 / 1.33 | 0.92 / 1.00 | 0.53 / 0.58 | 1.47 / 1.58 | 3.31 / 3.54 | 0.9999 |
| hnsw_ef800_k200__20261028-din-natural-seed0 | 800 | 0.73 / 0.84 | 0.74 / 0.86 | 1.64 / 2.03 | 0.07 / 0.09 | 1.71 / 2.10 | 1.17 / 1.45 | 6.21 / 7.52 | 7.40 / 8.93 | 9.88 / 11.74 | 0.9758 |
| exact_k100__20261028-din-natural-seed0 | — | 0.63 / 0.68 | 0.64 / 0.69 | 1.12 / 1.20 | 0.07 / 0.09 | 1.19 / 1.28 | 0.90 / 1.04 | 3.11 / 3.40 | 4.04 / 4.39 | 5.86 / 6.28 | 0.9999 |
| hnsw_ef800_k200__20261028-natural-matching-seed0 | 800 | 0.66 / 0.70 | 0.66 / 0.71 | 1.25 / 1.44 | 0.05 / 0.06 | 1.30 / 1.50 | 1.07 / 1.14 | 0.86 / 0.92 | 1.95 / 2.06 | 3.92 / 4.17 | 0.9758 |
| exact_k50__20261028-natural-matching-seed0 | — | 0.69 / 0.73 | 0.70 / 0.74 | 1.09 / 1.25 | 0.07 / 0.09 | 1.16 / 1.34 | 0.77 / 0.83 | 0.40 / 0.44 | 1.18 / 1.26 | 3.04 / 3.31 | 0.9999 |
| exact_k200__20261028-natural-matching-seed0 | — | 0.66 / 0.71 | 0.67 / 0.72 | 1.20 / 1.34 | 0.07 / 0.09 | 1.27 / 1.42 | 1.24 / 1.38 | 0.89 / 1.13 | 2.16 / 2.40 | 4.09 / 4.52 | 0.9999 |
| exact_k500__20261028-din-natural-seed0 | — | 0.96 / 1.14 | 0.99 / 1.18 | 2.00 / 2.25 | 0.14 / 0.17 | 2.14 / 2.40 | 3.03 / 3.44 | 21.97 / 23.74 | 25.10 / 27.03 | 28.26 / 30.49 | 0.9971 |
| exact_k200__20261028-din-natural-seed0 | — | 0.78 / 0.96 | 0.80 / 0.98 | 1.65 / 1.95 | 0.10 / 0.13 | 1.76 / 2.06 | 1.37 / 1.57 | 7.42 / 8.79 | 8.77 / 10.30 | 11.40 / 13.12 | 0.9999 |
| ivf_nprobe32_k200__20261028-natural-matching-seed0 | — | 0.58 / 0.73 | 0.59 / 0.74 | 0.15 / 0.17 | 0.04 / 0.06 | 0.19 / 0.22 | 0.96 / 1.08 | 0.87 / 0.93 | 1.85 / 2.00 | 2.68 / 2.91 | 0.7434 |
| ivf_nprobe4_k200__20261028-natural-matching-seed0 | — | 0.58 / 0.71 | 0.59 / 0.72 | 0.06 / 0.08 | 0.04 / 0.06 | 0.11 / 0.13 | 0.97 / 1.10 | 0.87 / 1.08 | 1.88 / 2.14 | 2.58 / 2.88 | 0.3089 |
| exact_k500__20261028-natural-matching-seed0 | — | 0.67 / 0.78 | 0.68 / 0.79 | 1.43 / 1.80 | 0.08 / 0.11 | 1.52 / 1.90 | 2.39 / 2.76 | 2.52 / 2.83 | 4.93 / 5.56 | 7.09 / 8.12 | 0.9971 |
| hnsw_ef200_k200__20261028-din-natural-seed0 | 450 | 0.73 / 0.84 | 0.74 / 0.85 | 0.90 / 1.15 | 0.07 / 0.09 | 0.97 / 1.23 | 1.23 / 1.45 | 6.75 / 7.82 | 8.01 / 9.28 | 9.71 / 11.22 | 0.9379 |
| ivf_nprobe4_k200__20261028-din-natural-seed0 | — | 0.69 / 0.80 | 0.71 / 0.81 | 0.08 / 0.10 | 0.06 / 0.08 | 0.14 / 0.18 | 1.08 / 1.32 | 6.08 / 7.09 | 7.18 / 8.41 | 8.03 / 9.42 | 0.3089 |
| hnsw_ef200_k200__20261028-natural-matching-seed0 | 450 | 0.61 / 0.68 | 0.62 / 0.69 | 0.63 / 0.81 | 0.05 / 0.06 | 0.68 / 0.86 | 1.01 / 1.19 | 0.94 / 1.17 | 2.01 / 2.33 | 3.31 / 3.73 | 0.9379 |
| exact_k50__20261028-din-natural-seed0 | — | 0.63 / 1.03 | 0.64 / 1.04 | 1.08 / 1.16 | 0.07 / 0.09 | 1.15 / 1.24 | 0.70 / 0.94 | 1.29 / 2.08 | 1.99 / 2.99 | 3.81 / 5.17 | 0.9999 |
| ivf_nprobe32_k200__20261028-din-natural-seed0 | — | 0.68 / 0.76 | 0.69 / 0.77 | 0.18 / 0.21 | 0.06 / 0.08 | 0.24 / 0.29 | 1.07 / 1.30 | 6.00 / 7.10 | 7.09 / 8.42 | 8.03 / 9.43 | 0.7434 |

Index build (single thread, not part of query latency): exact 0.0s, hnsw 19.7s, ivf 2.3s.
Rows are listed in the (shuffled) order they were timed. Sanity fields in metrics.json are on the timed queries only; they are not accuracy results.
