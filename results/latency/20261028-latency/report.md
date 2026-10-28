# Cascade latency (Val)

1000 queries one at a time after 50 warm-up queries (seed 0); Apple M2 Pro, macOS-26.6.2-arm64-arm-64bit; torch 2.5.1 (1 thread), FAISS 1.15.1 (1 thread). Load average before / after: [1.58, 2.02, 2.47] / [1.8, 2.03, 2.46].

Median / p95 in ms. User tower = history lookup + forward; retrieval = FAISS search (K1 + buffer) + §4.6 filtering; ranker = features + forward + top-10 ordering; total is timed end to end (sum of phases per query, so the median total is not the sum of component medians).

| job | efSearch used | user fwd | user tower | FAISS search | filter | retrieval | rank features | rank forward | ranker | total | overlap w/ exact ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_k200__20261028-natural-matching-seed0 | — | 0.66 / 0.73 | 0.67 / 0.74 | 1.20 / 1.30 | 0.08 / 0.10 | 1.28 / 1.38 | 1.21 / 1.36 | 0.92 / 1.11 | 2.16 / 2.45 | 4.11 / 4.47 | 0.9999 |
| hnsw_ef200_k200__20261028-natural-matching-seed0 | 450 | 0.59 / 0.62 | 0.60 / 0.63 | 0.61 / 0.75 | 0.04 / 0.06 | 0.66 / 0.80 | 0.95 / 1.04 | 0.80 / 0.89 | 1.76 / 1.92 | 3.03 / 3.28 | 0.9379 |
| exact_k500__20261028-natural-matching-seed0 | — | 0.68 / 0.75 | 0.69 / 0.76 | 1.42 / 1.66 | 0.08 / 0.10 | 1.50 / 1.76 | 2.29 / 2.59 | 2.07 / 2.57 | 4.40 / 5.13 | 6.61 / 7.53 | 0.9971 |
| ivf_nprobe4_k200__20261028-natural-matching-seed0 | — | 0.59 / 0.67 | 0.60 / 0.68 | 0.06 / 0.08 | 0.04 / 0.06 | 0.11 / 0.13 | 0.93 / 1.09 | 0.79 / 1.07 | 1.74 / 2.15 | 2.44 / 2.91 | 0.3089 |
| hnsw_ef800_k200__20261028-natural-matching-seed0 | 800 | 0.59 / 0.64 | 0.60 / 0.65 | 1.23 / 1.42 | 0.05 / 0.06 | 1.28 / 1.47 | 0.99 / 1.07 | 0.84 / 0.92 | 1.84 / 1.98 | 3.74 / 4.01 | 0.9758 |
| exact_k50__20261028-natural-matching-seed0 | — | 0.62 / 0.65 | 0.63 / 0.66 | 1.07 / 1.14 | 0.07 / 0.09 | 1.13 / 1.22 | 0.65 / 0.71 | 0.32 / 0.34 | 0.98 / 1.05 | 2.74 / 2.90 | 0.9999 |
| exact_k100__20261028-natural-matching-seed0 | — | 0.63 / 0.68 | 0.64 / 0.69 | 1.11 / 1.19 | 0.07 / 0.09 | 1.18 / 1.26 | 0.81 / 0.89 | 0.51 / 0.62 | 1.34 / 1.49 | 3.17 / 3.37 | 0.9999 |
| ivf_nprobe32_k200__20261028-natural-matching-seed0 | — | 0.57 / 0.60 | 0.58 / 0.61 | 0.15 / 0.16 | 0.04 / 0.06 | 0.19 / 0.21 | 0.92 / 0.99 | 0.82 / 0.90 | 1.76 / 1.88 | 2.53 / 2.67 | 0.7434 |

Index build (single thread, not part of query latency): exact 0.0s, hnsw 19.5s, ivf 2.3s.
Rows are listed in the (shuffled) order they were timed. Sanity fields in metrics.json are on the timed queries only; they are not accuracy results.
