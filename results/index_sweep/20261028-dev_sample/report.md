# Index sweep (val, K1 = 200) — DEVELOPMENT SAMPLE, do not report

Retriever `results/retriever/two_tower_text/seed0`; 4775 instances, 2553 users; 111480 items indexed; FAISS 1.15.1 (1 thread(s)), Apple M2 Pro. Search asks for K1 + buffer = 450 results, then §4.6 filtering.

| config | efSearch (configured → used) | build s | recall vs exact | Recall@200 macro | Recall@200 micro | new-in-period Recall@200 macro | short of K1 | mean #candidates | search ms/query (batch) |
|---|---|---|---|---|---|---|---|---|---|
| exact | — | 0.0 | 1.0000 | 0.0694 | 0.0731 | 0.0245 | 5 | 200.0 | 0.537 |
| hnsw_ef200 | 200 → 450 | 23.4 | 0.9354 | 0.0688 | 0.0725 | 0.0265 | 5 | 200.0 | 0.619 |
| hnsw_ef800 | 800 → 800 | 23.1 | 0.9740 | 0.0691 | 0.0727 | 0.0256 | 5 | 200.0 | 1.215 |
| ivf_nprobe4 | — | 2.3 | 0.3039 | 0.0455 | 0.0457 | 0.0283 | 1109 | 194.0 | 0.069 |
| ivf_nprobe32 | — | 2.3 | 0.7343 | 0.0684 | 0.0689 | 0.0341 | 17 | 200.0 | 0.154 |

Recall vs exact by segment:

| config | item_cold | item_tail | item_head | item_new_in_period | item_not_new | hist_1-2 | hist_3-5 | hist_6-10 | hist_11+ |
|---|---|---|---|---|---|---|---|---|---|
| exact | nan | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hnsw_ef200 | nan | 0.9336 | 0.9367 | 0.9359 | 0.9351 | 0.9344 | 0.9358 | 0.9382 | 0.9278 |
| hnsw_ef800 | nan | 0.9733 | 0.9745 | 0.9744 | 0.9738 | 0.9723 | 0.9740 | 0.9752 | 0.9732 |
| ivf_nprobe4 | nan | 0.3063 | 0.3022 | 0.2988 | 0.3064 | 0.3050 | 0.3054 | 0.3057 | 0.2913 |
| ivf_nprobe32 | nan | 0.7378 | 0.7317 | 0.7354 | 0.7338 | 0.7264 | 0.7316 | 0.7406 | 0.7417 |

Cross-check: FAISS exact filtered top-200 vs `data/processed/candidates/two_tower_text_seed0/val.npz` (first 200 columns, torch exact search): mean overlap 0.99991, rows with identical set 0.9879, identical order 0.3621, same target position 0.99979; target found by ours only / reference only: 0 / 0; 5 rows are short of K1 here (buffer exhausted; the reference ranks the full catalogue); 53 other rows differ in set, max |score − K1-th score| over their mismatched items 8.94e-08 (108 items, i.e. boundary ties). Order differences within a set are tied scores.

Search ms/query is batch throughput (4096-query batches, single-threaded FAISS on macOS), not the §5 latency protocol.
