# Index sweep (val, K1 = 200)

Retriever `results/retriever/two_tower_text/seed0`; 232904 instances, 124416 users; 111480 items indexed; FAISS 1.15.1 (1 thread(s)), Apple M2 Pro. Search asks for K1 + buffer = 450 results, then §4.6 filtering.

| config | efSearch (configured → used) | build s | recall vs exact | Recall@200 macro | Recall@200 micro | new-in-period Recall@200 macro | short of K1 | mean #candidates | search ms/query (batch) |
|---|---|---|---|---|---|---|---|---|---|
| exact | — | 0.0 | 1.0000 | 0.0687 | 0.0716 | 0.0261 | 80 | 200.0 | 0.413 |
| hnsw_ef200 | 200 → 450 | 19.9 | 0.9361 | 0.0671 | 0.0702 | 0.0272 | 125 | 200.0 | 0.530 |
| hnsw_ef800 | 800 → 800 | 19.7 | 0.9745 | 0.0680 | 0.0710 | 0.0265 | 108 | 200.0 | 1.092 |
| ivf_nprobe4 | — | 2.3 | 0.3089 | 0.0434 | 0.0466 | 0.0288 | 51709 | 194.4 | 0.064 |
| ivf_nprobe32 | — | 2.3 | 0.7392 | 0.0643 | 0.0676 | 0.0309 | 471 | 200.0 | 0.142 |

Recall vs exact by segment:

| config | item_cold | item_tail | item_head | item_new_in_period | item_not_new | hist_1-2 | hist_3-5 | hist_6-10 | hist_11+ |
|---|---|---|---|---|---|---|---|---|---|
| exact | nan | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hnsw_ef200 | nan | 0.9342 | 0.9373 | 0.9375 | 0.9353 | 0.9321 | 0.9366 | 0.9383 | 0.9333 |
| hnsw_ef800 | nan | 0.9738 | 0.9750 | 0.9753 | 0.9741 | 0.9711 | 0.9745 | 0.9763 | 0.9745 |
| ivf_nprobe4 | nan | 0.3115 | 0.3072 | 0.3073 | 0.3097 | 0.3123 | 0.3079 | 0.3111 | 0.3043 |
| ivf_nprobe32 | nan | 0.7413 | 0.7377 | 0.7426 | 0.7375 | 0.7257 | 0.7359 | 0.7475 | 0.7515 |

Cross-check: FAISS exact filtered top-200 vs `data/processed/candidates/two_tower_text_seed0/val.npz` (first 200 columns, torch exact search): mean overlap 0.99994, rows with identical set 0.9911, identical order 0.3650, same target position 0.99969; target found by ours only / reference only: 0 / 0; 80 rows are short of K1 here (buffer exhausted; the reference ranks the full catalogue); 1999 other rows differ in set, max |score − K1-th score| over their mismatched items 2.53e-07 (4054 items, i.e. boundary ties). Order differences within a set are tied scores.

Search ms/query is batch throughput (4096-query batches, single-threaded FAISS on macOS), not the §5 latency protocol.
