# Step A0: candidate pools for a fixed ranker (Val only, exploratory)

**Status label: Val only, exploratory.** Reused validation population (232,904 events / 124,416 users), previously
exposed in many studies. No Test file is opened, loaded or scored. No ranker is trained. Nothing is selected here
(no K, pool or ratio choice).

## Pre-stated hypothesis (written 2026-11-06 before any pool was built or scored)

- P3 (fused recency + two-tower) micro Recall@200 is about 8.5-9%, well above P1 (recency) at 6.4%.
- P3 loses some new-item recall relative to P1 and gains much more on warm targets.
- The target-conditioned retriever oracle (better of recency and two-tower per event) is far above both.

## Results (Val only, exploratory; computed 2026-11-06 after the hypothesis above)

Code: `src/cascade/analysis/pools_build.py` (steps `verify`, `regen`, `fuse`, `filters`, `report`, `provenance`);
tests: `tests/test_pools_build.py` (5 pass). Logs: `results/logs/20261106/pools-build-*.log`.

### Pool archives (same 8-file format as the recency study: cand, score, inst, user, target, retrieval_rank, top10, history)

| pool | B (ranker_train) | Val |
|---|---|---|
| P1 recency (reference, not copied) | `results/local_experiments/20261103-full-promising-ranking/candidates/ranker_train/` | `.../candidates/val/` |
| P2 two-tower (float32 regenerated) | `results/local_experiments/20261106-pools/candidates/P2_two_tower/ranker_train/` | `.../P2_two_tower/val/` |
| P3 fused (interleave + RRF score) | `results/local_experiments/20261106-pools/candidates/P3_fused_rrf/ranker_train/` | `.../P3_fused_rrf/val/` |

Analysis-only extras are kept outside the archive folders under `results/local_experiments/20261106-pools/aux/`
(P3 per-candidate source ranks in recency/two-tower, 0 = absent; P3 interleave-position target rank; P2 full-catalogue
target rank and the saved-float16 pessimistic rank).

P3 note: `cand` is stored in interleave order and `score` is the RRF sum, so `score` is not sorted along columns.
`retrieval_rank`/`top10` follow the stored score (pessimistic ties, stable tie-break = interleave order), the
same rule the recency archive uses. RRF creates systematic exact ties (single-source items at equal source rank), so
RRF-score order is worse than the interleave order itself; both are reported.

### Verification

- `verify_saved_two_tower.json`: saved two-tower B lists: columns 0-199 natural, score non-increasing, no padding or
  duplicates; 276,321 forced rows (0.8769) all have the target at column 200 and not in 0-199; 38,793 non-forced
  rows have the target in 0-199 at `target_pos`; `target_rank` = pos+1 for 38,584, larger by ties for 209, never
  smaller. inst/user order equals the recency archive exactly for both splits. PASS.
- `regen_two_tower.json`: float32 regeneration on MPS from `results/retriever/two_tower_text/seed0/model.pth`.
  B: 0 differing positions (bit-identical order and float16 values). Val: 474,783 differing positions in 145,075 rows
  (1,875 rows / 1,899 items differ at the 200 boundary), every one an exact float32 tie (max regenerated score gap
  at a differing position = 0.0); float16 casts match the saved scores at every position; full-catalogue target
  ranks match for every event; natural-hit membership identical (B 38,793; Val 16,673). The Val swaps follow from
  the saved list using top-500 versus top-200 here. PASS.
- `filter_check.json`: 10,000 seeded rows per split (seed 0): P1, P2, P3 and the saved two-tower top 200 have zero
  padding, duplicates, history items (full history before t) and unreleased items; archive histories equal harness
  histories. Every P2 row also passed `check_candidates` during regeneration. PASS.
- Recomputed `retrieval_rank`/`top10` from stored scores reproduce every archive exactly (P1, P2, P3; both splits).

### Tables

See `tables.md` (all rows) and `pool_report.json`. Headline (per-user macro over 124,416 users; micro in brackets):

| pool | R@200 overall | R@200 new | R@200 warm | nDCG@10 overall | nDCG@10 new | nDCG@10 warm |
|---|---|---|---|---|---|---|
| P1 recency | 0.054746 (0.064044) | 0.065043 | 0.050574 | 0.006284 | 0.007611 | 0.005606 |
| P2 two-tower | 0.068716 (0.071587) | 0.026113 | 0.087430 | 0.005642 | 0.000361 | 0.007983 |
| P3 fused, RRF order | 0.079521 (0.087766) | 0.057154 | 0.089439 | 0.006555 | 0.004106 | 0.007591 |
| P3 fused, interleave order | same pool | | | 0.007408 | 0.005487 | 0.008145 |
| oracle retriever (better of P1/P2 order per event) | | | | 0.011705 | 0.007889 | 0.013320 |
| candidate oracle (target in either top 200) | 0.112394 (0.123626) | 0.083060 | 0.125780 | | | |

B natural positives (of 315,114 rows): P1 20,676; P2 38,793; P3 38,861; either top 200 53,115; both 6,354.

The oracles are target-conditioned upper bounds only, not deployable systems.

### Hypothesis check

- P3 micro Recall@200 0.0878: inside the stated 8.5-9% range, above P1 0.0640. Consistent.
- P3 new-item Recall@200 is lower than P1 (−0.0079 macro [−0.0091, −0.0067]); warm gains +0.0389 [+0.0375, +0.0402]. Consistent.
- Oracle far above both on nDCG@10 overall (0.0117 vs 0.0063/0.0056) and candidate oracle R@200 0.112 vs 0.080.
  Not uniformly: on new items the oracle adds only +0.00028 nDCG@10 over recency, because two-tower rarely wins there.
