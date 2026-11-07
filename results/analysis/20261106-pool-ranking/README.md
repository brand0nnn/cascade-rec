# Fixed bce_text ranker on alternative candidate pools (Val only, exploratory)

**Status label: Val only, exploratory.** Reused Val population (232,904 events / 124,416 users), exposed in many
earlier studies. No Test file is opened, loaded or scored. Nothing is selected (no pool, K or ratio choice).

Question: holding the ranker fixed (the registered `bce_text` residual recipe from
`src/cascade/pipelines/full_data_ranking.py`), how does the candidate pool change what the ranker delivers?

## Pre-stated hypothesis (written 2026-11-06 02:08 +08, before any pool-ranking fit)

The ranker on the fused pool beats the ranker on P1 overall by clearly more than about 0.0003 nDCG@10, mostly through
warm targets; new-item quality may fall. A larger natural training set (P2/P3 have about 38.8k vs 20.7k natural B
positives, skewed warm) may also help.

## Plan (registered before fitting)

- Pools (read-only, built by pools-build): P3 fused (round-robin interleave, RRF skip score, k = 60) and P2 two-tower
  natural top 200. P1 = rebuilt `bce_text` evaluation in `results/local_experiments/20261103-full-promising-ranking/`.
- P3t (lead's tie-break variant, RRF − 1e-6 × interleave position) is gated: its epoch-0 Val nDCG@10 must reproduce the
  P3 interleave-order value 0.007408. See `p3t_gate.json` / `interim_P3t.md`.
- Recipe unchanged, imported from `full_data_ranking`: GraphLatentRanker R2, graph columns disabled, BCE, 50 uniform
  real non-target negatives per natural list, Adam 1e-3, batch 32 lists, ≤10 epochs, patience 2, chronological inner
  split (same `split_cutoff`), epoch 0 eligible, all-B refit for the selected epoch count from the same initial state.
  Seeds 0/1/2. Nine common columns from the same pre-T0 `BatchedGraphFeatures`, standardised over all 200 slots of
  this pool's prefix B queries. Initial-state hashes must equal the rebuilt P1 `bce_text` fits; epoch 0 must equal each
  pool's retrieval order exactly (inner tune split and full Val).
- Metrics: per-user macro over the full Val population (misses = 0), seed-mean event metrics, paired user bootstrap
  (1,000 draws, seed 0): nDCG@10, Recall@10, HR@3, MRR@10, conditional nDCG@10, five-way funnel; overall / new
  (target first_ts ≥ T1) / warm.
- Code: `src/cascade/pipelines/pool_ranking.py`; tests: `tests/test_pool_ranking.py`.
  Bulk: `results/local_experiments/20261106-pool-ranking-<pool>/`.

## P3t dropped (lead decision, 2026-11-06, after the gate failed)

The P3t gate failed (`p3t_gate.json`, `interim_P3t.md`). Epoch-0 Val nDCG@10 on the same P3 pool was RRF pessimistic
0.006555, P3t literal formula 0.007044, pure lexicographic tie-break 0.007041, interleave order 0.007408 (required).
B: 0.009973 / 0.010321 / 0.010314 / 0.011023. The lead confirmed the P3t definition was inconsistent and **dropped P3t
entirely**. No P3t archive was created and nothing was fitted. Runs continue as registered: P3 (RRF score, pessimistic
ties) seeds 0/1/2, then P2 seeds 0/1/2. A lexicographic tie-break variant may be requested separately; it has not been started.

## Results (Val only, exploratory; computed 2026-11-06 after the hypothesis above)

Full tables: `tables.md`; CSVs `metrics.csv`, `paired_differences.csv`, `funnel.csv`, `fits.csv`, `inner_curves.csv`;
`report_provenance.json`. Logs `results/logs/20261106/pools-fit-*.log`. 232,904 events / 124,416 users, per-user macro,
misses = 0, ranker values = seed means; intervals = 1,000 paired-user draws (seed 0) on seed-mean event metrics, unadjusted.

### Checks (all pass)
- Archives: P3/P2 events, users, targets and histories equal the P1 archive. There is no padding, no duplicates and no
  unreleased candidates, and the builder raised no history-candidate error on any row. Stored retrieval_rank and top10
  reproduce from the stored score. P3 columns are unsorted in 284,143 B rows and 199,186 Val rows, so every rank comes
  from pessimistic counts and stable-argsort top 10.
- Features: the same builder on P1 candidates with the P1 scaler reproduces the stored P1 common features exactly
  (max abs error 0.0, 256 B + 256 Val queries). Batched vs reference builder error is ≤ 3.7e-9.
- Initial-state sha256 equals the rebuilt P1 bce_text fit for every pool and seed (b71551…/5fa012…/a39a90…).
- Epoch 0 equals each pool's retrieval order exactly on the inner tune split (logits, ranks, top 10) and on full Val
  (P3 0.006555, P2 0.005642). Selected-epoch replay, portable reload and outer score replay are exact.
- Decomposition inputs (`20261106-pool-ranking-P1onP3/checks.json`): on 24,396,323 overlapping (event, item) slots the
  recomputed recency skip score equals the P1 archive bit-for-bit, and the re-standardised features are within 2.4e-7.
  P1 checkpoints replay their stored outer logits exactly (max diff 0.0). The recency order on P3 candidates gives
  Val nDCG@10 0.006284, equal to P1 retrieval.

### Selected epochs (seeds 0/1/2): P1 8/8/7, P3 7/8/9, P2 5/10/9. B natural positives: P1 20,676, P3 38,861, P2 38,793.

### Headline nDCG@10 (overall / new / warm)
| System | overall | new | warm |
|---|---:|---:|---:|
| P1 retrieval / ranker | 0.006284 / 0.008141 | 0.007611 / 0.008666 | 0.005606 / 0.007887 |
| P3 retrieval (RRF) / ranker | 0.006555 / 0.011019 | 0.004106 / 0.006965 | 0.007591 / 0.012684 |
| P2 retrieval / ranker | 0.005642 / 0.010081 | 0.000361 / 0.003698 | 0.007983 / 0.012748 |
| P1-trained ranker on P3 lists (no training) | 0.008811 | 0.005347 | 0.010308 |

### Key contrasts, nDCG@10 [95% CI]
| Contrast | overall | new | warm |
|---|---|---|---|
| P3 ranker − P1 ranker | +0.002878 [+0.002544, +0.003211] | −0.001701 [−0.002071, −0.001314] | +0.004797 [+0.004367, +0.005231] |
| P2 ranker − P1 ranker | +0.001940 [+0.001529, +0.002325] | −0.004967 [−0.005479, −0.004419] | +0.004861 [+0.004341, +0.005375] |
| P3 ranker − P2 ranker | +0.000938 [+0.000654, +0.001238] | +0.003267 [+0.002839, +0.003691] | −0.000064 [−0.000422, +0.000263] |
| P3 ranker − P3 RRF order | +0.004464 [+0.004088, +0.004796] | +0.002859 [+0.002386, +0.003317] | +0.005092 [+0.004609, +0.005559] |
| P3 ranker − P3 interleave order | +0.003611 [+0.003229, +0.003970] | +0.001478 [+0.000981, +0.001958] | +0.004538 [+0.004065, +0.004993] |
| P2 ranker − P2 order | +0.004439 [+0.004081, +0.004778] | +0.003338 [+0.003026, +0.003680] | +0.004765 [+0.004291, +0.005262] |
| P1 ranker − P1 order | +0.001857 [+0.001554, +0.002164] | +0.001055 [+0.000540, +0.001544] | +0.002280 [+0.001900, +0.002627] |
| (i) P1 ranker on P3 lists − P1 ranker | +0.000670 [+0.000397, +0.000947] | −0.003319 [−0.003595, −0.002999] | +0.002421 [+0.002048, +0.002785] |
| (ii) P3 ranker − P1 ranker on P3 lists | +0.002208 [+0.001974, +0.002430] | +0.001618 [+0.001356, +0.001884] | +0.002376 [+0.002078, +0.002675] |

Recall@10, HR@3 and MRR@10 move in the same direction for every contrast above (see `tables.md`).

### Decomposition and same-candidate subset
- The P1 checkpoints can be scored validly on P3 lists if their own input function is kept: the recency-cosine skip
  score is recomputed for every P3 candidate and the features are re-standardised with the P1 scaler. Feeding them the
  RRF skip would be invalid (different scale and order), so that was not done. Read on the path P1 ranker → P1 ranker
  on P3 lists → P3 ranker: (i) the better candidates alone give +0.000670 overall (warm +0.0024, new −0.0033), and
  (ii) training on P3 lists gives +0.002208, positive on both new and warm. Step (ii) bundles the larger and different
  natural training set with the change of skip function (RRF instead of cosine), the per-pool scaler and new plans, so
  it is not a pure training-set-size effect. The split depends on the order of the path; it is not an exact causal
  attribution.
- Same-candidate subset (11,047 Val events whose target is in both the P1 and P3 top 200): the P3 ranker **loses** to
  the P1 ranker, −0.042619 [−0.045990, −0.039311] (new −0.024, warm −0.054). The P1 checkpoints on P3 lists lose even
  more (−0.064608). On events P1 already covers, the 2× longer mixed list holds more strong two-tower distractors and
  costs rank. The whole overall gain comes from about 9,400 targets that only P3 retrieves (mostly warm, two-tower
  only). This subset is selected on retrieval outcomes, so it favours recency-friendly events (P1 retrieval nDCG@10
  0.153 vs P3 RRF order 0.123 there).

### Funnel (seed-mean counts, all events; vs own pool order)
| Ranker | retrieval miss | unrescued | rescued | demoted | retained | top-10 hits |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 217,988 | 9,325 | 2,454 | 1,568 | 1,569 | 4,023 |
| P3 | 212,463 | 13,296 | 3,534 | 1,877 | 1,734 | 5,268 |
| P2 | 216,231 | 10,753 | 3,307 | 1,308 | 1,305 | 4,612 |
| P1 ckpt on P3 | 212,463 | 14,024 | 2,806 | 2,073 | 1,538 | 4,344 |

### Hypothesis check
- P3 ranker beats the P1 ranker by clearly more than 0.0003 (+0.0029), mostly through warm targets (+0.0048), and new
  items fall (−0.0017). Consistent.
- The decomposition is consistent with the larger and different training set mattering (step (ii) +0.0022 vs (i)
  +0.0007), but (ii) is confounded with the skip-function change.
- Not anticipated: (1) P2 ranker beats the P1 ranker overall (+0.0019) even though P2 retrieval is worse than P1
  (−0.0006). (2) P3 and P2 rankers tie on warm targets; P3's edge over P2 is entirely new items. (3) On the
  same-candidate subset the P3 ranker is much worse than the P1 ranker.

### Caveats
Reused, previously exposed Val; unadjusted intervals; three training seeds. The RRF skip score has a much smaller
spread than the cosine scores (top1−top10 ≈ 0.015 vs ≈ 0.05–0.07), so the residual head dominates more on P3. That is
a property of the registered P3 pool, not separated here. The `both_*` rows in `tables.md` for P2 systems use the same
P1∩P3 event subset and are not a P2 same-candidate analysis. `pool_ranking.py` was extended after evaluation (cross_score,
report, post-evaluation verify). Every function used by prepare/fit/evaluate is checked identical to each study's frozen
source copy before reporting. No Test file was opened; nothing is selected.
