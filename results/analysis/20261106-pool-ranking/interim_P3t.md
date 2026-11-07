# Interim: P3t gate FAILED, P3t not built or fitted (Val only, exploratory)

Written 2026-11-06 ~02:15 +08, before any pool-ranking fit. Record: `p3t_gate.json`
(`python -m cascade.pipelines.pool_ranking --phase check_p3t`; log `results/logs/20261106/pools-fit-p3t-gate.log`).

Registered gate: P3t (score = P3 RRF score − 1e-6 × 0-based interleave position) must reproduce, at epoch 0, the
P3 interleave-order Val nDCG@10 of 0.007408 (`results/analysis/20261106-pools/tables.md`). It does not:

| Order on the same P3 pool | Val nDCG@10 (macro) | B nDCG@10 (macro) |
|---|---:|---:|
| RRF score, pessimistic ties (P3 as stored) | 0.006555 | 0.009973 |
| P3t (RRF − 1e-6 × position) | 0.007044 | 0.010321 |
| RRF with a pure lexicographic tie-break by interleave position | 0.007041 | 0.010314 |
| Interleave order itself | 0.007408 | 0.011023 |

Why the three instructions are mutually inconsistent:
1. **RRF order ≠ interleave order even without ties.** Items found by both sources get two RRF terms and move above
   single-source items placed earlier by the interleave. Only 14.5% of Val rows have P3t order equal to interleave order;
   14,579 of 20,441 natural Val targets get a different rank.
2. **The 1e-6 × position offset is not a pure tie-break.** Its range (up to 1.99e-4) exceeds many gaps between distinct
   RRF sums (≈1/(60+r)² ≈ 1.5e-5–4e-5 deep in a list), so it also reorders non-tied items: the full order differs from
   the lexicographic tie-break in 69% of Val rows (1,020 target ranks differ).

Action taken: per the stop rule, the P3t archive was **not** created (no folder under
`results/local_experiments/20261106-pools/candidates/`) and P3t was not fitted. P3 and P2 are unaffected by this
inconsistency and proceed as registered. A future P3t needs the lead to choose one definition: (a) pure lexicographic
tie-break (epoch-0 Val 0.007041), (b) a skip score whose order equals the interleave (epoch-0 0.007408; that is a
different scoring rule, not a tie-break), or (c) the literal formula (0.007044).
