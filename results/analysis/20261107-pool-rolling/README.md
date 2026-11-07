# P3 fused pool x rolling monthly co-review features (Val only, exploratory)

Written before any feature build, fit or score (7 Nov 2026, ~01:05 SGT).

**Question.** Combine the two strongest consistent effects from 6 Nov: the fused candidate pool P3
(`20261106-pools/candidates/P3_fused_rrf`) and rolling monthly co-review/common features
(`20261106-rolling-ranking`). Does the `bce_coreview` recipe, trained on P3 with rolling monthly B snapshots
and served on P3 with Val monthly snapshots, beat P3 `bce_text` served at T1 (0.011451) and P1 rolling
`bce_coreview` (0.010613)?

**Pre-stated expectation.** Yes on both, sub-additively (the pool gain under rolling co-review is smaller than
the pool gain measured with `bce_text` at T0). Co-review − text on P3 under rolling features is positive
(only measurable if the H3 text control runs).

**Design.** P3 candidates, P3 chronological prefix/tune split and P3 negative plans (inner = prefix natural,
refit = all-B natural), unchanged. Nine common + six graph features rebuilt per query from all reviews strictly
before its UTC month start (B queries 26-31 Jul 2020 use T0; Val Aug 2021 queries use 1 Aug 2021), query user
excluded, same cut-rule and reference checks as `rolling_ranking`. Prefix scalers refitted on the rolling P3
prefix rows (all 200 slots). Same recipe (BCE, 50 negatives, Adam 1e-3, 32 lists, <=10 epochs, patience 2,
epoch 0 eligible, fixed-epoch all-B refit). `bce_coreview` = graph columns 0,1,2,5 enabled (mask
`pairwise_coreview`) with the BCE objective, as in `objective_coreview`. Initial state must equal the P3
`bce_text` fits and the `bce_coreview` source fits (same model; only the mask differs). Fits on MPS,
evaluation on CPU over natural-hit rows (misses rank = inf).

**Metrics.** Per-user macro over 232,904 Val events / 124,416 users, seed mean (0/1/2), paired user bootstrap
(1,000, seed 0) on seed-mean event values. nDCG@10, Recall@10, HR@3, MRR@10; all / new-in-Val / warm.

No Test file is opened; nothing is selected. Results: `report.md` (written automatically by `analyse`).
