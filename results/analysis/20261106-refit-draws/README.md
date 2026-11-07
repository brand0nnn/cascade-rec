# Refit-draw noise for BCE co-review vs text (Val only, exploratory)

## Pre-stated hypothesis (written 2026-11-05T18:28:47Z, before any new plan was drawn or any fit started)

Under T0 (stale pre-T0 graph features, as trained), the co-review − text nDCG@10 gain pooled over
training draws is small (+0.0001 to +0.0002) with a 95% paired-user CI that may include zero. Under
period-start T1 features it is positive with a CI above zero. The draw-to-draw SD of each method's
Val nDCG@10 is about 0.0002.

## Registered design (before fitting)

- Methods: `bce_text` (graph columns masked) and `bce_coreview` (four optional co-review columns on),
  BCE objective, recipe of results/local_experiments/20261103-full-promising-ranking and
  results/local_experiments/20261104-objective-coreview, unchanged: same K200 frozen-recency candidates,
  scaled features, chronological B split, initial state per seed (asserted equal to the rebuilt fits'
  initial_state_sha256), Adam 1e-3, 32 lists, 50 negatives, <=10 epochs, patience 2, epoch 0 eligible,
  fixed-epoch all-B refit.
- Only change: a draw ID on the negative-sampling / epoch-order RNG of the inner and refit plans.
  Draw 0 = `np.random.default_rng(seed)` (the existing plans, checked equal array-for-array);
  draw d >= 1 = `np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(d,)))`.
  Torch initial state and dropout seeding are unchanged.
- Draws 1, 2, 3; seed 0 first (6 fits), then seeds 1 and 2 (12 fits) if not stopped.
- Val scoring on CPU of the 14,916 natural K200 hit rows (misses keep rank inf), twice: saved T0 features and
  the freshness study's period-start T1 features (results/local_experiments/20261106-freshness-val/features/T1).
- Primary pooled contrast: draws 1–3 (new draws). Draw 0 (existing, already-inspected fits) is reported separately
  and in a secondary pool 0–3.
- Metrics: nDCG@10, Recall@10, HR@3, MRR@10, user macro over all 232,904 Val events; groups all/new/warm
  (new = target first_ts >= T1). Contrasts: paired user bootstrap, 1,000 draws, seed 0, after averaging event
  metrics over seeds x draws. Draw-to-draw SD: ddof=1 across draws within each seed.
- Nothing is selected. No Test file is opened.

## Checks completed

- Unit tests: tests/test_refit_draws.py (4 pass). They show draw 0 equals `graph_latent_study.negative_plan` on toy data and on the saved full seed-0 inner plan. They also show new draws keep the eligible set, put the positive first, and draw 50 unique real non-target negatives.
- Prepare: the draw-0 regeneration equals all six saved parent plans array-for-array (seeds 0/1/2 × inner/refit). Draws 1–3 are saved under the bulk study's plans/ folder.
- Draw-0 scoring check: CPU scoring of the existing seed-0 bce_text and bce_coreview checkpoints reproduces the freshness study's saved T0 and T1 ranks exactly (draw0_scoring_check.json).
- Every fit asserts an initial_state_sha256 equal to the rebuilt bce_text and bce_coreview fits. It also asserts that epoch 0 equals retrieval, that the selected inner checkpoint replays and that the portable state reload is exact.
- Sample CPU smoke: results/local_experiments/20261106-refit-draws-sample-smoke. The draw-0 CPU control matches the sample parent's MPS fit at epoch 0. At epoch 1 the loss differs only by device-level float noise (0.992450 vs 0.992437).
- Code note: after prepare, only the analysis/plumbing of refit_draws.py changed (no plan or fit logic). All fits carry the same `refit_draws_sha256`, and a snapshot of that source is saved in the bulk study's source/ folder.

## Status

- Seed 0 × draws 1–3 × {bce_text, bce_coreview}: 6 fits done on MPS between 04:05 and 05:01. The interim table is interim_seed0.md.
- Seeds 1–2 have not been fitted yet; the lead asked for a report first.

## Pre-stated hypothesis for the 3-seed pool (written 2026-11-06T05:37:50Z, after the seed-0 interim and before any seed-1/2 fit or the final analysis)

Pooled over seeds 0–2 × draws 1–3: under T1 the co-review − text nDCG@10 gain is positive with a CI above zero.
Under T0 it is smaller, and its CI may include zero.
