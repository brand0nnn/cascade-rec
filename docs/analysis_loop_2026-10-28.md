# Exploratory analysis loop — 28/29 Oct 2026 (Val only, no training)

**Final summary at the top; the log below is chronological.**

**Scope:** exploratory analyses of saved Val outputs and frozen models. No Test, no training, no K₁/index selection,
no commits. Orchestrator edits this file, `docs/decisions.md` and `docs/evaluations.md`; subagents write only their
own scripts, tests and `results/analysis/<run-id>/`. Hard stop: no new task after 00:00; summary by 00:15.

## Final summary (23:40)

All four paths closed by 23:40; every result was reviewed, spot-checked against saved outputs, and pytest stayed
green (171 passed). Val only, no training, no Test, nothing committed, no setting selected.

### Findings, ranked by strength of evidence

1. **The frozen retriever is a popularity model in disguise (strong; exp 1b, 3, 3b).** 71% of its top-10 slots go
   to the top-1% products (coverage 2.6%); 82% of its @200 hits are head targets. Warm-target recall is governed by
   popularity *in the retriever's training window* (Recall@200 0.009 at 1–4 training reviews → 0.84 at 1,000+;
   Recall@10 ≈ 0 below 100). This links earlier results: natural ranker positives are 84% head because the
   retriever's hits are; approximate search and larger K₁ help new/tail because they loosen a head-heavy pool.
2. **A zero-training "similar to the last product" scorer beats the trained retriever on new and tail products
   (strong; exp 3).** New-in-period Recall@10 0.0110 vs 0.0009 (~12×) and Recall@200 0.049 vs 0.026; warm-tail
   Recall@200 0.050 vs 0.015. The hits are the same product line for another phone model or colour (legitimate).
   The two scorers find mostly different targets (14.6% overlap). Relevant to the frozen-retriever decision, which
   chose text for new products.
3. **Serving-time popularity correction trades head accuracy for new/tail coverage, end to end (strong; 3b, 3c).**
   Ranking candidates by u·v − β·τ·log q (unseen products given the rarest seen log q) at β = 0.25 doubles the
   new/tail Recall@200 ceiling for −5% overall. Through the natural rankers: DIN nDCG@10 new +0.00112 (+44%),
   tail +0.00079 (+96%), head −0.00231 (−13%), overall −0.00057 [−0.00078, −0.00038] (−5.8%); the MLP loses more
   overall (−12.8%). The better ranker absorbs more of the retriever's @10 loss, but the rankers rank newly admitted
   new products less well (conditional nDCG@10 for new falls) — they were trained on head-heavy natural positives.
   A handling detail matters: with the training convention (log q = 0 for unseen) new products disappear entirely.
4. **The forced-list ranker learned two score responses (moderate–strong; exp 4).** Its logit is U-shaped over
   retrieval rank (boost at the top and at the bottom), whereas natural rankers fall monotonically; natural training
   *increases* reliance on the retrieval score and text cosine. This refines the overnight shortcut story: the
   bottom-of-list boost is the shortcut, and it takes over at K₁ = 500. All rankers rely mostly on candidate identity.
5. **"Staleness over Val months" is mostly composition, not model decay (moderate; exp 1).** The crude Recall@200
   fall 0.089 → 0.053 comes from a rising new-product share (17% → 45%) and a growing catalogue; a real residual
   decline remains only for Val-released products (−0.015 Q4−Q1).
6. **Same-store targets are retrieved 2–3× better, via text similarity (moderate; exp 2, 2b).** At equal text
   similarity to the last product the store lift disappears; cross-category targets are not over-represented among
   deep misses, so "the text retriever misses complements" does not explain the far misses.

### Closed paths and why
- Exp 1 (staleness): answered after 1b — composition plus training-window popularity; the remaining mechanism check
  was covered by 3b.
- Exp 2 (similar vs complementary): answered after 2b — text similarity explains the store effect; crowding and
  history-centroid splits are second-order.
- Exp 3 (last product → serving correction → end to end): closed after 3c — the trade-off is measured end to end;
  going further means choosing β, which is a retrieval-setting selection (needs δ and the team).
- Exp 4 (permutation importance): answered — hypothesis rejected; correlated text inputs limit finer attribution.

### Caveats
- One retriever seed; Val only; exploratory; CIs are user-bootstrap unless stated (item-cluster in 1b) and
  unadjusted for the many comparisons.
- β values were explored on Val; they are probes, not selections. Any adoption would need the team, δ and a fresh
  confirmation before the single Test run.
- The serving correction's "minseen" convention for unseen products is our own choice.
- Permutation importance cannot separate the correlated text-derived inputs.
- Git hashes are recorded as `-dirty` because the new analysis modules are untracked.
- `results/analysis/20261028-debiased-lists/` holds ~0.84 GB of `.npz` lists; `.npz` is not git-ignored, so do
  not `git add` that folder (move or delete it after review).

### Suggested next steps (for the team)
1. Discuss finding 1–3 together: the retriever's popularity skew is the common cause behind the new-product
   trade-offs seen everywhere (natural lists, IVF, K₁). Decide whether new-product performance gets explicit weight
   in δ / the final selection rule (44% of Test targets are new).
2. If the team wants a retrieval-side fix, the cheapest options are serving-time correction (no training) or mixing
   last-product candidates into the pool — the latter is multi-source merging (out of scope unless agreed).
3. For the ranker: the conditional-nDCG drop on newly admitted new products suggests training lists with more
   new/tail positives (e.g. deeper K_train or debiased training lists) — a training-label change needing sign-off.

## Status

| # | Experiment | Subagent | Run ID | State | Verdict |
|---|---|---|---|---|---|
| 1 | Retriever staleness | exp1-staleness | `20261028-staleness-r2`, `-staleness-popularity` | **closed** after 1b | mixed: recall falls across Val mostly via mix + catalogue growth; real decline only for new products; popularity (training-window) is the driver |
| 2 | Similar vs complementary targets | exp2-complements | `20261028-complements`, `-mechanism-r2` | **closed** after 2b | same-store lift = text similarity to the last product; retriever also has a popularity route |
| 3 | Last-product-only baseline | exp3-lastitem | `20261028-last-item`, `-logq-serving`, `-debiased-lists`, `index_rank_eval/20261028-debiased-rank-{natural,din}` | **closed** 23:40 after 3c | **refuted, but surprising**: last-item reaches 62% overall yet beats the frozen retriever on new/tail products |
| 4 | Ranker permutation importance | exp4-importance | `20261028-permutation-importance`, `-r2` | **closed** 23:38 | hypothesis rejected: natural training *restores* use of the retrieval signal; forced models' rank response is U-shaped |

## Hypotheses (recorded before running)

1. **Staleness.** The frozen retriever trained on reviews before 26 Jul 2020. Prediction: Val Recall@200 falls
   across Val months (Aug 2021 → Jul 2022), and targets first reviewed after 26 Jul 2020 are retrieved worse than
   older ones, independent of item popularity. Null: flat recall across months once the product-age mix is held
   fixed.
2. **Similar vs complementary.** The text retriever finds products that look like the user's last product.
   Prediction: Recall@200 is much higher when the target shares the last history product's leaf category (and/or
   store) than when it does not; cross-category targets dominate the deep misses (rank > 5,000). Builds on Qing
   Rong's graph/warm-target analysis in `docs/evaluations.md` without repeating it.
3. **Last-product-only.** Prediction: scoring by the last history product's retriever vector alone gets most of the
   frozen retriever's Recall@200 (say ≥ 80%), with the gap growing little with history length — so the sequence
   model acts mostly like a last-product model, explaining flat recall across history buckets
   (0.0685 / 0.0695 / 0.0718 / 0.0662 for 1–2 / 3–5 / 6–10 / 11+).
4. **Feature importance.** Prediction: shuffling the retrieval score (and text cosine) hurts the forced MLP most —
   possibly *improving* it, given the inverted shortcut — while natural MLP/DIN rely more on history/category
   matching features.

## Log

- 23:02 Progress file created; hypotheses 1–4 recorded. Launched subagents for 1–3 concurrently.
- 23:06 **Exp 2 result** (`results/analysis/20261028-complements/`; reviewed `src/cascade/analysis/complements.py`,
  classifier + alignment checks; reproduces Qing Rong's same-leaf-last numbers 0.1008 / 0.0634 exactly; pytest 167
  passed). Frozen retriever Val Recall@200 (macro, 95% user-bootstrap CI) by target vs last history product:

  | Class | Share | R@200 | Median rank |
  |---|---|---|---|
  | same leaf & same store | 2.2% | 0.215 [0.204, 0.226] | 1,284 |
  | diff leaf, same store | 1.2% | 0.199 [0.185, 0.214] | 1,589 |
  | same leaf, diff store | 16.8% | 0.086 [0.082, 0.088] | 4,578 |
  | diff leaf & diff store | 75.9% | 0.061 [0.060, 0.063] | 8,304 |

  Holds within warm-head / warm-tail / new-in-period and after re-weighting to the overall item-type mix.
  **Verdict:** partly supported. Same leaf helps (1.6×), but the large lift is **same store** (~2×+, with or without
  a leaf match); same leaf with a different store is weak. **Refuted:** cross-category targets are *not*
  over-represented among deep misses (80.6% of rank > 5,000 vs 77.2% of all targets, 1.04×) — deep misses mirror
  the base rate, so "text misses complements" does not explain why most misses are far away. Given retrieval,
  the retriever's own top-10 rate is class-neutral (0.14–0.16) while the natural MLP's lift concentrates on
  same-store targets (+0.27 / +0.16) and is smallest for same-leaf/diff-store (+0.04).
  New vs Qing Rong: joint leaf × store split against the *last* item, depth distributions and deep-miss
  composition, ranking-given-retrieval by class. Caveat: "different leaf" is only a proxy for "complement"; same
  store with the last item covers just 3.4% of Val.
- 23:07 **Follow-up 2b** (same subagent), hypothesis recorded before running: the same-store lift comes from
  store/brand tokens in the item text. Predictions: target–last MiniLM and retriever-vector cosine is much higher
  for same-store pairs; Recall@200 rises steeply with target–last cosine; class adds little within a cosine decile.
  Run `20261028-complements-mechanism`.
- 23:08 Exp 4 started (exp 2 finished, so the "one of 1–3 done" condition holds; ~2.8 GB free, so the subagent was
  told to keep batches small). Added the forced DIN (`20261024-din-matching-seed{0,1,2}`) as an optional fourth model.
- 23:08 **Exp 3 result** (`results/analysis/20261028-last-item/`; reviewed `src/cascade/analysis/last_item.py`
  scorers; sanity re-score of 5,000 Val instances on MPS matches the saved CUDA ranks (98.3% exact, all hits equal);
  I independently recomputed R@10 by item type and the top-10 popularity concentration from the saved ranks;
  pytest 167 passed). Val, per-user macro, paired user bootstrap vs frozen retriever:

  | Group | Frozen R@10 | Last-item R@10 | Frozen R@200 | Last-item R@200 | Diff @200 [95% CI] |
  |---|---|---|---|---|---|
  | overall | 0.0104 | 0.0094 | 0.0687 | 0.0429 | −0.0258 [−0.0273, −0.0243] |
  | new-in-period | 0.0009 | **0.0110** | 0.0261 | **0.0490** | **+0.0229 [+0.0211, +0.0246]** |
  | warm-tail | 0.0004 | **0.0098** | 0.0153 | **0.0497** | **+0.0344 [+0.0322, +0.0365]** |
  | warm-head | 0.0207 | 0.0083 | 0.1180 | 0.0377 | −0.0802 [−0.0823, −0.0781] |

  Ratio last-item / frozen at @200 is flat (0.63–0.68) across history buckets. Hit overlap at @200 is small: last-item
  finds only 14.6% of the frozen hits, and 79.2% of its hits are ones the frozen retriever misses. Untrained mean-pool
  is worse (0.34 of frozen) and degrades with history length. **Popularity:** 71% of the frozen retriever's top-10
  slots go to the top-1% products (coverage 2.6% of the catalogue) vs 5% (coverage 79.8%) for last-item.
  **Verdict:** H3 refuted — the transformer is *not* a last-product model. But the result is a strong new finding:
  **the trained retriever is a popularity-heavy model, and a zero-training "similar to the last product" scorer is
  ~12× better at @10 and ~2× better at @200 on new-in-period products** (the segment Two-tower (text) was chosen for).
  This also links earlier results: natural ranker positives are 84% head because the retriever's hits are 82% head;
  approximate indexes and larger K₁ help new/tail because they loosen a head-heavy candidate set. Leakage check: both
  scorers use only history and frozen text vectors with the shared §4.6 filtering (history items excluded); a likely
  legitimate source is near-duplicate variants (checked in 3b).
- 23:09 **Follow-up 3b** (same subagent), hypothesis recorded before running: the head skew comes partly from
  serving with the raw score (logQ-corrected training lets the raw score absorb popularity), so subtracting
  β·log q at serving time (β ∈ {0, 0.25, 0.5, 1}; β = 0 must reproduce the frozen ranks) shifts recall toward
  new/tail without training. Probe only, no β selection. Also checks the near-duplicate explanation for last-item's
  new-product hits. Run `20261028-logq-serving`.
- 23:10 **Exp 1 result** (`results/analysis/20261028-staleness-r2/`, the run to cite; `-staleness` is the first,
  saved-ranks-only pass). Reviewed `src/cascade/analysis/staleness.py` (Val only, alignment and timestamp checks;
  inference-only re-score reproduces saved ranks, 100% hit agreement); pytest 167 passed.
  Crude Val Recall@200 falls month by month, 0.089 (Aug 2021) → 0.053 (Jul 2022). Decomposition:
  - **Mix:** the new-in-period share of instances rises 17% → 45% from Q1 to Q4; by era, R@200 is 0.143
    (products first reviewed before 26 Jul 2020) vs 0.028 (ranker-training era) vs 0.026 (Val era).
  - **Catalogue growth:** the released pool grows 94.6K → 102.8K; Val-released products take 0.1% → 12.6% of
    top-10 slots, pushing older targets down. Removing them as competitors ("frozen pool") makes the retriever-era
    and warm-head declines vanish (Q4−Q1 −0.001 and −0.003, CIs include 0).
  - **Real residual decline only for Val-era products:** −0.015 [−0.020, −0.011] Q4−Q1 even in the frozen pool;
    about a quarter of the crude fall survives both adjustments. Not explained by history length.
  - **Era vs popularity:** within warm-head, retriever-era vs ranker-era is 0.172 vs 0.039, but holding the total
    pre-T1 review count fixed the gap mostly disappears or reverses; holding only recent (ranker-period) popularity
    fixed, ranker-era products are much worse. Reading (inference): the retriever favours what was popular *in its
    own training window*.
  **Verdict:** mixed. "Staleness over months" is mostly composition and catalogue growth, not model decay; the
  substantive signal is popularity-in-the-training-window, consistent with exp 3's popularity-heavy finding.
  Caveat: user-only bootstrap CIs are over-confident for item-level contrasts.
- 23:11 **Follow-up 1b** (same subagent), hypothesis recorded before running: warm-target recall is explained
  mainly by popularity in the retriever-training period (0 for ranker-era products), not by current popularity
  or era; item-cluster bootstrap. Run `20261028-staleness-popularity`.
- 23:2x **Follow-up 2b result** (`results/analysis/20261028-complements-mechanism-r2/`; the first `-mechanism`
  folder had a wrong re-weighting table and is marked `SUPERSEDED.txt`; reviewed; spot re-score of 256 instances
  matches; pytest green).
  - Frozen MiniLM text similarity between target and last product orders the classes (same/same 0.82, diff
    leaf/same store 0.66, same leaf/diff store 0.61, diff/diff 0.37), and Val Recall@200 rises steadily with it
    (0.038 in the lowest decile → 0.165 in the highest).
  - **At equal MiniLM similarity the same-store lift disappears** (matched diff −0.003 [−0.014, +0.009]): it is a
    text-similarity effect (brand words / store tokens / product-line wording, not separated). At equal
    similarity, same-leaf/different-store targets are retrieved *less* than cross-category ones (−0.044) —
    possibly crowding by many similar same-leaf products (untested).
  - In the retriever's own vector space recall is **U-shaped** in target–last similarity: the least similar decile
    has high recall (0.146) because those targets are popular (median pre-T1 count 41 vs 5). So the retriever has
    two routes: text similarity to recent products, and popularity independent of similarity — consistent with exp 3.
  **Verdict / closed:** the store effect is explained (text similarity); further splitting (crowding, history
  centroid) is second-order next to the popularity finding, so this path is closed after two results.
- 23:15 **Follow-up 1b result** (`results/analysis/20261028-staleness-popularity/`; warm Val targets only,
  saved ranks, item-cluster bootstrap; reviewed; pytest green). Recall by the target's review count **in the
  retriever-training period**:

  | Training-period count | Items | R@10 | R@200 [item CI] |
  |---|---|---|---|
  | 0 (ranker-training era) | 10,042 | 0.0016 | 0.028 [0.025, 0.033] |
  | 1–4 | 3,705 | 0.0000 | **0.009** [0.007, 0.011] |
  | 5–19 | 5,421 | 0.0003 | 0.021 |
  | 20–49 | 2,809 | 0.0011 | 0.044 |
  | 50–99 | 1,396 | 0.0027 | 0.079 |
  | 100–299 | 1,076 | 0.020 | 0.207 |
  | 300–999 | 358 | 0.096 | 0.540 |
  | 1000+ | 69 | 0.373 | 0.837 [0.783, 0.877] |

  (a) supported: recall rises steeply with training-window popularity, and **Recall@10 is ≈ 0 below 100 training
  reviews**. (b) mostly supported: within a training-count bucket, later popularity adds little (CIs include 0 in
  the middle buckets). (c) refuted: products never seen in training (count 0) beat products seen 1–4 times, at
  every pre-T1 level. Plausible mechanism (inference, matches the logQ maths): during training a rarely seen item's
  logit is boosted by −log q, so the model learns a *low raw score* for it; never-seen items get only the text
  tower's prior. Exp 3b tests this directly by adding −β·log q back at serving time.
  **Closed** after two results: the staleness question is answered (composition + training-window popularity), and
  the remaining mechanism check is covered by 3b.
- 23:25 **Follow-up 3b result** (`results/analysis/20261028-logq-serving/`; β = 0 reproduces the saved ranks, 100%
  hit agreement, R@200 0.068714; I re-computed the R@200-by-item-type cells for β = 0 and 0.25-minseen from the saved
  ranks; pytest 171 passed). Serving score u·v − β·τ·log q (no training). 26% of products (all new-in-period ones)
  have no retriever-training interaction; "train" keeps their log q = 0 (= treated as the most popular → no boost),
  "minseen" gives them the rarest seen item's log q.

  | Run | R@10 | R@200 (Δ [CI]) | R@500 | R@200 new | R@200 tail | R@200 head | Top-1% share of top-10 |
  |---|---|---|---|---|---|---|---|
  | β = 0 (frozen) | 0.0104 | 0.0687 | 0.1175 | 0.0261 | 0.0153 | 0.1180 | 70% |
  | β = 0.25, minseen | 0.0062 | 0.0654 (−0.0033 [−0.0042, −0.0023]) | **0.1191** | **0.0553** | **0.0338** | 0.0852 | **10%** |
  | β = 0.5, minseen | 0.0044 | 0.0531 | 0.1037 | 0.0693 | 0.0430 | 0.0477 | 1.5% |
  | β = 0.25, train | 0.0097 | 0.0510 | 0.0794 | 0.0001 | 0.0152 | 0.0975 | 63% |

  **Verdict:** supported with a twist. A small serving-time popularity correction **doubles new-product and tail
  Recall@200 for a −5% overall cost** (R@500 even rises) and collapses popularity concentration — but only if unseen
  products are given a proper log q; the training convention (log q = 0) wipes out new products. Retriever-only
  Recall@10 falls at every β, so the gain is a candidate-pool effect, not a top-10 effect. Near-duplicate check:
  last-item's new-product @10 hits are same product line in another phone model/colour (median text cosine 0.88; 75%
  same store; 0.1% identical titles) — legitimate, not leakage. Probe only, no β selected.
- 23:26 **Follow-up 3c** (same subagent), hypothesis recorded before running: the retriever's @10 loss does not
  matter in the cascade because the ranker re-orders the top 200; with natural rankers on debiased lists,
  new/tail end-to-end nDCG@10 rises and overall changes little. Candidates are selected by the debiased score but
  keep the original score as the ranker feature; β = 0 lists must reproduce the exact lists. Runs
  `20261028-debiased-lists`, `results/index_rank_eval/20261028-debiased-rank-{natural,din}`.
- 23:38 **Exp 4 result** (`results/analysis/20261028-permutation-importance{,-r2}/`, perm seeds 0 and 1; 12
  checkpoints incl. the optional forced DIN; 16,673 natural-hit Val lists at K₁ = 200). Reviewed: baselines
  reproduce saved ranks (100% for natural models; 99.92–99.98% for forced ones, ties from a 2e-7 relative
  score_std difference, so the subagent's own loader uses a 1e-6 tolerance, documented); permuting unused inputs
  leaves ranks bit-identical; pytest 171 passed. Drop in conditional nDCG@10 (mean ± seed SD):

  | Group | Forced MLP | Natural MLP | Natural DIN | Forced DIN |
  |---|---|---|---|---|
  | item text embedding | 0.062 ± 0.002 | 0.080 ± 0.003 | 0.050 ± 0.004 | 0.055 ± 0.007 |
  | candidate item id (DIN query) | — | — | 0.071 ± 0.008 | 0.047 ± 0.004 |
  | store-in-history flag | 0.015 | 0.021 | 0.024 | 0.018 |
  | retrieval score | 0.014 ± 0.001 | **0.022 ± 0.001** | 0.010 ± 0.001 | 0.002 ± 0.003 |
  | history ids + leaf ids | 0.016 ± 0.003 | 0.004 ± 0.002 | 0.014 ± 0.004 | 0.012 ± 0.003 |
  | text cosine | 0.002 ± 0.001 | **0.011 ± 0.002** | 0.014 ± 0.003 | 0.002 ± 0.002 |
  | leaf == last-leaf flag | 0.001 | 0.002 | 0.001 | 0.001 |

  **Verdict: hypothesis rejected.** Shuffling the score *hurts* the forced MLP on natural-hit lists (it does not
  help), and natural training makes both models rely **more** on the retrieval score and text cosine, not less.
  All models rely mainly on candidate identity (text embedding; item id for DIN); store-in-history is a solid third;
  the leaf-match flag is negligible. Reconciliation with the overnight score diagnostic (Spearman −0.40 at 500
  candidates over uniform Val queries): within natural-hit lists at 200, the forced MLP's logit is **U-shaped**
  over retrieval-rank deciles (+0.2 top, −0.2 middle, +0.3 bottom; within-list Spearman −0.06, 59% of lists
  negative), while natural models fall monotonically (Spearman +0.22 to +0.24, ~1% negative). So the forced model
  learned *two* score responses — a genuine top-of-list boost and the shortcut boost at the bottom; natural-hit
  targets sit near the top, which is why shuffling the score still hurts it there, and the bottom boost is what
  takes over at K₁ = 500. **Closed:** the question (which inputs carry the shortcut) is answered as far as
  permutation can go; correlated text inputs limit finer attribution.
- 23:40 **Follow-up 3c result** (`results/analysis/20261028-debiased-lists/`, `results/index_rank_eval/20261028-debiased-rank-{natural,din}/`;
  β = 0 lists reproduce the exact lists (99.1% identical candidate sets; differences are near-ties) and the known
  end-to-end nDCG@10 to 6 decimals (MLP 0.008607, DIN 0.009824); I re-read both results.json files; pytest 171).

  | Ranker, list | nDCG@10 | Δ vs β = 0 [CI] | New | Tail | Head |
  |---|---|---|---|---|---|
  | MLP, β = 0 | 0.008607 | — | 0.00223 | 0.00079 | 0.01590 |
  | MLP, β = 0.25 minseen | 0.007509 | −0.00110 [−0.00128, −0.00092] | 0.00302 | 0.00147 | 0.01286 |
  | DIN, β = 0 | 0.009824 | — | 0.00254 | 0.00082 | 0.01811 |
  | DIN, β = 0.25 minseen | 0.009255 | −0.00057 [−0.00078, −0.00038] | **0.00366** | **0.00161** | 0.01579 |
  | DIN, β = 0.5 minseen | 0.006378 | −0.00345 | 0.00581 | 0.00240 | 0.00828 |

  **Verdict:** partly supported — new/tail rise and the overall loss (DIN −5.8%) is far smaller than the
  retriever-only Recall@10 drop (−40%), but it is not ≥ 0. Conditional nDCG@10 on new products falls (DIN 0.105 →
  0.071): the rankers rank the newly admitted new products less well. Score feature only moderately shifted (mean z
  +0.27 → +0.07). **Closed.**
