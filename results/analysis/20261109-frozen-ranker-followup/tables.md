# Frozen rankers on better candidate lists: Val follow-up and post-hoc Test check (9 Nov 2026)

Protocol `docs/test_protocol_2026-11-09.md` (sha256 3afecb44…c7d6, locked 05:10 UTC before Test was opened).
Code `src/cascade/pipelines/frozen_ranker_followup.py`. Post-hoc third look at Test; nothing trained, tuned or
selected; each system scored once. nDCG@10 ×10⁻³, per-user macro, seed means (3 seeds); intervals: 1,000 paired
user draws, seed 0, unadjusted. "In list" = share of purchases whose product is in the candidate list.

## Checks (all passed before Test was opened)

- Val anchor: all six family-A rankers reproduce their saved K200 Val ranks exactly (0 differences).
- Two-tower Val lists rebuild exactly; mixed-list two-tower scores equal the stored scores on 470,295 overlapping slots;
  the fresher-retriever lists equal the 6 Nov study's Val pool exactly.
- Family B: fresher-retriever list builder, P2 features (max error 0.0) and P2 ranks (3 seeds) replay exactly; m1 on
  mixed lists replays the saved Val P1onP3 ranks exactly on MPS (CPU flips one near-tied row, logits 4.8e-7 apart).
- Test: the candidate-file two-tower top 200 differs from the locked 7 Nov archive only in tie order/boundary ties
  (the same pre-existing difference exists between the two saved Val formats); 1 of 297,390 Test purchases differs
  in list membership.

## Results

| Test | Ranker (not retrained) | In list | Val: own → better | Test: own → better | Test change [95% CI] | Test new / warm |
|---|---|---|---|---|---|---|
| 500 vs 200 candidates | natural MLP (arm C) | Test 3.9% → 7.4% | 8.63 → 8.07 | 5.18 → 4.77 | −0.40 [−0.53, −0.28] | +0.41 / −0.66 |
| 500 vs 200 candidates | DIN | | 9.82 → 9.73 | 6.24 → 6.39 | +0.15 [+0.01, +0.27] | +0.21 / +0.19 |
| fresher retriever | natural MLP (arm C) | Test 3.9% → 6.0% | 8.63 → 8.00 | 5.18 → 5.11 | −0.07 [−0.26, +0.12] | +0.13 / −0.23 |
| fresher retriever | DIN | | 9.82 → 9.18 | 6.24 → 5.95 | −0.29 [−0.52, −0.05] | +0.66 / −0.76 |
| fresher retriever | score + correction MLP (P2) | | 10.08 → 9.93 | 6.35 → 5.93 | −0.43 [−0.62, −0.25] | −0.23 / −0.60 |
| fresher retriever | (retrieval order, no ranker) | | 5.64 → 10.04 | 2.76 → 5.79 | +3.03 [+2.78, +3.30] | −0.25 / +4.10 |
| mixed lists | natural MLP (arm C) | Test 3.9% → 6.4% | 8.63 → 8.82 | 5.18 → 6.42 | +1.24 [+1.06, +1.42] | +2.92 / +0.60 |
| mixed lists | DIN | | 9.82 → 9.62 | 6.24 → 6.83 | +0.60 [+0.43, +0.75] | +2.26 / +0.01 |
| mixed lists | recency-list MLP (m1) | Test 6.1% → 6.4% | 8.14 → 8.81 | 6.44 → 6.70 | +0.27 [+0.04, +0.50] | −2.30 / +1.19 |
| mixed lists | m1 frozen → retrained on mixed (S6) | | 8.81 → 10.39* | 6.70 → 7.86 | +1.16 [+0.97, +1.35] | +0.52 / +1.31 |

*Val retrained value is the size-matched P3 ranker; Test S6 is the locked 7 Nov system.
Frozen P2 ranker gain minus retrieval-order gain on the fresher lists (Test): −3.46 [−3.78, −3.16].

## Written predictions

| Prediction | Test result | Verdict |
|---|---|---|
| U: relative nDCG change < ⅓ of relative in-list change (8 cells) | held for 6; not held for arm C on mixed (+24.0% vs +62.1%) and m1 on mixed (+4.1% vs +4.5%) | 6/8 held |
| D1 arm C: 500 − 200 < 0 | −0.40 [−0.53, −0.28] | held |
| D2 arm C: fresher − own < 0 | −0.07 [−0.26, +0.12] | inconclusive |
| D3 DIN: fresher − own < 0 | −0.29 [−0.52, −0.05] | held |
| D4 DIN: mixed − own < 0 | +0.60 [+0.43, +0.75] | **not held** |
| D5 P2 ranker: fresher − own < 0 (new) | −0.23 [−0.49, +0.04] | inconclusive |
| D6 P2 ranker gain < retrieval-order gain | −3.46 [−3.78, −3.16] | held |
| D7 m1: mixed − own < 0 (new) | −2.30 [−2.55, −2.04] | held |
| D8 retrained on mixed beats m1 frozen on mixed | +1.16 [+0.97, +1.35] | held |

## Reading

- **Fresher retriever (strongest):** the retriever's own order more than doubles on Test (2.76 → 5.79), yet none
  of the three frozen rankers gains (−0.07 n.s., −0.29, −0.43). Replicated from Val.
- **500 candidates:** the natural MLP loses on Test as on Val (−8%); DIN gains slightly (+2%; Val −1%, n.s.).
  The loss at 500 is MLP-specific.
- **Mixed lists:** frozen rankers trained on two-tower lists do gain on Test (+24% MLP, +10% DIN), almost entirely
  on new products (44% of Test). The recency-list ranker gains only +4% frozen and loses badly on new products;
  retraining it on mixed lists adds +1.16 more. Mixed-list gains for frozen rankers are therefore real but
  ranker-dependent, and smaller than retraining where both were measured.
- **New products:** which group loses depends on the list and ranker; there is no single "new products get worse" pattern.

## Addendum: second fresher-retriever seed (Val only) and SASRec intervals (Test, saved ranks only)

**Fresher retriever, seed 1** (seed 2 never completed training). Val; purchases in list 7.2% → 11.7%:

| Ranker (not retrained) | Seed 0 change [95% CI] | Seed 1 change [95% CI] |
|---|---|---|
| natural MLP (arm C) | −0.63 [−0.87, −0.39] | −0.51 [−0.75, −0.27] |
| DIN | −0.64 [−0.91, −0.37] | −0.25 [−0.51, +0.03] |
| score + correction MLP (P2) | −0.15 [−0.37, +0.08] | +0.11 [−0.11, +0.35] |
| retrieval order, no ranker | +4.39 [+4.10, +4.70] | +4.35 [+4.05, +4.66] |

Frozen P2 ranker gain minus retrieval-order gain, seed 1: −4.25 [−4.59, −3.87]. With either seed no frozen ranker
gains (0 of 3 positive intervals), while the retriever's own order gains about +4.4.

**SASRec paired intervals on Test** (all_train, 3 seeds; reproduces 2.588; 112,081 users, 1,000 draws):

| System | nDCG@10 ×10⁻³ | × SASRec | minus SASRec [95% CI] |
|---|---|---|---|
| natural MLP on two-tower top 200 (planned recipe) | 5.18 | 2.0 | +2.59 [+2.36, +2.84] |
| DIN on two-tower top 200 | 6.24 | 2.4 | +3.65 [+3.40, +3.91] |
| S6 mixed lists, ranker retrained | 7.86 | 3.0 | +5.27 [+5.01, +5.58] |
| S8 mixed lists + timed statistics + co-review | 10.38 | 4.0 | +7.79 [+7.46, +8.15] |

## Addendum: where the frozen rankers' mixed-list gain comes from (Test)

Additive split of the overall gain (mixed − two-tower 200, seed means, per-user macro weights; contributions sum to
the total). File: `mixed_gain_decomposition_test.csv`. ×10⁻³.

| Purchase is… | MLP new | MLP established | DIN new | DIN established |
|---|---|---|---|---|
| added by mixing (only in mixed list; 11,048 purchases, 53% new) | +0.95 | +1.19 | +0.80 | +0.98 |
| in both lists (7,888) | −0.02 | −0.02 | −0.03 | −0.14 |
| dropped by mixing (only in two-tower list; 3,794, 26% new) | −0.07 | −0.79 | −0.08 | −0.92 |
| **net** | **+0.86** | **+0.38** | **+0.68** | **−0.09** |

Total: MLP +1.24, DIN +0.60. The gain is a recall effect: the frozen rankers place many of the purchases the recency
half adds (new and established alike); mixing also drops some two-tower purchases, mostly established ones, which is
why the net gain falls mainly on new products (MLP 69%, DIN all of it). Purchases in both lists are ranked about the
same. Descriptive; it locates the gain but does not prove why the rankers can use these added candidates.

## Addendum: top-10 purchase counts by list change (Test, frozen rankers, seed means)

Number of the 297,390 Test purchases placed in the top 10 (micro counts, mean of 3 seeds). Baseline: two-tower top 200,
11,682 purchases in the list. File: `top10_counts_by_list_change_test.csv`.

| Better list | Purchases added to list / dropped | MLP: top 10 before → after | MLP: added that reach top 10 | MLP: lost among kept | DIN: before → after | DIN: added that reach top 10 | DIN: lost among kept |
|---|---|---|---|---|---|---|---|
| 500 candidates | +10,317 / 0 | 2,372 → 2,280 | 546 (5.3%) | −638 | 2,784 → 2,875 | 662 (6.4%) | −571 |
| fresher retriever | +10,658 / −4,491 | 2,372 → 2,346 | 893 (8.4%) | −487 | 2,784 → 2,881 | 1,289 (12.1%) | −681 |
| mixed lists | +11,048 / −3,794 | 2,372 → 3,284 | 1,448 (13.1%) | −102 | 2,784 → 3,502 | 1,351 (12.2%) | −116 |

Also lost: purchases dropped from the list that were in the top 10 before (fresher −432 / −512, mixed −435 / −517).
DIN on the fresher lists places 97 more purchases in the top 10 but at lower positions, so nDCG@10 falls (−0.29 ×10⁻³).
Reading: mixing keeps the list at 200 and keeps every candidate's two-tower score, so purchases already in the list
are barely displaced; 500 candidates add 300 competitors; the fresher retriever changes the score of every candidate.

## Addendum: two-tower version of R3 (Val only; no training; file `two_tower_r3_val.csv`)

Frozen two-tower-list score-plus-correction MLP (pool-ranking-P2 bce_text, 3 seeds) on its own two-tower Val lists;
only the date of the nine common inputs changes. The T0 arm replays the saved Val ranks exactly. Recency-list ranker
(B1, the R3 ranker) on Val for comparison: `results/analysis/20261106-freshness-val/metrics.csv`.
nDCG@10 ×10⁻³, all / new / established.

| Statistics counted up to | Two-tower ranker | Recency ranker (R3's) |
|---|---|---|
| T0, 26 Jul 2020 (as originally served; 1–2 years old on Val) | 10.08 / 3.70 / 12.75 | 8.14 / 8.67 / 7.89 |
| T1, start of Val (same timing as in training) | 10.21 / 3.60 / 12.94 | 9.43 / 11.72 / 8.23 |
| each query's month start (fresher than in training) | 9.74 / 1.27 / 13.42 | 8.48 / 6.18 / 9.57 |

Two-tower ranker, paired differences: T1 − T0 +0.13 [−0.00, +0.26] (new −0.10 [−0.22, +0.01]);
monthly − T1 −0.47 [−0.57, −0.37] (new −2.33 [−2.60, −2.09]; established +0.48 [+0.42, +0.54]).
Reading: too-fresh statistics hurt new products on both pipelines; the period-start fix gives the recency ranker
+1.29 (+16%) on Val but the two-tower ranker no detectable gain.

## Addendum: score-swap test for R1 (Val only; 10 Nov; predictions `docs/score_swap_val_2026-11-10.md`, sha256 d6b70111…cd4)

Frozen family-A rankers (3 seeds), 200 candidates per list; only the products and/or the score column change. Files
`score_swap_val_{levels,contrasts}.csv`, `score_swap_val_checks.json`. Checks: saved fresher-list ranks replay exactly
(0 differences on 4,096 rows); swapped scores equal the stored scores on all 343,879 shared slots. nDCG@10 ×10⁻³, all / new / established.

| Arm | Products | Score input | In list | natural MLP | DIN | retriever order |
|---|---|---|---|---|---|---|
| A | original two-tower | original (as trained) | 7.2% | 8.62 / 2.10 / 11.50 | 9.82 / 2.54 / 12.97 | 5.65 |
| B | fresher retriever | fresher | 11.8% | 8.00 / 1.41 / 11.02 | 9.18 / 3.06 / 11.86 | 10.11 |
| C | fresher retriever | original two-tower score | 11.8% | 9.31 / 1.70 / 12.73 | 10.33 / 1.75 / 14.12 | 6.14 |
| D | fresher retriever | fresher order, original score values | 11.8% | **9.76** / 1.67 / 13.39 | **10.56** / 1.88 / 14.42 | 10.04 |
| E | original two-tower | fresher | 7.2% | 7.39 / 1.35 / 10.10 | 8.73 / 3.41 / 11.02 | 9.31 |

| Prediction | natural MLP | DIN | Verdict |
|---|---|---|---|
| S1 C − B > 0 | +1.31 [+1.13, +1.49] | +1.15 [+0.94, +1.36] | held / held |
| S2 D − B > 0 | +1.76 [+1.59, +1.94] | +1.37 [+1.18, +1.56] | held / held |
| S3 E − A < 0 | −1.23 [−1.40, −1.07] | −1.09 [−1.26, −0.92] | held / held |
| S4 C − A > 0 | +0.69 [+0.49, +0.88] | +0.51 [+0.34, +0.69] | held / held |

D − A (better list, familiar score values): MLP +1.13 [+0.90, +1.36] (+13%), DIN +0.73 [+0.53, +0.95] (+8%).
Reading: the frozen rankers' failure on the fresher lists is caused by the score values, not by the products. With
the fresher order but the score values they were trained on, the same rankers gain from the better list. The gain is
on established products; new products stay below the original lists (MLP −0.44, DIN −0.66 for D − A), and the frozen
MLP on D (9.76) still does not beat the fresher retriever's own order (10.04). Val reused; one retriever seed.
