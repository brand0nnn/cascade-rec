# Statistics timing: column swap and age curve (Val only, exploratory; no training)

## Pre-stated hypotheses (written 2026-11-07T12:25:44Z, before any arm is built or scored)

Models: the four frozen P1 rankers trained with T0 statistics (B1 bce_text, B4 pairwise_text, B5 pairwise_coreview,
B6 bce_coreview), seeds 0/1/2, scored on the 14,916 Val natural-hit rows (misses keep rank infinity).

**Run 1, T0 -> T1 column swap.** Start from T0 Val features and replace a subset with its T1 values.
- X1 known flag only; X2 known flag + log degree; X3 all graph-derived common columns (0, 1, 2, 7, 8);
  X6 all T1 values but only for candidates first reviewed before T0 (older products get updated counts);
  X7 all T1 values but only for candidates first reviewed in [T0, T1) (products released during B).
- H1a: X1 alone recovers a substantial part (at least a third) of the T0 -> T1 gain on new targets.
- H1b: the rest comes from the added counts (X2/X3 above X1; for co-review models, the co-review block).
- H1c: most of the T0 -> T1 gain comes from X7 (products released during B), not X6.

**Run 2, statistics-age curve.**
- (a) Fixed cut date T1 - k months, k in {0, 3, 6, 9} plus T0 (about 12.5 months). H2a: nDCG@10 falls
  monotonically from T1 to T0 for all four models, and the unknown share of non-target candidates rises from
  about 0.15 towards 0.44.
- (b) Fixed lag per query: reviews before (query month start - L months), L in {0, 1, 3, 6, 9, 12} (L = 0 is the
  monthly arm). Training statistics were 0-12.5 months old (mean about 6). H2b: if the learned rule depends on the
  statistics being about as old as in training, new-target nDCG@10 for BCE text is higher at some L in {3, 6, 9}
  than at L = 0, and L = 12 is below that peak. If freshness alone mattered, L = 0 would be best.

Each outcome is reported, including failed hypotheses. Nothing is selected. No Test file is opened.

## Results (computed after the hypotheses above)

### Run 1: T0 -> T1 column swap, nDCG@10 (all / new / warm)

| Arm | bce_text | pairwise_text | pairwise_coreview | bce_coreview |
|---|---|---|---|---|
| T0 | 0.008141 / 0.008666 / 0.007887 | 0.007787 / 0.008754 / 0.007298 | 0.007735 / 0.008575 / 0.007293 | 0.008167 / 0.008250 / 0.008105 |
| X1_known | 0.009264 / 0.014268 / 0.006865 | 0.008544 / 0.011887 / 0.006889 | 0.008385 / 0.011138 / 0.006985 | 0.009305 / 0.014258 / 0.006919 |
| X2_known_degree | 0.009617 / 0.012943 / 0.007997 | 0.009883 / 0.014660 / 0.007506 | 0.009941 / 0.015874 / 0.006972 | 0.009632 / 0.013398 / 0.007807 |
| X3_all_common | 0.009432 / 0.011721 / 0.008231 | 0.009684 / 0.013146 / 0.007841 | 0.009837 / 0.015242 / 0.007081 | 0.009456 / 0.012295 / 0.008019 |
| X6_old_products | 0.008326 / 0.007601 / 0.008631 | 0.008146 / 0.008046 / 0.008152 | 0.008205 / 0.008136 / 0.008156 | 0.008556 / 0.007342 / 0.009074 |
| X7_products_released_in_B | 0.009443 / 0.012792 / 0.007774 | 0.009627 / 0.014690 / 0.007076 | 0.009865 / 0.015052 / 0.007227 | 0.009665 / 0.012320 / 0.008312 |
| T1 | 0.009432 / 0.011721 / 0.008231 | 0.009684 / 0.013146 / 0.007841 | 0.010221 / 0.014365 / 0.008041 | 0.009801 / 0.011132 / 0.009055 |

Share of the T0 -> T1 gain recovered by each swap (arm − T0) / (T1 − T0):

| Arm | bce_text all / new | pairwise_text all / new | pairwise_coreview all / new | bce_coreview all / new |
|---|---|---|---|---|
| X1_known | 87% / 183% | 40% / 71% | 26% / 44% | 70% / 208% |
| X2_known_degree | 114% / 140% | 111% / 134% | 89% / 126% | 90% / 179% |
| X3_all_common | 100% / 100% | 100% / 100% | 85% / 115% | 79% / 140% |
| X6_old_products | 14% / -35% | 19% / -16% | 19% / -8% | 24% / -31% |
| X7_products_released_in_B | 101% / 135% | 97% / 135% | 86% / 112% | 92% / 141% |

Arm − T0, nDCG@10 all [95% CI]:

| Arm | bce_text | pairwise_text | pairwise_coreview | bce_coreview |
|---|---|---|---|---|
| X1_known | +0.001124 [+0.000977, +0.001267] | +0.000757 [+0.000673, +0.000849] | +0.000650 [+0.000567, +0.000731] | +0.001138 [+0.000987, +0.001284] |
| X2_known_degree | +0.001476 [+0.001296, +0.001645] | +0.002097 [+0.001891, +0.002301] | +0.002206 [+0.002001, +0.002405] | +0.001465 [+0.001283, +0.001658] |
| X3_all_common | +0.001291 [+0.001124, +0.001459] | +0.001897 [+0.001683, +0.002095] | +0.002102 [+0.001896, +0.002302] | +0.001289 [+0.001105, +0.001475] |
| X6_old_products | +0.000185 [+0.000086, +0.000290] | +0.000359 [+0.000275, +0.000457] | +0.000471 [+0.000389, +0.000554] | +0.000389 [+0.000289, +0.000497] |
| X7_products_released_in_B | +0.001303 [+0.001160, +0.001444] | +0.001841 [+0.001659, +0.002010] | +0.002130 [+0.001955, +0.002308] | +0.001498 [+0.001338, +0.001653] |
| T1 | +0.001291 [+0.001124, +0.001459] | +0.001897 [+0.001683, +0.002095] | +0.002486 [+0.002288, +0.002698] | +0.001634 [+0.001432, +0.001822] |

### Run 2a: fixed cut date, nDCG@10 (all / new / warm)

| Cut date | bce_text | pairwise_text | pairwise_coreview | bce_coreview |
|---|---|---|---|---|
| T0 | 0.008141 / 0.008666 / 0.007887 | 0.007787 / 0.008754 / 0.007298 | 0.007735 / 0.008575 / 0.007293 | 0.008167 / 0.008250 / 0.008105 |
| date_T1-9m | 0.008494 / 0.009662 / 0.007890 | 0.008214 / 0.009893 / 0.007299 | 0.008200 / 0.009934 / 0.007242 | 0.008543 / 0.009248 / 0.008128 |
| date_T1-6m | 0.008962 / 0.010954 / 0.007959 | 0.008967 / 0.011896 / 0.007431 | 0.009071 / 0.012354 / 0.007357 | 0.009091 / 0.010451 / 0.008360 |
| date_T1-3m | 0.009309 / 0.011414 / 0.008230 | 0.009437 / 0.012633 / 0.007772 | 0.009750 / 0.013449 / 0.007805 | 0.009565 / 0.010866 / 0.008836 |
| T1 | 0.009432 / 0.011721 / 0.008231 | 0.009684 / 0.013146 / 0.007841 | 0.010221 / 0.014365 / 0.008041 | 0.009801 / 0.011132 / 0.009055 |

### Run 2b: fixed lag before each query month, nDCG@10 (all / new / warm)

| Lag | bce_text | pairwise_text | pairwise_coreview | bce_coreview |
|---|---|---|---|---|
| monthly | 0.008477 / 0.006181 / 0.009570 | 0.008735 / 0.007061 / 0.009475 | 0.009804 / 0.009033 / 0.010022 | 0.009640 / 0.007294 / 0.010693 |
| lag1 | 0.008897 / 0.007918 / 0.009316 | 0.009164 / 0.008975 / 0.009136 | 0.009966 / 0.010518 / 0.009521 | 0.009629 / 0.008058 / 0.010275 |
| lag3 | 0.009211 / 0.009691 / 0.008904 | 0.009452 / 0.010885 / 0.008622 | 0.010041 / 0.012201 / 0.008824 | 0.009696 / 0.009374 / 0.009707 |
| lag6 | 0.009202 / 0.010703 / 0.008413 | 0.009372 / 0.011915 / 0.008006 | 0.009713 / 0.012987 / 0.007965 | 0.009490 / 0.010264 / 0.009013 |
| lag9 | 0.009064 / 0.010982 / 0.008080 | 0.009093 / 0.012045 / 0.007543 | 0.009240 / 0.012694 / 0.007427 | 0.009214 / 0.010483 / 0.008497 |
| lag12 | 0.008690 / 0.010195 / 0.007917 | 0.008572 / 0.010870 / 0.007381 | 0.008656 / 0.011112 / 0.007375 | 0.008775 / 0.009757 / 0.008238 |

Lag − monthly (lag 0), nDCG@10 new [95% CI]:

| Lag | bce_text | pairwise_text | pairwise_coreview | bce_coreview |
|---|---|---|---|---|
| lag1 | +0.001736 [+0.001490, +0.001977] | +0.001913 [+0.001638, +0.002163] | +0.001484 [+0.001207, +0.001772] | +0.000764 [+0.000512, +0.001004] |
| lag3 | +0.003510 [+0.003160, +0.003841] | +0.003823 [+0.003453, +0.004177] | +0.003168 [+0.002788, +0.003546] | +0.002080 [+0.001759, +0.002401] |
| lag6 | +0.004522 [+0.004156, +0.004913] | +0.004854 [+0.004458, +0.005304] | +0.003954 [+0.003555, +0.004415] | +0.002970 [+0.002594, +0.003339] |
| lag9 | +0.004801 [+0.004430, +0.005194] | +0.004984 [+0.004584, +0.005427] | +0.003661 [+0.003224, +0.004118] | +0.003189 [+0.002827, +0.003566] |
| lag12 | +0.004014 [+0.003654, +0.004390] | +0.003809 [+0.003414, +0.004228] | +0.002079 [+0.001679, +0.002506] | +0.002463 [+0.002082, +0.002835] |

### Age of statistics and the "no reviews yet" cue (Val natural-hit rows)

| Arm | age mean (months) | age range | target unknown | other unknown | AUC of unknown |
|---|---:|---|---:|---:|---:|
| training (B natural lists, T0) | 6.1 | 0.0–12.5 | 0.431 | 0.203 | 0.614 |
| T0 | 18.0 | 12.5–23.7 | 0.713 | 0.444 | 0.635 |
| date_T1-9m | 14.5 | 9.0–20.1 | 0.634 | 0.366 | 0.634 |
| date_T1-6m | 11.5 | 5.9–17.1 | 0.545 | 0.272 | 0.636 |
| date_T1-3m | 8.5 | 3.0–14.2 | 0.478 | 0.206 | 0.636 |
| T1 | 5.5 | 0.0–11.2 | 0.406 | 0.153 | 0.626 |
| monthly | 0.5 | 0.0–1.0 | 0.095 | 0.014 | 0.540 |
| lag1 | 1.5 | 0.9–2.0 | 0.182 | 0.043 | 0.570 |
| lag3 | 3.5 | 2.9–4.0 | 0.296 | 0.095 | 0.600 |
| lag6 | 6.5 | 5.9–7.1 | 0.416 | 0.166 | 0.625 |
| lag9 | 9.5 | 9.0–10.1 | 0.516 | 0.240 | 0.638 |
| lag12 | 12.5 | 12.0–13.0 | 0.600 | 0.316 | 0.642 |

Intervals: 1,000 paired-user draws (seed 0) on seed-mean event metrics, unadjusted. No Test file was opened.

## Verdicts on the pre-stated hypotheses

- **H1a held.** The known flag alone recovers 44–208% of the T0 -> T1 gain on new targets (26–87% overall). BCE overshoots T1 on new targets.
- **H1b held.** Known flag + log degree recovers 89–114% overall. For text rankers, all graph-derived common columns equal T1 exactly. For co-review rankers, the co-review block adds the remaining 15–21%.
- **H1c held.** Updating only products released during B (X7) recovers 86–101% of the overall gain. Updating only older products (X6) recovers 14–24% and lowers new-target quality.
- **H2a held.** Moving the cut date from T0 to T1 raises nDCG@10 monotonically for all four models. The "no reviews" share of non-target candidates falls 0.444 -> 0.366 -> 0.272 -> 0.206 -> 0.153.
- **H2b held.**
  - New-target nDCG@10 is lowest at lag 0 for every model, and peaks at lag 9 (BCE text, pairwise text, BCE co-review) or lag 6 (pairwise co-review).
  - Lag 12 is below the peak for all four.
  - Warm-target quality instead falls steadily with lag.
  - The peak sits where the Val statistics are about as old as in training: lag 6–9 gives target / other unknown shares of 0.42–0.52 / 0.17–0.24, against 0.43 / 0.20 in training.
