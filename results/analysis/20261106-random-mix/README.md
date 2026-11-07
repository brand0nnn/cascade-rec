# Random mixing of missed B targets (Val only, exploratory)

**Label: Val only, exploratory; TRAINING-LABEL CHANGE in an isolated study, pending team review; never replaces
natural-list results.** Nothing is selected. No Test file is opened.

Written 6 Nov 2026, 13:45 SGT, **before** any diagnostic, fit or Val score of this study was computed.

## Question

The historical forced-list rankers appended missed targets to their candidate lists, so the appended positive could
be recognised by its low retrieval score / last position (CLAUDE.md §9.1; the score-inversion and U-shape diagnoses
in `docs/evaluations.md` and `results/analysis/20261105-ushape-popularity/`). If a missed target is instead inserted
so that neither position nor retrieval score reveals it, does the shortcut disappear, and does the extra training
data help or hurt?

## Design (registered here)

- Pool, features, recipe: the rebuilt `results/local_experiments/20261103-full-promising-ranking` (P1, frozen-text
  90-day recency K200) and its `bce_text` recipe, imported unchanged from `cascade.pipelines.full_data_ranking`:
  GraphLatentRanker R2 with graph columns disabled, BCE, 50 uniform real non-target negatives, Adam 1e-3, 32 lists,
  <=10 epochs, patience 2, chronological inner selection with epoch 0 eligible, fixed-epoch all-B refit, seeds 0/1/2.
  Initial state must hash-equal the rebuilt `bce_text` fits per seed.
- Mixed list: for every B list whose target is not in the natural K200 list, the candidate at a uniformly random rank
  r in [1, 200] (`numpy.random.default_rng([seed, instance id])`) is replaced by the target. The target takes that
  candidate's retrieval score (the skip score array is unchanged) and the replaced candidate is dropped. The target's
  nine common columns are rebuilt with the same pre-T0 `BatchedGraphFeatures` builder and standardised with the P1
  scaler; text/leaf/store/cosine inputs are computed inside the model from IDs as usual. Natural lists are unchanged.
- Arms: **M1** = natural lists + all mixed lists. **M2** = natural lists + one fixed seeded subsample
  (`default_rng(20261106)`) of 20,676 mixed lists (doubling the positives, P2/P3 pool scale); the subsample is shared
  by all seeds. Inner training uses prefix natural + prefix mixed lists of the arm; refit uses all-B natural + all
  mixed lists of the arm. Inner selection uses the **natural** chronological tuning lists only (the registered
  selector). Val stays natural. If one M1 fit would take > 25 min on mps, M2 runs first and M1 is reported as not run.
- Diagnostics (before training, CPU): for inserted vs natural positives, within-list AUC against their own real
  negatives of each head input: skip (retrieval) score, text cosine to history (numeric column 4), leaf/store match,
  the nine common columns, and score minus cosine.
- After training: Val per-user macro metrics (232,904 events / 124,416 users, misses = 0, seed mean), nDCG@10,
  Recall@10, HR@3, MRR@10, overall/new/warm, five-way funnel; paired user bootstrap (1,000 draws, seed 0) for
  M1 − P1, M2 − P1, M1 − M2; and the U-shape-by-popularity curve (mean within-list-centred logit by retrieval-rank
  decile x head/tail/new candidate group, 50,000 seeded natural Val lists) for P1, M1, M2.

## Competing pre-stated hypotheses

- **H1** — random insertion removes the U-shape and the extra positives raise new-item quality. Operationalised:
  the M arm shows no low-rank rise (decile 10 minus decile 6 centred logit is not > 0 in every seed for every
  group, the `ushape_popularity` rule), and its new-in-Val nDCG@10 minus P1 has a 95% interval above 0.
- **H2** — the cosine feature still exposes inserted targets (inserted-positive within-list AUC of cosine well below
  0.5, operationalised as < 0.40, or of score minus cosine well above 0.5, > 0.60), so a content-similarity shortcut
  remains and Val quality drops versus `bce_text` (overall nDCG@10 M − P1 interval below 0).
- Pre-registered concern: the uniform-history cosine nearly equals the recency score, so true misses (inserted
  targets) may remain identifiable by a low cosine relative to their assigned score.

Each hypothesis is judged per arm; the report says which holds, both, or neither.

Results are appended below after computation.

## Results (appended 6 Nov 2026, 15:10 SGT, after computation)

Full tables: `tables.md`, `metrics.csv`, `paired_differences.csv`, `funnel.csv`, `diagnostics_auc.csv`,
`ushape_deciles_*.csv`, `fig_ushape_random_mix.png`, `results.json`. Bulk data:
`results/local_experiments/20261106-random-mix/`. Code: `src/cascade/pipelines/random_mix.py`,
`tests/test_random_mix.py`. Logs: `results/logs/20261106/random-mix-*.log`.

**Counts.** B: 315,114 lists; 20,676 natural; 294,438 missed → mixed (235,272 prefix / 59,166 tune); no padded
lists, mean seed-0 insertion rank 100.4. M2 subsample 20,676 mixed (16,485 prefix / 4,191 tune): inner 33,304
lists/epoch (16,819 natural + 16,485 mixed), refit 41,352.

**M1 not run.** Measured M2 mps speed (0.62 ms per training list, 22 s per tuning pass) projects one M1 fit
(252,091 inner / 315,114 refit lists per epoch) at about 25 / 38 / 56 min for selected epochs 3 / 5 / 8 (about 70 min
for 10), above the 25 min rule. M1 − P1 and M1 − M2 are therefore not available.

**Checks.** Builder reproduces stored P1 common features exactly (B and Val, 256 rows each; single-column target
builds vs stored natural targets, 512 rows: max error 0). Mixed rows: one target at its slot, scores equal the
natural score array, no duplicates, target features at the slot (4,096 rows x 3 seeds). Initial state = P1
`bce_text` hash for all seeds; epoch-0 tuning logits/ranks/top-10 = retrieval; selected inner and portable
reload replays exact; Val scoring path reproduces stored P1 logits (max |diff| 0); saved Val logits replay ranks.

**Diagnostics (before training; per-list AUC of the positive vs its own 199 negatives).**

| Input | natural positives | inserted positives |
|---|---:|---:|
| skip (retrieval) score | 0.684 | 0.501 |
| text cosine to history (col 4) | 0.587 | **0.103** (median 0) |
| score minus cosine | 0.611 | **0.889** |
| leaf match | 0.532 | 0.295 |
| store match | 0.620 | 0.411 |
| candidate log degree / graph known | 0.476 / 0.386 | 0.486 / 0.390 |

Random insertion removes the score/position cue (AUC 0.50), but the uniform-history cosine identifies inserted
targets almost perfectly from below at every insertion-rank decile (0.09–0.11), and leaf/store match also from below.

**Val (232,904 events / 124,416 users, seed means, paired-user 95% intervals).**

| System | nDCG@10 all | new | warm | Recall@10 | HR@3 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| retrieval | 0.006284 | 0.007611 | 0.005606 | 0.011506 | 0.005260 | 0.004703 |
| P1 bce_text | 0.008141 | 0.008666 | 0.007887 | 0.015447 | 0.006787 | 0.005932 |
| M2 | 0.007061 | 0.007263 | 0.006985 | 0.013884 | 0.005651 | 0.005008 |

M2 − P1 nDCG@10: all **−0.001080 [−0.001278, −0.000881]**, new −0.001402 [−0.001715, −0.001091], warm
−0.000902 [−0.001117, −0.000674]; every metric/group interval is below 0. M2 still beats retrieval overall
(+0.000777 [+0.000431, +0.001096]) and on warm, but not on new targets (−0.000348 [−0.000862, +0.000157]). Funnel:
M2 rescues about as many as P1 (2,424 vs 2,454) but demotes more (1,935 vs 1,568 of the 3,137 retrieval top-10 hits).
Selected epochs M2 8/5/10 (P1 8/8/7); M2 epoch 1 falls below retrieval on inner tuning in every seed.

**U-shape (50,000 seeded natural Val lists).** Neither P1 nor M2 has the forced-list U-shape: no group rises in
every seed under the `ushape_popularity` rule, except M2 new candidates (decile 10 − decile 6 = +0.056/+0.066/+0.058).
M2 flattens the downward slope: overall centred logit d1 − d10 is 0.58 for M2 versus 1.20 for P1 (head/tail d6–d10
flat at about 0.35 / −0.37), i.e. the residual partly cancels the retrieval score at low ranks, consistent with a
learned low-similarity preference.

**Verdict.** H2 holds for M2 (diagnostic condition met; overall nDCG@10 below P1 with interval below 0). H1 does not:
the U-shape is absent (as it already is for P1), but new-item quality falls rather than rises. Random insertion
moves the shortcut from position/score to content similarity. Exploratory Val evidence only; nothing selected.
