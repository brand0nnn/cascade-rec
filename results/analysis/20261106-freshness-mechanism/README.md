# Freshness mechanism: knownness as a newness cue (Val only, exploratory; diagnostic)

## Pre-stated hypothesis (written 2026-11-05T18:07:16Z, before any hybrid arm or probe was computed)

The frozen rankers use candidate "knownness" (no pre-snapshot reviews) as a newness signal learned from T0-era features.
If knownness columns are held at their period-start (T1) values while all other graph/common columns use the monthly
snapshot, new-item quality recovers to about the T1 level and warm quality keeps the monthly gain, so the hybrid is
≥ both T1 and monthly overall.

Label: mixed snapshots are a diagnostic of what the frozen models use, not a recommended serving recipe.

## Design

- **Checkpoints:** the same twelve as results/analysis/20261106-freshness-val. Hit-row-only CPU rescoring, same population, metrics and bootstrap
  (232,904 events / 124,416 users, seed-mean then user macro, 1,000 paired-user draws, seed 0). No training. No Test file opened.
- **Arms** (built from that study's saved standardised T1/monthly features; every column comes from a snapshot that ends before each of its queries):

  | Arm | Base snapshot | Columns swapped in | Source of swapped columns |
  |---|---|---|---|
  | H1 | monthly | candidate_graph_known | T1 |
  | H2 | monthly | candidate_graph_known and candidate_log_degree | T1 |
  | H3 | T1 | candidate_graph_known | monthly |
- **Checks:**
  - Rescoring the monthly arm reproduces its saved ranks exactly.
  - Each hybrid equals its definition column by column.
  - The probe changes only the flipped slots, because the head scores each candidate separately.
- **Diagnostic only.** H1 and H2 create input combinations the model never saw: candidate_graph_known = 0 alongside a positive monthly log-degree or co-review paths.

## Training data: was "unknown" a positive cue? (training_knownness.csv)

| Population (T0 snapshot) | Positives unknown | Negatives unknown | P(target) among unknown slots | P(target) among known slots | AUC of "unknown" as a score |
|---|---:|---:|---:|---:|---:|
| B prefix natural lists (inner fit), 16,819 | 0.405 | 0.177 | 0.0114 | 0.0036 | 0.614 |
| All B natural lists (refit), 20,676 | 0.431 | 0.203 | 0.0106 | 0.0036 | 0.614 |
| All B prefix events (including misses), 252,091 | 0.346 | – | – | – | – |
| Val natural hits, T0 / T1 / monthly | 0.713 / 0.406 / 0.095 | 0.444 / 0.153 / 0.014 | – | – | 0.635 / 0.626 / 0.540 |

- In the training lists, an unknown candidate was about 3.1 times as likely to be the target as a known one.
- On Val, T1 keeps that contrast (AUC 0.626). Monthly almost removes it (AUC 0.540).

## Frozen-model probe (knownness_probe.csv)

Setup: 10,000 fixed Val hit rows on T1 features. Only candidate_graph_known is flipped from 0 to 1, which is z −2.61 → +0.38.

| Method | Mean Δlogit | Within-list logit SD | Unknown new targets in top 10, before → after flip (n=4,085) | Mean target rank, before → after |
|---|---:|---:|---:|---:|
| bce_text | −1.30 | 1.39 | 0.335 → 0.139 | 32 → 66 |
| bce_coreview | −1.17 | 1.31 | 0.330 → 0.140 | 31 → 61 |
| pairwise_text | −0.39 | 1.63 | 0.349 → 0.288 | 30 → 36 |
| pairwise_coreview | −0.51 | 1.92 | 0.379 → 0.305 | 29 → 36 |

Δlogit is the seed-mean over 304,776 flipped slots. Every seed has a negative Δ.

## Arms, nDCG@10 (seed-mean)

| Method | T1 all | monthly all | H1 all | H2 all | H3 all | T1 new | monthly new | H1 new | H3 new | T1 warm | monthly warm | H1 warm | H3 warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bce_text | 0.009432 | 0.008477 | 0.010247 | 0.009297 | 0.008370 | 0.011721 | 0.006181 | 0.015547 | 0.007018 | 0.008231 | 0.009570 | 0.007711 | 0.008990 |
| pairwise_text | 0.009684 | 0.008735 | 0.009748 | 0.009585 | 0.009335 | 0.013146 | 0.007061 | 0.011209 | 0.011564 | 0.007841 | 0.009475 | 0.008947 | 0.008165 |
| pairwise_coreview | 0.010221 | 0.009804 | 0.010627 | 0.010529 | 0.009852 | 0.014365 | 0.009033 | 0.012794 | 0.012616 | 0.008041 | 0.010022 | 0.009412 | 0.008426 |
| bce_coreview | 0.009801 | 0.009640 | 0.011371 | 0.010602 | 0.008844 | 0.011132 | 0.007294 | 0.016145 | 0.006477 | 0.009055 | 0.010693 | 0.008959 | 0.009893 |

## Contrasts, nDCG@10 overall [95% CI] (new / warm point estimates)

| Method | H1 − T1 | H1 − monthly | H2 − T1 | H2 − monthly | H3 − T1 |
|---|---|---|---|---|---|
| bce_text | +0.000815 [0.000692, 0.000945] (+0.003826 / −0.000521) | +0.001770 [0.001583, 0.001966] (+0.009366 / −0.001859) | −0.000135 [−0.000159, −0.000112] | +0.000820 [0.000696, 0.000955] | −0.001062 [−0.001182, −0.000958] (−0.004703 / +0.000758) |
| pairwise_text | +0.000064 [−0.000047, 0.000180] (−0.001937 / +0.001106) | +0.001013 [0.000918, 0.001120] (+0.004148 / −0.000527) | −0.000098 [−0.000122, −0.000075] | +0.000850 [0.000699, 0.001014] | −0.000349 [−0.000420, −0.000287] (−0.001582 / +0.000324) |
| pairwise_coreview | +0.000406 [0.000287, 0.000531] (−0.001571 / +0.001371) | +0.000823 [0.000731, 0.000926] (+0.003761 / −0.000610) | +0.000308 [0.000239, 0.000378] | +0.000724 [0.000577, 0.000890] | −0.000369 [−0.000442, −0.000306] (−0.001749 / +0.000385) |
| bce_coreview | +0.001570 [0.001413, 0.001727] (+0.005012 / −0.000096) | +0.001731 [0.001546, 0.001925] (+0.008851 / −0.001734) | +0.000801 [0.000687, 0.000898] | +0.000961 [0.000825, 0.001106] | −0.000958 [−0.001073, −0.000858] (−0.004655 / +0.000838) |

Co-review − text, nDCG@10 overall [95% CI]:

| Objective | H1 | H2 |
|---|---|---|
| BCE | +0.001124 [0.000954, 0.001287] | +0.001305 [0.001142, 0.001467] |
| Pairwise | +0.000879 [0.000708, 0.001036] | +0.000944 [0.000699, 0.001100] |

Recall@10, HR@3 and MRR@10 have the same signs. The one exception is pairwise_text H1 − T1: Recall@10 is positive, while HR@3 and MRR@10 are inconclusive. See contrasts.csv.

## Verdict on the pre-stated hypothesis: supported for the cue, not for the clean decomposition

- **Supported: the frozen rankers use knownness as a newness cue.**
  - "Unknown" was a positive signal in the T0-era training lists.
  - Flipping only knownness lowers a candidate's logit by about 0.9 within-list SD for BCE and about 0.25 SD for pairwise.
  - Swapping only the knownness column moves new-item nDCG@10 by −0.0016 to −0.0047 (H3 versus T1) and by +0.0038 to +0.0094 (H1 versus monthly).
  - The cue is much stronger under BCE than under pairwise.
- **Partly supported:**
  - H1 is at least as good as both T1 and monthly overall for three methods. For pairwise_text, H1 − T1 is inconclusive.
  - New-item quality recovers to about T1 for pairwise, but stays below it: −0.0019 for text, −0.0016 for co-review.
  - For BCE, new-item quality overshoots T1: +0.0038 for text, +0.0050 for co-review.
- **Not supported: warm targets do not keep the monthly gain.**
  - H1 warm is below monthly for every method (−0.0005 to −0.0019).
  - For BCE, H1 warm is at or below T1.
  - Knownness is a relative cue. Under monthly, newly released competitors become "known" and are demoted, which helps warm targets.
  - H3 reproduces part of the monthly warm gain from the knownness column alone (+0.0003 to +0.0008 versus T1).
- **H2 (degree also from T1):**
  - Text models return almost exactly to T1 (−0.0001). So for text rankers, the monthly-versus-T1 differences come almost entirely from knownness plus degree.
  - Co-review models keep a gain over T1 (+0.0003 pairwise, +0.0008 BCE). This is consistent with fresher co-review paths carrying real information.

## Caveats

- Mixed snapshots are a diagnostic of what the frozen models use, not a recommended serving recipe.
- H1 feeds combinations the model never saw (unknown flag with a positive degree or co-review paths). Its large BCE new-item gains may be extrapolation, not recovery.
- Inputs are off the T0-era scaler distribution. These are inference substitutions in frozen models; no retraining was done.
- Val has been used before. The intervals are unadjusted and conditional on these checkpoints and candidates. Nothing is selected.

## Files

- metrics.csv, contrasts.csv, training_knownness.csv, knownness_probe.csv, score_checks.json, config.yaml, git_hash.txt
- Bulk ranks: results/local_experiments/20261106-freshness-mechanism/
- Code: src/cascade/analysis/freshness_mechanism.py
- Tests: tests/test_freshness_val.py (7 tests pass)
