# Meso-level segments: which system wins for which users/products? (Val only, exploratory)

**Label: Val only, exploratory; descriptive, nothing selected.** No Test file is opened or scored. No model is
trained or re-scored: every system is read from its saved per-event Val ranks (232,904 events / 124,416 users).

## Pre-stated expectation (written 2026-11-06 21:50 +0800, before any segment metric was computed)

1. The fused pool's (P3) advantage over P1 bce_text concentrates in **warm targets** and **longer histories**.
2. Context/newness systems (P3 + context at T1; P1 retrieval order, i.e. 90-day recency) win for **new-in-Val
   targets**.
3. **No single system wins every segment.**

## Question

The professor asked for meso-level analysis: explain why results change and how settings could be customised per
user group or product category. Here, for each pre-query segment, we report which saved system has the highest
per-user-macro nDCG@10, and its paired difference against the common reference P1 bce_text.

## Systems (saved per-event Val ranks; seed means where seeds exist)

| Key | System | Source | Published overall nDCG@10 |
|---|---|---|---:|
| P1_retrieval | P1 90-day recency retrieval order (K200) | `results/local_experiments/20261103-full-promising-ranking/candidates/val/retrieval_rank.npy` | 0.006284 |
| P1_bce_text | P1 bce_text ranker, T0 features (reference) | `.../20261103-full-promising-ranking/evaluation/bce_text/seed{0,1,2}/val_ranks.npz` | 0.008141 |
| P3_bce_text | P3 fused-pool bce_text ranker, T0 | `.../20261106-pool-ranking-P3/evaluation/bce_text/seed{0,1,2}/val_ranks.npz` | 0.011019 |
| P3_context_T1 | P3 + context ranker, all features at T1 | `.../20261106-pool-combo-context/evaluation/common_T1/seed{0,1,2}.npz` | 0.012167 |
| rolling_coreview | Rolling bce_coreview, served monthly | `.../20261106-rolling-ranking/evaluation/val_monthly/bce_coreview/seed{0,1,2}.npz` | 0.010613 |
| P2f_retrieval | Fresh two-tower (all pre-T1) P2' retrieval order | `.../20261106-fresh-two-tower/candidates/P2f_two_tower_all_train/val/retrieval_rank.npy` | 0.010035 |

Before use, each system's overall nDCG@10 (and per-seed values where seeds exist) must reproduce the published
values in `results/analysis/20261106-{pool-combo,fresh-two-tower,rolling-ranking}/metrics.csv` to 1e-9; event/user
alignment with the Val instance table is checked for every file.

## Segments (all from information available before the query; frozen in `config.yaml`)

- **History length** at query (prior interactions): 1–2, 3–5, 6–10, 11+ (CLAUDE.md §4.7).
- **Target newness:** new-in-Val (first interaction ≥ T1) vs warm; warm further split into terciles of the target's
  pre-T1 review count (cut points = 1/3 and 2/3 quantiles over warm Val events).
- **Target second-level category** (`categories[1]`, "(none)" when missing): the 8 most frequent values by Val event
  count, the rest pooled as "other".
- **User activity gap:** time since the user's last prior interaction: ≤ 7 days, > 7 and ≤ 90 days, > 90 days.

Per segment × system: per-user-macro nDCG@10 within the segment (users with ≥ 1 event in the segment), event and user
counts, and the paired difference vs P1 bce_text with a paired user bootstrap (1,000 draws, seed 0, seed-mean event
metrics). Per segment, the system with the highest point estimate and the bootstrap CI of (best − second best).

## Caveats

- Descriptive only. The "best per segment" is not a selection and must not be used to choose a deployed router:
  segment-wise maxima over six systems are optimistic (winner's curse), the intervals are unadjusted for
  multiplicity, and Val has been inspected repeatedly by earlier studies.
- The systems differ in several factors at once (pool, features, freshness, objective), so a segment gap is not
  attributable to a single cause.
- Tercile and top-8 category cut points use the Val event distribution of pre-query attributes (descriptive
  binning, not tuning).

Results: `tables.md`, `segment_metrics.csv`, `winners.csv`, `figure_meso.png`. Code:
`src/cascade/analysis/meso_segments.py`.

## Outcome against the pre-stated expectation (added after computing, 2026-11-06)

All six systems reproduced their published overall nDCG@10 (and per-seed values) to 1e-9 (`checks.json`).

1. **Partly supported.** P3 bce_text's gain over P1 bce_text is in warm targets (+0.0048 vs −0.0017 on new), but
   within warm it sits entirely in the most-reviewed tercile (+0.0180; low −0.0017, mid −0.0013). It does **not**
   grow with history length (+0.0019 at 11+ vs +0.0027–0.0030 below).
2. **Partly supported.** P3 + context (T1) has the highest new-in-Val point estimate (0.010815), but it is tied with
   rolling bce_coreview (+0.000175 [−0.000391, +0.000698]); P1 recency retrieval order does not win new targets.
3. **Supported.** Five different systems top at least one of 22 rows: P3 + context T1 (9 rows, incl. overall and all
   history buckets), rolling bce_coreview (7), fresh two-tower P2' order (4), P3 bce_text (1), P1 bce_text (1).

Small categories (< 500 events) have wide intervals; treat their winners as noise-prone.
