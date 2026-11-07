# Context-arm study: leak-safe context columns for `bce_text` (6 Nov 2026)

Status: **pre-registered, Phase 1 only** (code, tests and a CPU sample smoke). The full run has not started.

## Question
Does leak-safe context information improve the registered residual ranker beyond its text features?

## Pre-stated hypothesis (recorded before any full-data fit)
A small overall nDCG@10 gain over `bce_text`, mostly on warm targets (popularity and rating signal), with no gain on new items. The gain may fall within training noise.

## Design
- Recipe: identical to `bce_text` in `src/cascade/pipelines/full_data_ranking.py`. It uses GraphLatentRanker R2 with the graph columns zeroed, BCE, and the source study's 50-uniform-negative plans. Optimiser settings: Adam 1e-3, 32 lists per batch, at most 10 epochs, patience 2. Selection is chronological on the inner split, with epoch 0 eligible, followed by a fixed-epoch refit on all of B. Seeds are 0/1/2.
- Inputs, read only: `results/local_experiments/20261103-full-promising-ranking/`. This covers the candidates, the scaled 9-column common features, the plans and the prefix/tune split. Nothing is rebuilt.
- Change: 63 one-hot columns are appended to the "common" block, so common_dim goes from 9 to 72. They are standardised over all 200 slots of the chronological-prefix B queries.
  - candidate price decile (11 levels). The cut points come from pre-T0 items with a parseable price, plus a "missing" level.
  - candidate review count at the period start, as floor(log2(n+1)) (14 levels).
  - candidate mean rating at the period start: "none" plus rounded 1–5 (6 levels).
  - candidate item age at the period start, as floor(log2(1+days)) (14 levels), plus one shared "new" level for items first reviewed at or after the period start (15 levels in total).
  - user history-length bucket: 1–2, 3–5, 6–10, 11+ (4 levels).
  - user time since last interaction, as floor(log2(1+days)) (13 levels).
- Period start: T0 = 1595800947930 for B queries and T1 = 1628643414042 for Val queries. Candidate statistics use only interactions strictly before the period start, never anything relative to the query time. Metadata `average_rating` and `rating_number` are never used.
- Reuse: the review-count, rating and age tables come from `RankerFeatureBuilder` (§4.5 per-period tables). A strict `timestamp < cutoff` recount cross-checks them. Price is recomputed because the builder fits its deciles on the whole catalogue. Age is remapped because the builder puts unseen items in the padding code.
- Initial state: the shared parameters equal the `bce_text` initial state exactly; the hash is checked against the control's metrics. The new head-input columns are drawn U(±1/sqrt(276)) from generator 1000003+seed. The global RNG continues as it did for the control.
- Control: the `bce_text` runs and Val evaluation already in the source study, paired by event.

## Leak checks (full data, dry run, no writes)
- Val has exactly 656 candidates whose first review equals the query timestamp, and all of them are targets. B has 900 such candidates, also all targets.
- These candidates carry the same period-dependent codes as every other new candidate (review 0, rating none, age "new"). That leaves one distinct scaled row in each group, so the columns cannot separate them.
- Latest interaction used: B 1595800834936 < T0; Val 1628643402825 < T1.
- Caveat: price is static metadata from the crawl. Its distribution differs slightly between the same-timestamp targets and other new candidates (total variation 0.088 on Val, mostly in missingness). This is not period-dependent, but price-missingness may carry weak crawl-time information. The same caveat applies to the existing store and category fields.
- Interpretation caveat: on Val, the review-count and age columns reflect the T1 snapshot. The control's existing common column 0 is pre-T0 log degree. Part of any gain could therefore come from refreshing popularity to the period start rather than from new kinds of information.

Exploratory reuse of a previously used Val population. No Test data is opened.
