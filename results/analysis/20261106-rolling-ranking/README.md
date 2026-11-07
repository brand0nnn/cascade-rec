# Rolling-monthly-snapshot ranker training (Val only, exploratory)

## Pre-stated hypothesis (written 2026-11-05T18:23:26Z, before any rolling feature was built or model fitted)

Rolling-trained models served monthly beat the corresponding T0-trained models served at T1 overall, with gains on
warm and no large new-item loss, because knownness now means "no reviews in the last month" consistently;
co-review − text under rolling is positive with CI above zero.

Context: the coordinator's reading of 20261106-freshness-mechanism is that train/serve-consistent features are
"features at the query period's start". B was trained on T0 features (B's start) and Val is served on T1 features (Val's start).
The published T0-for-Val numbers were therefore a stale-serving mismatch, and monthly serving breaks the learned knownness semantics.

## Interim note: features built (prepare finished 2026-11-06 02:46 +08, 979 s on CPU, 3 threads)

- **B features:** 315,114 queries in 14 snapshots. The first is the T0 snapshot for the 4,213 queries on 26–31 Jul 2020 (their period start). The other 13 are month starts from 1 Aug 2020 to 1 Aug 2021, with 1.88M → 2.20M edges.
- **Checks per snapshot:**
  - The latest review is before the earliest query.
  - No query user has an edge to any of its candidates or its target.
  - Every query-user edge lies in that user's strict history.
  - The batched build matches the explicit-exclusion reference (max error ≤2.3e-8).
  - The 26–31 Jul 2020 rows reproduce the parent's T0 features exactly.
  - The metadata columns 3–6 are unchanged.
- **Prefix:** the inner prefix ends in May 2021, so the June–August 2021 snapshots feed only tune and refit rows.
  The rolling prefix scaler is fitted over 50,418,200 prefix slots (252,091 queries × 200).
- **Scaler shift (prefix mean, T0 → rolling):**
  - candidate_graph_known 0.872 → 0.988 (std 0.334 → 0.109): under rolling, only about 1.2% of prefix candidate slots are unknown.
  - candidate_log_degree 2.32 → 2.60.
  - history_graph_fraction 0.926 → 0.992.
- **Val features:** monthly (consistent) and T1 (diagnostic) features were rebuilt for the 14,916 natural-hit rows and standardised with the rolling scaler.

## Results (evaluated 2026-11-06; fits on MPS, Val scoring on CPU)

**Fits.** All six fits completed, each starting from exactly the same initial state as its source fit, with the source negative plans.

| Method | Selected epochs: rolling (seeds 0/1/2) | Selected epochs: source | Inner nDCG@10: rolling | Inner nDCG@10: source |
|---|---|---|---|---|
| bce_text | 8/6/6 | 8/8/7 | 0.01247 / 0.01204 / 0.01216 | 0.01074 / 0.01059 / 0.01071 |
| bce_coreview | 5/10/7 | 8/5/7 | 0.01322 / 0.01330 / 0.01323 | 0.01113 / 0.01057 / 0.01090 |

The inner values are not directly comparable: the B tune rows carry rolling features here and T0 features in the source.

**nDCG@10 (seed-mean, all 232,904 events / 124,416 users).**

| Method | Arm | All | New | Warm |
|---|---|---:|---:|---:|
| bce_text | existing, served at T0 | 0.008141 | 0.008666 | 0.007887 |
| bce_text | existing, served at T1 | 0.009432 | 0.011721 | 0.008231 |
| bce_text | rolling, served monthly | 0.009304 | 0.008699 | 0.009579 |
| bce_text | rolling, served at T1 (diagnostic) | 0.010174 | 0.016476 | 0.007114 |
| bce_coreview | existing, served at T0 | 0.008167 | 0.008250 | 0.008105 |
| bce_coreview | existing, served at T1 | 0.009801 | 0.011132 | 0.009055 |
| bce_coreview | rolling, served monthly | 0.010613 | 0.010639 | 0.010508 |
| bce_coreview | rolling, served at T1 (diagnostic) | 0.010605 | 0.017100 | 0.007370 |

**Contrasts, nDCG@10 [95% paired-user CI].**

| Contrast | All | New | Warm |
|---|---|---|---|
| bce_text: rolling monthly − existing T1 | −0.000128 [−0.000305, 0.000053] | −0.003022 [−0.003491, −0.002547] | +0.001347 [0.001187, 0.001507] |
| bce_text: rolling monthly − existing T0 | +0.001163 [0.000953, 0.001389] | +0.000034 [−0.000418, 0.000473] | +0.001692 [0.001467, 0.001933] |
| bce_coreview: rolling monthly − existing T1 | +0.000811 [0.000614, 0.001002] | −0.000493 [−0.000978, −0.000034] | +0.001454 [0.001266, 0.001629] |
| bce_coreview: rolling monthly − existing T0 | +0.002445 [0.002199, 0.002675] | +0.002390 [0.001932, 0.002869] | +0.002404 [0.002141, 0.002676] |
| rolling monthly: bce_coreview − bce_text | +0.001308 [0.001129, 0.001483] | +0.001940 [0.001667, 0.002241] | +0.000930 [0.000738, 0.001137] |
| rolling T1: bce_coreview − bce_text | +0.000431 [0.000282, 0.000585] | +0.000624 [0.000388, 0.000857] | +0.000256 [0.000072, 0.000437] |
| bce_text: rolling T1 − existing T1 (diagnostic) | +0.000742 [0.000577, 0.000903] | +0.004755 [0.004360, 0.005102] | −0.001118 [−0.001283, −0.000968] |
| bce_coreview: rolling T1 − existing T1 (diagnostic) | +0.000804 [0.000637, 0.000971] | +0.005968 [0.005600, 0.006344] | −0.001685 [−0.001858, −0.001521] |
| bce_text: rolling monthly − rolling T1 | −0.000870 [−0.001059, −0.000682] | −0.007777 | +0.002465 |
| bce_coreview: rolling monthly − rolling T1 | +0.000008 [−0.000207, 0.000205] | −0.006460 | +0.003139 |
| bce_text: rolling monthly − existing monthly | +0.000827 [0.000695, 0.000971] | +0.002518 | +0.000008 |
| bce_coreview: rolling monthly − existing monthly | +0.000972 [0.000832, 0.001119] | +0.003346 | −0.000185 |

Signs for Recall@10, HR@3 and MRR@10, rolling monthly − existing T1, all events:
- bce_coreview: positive on all three. Recall@10 +0.000788, HR@3 +0.000978, MRR@10 +0.000813, all with CIs above zero.
- bce_text: Recall@10 −0.000609 [−0.000942, −0.000327]. HR@3 −0.000120 and MRR@10 +0.000022 are inconclusive.
- Full table: contrasts.csv.

**Knownness under rolling training (B prefix natural lists, 16,819).**
- 11.2% of positives are unknown, versus 2.0% of negatives.
- An unknown slot is the target 6.1 times as often as a known one (0.0278 vs 0.0045). Under T0 features the ratio was 3.1 (0.0114 vs 0.0036).
- So "unknown" became rarer and more strongly tied to the target. It now means "released in the current month".

## Verdict on the pre-stated hypothesis: partly supported

- **bce_coreview: supported.** Rolling monthly beats existing T1 overall by +0.00081 [0.00061, 0.00100] and on warm by +0.00145. The new-item loss is small (−0.00049 [−0.00098, −0.00003]).
- **bce_text: not supported.** Overall the difference is inconclusive (−0.00013 [−0.00031, 0.00005]). Warm gains (+0.00135), but new loses −0.0030, which is large.
- **Co-review − text under rolling monthly: supported.** +0.00131 [0.00113, 0.00148], positive in every group. That is larger than under existing T1 (+0.00037) or existing T0 (+0.00003).
- **The knownness cue persists with a sharper meaning.**
  - If rolling-trained models are served the stale T1 snapshot, every Val-released product is "unknown". The models then boost new items strongly (new +0.0048 to +0.0060 versus existing T1) at a warm cost (−0.0011 to −0.0017).
  - Monthly serving gives the reverse trade.
  - For bce_coreview, the two serving rules tie overall (+0.00001). So for this model, consistent serving does not raise overall nDCG@10; it moves quality from new to warm.

## Caveats

- Each method has three seeds. The selected epochs differ from the source, so each rolling-minus-existing contrast mixes the change of features with the change of selected training epochs.
- The scalers differ between arms: rolling uses the rolling prefix scaler, existing uses the T0 scaler.
- Val has been used before, the intervals are unadjusted, and nothing is selected. Rolling-T1 serving is a diagnostic, not a serving proposal.
- The partial first B month (26–31 Jul 2020) uses the T0 snapshot.
- Val August 2021 queries use 1 Aug 2021, which is before T1.
- No Test file was opened.

## Files

- metrics.csv, contrasts.csv, epochs.csv, features_manifest.json, inner_history/, config.yaml, git_hash.txt
- Bulk (B rolling features 3.8 GB, Val features, weights, ranks): results/local_experiments/20261106-rolling-ranking/
- Code: src/cascade/pipelines/rolling_ranking.py
- Logs: results/logs/20261106/freshness-rolling-*.log
