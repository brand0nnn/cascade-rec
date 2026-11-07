# Experiment overview (Val only, exploratory)

A one-page index of every system scored on Val during the 5–7 Nov 2026 runs, plus the team's earlier runs they build on. Full write-ups:
- `docs/overnight_results_2026-11-05.md`
- `docs/overnight_results_2026-11-06.md`
- `docs/evaluations.md`

No system was selected, and no Test file was opened.

## Terms

### Time periods

| Period | Dates | Used for |
|---|---|---|
| A | before **T0** = 26 Jul 2020 | training the two-tower retriever |
| B | T0 to **T1** = 11 Aug 2021 | ranker training (315,114 queries) |
| Val | T1 to **T2** = 16 Jul 2022 | evaluation (232,904 events, 124,416 users) |
| Test | after T2 | untouched |

### Retrievers
- **Recency:** frozen MiniLM text vectors ("title | category path | store"). The user vector is a recency-weighted mean of the last 50 history items (90-day half-life), scored by cosine. No training.
- **Old two-tower:** a text item tower (MiniLM → MLP) and a self-attention user tower, trained on **A only**.
- **Fresh two-tower:** same recipe, trained on **A + B**. It has seen period B, so it is only used to serve Val lists; no ranker is trained on its lists.

### Candidate lists (200 per query, saved for every B and Val query)
- **P1:** recency top 200.
- **P2:** old two-tower top 200.
- **P3, the "mixed" pool:** alternate recency #1, two-tower #1, recency #2, … skipping duplicates until 200 unique items. That is about the top 100 of each, with ~10 overlapping. Its retrieval score is RRF: Σ 1/(60 + rank).
- **P2′ / P3′:** the same constructions with the fresh two-tower.

### Ranker (one design, many trained versions)
- **Score:** retrieval score + MLP correction (128→64→1). The MLP's last layer starts at zero, so an untrained ranker reproduces the retrieval order.
- **Training data:** period-B lists of the same pool, natural lists only (target in the 200). Each example is the target plus 50 random negatives from its list.
- **Loss:** BCE unless marked pairwise. Pairwise = softplus(negative − positive).
- **Recipe:** Adam 1e-3, batch 32, ≤ 10 epochs, patience 2. Epochs are chosen on the last 20% of B, then the model is refit on all of B. Seeds 0/1/2.

### Features per candidate
- **text:** text interactions with the history, leaf and store match, cosine to the history, and 9 support features.
- **co-review:** + 4 counts from the co-review graph (products linked when the same user reviewed both): other reviewers, witness paths, normalised path strength, supported-history fraction.
- **context:** + 63 one-hot columns: price decile; review count, rating and age at the period start; history length; time since the last review.

### Graph cut dates
- **T0:** reviews before T0.
- **T1:** reviews before T1, the start of Val.
- **Monthly:** reviews before the first day of each query's month (about 14 graphs for B, 12 for Val).

The graph also supplies 3 of the 9 support features in *every* ranker, including the "has any reviews" flag.

### Metric
- nDCG@10, per-user macro over the full Val population, misses = 0.
- Seed means; paired-user bootstrap CIs are in the run folders.
- **Rescored** = an existing model scored with different Val features, no retraining.

## A. Retrieval only (no ranker)

| # | Retriever(s) | List | Order | nDCG@10 |
|---|---|---|---|---|
| A1 | Recency | P1 | recency score | 0.0063 |
| A2 | Old two-tower | P2 | two-tower score | 0.0056 |
| A3 | Recency + old two-tower | P3 | RRF score | 0.0066 |
| A4 | Recency + old two-tower | P3 | alternating order | 0.0074 |
| A5 | Fresh two-tower (seeds 0/1) | P2′ | two-tower score | 0.0100 |
| A6 | Recency + fresh two-tower | P3′ | RRF score | 0.0090 |
| A7 | Recency + old two-tower | best of P1/P2 per event | oracle (knows the target) | 0.0117 (upper bound only) |
| A8 | Recency + device-upgrade rule | recency top 180 + 20 newer-model items | recency order | 0.0063 (top 10 unchanged; Recall@200 +0.0014) |

Sources: `results/analysis/20261106-pools`, `…-fresh-two-tower`, `…-device-upgrade`.

## B. Ranker on the recency pool (P1)

| # | List | Features (loss) | Train graph | Val graph | nDCG@10 | Note |
|---|---|---|---|---|---|---|
| B1 | P1 | text (BCE): **current pipeline** | T0 | T0 | **0.0081** | team model, rebuilt 5 Nov |
| B2 | P1 | text (BCE) | T0 | T1 | 0.0094 | rescored |
| B3 | P1 | text (BCE) | T0 | monthly | 0.0085 | rescored |
| B4 | P1 | text (pairwise) | T0 | T0 / T1 / monthly | 0.0078 / 0.0097 / 0.0087 | rescored |
| B5 | P1 | text + co-review (pairwise) | T0 | T0 / T1 / monthly | 0.0077 / 0.0102 / 0.0098 | rescored |
| B6 | P1 | text + co-review (BCE) | T0 | T0 / T1 / monthly | 0.0082 / 0.0098 / 0.0096 | rescored |
| B7 | P1 | text (BCE), 4 training draws × 3 seeds | T0 | T0 / T1 | 0.0081 / 0.0093 | robustness |
| B8 | P1 | text + co-review (BCE), 4 draws × 3 seeds | T0 | T0 / T1 | 0.0083 / 0.0098 | robustness |
| B9 | P1 | text (BCE) | monthly | monthly | 0.0093 | retrained |
| B10 | P1 | text + co-review (BCE) | monthly | monthly | **0.0106** | retrained |
| B11 | P1 | text + context (BCE) | T0 (context at period start) | T0 (context at T1) | 0.0109 | mixed dates |
| B12 | P1 | text + context (BCE), same model as B11 | — | all at T1 | 0.0107 | fair comparison |
| B13 | P1 | text + context (BCE), same model as B11 | — | all at T0 | 0.0091 | context made stale |
| B14 | P1 + randomly inserted missed targets | text (BCE), random mixing | T0 | T0 | 0.0071 | training-label change, pending team review |

Sources: `results/local_experiments/20261103-full-promising-ranking`, `…-20261104-objective-coreview`, and `results/analysis/20261106-`:
- `freshness-val`, `freshness-mechanism`
- `refit-draws`
- `rolling-ranking`
- `context`
- `random-mix`

## C. Ranker on the two-tower pool (P2)

| # | List | Features | Train graph | Val graph | nDCG@10 | Note |
|---|---|---|---|---|---|---|
| C1 | P2 | text (BCE) | T0 | T0 | 0.0101 | retrained |
| C2 | trained on P2, scored on P2′ | text (BCE) | T0 | T0 | 0.0099 | rescored on fresh lists |

Sources: `results/analysis/20261106-pool-ranking`, `…-fresh-two-tower`.

## D. Ranker on the mixed pool (P3)

| # | List | Features | Train graph | Val graph | nDCG@10 | Note |
|---|---|---|---|---|---|---|
| D1 | P3 | text (BCE) | T0 | T0 | 0.0110 | retrained |
| D1s | P3, trained on 20,676 positives (P1's count) | text (BCE) | T0 | T0 | 0.0104 | size-matched control (7 Nov) |
| D2 | P3 | text (BCE) | T0 | T1 | 0.0115 | rescored |
| D3 | trained on P3, scored on P3′ | text (BCE) | T0 | T0 | 0.0114 | rescored on fresh lists |
| D4 | P3 | text + context (BCE) | T0 (context at period start) | T0 (context at T1) | 0.0127 | mixed dates |
| D5 | P3 | text + context (BCE), same model as D4 | — | all at T1 | 0.0122 | fair comparison |
| D6 | P3 | text (BCE) | monthly | monthly | 0.0115 | retrained |
| D7 | P3 | text + co-review (BCE) | monthly | monthly | **0.0131** | **best fair system** |
| D8 | P3 | text + co-review (BCE), same model as D7 | — | T1 | 0.0136 | mixed dates (diagnostic) |
| D9 | P3 | text (BCE), same model as D6 | — | T1 | 0.0127 | mixed dates (diagnostic) |

Sources: `results/analysis/20261106-`:
- `pool-ranking`
- `pool-combo`
- `fresh-two-tower`

and `results/analysis/20261107-pool-rolling`, `…-20261107-pool-size-control` (D1s).

## E. Locked Test run (7 Nov 2026, protocol `docs/test_protocol_2026-11-07.md`)

Frozen models, each scored once on Test (297,390 events / 112,081 users; 44.4% new targets). Inputs were rebuilt with the same builders, which first reproduced every saved Val list, feature and rank exactly.

| System | Model (Val row) | Test lists | Test statistics | nDCG@10 all | new | warm |
|---|---|---|---|---:|---:|---:|
| S0 | none (recency order) | P1 | — | 0.0060 | 0.0076 | 0.0054 |
| S1 | B1 | P1 | T0 | 0.0064 | 0.0082 | 0.0057 |
| S2 | B1 | P1 | T1 | 0.0074 | 0.0099 | 0.0062 |
| S3 | B1 | P1 | T2 (period start) | 0.0078 | 0.0118 | 0.0058 |
| S4 | B9 | P1 | monthly | 0.0080 | 0.0102 | 0.0070 |
| S5 | B10 | P1 | monthly | 0.0097 | **0.0130** | 0.0082 |
| S6 | D1 | P3 | T0 | 0.0079 | 0.0064 | 0.0082 |
| S7 | D6 | P3 | monthly | 0.0088 | 0.0076 | 0.0088 |
| S8 | D7 | P3 | monthly | **0.0104** | 0.0092 | **0.0105** |

Primary contrasts (nDCG@10, all): C1 S3 − S1 +0.0014; C2 S6 − S1 +0.0014 (new −0.0018, warm +0.0025); C3 S5 − S4 +0.0017; C4 S8 − S1 +0.0039 (+61%). All 95% intervals exclude zero. Full tables: `results/test_final/20261107/tables.md`.

## F. Statistics-timing diagnostics (7 Nov 2026, Val; frozen B1/B4/B5/B6, no training)

`results/analysis/20261107-statistics-timing/` (hypotheses written before any arm was scored; all five held).
- **T0 -> T1 column swap.**
  - Swapping only the "has any reviews" flag recovers 26–87% of the T0 -> T1 gain overall, and 44–208% on new targets.
  - Flag + review count recovers about 90–114%.
  - Updating only products released during period B recovers 86–101%; updating only older products recovers 14–24%.
  - So the T0 problem is products released during B being misread as "new", plus their missing counts.
- **Fixed cut date T0 -> T1 - 9/6/3 months -> T1.** nDCG@10 rises monotonically for all four models.
- **Fixed lag before each query's month (0–12 months).**
  - New-product quality is lowest at lag 0 and peaks at lag 6–9, where the statistics are about as old as in training (mean 6.1 months).
  - Established-product quality falls steadily with lag.
  - So the learned rule wants statistics of training-matched age, while the counts themselves are better when fresh.
- **Test replication** (addendum protocol, same frozen models, T2 in place of T1): all five predictions held. New-product quality peaks at lag 6 for all four models. Results: `results/test_final/20261107/timing/`.

## How to read it

**Key single-change comparisons:**

| Comparison | Effect |
|---|---|
| B1 → D1 | pool, +35% |
| B1 → B2 | fresher graph at scoring time |
| B9 → B10 and D6 → D7 | co-review with fresh graphs (+0.0013 / +0.0015) |
| B2 → B12 | context, fairly compared (+0.0013) |
| B1 → D7 | best fair system, +61% |

**"Mixed dates" rows are diagnostics, not fair systems.** They pair training-time and scoring-time features from different dates, which over-rewards the "no reviews yet = new product" signal. This applies to B11, D4, D8 and D9.

**Fresh-retriever rows** (A5, A6, C2, D3) use a retriever that saw period B; no ranker was trained on its lists.

**Caveats:**
- Val was reused to choose directions; a locked Test run is needed before any number is final.
- Product titles leak future device names (at least 8.5–21% of top-10 slots).
- The 5-core filter, computed over all time, flatters newness signals.

## Non-ranking analyses (no table rows)

| Analysis | Folder |
|---|---|
| Case study of 24 targets missed by both retrievers, plus device-change counts | `results/analysis/20261106-switch-cases`, `…-switch-cases-devices` |
| Title-leak audit | `results/analysis/20261106-title-leak` |
| Winners by user group, popularity and category | `results/analysis/20261106-meso-segments` |
| Fresh-retriever seed robustness | `results/analysis/20261107-fresh-tt-seeds` |
| 5 Nov analyses: U-shape by popularity, ranker oracle and blend, HR@3/MRR@10, first case study | `docs/overnight_results_2026-11-05.md` |
