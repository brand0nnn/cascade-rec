# Project overview (read this first)

CP4285 Group 7: Brandon Kang, Phua Qing Rong, Tong Jing Yen. National University of Singapore.

This file explains the whole project in the order of the v2 poster story. It is self-contained. `POSTER_BRIEF.md` says what goes on the poster; this file explains why.

The files in `evidence/` are the full working records. They use internal codes (decoded in §8) and an older framing, built around "three shortcuts", including forced training lists. **That older framing is superseded:** the team decided not to present the forced-list finding. Use the evidence files only to look up numbers.

## 1. What the project is

Online shops can't run an expensive model over every product for every visitor, so they recommend in two stages:

1. A **retriever** quickly picks a shortlist of candidates. Here: 200 out of 111,480 products.
2. A **ranker**, a heavier model, orders the shortlist. The user sees the top 10.

Our critique plan (29 Sep 2026, `reference/critique_slides_29sep.pdf`) asked:

- **RQ1:** Does a trained two-tower retriever find more purchases than standard baselines (Popularity, ItemKNN, BPR-MF, SASRec)?
- **RQ2:** How do the number of candidates (K) and the search method (exact Flat, HNSW, IVF) affect the final top 10 and its speed?
- **RQ3:** On identical candidates, how do popularity order, retriever order, an MLP ranker and an advanced ranker (DIN) compare?
- **Final:** Does the best pipeline beat the best single-stage model? Where do failures happen?

The plan was presented to the professor at the start. `evidence/initial_plan_20260929.md` records it and how far we followed it.

## 2. Data and protocol

- **Data:** Amazon Reviews 2023, Cell Phones & Accessories (5-core): 2.75M reviews, 381k users, 111k products. Each review counts as a purchase.
- **Split by time:**

  | Period | Dates | Used for |
  |---|---|---|
  | Retriever training | before 26 Jul 2020 | training the retrievers |
  | Ranker training | 26 Jul 2020 – 11 Aug 2021 | training the rankers |
  | Validation | 11 Aug 2021 – 16 Jul 2022 | every choice; 232,904 purchases |
  | Test | from 16 Jul 2022 | opened once, with predictions written first; 297,390 purchases, 44% of them new products |

- **Metric:** nDCG@10. A purchase scores 1 if the bought product is ranked 1st, 0.5 if 3rd, 0.29 if 10th, and 0 if it is outside the top 10 or was never retrieved. The score is averaged per user, then over users, and shown ×10⁻³ (so "8.27" means 0.00827). It is low because about 94% of purchases never reach the 200 candidates.
- **Statistics:** 3 training seeds per ranker. 95% intervals come from 1,000 paired resamples of users.
- **Ranker training data:** rankers learn from lists where the retriever actually found the purchase (the "natural" lists). All poster results use this.

## 3. The systems

**Retrievers:**
- **Two-tower:** our trained model. Product text goes through a frozen MiniLM encoder; a SASRec-style user encoder reads the purchase history; in-batch softmax with logQ correction; served with FAISS.
- **Recency retriever:** no training. It averages the text vectors of the user's recent purchases, weighting recent ones more.
- **Mixed lists:** the recency and two-tower lists alternate, without duplicates, until there are 200 products. The best DIN system also uses a combined score: reciprocal-rank fusion of both retrievers.

**Rankers:**
- **MLP:** features such as the retrieval score, text similarity to the history, store/category match and product statistics.
- **DIN:** the MLP plus target attention over the history.
- Both are trained with binary cross-entropy.

**Product statistics** given to the ranker: review count, mean rating and product age, computed from reviews before some date.

## 4. Answers to the planned questions

**RQ1: yes.** Validation, share of purchases among the 200 candidates:

| Retriever | Share |
|---|---:|
| Popularity | 2.7% |
| ItemKNN | 0.8% |
| BPR-MF | 2.7% |
| SASRec | 4.0% |
| Two-tower | **6.9%** |
| Recency | 5.5% |
| Mixed | **8.0%** |

- The two retrievers share only 12 of their 200 products: two-tower finds popular ones, recency finds new and niche ones.
- Only text-based retrievers find new products.

**RQ2.** Two-tower lists, trained MLP ranker, Validation:

| K | Purchases in the list | Top-10 quality |
|---:|---:|---:|
| 50 | 3.0% | 8.33 |
| 100 | 4.6% | 8.61 |
| 200 | 6.9% | 8.57 |
| 500 | 11.8% | 7.92 |

- The best K is 100–200; they are statistically tied. K = 500 vs K = 100: −0.00069 [−0.00087, −0.00048].
- **Search method** (K = 200): HNSW keeps 99.4% of the exact top 200 in 1.5 ms vs 3.4 ms, with the same top-10 quality (7.37). IVF keeps 94.6% and drops to 6.75.

**RQ3.** Held-out Test, nDCG@10:

| Ordered by | Recency lists | Two-tower lists | Mixed lists |
|---|---:|---:|---:|
| Popularity | 1.51 | – | – |
| Retriever's own order | 6.04 | – | – |
| MLP | 6.44 | 5.19 | 7.30 |
| DIN | – | 6.37 | 8.27 |

- DIN beats the MLP on the same lists: +23% on two-tower lists, +13% on mixed lists (+0.0008 to +0.0011).
- Moving DIN from two-tower to mixed lists: +30%.
- Changing the loss (BCE → pairwise) moved Test by at most 0.0005.

## 5. The central finding: a better list does not help a ranker trained on other lists

The RQ2 result raised the question: if a list contains more purchases, why does the top 10 get worse? We tested it three ways. Each time the ranker was trained on one kind of list and then, without retraining, given a list containing more purchases. All Validation.

### Test 1: 500 instead of 200 candidates (same two-tower retriever, ranker trained on 200-lists)
- Purchases in the list: 6.9% → 11.8% (+71%).
- Top 10: 8.57 → 7.92 (−8%).
- The newly found purchases add +1.3, but the extra 300 candidates push down purchases the ranker already ranked well: −2.0 (3-seed mean).

### Test 2: a fresher retriever
The two-tower design was retrained on all data up to Aug 2021 instead of Jul 2020. Its lists went to the ranker trained on the old two-tower's lists.

Purchases in the list: 6.9% → 11.1% (+62%; 3 seeds: +0.0425 [0.0413, 0.0436]).

| nDCG@10 ×10⁻³ | Old list | Better list | Change [95% CI] |
|---|---:|---:|---|
| Retriever's own order, all purchases | 5.64 | 10.04 | +4.39 [+4.10, +4.70] |
| Frozen ranker, all purchases | 10.08 | 9.93 | −0.15 [−0.37, +0.08], no detectable change |
| Frozen ranker, new products | 3.70 | 2.82 | −0.88 [−1.17, −0.57] |
| Frozen ranker, products first reviewed before Jul 2020 | 20.33 | 18.32 | −2.00 [−2.37, −1.64] |
| Retriever order, products first reviewed before Jul 2020 | 14.84 | 23.27 | |

- On the better list, the ranker overall is no better than simply keeping the new retriever's own order: −0.10 [−0.45, +0.21].
- For established products it is worse than that order: −1.26 [−1.72, −0.79]. For products first reviewed before Jul 2020: −4.95 [−5.69, −4.18].
- **A second version** gave the fresher retriever's mixed lists to the mixed-list ranker: +0.37 [+0.16, +0.57] overall, and −0.84 on new products. The research story quotes this as "only +0.0004". Don't confuse the two versions.
- **Caveats:** single retriever seed; the fresher retriever was trained on a different device and code version; Validation also chose its epoch.

### Test 3: mixed lists given to a ranker trained on recency lists
- Purchases in the list: 5.5% → 8.0% (+45%).
- Top 10: 8.14 → 8.81 (+8%).
- New products dropped from 8.67 to 5.35 (no interval computed).

### Interpretation (ours)
A ranker learns patterns in the data it was trained on, for example what candidates from its retriever usually look like. When the lists or inputs change, those patterns no longer fit.

Overall the frozen ranker gets little or nothing from a better list. It is clearly worse only for some product groups. **Never claim it is "worse overall".**

## 6. The fix: train the ranker on what it will see

### Fix A: retrain on the new lists (mixed lists)

| Validation | nDCG@10 |
|---|---:|
| Recency lists, its own ranker | 8.14 |
| Mixed lists, same ranker (not retrained) | 8.81 (+8%) |
| Mixed lists, retrained, same amount of training data | 10.39 (+28%) |
| Mixed lists, retrained, all data | 11.02 (+35%) |

- Of the +0.0029 gain: 23% comes from the better candidates alone, 55% from retraining with the same amount of data, and 22% from the extra training examples a better retriever provides.
- **Test:** 6.44 → 7.86, +22% (+0.00142 [0.00115, 0.00170]).
- **Trade-off:** mixed lists help established products and hurt new ones (Test: established +0.0025, new −0.0018).

### Fix B: compute product statistics the same way as in training
- The rankers were trained with statistics computed at the start of their training period. The original pipeline scored 2022 purchases with statistics from Jul 2020.
- **The learned pattern:** in training, 43% of bought products had no reviews yet, against 20% of other candidates. The ranker learned "no reviews yet → probably bought".
  - With stale statistics, far too many products look new: 71% of bought vs 44% of others.
  - With very fresh monthly statistics, almost none do: 10% vs 1%.
- **Same frozen ranker on Test**, statistics computed at:
  - Jul 2020: 6.44;
  - Aug 2021: 7.38;
  - Jul 2022, the start of Test (the same timing as in training): 7.82.

  That is +21%: +0.00138 [0.00122, 0.00153].
- **Further evidence:**
  - Updating statistics only for products released after Jul 2020 recovers the whole gain (99% on Test).
  - Flipping only the "has any reviews" input cuts new products in the top 10 from 34% to 14%.
  - New products are ranked best with statistics about 6 months old (the age the ranker learned from); established products are best with fresh ones (figure 07, Test).
  - All 5 predictions written before opening Test held.
- **Training with monthly statistics** (consistent timing) lets the "bought together" (co-review) signal help: +0.00166 [0.00149, 0.00184] on Test.

## 7. Final result and who it helps (Test)

| System | Test nDCG@10 |
|---|---:|
| SASRec, single stage (best baseline) | 2.59 |
| Two-tower → MLP (the planned pipeline) | 5.19 |
| Two-tower → DIN | 6.37 |
| Mixed lists → MLP | 7.30 |
| Mixed lists → DIN | 8.27 (+59% vs planned, [0.00281, 0.00336]) |
| Mixed lists + monthly statistics + co-review → MLP-style ranker | 10.38 (×2 planned, ×4 SASRec; +61% vs recency→MLP 6.44, [0.00359, 0.00429]) |

- **Funnel:** 93.6 of 100 Test purchases never reach the ranker, even for the best system; 1.5 reach the top 10 (planned pipeline: 0.8).
- **Latency:** best DIN system 17.6 ms per user on one laptop CPU thread, with the faster combined score (Test 8.23 vs 8.27, no detectable loss). The planned pipeline takes 4.0 ms.
- **Who it helps** (Validation):
  - same phone, known store, fresh product: 21.8% reach the top 10 (was 3.0%);
  - an iPhone ↔ Android switcher at a new store: 0.0%.
- **Upgraders:** 37% of the recency retriever's candidates fit the old phone, 3% the new one.
- **Never run:**
  - popularity retrieval + a ranker (planned in the critique);
  - DIN combined with the statistics fix.
- **Limitations:**
  - Validation was reused to choose directions;
  - product titles are today's metadata, so some name later phones;
  - the 5-core filter uses the whole timeline;
  - new products, new stores and platform switchers remain unsolved.

## 8. Glossary of internal codes (in `evidence/`)

| Code | Meaning |
|---|---|
| T0 / T1 / T2 | 26 Jul 2020 / 11 Aug 2021 / 16 Jul 2022: the starts of ranker training, Validation and Test |
| P1 / P2 / P3 | Recency / two-tower / mixed candidate lists (200 per purchase). P2′ / P3′ use the fresher two-tower |
| B1 | Brandon's MLP-style ranker on recency lists (the "team pipeline": Test 6.44) |
| D1, D6, D7 | Rankers on mixed lists: D1 with T0 statistics; D6/D7 with monthly statistics; D7 adds co-review |
| S0–S8 | Systems in Brandon's locked Test run (`evidence/tables/test_final_20261107.md`); S8 = 10.38 |
| mlp_tt, din_tt, mlp_rrf, din_rrf | Jing Yen's final Test rankers on two-tower (tt) or mixed (rrf) lists |
| A / B / C / D, E–H | Jing Yen's training-list study. C = natural lists (used here). A, B and E–H involve forced lists: **not on the poster** |
| Forced lists, Shortcut 1 | Missed purchases appended to training lists. **Excluded from the poster by team decision** |

## 9. Who did what (confirm with the team before printing)

From the critique roles:
- **Qing Rong:** data pipeline, shared evaluation harness, baselines, failure and segment analysis.
- **Brandon:** two-tower retriever, FAISS search, recency and mixed retrievers, the statistics-timing investigation, the locked Test run S0–S8.
- **Jing Yen:** MLP and DIN rankers, ranker comparisons, the K-curve, DIN on mixed lists, the final ranker Test run, user and product cohort analysis, latency.
