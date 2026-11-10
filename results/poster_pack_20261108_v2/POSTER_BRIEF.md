# Poster brief v2: CP4285 Group 7 STePS poster (for Claude with the Canva connector)

**Team:** Brandon Kang, Phua Qing Rong, Tong Jing Yen · CP4285 · National University of Singapore
**Data:** Amazon Reviews 2023, Cell Phones & Accessories · 381k users · 111k products · 2.75M reviews · split by time

This replaces the 8 Nov brief (`poster_pack_20261108`). The story is now simpler, and it no longer covers how training lists were built from missed purchases ("forced lists" / "Shortcut 1"). The team agreed to leave that out.

## 0. Instructions for Claude

0. **Read `PROJECT_OVERVIEW.md` first.** It explains the whole project in the order of this story, with every result and a glossary of the internal codes used in `evidence/`. Raw result tables are in `evidence/tables/`. The longer files in `evidence/` are working records with an older framing; use them only to look up a number.
1. **Format.**
   - One **A1 portrait** poster (594 × 841 mm).
   - Before starting, ask the user to confirm:
     - the size;
     - whether to start a new design or update the existing Canva draft;
     - the supervisor's name for the byline.
2. **Assets.**
   - The user uploads `figures/` and `photos/` to Canva Uploads, in a folder named **"CP4285 poster v2"**.
   - Use those files; don't redraw charts that exist as files.
   - Build the funnel, small tables and text boxes natively in Canva.
3. **Numbers.**
   - Use only the numbers and wording in this brief.
   - `evidence/numbers_and_sources_v2.md` gives the source of every number.
   - nDCG@10 values are written ×10⁻³: "8.27" means 0.00827.
4. **Do not mention:**
   - forced lists, appending missed purchases, or Shortcut 1;
   - the score–logit correlation;
   - the dose-response chart.
   The older evidence files still describe them; ignore those parts.
5. **The story in one line.**
   - We set out to compare retrievers and rankers (RQ1–RQ3).
   - Answering RQ2 showed something surprising: **giving a trained ranker a better candidate list did not improve its top 10.**
   - The rest of the poster explains why, shows the fix, and shows the final result.
   - Every section should lead into the next one. The orange arrow text in §2 is the connective tissue; keep it.
6. **Style.**
   - **Visual references:** the two prize-winning posters in `reference/`.
     - Use from *CARA*: bold boxed section headers, one hero figure, three rounded takeaway boxes.
     - Use from *DataTales*: numbered sections on a clean grid.
     - Avoid CARA's busy background and DataTales' dense paragraphs.
   - **Colour:**
     - background `#fcfcfb`, text `#0b0b0b`;
     - accent orange `#d9730d`, dark orange `#a35400`, light orange `#eba154`;
     - gray `#5f5e5a` and `#898781`;
     - **no blue**.
   - **Type:**
     - one sans-serif family;
     - title ≥ 90 pt;
     - section headers ≥ 44 pt;
     - body ≥ 26 pt;
     - captions ≥ 18 pt.
7. **Text budget.**
   - **About 550 words in total.**
   - Every section: one headline sentence that states the finding, then at most 3 short bullets.
   - A visitor should get the story from the headlines alone.

## 1. Header (about 9% of height)

- **Title:** Better Candidates Aren't Enough
- **Subtitle:** How retrieval and ranking choices affect product recommendations (our original project title).
- **Byline:** Brandon Kang · Phua Qing Rong · Tong Jing Yen · CP4285 Group 7 · NUS · Supervisor: [confirm with user]
- **Takeaway banner** (light orange box):

  > *A ranker only knows what it was trained on. Giving it a better candidate list, or fresher data, did not improve its top 10. Training it on the lists and data it will actually see doubled our planned pipeline on held-out test data.*

## 2. Sections, top to bottom

### Section 1 · What we set out to do (about 13%)

**Left: funnel.** Build it in Canva as three stacked trapezoids:

| Tier | Label |
|---|---|
| 111,480 products | the whole catalogue |
| **Retriever** → 200 candidates | fast, picks a shortlist |
| **Ranker** → 10 shown | slower, orders the shortlist |

**Right: our three questions** (from the critique, 29 Sep), as three numbered chips:

- **RQ1** Does a trained two-tower retriever find more purchases than standard baselines?
- **RQ2** How do the number of candidates and the search method affect the final top 10?
- **RQ3** Which ranker orders the candidates best?

**Protocol line (small):**
- train on the past, choose on Validation (Aug 2021 – Jul 2022), test once on Jul 2022 onwards (297,390 purchases);
- metric: nDCG@10, how high the bought product appears in the top 10;
- 3 seeds; 95% intervals from resampling users.

### Section 2 · What we found for each question (about 20%). Three equal columns.

**RQ1 · Retriever**
- **Headline:** Yes. The two-tower finds the most purchases, and mixing it with a simple recency retriever finds even more.
- **Table** (Validation, share of purchases among the 200 candidates):

  | Retriever | Found in top 200 |
  |---|---:|
  | Popularity | 2.7% |
  | ItemKNN | 0.8% |
  | BPR-MF | 2.7% |
  | SASRec | 4.0% |
  | **Two-tower** | **6.9%** |
  | Recency (text of your recent purchases, no training) | 5.5% |
  | **Mixed: both, alternating** | **8.0%** |

- **Bullet:** The two retrievers share only 12 of 200 products. Optional small figure: `fig16_two_retrievers_orange.png`.

**RQ2 · Candidates and search**
- **Headline:** More candidates find more purchases, but the top 10 gets *worse* beyond 200.
- **Table** (Validation; two-tower retriever; trained ranker):

  | Candidates | Purchases in the list | Top-10 quality (nDCG@10) |
  |---:|---:|---:|
  | 50 | 3.0% | 8.33 |
  | 100 | 4.6% | **8.61** |
  | 200 | 6.9% | 8.57 |
  | 500 | **11.8%** | 7.92 (−8%) |

- **Bullet:** Search method: HNSW keeps 99.4% of the exact results at half the time (1.5 vs 3.4 ms). IVF loses 5% of them, and the ranker can't recover those.

**RQ3 · Ranker**
- **Headline:** DIN beats the MLP on the same lists, but changing the lists mattered more than changing the ranker.
- **Table** (held-out Test, nDCG@10; – = not run):

  | Ordered by | Recency lists | Two-tower lists | Mixed lists |
  |---|---:|---:|---:|
  | Popularity | 1.51 | – | – |
  | Retriever's own order | 6.04 | – | – |
  | MLP ranker | 6.44 | 5.19 | 7.30 |
  | **DIN ranker** | – | 6.37 | **8.27** |

- **Bullet:** DIN vs MLP on the same lists: +13% to +23%. Same DIN on mixed instead of two-tower lists: +30%.

**Orange arrow / connector under Section 2** (full width, large italic):

> *RQ2 was a surprise: 71% more purchases among the candidates, yet a worse top 10. So we asked: what happens when a trained ranker is given a better list?*

### Section 3 (hero) · A better list did not give a better top 10 (about 22%)

- **Headline:** Three times we gave a trained ranker a better candidate list without retraining it, and three times its top 10 barely moved.
- **Figure (large):** `N1_better_list_same_ranker.png`.
- **Bullets:**
  - **500 instead of 200 candidates:** 71% more purchases in the list, but −8% in the top 10. The newly found purchases add +1.3, but the 300 extra candidates push down purchases the ranker already had right (−2.0).
  - **A fresher retriever** (retrained up to Aug 2021): 62% more purchases in the list, no gain in the top 10 (10.08 → 9.93), and −24% on new products.
  - **Mixed lists:** 45% more purchases in the list, only +8% in the top 10.
- **"Why" box** (light orange, short):

  > *A ranker learns patterns in the data it was trained on. When the lists or data it is given look different, those patterns stop fitting.*

- **Status line (small):** Validation, 232,904 purchases; ranker values are the mean of 3 seeds.

### Section 4 · The fix: train the ranker on what it will see (about 18%). Two cards side by side.

**Card A · Retrain on the new lists**
- **Headline:** Retrained on mixed lists, the ranker finally gains: +35% on Validation, +22% on Test.
- **Figure:** `N2_retrain_on_new_lists.png`.
- **Bullet:** With the same amount of training data, the retrained ranker still beats the old one by +28%. Retraining, not just extra data, is the main source of the gain.

**Card B · Give it data computed the same way as in training**
- **Headline:** The same trained ranker scored +21% on Test when its product statistics were computed at the same point as in training.
- **Figure:** `N3_statistics_timing_test.png`. Optional second figure: `07_statistics_age_curve.png`.
- **Bullets:**
  - **The pattern it learned:** in training, 43% of bought products had no reviews yet, against 20% of other candidates. So it learned "no reviews yet → probably bought".
  - **What went wrong:** the original pipeline used 2020 statistics on 2022 purchases, so far too many products looked new. Even *fresher* monthly statistics hurt new products, because the ranker learned on statistics about 6 months old.
  - **Checked on Test:** all 5 predictions written before opening Test held.

### Section 5 · Final result: our plan's last question (about 12%). Two columns.

**Left: `N4_final_test_ladder.png`** (key).
- **Headline:** On held-out Test, training on what the ranker will see doubles our planned pipeline and quadruples the best single-stage baseline (SASRec).

**Right:** `fig15_funnel_flow_orange.png` (small), plus two bullets:
- **Where misses happen:** 93.6 of 100 Test purchases never reach the ranker, even for the best system. Most failures happen at retrieval.
- **Cost:** the best DIN system takes 17.6 ms per user on one laptop CPU thread, against 4.0 ms for the planned pipeline. That is well inside a real-time budget.

### Section 6 · Who it helps (about 9%)

- **Headline:** It helps people buy for a phone they already own, from a store they know, and not much else.
- **Figure:** `fig17_user_x_product_orange.png` (key):
  - same phone, known store, fresh product: 21.8% reach the top 10 (was 3.0%);
  - an iPhone ↔ Android switcher at a new store: 0.0%.
- **Bullet:** Upgraders get accessories for their *old* phone: 37% of the recency retriever's candidates fit the old phone, 3% the new one.

### Footer (about 7%)

**Three takeaway boxes** (rounded, orange outline):
1. **Check the retriever first.** Over 90% of misses happen before ranking.
2. **New retriever? Retrain the ranker.** Better candidates alone don't help.
3. **Compute inputs the same way in training and in use.** Even "fresher" data can break what the ranker learned.

**Limitations (small, right):**
- The 500-candidate and fresher-retriever tests are Validation-only. The mixed-list and statistics results are confirmed on Test.
- The best system (10.38) uses an MLP-style ranker. DIN with the statistics fix was not run.
- Product titles are today's metadata, so some name phones released after the purchase.
- New products, new stores and platform switchers remain unsolved.

## 3. Figure priority if space is tight

1. **Must keep:** `N1_better_list_same_ranker`, `N2_retrain_on_new_lists`, `N3_statistics_timing_test`, `N4_final_test_ladder`.
2. **Keep if possible:** `fig17_user_x_product_orange`, `fig15_funnel_flow_orange`.
3. **Optional:** `07_statistics_age_curve`, `fig16_two_retrievers_orange`, `fig23_accuracy_vs_speed_orange`, `fig21_upgrade_list_orange`, buyer photos in `photos/` (good / bad / ugly cards from the earlier draft).

## 4. Figure list (`figures/`, 300-dpi PNG with SVG twins where available)

| File | Section | Caption |
|---|---|---|
| `N1_better_list_same_ranker.png` | 3 (hero) | More purchases in the candidate list (gray) did not give a better top 10 from the same trained ranker (orange) |
| `N2_retrain_on_new_lists.png` | 4A | The gain arrives only when the ranker learns on the new lists |
| `N3_statistics_timing_test.png` | 4B | Same trained ranker; statistics computed at the same point as in training: +21% on Test |
| `07_statistics_age_curve.png` | 4B (optional) | New products are ranked best with statistics as old as in training |
| `N4_final_test_ladder.png` | 5 | Final Test comparison against the plan and SASRec |
| `fig15_funnel_flow_orange.png` | 5 | Where 100 Test purchases go |
| `fig23_accuracy_vs_speed_orange.png` | 5 (optional) | Accuracy vs latency per user |
| `fig17_user_x_product_orange.png` | 6 | Sweet spot: fresh product, known store, a phone you own |
| `fig21_upgrade_list_orange.png` | 6 (optional) | Upgraders get accessories for the old phone |
| `fig16_two_retrievers_orange.png` | 2 (optional) | The two retrievers find different products |

## 5. Numbers that must not be mixed up

- **Section 3 uses Validation; Sections 2 (RQ3) and 5 use Test.** Don't put Validation and Test values in the same table.
- **The "planned pipeline"** is two-tower → MLP (Test 5.19). The recency → MLP pipeline (Test 6.44) is the starting point of the statistics study and of Section 4A on Test (6.44 → 7.86, +22%).
- **10.38 uses an MLP-style ranker, not DIN.** Don't write "DIN + all fixes".
- **SASRec (2.59)** is on the same Test purchases and harness, but has no paired interval. Write "×4", not a CI.
- **The 500-candidate result:** −8% is vs the 200-candidate ranker (8.57 → 7.92). The best count was 100 (8.61), statistically tied with 200.
- **The fresher-retriever result** is "no gain" overall (interval −0.37 to +0.08 ×10⁻³). It is significantly worse only for new products and older established products. Never write "worse overall".
- **Popularity retrieval + a ranker** (planned in the critique) was not run. Don't claim a result for it.

## 6. A 60-second walkthrough for presenting (for the team, not printed)

1. "Shops pick 10 products out of 111,000 in two steps: a fast retriever makes a shortlist of 200, and a ranker orders it. We asked which retriever, how many candidates, and which ranker." *(Section 1)*
2. "Our two-tower retriever beats the baselines, and DIN beats the MLP. But something odd happened: with 500 candidates, the list held 71% more purchases, and the top 10 got *worse*." *(Section 2)*
3. "So we tested it three ways. A better list, without retraining the ranker, barely changed the top 10." *(Section 3)*
4. "The ranker learns patterns in its own training data. Retrain it on the new lists, or give it data computed the same way as in training, and the gains appear: +22% and +21% on Test." *(Section 4)*
5. "Together that doubles our planned pipeline on held-out Test, and it costs 17.6 ms per user. It works best for people buying for a phone they already own." *(Sections 5–6)*
