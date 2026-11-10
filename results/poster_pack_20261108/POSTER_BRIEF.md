# Poster brief: CP4285 Group 7 STePS poster (for Claude with the Canva connector)

**Team:** Brandon Kang, Phua Qing Rong, Tong Jing Yen · CP4285 · National University of Singapore · Supervisor: Dr Kan Min-Yen
**Data:** Amazon Reviews 2023, Cell Phones & Accessories · 381k users · 111k products · 2.75M reviews · split by time

## 0. Instructions for Claude

1. **Format.** One **A1 portrait** poster (594 × 841 mm). Before you start, ask the user to confirm the size and whether to start a new design or update an existing Canva draft.
2. **Assets.**
   - The user uploads `figures/` and `photos/` to Canva Uploads, in a folder named **"CP4285 poster 8 Nov"**. Find them there; don't recreate charts that exist as files.
   - Build tables, the funnel, the report card and text boxes natively in Canva.
3. **Numbers.**
   - Use only the numbers and wording in this brief.
   - `evidence/` has the full detail for a short clarification; never invent or round differently.
   - nDCG@10 values are written ×10⁻³: "8.27" means 0.00827.
4. **Reading order.** Six bands, top to bottom (§2). The two halves of the story:
   - **Bands 1–2 answer the questions we proposed** in the critique slides (`reference/critique_slides_29sep.pdf`, pages 4–7).
   - **Band 3 is the in-depth mechanism.** Give it the most space.
5. **Style.**
   - **Visual references:** the two prize-winning posters in `reference/`.
     - *CARA* (`sample_poster_cara.png`): bold boxed section headers, one large hero diagram, a short results table, three rounded "insight" boxes at the end.
     - *DataTales* (`sample_poster_datatales.png`): numbered sections on a clean grid.
   - **Reuse from these:**
     - numbered section headers;
     - one hero visual per band;
     - short tables;
     - rounded takeaway boxes;
     - generous white space.
   - **Avoid** CARA's busy background pattern and the dense paragraphs in DataTales.
   - **Colour:**
     - background `#fcfcfb`, text `#0b0b0b`;
     - accent orange `#d9730d`, dark orange `#a35400`, light orange `#eba154`;
     - comparison dark gray `#5f5e5a`, context gray `#898781`;
     - **no blue**. The figures already use this scheme.
   - **Type:**
     - one sans-serif family;
     - title ≥ 90 pt;
     - section headers ≥ 44 pt;
     - body ≥ 24 pt;
     - captions ≥ 18 pt.
6. **Text budget.**
   - About 700 words in total.
   - Each panel: a one-sentence headline that states the finding, then at most 3 bullets.
   - Readers stand 1–2 m away.
7. **Layout reference.** `reference/jingyen_draft3_preview.jpg` is the team's long web draft. It has all the content but is about five A1 posters long. This brief is the cut-down version; where they differ, follow the brief.

## 1. Header band (about 8% of height)

- **Title:** It's Not the Ranker, It's the Data
- **Subtitle:** How retrieval and ranking choices affect product recommendations. This is our original project title from the critique, so keep it.
- **Byline:** Brandon Kang · Phua Qing Rong · Tong Jing Yen · CP4285 Group 7 · NUS · Supervisor: Dr Kan Min-Yen
- **Takeaway banner** (light orange box under the title): *A two-stage recommender's quality was decided by what the ranker was fed (which retrievers built its lists, how its training lists were made, how old its product statistics were) far more than by the ranker model. Fixing these doubled the planned pipeline's held-out Test score.*

## 2. Bands

### Band 1: What we set out to do (about 13%)

**Left third: task funnel (hero shape, build in Canva).** Three stacked trapezoids:

| Tier | Label |
|---|---|
| 111,480 products | the whole catalogue |
| 200 candidates | the purchase is here for 6.1% of Test purchases |
| 10 shown | the purchase is here for 1.3% |

- Side labels: "Experiment A · retriever (RQ1, RQ2)" next to the 200 tier, and "Experiment B · ranker (RQ3)" next to the 10 tier.
- Caption: *Test, team pipeline (recency retriever → MLP ranker).*
- Protocol line (small):
  - time split: train → Aug 2021 → Validation → Jul 2022 → Test;
  - 3 seeds; paired user bootstrap;
  - every choice made on Validation; Test opened once, with predictions written first.

**Right two-thirds: "Report card": the questions from our critique slides, answered.** A 3-column table: Question | Answer | Panel.

| Question (critique, 29 Sep) | Answer | Panel |
|---|---|---|
| **RQ1** Does our two-tower retriever beat popularity, BPR-MF, ItemKNN, SASRec at a fixed K? | **Yes.** Val Recall@200 0.069 vs SASRec 0.040. Mixing it with a recency retriever beats both (0.080). | 2 |
| **RQ2** How do K and Flat / HNSW / IVF search affect Recall@K, nDCG@10, latency? | Recall rises with K, but quality peaks at K ≈ 100–200. HNSW is free speed; IVF drops purchases the ranker can't recover. | 3 |
| **RQ3** On identical lists: popularity order vs retrieval order vs MLP vs DIN? | Popularity < retrieval < MLP < DIN on the final lists, but **the training lists decide which ranker wins.** | 4 |
| **Output** Where do misses arise? | Mostly before ranking: 94% of Test purchases never reach the ranker. | 6 |
| **Final** Does the selected pipeline beat the best single-stage baseline? | **Yes, ×4 SASRec** on Test (10.38 vs 2.59). The planned pipeline alone is ×2. | 6 |

### Band 2: Experiments A and B (about 22%). Three equal columns.

**Panel 2 · RQ1 · Experiment A: retriever design**
- **Headline:** The trained two-tower beats every baseline, and mixing it with a recency retriever beats both.
- **Table** (Validation, retriever only, no ranker):

  | Retriever | Recall@200 | nDCG@10 |
  |---|---:|---:|
  | Popularity | 0.027 | 1.7 |
  | ItemKNN | 0.008 | 0.6 |
  | BPR-MF | 0.027 | 1.7 |
  | SASRec | 0.040 | 3.1 |
  | **Two-tower (text)** | **0.069** | 5.7 |
  | Recency (no training: text of your recent purchases) | 0.055 | 6.3 |
  | **Mixed: recency + two-tower, alternating** | **0.080** | **7.4** |

- **Figure:** `fig16_two_retrievers_orange.png` (small).
- **Bullet:** The two retrievers share only 12 of 200 products: two-tower finds popular ones, recency finds new and niche ones. Only text-based models find new products at all.

**Panel 3 · RQ2 · Experiment A: candidate count and search**
- **Headline:** Recall keeps rising with K, but final quality peaks at K ≈ 100–200.
- **Figure:** `fig25_k_curve_orange.png`.
- **Small table** (K = 200, 1,000 Validation queries, single CPU thread):

  | Search | Keeps of exact top 200 | nDCG@10 | Median latency |
  |---|---:|---:|---:|
  | Flat (exact) | 100% | 7.37 | 3.4 ms |
  | **HNSW** | 99.4% | 7.37 | **1.5 ms** |
  | IVF | 94.6% | 6.75 | 0.9 ms |

- **Bullets:**
  - Recall nearly doubles from K = 200 to 500, but the extra candidates are mostly distractors: −7% for the natural-list ranker. The original forced-list ranker collapses by −72% (Shortcut 1 explains why).
  - HNSW is free speed. IVF searches only 16 of its 256 clusters, so 5% of the exact top 200 disappear, some of them purchases, and the ranker can't recover them.

**Panel 4 · RQ3 · Experiment B: ranker choice**
- **Headline:** DIN beats the MLP, but what the ranker is fed matters more than its architecture or its loss.
- **Table** (Test nDCG@10 ×10⁻³; – = not run):

  | Ranker | Recency lists | Two-tower lists | Mixed lists |
  |---|---:|---:|---:|
  | Popularity order | 1.51 | – | – |
  | Retrieval-score order | 6.04 | – | – |
  | MLP | 6.53 | 5.19 | 7.30 |
  | **DIN** | – | 6.37 | **8.27** |

- **Bullets:**
  - DIN vs MLP on identical mixed lists: +13% on Test (95% CI +0.0008 to +0.0011), winning on popular, niche and new products.
  - **But:** trained on forced lists, the MLP beat DIN (6.84 vs 6.44, Val). The training lists flipped the verdict.
  - Changing the loss (BCE → pairwise) moved Test by at most 0.0005. Changing what the ranker is fed moved it by 21–35% (Band 3).

### Band 3 (hero, about 32%): Why. The ranker learns whatever its inputs make easy

- **Band header:** "What we found while answering the RQs: three shortcuts". Give it a dark band or orange rule so it reads as the poster's centre.
- **Layout:** row A has two cards side by side (Shortcuts 1 and 2). Row B is one full-width card (Shortcut 3, the centrepiece) laid out as a five-step strip.
- Each card has a small tag naming the RQ it explains.

**Card S1 · explains RQ2 and RQ3: forcing missed purchases into training lists taught "the lowest score is the answer"**
- **Figures:** `fig5_shortcut.png` (key), `S1_forced_dose_response.png`.
- **Bullets:**
  - **Mechanism:** in 88% of training lists the retriever missed the purchase, so it was appended, and it scored below every real candidate in 100% of them. The ranker learned to reward low retrieval scores (score–logit correlation −0.40; natural-list rankers +0.19).
  - **Intervention:** train only on lists where retrieval found the purchase: +24% at K = 200 (3 seeds), no collapse at K = 500, and DIN then beats the MLP. Dose-response: every 10–30% of forced lists added back deepens the K = 500 collapse (7.9 → 5.5 → 3.9 → 2.3).
  - **Result:** two earlier project conclusions, "more candidates hurt" and "MLP beats DIN", were artefacts of this shortcut.
- **Status line (small):** Validation, 3 seeds, replicated on two candidate sets (laptop and cluster).

**Card S2 · explains RQ1 and RQ3: the ranker learns the habits of the retriever that feeds it**
- **Figure:** `04_mixed_pool_gain_explained.png`.
- **Bullets:**
  - **Mixed lists:** +35% on Validation and +22% on Test (same ranker recipe).
  - **Size-matched control:** better candidates alone give only 23% of the gain; 55% needs the ranker retrained on the mixed lists with the same amount of data.
  - **Counter-test:** a fresher retriever under a frozen ranker gave +62% Recall@200 but only +0.0004 nDCG@10. **Retrain the ranker whenever the retriever changes.**

**Card S3 (centrepiece) · explains RQ3: "no reviews yet = likely purchase" only works with statistics as old as in training**
- **Layout:** five steps left to right, joined by arrows. Each step has a one-word header, one line and its figure where listed.

| Step | Header | Line | Figure |
|---|---|---|---|
| 1 | **Observe** | Recomputing product statistics at the period start: **+21% on Test** for the *same frozen ranker*. Fresher monthly statistics were worse for new products. | none (big number "+21%") |
| 2 | **Explain** | In training, 43% of bought products had no reviews yet, vs 20% of other candidates. The ranker learned "no reviews → new → likely bought". Year-old statistics make it fire too often (71% vs 44%); monthly ones erase it (10% vs 1%). | `05_learned_rule_and_statistics_timing.png` |
| 3 | **Intervene** | Flip only "has any reviews" in 10,000 lists: new products in the top 10 drop from 34% to 14%. Update statistics only for products released after T0: **recovers the whole gain** (99% on Test, 101% on Validation); updating older products recovers 11–14%. | `06_where_the_stale_statistics_gain_comes_from.png` |
| 4 | **Predict** | If the rule needs training-aged statistics, new products should peak at that age, and they do: best at ~6 months; established products best when up to date. | `07_statistics_age_curve.png` |
| 5 | **Confirm** | All 5 written predictions held on the untouched Test period. Training *with* monthly statistics keeps the rule consistent, and "bought together" inputs flip from useless to +0.0017 on Test. | `09_test_confirms_val.png` (optional) |

**Footer line under Band 3:** *Every shortcut is the same lesson: the ranker learns whatever its inputs make easy.*

### Band 4: Output. Where it fails, what it costs, who it helps (about 18%). Three columns.

**Panel 5 · Misses and final test**
- **Figure:** `fig15_funnel_flow_orange.png`. Caption: *Out of 100 Test purchases, 93.6 never reach the ranker, even for the best system.*
- **Figure:** `F1_final_test_ladder.png` (key). This is the planned "final combination" comparison.
- **Bullet:** popularity order on the same lists scores 1.51. Popularity retrieval finds only 1.6% of Test purchases in its top 200 (mixed lists: 6.4%), and none of the 44% that are new products.

**Panel 6 · Cost**
- **Figure:** `fig23_accuracy_vs_speed_orange.png`.
- **Bullet:** best DIN system: median 17.6 ms per user on one laptop CPU thread, with the faster combined score (Test 8.23 vs 8.27, no detectable loss), against 4.0 ms for the planned pipeline. Well inside a real-time budget.

**Panel 7 · Who it works for (meso and micro)**
- **Headline:** The system recommends more of what you already own.
- **Figure:** `fig17_user_x_product_orange.png` (key). Fresh product, known store, a phone you own: 21.8% reach the top 10 (was 3.0%); a platform switcher at a new store: 0.0%.
- **Bullets:**
  - **Upgraders:** 37% of recency candidates are for the old phone, 3% for the new one (`fig21_upgrade_list_orange.png`, optional).
  - **Same-store purchases** (9%): top-10 rate 6.6% → 14.7%. The other 91% gain nothing.
- **Optional, if space allows:** a strip of three mini cards from the draft: "good / bad / ugly" buyers with the phone photos in `photos/phones/` and the product photos in `photos/products/`. Good: same UAG case, iPhone 13 → 13 Pro, ranked #1. Bad: iPhone 6s Plus → XS Max, never reached. Ugly: Galaxy S20 → iPhone 11, never reached. Small credit line: "Product images: Amazon; phone images: Apple and retailer sites."

### Band 5: Footer (about 7%)

**Takeaways:** four rounded orange-outline boxes in a row (CARA style):
1. **Look before you rank.** 94% of misses happen at retrieval. Check where failures arise first.
2. **Train on lists built like the ones you'll rank.** Forced positives teach a shortcut.
3. **Retrain the ranker whenever the retriever changes.** It learns its retriever's habits.
4. **Compute statistics the same way in training and in use.** A timing mismatch breaks learned rules silently.

**Limitations (small, right side):**
- Product titles are today's metadata: sellers add newer phone names later (770 purchases made before the iPhone 13 announcement mention it). Optional figure: `fig22_future_titles_orange.png`.
- Validation was reused to choose directions; the main effects are confirmed on Test with pre-registered predictions. Shortcut 1 is Validation-only.
- New products, new stores and platform switchers remain unsolved.

## 3. Figure priority if space is tight

1. **Must keep:** `F1_final_test_ladder`, `fig25_k_curve_orange`, `fig5_shortcut`, `04_mixed_pool_gain_explained`, `05_learned_rule_and_statistics_timing`, `06_where_the_stale_statistics_gain_comes_from`, `07_statistics_age_curve`, `fig15_funnel_flow_orange`.
2. **Keep if possible:** `S1_forced_dose_response`, `fig17_user_x_product_orange`, `fig23_accuracy_vs_speed_orange`, `fig16_two_retrievers_orange`.
3. **Optional:** `09_test_confirms_val`, `02_what_moves_the_needle`, `fig21_upgrade_list_orange`, `fig22_future_titles_orange`, `fig19_waffles_orange`, `fig24_slot_flow_orange`, `fig12_iphone13_launch_orange`, buyer photos.

## 4. Figure list (`figures/`, 300-dpi PNG with SVG twins where available)

| File | Band / panel | Caption |
|---|---|---|
| `fig16_two_retrievers_orange.png` | 2 (RQ1) | The two retrievers find different products: two-tower popular, recency new and niche |
| `fig25_k_curve_orange.png` | 2 (RQ2) | Recall rises with K; quality peaks at K ≈ 100–200; the forced-list ranker collapses |
| `fig5_shortcut.png` | 3 (S1) | In forced lists the purchase sits below every real candidate |
| `S1_forced_dose_response.png` | 3 (S1) | More forced lists, deeper collapse at 500 candidates |
| `04_mixed_pool_gain_explained.png` | 3 (S2) | Most of the mixed-list gain needs the ranker retrained on mixed lists |
| `05_learned_rule_and_statistics_timing.png` | 3 (S3, Explain) | The "no reviews yet" rule only works with statistics as old as in training |
| `06_where_the_stale_statistics_gain_comes_from.png` | 3 (S3, Intervene) | Updating only products released after T0 recovers the whole gain |
| `07_statistics_age_curve.png` | 3 (S3, Predict) | New products are best with ~6-month-old statistics; established ones with fresh ones |
| `09_test_confirms_val.png` | 3 (S3, Confirm) | The locked Test run confirms each main effect |
| `fig15_funnel_flow_orange.png` | 4 | Where 100 Test purchases go: most never reach the ranker |
| `F1_final_test_ladder.png` | 4 | Final test: the planned pipeline doubles SASRec; the fixes double it again |
| `fig23_accuracy_vs_speed_orange.png` | 4 | Accuracy vs latency per user |
| `fig17_user_x_product_orange.png` | 4 | Sweet spot: fresh product, known store, phone you own |
| `fig21_upgrade_list_orange.png` | 4 (optional) | Upgraders get accessories for the old phone |
| `fig19_waffles_orange.png` | 4 (optional) | Store loyalty carries the gain |
| `fig24_slot_flow_orange.png` | 4 (optional) | The ranker moves slots toward the right phone but cannot add missing products |
| `fig12_iphone13_launch_orange.png` | 4 (optional) | The best system's edge spikes at the iPhone 13 launch, then fades |
| `fig22_future_titles_orange.png` | 5 (optional) | Titles name phones released after the purchase |
| `02_what_moves_the_needle.png` | optional | Changing the ranker vs changing what it is fed (Test, % change) |

## 5. Numbers that must not be mixed up

- **Two different "originals" exist.**
  - The *planned pipeline* is two-tower → MLP (Test 5.19).
  - The *team pipeline* is recency → MLP (Test 6.44).
  - Our best combined system (10.38) is +61% over the team pipeline and ×2.0 over the planned pipeline.
  - The DIN system (8.27) is +59% over the planned pipeline.
  - On the poster, use the ladder (`F1_final_test_ladder`) rather than quoting both percentages.
- **10.38 uses an MLP-style ranker, not DIN.** "DIN + statistics fix" was never run, so don't imply it was.
- **The SASRec comparison** (2.59) is on the same 297,390 Test purchases and harness, but has no paired confidence interval.
- **The planned "popularity retrieval + ranker" cascade was not run.** Don't claim a result for it; use the popularity retrieval coverage bullet in Panel 5 instead.
