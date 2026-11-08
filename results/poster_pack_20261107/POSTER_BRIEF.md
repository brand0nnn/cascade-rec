# Poster brief: CP4285 Group 7 (for Claude, to update our existing Canva draft)

**Team:** Brandon Kang, Tong Jing Yen, Phua Qing Rong. **Course:** CP4285.

**Data:** Amazon Reviews 2023, Cell Phones & Accessories. 381k users, 111k products, 2.75M reviews. Split by time.

## Instructions for Claude

1. **Update the existing draft. Do not create a new design.** Keep the draft's size, style, title block, logos and any sections not covered here, unless a section below replaces them.
2. **Replace the main content with the five acts below.** Map each act to a panel of the draft in reading order: left to right, then top to bottom. If the draft has fewer panels, merge Act 1 into the introduction and Act 5 into the conclusion.
3. **Use only the numbers and wording in this brief.** `research_story_full_evidence.md` has the full detail if a short clarification is needed. Do not invent numbers.
4. **Figures.**
   - Files are in `figures/`, as 300-dpi PNG with SVG twins. I will have uploaded them to Canva Uploads (folder "CP4285 poster").
   - Place each where its act says.
   - **Priority if space is tight:**
     1. 02, 05, 07, J1 (must keep)
     2. 04, 09, J2
     3. 01, 10, J4
     4. drop first: 03, 06, 08, J3
5. **Readability.** Keep text short, readable from 1–2 m. Each act has a headline (the finding) and at most 3 bullets.
6. **Visual consistency.** The poster uses an orange (Amazon) colour scheme with no blue.
   - Figures 02, 04, 05 and 07 use:
     - orange `#d9730d` for the highlight (what the ranker is given, purchased products, new products);
     - dark gray `#5f5e5a` for the comparison series (established products);
     - light gray `#898781` for context;
     - the ramp `#eba154` → `#d9730d` → `#a35400` for stepped bars.
   - Other figures use blue highlights; prefer the four orange figures.
   - Use orange for poster accents, with background `#fcfcfb` and text `#0b0b0b`. Sans-serif font throughout.

## Title and takeaway

**Title:** What the ranker learns from its training data: three hidden shortcuts in a two-stage recommender

**Takeaway (under the title):** How the ranker's training data is built (how missed products are added, which retriever made the lists, and how old the statistics are) mattered far more than the ranking model. Fixing it improved recommendations by 61% on held-out data.

## Setup box (small, near the start)

- A **retriever** picks 200 candidate products per user; a **ranker** reorders them and shows the top 10.
- **Retrievers:** recency (text similarity to recent purchases), two-tower (trained neural model), and a mixed list alternating both.
- **Rankers:** small neural rankers (MLP; DIN for comparison), trained on past candidate lists.
- **Time split:** retriever training < T0 (Jul 2020) < ranker training < T1 (Aug 2021) < Validation < T2 (Jul 2022) < Test.
- **Metric:** nDCG@10. Test was run once, with predictions written down in advance.

## Act 1: Most failures happen before ranking

- **Headline:** The ranker never sees the right product 94% of the time.
- **Bullets:**
  - Only 6.1% of Test purchases are among the 200 candidates; 1.3% reach the top 10.
  - 88% of Validation purchases are missed by both retrievers.
  - Case study: users move to a new phone; retrievers keep suggesting accessories for the old one.
- **Figure:** `01_where_failures_happen.png`.
- **Transition line:** *The ranker can only work with what it is given, and with what its training data taught it.*

## Act 2: Shortcut 1, how missed products entered the training lists (Validation)

- **Headline:** Forcing missed purchases into training lists taught the ranker that "the lowest score is the answer."
- **Bullets:**
  - In 88% of training lists the purchase was added at the bottom, below every real candidate.
  - Training only on natural lists: +24%. The collapse at 500 candidates disappears (−72% → nearly flat), and DIN overtakes the MLP.
  - Two earlier project conclusions ("more candidates hurt", "MLP beats DIN") were artefacts of this.
- **Figures:** `J1_forced_list_shortcut.png` (key), `J2_k_curve_forced_vs_natural.png`. Optional: `J3_training_lists_decide_mlp_vs_din.png`.
- **Credit line (small):** *Found 28–29 Oct; replicated on the cluster by Jing Yen, 3 seeds, two candidate sets.*
- **Transition line:** *If how the lists are built matters this much, what else is the ranker learning from its training data?*

## Act 3: Shortcut 2, which retriever produced the lists

- **Headline:** The ranker learns the habits of the retriever that feeds it.
- **Bullets:**
  - The two retrievers share only ~12 of 200 candidates. Mixing their lists: +35% (Validation), +22% (Test).
  - Size-matched control: 55% of the gain needs the ranker retrained on the mixed lists; better candidates alone give 23%.
  - So when the retriever changes, the ranker must be retrained.
- **Figures:** `04_mixed_pool_gain_explained.png` (key). Optional: `03_two_retrievers_find_different_products.png`.

## Act 4 (centrepiece): Shortcut 3, when the product statistics were calculated

- **Headline:** The ranker learned "no reviews yet = likely purchase", a rule that only works with statistics as old as in training.
- **Bullets:**
  - **The rule.** In training, 43% of purchased products had no reviews yet, against 20% of other candidates.
  - **What broke it.** The original setup scored with statistics a year older than in training. Recomputing them at the period start: +16–30% (Validation), +21% (Test), same frozen model.
  - **The evidence.**
    - Updating statistics only for recently released products recovers the whole gain.
    - New-product quality peaks when statistics are as old as in training; established products want fresh ones.
    - All five predictions held on Validation and again on Test.
- **Figures:** `05_learned_rule_and_statistics_timing.png`, `07_statistics_age_curve.png` (key). Optional: `06_where_the_stale_statistics_gain_comes_from.png`.
- **Small add-on:** consistent monthly statistics turn the "bought together" inputs from useless (+0.0002) into the largest input gain (+0.0017 on Test). Optional figure: `08_coreview_needs_consistent_timing.png`.

## Act 5: Result, and its cost

- **Headline:** Fixing what the ranker is given: +61% on Test. Changing the ranker itself: −7% to +1%.
- **Bullets:**
  - Combined system (mixed lists + consistent monthly statistics + "bought together" inputs): nDCG@10 0.0064 → 0.0104 on Test. 8 of 9 pre-registered predictions held.
  - **Cost:** every fix favours established products. Natural lists fill 56% of top-10 slots with bestsellers; mixed lists lose slightly on new products.
  - **Open problem:** the best system overall is not the best for new products. Choosing per purchase could gain up to 34–42% (upper bound).
- **Figures:** `02_what_moves_the_needle.png` (key, large), `09_test_confirms_val.png`. Optional: `10_who_wins_on_test.png`, `J4_routing_upper_bound.png`.

## Conclusion / takeaways box

1. Check where misses happen before tuning the ranker.
2. Train on lists built the way the ranker will be used.
3. Retrain the ranker whenever the retriever changes.
4. Compute product statistics the same way in training and in use.
5. New and established products need different setups.

## Limitations (small footnote)

- Act 2 is Validation-only; Acts 3–5 are confirmed on Test.
- Validation was reused to choose directions.
- Product titles leak future device names.
- The 5-core filter flatters "no reviews yet".
- Monthly statistics are an exploratory variant (they never use future data).
- The routing gain is an oracle upper bound.

## Figure list (all in `figures/`)

| File | Act | Caption |
|---|---|---|
| `01_where_failures_happen.png` | 1 | Only 6.1% of Test purchases reach the 200 candidates; 1.3% reach the top 10 |
| `J1_forced_list_shortcut.png` | 2 | In forced lists the purchase always sits below every candidate |
| `J2_k_curve_forced_vs_natural.png` | 2 | More candidates only hurt the shortcut-trained ranker |
| `J3_training_lists_decide_mlp_vs_din.png` | 2 | The training lists decide whether MLP or DIN wins |
| `03_two_retrievers_find_different_products.png` | 3 | The two retrievers find different products |
| `04_mixed_pool_gain_explained.png` | 3 | Most of the mixed-list gain needs retraining on the mixed lists |
| `05_learned_rule_and_statistics_timing.png` | 4 | The "no reviews yet" rule only works with statistics as old as in training |
| `06_where_the_stale_statistics_gain_comes_from.png` | 4 | Recently released products misread as new explain the whole gain (Validation and Test) |
| `07_statistics_age_curve.png` | 4 | Same ranker given product statistics of different ages (Test): new products best at the training age (~6 months), established products best when up to date |
| `08_coreview_needs_consistent_timing.png` | 4 | "Bought together" inputs only help with consistent timing |
| `02_what_moves_the_needle.png` | 5 | % change in Test nDCG@10: changing the ranker (−7% to +1%) vs changing what it is given (+21% to +61%) |
| `09_test_confirms_val.png` | 5 | Validation vs Test for the four pre-registered comparisons |
| `10_who_wins_on_test.png` | 5 | The best system overall is not the best for new products |
| `J4_routing_upper_bound.png` | 5 | Upper bound if the better ranker were chosen per purchase |
