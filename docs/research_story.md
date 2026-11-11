# What decides recommendation quality in a two-stage recommender? Findings and evidence

CP4285 Group 7: Brandon Kang, Tong Jing Yen, Phua Qing Rong. Original narrative 7 Nov 2026; review qualifications updated 11 Nov 2026.

Start with the [review guide](review_guide.md) and [poster evidence map](evidence_map.md) for the current interpretation, exact study identities and correction history. Dated experiments below preserve their individual scope. R1 now includes the 9 November post-hoc Test follow-up; score swaps are Validation-only. The stage split and snapshot construction may amplify the observed effects; their magnitude under an alternative training protocol remains untested.

- **Data:** Amazon Reviews 2023, Cell Phones & Accessories.
- **Status:**
  - The forced-vs-natural result (Finding 2) is Val-only, replicated on two candidate sets.
  - Findings 3–5 were found on Val, confirmed in a locked, pre-registered Test run, and their mechanism was replicated on Test (all five predictions held).
- **Figures:** `results/figures/20261107-story/` (ours) and `results/figures/poster_20261107/` (Jing Yen's).

## Summary

A two-stage recommender first **retrieves** 200 candidate products per user, then a **ranker** reorders them and shows the top 10. We asked what decides the quality of those 10 products.

**Thesis:** the fitted rankers depend on their training inputs. Candidate construction, retrieval-score inputs and statistics timing changed quality substantially in these experiments. The direction depends on the model and intervention; mixed lists sometimes help frozen rankers, and the original temporal protocol limits generalisation.

About 94% of the products users bought next never reach the 200 candidates, so the ranker can only work with what it is given. Beyond that, we found three ways the construction of the training data taught the ranker a shortcut:

| How the training data is built | Shortcut the ranker learned | Effect of fixing it | Evidence |
|---|---|---|---|
| **How missed targets enter the training lists** (forced in vs natural only) | "low retrieval score = the answer" | +24%; no collapse at large K; DIN overtakes MLP | Val, 3 seeds, two candidate sets |
| **Which retriever produced the lists** | the habits of that retriever | residual mixed-list system +35% Val / +22% Test; size control retains most of the Val gain | size-matched control; Test |
| **When the product statistics were calculated** | "no reviews yet = likely purchase" | +21% for the same frozen model on Test; co-review inputs start to help | interventions; replicated on Test |

**Result:** combining the fixes gives **+61% on Test** (nDCG@10 0.0064 → 0.0104). Changing the ranker's loss moved results by under 0.0005.

**Trade-offs:** the best residual system overall is not the best for new products. The 9 November frozen matching-ranker mixed-list gains, by contrast, are concentrated on new products. Group effects depend on the intervention and model family.

## The story we present

We set out to understand what really decides the quality of the ten products a recommender shows a user. Our system works in two stages: a retriever picks 200 candidate products, and a ranker reorders them to choose the top ten. When we looked at where the system fails, the first answer was clear. About 94% of the time, the product the user bought next was never among the 200 candidates, so no ranker could find it. Our case study showed why: many of these users had just moved to a new phone, while the retrievers kept suggesting accessories for the old one. The ranker can only work with what it is given, and with what its training data taught it.

The first lesson about training data came from how the original ranker was trained. Whenever the retriever missed the product a user bought, the pipeline forced that product into the training list anyway. Every forced product had a lower retrieval score than all the real candidates, so the ranker learned a shortcut: "the lowest-scored candidate is the answer." This had distorted two of the project's earlier conclusions:
- **Larger candidate counts looked harmful,** because quality collapsed at K = 500.
- **The simple MLP looked better than DIN.**

Training only on lists where the retriever had genuinely found the product removed the shortcut. Quality rose by 24%, the collapse disappeared, and DIN overtook the MLP. We found this in late October and Jing Yen replicated it on the cluster with the original candidates. It raised the question that drives the rest of the project: if the way the training lists are built matters this much, what else in the training data is the ranker learning from?

The second answer was the retriever itself. Our two retrievers find almost completely different products, sharing only about 12 of every 200. When we mixed their lists and trained the ranker on the mixed lists, results improved by 35% on Val and 22% on the held-out Test period. Retraining with the training-list count matched retained most of the gain. This rules out list count alone, while candidate scoring, scaling and optimisation exposure also change. Better candidates alone, without retraining, gave less than a quarter. This residual-family path shows that adapting training can capture additional gain. It does not establish that retraining is always necessary: the separate frozen matching MLP/DIN improve on mixed Test lists by about 24%/10%.

The third answer, and our deepest finding, was the timing of the product statistics the ranker uses, such as how many reviews a product has and how often it is reviewed together with the user's past products.
- **The setup.** The ranker was trained with statistics calculated at T0, the start of its training period. At test time it used those same T0 statistics, by then a year or more out of date. Recalculating them at the start of the test period improved the same frozen model by 16–30% on Val and 21% on Test.
- **The puzzle.** Statistics updated every month, which should be even better, made recommendations for new products worse.
- **The explanation.** The ranker had learned that "no reviews yet" means "new product, likely purchase", and the usefulness of this cue changed with the snapshot cutoff. The observed age preference is conditional on the training protocol.
- **The tests.**
  - Flipping that single input moved scores sharply.
  - Updating statistics only for products first reviewed during B recovered most of the gain in the registered controls.
  - New-product quality peaked when the statistics were as old as in training, while established products preferred fresh ones.

  We wrote these predictions down in advance, and they held on Val and again in the separately registered Test run. Training with monthly statistics kept the rule consistent and, as predicted, made the "bought together" inputs, which the team had found useless, start to help.

Putting the fixes together gives our best system: mixed candidate lists with consistently timed monthly statistics and the "bought together" inputs. It is 61% better than the original pipeline on Test, and eight of our nine pre-registered predictions held. Changing the ranker's training loss, by comparison, moved results by under 0.0005.

The residual-family improvements have a trade-off: several gains favour established products, and the best system overall is not the best for new products. The separate frozen matching-ranker mixed-list gains instead concentrate on new products. These outcomes depend on the model and intervention. Choosing the better ranker per purchase could gain up to 34–42%, an upper bound that needs the answer to choose. New products therefore need a different setup, and that is the open problem we leave.

Our conclusion: in a two-stage recommender, how the ranker's training data is built decides what it learns. That means how missed products are added, which retriever produced the lists, and how old the statistics are. Getting these right matters far more than choosing a ranking model. These are known pitfalls, namely sample selection bias between stages and training–serving skew. We measured how large they are on a public benchmark, explained why they happen, and showed that they had changed the conclusions our own project had drawn.

---

## 1. Setup and terms

### Time periods

| Period | Dates | Used for |
|---|---|---|
| A | before **T0** = 26 Jul 2020 | training the two-tower retriever |
| B | T0 to **T1** = 11 Aug 2021 | training the rankers (315,114 queries) |
| Val | T1 to **T2** = 16 Jul 2022 | choosing directions (232,904 events, 124,416 users; 33% new products) |
| Test | after T2 | 297,390 events, 112,081 users; 44% new products; multiple disclosed study-specific runs |

"New product" means the target was first reviewed during that evaluation period. All other products are "established".

### Retrievers and candidate lists (200 per query)

- **P1, recency retriever:**
  - Products are compared by their frozen MiniLM text vectors.
  - The user is represented by a recency-weighted average of their last 50 products.
  - No training.
- **P2, two-tower retriever:**
  - A text item tower and a self-attention user tower.
  - Trained on period A.
- **P3, mixed list:** alternate P1 #1, P2 #1, P1 #2, … skipping duplicates until 200 products.

### Ranker

There is one design: the retrieval score plus a small MLP correction. Many versions were trained, each defined by:

| Choice | Options |
|---|---|
| Lists it is trained on | P1, P2 or P3, from period-B queries |
| Inputs | Text inputs (text similarity, category/store match, basic statistics such as "has any reviews yet"); optionally **co-review inputs** (how often people who reviewed the user's past products also reviewed the candidate); optionally context inputs (price, age, review count) |
| Loss | BCE or pairwise |
| Statistics timing in training | T0 (start of period B) or monthly |

### Statistics timing

Product statistics (review counts, co-review counts, "has any reviews yet") are counted from all reviews before a cut date. The options:

| Label | Statistics for each query come from reviews before… |
|---|---|
| T0 | 26 Jul 2020 (the team's original setting, also used for Val and Test) |
| T1 / T2 | the start of Val / Test (the "period start", matching how training used T0) |
| Monthly | the first day of the query's month |

### Evaluation

- nDCG@10, per user then averaged over users. Misses count as 0.
- Seeds 0/1/2, averaged.
- 95% intervals from 1,000 paired user resamples.
- Row codes (B1, D7, …) refer to `docs/experiment_overview.md`.

---

## 2. Starting point

**The team's results (4 Nov):**
- The best system was the BCE text ranker on P1 lists with T0 statistics (**B1**, the "current pipeline").
- Pairwise loss and co-review inputs changed results by only about ±0.0003.
- On Test, rebuilding the graph statistics at T1 helped (+0.0009 to +0.0014).

**The professor's open comments:**
- explain *why* results change;
- report @10 and secondary metrics;
- explain the U-shaped score curve;
- try random mixing;
- try leak-free context and metadata;
- do a case study of 20–30 items;
- consider tailoring by user group.

---

## 3. Finding 1: most failures happen before the ranker sees the right product

**Observation (Test, current pipeline B1):**

| Stage | Share of targets | Count |
|---|---:|---:|
| All products users bought next | 100% | 297,390 |
| …that were among the 200 candidates | 6.1% | 18,116 |
| …that the ranker put in the top 10 | 1.3% | 3,884 |

**Where:** on Val, **87.6% of all targets are missed by both retrievers** (204,111 of 232,904).

**Why (case study and counts):**
- 24 cases drawn with rules fixed in advance, plus counts over all 204,111 double misses.
- Most double misses are category jumps away from the last product: the retrievers echo the user's last purchase.
- Of missed targets whose title names a device, 62% name a device the user never had, usually a newer model. The retrievers keep suggesting accessories for the old phone.

**Consequence:** the first stage is the bottleneck. A ranker can only reorder what it receives.

Figure: `01_where_failures_happen.png`. Sources: `results/test_final/20261107/`, `results/analysis/20261106-switch-cases*`.

---

## 4. Finding 2: the first shortcut, how missed targets entered the training lists

**Status:**
- Val only.
- Found 28–29 Oct (`docs/report_2026-10-28.md`; controlled A/B/C/D study in `docs/evaluations.md`).
- Replicated on the cluster on the original candidates by Jing Yen (6–7 Nov, 3 seeds; `results/targeted_study/policy-all-seeds-20261107/`).
- Uses the team's two-tower candidate lists and feature MLP/DIN rankers.
- Every ranker in Findings 3–5 is trained on natural lists *because* of this result.

**Observation.**
- The original training lists **forced** each missed target into the list: 276,321 of 315,114 lists (88%).
- The forced target scored below every retrieved candidate in **100%** of them.
- Under forced training:
  - quality collapsed as more candidates were passed to the ranker;
  - the simple MLP beat DIN.

**Explanation.** The ranker learned "lowest retrieval score = the answer":
- Its score/logit correlation is −0.40 under forced training and +0.16 under natural training.
- 92–96% of its top-10 slots went to candidates added beyond rank 200.

**Test: change only the training population** (natural = only lists where the retriever found the target; same initial weights):

| Comparison (Val nDCG@10) | Effect [95% CI] |
|---|---|
| Natural − forced MLP (cluster, original candidates, 3 seeds) | +0.00169 [+0.00145, +0.00192] (0.0069 → 0.0086, +24%) |
| Same, laptop replication on rebuilt candidates (29 Oct) | +0.00187 [+0.00163, +0.00210] |
| Natural DIN − natural MLP (28 Oct) | +0.00122 [+0.00101, +0.00141] (DIN 0.0098 vs MLP 0.0086) |
| K = 500 vs best K, forced MLP | −0.0060 (−72%) |
| K = 500 vs best K, natural MLP | −0.0007 (nearly flat) |

**Consequences.**
- The earlier conclusions "larger K₁ hurts" (RQ2) and "MLP beats DIN" (RQ3) were mostly artefacts of forced training.
- Masking the retrieval score does not fix it (−42%), so natural lists are the remedy.

**Cost.** Natural-list training loses on new products (−0.00076 [−0.00103, −0.00049]) and concentrates on bestsellers:

| Training lists | Top-10 slots from the top-1% bestsellers | Distinct products shown |
|---|---:|---:|
| Natural | 56% | 10,749 |
| Forced | 41% | 16,394 |

Figures: Jing Yen's `fig5_shortcut.png`, `fig1_k_curve.png` and `fig4_training_vs_architecture.png` in `results/figures/poster_20261107/`.

---

## 5. Finding 3: what the ranker chooses from matters more than the ranker itself

### 5.1 Ranker changes barely move the results

Same system with and without one change, Test nDCG@10:

| Change | Effect [95% CI] |
|---|---|
| Pairwise instead of BCE (text inputs) | −0.00029 [−0.00044, −0.00015] |
| Pairwise instead of BCE (with co-review inputs) | −0.00045 [−0.00061, −0.00030] |
| Add co-review inputs, stale T0 statistics | +0.00005 [−0.00008, +0.00016] |

These first three rows use the team's trained weights.

| Change | Effect [95% CI] |
|---|---|
| Statistics at the period start (T2), same frozen model | **+0.00138** [+0.00122, +0.00153] |
| Monthly statistics in training and scoring (B9 vs B1) | **+0.00160** [+0.00141, +0.00180] |
| Add co-review inputs, monthly statistics | **+0.00166** [+0.00149, +0.00184] |
| Mixed candidate list P3 | **+0.00142** [+0.00115, +0.00170] |
| Combined (D7) | **+0.00394** [+0.00359, +0.00429] |

Figure: `02_what_moves_the_needle.png`.

### 5.2 The two retrievers find different products

**Recall@200** (share of targets among the 200 candidates):

| List | Val | Test |
|---|---:|---:|
| P1 recency | 5.5% | 5.3% |
| P2 two-tower | 6.9% | 4.4% |
| P3 mixed | 8.0% | 6.2% |

- **Low overlap.** The P1 and P2 lists share only about 12 of 200 candidates on Val. Recency finds new products; the two-tower finds established ones.
- **Ceiling.** An oracle that picks the better retriever per event reaches nDCG@10 0.0117 on Val. That's an upper bound only.
- **P2 on Test.** The two-tower drops to 4.4% because it was trained on period A, two years before Test.

Figure: `03_two_retrievers_find_different_products.png`.

### 5.3 The mixed list helps, but only if the ranker learns on it

**Observation.** The same ranker design (BCE, text inputs, T0) trained on each list:

| System | Val | Test |
|---|---:|---:|
| B1, P1 lists | 0.0081 | 0.0064 |
| C1, P2 lists | 0.0101 | — |
| D1, P3 lists | 0.0110 | 0.0079 |

That is **+35% on Val and +22% on Test**.

**Explanation, tested with a size-matched control.** P3 lists contain the target more often, so the P3 ranker also had 1.9× more training examples (38,861 vs 20,676). We retrained it on a random 20,676 (D1s) to separate the two effects. The Val gain splits as follows:

| Source of the +0.0029 | Effect | Share |
|---|---:|---:|
| Better candidates alone (P1-trained ranker scored on P3 lists) | +0.00067 | 23% |
| **Ranker trained on P3 lists, same amount of data** | **+0.00158** [+0.00134, +0.00181] | **55%** |
| The extra training examples a better retriever provides | +0.00063 [+0.00046, +0.00079] | 22% |

All three size-matched seeds stopped at the 10-epoch cap, so the extra-data share may be slightly overstated.

**Trade-off.** The mixed list helps established products and hurts new ones:

| | Established | New |
|---|---:|---:|
| Val | +0.0048 | −0.0017 |
| Test | +0.0025 [+0.0021, +0.0028] | −0.0018 [−0.0021, −0.0014] |

**Supporting evidence.** A retriever retrained on A + B finds **62% more targets** (Val Recall@200 0.069 → 0.111, robust across seeds). But rankers trained on the old lists turn only +0.0004 of that into top-10 quality.

**Lesson:** check whether the ranker can use the changed candidates and score inputs. Retraining captured further gain in this control, while the later R1 Test study also found positive frozen-ranker mixed-list effects. The need for retraining depends on the change and the fitted ranker.

Figure: `04_mixed_pool_gain_explained.png`. Sources: `results/analysis/20261106-pool-ranking`, `…-20261107-pool-size-control`, `…-20261106-fresh-two-tower`.

---

## 6. Finding 4 (centrepiece): the ranker learned a rule that depends on the timing of its statistics

This finding goes through all five depth steps: observe, locate, explain, test, predict.

### Step 1, observe

The team scored Val with T0 statistics, which were a year out of date. Rescoring the same frozen models with T1 statistics improved every ranker:

| Model (frozen, trained with T0 statistics) | Val at T0 | Val at T1 | Val monthly |
|---|---:|---:|---:|
| BCE text (B1) | 0.0081 | **0.0094** | 0.0085 |
| Pairwise text (B4) | 0.0078 | **0.0097** | 0.0087 |
| Pairwise co-review (B5) | 0.0077 | **0.0102** | 0.0098 |
| BCE co-review (B6) | 0.0082 | **0.0098** | 0.0096 |

That's +16–30%. Test confirms it: B1 with T2 statistics gains **+0.00138** [+0.00122, +0.00153] (pre-registered prediction 1, held).

### Step 2, locate

Monthly statistics, which are *fresher*, did worse than T1. The loss is concentrated on new products. For B1 on Val, new products score 0.0117 at T1 and 0.0062 monthly.

### Step 3, explain

In the natural-hit training lists with T0 statistics, **40.5% of targets showed no prior reviews, against 17.7% of other candidates in the inner-fit B prefix**; the full-B refit shares were **43.1% / 20.3%**. The ranker learned **"no reviews yet → new product → likely purchase"**. The registered frozen models preferred particular statistics ages for new targets; this is conditional on their T0-based training construction:

| Statistics | Age per query (months) | Bought / other showing "no reviews" | AUC of the rule |
|---|---|---:|---:|
| **Training (B, T0)** | 0–12.5 (mean 6.1) | **43% / 20%** | 0.614 |
| Val at T0 | 12.5–23.7 | 71% / 44% | 0.635 |
| **Val at T1** | 0–11.2 (mean 5.5) | **41% / 15%** | 0.626 |
| Val monthly | 0–1 | 10% / 1.4% | 0.540 |
| Test at T0 | 23.7–37.2 | 87% / 66% | 0.604 |
| **Test at T2** | 0–13.6 (mean 4.9) | **50% / 21%** | 0.646 |
| Test monthly | 0–1 | 16% / 2.5% | 0.567 |

- **The period start (T1 on Val, T2 on Test)** recreates the training conditions.
- **T0** makes too many products look new, and much more so on Test, where T0 statistics are two to three years old.
- **Monthly statistics** nearly erase the rule.

Figure: `05_learned_rule_and_statistics_timing.png`.

### Step 4, test by intervention (no training, frozen models)

**Flip one input.**
- We changed only "has any reviews" from no to yes for candidates in 10,000 Val lists.
- The BCE text score fell by 1.30, about one within-list standard deviation.
- New targets in the top 10 fell from 34% to 14%. Pairwise rankers rely on the rule about 3× less.

**Column swap.** We started from T0 statistics and updated one piece at a time to the period-start values (T1 on Val, T2 on Test). Share of the full gain recovered, range across the four models (predictions written in advance; all held on Val and on Test):

| Updated | Val overall | Val new | Test overall | Test new |
|---|---:|---:|---:|---:|
| Only the "has any reviews" flag | 26–87% | 44–208% | 27–84% | 53–226% |
| Flag + review count | 89–114% | 126–179% | 84–116% | 131–193% |
| **All statistics, only for products released after T0** | **86–101%** | 112–141% | **92–100%** | 109–128% |
| All statistics, only for products released before T0 | 14–24% | negative | 11–16% | negative |

So the stale-statistics problem is products released after T0 being misread as brand new. Fixing their flag and counts recovers the whole gain. Updating older products barely matters.

**Statistics-age curve.** Every query gets statistics cut a fixed number of months before its month. B1 (BCE text), nDCG@10:

| Lag (months) | 0 | 1 | 3 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| Val, new products | 0.0062 | 0.0079 | 0.0097 | 0.0107 | **0.0110** | 0.0102 |
| Val, established products | **0.0096** | 0.0093 | 0.0089 | 0.0084 | 0.0081 | 0.0079 |
| Test, new products | 0.0076 | 0.0095 | 0.0107 | **0.0113** | 0.0113 | 0.0107 |
| Test, established products | **0.0070** | 0.0067 | 0.0063 | 0.0060 | 0.0059 | 0.0061 |

- **New products** do worst with the freshest statistics and peak near the training age: lag 6–9 on Val, lag 6 on Test.
  - That's where the statistics look as they did in training. On Val at lag 6–9, 42–52% of bought products and 17–24% of other candidates show "no reviews". On Test at lag 6, it's 55% and 24%. Training had 43% and 20%.
- **Established products** prefer the freshest statistics.
- All four models show this pattern on both Val and Test.
- On Test, the fixed-cut-date curve also rises steadily from T0 to T2 for all four models.

So two mechanisms are at work:
1. **The learned rule** wants statistics of training-matched age.
2. **The counts themselves** are more useful when fresh.

Figures: `06_where_the_stale_statistics_gain_comes_from.png`, `07_statistics_age_curve.png` (Val and Test). Sources: `results/analysis/20261107-statistics-timing/` (Val) and `results/test_final/20261107/timing/` (Test).

### Step 5, a prediction that came true

If the problem is mismatched timing, then *training* with monthly statistics as well should keep the rule consistent while using fresh counts. The "bought together" inputs should then start to help.

**Val:**

| Rankers trained and scored with monthly statistics | nDCG@10 |
|---|---:|
| BCE text (B9) | 0.0093 |
| **BCE + co-review (B10)** | **0.0106** |

Co-review − text is **+0.0013** [+0.0011, +0.0015], against +0.00016 with stale statistics.

**Test** (pre-registered prediction 3, held):
- co-review − text **+0.00166** [+0.00149, +0.00184];
- positive for both new (+0.0028) and established (+0.0012) products.

### Test replication of Steps 3–4 (summary)

The same diagnostics were rerun on Test (T2 in place of T1) with the same four frozen models. Predictions were written in `docs/test_protocol_2026-11-07_addendum_timing.md` before any Test arm was built. **All five held** (`results/test_final/20261107/timing/`).

| Prediction (all four models) | Test result | Verdict |
|---|---|---|
| The "has any reviews" flag alone recovers ≥ 1/3 of the gain on new products | 53–226% | held |
| Updating only products released after T0 recovers ≥ 75% of the gain; older products < 1/3 | 92–100% vs 11–16% | held |
| Quality rises steadily as the cut date moves from T0 to T2 | monotone for all four | held |
| New products: lowest at lag 0, peak at lag 3–9, lag 12 below the peak | peak at lag 6 for all four | held |
| Established products: better at lag 0 than lag 12 | all four | held |

The mechanism is therefore not a Val artefact. It was also observed in the registered Test diagnostics, with a higher share of new products (44% vs 33%). Earlier Test exposure and the unchanged temporal construction still limit generalisation.

**Lesson:** product statistics must be calculated the same way in training and at scoring time. A mismatch silently breaks rules the model learned, and can make a useful input look useless.

---

## 7. Finding 5: co-review inputs only help when their timing is consistent

Gain from adding co-review inputs to the BCE ranker on P1 lists:

| Setting | Co-review − text [95% CI] |
|---|---|
| Stale: trained T0, scored T0 (Val; 36 fits incl. extra negative draws) | +0.00016 [+0.00007, +0.00026] |
| Trained T0, scored T1 (Val) | +0.00049 [+0.00038, +0.00061] |
| Monthly in training and scoring (Val) | +0.00131 [+0.00113, +0.00148] |
| Monthly in training and scoring (Test) | **+0.00166** [+0.00149, +0.00184] |
| On mixed P3 lists, monthly (Val / Test) | +0.00154 / +0.00158 |

The team concluded that co-review barely helps because it was measured with stale statistics. The extra negative draws also showed that the 5 Nov "disappearing" co-review gain was noise from one training draw (draw-to-draw SD about 0.0001).

Figure: `08_coreview_needs_consistent_timing.png`. Sources: `results/analysis/20261106-refit-draws`, `…-rolling-ranking`, `…-20261107-pool-rolling`.

---

## 8. Putting it together, and the locked Test run

**Protocol.**
- `docs/test_protocol_2026-11-07.md` was written and hashed before Test was opened.
- The same code first reproduced every saved Val list, feature and rank exactly.
- Each system was scored once, with no training or selection on Test.
- The rebuilt recency retrieval reproduces the team's earlier Test numbers exactly (nDCG@10 0.006043, Recall@200 0.053114).

### Test results, nDCG@10

| System | All | New | Established |
|---|---:|---:|---:|
| S0 recency order, no ranker | 0.0060 | 0.0076 | 0.0054 |
| S1 current pipeline (B1, T0) | 0.0064 | 0.0082 | 0.0057 |
| S3 same model, statistics at T2 | 0.0078 | 0.0118 | 0.0058 |
| S4 P1, monthly, text (B9) | 0.0080 | 0.0102 | 0.0070 |
| S5 P1, monthly, co-review (B10) | 0.0097 | **0.0130** | 0.0082 |
| S6 mixed list (D1, T0) | 0.0079 | 0.0064 | 0.0082 |
| S7 mixed list, monthly, text (D6) | 0.0088 | 0.0076 | 0.0088 |
| **S8 mixed list, monthly, co-review (D7)** | **0.0104** | 0.0092 | **0.0105** |

### Pre-registered predictions: 8 of 9 held

| Prediction | Test result | Verdict |
|---|---|---|
| Period-start statistics help the same frozen model | +0.00138 [+0.00122, +0.00153] | held |
| T2 statistics at least as good as T1 | 0.0078 vs 0.0074 | held |
| Mixed list helps established products | +0.00249 [+0.00215, +0.00284] | held |
| Mixed list hurts new products | −0.00178 [−0.00210, −0.00141] | held |
| Mixed-list gain smaller than on Val (44% vs 33% new products) | +0.00142 vs +0.00288 | held |
| Co-review helps with monthly statistics | +0.00166 [+0.00149, +0.00184] | held |
| Combined system helps established products | +0.00480 [+0.00434, +0.00525] | held |
| Combined-system gain smaller than on Val | +0.00394 vs +0.00494 | held |
| Combined system does not help new products (vs S1) | +0.00095 [+0.00050, +0.00152] | **not held** |

**About the miss.** D7 beats the current pipeline on new products only because that pipeline uses year-old statistics. Against the fair baselines it still loses on new products: −0.0027 vs S3 and −0.0038 vs S5.

**Who wins:**
- **D7** is best overall and for established products.
- **B10** (P1 lists, monthly co-review) is best for new products.

The best system overall is not the best for new products. Jing Yen's per-event oracle points the same way: choosing the better of two rankers for every purchase would gain +34–42% on Val, with DIN much better on new products (+44%). That is an upper bound, because the choice uses the answer. Together, these are the clearest answer to the professor's question about tailoring.

Figures: `09_test_confirms_val.png`, `10_who_wins_on_test.png`. Source: `results/test_final/20261107/`.

---

## 9. Answers to the professor's specific comments

| Comment | Answer and evidence |
|---|---|
| Explain *why* results change | Findings 1–5: retrieval misses, ranker learning on its own lists (size-matched control), the "no reviews yet" rule (intervention, swap and age curve, replicated on Test) |
| @10 metrics and secondary metrics | nDCG@10 is the headline. HR@3, MRR@10 and Recall@10 rank systems the same way (Val and Test `contrasts.csv`) |
| U-shaped score curve | Not a popularity effect. It's a "low retrieval score = answer" shortcut present in every popularity group, from forced-insertion training; natural-list rankers show none (`docs/overnight_results_2026-11-05.md`) |
| Random mixing | Fails: −0.0011 on Val. Inserted targets stay easy to spot because their text is unlike the user's history (AUC 0.10). Missed targets are by definition dissimilar, so insertion cannot hide them (B14) |
| Leak-free context / metadata | +0.0013 in a fair comparison (all statistics at T1, B12 vs B2). It acts mostly as a "new product" prior; the leak test passes on 656 same-timestamp targets |
| Case study of 20–30 items | 24 pre-registered double misses plus counts: device upgrades and category jumps (Finding 1) |
| Tailoring by user group | Product popularity, not history length, decides the winner. On Test, D7 is best for established products and B10 for new ones. Nothing helps the least-reviewed products |

---

## 10. What did not work

- **Random mixing** (Section 8).
- **Device-upgrade candidates:**
  - adding 20 newer-model products to the list raised Recall@200 by only +0.0014;
  - the check exposed the title leak (Section 10).
- **A blend of BCE and pairwise rankers:**
  - an oracle picking per event would gain +0.0019;
  - weights tuned on period B do not transfer, because "new product" means something different in B (57% of top-10 slots) and Val (18%).
- **A fresher retriever under a frozen ranker:** +62% Recall@200 but only +0.0004 nDCG@10. Using it needs a ranker trained on its lists, which in turn needs a retriever retrained over time (not done).

---

## 11. Limitations

- **Val reuse.** Val was used many times to choose directions. The main effects are confirmed on Test with pre-registered predictions, and the mechanism diagnostics are replicated on Test. Exploratory Val-only results (context, random mixing, meso analysis) are not Test-confirmed.
- **Title leak.** Amazon metadata was crawled in 2023, so titles name devices released after the review date: at least 8.5–21% of top-10 slots on Val, and 13.4% of targets. This affects every text-based model equally.
- **5-core filter.** It keeps only products that eventually reach 5 reviews, which makes "no reviews yet" more predictive than it would be in practice.
- **Monthly statistics.** They depart from the agreed rule (statistics as of the period start). They never use future data, but they are an exploratory variant.
- **Rebuilt weights.** Our B1 is a rebuild of the team's model: Test 0.0064 vs the team's 0.0065. Within-study comparisons use one consistent set.
- **The forced-vs-natural result (Finding 2) is Val-only.** It was replicated on two candidate sets and three seeds, but not run on Test.
- **Diversity cost.** Natural-list training fills 56% of top-10 slots with bestsellers. Our gains are also concentrated on established products; their diversity was not measured.
- **New and rarely reviewed products remain unsolved.** No system improves the least-reviewed products. The routing gain is an oracle upper bound.

---

## 12. Takeaways

1. **Check where misses happen before tuning the ranker.** Most failures occur before ranking.
2. **Train on lists built the way the ranker will be used.** Forcing missed targets into training lists teaches a shortcut.
3. **Re-evaluate compatibility whenever retrieval changes.** Test candidate membership, score inputs and ranking outcomes; compare adaptation with the same frozen-model reference.
4. **Calculate product statistics the same way in training and in use.** A mismatch silently breaks learned rules and can make a useful input look useless.
5. **Different products need different setups.** The best system for established products is not the best for new ones.

---

## 13. Where the evidence lives

| Topic | Files |
|---|---|
| Experiment table and terms | `docs/experiment_overview.md` |
| Full write-ups | `docs/overnight_results_2026-11-05.md`, `docs/overnight_results_2026-11-06.md` |
| Decisions and approvals | `docs/decisions.md` |
| Test protocol and addendum | `docs/test_protocol_2026-11-07.md`, `docs/test_protocol_2026-11-07_addendum_timing.md` |
| Test results | `results/test_final/20261107/` (`tables.md`, `predictions.csv`, `timing/`) |
| Size-matched control | `results/analysis/20261107-pool-size-control/` |
| Statistics-timing diagnostics | `results/analysis/20261107-statistics-timing/` |
| Forced vs natural lists (Finding 2) | `docs/report_2026-10-28.md`; `results/targeted_study/policy-all-seeds-20261107/`, `e3-20261107-all-seeds/`, `top10-diversity-20261107/`, `meso-oracle-20261107/` |
| Figures | `results/figures/20261107-story/` (index in `README.md`); Jing Yen's `results/figures/poster_20261107/` |
| Code | `src/cascade/pipelines/test_final.py`, `pool_size_control.py`; `src/cascade/analysis/statistics_timing*.py`; `scripts/story_figures.py` |
