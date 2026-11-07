# Overnight research loop results — 6 Nov 2026

**Val only, exploratory; awaiting team review.** Run 6 Nov 01:50 – 7 Nov 03:23 (phase H §9) on Brandon's laptop (M2 Pro, 16 GB), led by Claude as researcher with subagents. A rate-limit pause from about 03:00 to 13:30 stopped the subagents, but their background jobs kept running. Brandon then extended the run past the original 11:30 stop.

- **Records:** run log and every research decision in `docs/analysis_loop_2026-11-06.md`; authorisation in `docs/decisions.md` (2026-11-06).
- **What was not done:**
  - no Test file was opened;
  - nothing (K, index, ranker, pool, blend, δ) was selected;
  - no existing code or result was changed;
  - nothing was committed.
- **Status:** complete.

All metrics follow these rules:
- per-user macro over the full Val population (232,904 events / 124,416 users), with misses counted as 0;
- seed-mean event metrics (seeds 0/1/2), then a paired user bootstrap (1,000 draws, seed 0);
- new = target first reviewed at or after T1; warm = the rest.

"P1" is the team's current pipeline: frozen MiniLM + 90-day recency, natural top 200. "bce_text" is the registered residual BCE ranker on it, rebuilt on 5 Nov: Val nDCG@10 0.008141.

## Findings, ranked by strength of evidence

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | **The candidate pool changes the ranker's result more than any ranker change tested.** The same bce_text recipe trained and served on a fixed recency + two-tower fused pool reaches **0.011019 vs 0.008141 (+0.002878 [0.002544, 0.003211], +35%)**. Warm targets gain +0.0048; new targets lose −0.0017. Most of the gain needs the ranker to be *trained* on the new pool. | 6 new fits, matched initial states, exact checks, decomposition | Strong |
| 2 | **The published Val numbers were scored with year-old graph/support features.** Frozen models scored with features from the start of Val (T1, the train/serve-consistent setting) gain +0.0013 to +0.0025 for every method. The rankers learned "no reviews yet" as a newness cue, and stale features distort it. | No-training rescoring of 12 checkpoints; mechanism probes | Strong |
| 3 | **Co-review features help, and more when the graph is fresh.** Pooled over 3 seeds × 4 training draws: T0 +0.000157 [0.000065, 0.000255]; **T1 +0.000492 [0.000375, 0.000609]**; all on warm targets. **Rolling monthly training and serving: +0.001308 [0.001129, 0.001483]**; rolling bce_coreview reaches 0.010613. | 36 refit-draw fits + freshness rescoring + rolling fits | Strong |
| 3b | **The gains are sub-additive.** Fused pool + context, all features at T1, reaches 0.012167 (+0.004026 [0.003658, 0.004415], about +49% over the current pipeline), against a sum of single gains of about +0.0055. Pool and freshness overlap. | 3 new fits + rescoring | Moderate |
| 4 | **Leak-safe context features add a modest real gain** (price, period-start review count/rating/age, user history length, time gap). Fair comparison, everything at the period start: **+0.001273 [0.001052, 0.001501]** over bce_text(T1). The driver is a "new at period start" prior. The headline +0.0027 over bce_text(T0) is about half stale-baseline effect. | 3 fits, attribution, leak re-check | Moderate |
| 5 | **Both retrievers miss 87.6% of Val targets, and the largest single reason is device change.** 62% of double-missed targets with a recognisable model name a device the user's history never names, against 34% for retrieved targets. Within the same device family, the target is usually newer (74%). | Frozen-rule case study (24 cards) + scale counts | Moderate (title mentions only) |
| 6 | **Crawl-time product titles leak future device names.** At least 8.5% of recency and 20.9% of two-tower top-10 slots, and 13.4% of Val targets, name one of 4 checked models before its release. The release filter cannot stop this. | Post-hoc audit against external release dates | Moderate (lower bound; a dataset limitation) |
| 7 | The retriever oracle (better of recency / two-tower per event) is +0.0054 over recency, almost all warm. Fusing without a ranker gives +0.0011 (interleave order). | Pools build | Strong (oracle is an upper bound only) |
| 8 | A rule-based "device-upgrade" candidate source raises Recall@200 by +0.0014 overall and +0.0139 for newer-model targets. Part of it rests on leaked future names. | No-training pool test | Weak/illustrative |

---

## 1. Retrieval pools with the ranker held fixed (Track A; professor items P-c, P-d)

**Question.** How does the candidate pool change what the same ranker can deliver? This is the project's core question, asked on the current valid pipeline.

**What was run.**
- **Pools:**
  - P1 = recency top 200;
  - P2 = frozen Two-tower (text) seed 0, natural top 200;
  - P3 = round-robin interleave of the two, de-duplicated to 200, with the skip score = RRF (k = 60). The rule was fixed before fitting; the Val overlap was seen at 01:45 and is disclosed.
- **Ranker:** the registered bce_text recipe (same selector, refit and initial states) trained on each pool, seeds 0/1/2.
- **Runs:** `results/analysis/20261106-pools`, `results/analysis/20261106-pool-ranking`.
- **Checks:**
  - saved two-tower lists verified (forced target always in column 201);
  - float32 regeneration matches;
  - filters, features and initial-state hashes match;
  - epoch 0 equals each pool's own order.

**Results (Val nDCG@10).**

| System | Overall | New | Warm |
|---|---|---|---|
| P1 recency order | 0.006284 | 0.007611 | 0.005606 |
| P2 two-tower order | 0.005642 | 0.000361 | 0.007983 |
| P3 fused, interleave order | 0.007408 | | |
| Retriever oracle (upper bound) | 0.011705 | 0.007889 | 0.013320 |
| P1 ranker (bce_text) | 0.008141 | | |
| P2 ranker | 0.010081 | | |
| **P3 ranker** | **0.011019** | | |

- **Pool recall:** Recall@200 is P1 0.0547, P2 0.0687, P3 0.0795. The two lists share on average only about 12 items.
- **P3 ranker − P1 ranker:** +0.002878 [0.002544, 0.003211]; new −0.001701; warm +0.004797.
- **P2 ranker − P1 ranker:** +0.001940; new −0.004967.
- **P3 ranker − P2 ranker:** +0.000938, all of it on new targets.
- **Decomposition** (P1 checkpoints scored on the P3 lists with their own cosine skip):
  - better candidates alone: +0.000670 [0.000397, 0.000947];
  - training on P3 lists: +0.002208. This step also changes the skip score and the scaler.
- **Same-candidate subset** (targets in both P1 and P3): the P3 ranker *loses* (−0.0426). The gain comes from the 9,394 targets only P3 retrieves; longer mixed lists add strong distractors.

**Finding.** Retrieval choices dominate what the cascade delivers. The same ranker is 35% better on a fused pool. Most of that needs the ranker to *learn* on that pool: the fused lists carry twice the natural positives, skewed warm. P2 retrieval is worse than P1, yet its ranker beats P1's, so retrieval-order quality does not predict cascade quality. The cost is new-item quality.

**Caveats.**
- The pools carry about 2× the natural positives. The decomposition separates candidates from training only roughly.
- The RRF skip score creates exact ties. A tie-broken variant was dropped after its gate exposed an inconsistent definition (lead error, recorded).

**Size-matched control (7 Nov, `results/analysis/20261107-pool-size-control`).** The P3 ranker was refit on a fixed random subset of P3 natural B queries with exactly P1's counts (prefix 16,819 / tune 3,857 = 20,676), with everything else unchanged. Hypothesis written before fitting.
- Size-matched P3 ranker: 0.010387 (new 0.006300, warm 0.012056).
- vs P1 ranker: **+0.002246 [0.001901, 0.002602]**; new −0.002366, warm +0.004170.
- Full P3 ranker − size-matched: +0.000632 [0.000464, 0.000790].
- Size-matched − P1 checkpoints on P3 lists: +0.001576 [0.001338, 0.001805].
- So the +0.002878 splits roughly into 23% better candidates alone, 55% training on the P3 distribution at equal size, and 22% from the extra positives.
- Caveat: all three seeds selected the epoch cap (10), so the size-matched fit may be slightly understated and the "extra positives" share slightly overstated.

**Additivity test: pool × freshness × context** (`results/analysis/20261106-pool-combo`). This is an additivity test, not a final-pipeline selection.

| System | nDCG@10 | New | Warm | vs P1 bce_text(T0) |
|---|---|---|---|---|
| P1 bce_text (T0) | 0.008141 | 0.008666 | 0.007887 | — |
| P3 bce_text (T0) | 0.011019 | 0.006965 | 0.012684 | +0.002878 |
| P3 bce_text (T1) | 0.011451 | 0.006751 | 0.013221 | +0.003310 |
| P3 + context (T1 context, T0 common) | **0.012697** | 0.015008 | 0.011203 | **+0.004557 [0.004183, 0.004961]** |
| P3 + context (all at T1, consistent) | 0.012167 | 0.010815 | 0.012332 | +0.004026 [0.003658, 0.004415] |

- **Freshness on P3:** only **+0.000432 [0.000272, 0.000589]** (new −0.0002), against +0.0013 on P1.
- **Context on P3:** +0.000716 [0.000458, 0.000977] in the consistent T1 comparison, mostly on new targets (+0.0041).
- **Pool gain once context is present:** P3+context − P1+context, both at T1, is +0.001462 [0.001094, 0.001824].

**Finding.** The gains are **sub-additive**. The single-factor gains sum to about +0.0055; the consistent combination gives +0.0040. Pool and freshness overlap, because both mostly add warm, graph-known targets. Context partly restores the new-item quality that the fused pool loses. The consistent combination is about **+49% over the current pipeline**, with better new and warm quality than P1. As with P1, mixing T0 common features with T1 context scores higher, by leaning hard on newness. That is a staleness artefact and is not recommended.

## 2. Graph freshness on Val, and why it matters (Track E; P-b)

**Question.** The team saw on Test that period-start graph features help. Does that hold on Val with correct sequencing, and why?

**What was run.**
- Frozen checkpoints (4 methods × 3 seeds) rescored with graph/common features from all reviews before T1, and with monthly snapshots.
- Then hybrid snapshots, training-data statistics and a logit probe.
- **Runs:** `results/analysis/20261106-freshness-val`, `…-freshness-mechanism`.

**Results.**
- **T1 − T0, overall nDCG@10:**
  - bce_text +0.00129 [0.00112, 0.00146] (new +0.0031);
  - pairwise_text +0.00190;
  - pairwise_coreview +0.00249;
  - bce_coreview +0.00163.
- **Monthly refresh:** worse than T1 overall for every method. New −0.004 to −0.006; warm +0.0013 to +0.0020.
- **Mechanism:**
  - In training, 40.5% of positives were "unknown" (no reviews before the snapshot), against 17.7% of negatives (AUC 0.614).
  - Flipping only "known" lowers the logit by −1.30 (bce_text), against a within-list SD of 1.39. Pairwise relies on it about 3× less.
  - Monthly snapshots make most new products "known" and erase the cue.

**Finding.** The consistent setting is "features as of the query period's start": training used T0 for B, so Val should use T1. The published Val numbers were scored with year-old T0 features. Fresh co-review paths carry real information, but the frozen rankers also lean on a staleness-dependent newness cue.

**Rolling-snapshot training** (`results/analysis/20261106-rolling-ranking`):
- **Setup:** bce_text and bce_coreview retrained with each B query's graph and common features taken from the month start before it, then served the same way on Val. Same initial states and negative plans; 3 seeds.
- **Results:**

  | Model | Rolling (monthly) | vs existing model at T1 | vs existing model at T0 |
  |---|---|---|---|
  | bce_coreview | **0.010613** | **+0.000811 [0.000614, 0.001002]** (warm +0.0015, new −0.0005) | +0.002445 |
  | bce_text | 0.009304 | −0.000128 (n.s.; new −0.0030, warm +0.0013) | |

  - Co-review − text under rolling: **+0.001308 [0.001129, 0.001483]**.
- **Mechanism:** the knownness cue persists but sharpens. "Unknown" now means "released this month": an unknown slot is the target 6.1× as often, against 3.1× under T0 features.
- **Finding:**
  - Co-review needs a fresh graph in *both* training and serving; then it is the largest feature gain tested (+0.0013 over text).
  - For text-only models, fresher training mostly trades new-item quality for warm quality.
- **Caveats:** 3 seeds; selected epochs changed, so feature and epoch changes are confounded.

## 3. Is the co-review gain robust? (Track B)

**Question.** Last night the published co-review gain disappeared after a one-positive change. Averaged over training draws, does co-review help?

**What was run.**
- bce_text and bce_coreview with the same initial states and three new negative/refit draws, scored at T0 and T1.
- **Run:** `results/analysis/20261106-refit-draws`.

**Seed-0 results.**
- **Co-review − text, pooled over draws:**
  - T0: +0.000196 [0.000072, 0.000325];
  - T1: +0.000387 [0.000244, 0.000536]. Every T1 draw has its CI above zero.
- Warm targets gain (+0.0004 / +0.0007); new targets lose (−0.0002 / −0.0004).
- The draw-to-draw SD is about 0.0001.

**Final, 3 seeds × 4 draws (36 fits incl. draw 0)** (`final.md`), co-review − text nDCG@10:

| Features | Overall | New | Warm |
|---|---|---|---|
| T0 | **+0.000157 [0.000065, 0.000255]** | −0.000173 | +0.000299 |
| T1 | **+0.000492 [0.000375, 0.000609]** | −0.000232 | +0.000811 |

- Draw-to-draw SD within a seed is 0.00011–0.00017 overall and 0.0003–0.0004 on new targets.
- Recall@10, HR@3 and MRR@10 agree.

**Finding.** Averaged over training draws, co-review gives a **small, robust warm-item gain** with a small new-item cost, and the gain is about three times larger with period-start features. The published single-draw +0.000295 was within draw noise; last night's "disappearance" was one unlucky draw, not a bug. With rolling training it grows to +0.0013 (§2).

## 4. Leak-safe context features (Track D; P-b)

**Question.** Does leak-safe context improve the residual ranker beyond text?

**What was run.**
- 63 one-hot context columns: price decile; review count, rating and item age as of the period start; history length; time since the last interaction.
- 3 fits, attribution and a leak re-check.
- Runs: `results/analysis/20261106-context/{report,followup}`.

**Results.**
- **vs bce_text(T0):** +0.002730 [0.002478, 0.002988]; new +0.0094; warm −0.0006.
- **Same checkpoints with T0 statistics:** +0.000952.
- **Everything at T1 vs bce_text(T1):** **+0.001273 [0.001052, 0.001501]** (new +0.0023, warm +0.0006).
- **Attribution:** neutralising review count + rating + age removes the gain entirely. Price contributes a little; the user features contribute almost nothing.
- **Leak re-check: passes.**
  - No column can see the gap between first review and query time.
  - The 656 same-timestamp targets are indistinguishable from other new candidates.
  - The gain holds without them (+0.0092 new).

**Finding.** Context helps mainly through an explicit newness prior. Once the baseline is fair, the gain is modest but real.

**Caveat.** The 5-core filter, computed over the whole timeline, guarantees that every new product gets later reviews. That likely inflates any newness prior (§4.4 limitation).

## 5. Why did users switch to products neither retriever found? (Track F; P-e, P-f)

**Question.** The professor asked for a case study of 20–30 targets the baselines did not retrieve.

**What was run.**
- Frozen buckets, with 8 events drawn per bucket (seed 0).
- Scale counts with a frozen device-model regex dictionary.
- A no-training device-upgrade candidate source.
- A title-leak audit.
- Runs: `results/analysis/20261106-switch-cases`, `…-switch-cases-devices`, `…-device-upgrade`, `…-title-leak`.

**Results.**
- **Double misses:** 204,111 events (87.6%) are missed by both retrievers.
- **Leaf category change** from the last product: 0.839 among double misses vs 0.550 among retrieved events.
- **Device change:** 62.2% of model-naming double-miss targets name a model the history never names, vs 34.1% for retrieved events. Within the same family, newer beats older 74 : 26.
- **The retrievers echo devices the user already has:** about half of their top-10 slots name a model the user owns.
- **Device-upgrade pool** (recency top 180 + 20 newer-model items): Recall@200 +0.001402 overall; +0.013870 for targets naming a newer model; the top 10 is unchanged by construction.
- **Title leak:** crawl-time titles name future models in at least 8.5% (recency) / 20.9% (two-tower) of top-10 slots, and in 13.4% of targets. 23% of the device-upgrade rescues rely on such names.

**Finding.** Most of what the cascade misses is users changing device or product type, while text retrieval echoes the last purchase. A simple device-upgrade rule helps the targeted group a little, but part of its evidence rests on leaked names.

**Caveats.**
- Title mentions are not compatibility claims; the device-audit gate is still blocked.
- The case coding is model-assisted and needs human review.
- The title leak is a dataset limitation affecting every text model and Test.

## 6. Random mixing (Track C; P-a)

**Label.** Training-label change in an isolated study, pending team review; it never replaces the natural-list results.

**Question.** Forced lists taught rankers a shortcut. If missed targets are inserted at a random rank with that rank's score, so neither position nor score reveals them, does the shortcut disappear?

**What was run.**
- B lists missed by retrieval (294,438 of them) became mixed lists: the target replaces the candidate at a seeded uniform rank and takes its score.
- **M2** = natural lists plus 20,676 mixed lists, using the bce_text recipe with the same initial states; seeds 0/1/2.
- M1 (all mixed lists) was not run: it would take about 25–70 min per fit.
- **Run:** `results/analysis/20261106-random-mix`.

**Results.**
- **Diagnostics (AUC against each list's own negatives):**

  | Input | Natural positives | Inserted positives |
  |---|---|---|
  | Skip score | 0.684 | 0.501 (the cue is hidden) |
  | Cosine to history | 0.587 | **0.103** |
  | Score − cosine | 0.611 | **0.889** |

  The inserted-target cosine AUC stays at 0.09–0.11 in every insertion-rank decile.
- **Val nDCG@10:** M2 0.007061 vs bce_text 0.008141, **−0.001080 [−0.001278, −0.000881]**.
  - New targets: −0.001402. Warm targets: −0.000902.
  - Every metric falls, and every CI is below zero.
  - On new targets, M2 is no better than retrieval.
- **Funnel:** rescues are similar to bce_text (2,424 vs 2,454), but M2 demotes more retrieval top-10 hits (1,935 vs 1,568).
- **Shape of the scores:** neither model shows the forced-list U-shape. M2 flattens the slope (decile 1→10 gap 0.58 vs 1.20), and its scores for new candidates rise at the bottom ranks. That is consistent with a learned preference for low-similarity items.

**Finding.** Random mixing cannot remove contamination. A missed target is by construction textually unlike the user's history, so giving it a random rank's score creates a visible score/content mismatch. The shortcut moves from position/score to content similarity, and Val quality drops, most on new items. This answers the professor's and Sahej's suggestion: **inserted ground truth is identifiable through any feature that encodes retrieval similarity, so the clean remedy is natural lists (or more natural positives, as with the fused pool in §1), not smarter insertion.**

## 7. Retriever freshness (Track G2)

**Question.** Graph freshness helped every ranker. Does retriever freshness help too?

**What was run.**
- Two-tower (text) seed 0 retrained with the identical recipe on all data before T1 (the frozen model used data before T0). It selected epoch 39 after 63 minutes.
- The fresh retriever only *serves* Val candidates: P2' (its natural top 200) and P3' (fused with recency, same rule as P3).
- The frozen P2/P3 rankers score these lists.
- **Run:** `results/analysis/20261106-fresh-two-tower`. Pool and feature rebuilds reproduce the stored archives exactly.

**Results.**
- **Retriever, full-Val Recall@200:**
  - overall 0.0687 → **0.1113 (+62%)**;
  - warm 0.0874 → 0.1478;
  - products first reviewed in [T0, T1): ×2.4;
  - new-in-Val 0.0261 → 0.0315.
- **Retrieval-order nDCG@10:** P2' 0.010035, that is +0.004393 [0.004100, 0.004699] over P2. That roughly equals the old P2 ranker.
- **Frozen rankers:**
  - P3' ranker 0.011384 vs P3 ranker 0.011019: **+0.000365 [0.000160, 0.000569]** (new −0.0008, warm +0.0009);
  - P2' ranker vs P2 ranker: −0.000149 (n.s.), and no better than P2' retrieval order.

**Seed robustness (phase H2, 7 Nov).** Full-Val macro R@200 by retriever seed:

| Retriever | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| Frozen (pre-T0) | 0.0687 | 0.0697 | 0.0675 | 0.0686 (SD ≈ 0.0011) |
| Fresh (pre-T1) | 0.1113 | 0.1108 | — | 0.1111 |

- Fresh seed 1 selected epoch 35, with R@100 0.0753 and nDCG@10 0.0100, both identical to seed 0.
- Fresh seed 2 was stopped by the lead at epoch 8 because free disk reached its floor.
- **The +62% gain is about 40× the seed-to-seed spread.**
- **Paired-user CIs** (`results/analysis/20261107-fresh-tt-seeds`), fresh mean (seeds 0/1) − frozen mean (seeds 0/1/2):
  - R@200 **+0.042460 [0.041310, 0.043569]**;
  - nDCG@10 **+0.004321 [0.004039, 0.004597]** (+76%).
- By group:
  - warm: R@200 +0.0602 (+69%);
  - new-in-Val: R@200 +0.0053 (+20%), nDCG@10 +0.000145 [0.000023, 0.000261].

**Finding.** A fresher retriever is the largest *retrieval-stage* gain of the night. P3' gains +0.0024 in retrieval order, but only about 15% of that survives ranking with rankers trained on the stale retriever's lists.

**Caveats.**
- Two explanations can't be separated: the rankers were trained on old-retriever lists, and Val also chose the retriever's epoch.
- There is one retriever seed, and the device and code version differ from the frozen run.
- Retraining rankers on fresh-retriever lists for B would need a time-consistent (rolling) retriever. A retriever trained on all data before T1 has seen B's targets, so its B lists would be in-sample.

## 8. Which system wins for whom? (P-c, P-f)

**What was run.**
- Saved per-event Val ranks for 6 systems: P1 retrieval, P1 bce_text, P3 ranker, P3+context (T1), rolling bce_coreview, fresh P2' order. Each reproduces its published nDCG@10 to 1e-9.
- Frozen segments: history length, new/warm, warm popularity terciles (cut at 13 and 68 pre-T1 reviews), second-level category, and the user's activity gap.
- **Run:** `results/analysis/20261106-meso-segments` (tables.md, figure_meso.png).

**Results** (highest point estimate per segment; descriptive, not a selection).

| Segment | Best | Margin over 2nd |
|---|---|---|
| All | P3+context T1 (0.01217) | +0.00115, CI > 0 |
| Every history-length bucket | P3+context T1 | CI > 0 for 3+ items |
| New-in-Val | P3+context T1 ≈ rolling co-review | tie |
| Warm, most-reviewed third | **fresh P2' order (0.0453, about 3.7× P1 bce_text)** | +0.0129, CI > 0 |
| Warm, middle third | rolling co-review | +0.0032, CI > 0 |
| Warm, least-reviewed third | P1 bce_text ≈ P1 retrieval | tie; every newer system is ≤ P1 |
| Accessories / Cases & Holsters | fresh P2' / rolling co-review | CI > 0 |
| Users inactive > 90 days (58% of events) | P3+context T1 | CI > 0 |
| Users active within 90 days | rolling co-review | CI includes 0 |

**Finding.**
- **Product popularity, not user history length, separates the systems.** The fresh two-tower is effectively a popularity retriever: about 3.7× better on the most-reviewed products, near zero on rarely reviewed and new ones. Co-review suits mid-popularity products and cases, which are complementary to a phone.
- **No newer system helps the long tail.** That is where a per-product-age or popularity-aware source could pay off.
- **Caveat:** the winners are optimistic (22 rows × 6 systems, unadjusted, Val reused) and the systems differ in several factors at once. This motivates a separately registered study; it does not justify a router.

## 9. Phase H (7 Nov): fused pool × rolling co-review

**Question.** Do tonight's two strongest consistent effects combine: the fused pool P3, and co-review features refreshed monthly in both training and serving?

**What was run.**
- bce_coreview and bce_text (control) on P3, trained and served with rolling monthly snapshots (14 B snapshots); seeds 0/1/2.
- Same P3 plans and split, with initial states matching the P3 fits.
- **Run:** `results/analysis/20261107-pool-rolling` (report.md).

**Results.**

| System (Val nDCG@10) | Overall | New | Warm |
|---|---|---|---|
| P1 bce_text (T0), the current pipeline | 0.008141 | 0.008666 | 0.007887 |
| P3 bce_text (T1) | 0.011451 | 0.006751 | 0.013221 |
| P3 + context (all T1) | 0.012167 | 0.010815 | 0.012332 |
| P3 rolling bce_text (monthly) | 0.011545 | 0.006092 | 0.013791 |
| **P3 rolling bce_coreview (monthly)** | **0.013084** | 0.006890 | **0.015571** |

- **Against the other P3 and P1 systems:**
  - vs P3 bce_text (T1): **+0.001633 [0.001391, 0.001870]**;
  - vs P1 rolling bce_coreview: +0.002471 [0.002095, 0.002847];
  - vs P3 + context (T1): +0.000917 [0.000602, 0.001217], but new −0.0039.
- **Co-review − text within P3 rolling: +0.001539 [0.001337, 0.001748].**
- **Interaction:** pool × rolling co-review is **−0.000407 [−0.000688, −0.000142]**, so the gains are sub-additive.

**Finding.**
- The pre-stated hypothesis holds on all three counts.
- The highest train/serve-consistent system of the night is P3 + rolling co-review, at 0.013084: **+61% over the current pipeline**, all of it on warm products.
- Co-review's value is largest on the fused pool, where the two-tower adds warm, graph-connected candidates.
- New-item quality stays at P1 level or below. Context (§4) is what lifts new items, so the two are complementary. They were not combined, because time ran out.

**Caveat.** Serving the rolling model with T1 snapshots scores 0.013568. That is a mixed-staleness diagnostic that over-rewards newness, not a recommended setting.

## Professor's items → evidence

| Item | Evidence now | Status |
|---|---|---|
| P-a random mixing for the forced-list effect | §6: insertion stays identifiable via content similarity (AUC 0.10); quality −0.0011 | answered (label change, pending team review) |
| P-b metadata/context without contamination | §2 freshness and mechanism; §3 co-review robustness; §4 context | answered (exploratory) |
| P-c meso-level groups and secondary metrics | new/warm in every table; HR@3/MRR@10 in every run folder; funnels in §1 | answered |
| P-d oracle between two retrievers | §1 retriever oracle (+0.0054, upper bound) | answered |
| P-e case study of 20–30 unretrieved targets | §5 (24 cards + scale counts) | answered (needs human review of coding) |
| P-f explain why and customise per group | §1 (warm vs new trade-off by pool), §2 mechanism, §5 device change, §8 winners by popularity/category/activity | answered (descriptive) |

## Research decisions taken during the night
See the progress file for the full reasoning. The main ones:
- **Device change follow-ups:** follow up the case study at scale, then test a device-upgrade rule, then audit title leakage. The path was closed as a dataset limitation.
- **Freshness mechanism:** test the "unknown = new" mechanism directly, which led to the rolling-snapshot training study.
- **Context:** demand attribution, a leak re-check and consistent-snapshot scoring before accepting the context gain.
- **P3t:** drop it after its gate failed (lead error).
- **Pool gain:** decompose it, then test additivity.
- **Co-review:** extend refit-draws to seeds 1–2.

## Caveats
- Val has been reused throughout. Everything is exploratory and unadjusted for multiple comparisons.
- The pipeline is regenerated, not the teammates' bit-identical arrays.
- The pool overlap was seen before registration.
- Oracles are target-conditioned upper bounds.
- Track C changes training labels.
- HR@3 and MRR@10 are exploratory metrics.
- Case coding is model-assisted.
- Mixed-snapshot hybrids are off-distribution diagnostics.
- Dataset limitations: crawl-time titles and parent-ASIN grouping; the 5-core filter over the whole timeline.

## Decisions the team needs to make
1. **Lead with retrieval.** Should the poster's RQ2 story be the fused-pool result (+35%, ranker retrained), with the new-item trade-off?
2. **Feature dates.** Should Val/Test evaluation use features from the start of each period (the train/serve-consistent setting), and should the published T0-for-Val numbers be reported as stale?
3. **Co-review.** State it as "small warm gain that needs a fresh graph" once the 3-seed refit-draw result lands.
4. **Dataset limitations.** Add the title leak and the 5-core newness inflation next to the existing 5-core caveat.
5. **Random mixing.** Decide whether the Track C label change belongs in the report at all; it is pending team review.
