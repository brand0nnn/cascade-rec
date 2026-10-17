# Evaluation and Failure Investigation Plan
Cascade recommender project (Group 7). This document covers what we investigate and how. It does not cover implementation.

## 0. Purpose and principles

**Purpose.** The project measures how retrieval choices limit the quality of the final ranked list. The professor's 14 Oct feedback asked for four things:
- fewer variables
- @10 as the headline metric, because the user sees 10 products
- an explanation of *why* results change
- one selected retriever–ranker combination tested end to end

**Principles.**
- **Zoom in only where needed.** Start from macro numbers, then locate the loss, then say for whom, then say why, then show examples.
- **No new model variables.** Everything after Step 1 reads existing results and explains them.
- **@10 is the headline.** Larger cut-offs (Recall@K₁) are used only to show the candidate-stage ceiling and to explain the @10 numbers.
- **Associations, not causes.** The explanatory steps describe what co-occurs with success or failure. They do not prove why.
- **Val for discovery, Test once.** Categories, thresholds and hypotheses are fixed on Val. Test is reported once. Nothing learned from Test changes a model, feature or definition.

### Overview of the five steps

| Step | Question | Scope |
|---|---|---|
| 1 | Which system wins? | Aggregate @10 metrics, with uncertainty |
| 2 | Which stage is to blame? | Five-way funnel per pipeline |
| 3 | For whom and for what does it fail? | Item and user segments (including history diversity) |
| 4 | What signals does it over- or under-trust? | Focused 3-signal profiles of candidates |
| 5 | Does a real case show the mechanism? | 3–5 hand-picked cases |

---

## Step 1 — Which system wins?

**What we investigate.** Whether each design choice improves the user-facing list, and by how much.

**Comparisons.**
- **RQ1, retrieval quality.** The two-tower retriever (`v2` text-only, `v4` text plus ID embeddings) against the four baselines used as retrievers (Popularity, ItemKNN, BPR-MF, SASRec), all trained on period A only so every row sees the same data.
- **RQ3, ranking skill.** On identical K₁ = 200 candidate lists: popularity order, retrieval-score order, MLP ranker, DIN ranker.
- **RQ2, cascade trade-off.** How the number of candidates passed on (K₁ ∈ {50, 100, 200, 500}) and the search setting (exact, HNSW, IVF) change final quality and latency.
- **Final test.** The selected pipeline against the best single-stage baseline (SASRec on A∪B) and against popularity retrieval plus the same ranker.

**How we measure.**
- **Headline:** Recall@10 and nDCG@10 (per-user macro average, mean ± std over 3 seeds).
- **Ceiling:** Recall@K₁, to show what the ranker could at most achieve.
- **Ranking skill alone:** conditional nDCG@10 (only instances where the target is in the candidate list) and GAUC.
- **Cost:** median and p95 latency.
- **Differences between systems:** paired bootstrap over users, reporting the 95% CI of the difference. A difference counts only if the CI excludes 0.

**What a finding looks like.** *"Pipeline X beats SASRec on nDCG@10 by Δ, CI [a, b]"* or *"K₁ beyond 200 does not improve nDCG@10 but raises latency."*

**Check built into the plan.** The MLP ranker must beat retrieval-score order on Val nDCG@10. If it does not, the cascade study is not meaningful and we stop before running the sweeps.

**Limits.** Step 1 says which system is better. It does not say why.

---

## Step 2 — Which stage is to blame?

**What we investigate.** When the final top 10 misses the true target, did retrieval or ranking fail, and does the ranker add value over the retriever's own order?

**Method.** Assign every instance to exactly one of five outcomes, per pipeline and per K₁ (where "retrieved at rank 1–10" refers to that pipeline's Stage-1 ordering: two-tower score order `R1` for `F2`, and Popularity order `R0` for `F1`):
1. **Retrieval miss.** The target is not among the K₁ candidates. Stage 1 is responsible, because the ranker never saw it.
2. **Ranking miss, unrescued.** The target was retrieved at rank 11…K₁ and the ranker left it outside the top 10.
3. **Ranking miss, demoted.** The target was retrieved at rank 1–10 and the ranker pushed it out of the top 10. Stage 2 actively hurt Stage 1.
4. **Retained hit.** The target was retrieved at rank 1–10 and stayed there.
5. **Rescued hit.** The target was retrieved at rank 11…K₁ and the ranker lifted it into the top 10.

**What we compare.**
- **Rescued hits against demoted misses.** If rescues clearly outnumber demotions, the ranker adds value. If they are close, the ranker mostly shuffles.
- **The retrieval-miss share against K₁.** This shows how much widening the funnel recovers, and whether any gain is then lost to ranking at longer lists. Rankers are trained on 200-candidate lists, so K₁ = 500 is a longer list than they have seen. Declining quality there is a result to note, not necessarily a bug.
- **For the index sweep,** the share of instances where the exact index would have retrieved the target but the approximate one did not.

**What a finding looks like.** *"At K₁ = 200, 52% of misses are retrieval misses, 30% unrescued and 18% demoted. DIN converts 6% of instances from miss to hit, while MLP demotes 4%."*

**Limits.** The funnel gives the stage, not the cause. A retrieval miss could come from a cold item, a short history, or a text mismatch. That is Steps 3 and 4.

---

## Step 3 — For whom and for what does it fail?

**What we investigate.** Where failures concentrate, so a mean metric does not hide a structural weakness.

**Axis A: item type (the cold-start question).** This matters most because 44.4% of Test targets are products with no interactions before the test period.
- **Segments:** `new-in-period` (use this on Val; on Test it equals `cold`), `tail`, `head`.
- **Expectations to check:**
  - ID-only baselines (Popularity, ItemKNN, BPR, SASRec) should do almost nothing on new items. Ties are broken pessimistically, so their recall on these is near zero.
  - `v2` and `v4` use item text, so they should retain recall on new items.
  - `v4` adds a learned ID signal for items seen in training. It should help head items and may crowd out new items.
  - **Ranker question:** because the ranker sees review count and item age, does it systematically demote new items in favour of established ones? Compare rescue and demotion rates for `new-in-period` targets against `head` targets.

**Axis B: user history length and category diversity** (`1–2`, `3–5`, `6–10`, `11+` prior purchases; with `11+` further split into **homogeneous** `1–2 distinct leaf categories` vs **mixed** `3+ distinct leaf categories`).
- **Short histories (`1–2`):** expect failures to be mostly at retrieval, from too little context.
- **Long, mixed histories (`11+` with `3+` distinct leaf categories):** expect failures at MLP ranking. A user who bought 12 phone cases has a homogeneous history where mean-pooling and target attention behave similarly; a user who bought across multiple categories has a mixed history where MLP mean-pooling blurs current intent. DIN's target attention is meant to address exactly this, so the tested claim is that DIN's rescues over the MLP are concentrated in **long, mixed histories**.

**How we measure.** The same metrics and the same five-way funnel, sliced by segment. For each model comparison (`v2` vs `v4`, MLP vs DIN), report where the difference is concentrated.

**What a finding looks like.** *"v4's gain over v2 appears only on head items; on new items v4 is X lower."* *"DIN rescues over MLP are concentrated in mixed histories of 11+ items (3+ leaf categories) and absent on new items."*

**Limits.** Segments are one axis at a time. They cannot show how multiple item-to-history relationships combine to drive a specific recommendation.

---

## Step 4 — What signals does it over- or under-trust? (signal profiles)

**What we investigate.** Which combinations of relationships between a user's history and a candidate product make the system rank it highly, and whether those same relationships predict what users really bought. Where the two diverge, the system trusts something users do not reward, or ignores something they do.

**The signal vocabulary.** For each candidate, record six simple yes/no relationships to the user's history:

| Signal | Meaning |
|---|---|
| `Co-buy` | Other training users reviewed both this candidate and an item in the user's history |
| `Same-device` | The candidate is for the same phone model as a history item (taken from the title; `"unknown"` tracked separately) |
| `Same-leaf` | Same leaf category as the user's last purchase |
| `Same-store` | Same store or brand as a history item |
| `Head item` | The candidate is among the top 20% most-interacted items |
| `Text-similar` | High text similarity to the history |

**Two focused 3-signal profile tables (to avoid a $2^6 = 64$ combination explosion).** Rather than crossing all six signals at once, we group candidates into two readable 8-row profile tables:
1. **Table 4A — Compatibility & Category Profile (`Same-device` × `Same-leaf` × `Text-similar`):** Tests whether the pipeline confuses substitutes (`Same-leaf = True`) for complements (`Same-leaf = False`) and whether high text similarity pulls in the wrong phone model (`Same-device = False`).
2. **Table 4B — Popularity & Collaborative Profile (`Head item` × `Co-buy` × `Same-leaf`):** Evaluated on warm targets (`pre_count > 0`) to test whether `Head item` + `Same-leaf` overrides specific collaborative links (`Co-buy`).

**Method and denominators.**
- **Compare slot share with target share (matched denominators).** We compare shares and not probabilities because a candidate is the target about 1 time in 200, so the two probabilities live on different scales:
  - *For Stage 1 (Retriever analysis, RQ1):* Compare a profile's share of the **$K_1$ retrieved candidate pool** against its share across **all true targets** (split by warm vs `new-in-period`), plus the profile distribution of missed targets vs retrieved items.
  - *For Stage 2 (Ranker analysis, RQ3):* Restrict to **instances where the target was retrieved (`ret_rank <= K₁`)**. Compare (a) the profile's share of **Top-10 ranked slots**, (b) its share of the **$K_1$ candidate pool offered to the ranker**, and (c) its share of **retrieved true targets**. If the ranker inflates a profile's share in the Top 10 far above its share in the $K_1$ candidate pool and true targets, it is a Stage 2 amplification effect.
- **Compare systems.** Run the Stage 2 profile tables for retrieval-score order (`R1`), MLP (`R2`) and DIN (`R3`) to show which signal bundles each ranker reweights.
- **Cold targets separately.** A new item cannot have a `Co-buy` or `Head item` signal, since no training user bought it. Report cold targets using Table 4A (text and attribute signals) only, and do not read absent `Co-buy` as a negative.
- **Optional summaries.**
  - A category-to-category transition view: what users buy after leaf category A against what the system recommends after A, to show self-loops such as recommending cases after a case.
  - A depth-3 decision tree that summarises the ranker's top-10 decisions, included only if we also report its fidelity (accuracy/AUC in reproducing the ranker's top-10 choices).

**Hypotheses to test (fixed on Val before Test).**

| Mechanism | Prediction |
|---|---|
| **Substitute trap** | `Same-leaf` + `Head item` candidates take many top-10 slots in Table 4B but are rarely the target (users seldom buy a second case right after a case) |
| **Wrong-device match** | In Table 4A, candidates with `Text-similar = True` + `Same-device = False` are over-recommended and almost never the target |
| **Popularity override** | A target with `Same-device = True` + `Co-buy = True` is displaced by a `Head item` lacking device match |
| **Cold-start suppression** | `Same-device` new items are under-ranked in Top-10 slots relative to their share in the $K_1$ candidate pool and true targets, because ranker features reward review count and age |
| **DIN attention** | DIN shifts Top-10 slot share toward `Same-device = True` + `Same-leaf = False` (complementary accessories) on warm items with mixed histories, but matches MLP on new items (whose item ID is a shared OOV vector in attention) |
| **`v4` vs `v2`** | `v4`'s learned IDs shift candidate-pool slots toward items with training history (`Head`/`Tail`) at the expense of `new-in-period` items |

**What a finding looks like.** *"In Table 4B (retrieved instances), `Same-leaf + Head` candidates fill X% of MLP top-10 slots, Y% of the K₁ = 200 candidate pool, and Z% of retrieved targets: the ranker amplifies a substitute bias that the retriever mostly lacks."*

**Limits (must be stated in the write-up).**
- **Association only.** A profile being over-recommended does not prove the model used that exact feature. Signals correlate (`Same-device` often co-occurs with `Same-leaf`; `Co-buy` tracks popularity). Report both single-signal marginal shares and the 3-signal profile combinations.
- **A divergence is not always a failure.** The other nine items shown may be fine products the user never reviewed. Non-targets are unobserved, not disliked. Findings are relative to observed reviews.
- **We only find what we chose to look for.** Always include the `"All false (No signal)"` row in Tables 4A and 4B to report the exact share of candidates and targets that no listed signal explains.
- **Coverage.** Extracting the device from titles will miss some products. Report regex coverage and treat `"device unknown"` as its own bucket.

---

## Step 5 — Does a real case show the mechanism?

**What we investigate.** Whether each pattern from Step 4 can be seen in a concrete example, so that the poster does not rest on percentages alone.

**Method.**
- Choose 3–5 cases, one per mechanism in Step 4, sampled systematically from the profile buckets flagged in Step 4 rather than cherry-picked by eye.
- For each case show the user's last 3–5 purchases, the true target, what `v2` retrieved, and what the MLP and DIN ranked at positions 1–3. Annotate each with its signal profile.
- Include at least one counter-example where the signal profile predicted a failure (e.g. a cold cross-category complement) yet the pipeline succeeded, so the examples are not only confirming ones.

**What a finding looks like.** One illustrated example per mechanism, such as *"the user bought an iPhone 13 Pro case; the MLP's top three were other iPhone 13 Pro cases (`Same-device + Same-leaf + Head`), while DIN rescued the target compatible screen protector (`Same-device + Different-leaf + Co-buy`)."*

**Limits.** Cases illustrate and do not prove. They carry no evidential weight beyond the counts in Step 4.

---

## Cross-cutting rules

**Validity**
- **Model-visible data only.** Any collaborative link (`Co-buy`) is built only from interactions the compared model could have seen: period A (`< T_AB`) for the A-only retrievers and baselines, A∪B (`< T1`) for the A∪B single-stage baseline and ranker analysis. Using the full pre-test graph for an A-only model would credit it with links it never saw.
- **Test discipline.** Hypotheses, signal definitions and thresholds are fixed on Val. Test is reported once. Nothing found on Test changes a model, feature or definition.
- **Replication.** A Step 3 or Step 4 pattern found on Val counts as evidence only if it replicates on Test. A pattern that does not replicate is reported as unstable.
- **Cold-item split.** Report `cold` or `new-in-period` targets separately everywhere, since they are 44.4% of Test and behave differently by construction.
- **Seed variation.** Report variation over seeds, so that a pattern is not one lucky run.
- **Honesty about unexplained failures.** For every explanatory table, include the `"no signal"` row showing the share of failures that no listed signal or segment accounts for.

### How the steps fit together

Step 1 gives the numbers. Step 2 says which stage lost the instance. Step 3 says for whom. Step 4 says what combination of signals the system relied on at the point of failure. Step 5 makes it visible. Each step narrows the previous one, and none introduces a new model.

### Summary of what each research question draws on

| Question | Primary steps | Explanation steps |
|---|---|---|
| **RQ1:** retriever vs baselines, text vs ID | 1 | 2, 3 (item type), 4 (Table 4A/4B of retrieved vs missed) |
| **RQ2:** K₁ and index setting | 1 | 2 (retrieval-miss share, index drops) |
| **RQ3:** ranker comparison | 1 | 2 (rescued vs demoted), 3 (history length × category diversity), 4 (Table 4A/4B slot vs pool vs target share) |
| **Final combination test** | 1 | 2, 3, 4 to explain a win or a loss |

If the final combination underperforms, that remains a valid result. The funnel, segments, and signal profiles explain it, with no further variants added.