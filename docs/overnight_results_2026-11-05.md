# Overnight rebuild and analysis results — 5 Nov 2026

**Val only, exploratory; awaiting team review.** Run 04:35–08:20 on Brandon's laptop (M2 Pro, 16 GB), orchestrated as a loop with subagents. Every run log and review note is in `docs/analysis_loop_2026-11-05.md`; the authorisation is in `docs/decisions.md` (2026-11-05). No Test file was opened, no model design was trained, and nothing (K, index, ranker, blend, δ) was selected.

## What was done and why

After the supervisor consultation and the PR #3 merge, the open questions were the professor's action items, applied to the team's current pipeline (plan.md §15: frozen MiniLM + 90-day-recency natural K200 candidates; BCE / pairwise rankers, with and without co-review features). That pipeline's bulk arrays were only on a teammate's machine, so the night had three parts:

1. **Rebuild (Track R):** regenerate the current pipeline on this laptop from our own processed data (byte-identical to the teammates' inputs), retraining the 15 registered ranker fits as a reproduction.
2. **Current-pipeline analyses (Track N):** oracle and serving-time blend, HR@3/MRR@10, and a traceable case study.
3. **Laptop-local evidence (Track L):** the forced-list U-shape split by popularity, and HR@3/MRR@10 for the archived pipelines.

## Findings, ranked by strength of evidence

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | The rebuilt pipeline reproduces retrieval exactly and three of five rankers within tolerance. **BCE's overall lead and every ranker's gain over retrieval are robust.** | Rebuild: 15 fits × reproduction table; two passing audits | Strong |
| 2 | **The co-review overall gain and pairwise's new-item lead are not robust to the training draw.** Both exist before the final refit and disappear after it in our regeneration. | Rebuild + inner-checkpoint diagnostic, paired-user CIs | Strong (for "not robust"); the cause is narrowed to the refit, not identified |
| 3 | **The forced-list U-shape is the retrieval-score shortcut acting within every popularity group, not a preference for unpopular products.** | U-shape split, 3 seeds, exact alignment | Strong |
| 4 | **A simple serving-time blend of BCE and pairwise does not deliver the new-item benefit; the target-conditioned oracle shows large headroom (+0.0019) that a blend cannot reach.** Blending mainly redistributes exposure between new and warm candidates. | Oracle + blend, weights chosen on tuning queries, paired CIs | Moderate |
| 5 | HR@3 and MRR@10 rank the systems the same way as nDCG@10, in the archived and current pipelines. Pairwise's new-item advantage over BCE is not significant at any metric in the regeneration. | Secondary metrics, reproduced saved nDCG to 1e-16 | Strong (confirmatory) |
| 6 | Event-level BCE vs pairwise differences are mostly seed-dependent (3 of 21 drawn cases hold in all seeds). | Case study, 28 events | Illustrative only |

---

## 1. Rebuild of the current pipeline (Track R)

**Question.** Can we regenerate the team's current pipeline here, and do its published Val results reproduce?

**What was run.**
- **Inputs:** `data/processed` `items`, `instances`, `item_segments` and `sequences` have exactly the sha256 recorded for the teammates' `data/processed/studies/20261027-targeted/`. `results/targeted_study/data.yaml` points at our folder.
- **Encoder:** a pinned encoder environment `.venv-encoder` (sentence-transformers 3.4.1 etc.) regenerated the MiniLM vectors; the revision 1110a243 matches.
- **Pipeline:** `frozen_text_study` prepare/evaluate → `full_data_ranking` prepare, 12 fits (4 methods × seeds 0/1/2) and evaluate → `objective_coreview` prepare, 3 fits and evaluate → local report and audit.
- **Outputs:** gitignored `results/local_experiments/` (~17 GB). Each folder has a `REGENERATED.md`.
- **Comparison table:** `results/logs/20261105/R3e_compare_published.txt`.

**Result.**

| Val, seed-mean nDCG@10 | Published | Regenerated | Difference | Verdict |
|---|---:|---:|---:|---|
| Frozen recency retrieval | 0.006284 | 0.006284 | exact (every digit; Recall@200 0.054746 too) | reproduced |
| BCE text | 0.008115 | 0.008141 | +0.000026 | reproduced |
| Pairwise text | 0.007767 | 0.007787 | +0.000019 | reproduced |
| Pairwise + co-review + metadata | 0.007930 | 0.007917 | −0.000013 | reproduced |
| Pairwise + co-review | 0.008065 | 0.007735 | −0.000331 | **not reproduced** |
| BCE + co-review | 0.008409 | 0.008167 | −0.000242 | **not reproduced** (seed 0, same epoch: 0.008357 vs 0.008364) |

Why the fits are not bit-identical:
- The text vectors differ by float noise only (max 1.6e-7). That moved one ranker-training positive across the K200 boundary (20,676 vs 20,675 natural positives), so every negative and refit plan is a different random draw.
- Initial weights match for all 15 fits.
- Val positives (14,916), the graph (1,877,548 pre-T0 edges), and the leaf/store/protocol/config hashes all match. Both local audits pass.

**Finding.** The rebuild is trustworthy: retrieval is exact, and the pipeline code, features and checks pass. The two co-review models are the only ones outside tolerance (see §2).

## 2. Is the co-review gain robust? (follow-up to §1)

**Question.** Is the co-review shortfall a feature bug, or training variance?

**Hypothesis (written before the diagnostic).** If the checkpoints chosen on the tuning data — before the final retraining on all of the ranker-training data — still show co-review > text on Val, then the reversal comes from that refit, not from the Val features.

**What was run.**
- Feature checks: batched graph features equal the reference builder (max error 7.5e-9); edge counts and metadata hashes match.
- The inner-selected, pre-refit checkpoints of all four factorial cells were scored on Val, post-lock and selecting nothing (`results/analysis/20261105-inner-checkpoints-on-val`).
- Paired user bootstrap: 1,000 draws, seed 0, seed-averaged.

**Result.**

| Co-review − text | Pre-refit checkpoints (overall) | Pre-refit (new) | After refit (overall) | After refit (new) | Published (overall / new) |
|---|---|---|---|---|---|
| Pairwise | **+0.000280 [0.000152, 0.000415]** | +0.000377 [0.000164, 0.000568] | −0.000052 [−0.000180, 0.000075] | −0.000180 [−0.000378, 0.000022] | +0.000298 / −0.000279 |
| BCE | **+0.000214 [0.000102, 0.000335]** | +0.000486 [0.000308, 0.000664] | +0.000026 [−0.000093, 0.000151] | −0.000416 [−0.000597, −0.000252] | +0.000295 / −0.000354 |

Other regenerated contrasts (published in brackets):
- pairwise − BCE, new items: **+0.000089, not significant** [+0.000455, significant];
- pairwise − BCE, overall: −0.000354 [−0.000347];
- BCE + co-review, new-item loss: −0.000416 [−0.000354]. Its direction reproduces.

**Finding.**
- The co-review features carry real signal: before the final refit they help, even on new products.
- Whether that gain survives the all-ranker-training refit depends on the random training draw. One changed positive was enough to remove it in our rebuild.
- So two published claims are **not robust to the training draw**:
  - "co-review improves overall quality" (+3.6%);
  - "pairwise leads on new products".
- Robust: BCE has the best overall point estimate, every ranker beats retrieval, and after refitting co-review costs new products under BCE.
- The refit draw is confounded with the prefix vs all-B data difference; this narrows the cause, not proves it.

## 3. The forced-list U-shape, split by candidate popularity (Track L1)

**Question.** The ranker trained on forced lists scores candidates in a U-shape along the retrieval rank. Is that because it prefers unpopular products, because of the retrieval-score shortcut, or both? (The professor's and Sahej's suggestion: connect the U-shape to popularity.)

**Hypothesis.** Arm A (all lists, forced positives) shows both effects; arm C (natural lists) shows neither.

**What was run.**
- `results/analysis/20261105-ushape-popularity`.
- Inputs: 50,000 random natural Val top-200 lists (seed 0) and policy-study arms A and C, seeds 0–2, re-scored with their frozen feature builder.
- Alignment check: recomputed ranks equal each checkpoint's saved ranks exactly.
- Candidate groups use only information before Val: new = no review before T1; head/tail = top 20% / rest by pre-T1 count.

**Result** (mean centred logit, seed mean):
- **A rises at the bottom of the list in every popularity group and seed.** Decile 10 minus decile 6: head +0.32 to +0.38, tail +0.42 to +0.52, new +0.44 to +0.54.
- C's curves fall monotonically (−0.04 to −0.11).
- **Tail is not preferred.** Tail − head is −0.07 to −0.13 in A. A's tail curve starts below head and crosses above it only at low ranks.
- **New products are preferred in both arms.** New − head is +0.67 to +0.89 in A and +0.27 to +0.37 in C.
- **Composition explains almost nothing.** Holding group shares fixed moves the curve by ≤ 0.03.
- Figure: `fig_ushape_popularity.png` (+ CSV).

**Finding.** The U-shape is the score/rank shortcut, applied within popular, tail and new products alike. It is not driven by which products sit at low ranks. Separately, both rankers favour new products, and the forced-list ranker does so about 2.4× more. That fits its stronger new-product results, but is not shown to cause them. This is the figure for the poster's methodological lesson box.

## 4. Oracle and serving-time blend of BCE and pairwise (Track N1)

**Question.** BCE is better on popular products and pairwise was better on new ones. How much could combining them recover? This is the professor's oracle suggestion, plus a version that could be served.

**Hypothesis (registered 04:58, before any score existed).** A candidate-level blend keeps BCE's warm quality, recovers part of pairwise's new-item advantage, and beats both overall by a small margin. The oracle gap is much larger than the blend gain.

**What was run** (`results/analysis/20261105-coreview-blend-oracle`).
- **(a) Target-conditioned oracles (upper bounds only).**
  - Event oracle: per event, the better of the two systems.
  - Group oracle: pairwise for new targets, BCE for warm.
- **(b) Candidate-level blend.**
  - Within each list, z-standardise each model's logits.
  - Score = (1−w)·z_BCE + w·z_PW, with w = b for new candidates and w = a for warm ones.
  - "New candidate" = no review before the period start, which is known at serving time. The code checks that no candidate's first review comes after the query.
- **Choosing (a, b).** A shared (a, b) was chosen on a 7×7 grid using only the 63,023 ranker-training tuning queries and each model's pre-refit selected-epoch logits.
  - Frozen at 06:56:38 (pair 2: 07:30:28).
  - The first Val read was at 06:57:32. I checked the code path: the choice step never loads Val logits.
- Paired user bootstrap: 1,000 draws, seed 0, seed-averaged.

**Result, BCE text + pairwise text** (blend a = 1/6, b = 1/2):

| nDCG@10 | All | New | Warm |
|---|---|---|---|
| BCE text | 0.008141 | 0.008666 | 0.007887 |
| Pairwise text | 0.007787 | 0.008754 | 0.007298 |
| **Blend − BCE** | +0.000037 [−0.000029, 0.000101] | **−0.000402 [−0.000566, −0.000246]** | **+0.000236 [0.000182, 0.000294]** |
| Group oracle − BCE | +0.000056 [−0.000032, 0.000144] | | |
| **Event oracle − BCE** | **+0.001963 [0.001864, 0.002070]** | +0.002424 | +0.001756 |

**Result, BCE + co-review + pairwise text** (a = 1/3, b = 1/2):
- Blend − BCE + co-review: overall **+0.000199 [0.000122, 0.000282]**, new +0.000211 [0.000059, 0.000351], warm +0.000188 [0.000094, 0.000279].
- This roughly equals the group oracle.
- It is tentative, because BCE + co-review only partly reproduced and has high seed variance (SD 0.000152). The blend may partly be smoothing its weak seed.

**Why the blend hurts new targets.**
- Mixing two z-scores shrinks score spread. New candidates already have compressed scores: within-list SD 0.45 (BCE) and 0.38 (pairwise), vs ≈ 1.0 for warm candidates.
- Giving new candidates more mixing (b = ½) clusters them nearer the list mean. Their top-10 share falls from 0.179 to 0.169.
- So b works as an exposure knob, not as "use pairwise for new products".
- Tuning chose this direction because "new" in the ranker-training period (since T0) fills 57% of top-10 slots, against 18% on Val. That population does not transfer.
- The Val-best point (descriptive and in-sample, not a valid choice) is the opposite direction (a = ½, b = 0).

**Finding.**
- The hypothesis is **not supported for the main pair**. A simple blend gives no reliable overall gain over BCE and moves quality from new to warm products.
- The event oracle shows large headroom: the two models win on different events, worth +0.0019 nDCG@10 if one could choose per event. A query-time rule this simple cannot capture it.
- Per the record's routing rules, no deployable routing gain is established.
- **The tuning-chosen weights fail to transfer, because "new product" means a different population in the ranker-training period than in Val.** This is a concrete reason to treat any new-vs-warm calibration knob with care.

## 5. HR@3 and MRR@10 (Tracks L2 and N2)

**Question.** Do shallow-rank metrics (does the user see the product in the top 3; how high is it) tell the same story as nDCG@10?

**Archived pipelines (L2, historical).**
- Run: `results/analysis/20261105-secondary-metrics-archived-r2`.
- Reproduction check: all 33 saved nDCG@10 / Recall@10 values reproduce from the same ranks (max diff 8e-17).
- HR@3 and MRR@10 order the systems the same way as nDCG@10:

  | System | HR@3 | MRR@10 |
  |---|---:|---:|
  | Popularity order | 0.00251 | 0.00208 |
  | Retrieval order | 0.00482 | 0.00419 |
  | Forced MLP | 0.00529 | 0.00473 |
  | Natural MLP | 0.00720 | 0.00639 |
  | Natural DIN, K_train 500 | 0.00858 | 0.00751 |

- Natural DIN K_train 500 − forced MLP, HR@3:
  - overall +0.00328 [0.00297, 0.00361];
  - head +0.00713 [0.00655, 0.00768];
  - new +0.00002 [−0.00031, 0.00037], so no longer a significant loss;
  - tail −0.00029 [−0.00056, −0.00003].
- Natural MLP − forced MLP still loses on new products: HR@3 −0.00055 [−0.00087, −0.00020].

**Current pipeline (N2,** `results/analysis/20261105-secondary-metrics-current-r2`**).** Seed means; co-review rows are flagged by their reproduction status.

| System | HR@3 all | MRR@10 all | HR@3 new | HR@3 warm |
|---|---:|---:|---:|---:|
| Popularity order | 0.002310 | 0.001936 | 0 | 0.003403 |
| Retrieval order | 0.005260 | 0.004703 | 0.006044 | 0.004762 |
| BCE text | 0.006787 | 0.005932 | 0.006758 | 0.006793 |
| Pairwise text | 0.006353 | 0.005629 | 0.007054 | 0.005965 |
| Pairwise + co-review + metadata | 0.006480 | 0.005727 | 0.006815 | 0.006232 |
| BCE + co-review (partly reproduced) | 0.006891 | 0.005953 | 0.006420 | 0.007066 |

Pairwise − BCE on new targets includes zero at every metric:
- nDCG@10 +0.000089 [−0.000187, 0.000330];
- HR@3 +0.000295 [−0.000053, 0.000631];
- MRR@10 +0.000167 [−0.000081, 0.000399].

Overall, pairwise is lower than BCE on every metric (HR@3 −0.000434 [−0.000636, −0.000225]). Every learned ranker beats retrieval overall on every metric.

**Finding.** The secondary metrics confirm the nDCG@10 conclusions. They close the archived part of the "Hit@3/MRR@10 unverified" status row.

## 6. Traceable case study (Track N3)

**Question.** What do individual retrieval misses, demotions and rescues look like in the current pipeline?

**What was run.**
- `results/analysis/20261105-case-study` (`cases.md` has 28 cards and a rubric table).
- The bucket rules and seeded draw were frozen at 04:52, before any case was seen; the draw itself ran at 07:01.
- 7 cases per bucket:

  | Bucket | Eligible events |
  |---|---:|
  | New-target retrieval miss | 70,715 |
  | Retrieval top-10 hit that BCE demotes and pairwise keeps | 342 |
  | BCE rescue from retrieval ranks 11–200 | 2,515 |
  | Counterexample: pairwise demotes, BCE keeps | 305 |

- The display uses seed 0, with target ranks recorded for all seeds.
- Rubric coding is model-assisted and needs human review. No device compatibility was checked.

**Result.**
- **Only 3 of the 21 BCE-vs-pairwise contrast cases meet their bucket rule in all three seeds** (0 of 7 in the demotion bucket).
- The contrast cases are mostly substitutes (17/21).
- Missed new products: in 6/7, the retrieval top 3 echo the store of the user's last product, while 6/7 targets are a different kind of product (e.g. a phone case after a Fitbit charger).
- Stable rescues are the same product type for a newer phone model.
- Counterexample 5991: for a Pixel 4a 5G user, pairwise's top 3 are Pixel 4a cases whose titles say "NOT Compatible with Pixel 4a 5G".

**Finding.** These cases illustrate the pattern; they don't prove it. Event-level differences between the two objectives are mostly seed noise, which fits §2's conclusion that the aggregate objective and co-review differences are fragile. The recency text query tends to echo the last purchase, which misses switches to a different kind of product.

---

## Closed paths

- **Co-review feature bug:** ruled out. The features equal the reference builder, the graph and metadata match, and seed 0 reproduces at the same epoch.
- **Popularity explanation of the U-shape:** ruled out. Composition moves the curve by ≤ 0.03.
- **Archived secondary metrics:** done; no further work needed.
- **Case study:** done; it is illustrative and should not be extended without a new question.

## Caveats

- **Val reuse:** everything is on Val, which the project has reused throughout. All results are exploratory and unadjusted.
- **Regenerated pipeline:** it is not bit-identical to the teammates'. It differs by one ranker-training positive and the order of tied candidates.
- **Diagnostic confound:** the inner-checkpoint diagnostic scores models trained on the prefix of B, so the refit draw is confounded with the data difference.
- **Oracle and blend:** the oracle is conditioned on the true target, so it is an upper bound only. The blend weights were chosen on tuning queries (see §4).
- **Exploratory metrics:** HR@3 and MRR@10 are extra metrics computed from the same ranks.
- **Case coding:** done by the model; it needs human review, and device compatibility was not audited.

## Unfinished or blocked

- All planned analyses finished by 08:00. The oracle/blend used both pairs, including BCE + co-review.
- Not done: more refit seeds to settle the co-review question (that would be new training); a human review of the case coding; a device-compatibility audit.
- No Test run, and none is needed for these questions. The team's locked Test (4 Nov) is unchanged.

## Decisions for the team

1. **How to report co-review and the pairwise new-item lead.** The published single-draw results don't survive a one-positive perturbation. Options:
   - report them as fragile;
   - rerun the registered fits with more refit seeds;
   - present the pre-refit evidence alongside.
2. **Poster lesson box:** use the U-shape-by-popularity figure (§3).
3. **Routing and blending:** don't present a blend or router as a gain. Present the oracle headroom (+0.0019) as evidence that BCE and pairwise win on different events. Note that a "new product" knob tuned on ranker-training data does not transfer to Val.
4. **A check to run:** 656 Val candidates have their first review exactly at the query time, and all 656 are the target. This is the cold-item admission rule (CLAUDE.md §4.6). Any feature that exposes item age or review count *relative to query time* would therefore leak the label. Check that no current feature does; the per-period statistics should not.
