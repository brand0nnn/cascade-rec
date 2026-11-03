# Architecture proposals: evidence, priorities and open decisions

Updated **4 November 2026** with the full-data follow-up; the original 1 November proposal review is retained below.

**4 November full-data follow-up:** after the deeper assessment, the user authorised the isolated replication in root plan §13. Twelve full-data fits are complete: BCE text/common control nDCG@10 **0.008115**, pairwise text **0.007767**, pairwise co-review **0.008065**, combined **0.007930**, on the same frozen-recency K200 pool. Pairwise remains stronger on new targets versus BCE but loses overall/warm; co-review's incremental gain shifts toward warm targets, and additional metadata loses overall/new quality. This qualifies the sample-based priority order below. Read [the governing full-data evidence](evaluations.md#full-data-promising-ranker-replication-4-november-2026). Historical proposal-only restrictions were superseded for that isolated study, without selecting canonical integration, routing/merging or a final Test system. No further experiment follows automatically.

**Status:** discussion and recommendations. This document does not approve model changes or replace [evaluations.md](evaluations.md), which governs evaluation and authorised follow-up. The frozen reference remains **Two-tower (text), seed 0 → matching MLP**, with DIN as the existing comparison. No new architecture has been implemented or trained by this review.

**2 November local extension:** the user subsequently authorised a separate experimental track in root [plan.md](../plan.md). Its three-seed sample metadata-neighbour encoder, two-interest retriever, real candidate fusion and natural-only semantic rankers are complete; none shows a clear gain over its matched control. These do not implement this document's proposed full-data ranker features or router. A simpler fixed-recency frozen-text control improves full-Val top-ten quality over archived trained retrieval, while lowering Recall@200; new-item gains coexist with warm-item losses. No recommender was trained for that full check. See [the current findings](evaluations.md#independent-local-experiments-2-november-2026). Historical proposal statuses below refer to the canonical full-data pipeline.

The Phase 1 local plan and its evidence checklist are complete, including the subsequently approved candidate-normalised training trial and a consolidated report. Actual-budget fusion beats graph alone but not clearly content. The Phase 1 local rankers demote more existing top-ten targets than they rescue. Source-bound evidence is in `results/local_experiments/20261102-plan-completion/report-final/`; root plan §9 and the evaluation record define closure. No full-data architecture integration, routing, changed-pool ranking or Test launch follows automatically.

**3 November sample ranking extension:** the user subsequently approved and completed root `plan.md` §10: text residual, explicit graph, BPR64, graph+BPR, PCA-text64, a learned self-only encoder and its matched metadata-neighbour encoder, three seeds each. This is separate from the canonical full-data proposal statuses below. All arms used the same strong frozen-text/recency pool, pre-T0 inputs, natural positives and chronological tuning. Every fit selected retrieval-preserving epoch 0 after two unsuccessful trained epochs. All selected outer-Val lists equal retrieval (nDCG@10 **0.012489**); no graph/latent ranking gain was demonstrated. This limits the current short BCE recipe rather than establishing that relational or latent signals cannot help. See [the evaluation record](evaluations.md#phase-2-graphlatent-ranking-completed-3-november-2026) and the local study's `report/report.md`. The subsequent objective comparison and conditional feature replication are completed in §1 below. Canonical integration, full-data training, routers/merges and Test remain separate decisions.

## 1. What we currently have

**Graph completion before assessment:** root plan §12 subsequently completed twelve fixed G0/GM/GC/GMC fits. Optional co-review features reach nDCG@10 **0.017728**, optional category/store-history fractions **0.016444**, both **0.018174**, and the text/common-control baseline **0.015566**. GC−GM has a positive overall interval; metadata added to GC and the factorial interaction are inconclusive. Existing matching/support inputs remain present, so these are additional-family comparisons. Frozen saved R6/R7 diagnostics show that the neighbour branch learns on inner cohorts despite its poorer final-refit Val result, without identifying a cause. Full evidence and limits are in [the completion record](evaluations.md#graph-family-completion-and-frozen-encoder-diagnostics-3-november-2026). Graph experiment execution is now paused for the deeper assessment; no additional architecture or integration is selected here.

**3 November objective follow-up:** the separately authorised root plan §11 is complete: 24 sample fits, exact fresh-BCE reproduction, an inner-only expansion gate and one outer-Val comparison. Pairwise text ranking reaches nDCG@10 **0.015566** versus retrieval/BCE **0.012489**; adding explicit graph/history scalars reaches **0.018174**, with R2−R1 **+0.002608 [0.001028, 0.004108]** and positive new/warm intervals versus R1. This family includes category/store-history shares alongside co-review inputs, so no co-review-specific mechanism is isolated. BPR on top is inconclusive; the learned neighbour encoder loses to the self-only control overall/new. Warm R2 remains slightly below retrieval in point estimate. The simpler scalar family is a useful robustness candidate, not a selected full-data architecture. See [the current evaluation](evaluations.md#ranking-objective-ablation-and-graphlatent-replication-3-november-2026). Canonical proposal statuses below remain separate; no full-data integration, router/merge or Test follows automatically.

The local analysis covers **232,904 validation events / 124,416 users**. Three-seed matching MLP reaches nDCG@10 **0.006843** at K200 versus retrieval order **0.005642**; the difference is **+0.001201, paired-user 95% CI [0.000902, 0.001494]**. DIN reaches **0.006436**. K100 has the highest observed MLP validation nDCG, **0.008363**, but K100-minus-K50 is **+0.000047 [-0.000145, 0.000241]**; no final K or acceptable-loss margin has been selected.

Absolute metrics need the right comparison. The harness evaluates full-catalogue scores with per-event availability/history filters, not 99 sampled alternatives. The eligible catalogue size varies by query. Sampling can change model comparisons; there is no supported universal “15–30×” conversion factor. See [Krichene & Rendle, KDD 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/). The catalogue contains 111,480 products, and **76,768 validation targets are new in the validation period**. New-in-period is distinct from unseen during retriever training: **152,195** validation targets are unseen before T0. An interaction-only model has no prior item-specific evidence for those products; a universal theoretical zero-recall claim is inappropriate across arbitrary feature-equipped models.

Retrieval misses remain the largest ceiling: only **16,672** targets enter the frozen natural K200 lists. The ranker's score mismatch is a strong diagnostic: **276,321/315,114 training targets (87.69%)** were force-added, and frozen-network score response favours low scores over the measured range. Controlled retraining is still needed to establish whether the training policy caused that response and whether a remedy improves natural-list ranking.

## 2. What the expanded graph evaluation establishes

Source: [expanded report](../results/evaluation_completion/20261101-kg-expanded-final/report.md). These are exploratory post-hoc associations, not internal model attributions. Differences below are nDCG@10, seed means followed by paired-user bootstrap (B=1000, seed=20261027); intervals are unadjusted.

| Validation slice | Events | MLP minus retrieval [95% CI] | DIN minus retrieval [95% CI] |
|---|---:|---|---|
| Warm, one other-reviewer witness | 17,836 | +0.002741 [0.001636, 0.003813] | +0.002412 [0.001269, 0.003570] |
| Warm, 6+ other-reviewer witnesses | 8,084 | -0.022474 [-0.027760, -0.017029] | -0.041376 [-0.046950, -0.035703] |
| Co-reviewed training-head, no same-store history | 30,525 | -0.006453 [-0.008211, -0.004790] | -0.011454 [-0.013217, -0.009806] |
| Co-reviewed training-head, same-store history | 5,186 | +0.022026 [0.016451, 0.027591] | +0.015481 [0.009524, 0.021227] |

The original 28,149-event head/co-review/different-last-leaf loss cell splits into **10,701 returning-category**, **13,980 novel-in-window**, and **3,468 unknown-category** events. Returning-category MLP loss is inconclusive; DIN loss occurs in both known continuity groups. All validation events partition into **44,157 immediate / 61,068 returning / 95,793 novel-in-window / 31,886 unknown** events. Unknown metadata stays separate, and novelty refers only to the observed last-50 window.

Saved-witness age also accompanies different outcomes. That archive retains the **smallest reviewer/history-item witness**, not the newest or representative witness. It cannot supply a recent-path share or establish phone obsolescence. Those quantities would require a new computation over every permitted path.

The implementation has four numeric matching inputs: standardised retrieval score, last-leaf equality, store-history match and text cosine. It also learns pooled history ID/category representations and a text projection; DIN adds candidate-conditioned attention. Thus it lacks explicit co-review counts and an explicit any-history category flag, but it is inaccurate to describe the model as having only four scalar inputs or no learned collaborative information. Graph counts are candidate features to test, not sufficient statistics equivalent to a learned GNN.

## 3. Recommendation and scope

Prioritise the existing controlled score-policy study before introducing graph inputs. It tests a diagnosed mismatch while preserving the frozen retrieval lists and current feature contract. If its result leaves a material graph-associated loss, consider one small, separately approved feature ablation. No accuracy improvement or additional latency is promised for any proposal.

| Proposal | Current state | Recommendation / dependency |
|---|---|---|
| T1: matching-MLP policy arms A/B/C/D | Implemented and authorised; no full-data remedy result | First experiment after original inputs are restored; execution remains user-managed |
| B1: ID-residual dropout | Implemented and authorised at p=0.5; no full-data result | Separate RQ1 control/treatment study; preserve the frozen text retriever for ranker comparisons |
| T2: graph/history feature group | Proposed; not implemented or approved | Consider after policy results; begin with a minimal non-device feature family |
| R1: nonnegative score residual | Proposed; not part of A/B/C/D | Separate architectural intervention, requiring its own baseline and decision |
| H1: target-conditioned oracle | Diagnostic concept; published draft estimates unverified | Recompute from aligned event metrics with full-population user weights before reporting a value |
| B2/H2: deployed retrieval merge or ranker router | Proposed; not approved | Needs complete candidate identities/scores, an inference-valid rule and fresh list evaluation |
| GIN / LightGCN / LLM GraphRAG | Outside current scope | Defer; current evidence does not require this expansion |

GIN is a different learned representation, not automatically a degenerate zero vector for isolated items: its update includes the node's own features. GIN and LightGCN are different architectures and should not share a claimed performance range. The [GIN paper](https://arxiv.org/abs/1810.00826) concerns expressive aggregation; it supplies no expected gain or guaranteed hub failure for this project. LLM extraction/reasoning would introduce additional model, audit and cost choices. Neither approach has been benchmarked here, so remove predicted GNN gains, GraphRAG latency ranges and claims of inevitable oversmoothing. An explicit graph-feature lookup is simply a graph feature, not an LLM GraphRAG system.

## 4. T1 and B1: preserve the authorised experiments

Tong's policy study is a fixed 2×2 matching-MLP experiment:

| Arm | Training lists | Retrieval-score input |
|---|---|---|
| A | All constructed lists | Enabled |
| B | All constructed lists | Uniformly masked |
| C | Natural-hit lists only | Enabled |
| D | Natural-hit lists only | Uniformly masked |

The natural subset contains **38,793 lists**. Fresh A is a reproduction/control arm, not a guarantee of exactly the historical three-seed mean. Shapes, initialisation within seed, normalisation, loss, negative sampling and checkpoint rule remain fixed. B−A and D−C test score availability; C−A also changes training support and optimisation exposure. Follow [the existing launch and acceptance protocol](evaluations.md#authorised-follow-up-controlled-ablations-not-results); do not add random score/rank mixing or a monotonic residual to these arms. The proposal's `scripts/slurm/retrain_ranker_policy.sh` does not exist; use the documented policy runner / `targeted_study.sbatch` commands.

Brandon's existing ID-dropout treatment is **p=0.5**, with a shared per-item mask per batch, inverted scaling, a dedicated RNG, and IDs retained at evaluation. It tests the co-adaptation hypothesis; it does not establish it in advance. Degree-gated IDs and a p=0.3–0.5 sweep are additional proposals. Neither changes the frozen text seed-0 candidate source used by the rankers.

## 5. T2: requirements for a future graph feature ablation

Potential inputs are any-history category continuity with knownness, known-category/store window shares, `log1p(other_reviewers)` and `log1p(witness_paths)`. Start with a small prespecified family against the selected score-policy control; do not add every relation at once. Tong owns scorer integration; Phua owns cutoff/missingness/evaluation checks. These are proposed responsibilities, not new task assignments.

Before implementation is approved:

- **All candidates:** compute identical features for every scored positive and unobserved alternative. `target_profiles.parquet` contains only true targets and must never be used as a target-only feature cache or label-dependent gate.
- **Causal snapshots:** ranker-training rows use review edges strictly before **T0=1595800947930**; validation rows use edges strictly before **T1=1628643414042**. The validation target profiles are pre-T1 evidence and cannot be copied into training. Histories remain the last 50 strictly prior items. No Test work is opened by this proposal.
- **Exact counts:** distinct reviewers are the candidate's reviewers intersected with the union of reviewers of distinct history items, excluding the query user and candidate-equals-history self-links. Path counts sum common-reviewer counts over distinct history items. Plain sparse multiplication counts paths and can count a reviewer repeatedly; it does not automatically count distinct witnesses. Avoid a dense item×item product.
- **Missingness:** category/store OOV IDs and padding must not count as semantic matches. Preserve knownness for new inputs. The historical last-leaf numeric feature compares mapped IDs directly, including equal OOV IDs; store features also use the training builder's rare-store mapping. These differ from raw metadata graph flags. Preserve that historical behaviour in A/B/C/D; any correction is a separate intervention.
- **Recency:** the archive's single witness cannot yield `recent_witness_share`. A training feature cannot count edges in [T0,T1), which are future evidence for those rows. Define any recency feature consistently at each permitted snapshot and recompute all required paths.
- **Interface:** `FEATURE_GROUPS` currently contains retrieval/matching/attributes/statistics, and both rankers use the existing four-column numeric contract. A proposed `graph` group needs explicit opt-in config/checkpoint metadata and unchanged historical loading/defaults. Keeping `numeric` at width four alone does not guarantee compatibility. Freeze source hashes per experiment and do not alter source during a policy run.
- **Device gate:** no device feature/evaluation integration before the frozen parser passes `assess_device_audit` on 100 labelled locked titles at precision ≥0.95. The current audit has **0/100 reviewed**; never tune on the locked set. Parsed-title coverage is not accuracy, and no measured MiniLM model-number confusion has been established locally.
- **Measurement:** restore original candidate matrices and frozen text vectors, pass development checks, then compare natural-list nDCG@10/Recall@10, retrieved coverage, churn, supports, paired-user intervals and actual feature+ranking cost. The proposed <0.3 ms lookup cost has not been measured.

## 6. Routing: diagnostic headroom versus a runnable ranking policy

A true target's category, degree or seen/new status can be computed after the event for analysis. A query-time router cannot know which candidate will become that target. Selecting each target's rank from its preferred scorer is therefore a **target-conditioned oracle diagnostic**, even if the relation for a specified candidate is available before ranking. It does not construct one ranked list. The draft values **0.075878 / 0.007442 / 0.00756** and its claimed latency saving are not verified deployable results. Slice means cannot be multiplied by event shares to recover user-macro performance; a recomputation must select aligned per-event metrics and retain the full user denominator. Selecting cell winners on the same validation data also creates selection optimism.

A practical router needs one of two explicit designs:

1. **Query-level gate:** choose one scorer from history/context available before seeing the target. Evaluate the complete list produced by that choice.
2. **Candidate-level mixture:** compute every candidate's relations and define how retrieval cosine, MLP logits and DIN logits become one comparable score/order. Sorting target ranks from separate scorers is invalid. Calibration, deduplication, ties and a fixed total K budget must be specified and evaluated on real candidate lists.

The proposed Stage-1 **140 seen-ID + 60 unseen-text** merge is a new budget allocation, not a consequence of a target-conditioned oracle. It needs actual restricted candidate lists, score/ordering rules, refill and the same filtering policy. Full-catalogue target ranks alone cannot reconstruct its recall. It is also multi-source retrieval merging, which current scope excludes. Report it only after a separate scope decision and full evaluation; a changed candidate source requires a separate cascade baseline.

## 7. Decisions still open

The immediate work is the authorised policy/dropout studies once their inputs are ready. Before further implementation, the team must decide whether to add a minimal graph feature experiment after those results, whether a monotonic residual merits its own arm, and whether to expand scope to an inference-valid router/merge. Final Test, index selection and full-cascade timing remain paused. The device gate and missing candidate/vector inputs remain explicit blockers.
