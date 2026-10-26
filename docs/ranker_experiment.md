# Ranker experiments: MLP, DIN and candidate-count diagnostics

## Current state (draft PR, 2026-10-26)

The matching-only MLP is the overall validation reference after feature discovery and three-seed replication. At K₁=200, end-to-end per-user macro nDCG@10 is **0.006843 ± 0.000233**, versus retrieval order **0.005642** and DIN **0.006436 ± 0.000499**. DIN improves new-item/tail ranking relative to MLP but loses on head items; the long-mixed-history hypothesis is inconclusive. These are exploratory results on one frozen retriever, with seed SD and paired-user uncertainty reported separately.

The K₁ sweep and score diagnostics are complete. **No final candidate count is selected:** the historical CI heuristic returned K50, but the updated evaluation plan requires an acceptable-loss margin δ and cost rule recorded before inspecting outcomes. No such margin was recorded for this sweep. K100 is the observed quality maximum; report the quality/cost trade-off without equivalence or cost-optimality claims. Saved reports preserve the historical heuristic output, not a final pipeline decision.

Training retains the original force-insertion/negative policy. Excluding retrieval rank/insertion markers and selecting on natural validation hits do not eliminate the score-related training mismatch. A controlled retraining comparison remains a separate decision. Ranker/cascade Test evaluation has not been run; retrieval/baseline Test results predate this work. Index experiments and full-cascade timing/final comparison remain pending.

Start with the [README](../README.md) for commands and [artifact guide](ranker_artifacts.md) for shared models, compact evidence and locally retained development files. The sections below are a chronological experiment record; early failed runs and next-step statements describe their date, not the current status.

**Final pre-merge verification, 2026-10-26:** the user authorised a final review, current tests and merge. Built the missing separately remapped sample fixtures from cached raw data and reused the existing full-catalogue text vectors for their items; the full-data experiment artifacts were unchanged. The current suite passes **66 tests with no skips**, including DIN algebra/masking/gradient checks and new run-directory/resume safety checks. Direct training now refuses existing run directories, and the incomplete CLI resume option is explicitly rejected. Independent code and artifact reviews found no blocker for merging this exploratory milestone; the research limitations above remain open.

## Historical: first real candidate-list run

## Question

Does the MLP improve ranking over the frozen two-tower retriever on the candidates the retriever returned? The primary validation checkpoint metric is per-user macro nDCG@10. Macro GAUC is reported beside it because it measures pairwise ordering across the candidate list, not just the top ten.

## Setup

- Frozen retriever: `two_tower_text/seed0`.
- Candidate window: top 200. Training targets are force-inserted to provide a positive for every training list; validation targets are force-inserted for bookkeeping but are excluded from conditional ranker evaluation when the retriever missed them.
- Training lists: 315,114; 50 uniformly sampled negatives per target list.
- Validation lists: 232,904; only 16,672 (7.16%) contain the target in retrieved top 200, so ranker-only metrics are conditional on retrieval success.
- Features: retrieval score, item/category/store/price and period-causal item statistics, recent user history, frozen text embeddings and context matches. Candidate rank is excluded because target rank is only known for a force-inserted positive.
- Checkpoint: epoch 1, selected by validation macro nDCG@10. Later epochs 2–4 did not improve it.
- Run: `results/ranker_mlp/20261021-causal-age-v2/`.
- Test split was not used.

## Validation result

| Metric | Retriever order (R1) | MLP reranker (R2) |
|---|---:|---:|
| Conditional macro nDCG@10 | 0.08282 | 0.05849 |
| Conditional macro GAUC | 0.62577 | 0.71725 |
| Conditional micro nDCG@10 | 0.08521 | 0.05825 |
| Conditional micro GAUC | 0.62899 | 0.71666 |
| End-to-end macro nDCG@10 | 0.00564 | 0.00263 |

Conditional metrics compare ranking only on the 16,672 lists where retrieval found the target. End-to-end nDCG counts every retrieval miss as zero. Thus the MLP improves pairwise discrimination (GAUC) but makes top-10 placement worse, and its end-to-end score is also lower than retrieval order on this validation split.

## Decision

The R2 > R1 go/no-go check **does not pass** for the chosen primary metric. Do not claim an MLP improvement or move this checkpoint into a final comparison as a winning ranker. The result is still informative: improving average positive-vs-negative ordering did not translate into better top-10 recommendations.

A plausible explanation is objective mismatch: the current pointwise BCE training samples 50 negatives uniformly, while macro nDCG@10 rewards getting the few highest positions right. This is a hypothesis, not yet established. A next ranker experiment should keep the retriever, split, candidate window and primary metric fixed while changing one training choice, for example a listwise softmax loss over each positive plus its sampled negatives or harder negatives. Compare against the saved R1 baseline and report both conditional and end-to-end outcomes.

## Validity note

An earlier run used a global first-interaction timestamp to derive item age, which leaked future information. That run is marked `INVALID.md`. A second causal-feature run exposed a fresh-run checkpoint initialization bug and is also marked invalid. The reported run uses item age computed only from interactions before each period and initializes the nDCG selector so the first valid epoch can be selected.


## 2026-10-24: completed feature ladder

All four cumulative feature setups completed full-data validation discovery with seed 0, using the same frozen retriever and candidate lists. Each setup starts from a fresh model; checkpoint selection remains conditional macro nDCG@10. Source snapshots, configurations, per-instance target ranks and comparisons are saved in `results/ranker_ladder/20261024-feature-ladder-seed0/` and the corresponding `results/ranker_mlp/20261024-feature-ladder-seed0-rung*/` directories.

| Setup | Conditional macro nDCG@10 | End-to-end macro nDCG@10 | Rescued | Demoted |
|---|---:|---:|---:|---:|
| Retriever order | 0.082816 | 0.005642 | — | — |
| Score-only MLP | 0.025049 | 0.001600 | 554 | 2,292 |
| + History/text matching | 0.102843 | 0.007105 | 2,533 | 1,707 |
| + Candidate attributes | 0.094674 | 0.006522 | 2,853 | 2,054 |
| + Historical item statistics (full model) | 0.056142 | 0.003914 | 1,745 | 2,249 |

The history/text matching rung is the discovery winner: end-to-end macro nDCG@10 is approximately 26% above retrieval order. Its 2,533 rescues versus 1,707 demotions give a net increase of 826 top-ten hits across all validation instances. Raw hit counts and macro nDCG have different weighting; both are reported. The full model's poorer result does not establish that historical statistics are always harmful; this is one cumulative feature order, training recipe and seed. The earlier full-model no-go therefore does not rule out MLP reranking: a reduced feature setup improves validation performance here.

The score diagnostic also shows a potential training-distribution problem: 276,321 of 315,114 training targets (87.69%) were force-inserted after retrieval missed them. Raw retrieval-score target-vs-alternative AUC is approximately 0.0804 across all training lists, versus 0.6528 on naturally retrieved training targets. On validation retrieval hits it is approximately 0.6290. The forced-positive policy may encourage a score-only model to learn a relationship that does not generalize to retrieved targets. This is a diagnostic association, not yet a causal explanation or an authorized change to the positive/negative definitions.

Next evidence needed: replicate shortlisted configurations across seeds, then use saved target ranks for item/user segment analysis and paired user bootstrap comparisons. No Test data was used for this ladder. These one-seed validation numbers are provisional.


## 2026-10-24: shortlist repetition and provisional groups

The user authorized repetitions and meso analysis. Four new full-data trainings are queued: matching and attributes for seeds 1 and 2. Seed 0 is reused from the completed discovery ladder. Training uses the same frozen retriever/candidate lists and the unchanged model/training implementation. `results/ranker_repeat/20261024-shortlist-repeat/status.json` records live job status; `meso.md`, `meso.json` and `group_gains.png` regenerate after each seed finishes.

The initial report uses seed 0 only. Its group findings are exploratory: both shortlisted rankers improve new-in-period and warm-tail nDCG relative to retriever order; candidate attributes improve new-item ranking further but reduce warm-head ranking. The matching-only setup has positive gains for history lengths 1–2, 3–5 and 6–10. For long mixed histories, the user-bootstrap interval includes zero. The long homogeneous group has only 248 cases from 138 users, so its apparent loss should be interpreted cautiously. These patterns must be reassessed after all three seeds finish.

Window-entry novelty is 37.57% of ranker-training instances and 32.96% of validation instances. These percentages are specific to eligible target cases with a nonempty prior history and describe targets whose items had no interaction before the relevant period start. No Test cases were analysed.

Checks before repetition: 28 passed, 13 skipped for absent separately preprocessed sample artefacts. Added meso/bootstrap checks: final suite 32 passed, 13 skipped. The full-data analysis asserts case/user alignment with the saved validation ranks and excludes all non-validation instances. The group plot was rendered and visually inspected. CIs describe paired-user sampling uncertainty; training-seed variation will be reported separately once repeats finish.


## Completed three-seed shortlist repetition

All four requested repeat trainings completed; both shortlisted configurations now have ranker seeds 0, 1 and 2 on identical validation cases and the same frozen seed-0 retriever candidates. The group report and plot have regenerated with all three seeds.

| Setup | End-to-end macro nDCG@10, mean ± sample seed SD | Relative change vs R1 |
|---|---:|---:|
| Retriever order | 0.005642 | — |
| History/text matching | 0.006843 ± 0.000233 | +21.3% |
| + Candidate attributes | 0.006551 ± 0.000290 | +16.1% |

Both models beat retrieval order in all three seeds overall. Matching has the higher mean, while attributes wins in seed 1; matching is not uniformly best on every seed. The paired-user CI for the mean matching gain over R1 is [0.000902, 0.001494]. The attributes-minus-matching difference is -0.000292, with paired-user CI [-0.000493, -0.000092]. These intervals describe user uncertainty conditional on the completed seed mean, with training-seed SD shown separately.

Meso findings after repetition: new-in-period and warm-tail gains are positive in all three seeds for both models, with positive user-bootstrap intervals. Warm-head improvement from matching is uncertain (positive in only two seeds, CI includes zero), while attributes underperforms R1 on warm-head items in all three seeds. History lengths 1–2, 3–5 and 6–10 show positive gains in all three seeds for both models. Long mixed histories have positive point estimates but intervals include zero. The homogeneous 11+ group has only 248 cases/138 users, with intervals spanning zero; do not claim a reliable loss for this group.

The matching model averages 2,476.3 rescues and 1,747.0 demotions, approximately 729 additional raw top-ten hits per seed. The attributes model averages more net additional hits, but lower nDCG: placement within the top ten and per-user weighting also matter.

These are exploratory validation findings on one frozen retriever, without multiplicity adjustment or causal claims. Test remains untouched. Full seed-specific and group results are in `results/ranker_repeat/20261024-shortlist-repeat/meso.json` and `meso.md`.


## Result figures

`python -m cascade.eval.plot_ranker_results` regenerates five paired-panel PNG/SVG figures and numeric CSVs from the completed saved results. Outputs are in `results/ranker_repeat/20261024-shortlist-repeat/figures/`, with source SHA-256 values in `manifest.json`. These are validation figures; no new training or Test evaluation was performed.

The figures cover (1) single-seed feature discovery and three-seed replication, (2) item/user group improvements with paired-user intervals, (3) per-instance retrieval coverage alongside conditional ranking quality, (4) rescues/demotions and per-user Recall@10, and (5) validation-window novelty and performance. Training-seed SD and user-bootstrap CIs are labelled separately. All figures were rendered and visually inspected; numeric summaries are checked against the underlying per-seed metrics.

## Next: candidate-specific history with DIN

Selected MLP reference: matching-only, all three seeds completed, end-to-end macro nDCG@10 0.006843 ± 0.000233. DIN uses the same feature groups, head, candidates, training recipe and checkpoint rule, replacing history averages with a candidate-conditioned weighted sum. Three fresh ranker seeds will be compared on validation. No Test evaluation is part of this step.

Example: with a history containing phone cases and chargers, the MLP builds one average history vector. DIN can assign different coefficients to those past products when scoring a case versus a charger. Those coefficients are learned signed values, not percentages. Whether this helps is measured; it is not assumed.

The selected MLP does not directly use candidate item/category identity in its head. DIN uses that identity inside its attention query, so this experiment includes that added candidate interaction as well as weighted pooling. Pool scale also changes from an average to an unnormalised sum. See the dated decision for interpretation limits and links to the [DIN paper](https://arxiv.org/abs/1706.06978) and [FuxiCTR reference](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/layers/attentions/target_attention.py). The reference source was inspected; the external package was not run. ReLU and the project training recipe are retained, so this is an adaptation, not a full paper reproduction.

Development run completed: `results/ranker_din/20261024-din-matching_sample/` (one epoch, fixed 2% users). It exercises training, full candidate ranking on naturally retrieved validation targets, checkpoint selection and saved ranks. These scores are excluded from research reports.

Historical full study command (the completed ID is reserved; use a fresh ID and a fresh development run when retraining):

```bash
.venv/bin/python -u -m cascade.pipelines.ranker_din_study \
  --candidate-dir data/processed/candidates/two_tower_text_seed0 \
  --device mps --run-id 20261024-din-matching \
  --development-run 20261024-din-matching_sample
```

Follow `results/ranker_din_study/20261024-din-matching/status.json` and `seed0.log` / `seed1.log` / `seed2.log`. Completed seeds trigger a separate validation comparison in that directory (`meso.json`, `meso.md`, `group_gains.png`). Partial reports list exactly which seeds are complete; CIs compare common ranker seeds with matching-only MLP. The long-mixed-history hypothesis and existing group definitions are fixed before full training; item/history/time groups and support sizes are all reported.

## 2026-10-25: completed DIN and four-system validation comparison

All DIN seeds 0/1/2 and their group reports completed. DIN end-to-end macro nDCG@10 is 0.006436 ± 0.000499 (sample seed SD), versus matching-only MLP 0.006843 ± 0.000233. DIN beats retrieval order in every seed and beats the corresponding MLP only in seed 2. The seed-mean DIN–MLP difference is -0.000406, paired-user 95% CI [-0.000578, -0.000227]. User CIs condition on the saved seed mean; they do not subsume training uncertainty.

Meso results versus MLP: DIN improves new-in-period targets (+0.001254, user CI [0.001013, 0.001501]) and warm-tail targets (+0.000223, [0.000048, 0.000409]), and reduces warm-head performance (-0.001673, [-0.001961, -0.001345]). Long mixed histories show no established benefit (-0.000250, [-0.000823, 0.000300]). Group intervals are exploratory and unadjusted. The attention hypothesis for long mixed histories is not supported by this comparison. Group macro scores have different user populations and must not be added to reconstruct the overall mean.

The complete R0–R3 report is `results/ranker_comparison/20261025-rq3/report.md` with structured output in `comparison.json`, numeric tables in `summary.csv` / `groups.csv`, aligned target ranks and raw per-query CPU timings. All four rows use identical exact top-200 validation candidates from the frozen retriever.

| System | End-to-end macro nDCG@10 | Macro Recall@10 | Conditional macro GAUC | CPU ranker total median / p95 ms |
|---|---:|---:|---:|---:|
| R0 popularity order | 0.003281 | 0.007270 | 0.586998 | 0.010 / 0.011 |
| R1 retrieval order | 0.005642 | 0.010446 | 0.625756 | 0.003 / 0.003 |
| R2 matching-only MLP | 0.006843 ± 0.000233 | 0.013868 ± 0.000252 | 0.711995 ± 0.001403 | 1.591 / 2.014 |
| R3 matching-only DIN | 0.006436 ± 0.000499 | 0.013629 ± 0.000582 | 0.743015 ± 0.005554 | 7.324 / 8.363 |

Accuracy ± values are sample SD across three ranker seeds. Timing is on Apple M1 Max, one CPU/BLAS thread, one query at a time, 1,000 common validation queries after 50 common warm-ups per run. Learned timing entries are means of the three seed-specific medians and p95s. Total includes the current feature builder, tensor conversion, scoring and ordering; retrieval, disk I/O, checkpoint loading and one-time feature tables are excluded. R1 is already ordered, so its ranker-stage work is only extracting the final list. These are validation development timings; final Test and full cascade latency remain pending. The full table retains preparation/model/ordering timing separately.

MLP improves nDCG by 21.3% relative to R1 and is approximately 4.6 times faster than DIN for this ranker-stage implementation. DIN has higher GAUC while lower top-10 nDCG, supporting the user's decision to select checkpoints by top-10 nDCG. This is an observed metric trade-off, not a causal claim about the loss. R2–R0 nDCG difference is 0.003562, user CI [0.003278, 0.003878]; R2–R1 is 0.001201, [0.000902, 0.001494].

Reproduce the development/full comparison and figures:

```bash
.venv/bin/python -m cascade.pipelines.ranker_comparison --run-id <new-id> --sample
.venv/bin/python -m cascade.pipelines.ranker_comparison --run-id <new-id>
.venv/bin/python -m cascade.eval.plot_ranker_comparison --run-id <new-id>
```

Run IDs are never overwritten. `configs/ranker/comparison.yaml` lists the six frozen checkpoints and timing protocol. The full run saves source/config/checkpoint hashes and case alignment. Current decision: retain matching-only MLP as the overall validation choice. Candidate-count/search sweeps and final Test remain separate next stages.

## 2026-10-25: candidate-count sweep protocol

The user approved evaluating K1=50/100/200/500 with all three selected matching-only MLP checkpoints frozen. `configs/cascade/k1_sweep.yaml` specifies the checkpoint paths, inference batch size, fixed feature contract, reproduction tolerance, bootstrap count, ranking-latency protocol and recommendation rule. The fixed 2% full-user development run `results/ranker_k1_sweep/20261025-k1_sample/` completed all three checkpoints, all four counts, CPU timing and group reporting. Its K200 target ranks exactly reproduce the corresponding saved subset ranks. Development values are not research results.

Full results go to `results/ranker_k1_sweep/20261025-k1/`. This stage uses saved exact retrieval prefixes and measures ranker-stage CPU latency; it does not measure full retrieval/cascade latency. The retriever remains frozen seed 0. MLP weights, feature preprocessing and training-score normalization are reused without fitting. Scores are computed once at K500 and reused at smaller prefixes because this MLP scores each pair independently of other candidates. Target retrieval misses remain zero end-to-end and excluded from conditional ranking metrics. K500 expands beyond the candidate pool used in ranker training.

The report includes a candidate-expansion decomposition: gains from newly retrieved targets and losses from added competitors on existing targets. Both contributions retain the same overall per-user weights and add to the overall nDCG difference. These contribution values are distinct from the separately reweighted meso group means. Item/history/time group definitions remain fixed, with group support and paired differences versus the observed best saved alongside the headline metrics.

The prespecified selection heuristic chooses the smallest K whose observed mean end-to-end macro nDCG lies inside the user-bootstrap interval of the best observed mean. This does not prove equivalence. The report shows the best mean, interval, eligible counts and paired gaps, retaining seed SD separately. See the dated decision for full interpretation limits.

```bash
.venv/bin/python -m cascade.pipelines.ranker_k1_sweep --run-id <new-id> --device mps --sample
.venv/bin/python -m cascade.pipelines.ranker_k1_sweep --run-id <new-id> --device mps
```

Outputs: `status.json`, source/config/checkpoint hashes, `seed{0,1,2}_ranks.npz` with ranks and AUCs by K, per-K raw timing queries/results, `sweep.json`, `summary.csv`, `groups.csv`, `report.md` and four PNG/SVG figure pairs. K200 reproduction is checked before interpreting the sweep, with changed-rank counts retained in the report.

### Completed full sweep

`results/ranker_k1_sweep/20261025-k1/` completed all three frozen MLP seeds on all 232,904 validation events / 124,416 users. K200 reproduces every saved target rank exactly for all three seeds: zero changed cases. No model was retrained and Test remains untouched.

| K1 | Retrieval Recall@K1 macro | End-to-end macro nDCG@10, mean ± seed SD | Macro Recall@10 | CPU ranker median / p95 ms |
|---:|---:|---:|---:|---:|
| 50 | 0.029751 | 0.008315 ± 0.000119 | 0.014825 | 0.662 / 0.986 |
| 100 | 0.045741 | 0.008363 ± 0.000109 | 0.015519 | 0.941 / 1.288 |
| 200 | 0.068714 | 0.006843 ± 0.000233 | 0.013868 | 1.594 / 2.288 |
| 500 | 0.117544 | 0.002210 ± 0.000095 | 0.004894 | 3.730 / 4.767 |

K100 has the highest mean in each ranker seed, although its gain over K50 is small (seed 1 gains only about 0.000004). The mean difference K100−K50 is 0.000047, paired-user 95% CI [-0.000145, 0.000241]. The best mean's bootstrap interval is [0.008019, 0.008687], so K50 and K100 satisfy the historical interval heuristic, whose output is K50. K50's CPU ranking median is about 30% below K100's. This does not establish equivalence; K100 also has higher Recall@10 and different group outcomes. **Superseded for final selection:** the updated evaluation plan requires a predeclared acceptable-loss margin δ; none was recorded before this sweep, so no final cost-optimal candidate count is claimed.

Both larger expansions reduce end-to-end nDCG in all three seeds despite increasing retrieval coverage. For 100→200, newly retrieved targets contribute +0.002251 macro nDCG while competition on already retrieved targets contributes -0.003771, net -0.001520. For 200→500, 11,903 additional targets enter retrieval; the model averages 929.3 new-target top-ten hits but 3,089.7 existing-target top-ten demotions. The nDCG contributions are +0.001757 and -0.006389, net -0.004633. These exactly additive contributions preserve the same user weights. The observation establishes the fixed-ranker competition trade-off, not why particular input features cause distractors to score highly. Training-distribution/objective explanations require further diagnostics.

Meso trade-off: K50 has the strongest warm-head point estimate; new-in-period and warm-tail point estimates peak at K200. Moving from K50 to K100 improves new-in-period nDCG by 0.000622 (exploratory user CI [0.000414, 0.000837]) and warm-tail by 0.000369 ([0.000194, 0.000562]), while warm-head falls by 0.000488 ([-0.000837, -0.000120]). Thus near-identical overall nDCG at K50/K100 hides opposing item-group changes. K200 improves new-in-period versus K100, while its warm-tail difference is uncertain. All history-length group point estimates are lower at K500 than K200. Group comparisons remain exploratory and unadjusted; their macro means do not add to the overall mean.

Retrieval coverage is measured by actual membership in each saved candidate prefix, matching the ranker's real opportunities. A source-artifact tie diagnostic found two K100 boundary events with target column 99 (rank 100 in the returned order) but pessimistic full-catalogue target rank 101. Therefore the actual event-level Recall@100 here is 0.047908151 versus 0.047899564 in the original metadata's full-rank summary. K50/K200/K500 have no membership/full-rank disagreement. This does not change candidate lists, ranking tie policy or ground truth; retain this distinction when comparing retrieval tables.

All four figure pairs were rendered and visually inspected: candidate-quality/time trade-off, failure breakdown, additive new-target/competition contributions and meso changes from K200. CPU timing uses Apple M1 Max, one thread, the same 1,000 validation queries and 50 warm-ups per K/seed. Retrieval and total cascade latency remain unmeasured by this stage; the timing columns are explicitly ranker-only. Detailed reports and raw data remain traceable to the saved source/config/checkpoint hashes.

## 2026-10-25: completed score and product diagnostics

The user approved explanation of the larger-pool loss. `results/ranker_diagnostics/20261025-score-diagnostic/` completed after a fixed 2% development run. It rescored 5,814 validation events: the union of all 4,845 unique events demoted from K200 top ten to outside K500 top ten in any MLP seed, plus 1,000 uniform random validation queries (with overlaps). Per-seed demotion event counts are 3,148 / 3,027 / 3,094. All six K200/K500 reproduction checks match saved ranks exactly, with zero differences. Weights, labels, training policy and Test are unchanged.

### Observed training/evaluation score mismatch

The full training audit reproduces the earlier raw-score diagnostic: 276,321 of 315,114 targets (87.69%) were force-added after retrieval missed them. Raw retrieval-score target-versus-alternative AUC is 0.080370 across all training lists, 0.652839 on naturally retrieved training positives, and approximately zero on forced positives. On naturally retrieved validation targets at K200 it is 0.628975. These are per-event AUCs; corresponding per-user values are 0.076231 / 0.649580 / approximately zero / 0.625756. Thus the overall constructed training task associates targets with lower retrieval scores, while naturally retrieved validation targets show the opposite pairwise direction. This is a distribution/label-construction observation, not yet a causal training-policy experiment.

### What the frozen MLP actually does

Across 1,000 uniform validation events, mean within-query Spearman correlations between raw retrieval score and MLP logit over all 500 candidates are -0.402 / -0.372 / -0.400 for seeds 0/1/2. Added positions 201–500 occupy 94.7% / 95.6% / 95.2% of stable-sorted top-ten slots. Rank-band mean logits rise strongly toward positions 401–500, after a low point in ranks 11–100. These are event-sample diagnostics, not macro accuracy or relevance labels for alternatives.

The prespecified one-input probe fixes the rank-100 candidate/history and varies only standardised retrieval score z=-3…3. At z=-3 versus z=+3, the lower score receives a higher logit in 99.9% / 99.7% / 100.0% of sampled events. A comparison within the same grid at z=-1 versus z=0 (raw scores approximately -0.259735 versus -0.179610) raises mean logit by 2.428 / 2.091 / 2.558, with the lower input favoured in 81.6% / 84.5% / 82.1% of events. The response is not globally monotonic: logits flatten near z=0…1 and increase modestly again toward z=3. Describe the strong low-score preference over this studied range, not a universal rule that every lower score is always better. This is an artificial feature intervention, potentially off the natural feature manifold, not a new ranking-accuracy result or calibrated probability.

Together, the inverse training signal, negative score association and direct frozen-network sensitivity support a retrieval-score mismatch explanation for the expansion loss. Establishing that force-insertion caused this learned response requires a separate controlled retraining comparison; none is run or silently introduced here.

### Meso profiles and real examples

Within all demotion events, added new-in-period items are 8.9% of available added candidate occurrences, 15.2% of overtakers and 27.9% of added top-ten slots. Warm-tail is 30.2% / 34.7% / 35.3%, and warm-head 60.9% / 50.1% / 36.9%. These are pooled seed–event–candidate occurrences; products and events repeat across seeds. Compare share against availability before describing overrepresentation. Warm-head alternatives still have the largest raw overtaker share. Leaf-category profiles show, for example, smartwatch bands at 6.2% availability versus 9.7% overtakers, while basic cases are 26.6% versus 25.6%; raw frequency alone is not enrichment.

The report includes one median-rank-loss seed-0 example per target group, selected before examining product titles. Event 26,725's Moto G fast target falls from MLP rank 4 to 55 with its logit unchanged at -2.5879. An added magnetic car mount from retrieval rank 418 scores 4.7203 and ranks first, alongside other added products above the target. The other representative target examples fall 4→50 (new-in-period case) and 4→55 (warm-tail screen protector). The last three prior product titles, exact target/competitor titles, ASINs, item groups, retrieval scores and MLP logits are saved. These products are unobserved alternatives, not proven disliked or irrelevant items.

Reproduction commands:

```bash
.venv/bin/python -m cascade.pipelines.ranker_diagnostics --run-id <new-id> --device mps --sample
.venv/bin/python -m cascade.pipelines.ranker_diagnostics --run-id <new-id> --device mps
```

Configuration: `configs/ranker/diagnostics.yaml`. Artifacts include `diagnostics.json`, `cohort_scores.npz`, `feature_probe.npz`, per-query correlations, rank-band summaries, demotion events and overtaker profiles, source/config/checkpoint hashes, `report.md`, and two PNG/SVG figure pairs. Both full figures were rendered and visually inspected. No extra unit-test suite or new model training was run in this diagnostic step.
