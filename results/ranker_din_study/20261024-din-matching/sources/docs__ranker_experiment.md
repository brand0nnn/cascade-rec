# MLP ranker: first real candidate-list run

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

Full study command:

```bash
.venv/bin/python -u -m cascade.pipelines.ranker_din_study \
  --candidate-dir data/processed/candidates/two_tower_text_seed0 \
  --device mps --run-id 20261024-din-matching \
  --development-run 20261024-din-matching_sample
```

Follow `results/ranker_din_study/20261024-din-matching/status.json` and `seed0.log` / `seed1.log` / `seed2.log`. Completed seeds trigger a separate validation comparison in that directory (`meso.json`, `meso.md`, `group_gains.png`). Partial reports list exactly which seeds are complete; CIs compare common ranker seeds with matching-only MLP. The long-mixed-history hypothesis and existing group definitions are fixed before full training; item/history/time groups and support sizes are all reported.
