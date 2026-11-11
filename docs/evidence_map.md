# Poster claims and implementation evidence

Use this map to trace the submitted poster's RQ1–RQ3 and R1–R4 panels. Each row identifies its own evaluation and model family. Scores are per-user macro; nDCG is unscaled unless `×10⁻³` is stated. The [review guide](review_guide.md) explains histories, filtering, chronology, Test exposure and missing bulk artifacts.

## Claim map

| Panel | Comparison and scope | Owning results | Implementation |
|---|---|---|---|
| RQ1 retrieval | Validation; exact filtered retrieval; A-trained baselines and old two-tower, deterministic recency/mixed pools. Catalogue 111,480; 232,904 eligible queries. | [RQ1 summary](../results/rq1/summary.json), [pool report](../results/analysis/20261106-pools/pool_report.json). Old two-tower macro Recall@200 is about 6.9%; recency 5.5%; mixed 8.0%. | [report_rq1.py](../src/cascade/eval/report_rq1.py), [pools_build.py](../src/cascade/analysis/pools_build.py) |
| RQ2 candidate count | Validation; original seed-0 two-tower and three natural-list MLP arm-C seeds; trained at K200, scored at K50/100/200/500. | [K curve](../results/targeted_study/e3-20261107-all-seeds/k_curve.csv), [gain/competition decomposition](../results/targeted_study/e3-20261107-all-seeds/expansion.csv). K200→500: 0.008572→0.007923. | [ranker_policy_report.py](../src/cascade/eval/ranker_policy_report.py), [ranker_k1_sweep.py](../src/cascade/pipelines/ranker_k1_sweep.py) |
| RQ2 search settings | Separate 1,000-query Validation benchmark, residual `bce_text`, K200. Candidate overlap against exact search is not target recall. | [Quality](../results/steps_restoration/20261104/indices_quality.csv), [settings](../results/steps_restoration/20261104/indices_settings.csv), [latency](../results/steps_restoration/20261104/indices_latency_summary.csv). | [steps_restoration_benchmark.py](../src/cascade/pipelines/steps_restoration_benchmark.py) |
| RQ3 same-list ranker | Test; matching-feature November MLP/DIN fits, three seeds each. Compare model choices within one candidate pool. | Submitted [levels](../results/test_final_ranker/20261108/levels.csv); current corrected [levels](../results/test_final_ranker/20261108-normalization-corrected/matching_report/levels.csv) and [paired contrasts](../results/test_final_ranker/20261108-normalization-corrected/matching_report/comparisons.csv). Corrected mixed DIN 0.008268565; two-tower DIN 0.006369208. | [run_hybrid_study.py](../scripts/run_hybrid_study.py), [score_test_final.py](../scripts/score_test_final.py), [correction audit](../scripts/audit_hybrid_normalization.py), [rank-only report](../scripts/report_corrected_ranker.py) |
| R1 frozen rankers | Post-hoc Test follow-up; October natural MLP/DIN, three seeds, same frozen weights across list interventions. One old and one fresh retriever checkpoint, both seed 0. | [Levels](../results/analysis/20261109-frozen-ranker-followup/levels.csv), [contrasts](../results/analysis/20261109-frozen-ranker-followup/contrasts.csv), [macro recall](../results/analysis/20261109-frozen-ranker-followup/recall_in_list_test_macro.csv), [full tables](../results/analysis/20261109-frozen-ranker-followup/tables.md). | [frozen_ranker_followup.py](../src/cascade/pipelines/frozen_ranker_followup.py), [9 November protocol](test_protocol_2026-11-09.md) |
| R1 score-input mechanism | Validation only; fixed fresh candidates and frozen rankers, change score inputs. Fresh products plus old scores improve MLP/DIN over fresh scores. | [Score-swap levels](../results/analysis/20261109-frozen-ranker-followup/score_swap_val_levels.csv), [contrasts](../results/analysis/20261109-frozen-ranker-followup/score_swap_val_contrasts.csv), [checks](../results/analysis/20261109-frozen-ranker-followup/score_swap_val_checks.json). | `phase_score_swap_val` in [follow-up code](../src/cascade/pipelines/frozen_ranker_followup.py); [10 November predictions](score_swap_val_2026-11-10.md) |
| R2 retraining | The four bars are Validation, not Test. Residual recency-trained MLP, then mixed lists with frozen / size-matched / full-data retraining. The distinct Test confirmation is S6−S1. | [Size-matched tables](../results/analysis/20261107-pool-size-control/tables.md), [metrics](../results/analysis/20261107-pool-size-control/metrics.csv). Four values: 0.008141 / 0.008811 / 0.010387 / 0.011019. [Test contrast](../results/test_final/20261107/contrasts.csv): S6−S1 +0.001423. | [pool_ranking.py](../src/cascade/pipelines/pool_ranking.py), [pool_size_control.py](../src/cascade/pipelines/pool_size_control.py), [test_final.py](../src/cascade/pipelines/test_final.py) |
| R3 statistics timing | Test, residual recency-list text ranker; candidates, weights and training scaler fixed. T0/T1/T2 statistics produce 0.006439 / 0.007377 / 0.007815. Lag/column-swap diagnostics use their own registered contrasts. | [Final-system table](../results/test_final/20261107/tables.md), [timing report](../results/test_final/20261107/timing/README.md), [age/knownness](../results/test_final/20261107/timing/age_and_unknown.csv), [training cue](../results/analysis/20261106-freshness-mechanism/training_knownness.csv). | [statistics_timing_test.py](../src/cascade/analysis/statistics_timing_test.py), [freshness_mechanism.py](../src/cascade/analysis/freshness_mechanism.py); [timing addendum](test_protocol_2026-11-07_addendum_timing.md) |
| R4 combined system | Test; residual family. S8 combines mixed lists, monthly statistics and co-review. Matched reference S1 is recency/T0. Poster RQ3 matching-family baselines are separate fits. | [Test levels and paired contrasts](../results/test_final/20261107/tables.md): S1 0.006439→S8 0.010379 (+61.20%). [Paired comparison to R1's planned MLP](../results/analysis/20261109-frozen-ranker-followup/best_vs_planned_test.csv): ratio 2.0047. | [pool_rolling.py](../src/cascade/pipelines/pool_rolling.py), [rolling_ranking.py](../src/cascade/pipelines/rolling_ranking.py), [test_final.py](../src/cascade/pipelines/test_final.py); [7 November protocol](test_protocol_2026-11-07.md) |

## R1 exact reading

R1 evaluates 297,390 Test queries from 112,081 users at or after T2, not the August 2021 training period. The fresh retriever trains strictly before T1 on A+B. Its checkpoint was selected on Validation, then frozen for the Test follow-up. The rankers remain trained on the old A-only retriever's B lists.

| Test list | Macro Recall@K | MLP nDCG@10 ×10⁻³ | DIN nDCG@10 ×10⁻³ |
|---|---:|---:|---:|
| Original two-tower K200 | 4.3849% | 5.1772 | 6.2382 |
| Original two-tower K500 | 7.9984% | 4.7733 | 6.3882 |
| Fresh two-tower K200 | 6.9892% | 5.1126 | 5.9526 |
| Mixed K200 | 6.2434% | 6.4173 | 6.8337 |

The exact K500 recall increase is **82.41%**; the submitted poster prints +83%. Fresh recall rises **59.39%**, and mixed recall **42.38%**. These are relative changes, not percentage-point changes. Event-micro recall is different: fresh 3.928%→6.002%, rather than macro 4.385%→6.989%.

Fresh MLP's −1.25% nDCG change is inconclusive, with absolute interval [−0.000255440, +0.000119984]. Fresh DIN changes −4.58%, interval [−0.000515729, −0.000051445]. Mixed lists improve frozen MLP/DIN by 23.95%/9.55%; larger K slightly improves DIN. The broad statement that a better list cannot help without retraining is not supported.

The [top-ten count decomposition](../results/analysis/20261109-frozen-ranker-followup/top10_counts_by_list_change_test.csv) records event-micro counts averaged over three ranker seeds. It reports 638 previously successful MLP events displaced at K500, **not the poster's 838**; fresh 487 and mixed 102 are the other kept-target loss counts. Added-target successes and targets dropped from the list must also be counted; none of these counts alone determines macro nDCG.

## Checkpoint identities and availability

Paths below identify owning runs; wildcard paths are inventory patterns, not executable commands.

| System | Owning run pattern | Availability in this repository |
|---|---|---|
| Original two-tower | `results/retriever/two_tower_text/seed0` | Selected `model.pth`, config and results are tracked |
| Fresh two-tower used by R1 | `results/retriever/local/20261106-two-tower-text-all-train-seed0` | Training metadata and Validation ranks tracked; weights require original bulk artifacts or regeneration |
| R1 natural MLP | `results/ranker_policy/policy-pilot-20261029/arm_C_seed0`; `policy-replication-20261029/arm_C_seed{1,2}` | Checkpoints not supplied by the older sharing manifest; paths fixed in follow-up code |
| R1 natural DIN | `results/ranker_din/20261028-din-natural-seed{0,1,2}` | Checkpoints not supplied by the older sharing manifest |
| RQ3 two-tower MLP/DIN | `results/hybrid_study/hybrid-20261107-s{0,1,2}/{mlp_tt,din_tt}_seed*` | Aggregate metadata/results tracked; weights and per-event ranks local/cluster |
| RQ3 mixed MLP/DIN | `results/hybrid_study/rrf-20261108-s{0,1,2}/{mlp_rrf,din_rrf}_seed*` | Corrected aggregate report supplied; rescoring requires the bulk checkpoints/candidates |
| R2/R3 recency residual text | `results/local_experiments/20261103-full-promising-ranking/runs/bce_text/seed*` | Compact reports supplied; bulk arrays and weights excluded |
| R2 mixed residual text | `results/local_experiments/20261106-pool-ranking-P3/runs/bce_text/seed*` | Compact reports supplied; size-matched variant has a separate study |
| R4 mixed monthly co-review | `results/local_experiments/20261107-pool-rolling/runs/bce_coreview/seed*` | Compact reports supplied; bulk arrays and weights excluded |

Do not substitute the tracked October 24 checkpoints for R1's October 28/29 natural checkpoints. Likewise, the older 253-artifact manifest is not a complete inventory of these November systems.

## Executable entry points

These are the owning programs for inspection, not a sequential fresh-clone recipe. Inputs, output guards and saved protocols must be satisfied before execution.

| Operation | Entry point | Input requirement |
|---|---|---|
| Rebuild companion aggregates | `python3 poster-companion/export_data.py` | Committed source summaries only; updates derived companion bundles |
| Check companion evidence | `python3 -m unittest discover -s poster-companion -p 'test_*.py' -v` | Standard library and committed files |
| Recompute corrected matching-family table | `.venv/bin/python scripts/report_corrected_ranker.py --output FRESH_DIRECTORY` | Original/corrected per-event Test ranks and processed subgroup labels; no model execution |
| Audit corrected model scoring | `scripts/audit_hybrid_normalization.py` | Original checkpoints, candidates, data and saved ranks; preserve previous audit outputs |
| R1 candidates and frozen scoring | `cascade.pipelines.frozen_ranker_followup` phases `din-val`, `din-test`, `report` | Owning October checkpoints and candidate archives; Test phases require the locked protocol |
| Fresh retriever training | `cascade.pipelines.fresh_two_tower` phase `train` | Processed data and frozen text vectors; original run IDs are already occupied |
| R2 size-matched control | `cascade.pipelines.pool_size_control` | Parent residual fits, frozen pools, scalers and negative plans |
| R3 Test diagnostics | `cascade.analysis.statistics_timing_test` | Frozen residual weights, Test candidates and registered snapshots |
| R4 Test systems | `cascade.pipelines.test_final` | Registered parent fits, Test pools/features and owning protocol |

## Interpretation corrections for review

The current interpretation is learned dependence on the training inputs. Changed retrieval causally changes outputs for the fitted systems, but the direction is not uniformly negative. R1 score swaps narrow the input mechanism on reused Validation. They do not establish whether another temporal training protocol would have the same sensitivity.

R2's size-matched control shows that its gain is not explained by training-list count alone. It also changes the candidate/scoring setup, per-pool scaler and optimisation exposure, so the descriptive 23%/55%/22% decomposition is not a unique causal allocation.

R3's lag preference applies to the registered snapshot-trained residual systems. The stage split, long windows, frozen snapshots and all-time dataset filtering can contribute to that cue. Matching-only rankers have no direct review-count input.

R4 bundles several interventions; it cannot attribute the combined gain to one component. The roughly 2× ratio to a separate planned MLP baseline is identified separately from S8−S1 within the residual family. Historical Test exposure is disclosed in the review guide.
