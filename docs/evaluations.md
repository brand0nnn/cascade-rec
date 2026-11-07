# Evaluation findings, protocol and next experiments

Updated **6 November 2026**. This is the single current evaluation record for teammates and agents. Read this document for results, interpretation, progress, limitations and the authorised follow-up. Append future findings here. Other experiment notes and generated reports are historical evidence, not separate instructions.

## Current progress

The available local evaluation is complete: **232,904 validation events / 124,416 users**, using archived predictions and restored histories. The saved-validation replay reproduced the historical comparisons and checked source identities; it is analysis verification, not independent training replication. No new model training, Test predictions, SSH or cluster jobs were used for this completion.

| Work | State |
|---|---|
| RQ1 retrieval comparisons and text/ID trade-off | Complete from saved evidence; existing retrieval Test exposure disclosed |
| RQ3 rankers, uncertainty and five-way failure funnel | Complete on Val, frozen text seed-0 candidates |
| RQ2 exact-search K sweep and expansion losses | Complete on Val; no final K selected |
| Item/history/diversity/time groups | Complete on Val; sparse diversity contrast inconclusive |
| Graph target/history relations and 12 joint warm-target cells | Complete on Val; descriptive, not causal |
| Five cases | Target ranks, histories and retrieval examples complete; ranker top-three identities unavailable |
| Candidate-pool/top-ten profile amplification and device/text analysis | Blocked by original candidates/vectors and locked manual device audit |
| Controlled policy retraining and ID dropout | Full-data A/B/C/D seed-0 pilot and report complete (job 1080925); natural-only improves overall but loses new/tail quality; seeds 1/2 and ID dropout remain unrun |
| Approximate-index study, full-cascade timing and final F0/F1/F2 Test | Deferred by user |

**Working conclusion:** retrieval misses remain the largest ceiling, while the current ranker also displaces many strong retrieved targets. Historical matching-MLP gains are concentrated in new products. The new seed-0 controlled pilot improves overall nDCG with natural-only training but loses new/tail quality; score masking alone is worse. This is a trade-off, not a qualified remedy. Complete seeds 1/2 before integrating a policy or making a training-mechanism claim. The strongest located historical warm-target loss is concentrated among training-head, co-reviewed targets in a different leaf category from the last interaction.

### 2 November: meeting follow-up and development execution

The supplied meeting summary prioritises natural-list performance and head/tail/new-item trade-offs. Keep forced-list effects as pipeline diagnostics; the summary alone does not establish that held-out Test labels entered training. The existing A/B/C/D study is the immediate next experiment. Reuse the completed segment/funnel analysis, then extend representative cases to 20–30 systematically selected items. An oracle comparison is a diagnostic upper bound, pending an explicit choice of the two systems; DIN here is a ranker, not a retriever. Randomising candidate positions alone does not remove score/item-distribution differences associated with force-added targets. No new model or label policy was introduced.

Local execution on source commit `7e10c06`:

- `.venv/bin/python -m pytest -q`: **140 passed**.
- Ranker preflight: **ready**, no missing inputs; `val.npz` and `meta.json` match the frozen candidate identities. Record: `results/targeted_study/policy-20261102-preflight.json`. This supersedes the older missing-local-input inventory below for this checkout only; the cluster still needs its own check.
- `policy-20261102-development_sample`: all four arms completed, seed 0, one epoch, fixed 2% full-data user sample, CPU. Validation has 4,775 events / 2,553 users; A/B train on 6,224 lists and C/D on 730 natural-hit lists. All arms have the same initial-parameter hash. K200 checkpoint macro nDCG reproduces exactly in all four arms; C has one target-rank difference with no headline metric difference, retained in its anchor diagnostics.
- The report completed at `results/targeted_study/policy-20261102-development-report/report.md`, including group metrics, training support, funnel, expansion and score diagnostics. These are **development checks, not research results**; do not select a policy from their scores. No Test predictions were loaded. Original candidates and historical results were preserved.
- This session exposes neither CUDA nor MPS. Full-data pilot/replication have not run; cluster submission remains user-managed. The existing Slurm wrapper passes `bash -n`.

Next, on the cluster after syncing this source and checking its original inputs, run from the repository root:

```bash
.venv/bin/python -m cascade.pipelines.study_preflight --study-kind ranker --require-ready \
  --output results/targeted_study/policy-20261102-cluster-preflight.json
mkdir -p results/logs
sbatch --export=ALL,STUDY_KIND=ranker,STAGE=pilot,RUN_ID=policy-20261102-pilot scripts/slurm/targeted_study.sbatch
```

Use a fresh RUN_ID if that directory already exists on the cluster. The pilot trains A/B/C/D seed 0 and automatically writes `results/targeted_study/policy-20261102-pilot-pilot-report/report.md`. Review validity and runtime, then complete seeds 1/2 for all valid arms under the fixed protocol below. Final Test stays paused.

### 6 November: cluster pilot launched through xlogin

The user explicitly authorised SSH via `xlogin`, superseding the earlier user-managed-SSH restriction for this work. An isolated checkout of source commit `7e10c06` is at `/home/t/tjingyen/cp4285-policy-20261106/cascade-rec`. All 13 copied processed-input/candidate files match local SHA-256 hashes; the pinned environment was installed with Python 3.11.18 and PyTorch 2.5.1+cu124. No training recipe, split, filtering or metric was changed.

- **Active Slurm job: `1080925`**, running on `xgpi17`, NVIDIA H100 NVL. Requested one `h100-47` GPU, four CPUs, 32 GB host RAM, six hours in `gpu-long`. NVIDIA reports 95,830 MiB. A pending A100 request (`1080918`) was cancelled before execution; a generic-GPU attempt (`1080920`, `xgpe2`) failed its CUDA check before any model training. Its outputs are not research evidence.
- The active job passed **127 tests, with 13 sample-data-dependent skips**. The full local suite previously passed all 140. CUDA is confirmed usable on the H100.
- GPU development run `policy-20261106-gpu-development_sample` completed all four arms and reporting. Copied report: `results/targeted_study/policy-20261106-gpu-development-development-report/report.md`. These remain development-only checks, not full-data research findings.
- Full pilot `policy-20261106-pilot` passed preflight and entered arm A epoch 1 at approximately **00:13 SGT on 6 November**. At this observation its status was `running`, with no completed arms. No full-pilot result or policy selection is claimed yet. Seeds 1/2 were not submitted; Test remains paused.
- The wrapper `/home/t/tjingyen/cp4285-policy-20261106/run-pilot.sh` runs tests, GPU development/reporting, then the pilot/reporting sequentially with failure stopping the job. It does not submit another job. Environment versions and input hashes are recorded beside the checkout. Preflight's static `cluster_execution` string predates this authorised SSH execution; the actual job history is recorded here.

Inspect the active job and log:

```bash
ssh xlogin 'squeue -j 1080925'
ssh xlogin 'tail -n 40 /home/t/tjingyen/cp4285-policy-20261106/cascade-rec/results/logs/slurm-cascade-policy-pilot-1080925.out'
```

On successful completion, the full report will be at `/home/t/tjingyen/cp4285-policy-20261106/cascade-rec/results/targeted_study/policy-20261106-pilot-pilot-report/report.md`. Check all four arms and output identities, review runtime, then launch the separate seeds-1/2 phase on this same source/input snapshot. Do not rerun completed fits or infer a three-seed remedy from the pilot alone.

### 6 November: full-data seed-0 pilot complete

Slurm job `1080925` completed with exit code `0:0` in **1:16:49**. All four arms and reporting finished on the H100. Checkpoints, selected natural-Val prefix predictions and provenance were copied locally to `results/ranker_policy/policy-20261106-pilot/`; the report and CSV evidence are at `results/targeted_study/policy-20261106-pilot-pilot-report/`. All **26 declared arm-artifact hashes** match after transfer. K200 selected-checkpoint nDCG reproduces exactly in every arm, with zero changed target ranks. No Test predictions were loaded.

Full Val: 232,904 events / 124,416 users. Same natural frozen-retriever K200 lists and user-macro aggregation throughout:

| Arm | Training population / score input | nDCG@10 | Recall@10 | New nDCG@10 | Warm-head nDCG@10 | Warm-tail nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| A | All lists (forced when missed), score enabled | 0.007076 | 0.014252 | 0.003541 | 0.011637 | 0.001244 |
| B | All lists, score masked | 0.004216 | 0.009691 | 0.002404 | 0.006649 | 0.001029 |
| C | Natural-hit-only, score enabled | 0.008617 | 0.016233 | 0.002104 | 0.015925 | 0.000891 |
| D | Natural-hit-only, score masked | 0.007780 | 0.014961 | 0.002496 | 0.014043 | 0.000735 |

**Measured pilot trade-off:** C versus fresh A improves overall nDCG by **0.001541** (21.8%), paired-user 95% CI **[0.001244, 0.001819]**, while new-item nDCG falls by **0.001436**, CI **[-0.001769, -0.001106]** (40.6% relative loss). Warm-head improves by 0.004287, CI [0.003747, 0.004791]; warm-tail falls by 0.000353, CI [-0.000612, -0.000097]. B-A is -0.002860, CI [-0.003147, -0.002591]; masking score alone is not a remedy in this pilot. D-C is -0.000837, CI [-0.001053, -0.000609]. These are exploratory user-sampling intervals conditional on one trained seed, not training-seed replication.

**Expansion and score response:** A drops from K200 nDCG 0.007076 to K500 0.002400; C drops from 0.008617 to 0.007863. A's added-position-201–500 top-ten slot share on the common 1,000-event diagnostic cohort is 93.7%, versus C's 37.6%. Mean score/logit Spearman changes from -0.401 to +0.156. In the fixed-feature score probe, mean logit(z=-1) minus logit(z=0) changes from +2.672 to -0.207. These accompany the recipe change, but are not a unique causal explanation; interventions may be off the natural feature distribution. C still loses quality under K expansion.

**Support and interpretation:** A/B use 315,114 training lists (158,634 users); C/D use 38,793 lists (32,829 users). New-in-ranker-period target lists fall from 118,392 to 5,303. Natural-only changes supervised support and optimizer exposure as well as removing forced targets. The matched within-seed initialization checks pass. Selected epochs A/B/C/D are 11/15/5/8, respectively. Do not present C-A as an isolated contamination effect.

**Decision:** no policy qualifies for integration. All three seeds are required, and C/D also show a new-item loss against A in this pilot. Keep the reference unchanged. Next is seeds 1/2 for **all four valid arms**, without pruning negative arms or introducing another model. Replication has not been submitted in this status check; final Test remains paused.

## Working picture

The two-tower retrievers improve candidate retrieval over matched-period baselines, and the matching MLP improves the final top ten. However, increasing candidate coverage eventually makes the frozen ranker's output worse. The main issue to investigate is the interaction between training-list construction, retrieval score and candidate expansion.

The current reference is the frozen text seed-0 retriever with matching MLP. K100 has the highest observed validation quality; K50 is cheaper and nearly tied overall. Neither is a final selected configuration. More elaborate rankers/features change which products benefit without improving overall nDCG@10.

## Evidence and interpretation

### 1. Retrieval improves, but coverage remains the largest constraint

All four rows use the same retriever-training period. Values are per-user macro means; stochastic models average three training seeds.

| Retriever | Val Recall@200 | Existing Test Recall@200 | Val nDCG@10 |
|---|---:|---:|---:|
| Popularity | 0.026985 | 0.0162 | 0.001655 |
| SASRec | 0.040180 | 0.0235 | 0.003138 |
| Two-tower text | 0.068630 | 0.0438 | 0.005695 |
| Two-tower text + ID | 0.072370 | 0.0438 | 0.006445 |

Text improves Val Recall@200 by about 71% over matched-period SASRec. However, SASRec trained on all training data reaches Val Recall@200 of 0.066174 and nDCG@10 of 0.004791. Its extra training period closes most of the coverage gap. Keep the matched-training RQ1 comparison separate from the stronger full-data baseline used in the final comparison. Complete models and objectives differ; this is not an isolated text-feature ablation.

Text + ID improves overall Val Recall@200 by 0.003740, paired-user 95% CI [0.002731, 0.004751], but reduces new-in-period Recall@200 by 0.007363, CI [-0.008591, -0.006134]. The frozen text choice reflects a new-product priority, rather than the highest overall Val score. Do not infer how a text + ID cascade would perform: the ranker experiments use text seed-0 candidates.

About 33% of Val events and 44.4% of Test events target products new in that period. Existing retrieval Test results are substantially worse than Val. Product turnover is consistent with this difficulty, but does not establish the cause of the entire decline. Canonical Val cold=0 follows from defining cold using pre-Test counts; Val still contains new-in-period products.


### 2. Matching MLP improves @10; GAUC alone would select poorly

All systems below rank identical natural K200 candidates from the frozen text seed-0 retriever. Retrieval misses receive zero end-to-end credit.

| Ranker/order | Val nDCG@10 | Val Recall@10 | Natural-hit GAUC |
|---|---:|---:|---:|
| Popularity order | 0.003281 | 0.007270 | 0.586998 |
| Retrieval order | 0.005642 | 0.010446 | 0.625756 |
| Matching MLP | 0.006843 | 0.013868 | 0.711995 |
| MLP + attributes | 0.006551 | 0.014099 | 0.750953 |
| DIN matching | 0.006436 | 0.013629 | 0.743015 |

Matching MLP improves nDCG by 21.3% over retrieval order; the absolute improvement is 0.001201, CI [0.000902, 0.001494]. Attributes and DIN improve pairwise GAUC but reduce nDCG relative to matching MLP. Ordering a target above more alternatives does not guarantee better placement in the top ten. The attributes Recall@10 advantage over matching is inconclusive.

DIN-minus-MLP nDCG is -0.000406, CI [-0.000578, -0.000227]. DIN takes about 7.324 ms versus MLP's 1.591 ms in the original local CPU ranking benchmark, approximately 4.6 times as long. These timings exclude retrieval and are specific to this implementation and hardware.

Score-only and statistics feature-ladder results are single-seed discovery evidence. Their poor results do not establish that those features are universally harmful. DIN also changes candidate identity interactions and pooling scale; the comparison does not isolate attention alone.


### 3. Aggregate gains hide item-group trade-offs

| Target group | Retrieval nDCG@10 | Matching MLP | DIN |
|---|---:|---:|---:|
| New in Val period | 0.000361 | 0.002820 | 0.004074 |
| Warm head | 0.011290 | 0.011538 | 0.009865 |
| Warm tail | 0.000166 | 0.001175 | 0.001398 |

MLP substantially improves the placement of retrieved new/tail targets. DIN improves these groups further, but loses on warm-head products; the head loss outweighs the gains under the overall metric. The strongest observed DIN distinction is this item-group redistribution. A benefit for long, mixed histories is not established. The homogeneous long-history group has only 248 events / 138 users, limiting that comparison.

The analysis tests the mixed-minus-homogeneous contrast in the actual DIN-minus-MLP effect: -0.001106, 95% CI [-0.003928, 0.001753]. This does not establish that DIN helps mixed histories more. Comparing significance in the two groups separately would not answer that question.

Performance also declines across the validation timeline itself. Comparing days 0-89 with days 270+, retrieval-order nDCG falls from 0.007170 to 0.004273, matching MLP from 0.008360 to 0.005304, and DIN from 0.008426 to 0.004600. MLP remains above retrieval order in each reported window. Temporal robustness therefore deserves follow-up alongside the item-group trade-off. These windows contain different events/users and potentially different target mixes; the comparison does not isolate model ageing or product turnover as a cause.

Each group averages its own qualifying events within user and then its qualifying users. Group means cannot be added or simply weighted by event shares to recover the headline metric. Group intervals are exploratory and unadjusted.

### 4. The ranker rescues targets and destroys existing successes

At K200, 216,232 of 232,904 events miss retrieval entirely. The matching MLP averages 2,476 rescued targets, 1,747 demoted targets and 866 retained hits across seeds. These are means of raw event counts, not ensemble predictions or user-macro shares.

The net improvement is real, but reranking replaces many successful retrieval-order hits. DIN rescues more and demotes more. Rescue counts alone cannot explain nDCG: demotions and movement within the top ten also matter. The five-way funnel uses additive user-macro contributions: rescues +0.004898, demotions −0.003284 and retained-hit movement −0.000413, summing to +0.001201. DIN averages 2,672.7 rescues, 1,977 demotions and 636 retained hits.

### 5. Larger candidate pools expose the central failure

| K1 | Membership Recall@K1 | Final nDCG@10 | Final Recall@10 | CPU ranking median, ms |
|---|---:|---:|---:|---:|
| 50 | 0.029751 | 0.008315 | 0.014825 | 0.662 |
| 100 | 0.045741 | 0.008363 | 0.015519 | 0.941 |
| 200 | 0.068714 | 0.006843 | 0.013868 | 1.594 |
| 500 | 0.117544 | 0.002210 | 0.004894 | 3.730 |

From K200 to K500, 11,903 additional target events become retrievable, yet final nDCG falls about 68%. Additive contributions retain the same full-population user weights: +0.001757 from newly retrieved targets and -0.006389 from competition against previously retrieved targets, giving -0.004633 overall. This is measurable displacement, not just a change in the conditional metric's denominator.

K100 is the observed nDCG winner in all three seeds. K100-minus-K50 is only 0.000047, CI [-0.000145, 0.000241]; this is inconclusive, not equivalence. K50 is about 30% cheaper in ranking. Their similar aggregate quality hides opposing effects: K100 improves new/tail results while K50 favours warm-head products.

The sweep freezes weights and training preprocessing. K500 extends beyond the K200 training pool. This diagnoses the current pipeline, not an inherent disadvantage of larger candidate sets. No final K or cost-optimality claim follows: the historical CI-containment heuristic was superseded, and no acceptable-loss margin was recorded before these outcomes were seen.

### 6. The score mismatch is the leading explanation, with a causal gap

- 276,321 of 315,114 training targets (87.69%) were force-added after retrieval missed them.
- Raw retrieval-score target-versus-alternative AUC is 0.080370 over constructed training lists, 0.652839 on naturally retrieved training positives and 0.628975 on naturally retrieved Val targets. Forced positives have approximately zero AUC. The dominant constructed training population associates the positive label with lower scores, opposite the natural-hit relationship.
- In 1,000 uniform Val events, raw retrieval score and MLP logit have mean within-query correlations of about -0.4. Candidates added at positions 201-500 occupy 94.7-95.6% of final top-ten slots.
- Holding product/history fixed and varying only the retrieval-score input confirms strong low-score preference over the studied range. The response is not globally monotonic; artificial inputs may be off the natural feature distribution.

Together these support a training-score mismatch hypothesis and show the frozen network's sensitivity to that input. They do not prove that force-insertion caused the learned response. Establishing that requires a controlled retraining comparison. Added products are unobserved alternatives, not known dislikes or proven irrelevant recommendations.

Sources for sections 3-6: [saved evaluation and funnel](../results/saved_evaluation/20261027-saved-validation/report.md), [original K1 sweep](../results/ranker_k1_sweep/20261025-k1/report.md), [score diagnostics and representative examples](../results/saved_evaluation/20261027-saved-validation/saved_diagnostic_report.md).


### Historical failures and feature discovery

The first valid full-feature MLP (`20261021-causal-age-v2`, epoch 1) lowered end-to-end nDCG from 0.00564 to 0.00263 while increasing conditional GAUC from 0.62577 to 0.71725. Earlier runs with future-derived item age or a checkpoint-initialisation defect were invalid and are excluded. The subsequent seed-0 feature ladder gave end-to-end nDCG 0.001600 (score only), 0.007105 (+history/text matching), 0.006522 (+attributes), and 0.003914 (+statistics). These are cumulative feature additions, not isolated feature effects. Three-seed replication established matching as the overall reference. The early failed model therefore does not show that reranking always fails.

All-training baseline context (nDCG@10 / Recall@200): Popularity 0.001990 / 0.037444; ItemKNN 0.001328 / 0.015147; BPR 0.002614 / 0.041678; SASRec 0.004791 / 0.066174. Matched-period ItemKNN is 0.000615 / 0.008296; BPR is 0.001678 / 0.027263. These complete the baseline comparison without confusing training periods.

Unseen ID residuals are already zeroed. Random unseen-ID embeddings do not explain the text+ID new-item loss; co-adaptation to the ID signal is a hypothesis for the dropout study.

## Graph: target/history relationships


User-macro metrics; matching/DIN average archived seeds 0/1/2. Each slice has its own user denominator.

| Axis | Group | Events | Users | R@200 | Retrieval nDCG@10 | MLP nDCG@10 | DIN nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | all | 232904 | 124416 | 0.068714 | 0.005642 | 0.006843 | 0.006436 |
| canonical_segment | head | 138699 | 90160 | 0.103605 | 0.009099 | 0.010191 | 0.009353 |
| canonical_segment | tail | 94205 | 66116 | 0.015143 | 0.000218 | 0.001550 | 0.001937 |
| item_type | new-in-period | 76768 | 52420 | 0.026113 | 0.000361 | 0.002820 | 0.004074 |
| item_type | warm-head | 109765 | 75877 | 0.117994 | 0.011290 | 0.011538 | 0.009865 |
| item_type | warm-tail | 46371 | 37944 | 0.015294 | 0.000166 | 0.001175 | 0.001398 |
| seen_retriever_train | false | 152195 | 93332 | 0.027314 | 0.000612 | 0.003287 | 0.003881 |
| seen_retriever_train | true | 80709 | 58710 | 0.143379 | 0.014841 | 0.012942 | 0.010721 |
| seen_all_train | false | 76768 | 52420 | 0.026113 | 0.000361 | 0.002820 | 0.004074 |
| seen_all_train | true | 156136 | 96764 | 0.087426 | 0.007983 | 0.008488 | 0.007336 |
| same_leaf_last | false | 179834 | 106661 | 0.063384 | 0.005312 | 0.006131 | 0.005662 |
| same_leaf_last | true | 44157 | 34704 | 0.100768 | 0.008144 | 0.010895 | 0.010655 |
| same_leaf_last | unknown | 8913 | 7249 | 0.045334 | 0.004312 | 0.004339 | 0.004804 |
| same_store_history | false | 211804 | 118502 | 0.059457 | 0.004892 | 0.004814 | 0.004191 |
| same_store_history | true | 21037 | 17115 | 0.170828 | 0.014117 | 0.029559 | 0.031422 |
| same_store_history | unknown | 63 | 63 | 0.031746 | 0.000000 | 0.000000 | 0.000000 |
| co_review_retriever_train | false | 212022 | 117079 | 0.037596 | 0.001333 | 0.003677 | 0.004014 |
| co_review_retriever_train | true | 20882 | 17609 | 0.383501 | 0.050884 | 0.038107 | 0.030495 |
| co_review_all_train | false | 192341 | 110112 | 0.031024 | 0.000947 | 0.002521 | 0.003006 |
| co_review_all_train | true | 40563 | 32081 | 0.248766 | 0.028548 | 0.027057 | 0.022679 |

## Selected exploratory differences

These overlapping profiles were highlighted after the first profile report; intervals are unadjusted, not independent confirmations.

| Profile | Group | MLP minus retrieval nDCG@10 | Paired-user 95% CI |
|---|---|---:|---|
| seen_retriever_train | false | 0.002676 | [0.002457, 0.002906] |
| seen_retriever_train | true | -0.001898 | [-0.002583, -0.001168] |
| same_leaf_last | false | 0.000818 | [0.000516, 0.001126] |
| same_leaf_last | true | 0.002751 | [0.001937, 0.003584] |
| same_leaf_last | unknown | 0.000026 | [-0.001249, 0.001266] |
| same_store_history | false | -0.000078 | [-0.000362, 0.000178] |
| same_store_history | true | 0.015443 | [0.013608, 0.017219] |
| same_store_history | unknown | 0.000000 | [0.000000, 0.000000] |
| co_review_retriever_train | false | 0.002344 | [0.002132, 0.002557] |
| co_review_retriever_train | true | -0.012776 | [-0.015236, -0.010371] |
| co_review_all_train | false | 0.001574 | [0.001391, 0.001783] |
| co_review_all_train | true | -0.001491 | [-0.003041, -0.000107] |

## Joint warm-target profiles


Training-head means the top 20% of positive-count products **before T1**, with item-ID tie breaking. It is distinct from the canonical pre-T2 head/tail labels. Co-review requires another reviewer linking a history item and the target before T1. These rows restrict to targets already present before validation.

| Training-head | Co-review | Same leaf as last item | Events | Retrieval nDCG | MLP nDCG | DIN nDCG | MLP change interval |
|---|---|---|---:|---:|---:|---:|---|
| No | Yes | No | 3,431 | 0.000667 | 0.003326 | 0.003484 | [0.001360, 0.004141] |
| No | Yes | Yes | 1,215 | 0.001010 | 0.011029 | 0.012111 | [0.006040, 0.014610] |
| Yes | Yes | No | 28,149 | 0.030451 | 0.027776 | 0.022595 | [-0.004378, -0.001052] |
| Yes | Yes | Yes | 6,625 | 0.044928 | 0.042607 | 0.036329 | [-0.006588, 0.002004] |

All 12 known/unknown cells and their supports are in [warm_joint_target_profiles.csv](../results/evaluation_completion/20261028-final/warm_joint_target_profiles.csv), with seed-specific funnel/at-risk rates in [warm_joint_funnel.csv](../results/evaluation_completion/20261028-final/warm_joint_funnel.csv). They partition all 156,136 warm events. This completes the available **target-outcome** part of Step 4B; candidate slot amplification is still unavailable.

Each slice has its own user denominator. Its mean difference is not its contribution to the overall difference: `overall_mlp_contribution` keeps the full-population user weights, and can even have a different sign. Warm contributions plus the new-target contribution reconcile exactly. Intervals use 1,000 paired-user draws, seed 0; all profile comparisons are exploratory and unadjusted.

Stage-1 target ranks here use the reconciled pessimistic full-catalogue convention, capped to K200 for actual members. The earlier graph report uses historical returned positions; the saved reconciliation identifies two nDCG-event differences and no top-ten hit-count change. No candidate membership was changed.

## Retrieval-side profile comparison

Both retrievers use the same training period, three seeds and exact catalogue evaluation. Co-review below uses the earlier retriever-training cutoff, so later evidence is not attributed to the retriever.

| Target profile | Text Recall@200 | Text + ID | Difference | Paired-user 95% CI |
|---|---:|---:|---:|---|
| Co-reviewed before retriever cutoff | 0.385771 | 0.427421 | +0.041650 | [0.036650, 0.046658] |
| No co-review before retriever cutoff | 0.037265 | 0.037273 | +0.000008 | [-0.000819, 0.000865] |
| Seen in retriever training | 0.144203 | 0.164591 | +0.020388 | [0.018392, 0.022273] |
| Unseen in retriever training | 0.026653 | 0.021166 | -0.005488 | [-0.006412, -0.004523] |
| Store occurs in history | 0.168174 | 0.130217 | -0.037957 | [-0.042232, -0.033478] |
| New in validation | 0.026315 | 0.018952 | -0.007363 | [-0.008591, -0.006134] |

Full nDCG/Recall values, supports and uncertainties: [retriever_target_profiles.csv](../results/evaluation_completion/20261028-final/retriever_target_profiles.csv). Marginal groups overlap and are not independent confirmations.

## Five representative cases

The five cases below record target identity, last 3–5 prior products, full-catalogue retrieval rank, actual K200 membership, seed-specific MLP/DIN target ranks, known profiles and the available retrieval top three. First three cases retain their earlier seeded selection; the last two use stated buckets, sorted IDs and seed 0. No extreme titles were selected manually.

| Case | Retrieval target rank | MLP seed 0 | DIN seed 0 | What it illustrates |
|---|---:|---:|---:|---|
| SUPCASE Galaxy A71 case | 29 | 3 | 5 | A co-reviewed target can be rescued; history includes the same store. |
| Yootech wireless charger | 9 | 26 | 18 | Strong collaborative evidence does not prevent demotion. |
| TOPGO cup-holder mount | 1,146 | Not retrieved | Not retrieved | A co-review path does not ensure retrieval; reranking cannot repair this miss. |
| New Anker charger | 74 | 10 | 26 | A new target can benefit, but the hit is seed-dependent. |
| New i-Blason case | 6 | 34 | 12 | Same store/category do not guarantee preservation of an existing hit. |

**Step 5 limitation:** MLP/DIN top-three product identities are unavailable without original candidate-level inputs/outputs. The report says so rather than reconstructing products from target ranks. Titles are crawl-time metadata, and compatibility is not independently established. The historical score-diagnostic examples retain actual added overtakers for their separately selected demotion cohort.


**Contribution to the overall gain:** new targets +0.0007941783; warm targets +0.0004065793; total +0.0012007576. These retain full-population user weights, unlike group means.

Case details (all rank lists are seeds 0/1/2; target IDs make each example traceable):

- Instance **2241582**, SUPCASE Galaxy A71 case `B08H8HZ3J3`: prior SUPCASE Tab 4, HTC One M9 and Galaxy S7 cases. Retrieval 29; MLP 3/5/12; DIN 5/13/9. Same store and pre-T1 co-review, different leaf: a rescue counterexample.
- Instance **1777200**, Yootech charger `B07P9S6R9Y`: history includes Fitbit protectors/bands, an Apple Watch band and an iPhone 11 case. Retrieval 9; MLP 26/32/32; DIN 18/37/47. A hit demoted by every ranker seed despite co-review evidence.
- Instance **1405134**, TOPGO mount `B0BTBQ9D9B`: history includes watch accessories, a charging stand and stylus tips. Retrieval 1,146; absent from K200 for every ranker. Retrieval top three are watch-related products. Reranking cannot rescue it.
- Instance **2328439**, new Anker charger `B0B4DN7W4Z`: history includes earlier Anker chargers, with an Apple Watch charger last. Retrieval 74; MLP 10/18/14; DIN 26/16/24. Same-store overlap accompanies a seed-dependent rescue.
- Instance **1814991**, new i-Blason case `B09CTCHDL9`: history includes an older i-Blason case and iPhone 13 Pro Max cases. Retrieval 6; MLP 34/8/38; DIN 12/9/8. Same-store/category overlap does not guarantee hit preservation.

The graph contains 111,480 products, 367,388 pre-T1 reviewers, 2,208,880 pre-T1 review edges (1,877,548 before the retriever cutoff), and 305,632 metadata edges. Title-parser statuses: parsed 51,678; unknown 45,905; ambiguous 9,684; universal 4,189; negated 24. The 200-title audit is split into 100 development and 100 locked titles; gold labels remain blank. Parser coverage is not precision, and no device-compatibility conclusion is established.

## Evaluation protocol



#### Step 1 — Which system wins?

**Comparisons.**
- **RQ1, retrieval quality.** Two-tower (text) and Two-tower (text + ID) against Popularity, ItemKNN, BPR-MF and SASRec, all trained on the retriever-training period only, with exact search. The retriever choice is already recorded in `decisions.md`; this is not a new selection exercise.
- **RQ3, ranking skill.** At K₁ = 200, exact search, identical natural candidate lists: R0 = popularity order, R1 = retrieval-score order, R2 = MLP, R3 = DIN.
- **RQ2, cascade trade-off.** Train each ranker once per planned seed and freeze it. The K₁ sweep uses the frozen MLP at K₁ ∈ {50, 100, 200, 500} with exact search. The separate index sweep holds K₁ = 200 and the MLP fixed: exact, two HNSW settings and two IVF settings from `CLAUDE.md` §8–10. This is not a full K₁ × index × ranker grid. Check that HNSW `efSearch` covers the actual number of results requested, including filtering buffer; record effective settings.
- **Final comparison.** F0 = SASRec trained on all training data; F1 = popularity retrieval plus the selected ranker; F2 = frozen two-tower retrieval plus the selected ranker. F1 reuses a ranker trained on two-tower candidates, so report its candidate/feature distribution shift; it is not a separately optimised popularity cascade.

**Metrics and populations.** All systems use the shared harness's filtering and pessimistic score-tie policy. Align by instance ID; cases a model cannot score count as misses in end-to-end comparisons.
- **Headline:** Recall@10 and nDCG@10 over all eligible instances, first averaged within user, then across users. Retrieval misses contribute zero. Also retain micro averages for diagnostics.
- **Ceiling:** Recall@K₁, the maximum attainable Recall@10 for that candidate set under perfect ranking. Verify target membership in saved candidates and report any boundary-tie discrepancy with full-catalogue rank.
- **Conditional nDCG@10:** the same metric restricted to natural candidate lists containing the target; average within user over qualifying instances, then over qualifying users. Report qualifying users/instances and retrieval coverage. R0–R3 share this subset in RQ3. Different retrievers/K₁ settings may select different subsets, so conditional scores alone cannot establish superior ranking across those settings.
- **Natural-list GAUC:** for each list containing its target, compute the fraction of non-target candidates scoring below the target, with ties worth 0.5; average within user, then across users (`eval/metrics.py`). Exclude padding. Lists without a target have no positive and are excluded from GAUC, not from end-to-end metrics. State the pool and coverage. Forced-list loss/GAUC, if retained, is a separately labelled diagnostic, not evidence of end-to-end ranking quality.
- **Cost:** median and p95 of user encoding, search/filtering, feature preparation/ranker scoring and total. Follow `CLAUDE.md` §5: same recorded CPU, single thread, one query at a time, 50 warm-ups and 1,000 fixed random instances. Use Val for selection and a fixed Test sample for final timing; retain query IDs and sampling seed. Time actual candidate processing, including filtering/refill if needed.

**Uncertainty.**
- Report mean ± std over training seeds {0, 1, 2} for stochastic models, with actual seed IDs and per-seed values. Popularity/ItemKNN are deterministic and need one run. For the cascade, ranker seeds vary while retriever seed 0 and its candidates stay fixed: this measures ranker variation conditional on that retriever, not full-pipeline training variation.
- Use the existing paired bootstrap over users: 1,000 resamples, bootstrap seed 0, 95% percentile CI of the difference. For seed-mean comparisons, average each instance's metric over the stated training seeds first, then compute paired user means and resample users. The CI measures user-sampling uncertainty conditional on those trained runs; it does not include retriever retraining uncertainty.
- Report effect sizes and intervals. A CI excluding zero supports a directional difference; a CI including zero is inconclusive, not proof of equivalence or no improvement. Mark Step 3/4 profile tests as exploratory; do not interpret many unadjusted intervals as independent confirmations. Freeze any small confirmatory hypothesis family and its multiplicity treatment on Val before Test.

**MLP checkpoint, not a success requirement.** Compare R2 with R1 on unforced Val nDCG@10 and inspect Recall@10 too. If R2 does not improve it, first check feature/label leakage, alignment, filtering and training. If the result persists, report that reranking did not improve the observed task; the study remains meaningful. The team decides whether expensive sweeps still answer a useful question. No new variants are added merely to obtain a win.

**Final selection (Val only).** Compare MLP/DIN at K₁ = 200 using Val nDCG@10; report uncertainty and any inconclusive choice explicitly. Use the MLP sweeps to shortlist retrieval settings, then validate the actual selected ranker at the shortlisted setting on Val before freezing F2. For a new selection study, the team must record before inspecting its outcomes an acceptable absolute nDCG@10 loss margin δ and the cost/tie-break rule in `decisions.md`. A lower-cost setting is acceptable only if the upper paired-CI bound on its loss relative to the stated reference is ≤ δ. CI overlap alone does not establish acceptable loss. Until δ is recorded, report the quality/latency trade-off without claiming equivalence or a selected cost-optimal setting. Record all selected run IDs, seeds, candidate hashes and settings before final Test reporting.

**Example finding.** “Pipeline X changes nDCG@10 by Δ, CI [a, b].” Or: “The K₁ = 500 gain is inconclusive and latency is higher,” rather than “there is no gain.”

---

### Step 2 — Where is the target lost?

**Method.** Assign every instance to exactly one outcome for each pipeline/K₁. Use actual candidate membership for retrieval; use the shared score-tie policy for Stage-1 and final target ranks. Stage-1 ordering is two-tower score order for F2 and popularity order for F1. Log boundary ties so membership/rank conventions can be checked.

1. **Retrieval miss:** target absent from the K₁ candidates.
2. **Ranking miss, unrescued:** target present, Stage-1 rank >10, final rank >10.
3. **Ranking miss, demoted:** target present, Stage-1 rank ≤10, final rank >10.
4. **Retained hit:** target present, Stage-1 rank ≤10, final rank ≤10.
5. **Rescued hit:** target present, Stage-1 rank >10, final rank ≤10.

**Denominators and interpretation.**
- Report raw counts and user-macro shares over all eligible instances. The five macro shares sum to one. If also reporting percentages of misses, label that different denominator; do not mix it with percentages of all instances.
- With the same instances and user weights, **ΔRecall@10 = macro rescued share − macro demoted share**. Raw rescue counts alone need not explain the headline macro metric.
- For nDCG, compute each instance's final-minus-Stage-1 nDCG@10. Within each outcome, zero out other outcomes but keep the full user/instance denominator; macro-average these contributions so they sum to total ΔnDCG@10. Include retained hits moving within the top 10: rescues outnumbering demotions does not guarantee improved nDCG.
- Also report rescue rate among target-present Stage-1 misses and demotion rate among Stage-1 hits. Use these at-risk denominators when comparing segments.
- Across K₁, show which retrieval misses become available and whether ranking converts them into hits. Rankers trained on top-200 lists face a changed candidate distribution at K₁ = 500; report this caveat.
- For approximate search, report overlap with exact filtered top-K₁, exact-hit/approximate-miss and exact-miss/approximate-hit counts, and final metric changes. Log short candidate lists; separate approximation effects from filtering shortfalls.

**Limits.** This identifies where the observed target is lost, not its cause or the quality of every non-target recommendation.

---

### Step 3 — For whom and for what does it fail?

**Axis A: item type.** Report two distinct breakdowns; do not present them as one partition:
- **New-in-period vs not-new:** first interaction at/after the split start (T1 on Val, T2 on Test). On Test, new-in-period equals the canonical cold group; it is 44.4% of Test targets.
- **Canonical cold/tail/head:** retain the existing labels based on pre-T2 counts (`CLAUDE.md` §4.7). On Val, new-in-period items already belong to tail/head, so these labels overlap with new-in-period. They are retrospective popularity descriptions, not evidence that a model knew those counts. Never feed them into model features or treat them as validation-period feature statistics.
- When comparing new items with head items, use **new-in-period vs not-new-and-head** to avoid overlap. Keep the original labels unchanged. Report whether each target was seen in the compared model's training period as a separate flag: unseen-in-training is not the same as new-in-period.

**Questions.** Do text models retain new-item recall? Does Two-tower (text + ID) trade new-item recall for warm-item performance? Do rankers demote retrieved new targets more often than not-new head targets, after accounting for the different numbers of Stage-1 hits available to demote?

**Axis B: history length and category diversity.** Keep existing full-history length bins: 1–2, 3–5, 6–10, 11+. Within 11+, count distinct known full leaf-category paths in the model-visible last 50 items, strictly before the target. Split into 1–2 known categories (homogeneous), 3+ (mixed), and zero known categories (unknown); report missing-category coverage. Unknown categories are not an extra category. These are descriptive thresholds, not established boundaries of user intent; freeze them before Test.

**Question.** Is DIN's improvement over MLP larger on long, mixed histories? Compare the actual DIN-minus-MLP effect between groups, not “significant in one group, non-significant in another.” Category diversity is a proxy; neither architecture guarantees the predicted result.

**Measurement.** Report metrics, funnel shares/rates, users, instances, retrieved targets and hit counts for each slice. Check Val support before committing to profile intersections; sparse cells get counts and uncertain/descriptive estimates rather than strong claims. No universal minimum instance count guarantees adequate power. Report observed effect sizes/CIs and mark unsupported comparisons. Do not redefine bins after seeing Test.

---

### Step 4 — Which signal profiles accompany the gap?

**Purpose.** Describe which candidate relationships are amplified between the retrieved pool and top 10, and relate that redistribution to observed target outcomes. Concentrate on gaps located in Steps 2–3; do not infer that the model used a particular feature.

**Definitions.** Unless stated otherwise, history means the last 50 items strictly before the target, matching the model input. The last purchase means the most recent such event under the existing deterministic event ordering. Missing category/store values do not match each other.

| Signal | Definition |
|---|---|
| `Co-buy` | At least one other user in the permitted training graph reviewed both candidate and a history item; exclude the current user's own links. All reviews count, matching the interaction protocol. |
| `Same-device` | Parse canonical phone-model sets from titles. True if known sets overlap with a history item's set; false if candidate and at least one history device are known and all known sets are disjoint; otherwise unknown. Generic/universal compatibility needs an explicit parser rule, not an assumed mismatch. |
| `Same-leaf` | Candidate's known full leaf-category path equals the last purchase's known path. |
| `Same-store` | Candidate's known store appears in history; store is not assumed identical to brand. Report as a marginal signal only. |
| `Head item` | Top 20% of positive-count items in the permitted training graph, count descending/item-ID ascending. This training-head signal is separate from canonical pre-T2 head/tail reporting. |
| `Text-similar` | Cosine similarity of frozen candidate MiniLM text to the mean of available history text vectors exceeds a Val-fixed threshold; do not use learned ranker scores to define it. Missing/zero-vector text is unknown. |

**Definition freeze before Test.** Save the parser/rule version, training-graph cutoffs, similarity aggregation and numeric threshold, reference-pool sampling method/seed and artifact hash, and missing-value handling. Evaluation chooses the text-similarity threshold from the Val reference distribution and records its rationale before applying it to Test; it is not yet a measured value. Audit a reproducible Val title sample for device-extraction correctness as well as coverage, by cold/warm status. If device evidence is inadequate, label the device hypothesis untested; text similarity alone cannot establish compatibility.

**Two focused tables, for readability rather than independence.**
1. **4A:** `Same-device` × `Same-leaf` × `Text-similar`. Eight Boolean profiles for known device/text values, plus separate unknown/missing groups that preserve the full denominator. Same/different category are proxies; they do not establish substitutes/complements.
2. **4B:** `Head item` × `Co-buy` × `Same-leaf`, on instances with not-new targets. Cold/new targets lack training-head and co-buy evidence by construction, so use 4A and cold-specific outcome rates for them. 4B describes warm-target instances, not all failures. The two tables share `Same-leaf`; their findings are not independent replications.

**Populations, weighting and outcomes.**
- **Stage 1:** within the same user/instance population, compare a profile's candidate-pool slot share with its true-target share; also show profiles of retrieved vs missed targets. Separate new/not-new targets. Target share uses all targets, including retrieval misses.
- **Stage 2:** restrict to natural lists containing the target. For the identical instances, compare profile shares in the candidate pool, retrieval-order top 10, MLP top 10, DIN top 10, and retrieved targets. Show the broader all-instance slot distribution separately so conditioning does not hide behaviour on retrieval-miss instances.
- Compute slot/profile shares within each instance, average within user, then across users, matching headline weighting; retain raw counts. Exclude padding and report actual list lengths. Shares are distributions with different units, not interchangeable probabilities of relevance.
- Add target retrieval/hit rates by target profile, and rescue/demotion rates with the at-risk denominators from Step 2. Slot amplification alone is not evidence of an error. Report support and uncertainty alongside each claimed gap; matched ranker comparisons use identical users/instances.

**Hypotheses to examine on Val and freeze before further Test analysis.** These are proposed associations, not established mechanisms or a claim of preregistration before existing retrieval Test exposure.

| Hypothesis | Evidence to examine |
|---|---|
| Same-category concentration | Same-leaf + training-head items gain top-10 share while same-profile targets have low hit rates. Check observed category transitions; do not assume users seldom buy another item in the same category. |
| Apparent wrong-device match | Known-device-mismatch + text-similar candidates gain top-10 share relative to pool and targets. Validate parser errors and universal accessories before describing a compatibility failure. |
| Popularity override | On matched within-instance rank transitions, a retrieved device-matching/co-buy target loses to training-head candidates without device match. This needs a targeted comparison beyond either three-signal table alone. |
| New-item suppression | Retrieved new targets have lower promotion or higher demotion rates, alongside changes in new-item slot share. Review count/age are possible explanations; the analysis does not establish that those features caused it. |
| DIN and mixed histories | DIN-minus-MLP outcomes improve more for long, mixed histories; inspect device/category profiles. Shared OOV item IDs do not require equal cold-item performance: attention also uses category embeddings, and other features/learned weights differ. |
| Text + ID versus text | Learned IDs accompany a shift in candidate slots toward items seen during retriever training and a change in new-item retrieval rates. |

**Limits.** Non-targets are unobserved, not disliked. Co-buy, popularity, category and text correlate. Report marginal shares alongside profiles. Include the all-false row as **none of these indicators present**, not “unexplained failure”; unknown/missing rows are separate. Counts cannot quantify all causes of failure. A category-transition view is optional; a surrogate decision tree is not required for this plan.

---

### Step 5 — What does a representative case show?

- Select 3–5 cases covering the strongest supported patterns, including at least one counterexample. There is no requirement to illustrate every proposed hypothesis.
- Freeze qualifying bucket definitions and the selection rule on Val. Within each selected bucket, sort stable instance IDs and take one with a seeded RNG (seed 0), recording the eligible count and chosen ID. Fix the bucket order and avoid duplicate IDs. Apply the same frozen rule if illustrating Test; disclose any additional presentation-based choice rather than replacing an inconvenient result silently.
- Show the last 3–5 purchases, true target, Two-tower (text) retrieval rank and candidate membership, MLP/DIN top three, and the target's final rank/hit status. Top-three examples alone cannot explain a top-10 metric. Annotate profiles and unknowns; describe compatibility only where checked.
- Cases illustrate the aggregate analysis; they do not provide independent proof.

---

### Cross-cutting rules and completion checklist

- **Permitted evidence:** build co-buy/training-head signals from retriever-training interactions for RQ1 retrievers, and all training interactions for ranker analysis. Record each cutoff; do not credit an earlier model with later collaborative evidence. Keep descriptive canonical pre-T2 popularity labels separate from these signals and from model features.
- **Prior Test exposure:** record existing retrieval Test reporting. Before any new ranker/cascade Test inspection, save run IDs, training/early-stopping policy, definitions, thresholds, hypothesis family, selection rule and planned comparisons. Additional hypotheses prompted by Test are explicitly exploratory, not independent confirmation.
- **Replication:** compare direction, effect size and uncertainty across Val/Test. Non-replication may reflect uncertainty or population shift; report it as unstable or inconclusive rather than redefining the hypothesis. Neither matching direction nor two nominal p-values establishes a mechanism.
- **New-item reporting:** provide new/not-new results wherever support permits; preserve the distinct canonical segment definitions and denominators.
- **Provenance:** save configs, training seeds, bootstrap seed, git hashes, candidate source/hash and analysis definitions. Identify which components are fixed. Sample runs are for development, never reported as full-data results.
- **Open choices before dependent work:** the controlled training protocol below is authorised; changes beyond it need a recorded decision (Tong/team); profile thresholds/parser audit and support (evaluation); acceptable loss margin δ and final selection rule (team). Recording these dependencies does not assign ranker implementation to evaluation.

| Research question | Primary evidence | Explanation |
|---|---|---|
| RQ1: retrieval, text vs ID | Step 1 | Item segments, retrieved/missed target profiles; no ranker funnel is needed for a retriever-only row |
| RQ2: K₁ and search index | Step 1 | Step 2 target losses/recoveries and exact-search overlap |
| RQ3: rankers on fixed candidates | Step 1 | Step 2 metric contributions, Step 3 history groups, focused Step 4 profiles |
| Final F0/F1/F2 comparison | Step 1, frozen Val selection | Steps 2–4 explain supported differences; Step 5 illustrates them |

An underperforming ranker or final pipeline is a valid result. Report the verified outcome, its uncertainty and the supported failure analysis without adding variants to manufacture an improvement.

## Authorised follow-up: controlled ablations (not results)

### Fixed decisions

- Final Test, index tuning and full-cascade timing stay paused. SSH and cluster execution are user-managed.
- Ranker study: matching MLP A/B/C/D = all training lists versus natural-hit-only, crossed with score enabled versus uniformly masked. Four seed-0 pilots, then seeds 1/2 for all valid arms: 12 full-data fits. Do not prune the factorial based on a negative seed-0 result.
- All arms have identical parameter shapes and initialization within seed. Full-training feature normalization is fixed before cohort selection. BCE, learning rate, 50 uniform negatives, max 30 epochs and patience 3 stay fixed. Shared events receive paired negatives and relative ordering. Checkpoints use natural-hit conditional macro nDCG@10 at K200.
- B-A and D-C test score availability within a fixed population. C-A is the total natural-only recipe effect, including fewer supervised users/items and optimizer updates. The direct interaction is (D-C)-(B-A).
- A fix qualifies for validation integration only with all three seeds, overall K200 paired nDCG interval lower bound >0 and no observed new-item nDCG loss versus fresh A. Highest qualifying overall mean wins; exact ties use B/C/D order. If none qualifies, A remains the reference without a claim that its policy is optimal.
- Retriever study: fresh unregularised text+ID versus ID-residual dropout p=0.5, with shared per-item batch masks and dedicated RNG. The historical Recall@100 selector, objective and static q remain fixed. Pilot seed 0; replication seeds 1/2 requires a separate launch after the pilot guard is reviewed.
- Retriever pilot guard: higher observed overall nDCG@10 and no observed new-item Recall@200 loss versus text-only. Report the dropout-minus-ID-control comparison separately. Repeat guardrails after replication; a point-estimate guard is not noninferiority.
- Graph work explains relationships only. Device-title extraction requires the locked audit before quantitative device claims. Graph paths are not causal model explanations.

### Local prerequisites and completed evidence

The original saved-results evaluation described above is complete within its available saved evidence. Candidate profiles and final held-out comparisons are separate dependencies.

An isolated structural reconstruction is under `data/processed/studies/20261027-targeted/`, configured by `results/targeted_study/data.yaml`. Original source tables are preserved. The local preflight report (`results/targeted_study/preflight.json`) lists remaining files; graph structure can be analysed before original embeddings/candidates are recovered.

Check the inputs **on the cluster**, from its repository root, before reserving a GPU. These commands return exit code 2 and list missing/mismatched inputs rather than allowing training to start:

```bash
.venv/bin/python -m cascade.pipelines.study_preflight --study-kind ranker --require-ready
.venv/bin/python -m cascade.pipelines.study_preflight --study-kind retriever \
  --text-reference results/retriever/two_tower_text --require-ready
```

The retriever check does not require ranker candidates. Ranker checks also include the historical comparison/config and frozen candidate identities. If processed inputs are elsewhere, pass `--data-config YOUR_CONFIG.yaml` (same frozen split/mappings). Supply `--candidate-dir YOUR_CANDIDATES` for ranker or `--text-reference YOUR_TEXT_RUNS` for retrieval as needed. Both checks must report `ready` for their respective jobs; this does not establish CUDA runtime or cluster allocation availability.

The policy runner checks original candidate hashes and the original feature contract. A regenerated archive is not silently accepted as an exact historical reproduction; establish a separately documented baseline before changing those checks.

### Explicit experiment commands

Run from the repository root after recovering the original inputs and confirming execution readiness. No additional test run is requested for this documentation handoff. On the cluster, `configs/data.yaml` points to its original full processed data. `--device cpu` supports technical checks locally; use CUDA for full fits.

```bash
# Development functionality only: four arms, fixed 2% full-data users, one epoch.
.venv/bin/python -m cascade.pipelines.ranker_policy_study \
  --stage development --run-id policy-development --device cuda

# Full-data seed-0 pilot, then separately launch seeds 1/2. Fresh roots per stage.
.venv/bin/python -m cascade.pipelines.ranker_policy_study \
  --stage pilot --run-id policy-pilot --device cuda
.venv/bin/python -m cascade.pipelines.ranker_policy_study \
  --stage replication --run-id policy-replication --device cuda

# Combine completed stages without averaging model scores into an ensemble.
.venv/bin/python -m cascade.eval.ranker_policy_report \
  --studies results/ranker_policy/policy-pilot results/ranker_policy/policy-replication \
  --output results/targeted_study/policy-report

# Frozen sensitivity probe; not a retraining or causal effect estimate.
.venv/bin/python -m cascade.pipelines.retriever_ablation \
  --run-id id-probe --probe-only \
  --probe-run-dir results/retriever/two_tower_text_id/seed0 --device cuda

# Matched retriever pilot pair, then inspect the text-reference guard.
.venv/bin/python -m cascade.pipelines.retriever_ablation \
  --run-id id-pilot --phase pilot --device cuda
.venv/bin/python -m cascade.pipelines.retriever_ablation \
  --run-id id-pilot --phase pilot --summarize-only \
  --text-reference results/retriever/two_tower_text

# Only after reviewing a promising pilot, explicitly complete the same pair.
.venv/bin/python -m cascade.pipelines.retriever_ablation \
  --run-id id-pilot --phase replication --device cuda
.venv/bin/python -m cascade.pipelines.retriever_ablation \
  --run-id id-pilot --phase replication --summarize-only \
  --text-reference results/retriever/two_tower_text
```

Every completed policy arm saves selected K50/100/200/500 ranks and AUCs, K200 reproduction diagnostics, group labels, training support and optimizer-step counts. Its report exports the five-way funnel, paired effects/interaction and additive expansion losses. Retain inconclusive and negative findings.

### User-managed Slurm submission

**Run instructions updated 29 October 2026.** Sync the edited source/script files to the cluster first. Run from the repository root with the existing `.venv` and **original full processed inputs**. The script defaults to the repository's `gpu-long` / `gpu:a100-40:1`, 32 GB RAM, four CPUs and six hours. Override `--partition`, `--gres`, `--account` and `--time` on `sbatch` for **your own allocation** where required; no teammate allocation is assumed.

After the prerequisite checks above succeed, submit the two independent seed-0 pilots (use fresh IDs):

```bash
mkdir -p results/logs
sbatch --export=ALL,STUDY_KIND=ranker,STAGE=pilot,RUN_ID=policy-20261029-pilot scripts/slurm/targeted_study.sbatch
sbatch --export=ALL,STUDY_KIND=retriever,STAGE=pilot,RUN_ID=id-20261029 scripts/slurm/targeted_study.sbatch
```

Each job checks its inputs/CUDA, runs only the requested phase, and generates its report automatically. Neither job runs Test, SSH or another `sbatch`. Logs: `results/logs/slurm-cascade-targeted-JOBID.out`.

| Pilot result | Where to read it |
|---|---|
| Four-arm MLP quality, groups, rescue/demotion, expansion and A/C score response | `results/targeted_study/policy-20261029-pilot-pilot-report/report.md` |
| ID dropout versus fresh ID control and text reference, overall/new-item guard | `results/retriever_ablation/id-20261029/pilot_comparison.json` |
| Trained models and selected predictions | `results/ranker_policy/policy-20261029-pilot/` and `results/retriever/id_dropout/id-20261029/` |

After reviewing the pilot outputs, explicitly submit seeds 1/2. Keep the full four-arm ranker comparison even when a valid pilot is negative. Retriever replication **requires** `text_reference_guard.status = "promising"`; the runner rejects replication otherwise. Keep the same retriever RUN_ID to include seed 0; ranker phases use different output roots.

```bash
sbatch --export=ALL,STUDY_KIND=ranker,STAGE=replication,RUN_ID=policy-20261029-replication scripts/slurm/targeted_study.sbatch
# Only if the retriever pilot guard is promising:
sbatch --export=ALL,STUDY_KIND=retriever,STAGE=replication,RUN_ID=id-20261029 scripts/slurm/targeted_study.sbatch
```

After ranker replication finishes, combine all three seeds (CPU reporting; no training):

```bash
.venv/bin/python -m cascade.eval.ranker_policy_report \
  --studies results/ranker_policy/policy-20261029-pilot results/ranker_policy/policy-20261029-replication \
  --output results/targeted_study/policy-20261029-all-seeds
```

Retriever replication automatically writes `results/retriever_ablation/id-20261029/replication_comparison.json`, including all three seed pairs and the text guard. Across arms/seeds the report checks recorded seed/population/dropout, training recipe, data/source/checkpoint identities and aligned Val ranks; differences are rejected rather than silently pooled.

For non-default input locations, append `DATA_CONFIG=...`, `CANDIDATE_DIR=...` (ranker), or `TEXT_REFERENCE=...` (retriever) to `--export=ALL,...`. For an optional one-epoch ranker development check, submit `STUDY_KIND=ranker,STAGE=development,RUN_ID=YOUR_FRESH_ID`. Retriever development is not exposed by this wrapper: the pilot is the full-data control/treatment pair. Existing fit/report directories are never overwritten. If training completed but reporting failed, run only the relevant report command with a fresh report output; do not resubmit completed fits.

Historical CUDA fit-plus-epoch-validation times were 12–13 minutes for text+ID and 27–32 minutes for text-only. Budget one GPU-hour per retriever fit (two for the pilot; six for all seed pairs), excluding candidate generation. This is an allowance, not a measured dropout runtime. Ranker runtime remains unmeasured: pilot is four fits and replication eight sequential fits; override `--time` if your measured pilot indicates six hours is insufficient.

**Readiness evidence:** shell syntax and CLI imports/help passed; local preflight correctly exits 2 for the six missing ranker artifacts / two missing retriever vectors. No cluster submission, CUDA fit or new results were run locally. Submission tooling is prepared, but each cluster's real inputs and allocation must pass its checks.

### Evidence to write after execution

Append measured findings to this document: run IDs and supports; which targets were rescued/demoted; effect sizes/intervals; new-item and temporal changes; whether the score response changed; any guardrail failure; and which competing explanations remain. Keep measured findings, sensitivity probes and training-policy interpretations separate. A failed fix is a result.

If the retriever qualifies, rebuild only training/Val candidates in a new namespace and train three matching MLP seeds using the same chosen recipe on each retriever. Applying a frozen old ranker to new candidates is only a distribution-shift diagnostic. Final Test remains paused.

## Exact remaining inputs and handoff

As of 2 November, this checkout contains `data/processed/item_text_emb.npy`, `item_text_emb.json`, and `data/processed/candidates/two_tower_text_seed0/{ranker_train.npz,val.npz,val_target_score.npz,meta.json}`. Ranker preflight and the four-arm development run pass; checked validation/metadata hashes match the frozen source. The earlier missing-file diagnosis is historical. Verify these original inputs independently on the cluster before full fits. User handles SSH and cluster commands; no teammate allocation is assumed available.

Candidate-slot amplification and the original MLP/DIN top-three products still require their respective candidate-level analyses/replay; recovering inputs alone does not complete them. Complete the locked annotation before device claims. Final Test and index/cost work stay paused. The ablation runners have not produced full-data results; the updated submission script runs blocking input checks and exports comparison reports. Cluster execution has not been verified here. The 2 November test/development evidence above verifies the current source; older implementation-review/test counts remain historical.

## Evidence and reproduction

The narrative above is self-contained. Compact CSV/JSON evidence is retained for audit; agents do not need to read generated report Markdown to determine status. Large generated graphs, copied result packages, local processed data and development outputs stay local/ignored. Historical source artifacts remain unchanged.

| Evidence | Location |
|---|---|
| Saved-validation aggregates, seed/group/funnel/CI tables and rank reconciliation | `results/saved_evaluation/20261027-final-validation/` (CSV/JSON only in this handoff) |
| Original candidate identities used by the prerequisite checker | `results/saved_evaluation/20261027-saved-validation/provenance.json` |
| Verified marginal target/history profiles | `results/explanation_graph/20261027-target-profiles-verified/` (compact tables/provenance) |
| Joint 12-cell warm profiles, retrieval profiles, five cases and contribution reconciliation | `results/evaluation_completion/20261028-final/` (CSV/JSON) |
| Device audit samples, still unlabelled | `results/explanation_graph/20261027-base/device_audit_{development,locked}.csv` |
| Historical checkpoints/candidate hashes and exclusions | `results/ranker_artifacts.json` |

To replay archived validation predictions without training, original processed data or Test prediction loading:

```bash
.venv/bin/python -m cascade.eval.saved_results --run-id YOUR_FRESH_RUN_ID
```

Use a fresh ID; existing outputs are preserved. Config: `configs/evaluation/saved_results.yaml`. `--sample` is a development subset and must never be reported as a full-data result. Graph/profile reproduction additionally needs the original/restored processed inputs; the one-off `notebooks/complete_evaluation.py` records the completion analysis, with local input/output paths that must be adjusted for a fresh run. Do not launch retraining merely to reproduce these saved-result findings.

### 7 November: seeds 1/2 replication submitted

The user authorised submission. Slurm job `1089445` (`STAGE=replication`, `RUN_ID=policy-20261107-replication`) was submitted on the same checkout/inputs as the pilot, with the pilot's resources (gpu-long, h100-47, 4 CPU, 32G, 6h). It covers all four arms × seeds 1/2 (eight fits). At submission it was pending (Priority). No result is claimed; Test remains paused. After completion, run the all-seeds report with `--studies results/ranker_policy/policy-20261106-pilot results/ranker_policy/policy-20261107-replication`.

### 7 November: E3 candidate-count curve per policy arm (pilot seed 0, no training)

Report `results/targeted_study/e3-20261107-pilot-kcurve/` (`cascade.eval.ranker_policy_extension_report`, 1,000 user resamples). Overall macro nDCG@10 at K50/100/200/500 and paired gap of each K versus that arm's best observed K:

| Arm | K50 | K100 | K200 | K500 | Best | K200−best CI | K500−best CI |
|---|---:|---:|---:|---:|---:|---|---|
| A | 0.008026 | 0.008370 | 0.007076 | 0.002400 | 100 | [-0.001524, -0.001053] | [-0.006317, -0.005631] |
| B | 0.006598 | 0.006023 | 0.004216 | 0.001711 | 50 | [-0.002683, -0.002052] | [-0.005227, -0.004540] |
| C | 0.008406 | 0.008684 | 0.008617 | 0.007863 | 100 | [-0.000234, +0.000113] | [-0.001031, -0.000579] |
| D | 0.007677 | 0.007932 | 0.007780 | 0.006937 | 100 | [-0.000326, +0.000035] | [-0.001245, -0.000731] |

Under natural-only training (C/D) the K200 interval versus K100 includes zero and the K500 loss is about 7× smaller than A's. C's 200→500 competition loss is −0.0021 versus A's −0.0065 with similar new-target gain (+0.0013 / +0.0018). The earlier RQ2 conclusion that larger K₁ strongly hurts is therefore largely conditional on the forced-positive training recipe. Single seed; interval excludes training-seed variance; no K₁ selection is made here (no δ recorded).

### 7 November: extension arms authorised (pre-registered before any outcome)

The user authorised additional controlled arms after seeing the seed-0 pilot. They are therefore **post-hoc and exploratory**; any qualification is not confirmatory. Same frozen candidates, feature contract, normalisation, initialisation per seed, BCE/50 negatives, selector and K prefixes as A–D:

- **E (score imputation):** all lists; each forced target's retrieval score replaced by the score of a uniformly drawn retrieved candidate from the same list (RNG keyed by instance ID, seed-independent). Natural lists and all other cells unchanged (verified: 276,320 cell changes over 276,321 forced lists; mean forced-target score −0.342 → −0.180, natural targets −0.154). Tests whether removing the score shortcut while keeping forced supervision recovers new-item quality.
- **F (longer natural-only schedule):** C with max 100 epochs and patience 10. Tests whether C's new-item loss is an exposure artefact (C stopped at epoch 5 with 4,850 selected steps versus A's 86,658).
- **G / H (partial forced lists):** natural lists plus a fixed 10% / 30% subsample of forced lists (RNG 20261107 over the full universe): 66,578 / 121,948 lists. Tests an intermediate point on the overall-versus-new-item trade-off.
- Fresh A and C controls are retrained inside each extension run; seeds 0/1/2 run as three independent jobs. Report with `cascade.eval.ranker_policy_extension_report`; the same guard (3 seeds, overall K200 CI lower bound > 0, no new-item loss versus A) is applied, labelled exploratory.
- Development check `ext-20261107-localdev_sample` (2% users, 1 epoch, CPU) completed all six arms with identical initial parameters. Development values are not reported.
- Cluster execution (7 Nov, user-authorised via xlogin): separate checkout `/home/t/tjingyen/cp4285-ext-20261107/cascade-rec` (copied from the pilot checkout so job 1089445's source-hash guard is untouched), plus the three changed files (SHA-256 in `staging/sha256.txt`). A cluster-side driver (`staging/ext_driver.sh`, log `driver.log`) waits for the copy, verifies hashes and that `cascade` imports from the new checkout, then submits `scripts/slurm/policy_extension.sbatch` for seeds 0/1/2 as `ext-20261107-s{0,1,2}`. Check `driver.log` for job IDs; if the import assertion fails no job is submitted. Afterwards: `python -m cascade.eval.ranker_policy_extension_report --studies results/ranker_policy/ext-20261107-s0 results/ranker_policy/ext-20261107-s1 results/ranker_policy/ext-20261107-s2 --output results/targeted_study/ext-20261107-all-seeds`.
- Correction (7 Nov 15:40 SGT): the full checkout copy was too slow (many small `.venv`/data files), so the driver was disabled (its checksum file replaced so it exits before submitting; original hashes kept in `staging/sha256.ok`). The extension instead runs from `/home/t/tjingyen/cp4285-ext-20261107/fast/cascade-rec`: code/results copied, `data` symlinked read-only to the pilot checkout, the pilot checkout's Python environment used via `PY=...` with `PYTHONPATH` pointing at the new `src` (verified `cascade` imports from the new checkout). Submitted Slurm jobs `1089821/1089822/1089823` = seeds 0/1/2 (`ext-20261107-s{0,1,2}`). Outputs land in that checkout's `results/ranker_policy/`.
- Jobs 1089821–23 failed at import within 5 s (no run directories created): the lightweight copy's `--exclude data` also dropped `src/cascade/data`. Re-synced with root-anchored excludes; `diff -r` against the pilot `src/` and `configs/` now shows only the two intended files. Resubmitted as **1090001/1090002/1090003** (seeds 0/1/2, same RUN_IDs).

### 7 November: A–D replicated over three seeds

Replication job `1089445` completed (exit 0, 1:47:06; 8 fits). Reports were generated on the cluster as CPU job `1090722` (login-node memory cap too small) and copied back: `results/targeted_study/policy-all-seeds-20261107/` (factorial) and `results/targeted_study/e3-20261107-all-seeds/` (K-curve). The local copy of the replication archive was incomplete at report time because of SSH drops; the reports read the complete cluster copy.

K200 macro nDCG@10, mean over seeds 0/1/2 (seed SD): A 0.006887 (0.000316), B 0.004200 (0.000269), C 0.008572 (0.000124), D 0.007625 (0.000421). Seed-mean paired user-bootstrap differences:

| Comparison | Overall | New-in-period | Warm-head | Warm-tail |
|---|---|---|---|---|
| C−A | +0.001685 [0.001445, 0.001920] | −0.000757 [−0.001030, −0.000489] | +0.004222 [0.003809, 0.004643] | −0.000469 [−0.000674, −0.000263] |
| B−A | −0.002687 [−0.002896, −0.002478] | −0.000599 [−0.000863, −0.000364] | −0.004939 | −0.000170 [−0.000370, 0.000010] |
| D−C | −0.000946 [−0.001106, −0.000787] | +0.000124 [−0.000016, 0.000262] | −0.002026 | −0.000005 |

C beats A in every seed (overall 0.008617/0.008431/0.008667 versus 0.007076/0.007063/0.006522) and has the lowest seed SD. The pilot's new-item loss also replicates but is smaller in the mean (−0.000757, about −24% relative, versus −40.6% at seed 0). Seed-to-seed new-item values overlap (A 0.0028–0.0035; C 0.0021–0.0026). **No arm qualifies** under the pre-registered guard because of the new-item loss; A remains the integration reference without a claim that it is optimal.

Three-seed K-curve: C's best observed K is 100 (0.008609). K200 versus K100 is −0.000037 [−0.000194, 0.000118]; K500 versus K100 is −0.000686 [−0.000870, −0.000477]. A's K500 loss versus its best K is −0.005971, so C's is about 8.7× smaller. The large-K collapse is largely tied to the forced-positive recipe, replicated over seeds.

### 7 November: meso secondary metrics and per-event oracles (saved predictions, no training)

`scripts/analysis_meso_oracle.py` → `results/targeted_study/meso-oracle-20261107/`. K200 natural Val, user-macro, seed-mean (policy A/B/C and R2/R3 three seeds; D seed 0; R1 deterministic). Oracle = per event and seed, the better target rank of two same-seed systems; an upper bound for a perfect per-event selector, not an achievable system.

| System | nDCG@10 | Hit@3 | MRR | Head nDCG | Tail nDCG | New nDCG |
|---|---:|---:|---:|---:|---:|---:|
| R1 retrieval order | 0.005642 | 0.004818 | 0.005390 | 0.011290 | 0.000166 | 0.000361 |
| A (forced, MLP) | 0.006887 | 0.005415 | 0.006130 | 0.011438 | 0.001316 | 0.003147 |
| C (natural-only) | 0.008572 | 0.007328 | 0.007676 | 0.015660 | 0.000847 | 0.002390 |
| R3 DIN (forced) | 0.006436 | 0.004708 | 0.005799 | 0.009865 | 0.001398 | 0.004074 |
| Oracle(A, C) | 0.011497 | 0.010005 | 0.009987 | 0.019907 | 0.001755 | 0.004511 |
| Oracle(MLP, DIN) | 0.009728 | 0.007775 | 0.008401 | 0.015513 | 0.001980 | 0.005210 |

- C−A Hit@3 +35% versus nDCG@10 +24%: natural-only gains are concentrated at the very top.
- Oracle(A, C) exceeds C by +0.002925 [0.002782, 0.003069] (+34%); oracle(MLP, DIN) exceeds MLP by +0.002885 [0.002757, 0.003021] (+42%). Seed-0 top-ten hits overlap only 2,117 of 5,319 (A/C) and 2,022 of 4,634 (MLP/DIN): the systems succeed on substantially different events.
- DIN−MLP: new-in-period +0.001254 [0.001013, 0.001501] (+44%), warm-head −0.001673 [−0.001961, −0.001345]. Learned rankers raise new-item nDCG 6.6–11× over retrieval order, whose new/tail values are near zero.
- An achievable router would need a selector using pre-event information only (for example candidate item-age); none is trained or claimed here.

### 7 November: 24 micro cases (seed 0)

`scripts/analysis_micro_cases.py` → `results/targeted_study/micro-cases-20261107/{cases.md,cases.json}`. Four prespecified categories, 6 events each drawn with RNG 0 from all eligible natural-K200 events: C rescues warm-head target missed by A (pool 1,606); A keeps new-in-period target missed by C (387); DIN rescues new target missed by MLP (342); retrieval miss at K500 (204,329). Ranker top-3 products were recomputed from the saved A/C/DIN seed-0 checkpoints; all recomputed target ranks equal the saved ranks. Illustrative only: six cases per category do not estimate prevalence.

Observed patterns (to be quantified before being stated as general):
- **C concentrates on a few bestsellers.** Across the 24 lists, C's 72 top-3 slots contain 38 distinct products versus 59 for A and 52 for DIN; one smartwatch case appears 9 times and one wireless charger 7 times. C's warm-head rescues are mostly purchases of those same products, with the target's leaf category in the recent history in only 1 of 6 cases. This suggests the head gain partly reflects popularity concentration (a diversity cost), not personal relevance.
- **A/DIN keep new items through brand and model continuity.** Examples: TORRAS iPhone 12 case → TORRAS iPhone 13 case (DIN rank 1 and 4); Moto G6 owner → Motorola One 5G; three single-ear headsets → another headset; Anker → Anker stand; BELONGME iPhone case → BELONGME watch protector. Target store appears in the five-item history in 3/6 DIN-rescue cases versus 0/6 C-rescue cases.
- **Many new-in-period targets are iPhone 13 launch accessories** (first seen 13–25 Sep 2021), shortly after Val start (11 Aug 2021): the new-item segment is partly a product-launch wave.
- **A/DIN top-3 come from deep retrieval positions** (median retrieval rank 107 / 140.5 versus 53 for C), consistent at micro level with the low-score preference.
- **Retrieval misses are often device or category switches** (iPhone case buyer → Apple Watch band; Fitbit accessories → iPhone screen protector; renewed Samsung watch), with full-catalogue ranks of 649–28,699. No ranker can recover them.
- Next check: full-validation top-10 catalogue coverage and popularity concentration per arm, to test the diversity hypothesis.
