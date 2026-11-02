# Evaluation findings, protocol and next experiments

Updated **2 November 2026** (independent local experiments and frozen-control full-catalogue check; retaining expanded graph evidence and the supervisor consultation review). This is the single current evaluation record for teammates and agents. Read this document for results, interpretation, progress, limitations and the authorised follow-up. Append future findings here. Other experiment notes and generated reports are historical evidence, not separate instructions.

## Current progress

The archived full-data evaluation is complete: **232,904 validation events / 124,416 users**, using archived predictions and restored histories. The saved-validation replay reproduced the historical comparisons and checked source identities; it is analysis verification, not independent training replication. That completion used no new training, Test predictions, SSH or cluster jobs. A separate, subsequently authorised sample experiment programme is also complete; its fresh MPS fits and follow-up frozen-control validation are recorded below.

| Work | State |
|---|---|
| RQ1 retrieval comparisons and text/ID trade-off | Complete from saved evidence; existing retrieval Test exposure disclosed |
| RQ3 rankers, uncertainty and five-way failure funnel | Complete on Val, frozen text seed-0 candidates |
| RQ2 exact-search K sweep, popularity-segment curve and expansion losses | Complete on Val (`warm-head` peaks at K50, `new`/`tail` peak at K200, aggregate peaks at K100); no final K selected |
| Item/history/diversity/time groups; secondary metrics | Core group analysis complete; Hit@3/MRR@10 completion is unverified because no reproducible secondary-metric artifact was found |
| Graph target/history relations and 12 joint warm-target cells | Complete on Val; descriptive, not causal |
| Graph witness multiplicity/recency, full-window category continuity and store interactions | Expanded on 1 Nov across all 232,904 Val events; exploratory, with paired-user intervals |
| Microscopic case study | Five source-bound cases complete; local 25-item consultation discussion is a draft, not a verified 25-event audit; ranker top-three identities unavailable |
| Candidate-pool/top-ten profile amplification and device/text analysis | Blocked by original candidates/vectors and locked manual device audit |
| Controlled policy retraining (A/B/C/D: all/natural-only × enabled/masked score) and ID dropout | Submission tooling updated 29 Oct; cluster input checks required; no full-data remedy result |
| Approximate-index study, full-cascade timing and final F0/F1/F2 Test | Deferred by user |
| Independent local track, root `plan.md` | Sample programme complete: 18 strict-pool retriever and six natural-only ranker fits, including the subsequently approved candidate-normalisation trial; frozen diagnosis, fusion, reports and audits complete |
| Full-catalogue frozen-text check | Complete on all Val events: recency improves top-ten quality over archived retrieval but lowers Recall@200; strong new/warm trade-off |

**Working conclusion:** retrieval misses remain the largest ceiling, while reranking also displaces strong retrieved targets. Forced-positive training and the inverted score response motivate the authorised policy experiment; controlled retraining is still needed for a causal claim. New products supply about two-thirds of MLP's net gain. The expanded graph localises substantial losses to dense co-reviewed targets and shows gains for same-store co-reviewed head targets. These target associations motivate [architecture proposals](architecture_proposals.md); integration into the canonical full-data pipeline remains unapproved. The separate local experiments below are authorised through root `plan.md`. Target-conditioned oracle estimates do not establish deployable routing gains.

## Local original RecBole sample weights (2 November 2026)

Fresh original `SASRecBCE` seed-0 fits completed on Apple MPS, after a separate one-epoch functionality check. This is local development on **30,389 products / 7,619 sampled users**, evaluated on **4,825 validation events / 2,534 users**. These newly trained sample models are not comparable numerically with the full-catalogue tables below and do not recover the missing historical checkpoints. Their original sample cutoff **T0=1596679591746** is retained, distinct from full-data T0; T1/T2 are unchanged.

| Training period | Selected epoch (one-based) | Training/selection seconds | Val Recall@200 | Val Recall@10 | Val nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Retriever-training only | 8 | 24.318 | 0.024158 | 0.003393 | 0.001421 |
| All training data | 13 | 38.729 | 0.032763 | 0.002470 | 0.001187 |

These are single-seed descriptive levels, not a claimed significant comparison. The original RecBole validation Recall@100 selects the checkpoint; the table uses the shared filtered user-macro harness. The runner now defaults to validation-only evaluation, rejects existing destinations and creates fresh per-run atomic exports. No held-out Test prediction was generated or loaded.

Reusable selected weights are `results/baselines/local/20261102-sasrec-retriever-sample-seed0/model.pth` and `results/baselines/local/20261102-sasrec-all-train-sample-seed0/model.pth`, approximately 8 MB each. Each run records config, ID mappings, selected checkpoint, input/source hashes and `weights_verification.json`. Both standalone states load strictly and produce exactly the selected checkpoint's CPU scores on **32 validation histories × 30,389 items**, with maximum difference **0**. RecBole ID mappings must accompany the weights. Local weights/exports remain ignored by Git; the original baseline results remain unchanged. The independent experimental track is documented in root `plan.md`.

## Independent local experiments (2 November 2026)

The user authorised an independent experiment track, including fresh controls with **pre-T0-only negatives** after the original RecBole catalogue-wide negative pool was explained. The strict sample pool has **20,740 products**, excluding **9,649 future products**. The original fits above remain historical references. All new learned retrievers retain the original BCE, sequence backbone, training pairs and RecBole unfiltered Recall@100 selector; final metrics use the shared full-catalogue event-time availability/full-history filters. These fresh models do not change teammates' checkpoints or constitute the full-data policy ablation.

All **15 retriever fits** and **six natural-only ranker fits** completed on MPS, seeds 0/1/2. Sample evaluation covers **4,825 Val events / 2,534 users**, with **30,389 products** and the original sample split. The frozen MiniLM revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; vectors were generated with the recorded encoder library pins. Text is a crawl-time snapshot and first review is an availability proxy.

### Retrieval findings

| Method | Recall@200 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| ID SASRec, strict pool | 0.023005 | 0.002401 | 0.001298 |
| Content SASRec | 0.056750 | 0.003862 | 0.001836 |
| Content + warm-item ID residual | 0.043602 | 0.004994 | 0.002225 |
| Content + metadata neighbours | 0.053690 | 0.003040 | 0.001248 |
| Two-interest pooling | 0.052636 | 0.003269 | 0.001727 |
| Actual content/graph RRF, budget 200 | 0.057858 | 0.004006 | 0.001931 |
| Pre-T0 popularity | 0.022277 | 0.002494 | 0.000842 |
| Frozen text mean | 0.071513 | 0.013313 | 0.006961 |
| Frozen text, fixed 90-day recency | **0.100990** | **0.022309** | **0.012489** |

These are **retrieval scores/order**, including the top-ten metrics; no reranker is applied in this table. Learned entries are three-seed means; deterministic controls have no task-specific recommender training. All differences below use unadjusted paired-user 95% intervals, B=1000, seed 0, after averaging event metrics across training seeds.

- Content minus matched ID: Recall@200 **+0.033746 [0.026206, 0.041296]**; nDCG@10 **+0.000538 [-0.000521, 0.001606]**. New-in-Val retrieval improves, while seen-before-T0 retrieval decreases. This is a trade-off, not a universal improvement.
- Content+ID minus content: Recall@200 **-0.013148 [-0.020393, -0.006511]**; nDCG@10 **+0.000389 [-0.000631, 0.001379]**. The unseen-item ID residual is masked at training and inference.
- Graph, two-interest and fusion differences versus content include zero for Recall@200 and nDCG@10. Graph also changes encoder capacity. Two-interest heads have mean cosine **0.958** despite both winning recommendation slots; distinct learned interests are not established. Fusion uses actual source candidates under a unique 200-item budget, rather than target-conditioned oracle ranks.
- Fixed recency minus uniform frozen text: Recall@200 **+0.029476 [0.021533, 0.038459]**, nDCG@10 **+0.005527 [0.003225, 0.008073]**. The 90-day half-life was specified before these outcomes. Independent CPU float64 reconstruction on 100 fixed queries matches top-10/top-200 hits, with one rank differing by one; strictly prior histories and target exclusion pass.

The sample has **3,240 new-in-Val events (67.2%)**, versus approximately 33% in full data, and a different catalogue/first-observation process. Do not compare these absolute scores with full-data tables or extrapolate the gain.

### Natural-only ranking and other analyses

The matched mean and candidate-conditioned semantic-attention rankers train on **458/6,306 naturally retrieved training positives**, excluding 5,848 misses, with identical content-seed-0 candidate lists, inputs, common initial weights and scoring head. No insertion, target-only feature or retrieval-score input. This attention model is DIN-inspired; it is not the historical DIN implementation.

End-to-end nDCG@10 is approximately **0.000827 mean / 0.000858 attention**, versus **0.001557 retrieval order**. Neither improves clearly over retrieval. Attention-minus-mean is only about **+0.000031** under an unadjusted interval; it supplies little practical support for additional ranker complexity here. Training support is small, so this does not settle the value of DIN on the full dataset. Frozen learned-history truncation to 5/10 items is inconclusive.

All 232,904 archived full-data Val events were also partitioned into **12 UTC months**, with warm/new metrics and category/store mix. Retrieval Recall@200 is **0.089171** in partial August 2021 and **0.052947** in partial July 2022; new-in-Val event share rises from **4.6% to 47.3%**. These are descriptive endpoints, not evidence of an annual causal effect. New-in-Val is fixed relative to T1, so its share also reflects elapsed time; product ageing, turnover and calendar effects remain confounded. Review timestamps lag purchases. No split was changed.

Artifacts: `results/local_experiments/20261102-sasrec-pre-t0/summary-final/` contains combined metric/group/composition tables and twenty query examples fixed before outcomes. Run directories retain selected standalone weights, mappings, configs and source/input hashes. `final_audit.json` verifies portable weights/snapshots, shared initial backbones and initial negative traces, graph temporal validity, archive alignment and unique/available/history-free lists. Monthly evidence is in `results/local_experiments/20261102-full-val-calendar/`. Generated artifacts remain local/ignored. The post-sample suite passed **154 tests, with four legacy sample-embedding tests skipped**.

### Full-catalogue frozen-control check

After pausing, the user resumed following the recommendation to check the strongest simple control before further model training. `cascade.pipelines.frozen_text_study` completed on the restored **111,480-item catalogue / 232,904 Val events / 124,416 users**, canonical T0, identical frozen encoder revision, unchanged uniform/90-day pooling and pre-T0 popularity. This full Val population includes pilot users; it is follow-up validation, not an untouched independent holdout. No recommender training, half-life tuning or new Test evaluation.

| Full-catalogue retrieval | Recall@200 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| Pre-T0 popularity | 0.026985 | 0.004155 | 0.001655 |
| Frozen text mean | 0.034183 | 0.005675 | 0.002920 |
| Frozen text, 90-day recency | 0.054746 | **0.011506** | **0.006284** |
| Archived two-tower text, three-seed mean | **0.068630** | 0.010550 | 0.005695 |

Recency minus uniform text: **+0.020564 [0.019682, 0.021512] Recall@200**, **+0.003363 [0.003096, 0.003623] nDCG@10**. Recency minus archived two-tower text: **-0.013883 [-0.015333, -0.012485] Recall@200**, **+0.000956 [0.000318, 0.001643] Recall@10**, **+0.000589 [0.000166, 0.000992] nDCG@10**. Intervals remain unadjusted paired-user bootstrap, B=1000, seed 0. Archived entries average event metrics across existing seeds 0/1/2 before bootstrapping. Architecture, objective and encoder artifacts differ, so the archived comparison does not isolate a causal recency effect on the trained model.

The novelty split explains the aggregate trade-off:

| Target group | Method | Recall@200 | nDCG@10 |
|---|---|---:|---:|
| New in Val | Frozen recency | 0.065043 | 0.007611 |
| New in Val | Archived two-tower text | 0.026315 | 0.000352 |
| Warm in Val | Frozen recency | 0.050574 | 0.005606 |
| Warm in Val | Archived two-tower text | 0.087112 | 0.008098 |

For new targets, recency-minus-archive nDCG@10 is **+0.007259 [0.006751, 0.007769]**; for warm targets it is **-0.002491 [-0.002975, -0.001997]**. These are target-conditioned diagnostic groups, not an inference-time router. Merging the methods would require actual candidate identities/scores under one budget; separate target ranks cannot establish fusion gains. Results justify investigating generalisation and objective/history choices, rather than assuming a more complex retriever is better.

All 100 independent float64 query reconstructions match target ranks exactly, and top-10/top-200 hits agree. The full audit verifies all 232,904 histories are strictly prior and target-free, all saved top-ten lists respect availability/full-history exclusion, and source/input/vector/rank hashes match. Encoder parameter hashes are unchanged. Catalogue encoding took **84.6 s** on MPS; complete validation evaluation took **22.1 s popularity / 40.2 s uniform / 41.4 s recency**, including batch work and metric aggregation. These are offline timings, not single-query serving latency or a complete cascade benchmark.

Fresh output: `results/local_experiments/20261102-full-frozen-text/`; use `report-final/report.md`, `report-final/comparison.csv` (including months/groups), `report-final/cases.json` (20 fixed-before-results examples), `evaluation/benchmark.json`, `evaluation/independent_recency_check.json` and `final_audit.json`. The first report attempt stopped during example export; its partial CSV is retained under `report/`, with final findings sourced from `report-final/`. Saved scores were unaffected. Final suite: **156 passed, 4 skipped**. Full-data learned-model training remains a separate decision.

### Frozen diagnosis of lost new-item signal

The user selected this follow-up after the full-catalogue check. `cascade.pipelines.content_signal_diagnostic` evaluates the **saved sample content-only seeds 0/1/2**, with seven fixed probes per seed. The probe manifest, geometry samples and neighbourhood queries were written before scoring. No training, checkpoint reselection, half-life tuning or Test evaluation. Inference substitutions locate signal loss in saved functions; they do not isolate the causal effect of a loss, optimiser or training policy.

| Frozen function | Overall nDCG@10 | New-in-Val nDCG@10 |
|---|---:|---:|
| Raw text + recency/cosine | 0.012489 | 0.013470 |
| Initial shared projection + recency/cosine | 0.009796 | 0.010108 |
| Trained shared projection + recency/cosine | 0.006607 | 0.007400 |
| Trained shared projection + recency/dot | 0.001310 | 0.001172 |
| Trained sequence/dot (original content model) | 0.001836 | 0.001112 |
| Trained sequence/cosine | 0.002575 | 0.001396 |

**The learned item space retains less of the transferable raw-text signal in the pooling control.** With recency/cosine pooling fixed, final-minus-initial projection nDCG@10 is **-0.003190 [-0.005300, -0.001180]** overall and **-0.002708 [-0.005145, -0.000342]** on new-in-Val events. Initial projection versus raw text also loses **-0.002692 [-0.004107, -0.001287]** overall. Initialisation is recreated deterministically from the recorded projection seed; both projected spaces have width 64. This is a contrast between two frozen representations inside identical pooling, not the performance of an untrained versus trained sequence model.

Geometry corroborates a change in structure: mean raw top-20 neighbour preservation for fixed post-T0 probes falls from **42.3% initial to 16.6% trained**. References are pre-T0 products only, with self excluded. In the fixed pre-T0 geometry sample, centred covariance effective rank falls from **38.1 to 10.5** out of 64 dimensions. Random-pair cosine falls rather than rising, so do not describe every product vector as collapsing to one common direction. Neighbour overlap measures preservation of MiniLM structure, not known item relevance or a requirement that all training preserve it.

**Dot scoring makes candidate norm a strong preference.** For the same final projected recency query, cosine-minus-dot nDCG@10 is **+0.005296 [0.003596, 0.007206]**, or **+0.006228 [0.004050, 0.008636]** for new-in-Val events. Query normalisation is a positive per-query scale and does not change ordering; candidate normalisation removes item-specific magnitude effects. Global top-1%-norm products occupy **17.0%** of dot-product recommendation slots versus **0.25%** with cosine. Dot/cosine lists cover approximately **3,398 / 15,483 distinct products**, respectively. These slot measures describe concentration, not relevance of alternatives. Norm/pre-T0-review-count Spearman is approximately **-0.05**, so simple popularity amplification is not established as the cause.

**The sequence model is useful relative to a matched uniform-pooling control.** Native sequence/dot beats projected-mean/dot by **+0.001053 [0.000280, 0.001930] nDCG@10** overall. Sequence/cosine minus projected-mean/cosine is inconclusive (**+0.000182 [-0.001232, 0.001530]**). It is therefore unsupported to assign the entire raw-text gap to the transformer. Recency pooling and sequence output are different query representations, and bypassing the learned sequence changes its inference distribution.

**Normalising at inference is insufficient as a new-item remedy.** Native sequence cosine improves overall nDCG@10 by **+0.000739 [0.000049, 0.001464]**, but the new-in-Val difference is **+0.000284 [-0.000293, 0.000834]**. Normalising each history item before projected recency pooling has no clear gain (**-0.000206 [-0.000659, 0.000135]** overall). Preserve these negative/inconclusive outcomes.

All intervals use 1,000 paired-user draws, seed 0, unadjusted after training-seed event averaging. All 21 archives align on 4,825 events and pass availability, uniqueness and full-history filtering. Source/input/weight hashes and unchanged parameters pass. Native ranks/top-ten lists reconstruct exactly on 128 queries per seed. **159 tests passed, 4 legacy sample-embedding tests skipped.** No model was retrained. Artifacts: `results/local_experiments/20261102-sasrec-pre-t0/diagnostics/20261102-frozen-decomposition/`, including `report/`, geometry/neighbour CSVs, `audit/verification.json`, norm-slot evidence and `figures/frozen_diagnosis.png`.

These findings motivated the candidate-normalised content-SASRec pilot subsequently approved and completed below: normalise candidate vectors consistently inside BCE and final scoring, leave query/history normalisation unchanged, and match the existing sampler, pairs, seeds, backbone and selector. A separate geometry-preservation intervention still needs its own design. The frozen results do not establish whether BCE, optimisation or the selector caused the learned geometry change.

### Approved candidate-normalised training: completed sample result

The user subsequently approved the matched sample training trial. Opt-in `UnitCandidateSASRec` / `content_unit` uses `q · (e / max(||e||, 1e-12))` in **BCE, native prediction/checkpoint selection and shared final retrieval**. Query magnitude and raw history inputs remain unchanged; there is no new temperature or parameter. The strict sampler, seed-specific initial backbone, first negative traces, atomic exports, mappings, pairs, native recipe and selector rule match the original content controls. No full-data learned-model fit or Test evaluation.

| Method, three-seed means unless deterministic | Recall@200 | Recall@10 | nDCG@10 | New-in-Val nDCG@10 |
|---|---:|---:|---:|---:|
| Original dot-trained/scored content model | 0.056750 | 0.003862 | 0.001836 | 0.001112 |
| Original weights, frozen cosine scoring | 0.060244 | 0.005415 | 0.002575 | 0.001396 |
| Candidate-normalised training and scoring | **0.075097** | **0.008298** | **0.003978** | **0.002548** |
| Frozen text + 90-day recency reference | 0.100990 | 0.022309 | 0.012489 | 0.013470 |

Treatment minus original dot: overall nDCG@10 **+0.002142 [0.000814, 0.003608]**, Recall@200 **+0.018347 [0.011714, 0.024630]**. New-in-Val nDCG@10 **+0.001436 [0.000356, 0.002610]**, Recall@200 **+0.015421 [0.008113, 0.022083]**. It also beats changing the original model's scoring only: nDCG@10 **+0.001403 [0.000153, 0.002757]** overall and **+0.001152 [0.000050, 0.002325]** on new-in-Val events. These are unadjusted exploratory paired-user 95% intervals (B=1000, seed 0, event metrics averaged over seeds); the last comparison has a small positive lower bound and is not a final confirmatory result.

The pipeline intervention provides evidence that consistent candidate normalisation can improve this sample model, including new-item retrieval. It does not isolate the effect of the loss from checkpoint selection or number of updates. Selected epochs **33/38/24** versus **12/14/24** and training times **77.0/86.6/58.7 s** versus **32.4/37.2/55.9 s** are retained. All fits use MPS and the same stopping rule, not equal update counts. No serving latency was measured in this trial.

Geometry is partly preserved: post-T0 raw top-20 neighbour overlap averages **26.2%**, versus **16.6%** in the original trained projection and **42.3%** initially. Pre-T0 centred effective rank rises to **17.4**, versus **10.5** in the original trained projection. Geometry describes projected vectors before candidate output normalisation; neighbour comparisons use cosine. These observations do not establish geometry as a causal mediator.

Frozen recency remains substantially better: treatment-minus-recency nDCG@10 **-0.008511 [-0.011727, -0.005760]** overall and **-0.010922 [-0.014650, -0.007698]** for new-in-Val events. Candidate normalisation addresses part of the observed gap; it is not a complete future-item solution or an automatic full-data launch decision.

Artifacts: `results/local_experiments/20261102-sasrec-pre-t0/unit_candidate_report/`, plus selected weights under `runs/content_unit/seed<seed>/` and actual top-200 IDs/scores under `retrieved/content_unit/seed<seed>/`. Matched-input/recipe/initialisation checks, source/weight hashes, exact standalone reload on 128 queries per seed, unchanged analysis-time parameters and all saved list filters pass. Actual top-200 target membership has zero differences from pessimistic rank-based hit@200 in all seeds. Latest suite: **162 passed, 4 skipped**. Historical controls and the frozen diagnosis remain unchanged.

## Working picture (historical full-data pipeline)

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

### Expanded graph target evaluation (1 November 2026)

The [expanded report](../results/evaluation_completion/20261101-kg-expanded-final/report.md) and its CSVs replay **232,904 events / 124,416 users**, including **156,136 warm events**, from the same verified archived predictions. Every new model difference uses **1,000 paired user-clustered bootstrap resamples, seed 20261027**, after averaging per-event metrics over training seeds 0/1/2. This explicitly requested bootstrap seed differs from the historical tables' seed 0; historical estimates/intervals above remain unchanged. All intervals are exploratory, unadjusted and conditional on those trained runs. Each slice has its own user denominator; do not infer an interaction from significance in one slice and not another.

**Witness multiplicity.** Distinct other-reviewer counts use only pre-T1 edges and exclude the query user's links. The four buckets partition all warm targets:

| Other reviewers | Events | Entered@200 | Retrieval nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 115,573 | 4,191 | 0.001378 | 0.002355 | 0.002313 | +0.000977 [0.000748, 0.001227] | +0.000935 [0.000692, 0.001176] |
| 1 | 17,836 | 1,932 | 0.005509 | 0.008250 | 0.007921 | +0.002741 [0.001636, 0.003813] | +0.002412 [0.001269, 0.003570] |
| 2–5 | 14,643 | 3,488 | 0.019081 | 0.021691 | 0.019272 | +0.002610 [0.000474, 0.004583] | +0.000191 [-0.001923, 0.002207] |
| 6+ | 8,084 | 4,837 | 0.106920 | 0.084446 | 0.065543 | -0.022474 [-0.027760, -0.017029] | -0.041376 [-0.046950, -0.035703] |

The dense group has high retrieval coverage and loses under both rankers. Within **training-head / different known last leaf**, the 6+ group contains **6,195 events**: retrieval 0.100146, MLP 0.078015, DIN 0.059986; MLP change **-0.022132 [-0.028932, -0.015792]**, DIN change **-0.040161 [-0.047145, -0.033665]**. This localises the association further without establishing a causal score penalty or a monotonic degree effect. Raw retrieval scores are unavailable in these target archives.

The separate path-count buckets contain **115,573 / 17,681 / 14,698 / 8,184** warm events for 0 / 1 / 2–5 / 6+. In the 6+ path group, MLP changes nDCG by **-0.021981 [-0.027377, -0.015657]**. All 23 nonempty head × last-leaf × degree cells, and the analogous 23 path-count cells, are saved in [warm_joint_other_reviewers.csv](../results/evaluation_completion/20261101-kg-expanded-final/warm_joint_other_reviewers.csv) and [warm_joint_witness_paths.csv](../results/evaluation_completion/20261101-kg-expanded-final/warm_joint_witness_paths.csv). Known/unknown last-leaf values are retained. Each row also gives users, natural membership, exact per-seed rescue/demotion counts, seed-mean counts and at-risk macro rates. `net_rescue_ratio = (rescued−demoted)/(rescued+demoted)` uses seed-mean raw event counts; it is undefined without churn and is not a user-macro metric contribution.

**Saved-witness age and era.** The profile archive retains the lexicographically first reviewer/history-item witness, not the newest witness or an age distribution. The era below refers to its **target edge**; both witness legs strictly satisfy the snapshot cutoff. These rows therefore cannot establish that recent collaborative evidence itself protects an item.

| Saved witness target edge | Events | MLP change in nDCG@10 [95% CI] |
|---|---:|---|
| Pre-T0 | 18,298 | -0.018049 [-0.020463, -0.015714] |
| [T0,T1) | 22,265 | +0.011463 [0.009951, 0.013232] |
| No witness | 192,341 | +0.001574 [0.001382, 0.001765] |
| Less than 365 days before event | 14,497 | +0.014749 [0.012809, 0.016844] |
| 365–729 days before event | 13,660 | -0.000720 [-0.003145, 0.001767] |
| 730+ days before event | 12,406 | -0.022559 [-0.025321, -0.019642] |

Era and age are separate overlapping partitions, not six mutually exclusive groups. The CSVs include retrieval/MLP/DIN levels, both model intervals and churn counts.

**Full-window category continuity.** `same_leaf_any_history` compares the target with all non-padding items in the model-visible last 50 interactions. A known match establishes true. Without a match, any missing target/history category leaves the flag unknown; false requires a known target and a fully known nonempty window. This conservative treatment prevents missing categories from becoming evidence of novelty. “Novel category for user” means absent from that observed window, not lifetime novelty. Returning requires a known different last category; an unknown last category stays in the unknown continuity group even if an earlier match is observed.

Across all validation events there are **44,157 immediate**, **61,068 returning**, **95,793 novel-in-window**, and **31,886 unknown** events. The original **28,149-event training-head / co-reviewed / different-last-leaf** problem cell partitions exactly as follows:

| Continuity within problem cell | Events | Retrieval nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---:|---:|---:|---:|---|---|
| Returning category | 10,701 | 0.033141 | 0.030913 | 0.026531 | -0.002228 [-0.004934, 0.000616] | -0.006610 [-0.009533, -0.003774] |
| Novel category for user | 13,980 | 0.030794 | 0.027112 | 0.021430 | -0.003681 [-0.006172, -0.001140] | -0.009363 [-0.011696, -0.006875] |
| Unknown metadata | 3,468 | 0.022887 | 0.021958 | 0.015552 | -0.000929 [-0.005147, 0.003078] | -0.007335 [-0.011427, -0.003593] |

The MLP point estimate is negative in both known continuity groups; its returning-category interval includes zero. DIN loses in both. These estimates do not establish a returning-versus-novel interaction. Per-event flags and IDs are saved in `expanded_target_profiles.parquet`; the all/warm/problem-cell CSVs retain supports and both intervals.

**Store × co-review × training-head.** The full joint tables retain the eight Boolean cells and two nonempty unknown-store cells, separately for all events and warm targets. Among warm co-reviewed training-head targets:

| Same store in history | Events | Retrieval nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] | DIN change [95% CI] |
|---|---:|---:|---:|---:|---|---|
| No | 30,525 | 0.029973 | 0.023520 | 0.018519 | -0.006453 [-0.008211, -0.004790] | -0.011454 [-0.013217, -0.009806] |
| Yes | 5,186 | 0.050207 | 0.072233 | 0.065689 | +0.022026 [0.016451, 0.027591] | +0.015481 [0.009524, 0.021227] |

Same-store continuity accompanies gains even within co-reviewed head targets, while the no-store group loses. This is an exploratory conditional association, not evidence that a store feature causally protects a recommendation. Full supports, churn and intervals are in [warm_store_joint.csv](../results/evaluation_completion/20261101-kg-expanded-final/warm_store_joint.csv) and [all_store_joint.csv](../results/evaluation_completion/20261101-kg-expanded-final/all_store_joint.csv).

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

### Expanded retriever co-review gradations (1 November 2026)

The [new retriever table](../results/evaluation_completion/20261101-kg-expanded-final/retriever_gradations.csv) uses **pre-T0** witness counts, not the ranker graph. Seed-mean Text+ID-minus-Text differences use 1,000 paired-user resamples with bootstrap seed **20261027**. The following four rows partition all 232,904 validation events:

| Other reviewers before T0 | Events | Text R@200 | Text+ID R@200 | R@200 change [95% CI] | Text nDCG@10 | Text+ID nDCG@10 | nDCG@10 change [95% CI] |
|---|---:|---:|---:|---|---:|---:|---|
| 0 | 212,022 | 0.037265 | 0.037273 | +0.000008 [-0.000869, 0.000849] | 0.001343 | 0.001214 | -0.000129 [-0.000244, -0.000015] |
| 1 | 8,663 | 0.207662 | 0.255254 | +0.047592 [0.040887, 0.055124] | 0.014007 | 0.019692 | +0.005685 [0.003862, 0.007273] |
| 2–5 | 7,446 | 0.396569 | 0.451804 | +0.055235 [0.047109, 0.063359] | 0.040842 | 0.053371 | +0.012529 [0.009778, 0.015286] |
| 6+ | 4,773 | 0.718867 | 0.729307 | +0.010441 [0.000601, 0.019824] | 0.148279 | 0.164287 | +0.016008 [0.010421, 0.020868] |

The Text+ID recall gain is positive in each nonzero witness bucket, with a smaller point gain at 6+ where Text already has high recall. Without a witness, the Recall@200 interval includes zero, while nDCG@10 decreases slightly. This is a target-profile association, not evidence that the retriever internally traverses these graph paths. The separate pre-T0 path-count partition contains **212,022 / 8,594 / 7,471 / 4,817** events; all Recall@200 and nDCG@10 estimates/intervals are in the same CSV.

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

## Supervisor consultation follow-up: evidence review (1 November 2026)

The local `20261029-consult/` report, popularity CSV, oracle CSV and 25-item case discussion are **draft synthesis**, not a new independently verified evaluation run. Review found contradictions with the owning saved sources: Text+ID overall Recall@200 is **0.072370**, not 0.068089; DIN nDCG@10 is **0.006436**, not 0.006262. Retriever-training-seen targets number **80,709**, not 117,202; pre-T0 co-reviewed targets number **20,882**, not 29,894. New-in-Val and unseen-before-T0 are different sets. The previous consultation summary's subgroup Recall@10 values also disagreed with the original K-sweep CSV. Use the original source tables and expanded graph evidence above.

The consultation's useful popularity/K₁ pattern is supported by the [original sweep group table](../results/ranker_k1_sweep/20261025-k1/groups.csv). These are descriptive three-seed means on natural lists; training-head in the graph is distinct from the canonical warm-head segmentation below.

| Target segment | Events | Metric | K50 | K100 | K200 | K500 |
|---|---:|---|---:|---:|---:|---:|
| Warm-head | 109,765 | nDCG@10 | 0.015532 | 0.015044 | 0.011538 | 0.002863 |
| Warm-head | 109,765 | Recall@10 | 0.027473 | 0.027670 | 0.023407 | 0.006533 |
| New-in-period | 76,768 | nDCG@10 | 0.001875 | 0.002497 | 0.002820 | 0.001915 |
| New-in-period | 76,768 | Recall@10 | 0.003845 | 0.005161 | 0.005755 | 0.003943 |
| Warm-tail | 46,371 | nDCG@10 | 0.000623 | 0.000992 | 0.001175 | 0.001030 |
| Warm-tail | 46,371 | Recall@10 | 0.001118 | 0.001878 | 0.002371 | 0.002269 |

Warm-head nDCG peaks at K50 while its Recall@10 peaks at K100; new/tail metrics peak at K200. Aggregate nDCG still peaks at K100, without a final K selection. The [archived rank-band scores](../results/ranker_diagnostics/20261025-score-diagnostic/rank_band_scores.csv) support the observed low-score/logit pattern; they do not by themselves establish that forced-list training caused it.

**Oracle and routing status.** The draft values 0.075878 / 0.007442 / 0.00756 and its latency-saving claim lack a reproducible full-list evaluation and paired-user uncertainty. Choosing a scorer from the true target's group is a target-conditioned oracle diagnostic. It cannot choose the scorer for a query before the target is known, or produce a candidate-level mixture by selecting only target ranks. Group means also cannot be weighted by event shares to recover user-macro performance. An oracle diagnostic needs aligned per-event metrics, full-population user weights, paired uncertainty and disclosure that cell choices were made on Val. A practical router additionally needs an inference-valid query gate or a calibrated candidate ordering and actual candidate matrices. No deployable routing gain is established.

**Secondary metrics and cases.** Hit@3/MRR@10 can be computed from archived ranks but were not verified in a saved reproducible output here. The 25-item draft mixes five source-bound validation examples, historical score-diagnostic cases and metadata illustrations; it is not a traceable 25-event audit of ranker overtakers. Retain the five verified cases above. Future expanded cases must satisfy Step 5 and the device gate, without treating parser errors, compatibility or user intent as known.

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
| `Same-store` | Candidate's known store appears in history; store is not assumed identical to brand. Retain the marginal signal alongside the authorised exploratory store × co-review × training-head target table. |
| `Head item` | Top 20% of positive-count items in the permitted training graph, count descending/item-ID ascending. This training-head signal is separate from canonical pre-T2 head/tail reporting. |
| `Text-similar` | Cosine similarity of frozen candidate MiniLM text to the mean of available history text vectors exceeds a Val-fixed threshold; do not use learned ranker scores to define it. Missing/zero-vector text is unknown. |

**Definition freeze before Test.** Save the parser/rule version, training-graph cutoffs, similarity aggregation and numeric threshold, reference-pool sampling method/seed and artifact hash, and missing-value handling. Evaluation chooses the text-similarity threshold from the Val reference distribution and records its rationale before applying it to Test; it is not yet a measured value. Audit a reproducible Val title sample for device-extraction correctness as well as coverage, by cold/warm status. If device evidence is inadequate, label the device hypothesis untested; text similarity alone cannot establish compatibility.

**1 November target-only expansion.** Witness degree/path bins are fixed at 0, 1, 2–5 and 6+; saved-witness target-edge age is <365 / 365–729 / 730+ days before the event, with a separate no-witness group and pre-T0 / [T0,T1) era view. Do not interpret the retained deterministic witness as the latest witness. Ranker edges must be strictly before **T1=1628643414042**; retriever edges must be strictly before the verified **T0=1595800947930**, as bound by `split.json`, the graph manifest and previous provenance. The task's alternate T0=1598016867874 is not the cutoff used to construct these profiles and is not substituted. This changes no training/evaluation split. Last-50 histories are reconstructed from the verified sequences because the requested standalone `val.npz` is absent; catalogue fields are in `products.parquet`, not `product_attributes.parquet`. New paired intervals use bootstrap seed **20261027** by explicit request; the historical protocol seed 0 remains applicable to historical tables.

Full-window category matches preserve unknowns: no match with incomplete category coverage is unknown, and a “novel” label refers only to the observed window. The warm joint degree/path tables preserve unknown last-leaf cells; the store tables preserve unknown-store cells. These expansions describe target outcomes and do not complete candidate-slot amplification or identify ranker top-three overtakers. **Steps 4A/4B candidate overlays and Step 5 overtaker inspection remain blocked** until `item_text_emb.npy` and `data/processed/candidates/two_tower_text_seed0/val.npz` are restored.

**Device audit gate.** Never tune regexes on `device_audit_locked.csv`. Do not enable `Same-device` in downstream evaluation tables unless `assess_device_audit` passes on **100 locked human-reviewed titles with positive-edge precision ≥0.95**. The 1 November gate is still **blocked: 0/100 reviewed**; neither audit file was edited. Development statuses are **30 unknown / 20 parsed / 19 ambiguous / 18 universal / 13 negated**, with all 100 review labels blank. Slash-separated models/variants (for example iPhone SE/5/5S and iPhone 7/8 Plus), universal claims and negation clauses need human review. Without gold labels, these are ambiguity/review patterns, not measured false-positive rates. No parser tuning or device-based evaluation was performed.

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
- Graph work explains relationships only. Device-title extraction requires the locked audit before quantitative device claims or model-feature integration. Graph paths are not causal model explanations. The architecture review in `architecture_proposals.md` recommends completing these existing controls first; new graph features, monotonic score residuals and deployed routing/merging require a separate decision. Random score/rank mixing is not an additional authorised arm.

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

Recover the **original** `data/processed/item_text_emb.npy`, `item_text_emb.json`, and `data/processed/candidates/two_tower_text_seed0/{ranker_train.npz,val.npz,val_target_score.npz,meta.json}`. The restored structural data preserve original item/user mappings; they do not replace these missing artifacts. Check original hashes before claiming historical reproduction. User handles SSH and cluster commands; no teammate allocation is assumed available.

Until recovery, target-only graph findings cannot establish candidate-slot amplification, and saved target ranks cannot identify the MLP/DIN top-three products. Complete the locked annotation before device claims. Final Test and index/cost work stay paused. The ablation runners have not produced full-data results; the updated submission script runs blocking input checks and exports comparison reports. Cluster execution has not been verified here. Earlier implementation-review/test counts are historical, not verification of the final source snapshot.

## Evidence and reproduction

The narrative above is self-contained. Compact CSV/JSON evidence is retained for audit; agents do not need to read generated report Markdown to determine status. Large generated graphs, copied result packages, local processed data and development outputs stay local/ignored. Historical source artifacts remain unchanged.

| Evidence | Location |
|---|---|
| Saved-validation aggregates, seed/group/funnel/CI tables and rank reconciliation | `results/saved_evaluation/20261027-final-validation/` (CSV/JSON only in this handoff) |
| Original candidate identities used by the prerequisite checker | `results/saved_evaluation/20261027-saved-validation/provenance.json` |
| Verified marginal target/history profiles | `results/explanation_graph/20261027-target-profiles-verified/` (compact tables/provenance) |
| Joint 12-cell warm profiles, retrieval profiles, five cases and contribution reconciliation | `results/evaluation_completion/20261028-final/` (CSV/JSON) |
| Expanded witness degree/path/age, category continuity, store intersections and retriever gradations | `results/evaluation_completion/20261101-kg-expanded-final/` (CSV, report and provenance; per-event Parquet stays local) |
| Architecture evidence review and proposals awaiting decisions | `docs/architecture_proposals.md` |
| Device audit samples, still unlabelled | `results/explanation_graph/20261027-base/device_audit_{development,locked}.csv` |
| Historical checkpoints/candidate hashes and exclusions | `results/ranker_artifacts.json` |

To replay archived validation predictions without training, original processed data or Test prediction loading:

```bash
.venv/bin/python -m cascade.eval.saved_results --run-id YOUR_FRESH_RUN_ID
```

Use a fresh ID; existing outputs are preserved. Config: `configs/evaluation/saved_results.yaml`. `--sample` is a development subset and must never be reported as a full-data result. Graph/profile reproduction additionally needs the original/restored processed inputs; the one-off `notebooks/complete_evaluation.py` records the completion analysis, with local input/output paths that must be adjusted for a fresh run. Do not launch retraining merely to reproduce these saved-result findings.

Replay the expanded graph evaluation from the repository root with `python -m cascade.pipelines.expand_kg_evaluation` in the project environment. It verifies historical profile/rank/structure hashes and creates a fresh timestamped directory. Its `--sample` option is a 512-user development replay, distinct from the training pipeline's 2% sample; neither sample is reportable as full data. Device audit inputs are read-only and the gate status is recorded without enabling device evaluation.
