# Design decisions and deviations

Every non-trivial choice or deviation from CLAUDE.md, with date and reason.

## 2026-11-11 — Make the code reviewable from a fresh checkout

The user requested the code portion of the review cleanup. Correct package metadata to Python 3.11 and declare the existing 18 dependency pins from `requirements.txt`, with an explicit setuptools backend. Add one local check command and a CI workflow for code/evidence; hosted execution remains pending a push.

Thirteen core tests previously skipped without prepared sample data. Run the same checks on an isolated synthetic dataset as well as the optional real sample. Use production split/instance/read/filter/candidate paths, random test vectors and no downloaded model. Preserve research inputs and predictions. Add non-writing companion export freshness checks, using `math.fsum` to remove Python-version differences in seed-mean rounding while preserving the committed bundle bytes.

Document and test the existing undefined relative-bootstrap behaviour for zero baselines, suppressing only the expected numerical warnings without dropping draws or changing finite calculations. Leave original reports intact. Record the remaining large-runner, repeated-recipe and artifact-manifest refactoring in `docs/code_review.md`; this cleanup does not rerun training or Test scoring.

## 2026-11-11 — Prepare the repository for professor review

The user authorised the first review-cleanup batch. Reconcile the local companion/correction work with main's later R1 evidence on `codex/review-ready`, preserve the old README under `docs/history/`, and provide a concise entrance, review guide and claim-to-code/checkpoint evidence map. Keep original poster assets, raw result tables, registered protocols and hashed source snapshots intact.

Update the companion to the corrected RQ3 scores and measured R1 Test follow-up. Recompute matching-family aggregate levels and paired intervals from existing original/corrected ranks only, using the existing macro metric/bootstrap and subgroup definitions; record input/output/script hashes in a separate `matching_report` directory. This executes no models and creates no new Test predictions.

Qualify claims by model family and evaluation period. The A/B split and frozen snapshots can amplify the observed dependence; consistently updated/shared-period training remains untested. Mixed lists can help frozen rankers, the fresh-MLP Test decline is inconclusive, and the R2 decomposition is descriptive. Repair stale current status and missing-report links while retaining dated history. Publication and further experiments are outside this cleanup.

## 2026-11-08 — Correct mixed-list Test score normalisation

Jing Yen requested continuation of Claude Code's unfinished evening check on xlogin. QR had identified a training/Test scaling discrepancy. Inspection confirmed that `run_hybrid_study.py` fits RRF/RRFL retrieval-score mean and standard deviation on natural-only ranker-training lists, whereas both Test scorers used all ranker-training lists. The correction is to reuse the training population and identical float32 reductions at inference.

- No retraining, checkpoint selection, candidate regeneration, split, label, filtering or metric change. Apply the same correction to every affected evaluated arm (`mlp_rrf`, `din_rrf`, `din_rrf_res`, `din_rrfl`), seeds 0/1/2, irrespective of the outcome.
- First run 128-list smoke checks, then full frozen-checkpoint replays. Require exact reproduction of saved Validation ranks with the training scaler and of historical Test ranks with the old scaler before accepting corrected Test ranks. Check model SHA-256 before and after scoring.
- Keep historical results intact. Save the audit, corrected ranks and paired-user bootstrap comparisons in `results/test_final_ranker/20261108-normalization-corrected/`; smoke outputs are separate. This is a disclosed correction after Test exposure, not a new untouched Test or a tuning opportunity.
- Centralise the training-only scaling formula, record scaling alongside future hybrid fits, and make Test scorers refuse overwriting earlier evidence. Cluster job `1097334` runs the smoke and full audit sequentially on xlogin's GPU partition. Numerical outcomes are recorded in `docs/evaluations.md`.

## 2026-11-07 — Locked Test run of the 5–7 Nov systems (approved)

Brandon approved the one-time Test run in `docs/test_protocol_2026-11-07.md` (sha256 `ed463472b4f437ae27a768975fbc622b69a2a49b2d5792cd3cf60be45fd30ac7`, frozen at approval), including decisions 2 (T2 statistics) and 3 (monthly statistics on Test).

- Frozen models only: no training, tuning or selection on Test. Each system is scored once and every result is reported.
- New outputs only, in `results/test_final/20261107/` and `results/local_experiments/20261107-test-final/`. No earlier study, result or file is modified.
- This supersedes, for these systems only, the 4 Nov restriction on further Test runs and on a T2 graph.
- **Outcome (7 Nov, 20:07–20:19):** the Val reproduction gate passed (lists, features and all six models' ranks exact; first attempt caught a 1-row CPU/MPS tie for D1, fixed by scoring each model on its original evaluation device). Every system was scored once. 8 of 9 written predictions held; 4b (combined system does not help new targets vs S1) did not: +0.000948. Results: `results/test_final/20261107/`.
- **Correction (7 Nov, after the run):** the protocol's rationale for prediction 2 says 18% of Val targets are new; the correct share is 33% (76,768 of 232,904; Test 44%). The 18% was the share of top-10 *slots* filled by new products in an earlier analysis. The prediction itself (the mixed-list gain is smaller on Test) and its verdict are unaffected. The frozen protocol file is left unchanged.
- **Addendum (7 Nov, approved by Brandon):** a Test replication of the Val statistics-timing diagnostics on the four frozen T0-trained P1 rankers. Protocol `docs/test_protocol_2026-11-07_addendum_timing.md` (sha256 `cd8228c4ea766fdb767e9081476ae58d979f47d0d6ede2b73089a49237a301a9`), with predictions written before any Test arm was built. No system or earlier result changes. **Outcome:** all five predictions held (`results/test_final/20261107/timing/`).

## 2026-11-04 — Close the objective x co-review factorial

Three fresh BCE+co-review seeds complete root plan §14 using the exact immutable parent inputs and nine locked control fits. Selected epochs are 8/9/8. Full-Val nDCG@10 is **0.008409**, compared with BCE-off **0.008115**: graph gain **+0.000295 [0.000169, 0.000425]**. Warm gains accompany new-item losses under both objectives. The direct interaction `(PW_on−PW_off)−(BCE_on−BCE_off)` is **+0.000003 [−0.000159, 0.000163]**; the joint-user new-minus-warm interaction is **+0.000146 [−0.000162, 0.000511]**. Neither establishes objective dependence or equivalence. Preserve the whole training/selection/refit-recipe interpretation and exploratory/unadjusted reused-Val limitations.

All new checkpoints/logits remain local; compact four-cell/group/funnel/contrast/source/provenance evidence and the checked interaction figure are shared. The exact numerical-analysis source snapshot is retained separately from a later label-formatting adjustment. The sample off-control reproduces the earlier BCE first epoch. Final checks: **233 passed / 4 legacy skips**, with 33 inner/three outer score replays, twelve factorial locks, 675 reconstructed metric rows, eighteen direct/main intervals and one jointly resampled novelty interval independently passing. Commit the results and documentation. This completes the authorised extension; no automatic graph retrieval, new encoder, canonical integration, serving/index work or Test launch.

## 2026-11-04 — Complete the controlled objective x co-review question

Following discussion of the original design, the user authorised proceeding with the missing BCE+co-review comparison. Root plan §14 registers three new full-data fits; the nine existing BCE-off/pairwise-off/pairwise-on controls, candidates, graph, feature normalisation, split, initialisation and negative plans remain immutable. The question is when additional relational evidence helps, including objective and novelty interactions, rather than requiring a new model to win overall. Direct paired-user factorial and joint-user new-minus-warm interaction intervals are registered before fitting. Final Test, graph retrieval and further architectures remain separate decisions.

## 2026-11-04 — Complete the full-data replication and qualify the sample conclusions

All twelve registered fits and full-Val scoring completed on the unchanged canonical split and frozen-recency natural K200 pool. BCE text/common control nDCG@10 is **0.008115**, pairwise text **0.007767**, pairwise co-review **0.008065**, combined **0.007930**, versus retrieval **0.006284**. Pairwise loses to BCE overall/warm while improving new items. Co-review improves its pairwise control overall/warm but loses new-item quality; additional category/store-history fractions lose overall/new quality versus co-review. Full BCE selects trained epochs 10/6/6 rather than the sample's epoch zero; different populations and update exposure prevent assigning this to data quantity alone.

Missing arm-pair comparisons are explicitly secondary, not registered primary hypotheses. BCE-minus-retrieval is **+0.001831 [0.001519, 0.002117]**; co-review-minus-BCE is inconclusive. Aligned historical context shows a gain over archived MLP K200/SASRec but no established superiority over archived MLP K100's observed validation maximum. All intervals remain exploratory/unadjusted on reused Val, conditional on the existing seeds/candidates. No final architecture or candidate count is selected from these results.

Share the compact report/CSV/config/environment/source/provenance/audit bundle and checked figure; keep bulk arrays, logits and weights local. All 548,018 candidate lists, 104 retained inner score sets, twelve outer outputs/locks, 675 metric rows and eighteen main nDCG intervals independently pass. Final unit suite: **229 passed / 4 legacy embedding skips**. Update README/CLAUDE/plan/evaluations/proposals and commit. This completes the bounded request; no automatic further training, changed-pool integration, serving/index work or held-out Test launch.

## 2026-11-03 — Resume for bounded full-data replication

After reviewing the completed evaluations, the user requested full-data testing of the most promising methods, adding results and committing. This authorises root plan §13: matched BCE text and pairwise text/co-review/combined residual rankers, three seeds, frozen-recency natural K200 candidates, canonical full-data split, chronological B tuning and fixed-epoch refits. It supersedes the previous experiment pause for this study. Full Val is reused exploratory evidence; held-out Test and the independent historical policy/dropout/index studies retain their separate status. Preserve every original study and teammate checkpoint. Register the configuration before fitting and lock every selected weight before outer scoring; include negative outcomes and warm-item guardrails.

Names follow the 2026-10-20 renaming (see the last entry): **retriever-training period** (formerly period A),
**ranker-training period** (period B), **all training data** (A ∪ B), **Two-tower (text)** (v2), **Two-tower (text + ID)**
(v4), **sample** (toy). Commits and run files from before that date use the old names.

## 2026-11-02 — Independent local experiments and future-item controls

The user authorised an independent local track in root `plan.md`, starting from the original RecBole baseline, using MPS and pausing at consequential choices. This extends the historical narrowed scope for separate local experiments; it does not replace teammates' controls or open final Test evaluation.

- After the sampler problem was explained, the user selected **option 1: pre-T0-only negatives** for fresh matched ID/content controls. The original RecBole vocabulary includes future products. Exposing their text as negatives would test a known-catalogue setting. Keep the original fits as historical references; match fresh BCE, training pairs, backbone, selector and final evaluation. Sample pool: 20,740 allowed / 9,649 excluded items. Initial shared backbones and initial negative traces match within seed.
- The local programme completed ID/content/content+ID/graph/two-interest retrievers at seeds 0/1/2; mean/semantic-attention rankers at seeds 0/1/2; actual unique-budget RRF; frozen history and fixed 90-day text controls; popularity; twenty fixed examples; and a full archived-Val monthly audit. All learned sample fits use MPS. Natural rankers use only 458 retrieved training positives, with no insertion or retrieval-score input. These are new models, not reproductions of the historical DIN or two-tower architecture.
- Frozen text with recency weighting is the strongest sample control. This is a pretrained MiniLM encoder with **no task-specific recommender training**, not an untrained language model. The 90-day half-life was specified before outcomes. Graph/two-interest/fusion/ranker variants do not supply a clear gain over their controls. Preserve negative results. See [evaluations.md](evaluations.md#independent-local-experiments-2-november-2026).
- The sample catalogue/availability process and 67.2% new-in-Val target share differ from full data. No numerical extrapolation or claim that graph relations/attention caused an effect. Graph also changes capacity; the two interest heads are similar. Natural-only ranking changes both policy and support relative to historical training; this is not the teammates' controlled policy ablation.
- The user paused, then resumed after the recommendation to check frozen text on the full catalogue. Proceed with the same pinned encoder and fixed uniform/90-day pooling, all original Val events and canonical full split, against pre-T0 popularity and archived retrieval references. Encoding/inference does not retrain a recommender. Preserve a benchmark and independent float64 check. No half-life search, learned full-data fit, candidate router or Test launch follows automatically.
- Full frozen-control validation completed: recency improves nDCG@10 relative to archived three-seed two-tower retrieval (**+0.000589 [0.000166, 0.000992]**) while reducing Recall@200 (**-0.013883 [-0.015333, -0.012485]**). Strong new-item gains coexist with warm-item losses. Preserve this trade-off; do not select a target-conditioned router or claim causal superiority of recency within the trained model. The full Val population overlaps the pilot users. Full audits and independent rank reconstruction pass; no recommender training or Test evaluation. Details and final artifact pointers are in the evaluation record.
- The user then chose **diagnose why training loses new-item signal on the sample**. Complete 21 frozen, predeclared projection/pooling/norm probes across saved content seeds 0/1/2, with no retraining. Final-versus-initial projection loses transfer quality under identical recency/cosine pooling; candidate magnitude strongly affects dot-product lists. Native sequence scoring normalisation improves overall quality but does not clearly improve new-item quality, and the sequence model beats uniform pooling in its own dot space. These locate effects in frozen functions, not the causal role of BCE/optimisation/selection. Do not call the transformer alone the cause, interpret lower effective rank as all directions becoming identical, or equate norm effects with popularity bias. The next candidate-normalised training pilot is a proposal requiring a new go-ahead. See the evaluation record and root plan §7 for evidence and exact limits.
- After that diagnosis the user explicitly approved **matched sample candidate-normalised training**, seeds 0/1/2 after a separate smoke. Normalise only the candidate vectors in BCE, native selection/prediction and shared scoring; leave the query magnitude, history inputs, seed/negative policy, pairs, backbone, selector and original controls unchanged. No temperature parameter. `content_unit` is opt-in with unchanged checkpoint keys. Compare against dot-trained native scoring and its existing frozen cosine probe, with raw recency as a reference. Full-data learned fits and Test remain paused. Root plan §8 records progress.
- Candidate-normalised sample training completed on MPS. Overall nDCG@10 **0.003978**, compared with original **0.001836** and frozen cosine **0.002575**; new-in-Val **0.002548**, compared with **0.001112 / 0.001396**. Paired intervals support improvements, with all matched-input and output audits passing. Frozen recency still leads by a substantial margin. This is a pipeline intervention including training/selection, not an equal-update loss-only comparison or proof of a geometry-mediated mechanism. Keep all seeds and selected epochs; do not replace canonical full-data controls or start full-data training automatically. Details are in the evaluation record and root plan §8.
- The user requested **complete all plan.md work** after the status audit identified five unfinished evidence tasks. Those are now complete without training: monthly/history intersections on both populations; exact local ranker failure/rescue/demotion analysis; fusion-versus-both-source CIs using actual 200-item lists; frozen recovery of six raw candidate / twelve ranker-logit archives with exact Val replay; and one consolidated report with twenty fixed examples, composition/concentration, monthly catalogue introductions, costs and audits. Use `results/local_experiments/20261102-plan-completion/report-final/`; root plan §9 records closure. Fusion beats graph but not clearly content. Rankers demote 13 original top-ten targets on average while rescuing 9 attention / 8.67 mean. Source/support/funnel checks pass; 168 tests pass with four legacy skips. Nominate native content versus candidate-normalised content as the later matched full-data pair, retaining frozen recency as reference. No new architecture, full-data fit, changed-pool ranking, serving study or Test launch is implied by closing the local plan.

## 2026-11-01 — Expanded graph findings and architecture proposal review

- Expanded graph target evaluation and the architecture evidence review are complete. The existing authorised experiments below define the next work; this review adopts no new architecture or experiment arm.
- Expanded target-only evaluation covers 232,904 validation events / 124,416 users. Retain verified graph cutoffs T0=1595800947930 and T1=1628643414042; the task's alternate T0 is inconsistent with the source graph. New intervals use the explicitly requested bootstrap seed 20261027, B=1000; historical tables retain seed 0. Histories are restored from the verified last-50 sequence pointers, with category/store unknowns preserved. See [evaluations.md](evaluations.md#expanded-graph-target-evaluation-1-november-2026) for all estimates and caveats.
- The graph finds substantial losses on dense co-reviewed targets and opposite-sign outcomes with versus without store continuity within co-reviewed head targets. These associations motivate experiments; they do not establish feature causality, a GNN equivalence or a guaranteed fix.
- Existing authorised work stays first: matching-MLP policy arms A/B/C/D and the separate ID-dropout p=0.5 control/treatment study, following the acceptance and replication rules in [evaluations.md](evaluations.md#authorised-follow-up-controlled-ablations-not-results). No full-data remedy result exists. Preserve the frozen text seed-0 candidate source and unchanged control feature/checkpoint semantics. Random score/rank mixing is not an implemented or authorised additional arm.
- [Architecture proposals](architecture_proposals.md) now distinguish hypotheses from evidence. Graph/history inputs, a monotonic score residual, degree-gated IDs and inference-time routing/merging remain open decisions. GNN/LLM GraphRAG and multi-source retrieval merging remain outside current scope. No model code was changed by the architecture review.
- A target-conditioned oracle uses the true target's group and separate scorers' target ranks. It is a diagnostic, not one deployable ranked list. Draft routing gains/latency claims are unverified; candidate-level mixing needs comparable scores and full-list evaluation. Final Test, index work and full-cascade timing remain paused.
- Candidate overlays/overtakers still require original candidates and text vectors. `Same-device` remains gated by 100 locked human-reviewed titles at precision ≥0.95; 0/100 are labelled. No locked-set tuning or device features are enabled.

## 2026-10-29 — Supervisor consultation: diagnostic framing and follow-up

The consultation notes motivate popularity/K₁ breakdowns, complementary ranker analysis, secondary metrics and a broader case discussion. They do not establish new architecture results. Duplicate draft entries were consolidated on 1 November.

- Treat forced-positive score inversion as a pipeline diagnostic and hypothesis for controlled retraining. It is a constructed-list distribution mismatch; the existing evidence does not establish leakage from held-out Test into training or isolate force-insertion as the cause of the learned response.
- All headline ranker results use natural unforced candidate lists. The original popularity-segment K₁ sweep and archived ranker comparison remain the numerical sources; see [evaluations.md](evaluations.md#supervisor-consultation-follow-up-evidence-review-1-november-2026).
- The 29 October consultation report, oracle CSV and 25-item discussion exist as local draft material. Their target-conditioned router estimates, inconsistent counts/metrics and broad case interpretations need verification. Hit@3/MRR@10 were requested, but no reproducible secondary-metric output was found in this review; do not mark them complete or cite the draft as an independent experiment.

## 2026-09-28 — Data

- **Sources.** Interactions: official 5-core rating-only file (URL in `configs/data.yaml`). Metadata:
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Cell_Phones_and_Accessories.jsonl.gz`
  (verified reachable, 858 MB). SHA-256 + download date in `data/raw/MANIFEST.json`.
- **Counts.** 380,999 users, 111,480 items, 2,752,785 interactions (no exact or (user,item) duplicates in the
  5-core file). All 111,480 items have a metadata row.
- **Split cross-check.** Our Val / Test periods contain exactly 239,437 / 304,468 interactions — identical to the
  official `timestamp_w_his` valid/test files, confirming the T1/T2 cut-offs are applied correctly.
- **`t_ranker_train`** (start of the ranker-training period) = 1595800947930 (2020-07-26 22:02 UTC), the 85th
  percentile of pre-T1 timestamps.
- **Leaf category** = the full category path joined with `" > "` (139 distinct), so identical leaf names under
  different parents are not merged. **Top-level category is constant** ("Cell Phones & Accessories") for every item
  with categories → useless as a feature; keep it for completeness but expect zero signal.
- **Price** parseable for only 38% of items → the "missing" bucket will dominate.
- **Item segments / top-1% popularity** use a deterministic tie-break (count desc, item id asc).
- **Cold-item share (important for the critique).** 44.4% of Test instances target an item with no interaction
  before T2 (0% in Val, by construction of the segment definition). ID-only models (Pop, ItemKNN, BPR, SASRec)
  cannot rank these items above others, so their Test recall is capped at ~55.6% of instances. Val and Test are
  therefore not exchangeable: Val under-represents cold items. This is a property of the official split plus the
  5-core filter (items released after T2 must have ≥ 5 reviews, all in Test). Report metrics per segment.

## 2026-09-28 — Evaluation harness

- **Ties** are broken pessimistically: rank = 1 + #(other unmasked items with score ≥ target score). Matters for
  Popularity (many items share a count, e.g. all cold items score 0). CLAUDE.md does not specify tie handling.
- Not-yet-released filter: items with `first_ts > t` are removed; `first_ts == t` is allowed (so a cold target at its
  own first interaction is rankable). `first_ts` is computed over the full timeline (as §4.6 prescribes).
- Filtering uses the user's *full* history before t; models only see the last L = 50.

## 2026-09-28 — Baselines via RecBole (deviation from §6, decided by the user)

- **Decision:** the four baselines are trained with **RecBole 1.2.1** (primary implementation) rather than being
  re-implemented in our codebase. RecBole is therefore no longer an independent cross-check for BPR/SASRec.
- **Evaluation still goes through our harness.** RecBole's own evaluator cannot apply our §4.6 filtering
  (not-yet-released items), per-instance targets with histories that include Val events, or per-user macro averaging.
  Each trained RecBole model is wrapped in a scorer (`src/cascade/models/recbole_baselines.py`) and evaluated by
  `cascade.eval.harness`, like every later model. RecBole's own validation numbers are only used for early stopping
  and are logged as `recbole_best_valid_subset`.
- **Early stopping** uses RecBole's validation on a fixed random subset of 20,000 Val instances (seed 0), metric
  Recall@100 (full ranking, no release filtering). RecBole's "test" file is a copy of the valid file, so Test data is
  never loaded by RecBole.
- **Pop:** RecBole's `Pop` counts items inside `calculate_loss` with `item_cnt[item] += 1`, which (a) also counts
  sampled negatives (negative sampling is on by default) and (b) does not accumulate duplicate items in a batch.
  Verified on the sample data that this changes the results. We use `PopExact` (a RecBole `GeneralRecommender` subclass)
  that counts from RecBole's training interaction matrix; verified identical to an independent count.
- **ItemKNN:** RecBole's `ComputeSimilarity` (cosine, top-100 neighbours per item, shrink 0) on the binary
  user–item matrix. RecBole's `ItemKNN` also precomputes `X·W` for all 381K users (~10⁸ non-zeros), so we use
  `ItemKNNLite`, which keeps only `W`; scores are computed per instance from the last L = 50 history items
  (score_j = Σ_h W[h, j], the same formula as RecBole).
- **BPR:** RecBole `BPR`, 64-d, 1 uniform negative, Adam lr 1e-3, batch 4096, eval every 2 epochs, patience 5 evals.
  Users with no training interactions are unscorable: skipped and counted (`n_unscorable`); metrics with them counted
  as misses are also saved (`with_unscorable_as_miss`).
- **SASRec:** RecBole `SASRec` with 2 blocks, 1 head, d = 64, FFN dim 64 (the original paper's d×d point-wise FFN),
  dropout 0.2, L = 50. RecBole only offers CE and BPR losses; `SASRecBCE` overrides `calculate_loss` with the
  original **BCE with 1 random negative**. RecBole trains one target per row (not every position in one pass), so we
  export every training instance (prefix → next item) as a row: the same set of training pairs, less compute-efficient.
- Training rows for SASRec use histories that may include events from earlier periods (retriever-training-period-only training: retriever-training instances).
- **Deterministic baselines** (Pop, ItemKNN) are run for seed 0 only; their std over seeds is 0 by construction.
- **RecBole anomaly-mode bug:** `recbole/model/sequential_recommender/sine.py` calls
  `torch.autograd.set_detect_anomaly(True)` at import time, which silently enables autograd anomaly detection for
  every model once RecBole's model package is imported (≈2–5× slower training). We switch it off after import.
  Found after the first full launch was far slower than profiled; those runs were killed and restarted.
- **Device:** SASRec trains on Apple MPS (≈0.023 s per batch of 512 vs ≈0.19 s on 3 CPU threads). BPR, Pop and
  ItemKNN run on CPU (RecBole's Pop hits float64 on MPS; BPR is faster on CPU). MPS vs CPU results differ slightly
  (floating-point nondeterminism), so all seeds of one model are run on the same device. Use `--device cuda` on a
  GPU machine.
- **Environment:** Python 3.11, torch 2.5, numpy < 2, pandas < 3 (RecBole 1.2.1 is not compatible with newer).

## 2026-10-06 — Retriever (Stage 1)

- **Text embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, L2-normalised, cached as
  `data/processed/item_text_emb.npy` [N+1, 384] (row 0 = padding) with a text hash in `item_text_emb.json`.
- **Training windows:** each user's retriever-training sequence; every position predicts the next item in one forward pass.
  Sequences longer than L+1 are cut into consecutive windows (stride L) from the end, so every retriever-training target is predicted
  exactly once, but a window's context does not reach back into the previous window.
- Same-timestamp events of one user are ordered by `parent_asin` in training sequences (only evaluation histories
  apply the strict "before t" rule).
- **Attention mask:** causal + key padding; each position may always attend to itself so padded (left) positions do
  not become fully masked (NaN). Pre-LN blocks + final LayerNorm, ReLU FFN (as SASRec), embedding dropout 0.2.
- **ID embeddings** (`id_text`, `id_only`) are multiplied by a "seen in the retriever-training period" mask, so items without such interactions
  get no ID signal (`id_only` gives them a zero vector → score 0).
- **logQ:** `q_j` = static retriever-training frequency; items unseen there get log q = 0 (never used, since training
  targets are retriever-training items).
- **Early stopping:** Val-subset Recall@100 (20K fixed instances, seed 0) through the harness, i.e. with §4.6 filtering.
- **Comparison caveat:** the retriever trains on the retriever-training period only; compare it with the baselines
  trained on that period (e.g. `sasrec/retriever_train`), not the all-training-data ones.

## 2026-10-14 — Scope revision after the critique (slides + professor's consult)

Professor's feedback: the two-stage design and "hold one stage fixed" are sound; the risk is scope. Use fewer
variables, match metrics to what the user sees (@10), explain *why* results change, and after the separate
comparisons test one selected retriever–ranker combination (an underperforming one is fine if explained).

Changes to CLAUDE.md:
- **RQ1** is now "two-tower vs the four baselines as retrievers" (as on the slides), all trained on the **retriever-training period**
  for a fair comparison. Only one ablation is kept: Two-tower (text + ID) (ID memorisation for head items vs text
  for new ones). The other three variant configs are kept but not run.
- **Rankers:** LightGBM and the MLP-without-retrieval-score variant are **dropped**. The advanced ranker is **DIN**
  (chosen over DCN-V2: users are product sequences, and target attention models "which past products matter for this
  candidate", e.g. a case for the phone bought earlier). DIN is now core, not stretch.
- **RQ3 rows** follow the slides: popularity order, retrieval-score order, MLP, DIN on identical candidate lists.
- **RQ2:** MLP ranker only; K₁ ∈ {50, 100, 200, 500} (1000 dropped); HNSW efSearch ∈ {K₁, 4K₁}, IVF nprobe ∈ {4, 32}
  (one fast and one accurate point each). The sweep over all five retriever variants is dropped.
- **New §10.1 final combination test:** frozen retriever + better ranker at the recommended K₁, compared on Test with
  the best single-stage baseline (SASRec on all training data) and popularity retrieval + the same ranker, with bootstrap CIs.
- **Metrics:** @10 is the headline; retrieval Recall@K adds @10 (slides) and larger K only shows the ceiling.
- **New segment `new-in-period`** (target first reviewed at/after the split start). Reason: under the `cold`
  definition Val has 0% cold targets, but 44% of Test targets are new products, so selecting the frozen retriever on
  overall Val recall alone would ignore cold-start. Implemented in the harness; `cascade.eval.resegment` backfills it
  for earlier runs from their saved ranks into `extra_segments.json` (metrics.json is not modified).
- **Roles** updated to the slides (Phua: data/eval/failure analysis; Brandon: retriever + FAISS + sweeps;
  Tong: rankers). Dropped from the plan: DIN + DCN-V2, MMR.
- Schedule kept as on the slides; Phase 1 has slipped (no candidate lists yet as of 14 Oct).

Slide wording to correct in the report/poster: the Popularity baseline counts **all** reviews (not "positive"
reviews); the ranker-training period is 26 Jul 2020 – 11 Aug 2021 (not "2021"); the item text also includes the store name; the slide-6
BPR row skips 74,437 test cases — the comparable row is 0.0027 / 0.0013 / 0.0154.

## 2026-10-14 — Candidate lists, FAISS module, cluster scripts

- **Candidate lists** (`build_candidates.py`): exact search = full scoring + the harness's own §4.6 filter
  (`apply_filters`), then top-k. Built once per split: ranker-training with k = 200 (target force-appended as column 201 when
  missed), Val/Test with k = 500 (never forced). Any K₁ <= 500 is a prefix; ranker-val lists (Val, 200, forced) are
  derived at load time (`load_candidates(..., force=True)`). Verified: candidate Recall@200/500 equals the harness
  Recall for the same model exactly. Scores are stored as float16 to keep test lists small.
- **FAISS on macOS:** torch and faiss bundle separate `libomp` copies; using both multi-threaded in one process
  segfaults (reproduced). On macOS faiss runs single-threaded (`faiss.omp_set_num_threads(1)`); this matches the §5
  single-thread latency protocol, and Linux is unaffected.
- **SoC cluster:** Slurm array jobs in `scripts/slurm/`. Two-tower (text) seed 0 and SASRec (retriever-training) seed 0 are re-run there so all seeds of
  a model come from the same device type; the earlier Mac (MPS) runs are kept as development runs only.
- **RQ1 report (2026-10-15, `cascade.eval.report_rq1`):** recomputed from each run's saved ranks. All Val/Test cases
  count; cases a model cannot score (users with no retriever-training interactions, for BPR trained on that period) are misses, since a retriever must
  return candidates for every user (this is the "skipped = miss" row of the baseline table). The frozen-retriever check
  averages each case's Recall@200 hit over seeds, then runs the paired bootstrap over users ((text + ID) − (text)), overall and on
  new-in-period cases. A difference counts only if its 95% CI excludes 0; a tie keeps Two-tower (text) (the main retriever).
  Verified: on existing runs the recomputed numbers equal `metrics.json` / `extra_segments.json` exactly.
- **Cluster partitions (2026-10-15):** training jobs run on `gpu-long` with `--gres=gpu:a100-40:1` (one fixed GPU type
  for every seed); data prep and candidate building use `gpu` (3 h limit, any GPU type).

## 2026-10-20 — Readable names

Teammates found "period A/B", "v2/v4" and "toy" hard to follow, so everything user-facing was renamed (code, configs,
CLI flags, result folders, reports, docs). Definitions and numbers are unchanged: the regenerated RQ1 and baseline
tables match the earlier ones value for value.

| Old | New (code / display) |
|---|---|
| period A, `A` | `retriever_train`, "retriever-training period" |
| period B, `B` | `ranker_train`, "ranker-training period" |
| A ∪ B, `AB` | `all_train`, "all training data" |
| `T_AB`, `period_b_fraction` | `t_ranker_train`, `ranker_train_fraction` |
| `v2_transformer` (v2) | `two_tower_text`, "Two-tower (text)" |
| `v4_id_text` (v4) | `two_tower_text_id`, "Two-tower (text + ID)" |
| `v1_meanpool`, `v3_no_logq`, `v5_id_only` | `two_tower_text_meanpool`, `two_tower_text_no_logq`, `two_tower_id_only` |
| `toy`, `--toy`, `data/processed/toy` | `sample`, `--sample`, `data/processed/sample` (~2% user sample of the same data) |
| `results/baselines/SASRec_A/` | `results/baselines/sasrec/retriever_train/` (same pattern for every baseline) |
| `configs/baselines/pop.yaml` | `configs/baselines/popularity.yaml` |
| candidate file `B.npz` | `ranker_train.npz` |

- Run files saved before the rename keep their old `name` inside `config.yaml` / `metrics.json` (they record what ran).
  Candidate folders are now named after the run folder, not the stored config name.
- `scripts/migrate_names.py` updates each machine's untracked files once: the data folder, keys inside the processed
  data's JSON files, candidate lists, and model weights left in old result folders after `git pull`.
- RecBole's cached exports in `data/processed/recbole/` keep old names; they are re-created under new names if a
  baseline is retrained.

## 2026-10-21 — Frozen retriever: Two-tower (text), seed 0

Decided by Brandon on 21 Oct (to confirm with Tong and Phua). Run: `results/retriever/two_tower_text/seed0`
(`config.yaml` + `model.pth`, both in git).

RQ1 numbers behind the choice (`results/rq1/summary.md`; per-user macro, mean ± std over seeds 0–2):

| | Two-tower (text) | Two-tower (text + ID) |
|---|---|---|
| Val Recall@200, all 232,904 cases | 0.0686 ± 0.0009 | 0.0724 ± 0.0008 |
| Val Recall@200, 76,768 new-in-period cases | 0.0263 | 0.0190 |
| Test Recall@200 | 0.0438 ± 0.0008 | 0.0438 ± 0.0007 |
| Test Recall@200, new products (cold) | 0.0193 | 0.0119 |

Frozen-retriever check (paired bootstrap over users, (text + ID) − (text), 95% CI): overall +0.0037 [+0.0027, +0.0048],
new-in-period −0.0074 [−0.0086, −0.0061]. The two disagree, which §7.4 says the team must settle. Reasons for text:
- It loses 5% overall but gains 38% on new products, and 44% of Test targets are new products. (Test was not used for
  the choice; it only confirms that the overall gain does not carry over: both have Test Recall@200 0.0438.)
- Text + ID overfits early (seed 0: Val peaks at epoch 10, then falls while training loss keeps dropping): the per-product
  vectors memorise the retriever-training period.
- The ranker has its own item-ID embedding, so ID memorisation still enters the pipeline at the ranking stage.
- Simpler and smaller (0.8 MB vs 28 MB of weights).

**Seed 0** by convention, not as the best seed (seeds 0 / 1 / 2: 0.0687 / 0.0697 / 0.0675 Val Recall@200; picking
the best would select on noise). Two-tower (text + ID) stays an RQ1 ablation only.

**No hyperparameter tuning.** The settings are paper recipes (SASRec: 2 blocks, 1 head, L = 50, Adam lr 1e-3,
dropout 0.2; Yi et al.: in-batch softmax + logQ) and our own choices (τ = 0.05, d = 64, MLP 384 → 256 → 64,
batch 256 sequences, 4,096 pairs per softmax); none was varied on our data. A one-night Val check (τ, batch size, lr)
was considered and skipped: the margin over the baselines is large (Val Recall@200 0.069 vs SASRec 0.040), a new
retriever would mean rebuilding every candidate list and retraining every ranker, and the baselines are untuned too.
Limitation for the report: SASRec used dropout 0.5 on its sparse Amazon datasets; we (and our SASRec baseline) use 0.2.

### Candidate lists for the rankers

- Built **once, on the cluster**, from this run:
  `sbatch --export=ALL,RUN_DIR=results/retriever/two_tower_text/seed0 scripts/slurm/build_candidates.sbatch`
  → `data/processed/candidates/two_tower_text_seed0/`. Everyone uses that one copy. `meta.json` records `source_run`.
- Not in git: about 2 GB (ranker_train ≈ 0.4 GB, val ≈ 0.7 GB, test ≈ 0.9 GB, scaled from the sample lists), and
  GitHub rejects files over 100 MB. Ranker training needs only `ranker_train.npz`, `val.npz`, `val_target_score.npz`,
  `meta.json` (≈ 1.1 GB). Share via the cluster (or a drive copy), not by rebuilding: each machine's `prepare_data`
  re-encodes `item_text_emb.npy`, which can differ in the last decimals across GPUs and swap near-tied candidates.
- **One ranker for all retrieval settings.** Rankers are trained once on these top-200 lists and frozen; RQ2 varies
  only K₁ and the search index of this same retriever, so differences come from retrieval alone. Comparing retriever
  *models* end to end would need one ranker per retriever (scores and ranks are retriever-specific features); not
  planned (optional extra if the team wants it after the main ranker works).

### Open, for the team to decide before ranker training (§15 rule 1: positive/negative definitions)

1. **Forced-positive leakage is an unresolved training-design risk.** The sample lists force-append the target
   in 94.2% of ranker-training instances. No ranker exists yet (status confirmed 2026-10-24), so this is not an
   observed ranker failure. See the detailed 2026-10-24 entry below for the mechanism, unsafe shortcuts, options
   and checks to settle before ranker training.
2. **Later reviews are labelled 0.** Each case has one positive (the next review). The user's later reviews in the same
   period can sit in the list as negatives. Sample lists: ranker-training 3.7% of lists contain one in the top 200
   (vs 5.8% containing the target itself), Val 2.9% (vs 4.2%). Proposal: exclude the user's later same-period items
   from negative sampling (load-time change, no rebuild). Evaluation stays next-item (benchmark protocol); an optional
   secondary "any later review in the top 10" metric would show how much next-item scoring understates all models.


## 2026-10-24 — Forced-positive leakage before ranker training (open)

**Current status and scope.** Retrieval is done; there is no ranker yet. This entry documents a risk in the
planned ranker training and validation protocol, not a failure in an existing trained ranker. It does not
invalidate the saved retrieval results. No training policy, feature set, labels or metric definition is changed
by this note. The ranker owner (Tong) and team must choose a policy before training; evaluation checks it.

### How the shortcut arises

- `build_candidates.py::force_insert` keeps the natural top-200 candidates and appends a missed target as column
  201, using its actual retrieval score. The existing sample audit found `forced_fraction = 0.942`; this is a
  sample finding, not a measured full-data fraction.
- When inserted, the target has a retrieval score no greater than the top-200 candidates (ties are possible).
  If the feature builder uses the appended position as rank, it gets rank 201; if it uses the full-catalogue
  retrieval rank, that also identifies a target outside the natural top 200. Both may expose insertion.
- All natural candidates in such a list are labelled 0; the inserted candidate is the only label 1. For example,
  a positive with score 0.2 and rank >200 competes with negatives scoring 0.5–0.9 at ranks 1–200. A model can
  exploit the selection procedure instead of learning which naturally retrieved product is the next target.
- Column 201 by itself is not automatically an input to a pointwise MLP. Leakage occurs if position, retrieval
  rank/score, an insertion flag or another label-dependent marker reaches its input. This is training-label
  leakage through candidate construction, not future-interaction leakage.

### Why forced validation can hide it

`load_candidates(..., force=True)` also inserts missed validation targets. A model using the same shortcut
could score well on forced-list GAUC or loss, including the planned early-stopping criterion. In actual
end-to-end Val/Test lists no target is inserted: retrieval misses stay misses and every scored candidate is
naturally retrieved. Good forced-list GAUC therefore does not establish good ranking on that population.
Naturally retrieved positives still provide legitimate training examples; the concern is the shortcut on the
forced majority, not a claim that training must necessarily fail.

### Options for the team to assess (none adopted here)

1. **Train on naturally retrieved positives.** Keep only training lists whose target is naturally in the top 200;
   retain retrieval rank/score. This matches the ranker's inference task, but the sample audit suggests only
   about 5.8% of lists remain. Check full-data support and item/user coverage before choosing it.
2. **Use forced lists with a feature policy applied equally to all candidates.** For example, excluding retrieval
   score/rank and all insertion markers from every row removes those direct shortcuts. This changes the planned
   feature set and needs team agreement; it also leaves a mismatch between forced and natural positives.
   It is an alternative policy, not a request to add another model ablation.
3. **Another documented sampling/feature policy.** Explain why it does not reveal which row was inserted and
   validate it on natural candidate lists. Giving only forced rows random ranks/scores is not automatically a
   solution: the synthetic joint feature pattern can still reveal insertion.

**Insufficient fixes.** Shuffling rows does not remove score/rank clues. Masking or zeroing retrieval features
only for forced targets, or exposing `forced`/`target_pos`, creates a positive-only marker. Keep these fields
for labels/audits, not as model inputs.

### Checks and responsibilities before interpreting ranker results

- **Ranker owner/team:** record the chosen policy and its trade-offs here, including the early-stopping population.
  Inspect the actual feature tensors, not just the candidate-file column order. Fit any standardisation using
  training data only.
- **Evaluation:** report the full-data training forced fraction and retained support. On Val, separately label
  forced-list diagnostics and natural-list results; do not pool them under one GAUC number. For natural-list
  GAUC/conditional nDCG, state that only lists containing the target are included and report their coverage.
- **Evaluation:** measure end-to-end Recall@10/nDCG@10 on all eligible, unforced Val instances, with retrieval
  misses counted as zero, and compare with retrieval-score order on the same instances. Use Val to verify the
  chosen protocol; Test remains for final reporting under the frozen protocol.
- No retriever retraining is required merely to document or address this ranker-training risk. No existing
  ranker needs retraining because none has been trained yet.


## 2026-10-24 — Evaluation-plan review and current scope

- Updated `docs/evaluations.md` after review. Retrieval is complete; ranker/cascade results remain future work.
  The forced-positive policy remains an open dependency owned by ranking/the team, not an implemented fix.
- Clarified natural versus forced validation, GAUC populations, fixed retriever versus ranker seed variation,
  paired user-bootstrap interpretation, and prior exposure to retrieval Test results. Existing retrieval results
  and canonical metric/segment definitions are unchanged.
- Evaluation will report new/not-new separately from canonical pre-T2 cold/tail/head labels, use matching macro
  weights for the funnel, include nDCG contributions and within-top-10 movements, and check profile support and
  proxy accuracy. Remaining analysis thresholds/parser rules must be saved on Val before further Test analysis.
- A verified MLP loss against retrieval-score order is a valid finding, not a declaration that the study is
  meaningless. Investigate protocol/implementation first, then agree whether expensive sweeps remain useful.
- Replaced the CI-overlap selection wording in `CLAUDE.md`: the team must record an acceptable absolute Val
  nDCG@10 loss margin δ and cost/tie-break rule before inspecting sweep outcomes. No numeric margin is chosen
  here. MLP sweeps shortlist settings; the actual selected ranker must be checked there on Val before Test.
- These are documentation/protocol clarifications. No model training, feature implementation, label policy,
  data split or metric implementation was changed; no experiments were rerun.

## 2026-10-24 — Controlled ranker feature ladder

The user approved the cumulative feature ladder after reviewing the ranker dependency graph. This is validation development for RQ3; the retriever, candidate lists (K1=200), BCE loss, 50 uniform negatives, hidden widths, dropout and learning rate stay fixed across rungs.

- `retrieval`: standardized retrieval score only.
- `matching`: also include category/store matching and text cosine, candidate frozen text projection, mean-pooled learned history item/category embeddings, history length and recency. History embedding tables exist here, but candidate ID/category embeddings do not enter the scorer until the attributes rung.
- `attributes`: also include candidate item ID, leaf/top category, store and price embeddings.
- `statistics`: also include period-causal review-count, rating and age embeddings. This reconstructs the original full feature set.

This is a cumulative comparison, so an increment measures the added group's value in the presence of earlier groups. Group conclusions are order-dependent. Hidden widths stay fixed but input size/parameter count change; parameter counts are recorded. The final full rung is retrained alongside the reduced rungs for a controlled comparison.

Use seed 0 for full-data discovery; three-seed replication belongs to the later shortlist stage. Seed model initialization before constructing its parameters (the earlier exploratory run seeded the training loop only). Keep the existing checkpoint rule: conditional per-user macro nDCG@10, with GAUC beside it. Report end-to-end nDCG/Recall@10 as well; conditional macro and end-to-end macro have different user denominators and should not be interchanged.

Save target ranks for every validation instance and the five exhaustive outcomes: retrieval miss, unrescued, demoted, retained, rescued. These counts use all validation instances as denominator. They describe crossing the top-ten boundary; changes within top ten still affect nDCG.

The new ladder first runs with `--sample`: a fixed 2% subset of full-data user IDs, sampled with RNG seed 0, one epoch, retaining the full catalogue, original IDs and time boundaries. This differs from the repository's separately preprocessed raw-user sample because those sample artefacts are absent locally; it checks the actual candidate-list interface without remapping IDs. It is marked development-only and is never reported as research evidence. Existing sample-data-dependent tests skip when those artefacts are absent; the feature-exclusion and rescue/demotion tests use hand-computed inputs and run independently.

Each rung saves its config, parameter count, checkpoint, epoch history, validation comparison and per-instance ranks. Existing earlier runs and checkpoints are preserved.

## 2026-10-24 — Shortlist repetition and meso analysis

The user explicitly requested repeats and analysis of which groups benefit. Repeat both shortlisted configurations (matching and attributes) with ranker seeds 1 and 2, retaining the original seed-0 runs. The frozen retriever stays seed 0, so this measures ranker-training variability conditional on a single fixed candidate builder. Training code, features, loss, candidate lists, checkpoint rule and hyperparameters remain identical to discovery. The runner regenerates the group report after each completed seed; incomplete reports are labelled provisional.

Before examining group outputs, fix the analysis as follows:

- Item groups: new-in-period at validation entry (`first_ts >= T1`), then warm tail/head using the existing project segments. Existing head/tail labels are based on counts before T2; this is a descriptive validation grouping, not a new time-causal training feature or a model input. Do not relabel validation items as test-cold.
- History length: 1–2, 3–5, 6–10, 11+ strictly prior interactions. For long histories, count distinct known leaf categories in the last 50 prior items; 1–2 is homogeneous, 3+ is mixed. With no known category, report an explicit unknown group.
- Window-entry novelty: report new-in-period proportions for ranker-training and validation windows. Also report validation in fixed elapsed-day bins 0–89, 90–179, 180–269, 270+, with novelty always defined at T1. Do not inspect Test to choose definitions.
- For each group, compare the same instances against retrieval order; report case/user support, retrieval Recall@200, end-to-end macro/micro and conditional nDCG@10, Recall@10, and mean rescue/demotion counts across seeds. Average per-instance metric values across seeds rather than ranks/scores; this is not an ensemble. Per-user averaging is performed within each group.
- Show training-seed SD separately from paired user-bootstrap CIs (1,000 resamples). The CI on seed-mean per-instance metric differences describes user sampling uncertainty only. It does not capture seed uncertainty. Also compare attributes versus matching on common completed seeds.
- Group intervals are exploratory, unadjusted across comparisons, and cannot establish causal feature effects or group differences. With fewer than three seeds, label all findings provisional. Test remains untouched.

Bootstrap implementation now processes user resamples in bounded-memory chunks, preserving the exact RNG draws, estimator and percentile interval of the original dense implementation. A direct equality test checks this.

## 2026-10-24 — DIN comparison after MLP replication

Proceed to the planned RQ3 DIN comparison after the matching-only MLP beats retrieval order in all three ranker seeds. Retain the selected `retrieval + matching` feature groups, frozen seed-0 retriever/candidates, K1=200, BCE, 50 uniform negatives per training list, Adam 0.001, batch setting 2048 (40 lists / up to 2040 rows), dropout 0.1, head 256/128/64, patience 3 and maximum 30 epochs. Continue checkpoint selection by conditional per-user macro nDCG@10, as the user approved, with GAUC alongside. Train fresh DIN seeds 0/1/2; compare their metrics with the existing corresponding MLP seeds. Test remains untouched.

DIN concatenates learned item32 and leaf-category16 embeddings for history and query. Its local activation is `[query, history, query-history, query*history]` through 64/32 ReLU hidden layers to one signed coefficient, padding masked to zero, then an unnormalised weighted sum of history vectors. This is the same input algebra as [FuxiCTR's DIN_Attention](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/pytorch/layers/attentions/target_attention.py) with `use_softmax=False`, checked by reading the reference source. The [DIN paper](https://arxiv.org/abs/1706.06978) motivates candidate-specific interest and unnormalised activation. This is a project adaptation retaining our head/ReLU/BCE; it does not reproduce the full paper's Dice/mini-batch-aware regularisation recipe. We have not executed the FuxiCTR package or reproduced the paper's benchmarks.

Interpretation detail: the selected matching-only MLP uses item/category embeddings for history pooling but does not expose candidate ID/category embeddings directly to its head. DIN uses these candidate embeddings inside the attention query. Thus this comparison includes the candidate identity interactions introduced by attention; it cannot attribute gains to attention independently of that extra use of candidate information. It also changes mean pooling to an unnormalised weighted sum, including the associated scale change. No direct candidate ID/category input, item statistics, or other metadata is added to the shared head. A stricter control with direct candidate attributes would be a separate future comparison.

Refactor MLP history pooling into an overridable method without changing parameters, concatenation order, mean-pooling formula or checkpoint keys. Initialise the shared embedding/projection/head modules before DIN's attention, preserving the same shared initial weights for a given seed. Use the same grouped training/evaluation pipeline for both models.

Before full training, run DIN on the existing fixed 2% full-data user development subset for one epoch, preserving catalogue IDs and time boundaries. `20261024-din-matching_sample` completed 6,224 training lists and 4,775 validation lists (349 natural retrieval hits), with finite training/validation outputs and saved checkpoint/ranks. Its scores are development-only and must not appear in research result tables. No additional unit-test suite was run in this step.

Analysis definitions remain fixed: new-in-period, warm-tail, warm-head; prior-history lengths; long homogeneous/mixed histories; validation elapsed-day windows and window-entry novelty. Main mechanistic hypothesis, specified before full DIN results: candidate attention may help long mixed histories by reducing dilution from unrelated prior purchases. This is a hypothesis rather than an expected result. Report all prespecified groups/supports, including small groups and negative results, and paired DIN-minus-matching differences on common completed seeds. Report seed SD separately from exploratory user-bootstrap CIs. Fewer than three DIN seeds means provisional evidence.

`cascade.pipelines.ranker_din_study` saves source/config snapshots, hashes, per-seed logs and status; completed model training remains distinguished from subsequent analysis failures. `cascade.eval.ranker_meso --din-id <id>` saves a separate report alongside DIN runs and does not overwrite the completed MLP report.

## 2026-10-25 — Complete R0–R3 validation comparison and ranker CPU timing

All three DIN seeds completed. The user authorized the next stage after the completed three-seed comparison: add popularity order on the same candidate lists and measure CPU ranking latency. No models are retrained. Use `configs/ranker/comparison.yaml` to select the saved matching-only MLP and DIN checkpoints for seeds 0/1/2. Preserve their saved validation ranks, selected-checkpoint GAUC and exact checkpoint configuration, with case/user/retrieval-rank alignment assertions and source/checkpoint hashes.

R0 uses raw review-event counts strictly before T1, over the same top-200 retrieval candidates; no future metadata or bucketed popularity is used. Evaluate target ties pessimistically through the existing ranker metric helper. R1 keeps its original cached retrieval order; compute its GAUC from the float16 cached scores with half credit for ties. This score quantisation may introduce ties in GAUC without changing the recorded retrieval ordering. Report this distinction and the frequency of target-score ties. Both deterministic baselines have one accuracy result rather than artificial training seeds.

Add conditional and end-to-end nDCG@10, end-to-end Recall@10, conditional GAUC, rescue/demotion counts, the existing item/history/time groups and paired user-bootstrap comparisons R2–R0, R2–R1, R3–R2. Group tables retain macro/micro nDCG, Recall and support. Global learned GAUC is reused from the selected checkpoint; per-instance learned AUC/tie data were not archived, so do not fabricate group GAUC from ranks. CIs remain exploratory and conditional on the seed-mean metric, with sample training-seed SD separate.

CPU timing is validation development, keeping Test untouched until final configuration selection. This is deliberately a validation counterpart to §5's final Test timing protocol, not the final Test measurement. Use the same 1,000 uniformly sampled validation instances for every system/seed, one query at a time with all 200 natural candidates, 50 identical warm-up queries, PyTorch intra/inter-op and BLAS thread limits of one. Record CPU/runtime and raw per-query timings/instance IDs. Exclude candidate retrieval, disk I/O, checkpoint loading and one-time lookup-table creation. Measure preparation (current common feature builder plus CPU tensor conversion), scoring, ordering and directly timed total separately; component medians need not sum to total median. Baselines use their natural work: raw-count lookup/sort for R0, first-ten extraction from the already ordered list for R1. Sorting ties during timing uses stable order solely to perform the ordering work; it never changes the pessimistic accuracy policy. Learned accuracy retains the saved training-device outputs while latency executes the same checkpoints on CPU.

Begin with the same fixed 2% full-data user development subset, 20 measured queries, five warm-up queries and 50 bootstrap resamples. `20261025-rq3_sample` completed all four systems, six checkpoint loads and group reports. Its numbers are development-only. Full measurements restore 1,000 queries, 50 warm-ups and 1,000 bootstrap resamples. No extra unit-test suite is run in this step. Local timings describe this Python implementation, including some unused fields prepared by the common feature builder, rather than a production serving system.

Completed full run `results/ranker_comparison/20261025-rq3/`: R0/R1/R2/R3 end-to-end macro nDCG@10 = 0.003281 / 0.005642 / 0.006843 / 0.006436. MLP remains the overall validation choice. Single-thread Apple M1 Max ranker-stage median times are 0.010 / 0.003 / 1.591 / 7.324 ms, respectively, with learned entries averaged over seed-specific medians. DIN has higher global conditional GAUC (0.743015 versus 0.711995), lower overall nDCG and greater ranking cost; its new-item/tail improvements and head loss remain useful exploratory findings. Preserve all three seeds; do not choose an unusually strong DIN seed as the representative accuracy result. No Test evaluation or new model training was performed.

## 2026-10-25 — Frozen-MLP candidate-count sweep

The user explicitly approved the proposed K1 sweep. Evaluate K1=50/100/200/500 on validation with the same frozen matching-only MLP checkpoints for seeds 0/1/2 and the frozen seed-0 retriever. Use natural prefixes of the already generated exact top-500 candidates; retain the original filtering and never insert missing targets. Preserve training-time feature tables/normalisation. No new training or Test evaluation is authorised by this step. At K500 the candidate distribution extends beyond the K200 pool used in training; report that limitation.

The pointwise MLP's inputs for a user–candidate pair do not depend on which other candidates are present. Score each pair once at K500 for cases whose target is retrieved there; reuse those fixed scores to derive ranks and tie-aware AUCs for all four prefixes. Cases whose target is absent at K500 are misses at every smaller K and receive zero end-to-end credit. Latency independently scores all candidate types on uniform validation queries, including retrieval misses. Before interpreting results, require the K200 rescored macro nDCG to match each checkpoint's saved result within absolute 0.000001; retain the count of rank-level differences as a numerical diagnostic. Enforce that adding candidates cannot improve the rank of an already retrieved target for this fixed pointwise model.

Report retrieval Recall@K1, end-to-end macro/micro nDCG@10, macro Recall@10, conditional nDCG and GAUC, failure counts, and the existing item/history/diversity/time groups. Conditional populations change with K, so changes in conditional metrics alone are not same-case ranking changes. Retain three-seed SD separately from paired user-bootstrap intervals on seed-mean per-instance values. Compute overall and group paired differences versus the observed best candidate count.

Implement the plan's recommendation literally: find the K with highest observed mean end-to-end macro nDCG@10, compute that mean's user-bootstrap 95% interval, and choose the smallest tested K whose observed mean lies inside it. This is a descriptive heuristic rather than proof of equivalence or noninferiority. Report the eligible counts, the interval, and paired gaps to the best alongside it. Intervals are exploratory and do not adjust for selecting the maximum or multiple groups.

Also decompose each adjacent expansion (50→100, 100→200, 200→500) into (a) gain from targets newly entering retrieval and (b) loss from added competitors on previously retrieved targets. Preserve identical overall per-user weights in both channels by averaging masked per-instance differences over the full population; these contributions add exactly to the overall macro change. Do not substitute separately reweighted subgroup macro means. For fixed pointwise scores the new-target contribution is nonnegative and the existing-target competition contribution nonpositive. Save raw new-target top-ten hit and existing-target demotion counts as an intuitive explanation.

CPU ranking timing uses the same 1,000 validation queries after 50 warm-ups for every K/seed, one CPU/BLAS thread, with feature preparation/scoring/ordering and directly timed total. Randomise K order deterministically. These are ranker-stage times only; cached prefixes do not measure retrieval or total cascade latency, which remains separate work. Begin on the fixed 2% user development subset with 20 timed queries, five warm-ups and 50 bootstrap resamples, then run full validation with 1,000 resamples. No additional unit-test suite is run in this step. Save config/source/checkpoint hashes, per-seed per-K ranks/AUCs, timing query IDs, group supports and reports.

Completed full sweep `20261025-k1`: mean macro nDCG at K50/100/200/500 is 0.008315 / 0.008363 / 0.006843 / 0.002210; CPU ranker medians are 0.662 / 0.941 / 1.594 / 3.730 ms. All K200 ranks reproduce exactly across all three seeds. K100 is the point-estimate winner in each seed. K50 and K100 both fall inside the best mean's user-bootstrap CI, so the prespecified heuristic recommends K50. K100−K50 paired interval [-0.000145, 0.000241] includes zero; do not claim equivalence. Preserve the item-group trade-off: K100 versus K50 improves new-in-period and warm-tail but reduces warm-head, and K200 has the highest new-item/tail point estimates. No actual deployment, final Test or retrieval-latency claim follows from this recommendation.

Candidate expansion has two empirically distinct effects. At 200→500, 11,903 extra target events become retrievable, but new top-ten hits average 929.3 while previous top-ten targets demoted average 3,089.7. The additive macro contributions are +0.001757 from newly retrieved targets and -0.006389 from competition, net -0.004633. This supports a candidate-quality versus ranking-competition explanation for this fixed model. It does not establish that any particular feature, training loss or negative-sampling policy causes the loss. More detailed cause analysis would be separate work.

Source tie note: actual K100 candidate membership includes two targets at returned rank 100 whose pessimistic full-catalogue ranks are 101. Use actual membership to describe the ranker's retrieval ceiling and record the two-event difference versus the source metadata's Recall@100 summary. No such discrepancy appears at K50/K200/K500.

## 2026-10-25 — Explain frozen-ranker demotions and retrieval-score use

The user approved the proposed diagnostic: inspect concrete 200→500 target demotions, profile the added products that overtake them, and investigate whether force-added training positives are associated with a reversed retrieval-score signal. Keep all three matching-only MLP checkpoints frozen. Do not retrain, change labels/sampling, alter the candidate-count recommendation, or evaluate Test in this step.

Fix the cohorts before examining new outputs: include all eligible validation events demoted from rank ≤10 at K200 to rank >10 at K500 in any saved ranker seed, plus 1,000 uniformly selected validation events (RNG seed 0). Rescore the union at K500 for every model, and require exact reproduction of all cohort K200 and K500 target ranks. Use the uniform cohort for general within-query score correlations, retrieval-rank-band mean logits, and added-item shares of the top ten. Use the complete per-seed demotion events for overtaker profiles. Label denominators explicitly: seed–event–candidate occurrences can repeat products and events; they are not independent unique items, equally weighted users, or global relevance measures.

Profile added positions 201–500 by new-at-T1 / warm-tail / warm-head and leaf category. Compare their overtaker/top-ten shares with their availability in the SAME demotion events. Count both strict score>target and equal-score overtakers separately; the latter follows the existing pessimistic target-rank policy. Descriptive slot lists use stable sorting, without changing target metrics. Preserve item IDs, category/store/text-match inputs, scores and case IDs in local artifacts. Unobserved alternatives must not be described as confirmed disliked or irrelevant products.

Recompute raw retrieval-score target-versus-alternative AUC on all training lists, naturally retrieved training positives, force-added training positives, and naturally retrieved validation targets at K200. Report per-event and per-user averages. Use all valid alternatives for exact AUC and one uniform unobserved alternative per training list (fixed RNG) to describe the expected uniform-negative score distribution. Use existing training normalisation unchanged. An inverse raw-score training signal alone does not establish the learned model's response or a causal training-policy effect.

Add a feature-sensitivity probe: in each uniform validation event, take the candidate at retrieval rank 100 and hold the product/history/other features fixed. Replace only standardised retrieval-score input 0 with z ∈ {-3,-2,-1,0,1,2,3}, saving every logit and its query ID for all seeds. This is an artificial input intervention, not a data edit, new training set, genuine accuracy estimate or calibrated probability. Some combinations may be off the natural data manifold. It isolates the frozen network's response to that input over the specified grid; causal claims about why force-insertion produced the response require a separate controlled retraining study.

Select one seed-0 product example for each target group (warm-head, new-in-period, warm-tail): among demotions with a target title, choose the event nearest that group's median rank loss, tie-breaking by event ID. Show target rank200/rank500, unchanged target score, last three prior products and up to five highest-scoring strictly overtaking added products with metadata and retrieval/MLP scores. This is a prespecified illustration rule, not selection of extreme cases.

Start with the fixed 2% full-user development subset and 20 uniform queries; training-normalisation tables still use full training candidates. Then run all eligible validation demotions and 1,000 uniform events. Save source/config/checkpoint hashes, cohort logits/numeric features, probe logits, per-query correlations, per-seed profiles and reports/figures. No additional unit-test suite is run in this diagnostic step.

Completed full diagnostic `20261025-score-diagnostic`: 5,814 rescored union events, including all 4,845 unique demoted events across seeds and 1,000 uniform events with overlaps. Six K200/K500 rank reproduction checks have zero differences. Training raw-score AUC is 0.080370 overall versus 0.628975 on naturally retrieved validation targets; force-insertion fraction is 87.69%. Uniform-query raw-score/logit Spearman means are -0.402/-0.372/-0.400; added positions 201–500 occupy 94.7%/95.6%/95.2% of top-ten slots. Fixed-product/history probes show a strong response favouring low retrieval-score input in the studied low-score range, with a modest upturn again at high z; do not claim global monotonicity. At z=-1 versus z=0 the mean logit increase is 2.428/2.091/2.558 and the lower input is favoured in 81.6%/84.5%/82.1% of queries.

These observations triangulate a specific frozen-model behaviour consistent with the forced-positive training-signal hypothesis. They do not identify the causal effect of changing that training policy. Any retraining comparison, label/population change, removal of a feature or new loss remains a separate decision. Preserve current checkpoints and candidate-count results. New-at-entry and warm-tail added products are overrepresented relative to availability among overtakers, with head items still the largest raw share. Interpret these as conditional occurrence profiles, not global accuracy or known relevance of alternatives. Three prespecified median-rank-loss examples and both inspected figures are retained in the report.

## 2026-10-26 — Draft ranker PR and reconciliation with the updated evaluation plan

The user authorised closing obsolete PR #1 and creating a separate draft ranker PR. Preserve the existing retrieval results and the newer evaluation-plan changes from `main`. The earlier entries above are a chronological record: statements that no ranker existed and next-step plans describe the state at their time. Exploratory MLP/DIN validation runs now exist under the original force-insertion/negative policy; no new training remedy is ratified by packaging them for review.

- The user approved checkpoint selection by conditional per-user macro Val nDCG@10 on natural retrieval hits, with GAUC beside it. The grouped-candidate training path implements this; the legacy aligned-NPZ toy path retains its older GAUC selector and is not the research training entry point. End-to-end results retain all eligible natural-list cases, including retrieval misses.
- Reconcile §9 with the implementation: retrieval rank/insertion markers are excluded; the actual retrieval score remains and has an observed training-signal mismatch. DIN adds candidate identity interactions and changes pooling scale, so the MLP–DIN comparison cannot isolate attention alone.
- The saved K₁ sweep used the earlier CI-containment heuristic and returned K50. The updated plan requires a predeclared acceptable-loss margin δ and cost/tie-break rule. **No δ was recorded before these outcomes were seen.** Preserve the old artifacts as historical evidence, label the heuristic explicitly in current documentation/report code, and claim no final candidate count, equivalence or cost-optimality. Do not manufacture a post-hoc margin. A final protocol decision and validation of the actual chosen pipeline remain pending before ranker/cascade Test reporting.
- Share the six selected matching-only MLP/DIN checkpoints and compact numerical evidence needed by current evaluation configs. Keep development/invalid checkpoints, logs, caches and bulk diagnostic arrays local; record an explicit hash/size manifest in `results/ranker_artifacts.json`. Historical configs and source snapshots retain their original bytes/paths for provenance.
- Requirements retain the text-encoder stack used for the successful local runs (sentence-transformers 3.4.1, transformers 4.48.3, huggingface-hub 0.28.1, tokenizers 0.21.0); `threadpoolctl` supports single-thread timing. These environment pins are not an encoder/model ablation. Cross-machine numerical identity is not asserted; use the shared frozen text/candidate artifacts for exact original comparisons.
- No new training, Test evaluation or unit suite is run during PR preparation. Earlier 32-pass/13-skip evidence predates the DIN/comparison/sweep/diagnostic additions; their completed development/full pipeline runs and reproduction assertions are recorded separately. The draft must not describe that older suite as verification of all later additions.

## 2026-10-26 — Final review, current tests and authorised merge

The user explicitly authorised a final review, current test run and merging PR #2. Independent reviews found no blocker in the completed fresh grouped-training/evaluation path or shared artifact dependencies. All 253 artifacts and 75 historical source snapshots match their recorded hashes.

The initial suite passed 32 tests and skipped 13 because the separately remapped sample data was absent. Built those existing sample fixtures from cached raw inputs and reused full-data text embeddings by item mapping; this only creates ignored sample files, without changing full-data experiments. Added 16 deterministic DIN checks and five run-safety checks. The final command `.venv/bin/python -m pytest -q -ra` passes **66 tests with no skips**.

Fixed two future rerun hazards before merging: direct grouped/toy ranker training atomically refuses an existing run directory; CLI `--resume` is rejected before loading config/data because exact optimizer/RNG restoration and selected-artifact carryover are incomplete. Other callers of the shared run-directory helper retain its default behaviour. The internal historical fitting function, models, features, labels, metric definitions, saved checkpoints and research results are unchanged. No research retraining or final Test comparison was performed. The forced-positive policy remedy and final pipeline selection remain open research decisions.

## 2026-10-28 — Exploratory runs (Val only; awaiting team sign-off)

Exploratory session at Brandon's request; running log and results in `docs/report_2026-10-28.md`.
Everything below is **exploratory**: no Test data is loaded, nothing is committed, and no setting is selected.

- **Candidate lists rebuilt locally** (`build_candidates.py`, frozen Two-tower (text) seed 0, `--k-train 200 --k-eval 500
  --splits ranker_train val`), because the team's copy lived only on one laptop. Training lists match the team's
  exactly (315,114; 276,321 forced; 38,793 natural). Val natural@200 is 16,673 vs 16,672, and 58 of the natural Val
  target positions differ by near-tied scores (float differences across machines). Paired comparisons with the
  team's runs use each run's own saved ranks over the same 232,904 Val instances.
- **Natural-only ranker training** (`train_population: natural`, opt-in; default unchanged): drop ranker-training
  lists whose target was force-appended. Filtered after `RankerFeatureBuilder.create`, so vocabularies and score
  standardisation are identical to the forced-list runs; only the training population changes. This changes the
  positive/negative definition (§15 rule 1), so it is a candidate remedy for the 2026-10-24 forced-positive issue, not
  an adopted policy. Configs `configs/ranker/natural_matching_seed{0,1,2}.yaml` (matching-only MLP recipe otherwise).
- **FAISS index sweep (retrieval half)**: `src/cascade/pipelines/index_sweep.py`, `configs/cascade/index_sweep.yaml`.
  FAISS 1.15.1 does **not** raise HNSW efSearch to the number of results requested (verified on synthetic and real
  data). Because filtered search asks for K₁ + buffer = 450 results and §8 requires efSearch ≥ results requested, the
  sweep uses `efSearch_effective = max(efSearch, 450)` and reports both. **Consequence:** the "efSearch = 200" point
  runs at 450, so the two HNSW points are effectively 450 vs 800. A genuine efSearch = 200 point needs a team decision
  (break the §8 rule, or use a smaller buffer for HNSW). Also: with IVF nprobe = 4, the 4 probed cells hold ~436 items
  (< 450), so many queries end up short of K₁ after filtering; exact FAISS search with the 250 buffer leaves ~0.1% of
  queries short of K₁ (early-Val instances where many products are not yet released), which §4.6 says should be ~never.
  Import torch before faiss on macOS (libomp clash segfaults otherwise).
- **Further runs** (all Val, exploratory; details and numbers in the 28 Oct report):
  natural-only seeds 0–2 (`results/ranker_mlp/20261028-natural-matching-seed{0,1,2}`); comparison with the forced-list
  MLP (`results/ranker_compare/20261028-natural-vs-forced-mlp`, new `src/cascade/eval/compare_rankers.py`); K₁ sweep
  and score diagnostic rerun with natural checkpoints (`configs/cascade/k1_sweep_natural.yaml`,
  `configs/ranker/diagnostics_natural.yaml`, `configs/ranker/comparison_natural.yaml`; the saved "recommended_k1"
  is the historical heuristic, not a selection); full index sweep and the natural MLP on its lists
  (`src/cascade/pipelines/index_rank_eval.py`); §5 latency (`src/cascade/pipelines/latency.py`; first run provisional
  because the machine was not idle).
- **`feature_groups: [matching]` (no retrieval-score input)** is now accepted by `ranker_mlp.py` as an opt-in
  exploratory setting (numeric column 0 dropped; default and saved checkpoints unchanged; test added). Used for
  `configs/ranker/{natural,forced}_noscore_seed0.yaml`.
- **Phase 2 runs (exploratory, Val):** no-score MLPs (`20261028-{natural,forced}-noscore-seed0`; forced one stopped
  after 3 epochs, `STOPPED.md`); opt-in `negative_sampling.exclude_later_reviews` (`later_review_cells` in
  `ranker_training.py`; flag off reproduces the old sampler exactly, tested) and `20261028-natural-laterneg-seed0`;
  K_train = 500 lists via the new opt-in `build_candidates.py --out-dir`
  (`data/processed/candidates/two_tower_text_seed0_ktrain500/`; on the 28 Oct merge `--out-dir` was replaced by the
  equivalent `--output-name two_tower_text_seed0_ktrain500 --k-train 500`) and MLP/DIN runs on them; DIN on natural lists
  (`20261028-din-natural-seed{0,1,2}`, `20261028-din-natural-ktrain500-seed{0,1,2}`); K₁ curves via
  `index_rank_eval.py` (`k1_counts`); final latency `20261028-latency-final`. Full-data correction to the 2026-10-21
  entry: later same-period reviews appear in 8.9% of all ranker-training lists and 13.1% of natural ones (3.7% was the
  sample figure). **Summary and team decisions: `docs/report_2026-10-28.md`, "Summary".**
- Final 28 Oct evaluations: K₁ and index sweeps for the K_train-500 DIN (`results/index_rank_eval/20261028-k1-din-ktrain500`,
  `.../20261028-index-rank-din-ktrain500`). Its nDCG@10 rises up to K₁ = 500 (+0.00024 vs 200). No K₁ or index is
  selected (δ not recorded).
- **Opt-in `loss: listwise_softmax`** in `ranker_training.py` (−log softmax of the positive over each list's sampled
  columns; default BCE unchanged; test added). H9 runs: `20261028-natural-ktrain500-listwise-seed{0,1}`,
  `20261028-din-natural-ktrain500-listwise-seed0`. Exploratory.

## 2026-10-28 — Exploratory retrieval/ranking analysis loop (Val only, no training)

Exploratory analyses of saved Val outputs and frozen models; no training, no Test, no K₁/index selection, no change
to the split, filtering or metrics. New code only (`src/cascade/analysis/{staleness,complements,complements_mechanism,
last_item}.py` and their tests); results under `results/analysis/`. Progress and numbers:
`docs/analysis_loop_2026-10-28.md`; measured findings are summarised in `docs/evaluations.md` (exploratory section).
- `20261028-staleness` (saved ranks) and `20261028-staleness-r2` (adds an inference-only re-score that reproduces the
  saved ranks, plus a "frozen pool" counterfactual removing Val-released products as competitors): Val Recall@200
  by month and product era.
- `20261028-staleness-popularity`: warm targets by review count in the retriever-training period, item-cluster
  bootstrap.
- `20261028-complements`: target vs last history product (leaf × store), deep-miss composition, ranking given
  retrieval by class. `20261028-complements-mechanism-r2`: text/retriever cosine to the last product
  (`20261028-complements-mechanism` superseded: a re-weighting table dropped empty cells; marked `SUPERSEDED.txt`).
- `20261028-last-item`: untrained last-product and mean-pool scorers through the shared harness (Val only); a
  5,000-instance re-score with the frozen scorer reproduces the saved ranks first. These are analysis probes, not
  retriever variants and not candidates for selection.
- Added later in the loop: `20261028-logq-serving` (frozen retriever re-scored with u·v − β·τ·log q, β ∈ {0, 0.25,
  0.5, 1}, two conventions for unseen products; β = 0 reproduces the saved ranks; probe, no β selected) and
  `20261028-permutation-importance{,-r2}` (permutation importance for 12 frozen ranker checkpoints on natural Val
  lists; the analysis loader uses a 1e-6 tolerance on the stored score standardisation because the forced
  checkpoints differ from the current builder by 2e-7 relative). Follow-up 3c (`20261028-debiased-lists`,
  `results/index_rank_eval/20261028-debiased-rank-*`, config `configs/analysis/debiased_rank_eval.yaml`) scores
  the natural rankers on debiased candidate lists; probe only, not a retrieval change.

## 2026-10-29 — Cluster scripts for the ranker policy study

Added `scripts/slurm/policy_study.sbatch` (one stage per job; optional `SEEDS` so replication seeds 1 and 2 run as
separate jobs; 10 h limit, since one 8-fit replication job could exceed the 6 h limit of `targeted_study.sbatch`)
and `scripts/slurm/submit_policy_study.sh` (`check` → `development` → `pilot` → `replication` → `report`; `check`
verifies inputs and the frozen candidate hashes before anything is queued). No change to the study protocol,
arms, seeds or pass rule in `docs/evaluations.md`; `targeted_study.sbatch` is unchanged. The study needs the
original frozen candidate lists; the locally rebuilt `two_tower_text_seed0` lists fail the hash check (verified).
- **Opt-in regenerated-candidate reference** (same day). The original frozen candidate files are unavailable to
  Brandon (separate cluster accounts), so `ranker_policy_study.py` gained `--candidate-reference <json>` and
  `--write-candidate-reference <out>`. With a reference, the runner checks the candidate hashes (meta, val and
  ranker_train) and the full-training score mean/std against that record instead of the frozen 2026-10-25 study;
  cardinalities are still checked against the reference checkpoint, and the record is copied into `study.json`
  (`candidate_source.kind = "regenerated"`). Default (no flag) is unchanged and still refuses the rebuilt lists.
  Record for the locally rebuilt lists: `configs/ranker/policy_candidate_reference_rebuilt.json` (score_std differs
  from the original checkpoint by 1.5e-8, which is what tripped the 1e-8 check). Validity: all four arms are trained
  fresh on identical lists, so A–D comparisons are unaffected; the new arm A is not byte-comparable with the
  historical forced MLP. Verified with a local development-stage run
  (`results/ranker_policy/policy-development-localcheck_sample`, not reportable) and its report; tests 171 passed.
  To be confirmed with Qing Rong (owner of the protocol).
- **Cluster scripts removed** (same day): the policy study will run on Brandon's laptop instead, so
  `scripts/slurm/policy_study.sbatch` and `scripts/slurm/submit_policy_study.sh` were deleted. The opt-in
  `--candidate-reference` option and `configs/ranker/policy_candidate_reference_rebuilt.json` stay: the laptop's
  rebuilt lists still need them.
- **Figures and a saved miss analysis** (same day): `src/cascade/analysis/retrieval_misses.py` →
  `results/analysis/20261029-retrieval-misses` (Val target rank buckets and the share of targets without an ID-era
  history; previously computed ad hoc in chat) and `src/cascade/analysis/plot_findings.py` →
  `results/figures/20261029-findings/` (six figures, each with a CSV of the plotted values and `sources.json`). The
  training-positive shares in fig. 4 are recomputed from `ranker_train.npz` with the 28 Oct report's H6 definitions
  (new = first interaction ≥ t_ranker_train) and reproduce 83.9% / 43.1% popular exactly. Summary added to
  `docs/evaluations.md`.
- **Policy study run on the laptop** (same day): pilot `results/ranker_policy/policy-pilot-20261029` and
  replication `results/ranker_policy/policy-replication-20261029` (MPS, `--candidate-reference` as above), report
  `results/targeted_study/policy-report-20261029`. Pre-registered rule applied unchanged: no fix qualifies (C and D
  lose new-in-period nDCG vs A), so A remains the reference. Results in `docs/evaluations.md`.
- **Policy-study figure** (same day): `plot_findings.py` gained `fig_policy_study` and a `--figures` selector
  (default still draws all); `results/figures/20261029-policy-study/fig7_policy_study.png` (+ CSV, `sources.json`),
  drawn only from `results/targeted_study/policy-report-20261029`.

## 2026-11-03 — Complete the independent graph/latent ranking sample study

The user authorised extending root `plan.md` with graph/latent ranking, chose new-item generalisation, one learned graph arm and chronological tuning, then instructed implementation. Execution paused on request after the 21 fits finished; the subsequent “continue work” authorised the scheduled validation/reporting. This decision applies to the isolated sample study, not canonical full-data integration.

- Preserve the frozen MiniLM/90-day-recency K200 pool, natural positives, pre-T0 banks and prefix-fitted preprocessing. R1–R7 share initial parameters and sampling plans within seeds 0/1/2. Tune on the chronological B holdout with epoch 0 eligible; lock every selected epoch/weight before one outer-Val report.
- Three fixed BPR64 fits, PCA-text64, all seven ranker arms and their scheduled reporting completed. Each ranker fit trained two epochs but selected epoch 0. Every selected Val logit/rank/top ten equals frozen retrieval: nDCG@10 **0.012489**, Recall@10 **0.022309**, Recall@200 **0.100990**. Retain all failed training curves and identical outcomes; do not pick a different checkpoint on outer Val to obtain a gain.
- The outcome limits this short BCE recipe. It does not isolate an objective/scale cause or establish that graph/latent features lack useful information. Nonselected trained weights/logits were not retained. A fresh matched R1 objective pilot with inner-only diagnostics is proposed in §10, not executed or silently added to the completed study.
- Source-of-truth findings are in `docs/evaluations.md`; local evidence is `results/local_experiments/20261102-graph-latent-ranking/report/`. Keep bulk weights/arrays/reports local/ignored. Full-data training, changed-pool replication, canonical integration, serving timing and final Test remain separate decisions.

## 2026-11-03 — Continue with a matched objective ablation and conditional feature replication

The user explicitly authorised continued ablation/evaluation and README/Markdown updates. Registered root plan §11 before fitting: fresh R1 BCE versus mean pairwise softplus of negative-minus-positive scores, same candidates/sampling/initial states/selector, seeds 0/1/2, and full retention of inner checkpoints/logits/probe/gradient diagnostics. A predeclared inner-only gate required a higher mean selected nDCG, at least two improving seeds and no mean post-T0-target nDCG loss before expanding to pairwise R2–R7. The gate passed on three seeds and was independently replayed before any new outer predictions; all 24 final epochs/weights were locked before outer evaluation.

All three BCE curves exactly reproduce Phase 2 and select epoch zero. Pairwise R1 reaches Val nDCG@10 **0.015566** versus retrieval/BCE **0.012489**; R2 with graph/history scalars reaches **0.018174**. Both objective and scalar-family differences have positive exploratory paired-user intervals. R2 combines co-review and category/store-history information; no individual relation is isolated. Extra BPR over R2 is inconclusive; neighbour R7 loses to self-only R6 overall/new. Warm R2 remains slightly below retrieval in point estimate. A common score shift alone cannot change ranking, and the objective contrast does not prove a calibration mechanism.

Keep all outcomes and retained trained epochs, the unchanged parent studies and teammates' workflows. Update README, CLAUDE, plan, proposals and the evaluation record, sharing one compact checked diagnostic figure; bulk local outputs stay ignored. Final suite **207 passed / 4 legacy skips**; all real score/probe/gate/weight replays and completion hashes pass. No full-data fit, candidate-pool change, canonical integration, serving study or Test evaluation. A later category/store-versus-co-review decomposition should register its own matched arms before execution rather than extending this completed study after seeing Val.

## 2026-11-03 — Finish the graph study and stop experiment execution

The user approved the fixed four-arm metadata/co-review decomposition and frozen encoder diagnosis, explicitly requiring a pause after completion. Registered root plan §12 before fitting; all twelve full fits and two technical smokes completed. G0/GMC reproduce the old controls, including final logits; all weights were locked before outer evaluation. GC improves overall/new/warm versus G0, GC beats GM overall/new, and extra metadata over GC plus the factorial interaction remain inconclusive. Existing matching/support features remain present; no individual co-review feature or causal mechanism is isolated.

The frozen R6/R7 diagnosis used twelve saved epoch2/selected checkpoints over all prefix queries and reused tuning/Val scores. Stronger inner neighbour scores coexist with poorer final-refit Val scores; different weights/populations prevent identifying the cause. Preserve all checkpoint, margin, residual, geometry and gradient evidence for the deeper assessment. Final suite **226 passed / 4 legacy skips** and completion/independent audits pass. Update the project records and checked comparison figure, commit, then **stop experiment execution**. No new GNN, automatic follow-up fit, temporal replication, full-data training or Test evaluation is authorised by this completion handoff.

## 2026-11-07 — Loop phase H (Val only, exploratory)

Brandon extended the loop to 08:00 on 7 Nov. Claude, as lead researcher, chose two new runs. All earlier rules hold.

- **H1:** the fused P3 pool trained with rolling monthly co-review features (bce_coreview recipe, seeds 0/1/2), plus a rolling bce_text control if time allows.
- **H2:** fresh Two-tower (text) retrained on all data before T1, seeds 1 and 2, with retriever-level Val metrics only.

## 2026-11-06 — Overnight research loop: retrieval pools, robustness, random mixing, context, freshness (Val only, exploratory)

Brandon authorised, on 6 Nov 2026 (01:50), an overnight loop led by Claude as researcher. Allowed work:

1. **New ranker training on changed candidate pools.**
   - Frozen two-tower seed-0 natural top 200 (P2).
   - A fixed round-robin fusion of recency and two-tower (P3). The rule is fixed, not tuned: skip score = reciprocal-rank fusion with k = 60.
   - The ranker is unchanged: the registered `bce_text` residual recipe, seeds 0/1/2.
2. **Extra refit-draw repeats** of `bce_text` and `bce_coreview` on the recency pool. Same initial states, new negative/refit draws.
3. **One isolated random-mixing study** (the professor's and Sahej's item):
   - Missed ranker-training targets replace a candidate at a seeded uniformly random rank, taking that candidate's retrieval score.
   - Arms: all mixed lists, or a subsample matched to the natural positive count.
   - This is a **training-label change in its own study, pending team review**; Val stays natural.
4. **One context-feature arm:** price, period-start review count, rating and item age, user history length and time since last interaction. All candidate statistics come from §4.5 period-start tables, never relative to query time.
5. **No-training analyses:**
   - a retriever oracle;
   - Val graph freshness (pre-T1 and monthly snapshots on frozen checkpoints);
   - a frozen-rule case study of targets missed by both retrievers;
   - meso, latency and diversity tables.
6. **Optional, only if the loop's recorded decision rules favour them:**
   - a DIN-style residual ranker;
   - one Two-tower (text) retrain on all pre-T1 data, used only to serve Val candidates.

- The Val overlap of the two retrievers (recency 6.4%, two-tower 7.2%, union 12.4%, top-100 union 8.7%, micro) was seen at 01:45, before registration. The fusion rule was then fixed without tuning.
- **Not allowed:**
  - no Test file;
  - no selection of K, index, ranker, pool, blend or δ;
  - no edits to existing code, configs or earlier studies;
  - no commits.
- Run IDs use the prefix `20261106-`. Progress: `docs/analysis_loop_2026-11-06.md`. Results: `docs/overnight_results_2026-11-06.md`.

## 2026-11-05 — Local rebuild of the current pipeline and overnight analyses (Val only, exploratory)

Brandon authorised, after the professor's consultation and the PR #3 merge: (1) a separate pinned environment `.venv-encoder/` (uv) used only to regenerate the frozen MiniLM encoder and vectors with the library versions `frozen_text_study.py` requires; (2) retraining the 15 registered ranker fits (root `plan.md` §13: four methods × seeds 0/1/2; §14: `bce_coreview` × seeds 0/1/2) as a **reproduction** with unchanged configs, seeds and recipe, because the teammates' bulk arrays are not on this laptop; (3) no-training analyses: oracle and serving-time blend, HR@3/MRR@10, a Step-5 case study, the U-shape by popularity and secondary metrics for archived pipelines.

- The processed data are ours: `items`, `instances`, `item_segments` and `sequences` hashes recorded in `results/full_data_replication/20261103-promising/study_provenance.json` equal `data/processed/*`. No script builds `data/processed/studies/20261027-targeted/`; a local `results/targeted_study/data.yaml` points at `data/processed` instead.
- Rebuilt folders reuse the teammates' study names under `results/local_experiments/` only because the code hard-codes them; each carries `REGENERATED.md` with every difference from the published numbers. Report/audit/publish phases are not run, so no tracked folder is written.
- No Test file is opened, no new model design is trained, and no K, index, ranker, blend or δ is selected. Progress: `docs/analysis_loop_2026-11-05.md`; results: `docs/overnight_results_2026-11-05.md`.
- **Runs and outcomes (5 Nov, same day).**
  - Rebuild completed 04:35–07:55. Retrieval is exact; bce_text, pairwise_text and pairwise_combined reproduce; both co-review models do not.
  - Approved during the loop:
    - local `report` + `*_audit` phases on the regenerated studies, which were checked to write only inside gitignored `results/local_experiments/`;
    - a post-lock diagnostic scoring the inner-selected checkpoints on Val (`src/cascade/analysis/regen_inner_on_val.py`).
  - New analysis code, all new files, with tests where needed:
    - `ushape_popularity.py`;
    - `secondary_metrics_archived.py` and `secondary_metrics_current.py` (HR@3 = Recall@3 and MRR@10 as extra exploratory metrics from the same ranks; no metric definition changed);
    - `coreview_blend_oracle.py` (blend weights chosen only on inner tuning queries);
    - `case_study_current.py`.
  - The first `20261105-secondary-metrics-archived` run crashed before computing anything and is kept as is; `-r2` is the result.
  - No publish phase, Test file, commit, or edit to existing code. Pytest: 301 passed.

