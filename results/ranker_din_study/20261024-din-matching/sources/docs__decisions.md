# Design decisions and deviations

Every non-trivial choice or deviation from CLAUDE.md, with date and reason.

Names follow the 2026-10-20 renaming (see the last entry): **retriever-training period** (formerly period A),
**ranker-training period** (period B), **all training data** (A ∪ B), **Two-tower (text)** (v2), **Two-tower (text + ID)**
(v4), **sample** (toy). Commits and run files from before that date use the old names.

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
