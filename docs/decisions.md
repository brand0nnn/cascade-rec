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
