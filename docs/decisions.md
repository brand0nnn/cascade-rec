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
