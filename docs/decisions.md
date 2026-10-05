# Design decisions and deviations

Every non-trivial choice or deviation from CLAUDE.md, with date and reason.

## 2026-09-28 — Data

- **Sources.** Interactions: official 5-core rating-only file (URL in `configs/data.yaml`). Metadata:
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Cell_Phones_and_Accessories.jsonl.gz`
  (verified reachable, 858 MB). SHA-256 + download date in `data/raw/MANIFEST.json`.
- **Counts.** 380,999 users, 111,480 items, 2,752,785 interactions (no exact or (user,item) duplicates in the
  5-core file). All 111,480 items have a metadata row.
- **Split cross-check.** Our Val / Test periods contain exactly 239,437 / 304,468 interactions — identical to the
  official `timestamp_w_his` valid/test files, confirming the T1/T2 cut-offs are applied correctly.
- **T_AB** = 1595800947930 (2020-07-26 22:02 UTC), the 85th percentile of pre-T1 timestamps.
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
  Verified on toy data that this changes the results. We use `PopExact` (a RecBole `GeneralRecommender` subclass)
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
- Training rows for SASRec use histories that may include events from earlier periods (A-only training: A instances).
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
- **Training windows:** each user's period-A sequence; every position predicts the next item in one forward pass.
  Sequences longer than L+1 are cut into consecutive windows (stride L) from the end, so every A target is predicted
  exactly once, but a window's context does not reach back into the previous window.
- Same-timestamp events of one user are ordered by `parent_asin` in training sequences (only evaluation histories
  apply the strict "before t" rule).
- **Attention mask:** causal + key padding; each position may always attend to itself so padded (left) positions do
  not become fully masked (NaN). Pre-LN blocks + final LayerNorm, ReLU FFN (as SASRec), embedding dropout 0.2.
- **ID embeddings** (`id_text`, `id_only`) are multiplied by a "seen in period A" mask, so items without A interactions
  get no ID signal (`id_only` gives them a zero vector → score 0).
- **logQ:** `q_j` = static period-A frequency; items unseen in A get log q = 0 (never used, since training targets are
  A items).
- **Early stopping:** Val-subset Recall@100 (20K fixed instances, seed 0) through the harness, i.e. with §4.6 filtering.
- **Comparison caveat:** the retriever trains on period A only; compare it with the A-only baselines
  (e.g. SASRec-A), not the A∪B ones.
