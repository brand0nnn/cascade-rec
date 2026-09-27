# CLAUDE.md — Cascade Recommender Project (CP4285)

**Project title:** Retrieve, Then Rank: How Much Does Candidate Retrieval Limit Product Recommendation?

This file is the source of truth for implementing the project. Read it fully before writing code. If something here is ambiguous or seems wrong, **stop and ask** rather than guessing — especially anything touching the data split, evaluation protocol, or metric definitions, because those decide whether results are valid.

---

## 1. What we are building and why

We build a **two-stage retrieve-then-rank recommender** on the **Cell_Phones_and_Accessories** category of **Amazon Reviews 2023**, then run a controlled **cascade study** measuring how retrieval choices limit the quality of the final ranked list.

- **Stage 1 — Retrieval:** a two-tower model (text-based item tower + SASRec-style self-attention user tower), trained with in-batch softmax and logQ correction (Yi et al., 2019), served with FAISS (exact / HNSW / IVF).
- **Stage 2 — Ranking:** a feature-based MLP ranker trained on candidate lists produced by the frozen retriever. LightGBM LambdaRank is a second ranker used as a robustness check.
- **Cascade study:** freeze the ranker, vary one retrieval setting at a time (candidate count K₁, search approximation, retriever variant), and measure retrieval recall, end-to-end nDCG@10 and latency.

### Research questions

| ID | Question | Status |
|---|---|---|
| RQ1 | Which retriever components improve candidate recall? (transformer vs mean-pool user tower; logQ on/off; text-only vs ID+text vs ID-only item tower) | Core |
| RQ2 | How do K₁, approximate search settings and retriever quality bound end-to-end nDCG@10? Do conclusions hold across two rankers (MLP, LightGBM)? | **Core — headline** |
| RQ3 | Do DIN attention and DCN-V2 feature crosses improve ranking over the MLP? | **Stretch only** |

### Scope

**Core (must deliver):** data pipeline, baselines, retriever + RQ1 ablations, FAISS index configs, MLP + LightGBM rankers, cascade sweeps (RQ2), segment analysis, significance tests.

**Stretch (only after core results exist; go/no-go ≈ 26 Oct 2026):** DIN ranker, DIN + DCN-V2 ranker (RQ3), K₁ sweep with the strongest ranker, MMR diversity re-ranker.

**Out of scope:** multi-source retrieval merging, pre-ranking stage, multi-task ranking, online/A-B evaluation, serving infrastructure, knowledge graphs, fine-tuning the text encoder.

---

## 2. Notation (use consistently in code, logs and plots)

| Symbol | Meaning |
|---|---|
| **K₁** (`k1`) | Number of candidates retrieval passes to the ranker |
| **10** | Length of the final list shown to the user |
| **Recall@K / nDCG@K** | Metric computed at cut-off K |
| **L** (`max_hist_len`) | Max history length fed to models (= 50) |
| **N** | Catalogue size (number of items) |
| **τ** (`temperature`) | Softmax temperature in the retriever (= 0.05) |

Never use a bare `k` for more than one of these meanings in the same function.

---

## 3. Repository layout

```
cascade-rec/
├── CLAUDE.md                  # this file
├── README.md                  # how to reproduce every result
├── requirements.txt
├── configs/                   # one YAML per experiment; nothing hard-coded
│   ├── data.yaml
│   ├── baselines/*.yaml
│   ├── retriever/*.yaml       # v1_meanpool, v2_transformer, v3_no_logq, v4_id_text, v5_id_only
│   ├── ranker/*.yaml          # mlp, mlp_no_retrieval_score, lgbm, (stretch) din, din_dcnv2
│   └── cascade/*.yaml         # k1_sweep, index_sweep, retriever_sweep
├── data/                      # gitignored
│   ├── raw/
│   └── processed/
├── src/cascade/
│   ├── data/        download.py, preprocess.py, split.py, instances.py, features.py, stats.py
│   ├── text/        encode.py
│   ├── models/      popularity.py, itemknn.py, bpr.py, sasrec_baseline.py,
│   │                retriever.py, ranker_mlp.py, ranker_lgbm.py,
│   │                (stretch) din.py, dcnv2.py, rerank_mmr.py
│   ├── index/       faiss_index.py
│   ├── eval/        metrics.py, harness.py, bootstrap.py, segments.py, latency.py
│   ├── pipelines/   train_baseline.py, train_retriever.py, build_candidates.py,
│   │                train_ranker.py, run_cascade.py
│   └── utils/       config.py, seed.py, io.py, logging.py
├── scripts/                   # thin CLI entry points that call src/
├── tests/                     # pytest unit tests (see §12)
├── results/                   # results/<experiment>/<run_id>/{metrics.json, config.yaml, git_hash.txt}
├── notebooks/                 # analysis and plotting ONLY — never training
└── docs/decisions.md          # log every design decision and deviation from this plan
```

### Engineering conventions

- Python 3.10+, PyTorch, type hints, docstrings on public functions.
- **Config-driven:** every run is launched from a YAML config; the config, git hash and seed are saved with the results.
- **Seeds:** all experiments run with seeds `{0, 1, 2}`; set Python, NumPy and PyTorch seeds, and cuDNN deterministic where feasible.
- **Toy mode:** every pipeline accepts `--toy` which uses a fixed ~2% user sample, for fast debugging. Never report toy results.
- **One evaluation harness** (`src/cascade/eval/harness.py`) is used by *every* model, baseline included. No model computes its own metrics.
- Cache expensive artefacts (text embeddings, candidate lists, item vectors) to `data/processed/` with the config hash in the filename.
- Record every non-trivial design decision or deviation in `docs/decisions.md` with date and reason.

### Dependencies

`torch`, `numpy`, `pandas` (or `polars`), `pyarrow`, `sentence-transformers`, `faiss-cpu` (and `faiss-gpu` if available), `lightgbm`, `scikit-learn`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, `pytest`. Optional: `recbole` (cross-check only, see §6).

---

## 4. Data

### 4.1 Source files

Dataset: **Amazon Reviews 2023** (Hou et al., 2024), McAuley Lab. Site: https://amazon-reviews-2023.github.io/

- **Interactions (5-core, ratings only)** — columns `user_id, parent_asin, rating, timestamp` (timestamp in **milliseconds**):
  `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Cell_Phones_and_Accessories.csv.gz`
  Expected size after 5-core: ≈ 381.0K users, 111.5K items, 2.8M ratings. Verify and log the actual numbers.
- **Item metadata** — `meta_Cell_Phones_and_Accessories.jsonl(.gz)`. **Verify the download URL on the dataset site before use** (it is also on Hugging Face as `McAuley-Lab/Amazon-Reviews-2023`, config `raw_meta_Cell_Phones_and_Accessories`). Join on `parent_asin`.

`download.py` must record source URL, download date and SHA-256 checksum of every file in `data/raw/MANIFEST.json`.

### 4.2 Metadata fields

| Field | Use |
|---|---|
| `title` | Item text (item tower), ranker text feature |
| `categories` (list, coarse → fine) | Item text; leaf category and top-level category IDs |
| `store` | Item text; ranker categorical feature |
| `price` | Ranker feature (bucketed; may be missing or non-numeric — parse robustly, missing → own bucket) |
| `average_rating`, `rating_number` | **DO NOT USE.** Computed at crawl time, includes future reviews → leakage. Compute rating stats from our own interaction data instead (§4.5). |

Items in the interaction file with no metadata: keep them, with empty text fields; log how many.

### 4.3 Interactions

- One interaction = one row (a review with a rating) = (user, item, timestamp). All ratings count as positive interactions regardless of star value.
- Deduplicate exact duplicates; if a user reviews the same item twice, keep the earliest.
- Sort each user's interactions by timestamp (tie-break by `parent_asin` for determinism).
- Map `user_id` and `parent_asin` to contiguous integer IDs; index 0 is reserved for padding.

### 4.4 Split (global, time-based)

Use the **official benchmark cut-offs** so our test set matches the public absolute-timestamp split:

- `T1 = 1628643414042` (≈ 2021-08-11) — start of validation
- `T2 = 1658002729837` (≈ 2022-07-16) — start of test

The training portion (`timestamp < T1`) is further cut at `T_AB` into two periods:

| Period | Time range | Used for |
|---|---|---|
| **A** | `< T_AB` | Retriever training; baseline training |
| **B** | `[T_AB, T1)` | Ranker training (candidates from the frozen retriever) |
| **Val** | `[T1, T2)` | Early stopping and tuning for all models |
| **Test** | `≥ T2` | Final evaluation only |

- `T_AB` = the timestamp at the **85th percentile** of training-period (`< T1`) interactions (configurable: `period_b_fraction: 0.15`). Log the resulting date.
- **Baselines** train on A ∪ B (all training data) *and* on A only; report the A∪B version as the main baseline, since that is their fair full-data setting. Log both.
- Known limitation (document it, do not fix): the official 5-core filter was computed over the whole timeline, so it uses future information about which users/items survive filtering.
- **The test period is never used for any tuning, early stopping or feature statistics.** Val is used for all selection. Test is run once per final configuration.

### 4.5 Instances

An **instance** = (user, history, target item, timestamp, period).

- For every interaction at time `t` in a period, the history is **all of that user's interactions strictly before `t`** (from any earlier period, including earlier interactions in the same period), truncated to the most recent `L = 50`.
- Instances with **empty history are skipped** (cold users are out of scope). Log the count per period.
- **Retriever training** uses SASRec-style sequences over period A (every position predicts the next item; see §7).
- **Ranker training** uses instances whose target falls in period B.
- **Val / test** instances are all interactions in Val / Test with non-empty history.

**Item statistics as features** (review count, mean rating, item age = time since the item's first interaction): for each instance, compute them from **all interactions strictly before the start of the instance's period** (A-start for A, B-start for B, T1 for Val, T2 for Test). Implement as precomputed per-period tables, not per-instance.

### 4.6 Item pool and candidate filtering

- The item pool is every item in the 5-core file.
- At retrieval and ranking time, for each instance remove from the candidate list:
  1. items already in the user's full history before `t`;
  2. items whose **first interaction is after `t`** (not yet "released" at `t`). An item whose first interaction is exactly the target event is allowed — this is how cold items enter the test set.
- Implementation: query FAISS for `K₁ + buffer` (buffer = `max_hist_len + 200`, configurable), filter, keep the top K₁. Log how often filtering leaves fewer than K₁ candidates (should be ~never).

### 4.7 Item segments (for analysis)

Based on interaction counts **before T2** (i.e. all training + val data):

- **cold**: 0 interactions before T2
- **tail**: bottom 80% of items with ≥1 interaction
- **head**: top 20% of items with ≥1 interaction

User segments by history length at instance time: `1–2`, `3–5`, `6–10`, `11+`.

### 4.8 Dataset statistics (needed for the critique — do this first)

`stats.py` writes `results/stats/stats.json` and figures:
- #users, #items, #interactions, density, per period
- median and mean history length; distribution plot
- interactions-per-item histogram (log–log) — shows the long tail
- % of test instances whose target is a **cold** item; % head / tail
- number of instances per period; number skipped for empty history
- metadata coverage (% items with title, categories, store, parseable price)

---

## 5. Evaluation harness (build before any model)

All models expose one interface to the harness:

```python
class Recommender(Protocol):
    def score_candidates(self, batch: InstanceBatch, candidates: LongTensor) -> Tensor: ...
    # or, for full-catalogue models:
    def score_all(self, batch: InstanceBatch) -> Tensor: ...  # [B, N]
```

The harness applies the §4.6 filtering, ranks, and computes metrics. **Metrics are full-ranking over the filtered catalogue — never sampled negatives** (Krichene & Rendle, 2020).

### Metric definitions (single relevant item per instance)

- `rank` = 1-based position of the target in the ranked list (∞ if absent).
- **Recall@K** = 1 if `rank ≤ K` else 0.
- **nDCG@K** = `1 / log2(rank + 1)` if `rank ≤ K` else 0.
- **Conditional nDCG@10** = nDCG@10 computed only over instances where the target is in the K₁ candidate list (isolates ranking skill).
- **GAUC** (rankers only): AUC of the ranker's scores within each instance's candidate list (target vs the other candidates), averaged per user, then over users.
- **Coverage** = number of distinct items appearing in any top-10 list / N.
- **Popularity concentration** = share of all top-10 slots filled by the 1% most-interacted items (counts before T2).

### Aggregation and significance

- **Per-user macro average:** average each metric over a user's instances, then over users. Also log the plain per-instance (micro) average.
- **Seeds:** report mean ± std over seeds `{0, 1, 2}`.
- **Significance:** paired bootstrap over users (1,000 resamples) for the difference between two systems; report the 95% CI of the difference. Put this in `bootstrap.py`.
- Every result table row also reports the metric per segment (§4.7).

### Latency (`latency.py`)

- CPU, single thread (`faiss.omp_set_num_threads(1)`, `torch.set_num_threads(1)`), queries issued one at a time, 1,000 random test instances, after 50 warm-up queries.
- Report median and p95 separately for: user-tower forward pass, FAISS search, ranker scoring of K₁ candidates, and total.
- Always on the same machine; record CPU model in the results.

---

## 6. Baselines (Phase 0 — needed by Mon 29 Sep 2026)

All baselines are evaluated through **our harness** on **our split** (full ranking, §5).

| Baseline | Implementation |
|---|---|
| **Popularity** | Rank items by interaction count in the training data. |
| **ItemKNN** (Sarwar et al., 2001) | Cosine similarity on item co-occurrence (user–item matrix from training data); keep top-100 neighbours per item; score = sum of similarities to the user's history items. |
| **BPR-MF** (Rendle et al., 2009) | 64-d user and item ID embeddings, BPR loss with 1 uniform negative, Adam, early stop on Val Recall@100. Users not seen in training → skip and log. |
| **SASRec** (Kang & McAuley, 2018) | ID embeddings; 2 self-attention blocks, **1 head**, d = 64, L = 50, dropout 0.2; **binary cross-entropy with 1 random negative per position** (the original recipe). Implement in `sasrec_baseline.py` by reusing the user-tower code from `retriever.py` with an ID-only item table. |

**RecBole** (Zhao et al., 2021) is used **only as a cross-check**: run BPR and SASRec in RecBole on a converted copy of our split and confirm our implementations land in the same ballpark. If RecBole's numbers differ a lot, investigate before trusting either. Note in `docs/decisions.md` that the baselines are implemented in our codebase with RecBole as a reference, so the critique slides can be kept accurate.

**Phase 0 acceptance:** a results table (Recall@10, nDCG@10, Recall@100; per-user macro; mean over 3 seeds) for all four baselines on Val and Test, plus `stats.json`. Popularity must clearly beat random; if SASRec does not beat popularity, investigate before continuing.

---

## 7. Stage 1 — Retriever (two-tower)

### 7.1 Item tower

1. **Text:** `f"{title} | {' > '.join(categories)} | {store}"` (skip empty parts).
2. **Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (384-d), **frozen**. Encode every item once, normalise, cache to `data/processed/item_text_emb.npy`.
3. **Projection:** MLP 384 → 256 → 64 (GELU, dropout 0.1). Output `e_i` (64-d, unnormalised; used as input tokens to the user tower).
4. **Variants:**
   - `text` (default): as above.
   - `id_text`: add a learned 64-d ID embedding to `e_i` for items with ≥1 interaction in period A; zeros otherwise (so cold items fall back to text).
   - `id_only`: learned 64-d ID embedding only (no text).
5. **Scoring vector:** `v_i = normalize(e_i)`.

### 7.2 User tower

- Input: the last L = 50 item vectors `e_i` from the item tower (**shared weights**), left-padded, plus learned positional embeddings.
- 2 causal self-attention blocks, **1 head** (matches SASRec), d = 64, FFN dim 256, dropout 0.2, pre-LayerNorm, residuals, padding mask + causal mask.
- Output `u` = hidden state at the last non-padded position, L2-normalised.
- Variant `meanpool`: `u = normalize(mean of non-padded e_i)` — no attention.

### 7.3 Score and loss (Yi et al., 2019)

- Score: `s(u, i) = (u · v_i) / τ`, τ = 0.05.
- **Training targets:** SASRec-style — for each sequence in period A, every position `p` predicts item `p+1`; all positions computed in one forward pass via the causal mask.
- **In-batch softmax:** flatten all valid (position, target) pairs in the batch; if more than `max_inbatch = 4096`, subsample 4096. For each pair, the negatives are the other targets in that set.
- **logQ correction:** subtract `log(q_j)` from every logit, where `q_j` = item j's frequency among period-A interactions (count / total). Applied to all logits including the positive (standard formulation). Yi et al. estimate `q_j` from streaming data; we use static counts — document this simplification.
- **Duplicate masking:** if item j equals the target of pair i but j is a different pair, set that logit to `-inf`.
- Optimiser Adam, lr 1e-3, batch 256 sequences, up to 50 epochs, early stopping (patience 5) on **Val Recall@100 with exact search** computed on a fixed random subset of 20K val instances.

### 7.4 Retriever variants (RQ1)

| Config | User tower | Item tower | Loss |
|---|---|---|---|
| `v1_meanpool` | mean-pool | text | in-batch softmax + logQ |
| `v2_transformer` (**main**) | transformer | text | in-batch softmax + logQ |
| `v3_no_logq` | transformer | text | in-batch softmax, **no** logQ |
| `v4_id_text` | transformer | id_text | in-batch softmax + logQ |
| `v5_id_only` | transformer | id_only | in-batch softmax + logQ |

Comparisons this enables: v1 vs v2 (attention), v2 vs v3 (logQ), v2 vs v4 vs v5 (item representation), **SASRec baseline vs v5** (isolates the loss change: BCE-1-negative vs in-batch softmax + logQ).

Report for every variant, with **exact search**: Recall@{50, 100, 200, 500}, per segment (especially **cold items** and history-length buckets). Select the variant with the best Val Recall@200 as the **frozen retriever** for Stage 2 (expected: v2 or v4). Record the choice in `docs/decisions.md`.

---

## 8. Nearest-neighbour index (FAISS)

Item vectors `v_i` are precomputed; inner product = cosine (vectors are normalised).

| Index | Construction | Search settings |
|---|---|---|
| Exact | `IndexFlatIP` | — (reference) |
| HNSW | `IndexHNSWFlat(d=64, M=32, METRIC_INNER_PRODUCT)`, `efConstruction = 200` | `efSearch ∈ {K₁, 2K₁, 4K₁}` — **efSearch must be ≥ the number of results requested** |
| IVF | `IndexIVFFlat(quantizer=IndexFlatIP, d=64, nlist, METRIC_INNER_PRODUCT)`, `nlist = 1024` (≈ 3–4·√N), trained on all item vectors | `nprobe ∈ {1, 8, 32}` |

For each approximate config, report **recall relative to exact search** (fraction of the exact top-K₁ that the index returns) alongside end-to-end metrics and latency.

Sanity test: IVF with `nprobe = nlist` must return exactly the exact-search results.

---

## 9. Stage 2 — Rankers (core)

### 9.1 Candidate lists

`build_candidates.py`, using the **frozen retriever** and **exact search**:

| Split | Instances | List size | Force-insert positive? |
|---|---|---|---|
| Ranker train | Period B | 200 | **Yes** — if the target is missing, append it (label 1) |
| Ranker val (loss / GAUC / early stopping) | Val | 200 | Yes |
| Val / Test end-to-end | Val / Test | K₁ (from the sweep) | **No** — misses are real misses |

Labels: target = 1, other candidates = 0. These zeros are *unobserved*, not confirmed dislikes; say so in the report.

For MLP training, subsample **50 negatives per list** uniformly from the non-target candidates (configurable). LightGBM uses full lists.

Save for every list: instance id, candidate ids, retrieval scores, retrieval ranks.

### 9.2 Features (`features.py`)

All item statistics come from the per-period tables in §4.5 (no leakage).

| Group | Feature | Encoding |
|---|---|---|
| Candidate | item ID | embedding 32-d; items with < 2 training interactions → shared OOV id |
| | leaf category, top-level category | embedding 16-d each |
| | store | embedding 16-d; stores with < 5 items → OOV |
| | price | quantile bucket (10) + "missing" → embedding 8-d |
| | review count (as of period start) | log-bucket → embedding 8-d |
| | mean rating (as of period start) | bucket (incl. "no ratings") → embedding 8-d |
| | item age (time since first interaction) | log-bucket → embedding 8-d |
| | text embedding | frozen 384-d MiniLM vector → learned linear 384 → 32 |
| User | history item IDs (last 50) | same item-ID embedding table, **mean-pooled** (masked) |
| | history leaf categories | same category table, mean-pooled |
| | history length | bucket → embedding 8-d |
| Pair / context | retrieval score | float (standardised) |
| | retrieval rank | `log1p(rank)` |
| | candidate leaf category == last history item's leaf category | binary |
| | candidate store appears in history | binary |
| | time since user's last interaction | log-bucket → embedding 8-d |
| | cosine(candidate text emb, mean history text emb) | float |

### 9.3 MLP ranker (main)

- Concatenate all feature embeddings and floats → MLP 256 → 128 → 64 (ReLU, dropout 0.1) → 1 logit.
- Binary cross-entropy, Adam lr 1e-3, batch 2048 rows, early stopping (patience 3) on **ranker-val GAUC**.
- Variant `mlp_no_retrieval_score`: drop retrieval score **and** retrieval rank.

### 9.4 LightGBM ranker (robustness check)

- `lightgbm.LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[10])`, `n_estimators` up to 1000 with early stopping (50 rounds) on ranker-val, `learning_rate=0.05`, `num_leaves=63`. One group per candidate list.
- Same features, flattened: categorical IDs as LightGBM categorical features (item ID excluded — too high-cardinality), history summarised as counts (history items in same leaf category; in same store), plus the float features.
- Not implemented via RecBole (its LightGBM wrapper is set up for pointwise prediction; we need grouped lists).

### 9.5 Ranker comparison (reported at K₁ = 200, exact search)

| Row | System |
|---|---|
| R0 | Retrieval order (no ranker) |
| R1 | LightGBM LambdaRank |
| R2 | MLP ranker |
| R2′ | MLP without retrieval score/rank |

**Required:** R2 must beat R0 on Val nDCG@10. If it does not, the cascade study is not meaningful — stop and report to the team before running RQ2.

---

## 10. Stage 3 — Cascade study (RQ2)

**Rankers are trained once and frozen.** Only retrieval settings change. Each sweep is run with **both** the MLP (R2) and LightGBM (R1).

| Sweep | Varied | Fixed |
|---|---|---|
| 1. K₁ (headline) | K₁ ∈ {50, 100, 200, 500, 1000} | exact search, frozen retriever |
| 2. Index | exact; HNSW efSearch ∈ {200, 400, 800}; IVF nprobe ∈ {1, 8, 32} | K₁ = 200, frozen retriever |
| 3. Retriever | v1–v5 | K₁ = 200, exact search |

For every run, record on Test (and Val):
- Retrieval **Recall@K₁** (the ceiling)
- End-to-end **Recall@10** and **nDCG@10**
- **Conditional nDCG@10**
- **Failure breakdown:** % of instances that are retrieval misses (target not in K₁) vs ranking misses (in K₁ but ranked > 10) vs hits
- Latency breakdown (§5)
- For index sweep: recall relative to exact search
- All metrics per segment

Caveat to log in `docs/decisions.md`: rankers were trained on 200-candidate lists; at K₁ = 1000 they see longer lists than in training.

### Required outputs (plots in `results/figures/`)

1. **Headline:** Recall@K₁, nDCG@10 (MLP and LightGBM) and total latency vs K₁ (log x-axis).
2. Index sweep: recall-vs-exact and nDCG@10 vs latency, one point per config.
3. Retriever sweep: Recall@200 vs end-to-end nDCG@10, one point per variant.
4. Failure breakdown stacked bars per K₁.
5. Segment breakdowns (cold / tail / head; history length).

Deliverable decision: the **smallest K₁ at which nDCG@10 is within the bootstrap CI of its maximum** — report it as the recommended K₁.

---

## 11. Stretch work (only after core results exist)

Only start when §10 sweep 1 is complete for both rankers and the team gives the go-ahead.

### 11.1 DIN ranker (Zhou et al., 2018)
Replace the MLP's mean-pooled history with **target attention**: for each history item `h_j` (item emb ⊕ category emb) and candidate `c` (same fields), weight `a_j = MLP_att([h_j, c, h_j − c, h_j ⊙ c])` (hidden 64 → 32, ReLU → 1, **no softmax**, masked for padding); interest = Σ a_j h_j. Everything else as the MLP ranker. Compare against R2 via GAUC and nDCG@10.

### 11.2 DIN + DCN-V2 (Wang et al., 2021)
Concatenated input `x0` → 3 cross layers `x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l` (full-rank W) **in parallel** with the deep MLP; concatenate both outputs → linear → logit.

Reference implementations to check against (not to import blindly): FuxiCTR and RecBole's DIN / DCNV2.

### 11.3 RQ3 ladder and cross-check
Report R0, R1, R2, R3 = DIN, R4 = DIN + DCN-V2 on identical candidate lists. Then re-run cascade sweep 1 with R4 and plot it alongside the MLP and LightGBM curves.

### 11.4 MMR diversity re-ranker
Greedy MMR over the ranker's top-50: `argmax λ·score(i) − (1−λ)·max_{j∈S} sim(i, j)`, sim = text-embedding cosine, λ ∈ {1.0, 0.9, 0.7}. Report nDCG@10 vs intra-list diversity and category coverage.

---

## 12. Tests (pytest, `tests/`)

Write these alongside the code they test:

- `test_metrics.py` — Recall/nDCG/GAUC on hand-computed toy examples; nDCG = 1 when target ranked first.
- `test_split.py` — no Val/Test timestamps in training data; period boundaries correct; T_AB between min and T1.
- `test_no_leakage.py` — item statistics for an instance never use interactions at or after its period start; histories never contain the target or later events.
- `test_filtering.py` — candidates never include history items or not-yet-released items.
- `test_logq.py` — logQ correction subtracts the right values; duplicate masking works.
- `test_index.py` — exact FAISS top-K equals brute-force NumPy top-K; IVF with `nprobe = nlist` equals exact.
- `test_candidates.py` — train/ranker-val lists always contain the positive; test lists never force it.

Run tests before every experiment batch; do not proceed on a red test.

---

## 13. Phases, milestones and acceptance criteria

| Phase | Dates (2026) | Work | Done when |
|---|---|---|---|
| 0 | now → **29 Sep** (critique) | Download, preprocess, split, stats, harness, 4 baselines | `stats.json` + baseline table on Val/Test, tests green |
| 1 | 30 Sep – 12 Oct | Text encoding, retriever v1–v5, RQ1 table; FAISS index module; ranker feature pipeline | RQ1 table with segments; frozen retriever chosen |
| 2 | 13 – 26 Oct | Candidate lists; MLP and LightGBM rankers; R0–R2′ table; cascade sweeps 1–3 | R2 beats R0; all sweep results + headline plot |
| 3 | 27 Oct – 4 Nov | Significance tests, segment analysis, failure breakdown, ethics metrics; **stretch work if go/no-go passed** | Final tables and figures reproducible from configs |
| 4 | 5 – 11 Nov | STePS poster (11 Nov), README reproduction guide | Every poster number traceable to a `results/` run |

### Team ownership (coordinate through the shared harness)
- **[Member A] Evaluation & analysis:** §4 data, §5 harness, §6 baselines, §10 sweeps, significance, ethics metrics.
- **[Member B] Retrieval:** §7 retriever and RQ1 variants.
- **[Member C] Ranking & search:** §8 FAISS index + latency, §9 rankers, §11 stretch.

---

## 14. Compute and practical notes

- Target: one GPU (Google Colab or NUS SoC cluster) for training; CPU for FAISS latency.
- MiniLM encoding of ~111.5K items is a one-off job; cache it.
- Candidate lists at K₁ = 1000 for all test instances can be large — store as int32 arrays in `.npy`/Parquet, not Python lists.
- Prefer command-line scripts for training (reproducible, resumable); notebooks only read from `results/`.

---

## 15. Rules for Claude Code

1. **Ask before changing** the split, filtering, metric definitions, positive/negative definitions, or anything in §4–§5.
2. **Never** use Test data for tuning, early stopping, feature statistics or model selection.
3. **Never** use metadata `average_rating` or `rating_number`.
4. Build and test the harness (§5) before any model; every model goes through it.
5. Start every new pipeline in `--toy` mode; only scale up once it runs end to end.
6. Keep implementations simple and readable; follow the paper recipes cited here and note any deviation in `docs/decisions.md`.
7. Every reported number must come from a saved run in `results/` with its config, seed and git hash.
8. Don't start stretch work (§11) unless asked.
9. When results look surprising (e.g. a baseline beating the retriever, recall near 0 or 1), investigate for bugs and leakage before reporting.

---

## 16. References

- Burges, C.J.C. (2010) *From RankNet to LambdaRank to LambdaMART: An overview*. Microsoft Research Technical Report MSR-TR-2010-82.
- Covington, P., Adams, J. and Sargin, E. (2016) 'Deep neural networks for YouTube recommendations', *RecSys '16*.
- Hou, Y. et al. (2024) 'Bridging language and items for retrieval and recommendation' (Amazon Reviews 2023), arXiv preprint.
- Johnson, J., Douze, M. and Jégou, H. (2021) 'Billion-scale similarity search with GPUs', *IEEE Transactions on Big Data*, 7(3).
- Kang, W.-C. and McAuley, J. (2018) 'Self-attentive sequential recommendation', *ICDM 2018*.
- Ke, G. et al. (2017) 'LightGBM: A highly efficient gradient boosting decision tree', *NeurIPS 2017*.
- Krichene, W. and Rendle, S. (2020) 'On sampled metrics for item recommendation', *KDD '20*.
- Malkov, Y.A. and Yashunin, D.A. (2020) 'Efficient and robust approximate nearest neighbor search using HNSW graphs', *IEEE TPAMI*, 42(4).
- Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT', *EMNLP-IJCNLP 2019*.
- Rendle, S. et al. (2009) 'BPR: Bayesian personalized ranking from implicit feedback', *UAI 2009*.
- Sarwar, B. et al. (2001) 'Item-based collaborative filtering recommendation algorithms', *WWW '01*.
- Wang, R. et al. (2021) 'DCN V2', *WWW '21*. (stretch)
- Yi, X. et al. (2019) 'Sampling-bias-corrected neural modeling for large corpus item recommendations', *RecSys '19*.
- Zhao, W.X. et al. (2021) 'RecBole', *CIKM '21*.
- Zhou, G. et al. (2018) 'Deep interest network for click-through rate prediction', *KDD '18*. (stretch)
