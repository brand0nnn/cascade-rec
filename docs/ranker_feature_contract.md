# Ranker feature interface

The candidate builder stores each split as grouped arrays (`cand`, `score`, `target_pos`, `target_rank`, `inst`, `user`). `RankerFeatureBuilder` expands features only for the current training or scoring batch, keeping the full run within local memory and disk limits.

For a force-inserted target, the builder records its real retriever score and full-catalogue rank for retrieval diagnostics. The MLP uses the score and deliberately excludes retrieval rank: exact rank is available only because we know the target and score it separately, so using it would leak the positive label. The saved `target_rank` remains available for Recall@K and failure analysis, but never enters the ranker.

| Array | Shape | Contents |
|---|---:|---|
| `item_id`, `leaf_category`, `top_category`, `store` | `[rows]` | Categorical IDs; 0 is padding and 1 is OOV |
| `price_bucket`, `review_count_bucket`, `rating_bucket`, `item_age_bucket` | `[rows]` | Static price and period-causal item-stat buckets |
| `history_length_bucket`, `recency_bucket` | `[rows]` | User/context buckets |
| `history_item_ids`, `history_leaf_ids` | `[rows, 50]` | Left-padded, most recent history; shared embedding vocabularies |
| `numeric` | `[rows, 4]` | Standardized retriever score, last-history category match, store-in-history, text cosine |
| `item_text` | `[rows, 384]` | Frozen MiniLM vector, projected by a learned layer |

Item review count, rating, and age use interactions strictly before the start of the candidate's period. An item with no prior interaction receives an unseen age value; future first-interaction timestamps are never used. Candidate training lists force-insert the logged target, with its actual retriever score; this makes every training interaction usable while avoiding a target-only rank feature. Validation early stopping measures macro nDCG@10 only on lists where the frozen retriever actually returned the target, with GAUC reported beside it. The final end-to-end evaluation uses unforced lists so retrieval misses count as misses.

The selected MLP uses `retrieval + matching` features. Loading the six shared MLP/DIN checkpoint bundles and
fresh retraining are separate workflows; see the [README ranker guide](../README.md#ranker-draft--mlp-din-and-validation-experiments)
and [artifact inventory](ranker_artifacts.md). Full processed data, frozen text vectors and the original candidate
files remain local/external prerequisites, including for the fixed-user development run.

Run the selected MLP recipe on the development subset first, then train with an unused full-run ID:

```bash
.venv/bin/python -m cascade.pipelines.ranker_training \
  --config configs/ranker/ladder_1_matching.yaml \
  --candidate-dir data/processed/candidates/two_tower_text_seed0 \
  --run-id my-matching-check --device cpu --sample
.venv/bin/python -m cascade.pipelines.ranker_training \
  --config configs/ranker/ladder_1_matching.yaml \
  --candidate-dir data/processed/candidates/two_tower_text_seed0 \
  --run-id my-matching-seed0 --device cpu
```

Use `cuda` or `mps` instead of `cpu` on a compatible machine. `configs/ranker/mlp.yaml` retains the full feature
setup for comparison; it is not the selected matching-only model. Each config records its seed. Use copied configs
with seeds 0/1/2 and distinct IDs for a three-seed rerun. The grouped `--sample` workflow selects fixed full-data
users and runs one epoch; its values are development checks, not research results.

The separate `--train` / `--val` aligned-NPZ interface remains available for toy examples. For full candidate lists, the pipeline samples negative positions per list and gathers features batch by batch rather than materializing repeated history and text arrays for every candidate.

## Cumulative feature experiments

`configs/ranker/ladder_*.yaml` enables groups explicitly. `retrieval` is mandatory. `matching` adds history/context and text relationships; `attributes` adds candidate item identity, categories, store and price; `statistics` adds period-causal item aggregates. The full group set keeps the original full-model checkpoint layout. Disabled groups do not enter the scorer and do not create unused trainable tables. The score-only builder bypasses history expansion.

Run the development check and then full-data discovery:

```bash
.venv/bin/python -m cascade.pipelines.ranker_ladder --candidate-dir data/processed/candidates/two_tower_text_seed0 --run-id my-ladder-check --device cpu --sample
.venv/bin/python -m cascade.pipelines.ranker_ladder --candidate-dir data/processed/candidates/two_tower_text_seed0 --run-id my-ladder-discovery --device cpu
```

The current meso/plot scripts hard-code the original seed-0 discovery ID `20261024-feature-ladder-seed0`.
A fresh ladder ID does not change that reference. Adjust analysis references explicitly before claiming a report
summarises an entirely fresh rerun. Checkpoint selection remains conditional per-user macro nDCG@10; GAUC is secondary.

Validation records both nDCG and the five outcomes. A rescue moves a retrieved target from rank 11–200 to rank 1–10; a demotion moves it from rank 1–10 to rank 11–200. Missing targets stay retrieval misses. All saved rank comparisons use pessimistic score ties and exclude padding.
