# Candidate-list contract for ranking work

This records the interface specified in `CLAUDE.md` §9.1 so retrieval and ranking work can proceed together. It does not change the split, filtering rules, or metrics.

## One row per instance and candidate

Store candidate lists as a Parquet table with these fields:

| Field | Meaning |
|---|---|
| `inst` | Stable instance ID from `data/processed/instances.parquet` |
| `item` | Candidate item ID from `data/processed/items.parquet` |
| `retrieval_score` | Two-tower similarity before ranking |
| `retrieval_rank` | 1-based position after seen-item and release-time filtering |
| `label` | 1 if the candidate is the instance's target item, otherwise 0 |

Save the split, retriever configuration/hash, `k1`, exact-search setting, and whether a positive was force-inserted alongside the table. Rows must be unique by (`inst`, `item`). Keep a separate instance table with the user, timestamp, and full target ID so misses remain measurable even when the target is absent from a candidate list.

**Training caveat:** when a target is force-inserted, a special retrieval rank or missing-score value appearing only on positive rows would reveal the label to the MLP. Record insertion for auditing, but do not feed such a flag or sentinel to the model. Resolve the score/rank feature treatment with the team before training the main MLP; the documented `mlp_no_retrieval_score` variant is a useful first implementation.

## Required lists

| Use | Period | Size | Missing positive |
|---|---|---:|---|
| Ranker training | B | 200 retrieved candidates, then append target if absent | Force-insert for supervised training |
| Ranker validation/early stopping | Val | 200 retrieved candidates, then append target if absent | Force-insert for supervised validation |
| End-to-end evaluation | Val and Test | Sweep value of `k1` | Never force-insert; count as retrieval miss |

Use `CLAUDE.md` §4.6 filtering before writing lists: exclude items in the user's full preceding history and items whose first interaction is after the request timestamp. The target is allowed at its own first interaction. Candidate labels of 0 mean **unobserved**, not confirmed dislikes.

## Work that can start before retriever candidates arrive

- Implement causal feature extraction from the existing processed items and histories; use period-start item statistics as specified in `CLAUDE.md` §4.5 and §9.2.
- Build and test the MLP training/scoring code against small synthetic candidate lists following this schema.
- Add a deterministic retrieval-order baseline (R0) and compare it with the MLP (R2) on identical validation lists once Brandon's exact-search candidates are ready.
- Use Val for early stopping and model selection. Do not tune on Test. Save the config, seed, retriever identity, and git hash with each run.
