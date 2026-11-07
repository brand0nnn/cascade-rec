# Context-arm follow-ups (no training; Val only) — pre-stated before running

Written 6 Nov 2026 before any follow-up scoring. These analyses use the three locked `bce_context` checkpoints from `results/local_experiments/20261106-context-ranking/`. They do no retraining and select nothing.

1. **Attribution.** Each context group is replaced in turn by its prefix-B mean, which is standardised value 0: price, review_count, mean_rating, item_age, history_length, time_since_last. A second arm neutralises all three candidate period columns together (review_count + mean_rating + item_age). Each arm reports the nDCG@10 drop versus full `bce_context`, overall/new/warm, with paired-user CIs.
2. **Leak re-check.** For the strongest group, compare target and non-target new candidates (first_ts ≥ T1) and report the target rate per bucket. Among new candidates, report the target rate by the gap between query time and first review, and confirm that no context column varies with that gap. Split the new-item gain into same-timestamp targets and the remaining new targets.
3. **Consistent snapshot.** Score with common columns at T1 (from `20261106-freshness-val`) plus context at T1, so every input is taken at the period start. Compare against `bce_text` at T1 and against `bce_context`.

**Pre-stated expectations:**
- item_age/review_count (newness) drives most of the new-item gain.
- The leak check passes.
- Under full period-start consistency, the context gain over bce_text(T1) is smaller than the gain over bce_text(T0), but still positive.

## Results (added after scoring; CPU, natural-hit rows only)
Reproduction: CPU re-scoring matches the saved MPS Val ranks exactly for seeds 0 and 2. Seed 1 differs on one event, with no top-10 flips.

**Attribution.** nDCG@10 drop versus full bce_context [95% CI]:

| Neutralised | Overall | New | Warm |
|---|---|---|---|
| Candidate period columns (all three) | −0.004132 [−0.004406, −0.003868] | −0.013307 | +0.000364 |
| item_age | −0.001794 | −0.003421 | −0.000981 |
| review_count | −0.001662 | −0.003245 | −0.000952 |
| mean_rating | −0.000683 | −0.004683 | +0.001226 |
| price | −0.000591 | +0.000204 (CI includes 0) | −0.000924 |
| time_since_last | −0.000144 | | |
| history_length | −0.000004 (CI includes 0) | | |

The three period columns overlap almost completely for new items: "new" age, zero reviews and no rating identify the same items. Their individual drops therefore understate the joint drop.

**Leak re-check.**
- Every new Val candidate has one identical row of period columns, whatever the gap between query time and first review.
- Target rate among new candidates by gap: same timestamp 656/656; (0,1d] 0.23%; (1,7d] 0.21%; (7,30d] 0.17%; (30,90d] 0.11%; >90d 0.08%. The model cannot see this gradient.
- Excluding the 656, the "new" bucket's target rate is still 0.105%. That is above most warm buckets and well above the all-slot base rate of 0.032%.
- The gain is not driven by same-timestamp targets. Context(T1) minus bce_text(T0) is +0.009205 [0.008598, 0.009738] on the 68,518 new events with gap > 0.
- Caveat: the 5-core filter was computed over the whole timeline, so every new product is guaranteed later reviews. That plausibly inflates the target rate of recently released products. This is the known §4.4 limitation, and a newness prior partly exploits it.

**Consistent snapshot (everything at T1).** nDCG@10:

| Contrast | Overall | New | Warm |
|---|---|---|---|
| Versus bce_text(T1) | +0.001273 [0.001052, 0.001501] | +0.002334 [0.001982, 0.002706] | +0.000622 [0.000372, 0.000857] |
| Versus bce_text(T0) | +0.002564 | | |
| Versus full bce_context | −0.000165 [−0.000285, −0.000038] | −0.004020 | +0.001521 |

All three pre-stated expectations hold. The gain under full period-start consistency is smaller but still positive.
