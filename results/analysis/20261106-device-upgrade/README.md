# Device-upgrade candidates in the recency pool (no training)

**Val only, exploratory; title mentions only, not compatibility claims; no training.**

Follows `results/analysis/20261106-switch-cases-devices`. Can retrieval act on device change? Pool D keeps P1
(recency, frozen MiniLM, 90-day half-life) ranks 1-180 and adds the top 20 eligible items whose title names a strictly
newer model of the same family as the user's current device. The 20 are ranked by the same recency score, with fill so
that |D| = 200. Rules are frozen in `config.yaml`; the parser is imported unchanged (sha256 checked).

## Pre-stated hypothesis (written 2026-11-06 02:07 +0800, before any upgrade pool was computed)

D increases Recall@200 over P1 for events whose target names a newer same-family model, by a clearly positive amount.
The overall Recall@200 change is small but positive (the removed ranks 181-200 hold few targets), and nDCG@10 changes
little in the append order.

Operational test: the D_append − P1 Recall@200 CI lower bound is > 0 in the newer-same-family subgroup, and the
overall diff is > 0 with CI lower bound > 0. Append order leaves ranks 1-180 unchanged, so the nDCG@10 clause holds by
construction and is reported as such.

Disclosure: before this freeze, a 512-row technical smoke run of the same code in a scratchpad directory checked
filters, CPU-vs-archive score agreement and runtime. Only those checks were read, not its metrics.

## Result (Val only, exploratory; per-user macro over 232,904 events / 124,416 users, misses = 0)

Checks: every one of the 199,507 changed lists passes `pools_build.filter_violations` (0 padding, duplicates, history
items or unreleased items). CPU recency scores match the archive at P1 columns to within 4.8e-7. The CPU top-200 set
equals the archive's in 211,789 of 214,685 rows (98.7%); the differences are ties and float precision. P1 ranks 1-180
are always taken from the archive.

Coverage: 220,264 events have a current device; 214,685 have an ordered line. 199,507 events (85.7%) and 113,164 users
(91.0%) get at least 1 upgrade candidate, and 174,938 events have at least 20. |U| is large: mean 3,494 and median
1,107 per ordered-device event.

| Group | Events / users | Recall@200 P1 → D | D_append − P1 Recall@200 [95% CI] | Gained / lost |
|---|---|---|---|---|
| Overall | 232,904 / 124,416 | 0.054746 → 0.056148 | **+0.001402 [+0.001109, +0.001742]** | 751 / 474 |
| New | 76,768 / 52,420 | 0.065043 → 0.068717 | +0.003673 [+0.002975, +0.004358] | 442 / 175 |
| Warm | 156,136 / 96,764 | 0.050574 → 0.050836 | +0.000262 [−0.000057, +0.000599] | 309 / 299 |
| Target names newer same-family model | 48,364 / 42,971 | 0.056940 → 0.070810 | **+0.013870 [+0.012779, +0.015016]** | 751 / 63 |
| ... and target has at most 3 keys | 30,627 / 28,314 | 0.059207 → 0.083345 | +0.024137 [+0.022374, +0.025797] | 751 / 0 |
| User has a recognised device | 220,264 / 119,162 | 0.054785 → 0.056281 | +0.001496 [+0.001155, +0.001834] | 751 / 474 |

Append order: Recall@10 and nDCG@10 are identical to P1 in every group (diff 0 by construction).

Interleave order (descriptive): overall Recall@10 −0.000248 [−0.000390, −0.000122] and nDCG@10 −0.000072 [−0.000113,
−0.000035]. In the newer-same-family subgroup, Recall@10 is +0.000782 [+0.000295, +0.001237] and nDCG@10 +0.000226
[+0.000085, +0.000358]. Recall@200 is identical to append (same set).

Rescued targets are spread across upgrade ranks 1-20 (91 at rank 1, 35 at rank 20). All 751 gains are, by construction,
targets naming a newer model. The 474 losses are P1 rank 181-200 targets pushed out, mostly warm (299).

Hypothesis: **holds under the frozen test.** The subgroup CI lower bound is > 0 (+0.0139, a 24% relative gain), and
the overall gain is small but positive (+0.0014, 2.6% relative). The nDCG@10 clause holds trivially in append order.
Absolute impact is small: 751 gained targets out of 48,364 newer-model events. The upgrade set is broad (median 1,107
items), and the recency score cannot pick the right one of them.

Seeded rescue examples (default_rng(0)): see `metrics.json` `examples` (inst 824409, 922371, 1533446, 1851427,
2383540). Caveat from 2383540: a listing that names both the current model and a newer one ("iPhone 14 Plus/13 Pro
Max") counts as an upgrade candidate. This is a shared-fit listing, not an upgrade.

Files: `config.yaml` (frozen 02:07:55 +08, with module and parser sha256), `metrics.json`, `bootstrap.csv`,
`val_event_ranks.npz`, `git_hash.txt`. Logs: `results/logs/20261106/device-upgrade-*.log`. Tests:
`tests/test_device_upgrade_retrieval.py` (2 pass).
