# Overnight research loop — 6 Nov 2026 (Val only, exploratory)

Instructions: scratchpad `overnight_loop_prompt_2026-11-06.md` (copied into the results file at the end). Authorisation: `docs/decisions.md` 2026-11-06. Limits: no new training after 09:30, no new task after 10:15, results by 11:00, stop by 11:30.

## Status board

| Track | Task | Subagent | State |
|---|---|---|---|
| A0 | Pools build + retriever oracle | pools-build | launched 01:55 |
| A1 | bce_text on P2/P3 | pools-fit | waiting on A0 |
| B | Refit-draw robustness | refit-draws | queued |
| C | Random mixing | random-mix | queued |
| D | Context features | context-arm | code + tests + sample smoke launched 01:55 (no full prep until GPU free) |
| E | Graph freshness on Val | freshness-val | launched 01:55 |
| F | Double-miss case study | switch-cases | launched 01:55 |
| G | Optional DIN / two-tower retrain | — | decide after A, B |

## Log

### 01:50 — Registration
- Pre-registration context: the Val retriever overlap was measured at 01:45 (recency 6.4%, two-tower 7.2%, union 12.4%, top-100 union 8.7%, micro). The fusion rule (round-robin interleave, RRF k = 60) was fixed without tuning.
- Timing reference (5 Nov logs): prepare ≈ 25 min, fit ≈ 8 min, evaluate ≈ 16 min.

### 01:55 — Launched A0 pools-build (GPU), E freshness-val (CPU), F switch-cases (CPU), D context-arm (code/smoke only)

### 02:00 — F switch-cases (P-e) — ACCEPTED, interesting
- **Question:** why did users review products that neither retriever found?
- **Hypothesis (README 01:48, config 01:50:33, before the draw at 01:50:51):** most double misses are leaf-category jumps from the last product.
- **Run:** `results/analysis/20261106-switch-cases`; `src/cascade/analysis/switch_cases.py`. Seeded draw of 8 per bucket.
- **Result:**
  - Double misses: 204,111 of 232,904 Val events (87.6%). Eligible: F1 new 69,185; F2 warm with ≥ 50 reviews 47,470; F3 warm with 1–49 reviews 87,456.
  - Leaf-category change from the last product: 0.839 among double misses vs 0.550 among events retrieved by at least one retriever.
  - Store change: 0.986 vs 0.817.
  - Target leaf never in the history: 0.564 vs 0.318.
  - Top-level category is uninformative: every item is "Cell Phones & Accessories".
  - In the cards, the dominant pattern is **device change**: the target suits a newer or different device than the one the history names, while the retrievers return accessories for the old device. Examples: Pixel XL case → Pixel 5a case (1756859); iPhone XR → iPhone 13 mini (114423). Other patterns: Apple Watch bands after phone accessories (4/24), and generic accessories after long gaps.
- **Checks:**
  - inst/user alignment passes for both retrievers;
  - the Test-name guard is present, and later-review context stops at T2;
  - 4 tests pass.
- **Verdict:** interesting; the hypothesis passes. Caveats: double misses are the normal case (the leaf-change share over all events is 0.803), and part of the gap is built in because both retrievers rank by text similarity.
- **Decision:** a follow-up to the same subagent, to quantify "device change" at scale with a frozen regex dictionary of model names. Hypothesis: a majority of device-naming double-miss targets name a model the history never names, and more often than among retrieved events; the retrievers' top 10 mostly echo history models. **Why:** if it holds, it gives a concrete, scale-backed answer to "why did users switch?" (they changed devices) and a design implication (retrieval echoes old devices). That answers P-e and P-f without a new model.

### 02:10 — D context-arm, phase 1 (code, tests, sample smoke) — ACCEPTED; full run launched
- **Code:** `src/cascade/pipelines/context_ranking.py`. It adds 63 one-hot context columns to the common block (common_dim 9 → 72): price decile, review count, mean rating and item age at the period start, history length, and time since the last interaction. Shared parameters start from the exact `bce_text` initial state; the hash is checked.
- **Leak tests (full data):**
  - Val: 656 candidates have their first review at the query time, and all are targets. B: 900, all targets.
  - Both groups get the same codes as other new candidates, so the model cannot tell them apart.
  - Statistics use only rows before the period start.
  - Price, which is crawl-time metadata, differs mildly for those groups (total variation 0.088, mainly whether it is missing); flagged as a weak signal.
- **Smoke:** the sample run selected epoch 0, so the nonzero-epoch refit path was not exercised there. 9 tests pass; full suite 324 passed.
- **Decision:** go for the full run (~35 min on mps), plus an **evaluation-only stale-context arm**: the same checkpoints scored with candidate statistics at T0. **Why:** the context arm's Val statistics are at T1 while the control's are at T0, so any gain could just be freshness, which Track E measures separately. Pre-stated: if the gain disappears under T0 statistics, it is a freshness effect, not a context effect.
- **GPU order:** context-arm now. A1 pool fits go after it, or after A0 if A0 finishes first.

### 02:20 — A0 pools-build — ACCEPTED, interesting
- **Run:** `results/analysis/20261106-pools`; `src/cascade/analysis/pools_build.py`; 5 tests.
- **Checks, all pass:**
  - In the saved two-tower B lists, columns 0–199 are natural, with the forced target at column 200 in all 276,321 forced rows.
  - Float32 regeneration from the checkpoint is bit-identical on B; on Val, only exact-tie reorderings differ.
  - Filter checks pass, and inst/user align for both splits.
  - P1 reproduces 0.006284 / 0.054746.
- **Val retrieval-order results** (macro; paired CIs vs P1):

  | Pool | R@200 | nDCG@10 | new nDCG@10 | warm nDCG@10 |
  |---|---|---|---|---|
  | P1 recency | 0.0547 | 0.006284 | 0.007611 | 0.005606 |
  | P2 two-tower | 0.0687 | 0.005642 | 0.000361 | 0.007983 |
  | P3 fused, RRF order | 0.0795 (+0.0248 [0.0237, 0.0258]) | 0.006555 (+0.00027 [0.00001, 0.00055]) | | |
  | P3 fused, interleave order | | **0.007408 (+0.00112 [0.00093, 0.00132])**; new −0.0021, warm +0.0025 | | |
  | Retriever oracle | candidate stage 0.1124 | +0.00542 [0.00515, 0.00569], almost all warm (new +0.00028) | | |

- **B natural positives:** P1 20,676; P2 38,793; P3 38,861. The pools carry about twice as many positives, skewed warm, which confounds any ranker comparison across pools.
- **Verdict:** the hypothesis holds: P3 micro R@200 is 0.0878, new recall is lower and warm is higher. **Fusion with no ranker gains about 60% as much as the BCE ranker does over recency (+0.0018).**
- **Problem found:** the fixed RRF score creates systematic exact ties (single-source items at the same source rank). Under pessimistic ties this lowers P3's own order below its interleave order.
- **Decision:** keep P3 with RRF as registered and primary. Add **P3t** = RRF − 1e-6 × interleave position, a mechanical tie-break disclosed as post-hoc, and run P3t first. Then P2. Launched pools-fit (9 fits), which waits for the GPU behind context-arm.
- **Planned next step:** if the P3 ranker wins, run a matched-positive-count control: P3 with positives subsampled to 20,676. **Why:** it separates "better candidates" from "twice the training positives".

### 02:35 — F follow-up: device change at scale — ACCEPTED, interesting
- **Run:** `results/analysis/20261106-switch-cases-devices`. The regex dictionary and module hash were frozen at 02:00:45, before any bulk computation. Parser coverage: 71,174 of 111,480 titles. 4 tests pass.
- **Results:**
  - Among double misses whose target names a device model (150,942), **62.2% name a model that no history title names**, against 34.1% among retrieved events.
  - The pattern holds with universal multi-model listings removed (0.687 vs 0.398), and when compared against the last product only (0.734 vs 0.465).
  - Within the same family, the target model is newer in **74%** of double misses (newer+older denominator), against 65% for retrieved events.
  - Both retrievers fill about half their top-10 slots with models already in the history, in both populations. In double misses only ~16% of slots name the target's model, against 29–34% when retrieval succeeds.
  - A target whose model is absent from the history reaches either top 200 7.7% of the time; with the model in the history, 20.9%.
- **Verdict:** interesting; the hypothesis holds. Device change, mostly upgrades to newer models, is the largest single reason for double misses. It is not the only one: 28% name a model the user already owns, and 26% name no recognised model.
- **Caveats:**
  - These are title mentions, not compatibility; the device-audit gate is still blocked.
  - Crawl-time titles can name later models.
  - Parser precision is unmeasured; a 60-title audit sample is blank, awaiting human review.
- **Decision:** a no-training test of whether retrieval can act on this. The "device-upgrade" pool D = recency top 180 + the top-20 eligible items naming a strictly newer same-family model (ranked by the recency score); users with no recognised device are unchanged. Hypothesis: Recall@200 rises clearly for targets naming a newer model, with a small positive overall change. **Why:** this follows the professor's suggestion that the case study could lead to a model that helps in specific cases (P-e, P-f), cheaply and on CPU.

### 02:45 — E freshness-val (P-b) — ACCEPTED, highly interesting
- **Run:** `results/analysis/20261106-freshness-val`; `src/cascade/analysis/freshness_val.py`.
- **Checks:**
  - T0 reproduction is exact on CPU. Across all 12 checkpoints, 7 of 2.79M ranks shift by one place against the saved MPS ranks, with no @10 change.
  - Snapshot cut checks pass: 2,208,880 edges at T1, 12 monthly snapshots, query user excluded. Tests pass.
- **T1 − T0, nDCG@10, all positive** (frozen models, Val):

  | Method | Overall | New | Warm |
  |---|---|---|---|
  | bce_text | +0.00129 [0.00112, 0.00146] | +0.00306 | +0.00035 |
  | pairwise_text | +0.00190 | +0.00439 | |
  | pairwise_coreview | +0.00249 | +0.00579 | |
  | bce_coreview | +0.00163 | +0.00288 | |

  Overall at T1: bce_text 0.009432, pairwise_coreview **0.010221**.
- **Monthly − T1:** worse overall for every method (−0.00016 to −0.00096). New products lose heavily (−0.0038 to −0.0061); warm products gain (+0.0013 to +0.0020).
- **Co-review − text grows with freshness:**

  | Objective | T0 | T1 | Monthly |
  |---|---|---|---|
  | BCE | +0.000026 (n.s.) | +0.000369 [0.000229, 0.000520] | +0.001164 |
  | Pairwise | −0.000052 (n.s.) | +0.000537 [0.000384, 0.000702] | +0.001070 |

  This fits the team's Test result: the "T1" graph is about 11 months stale for Test, as T0 is for Val, and the co-review increment was inconclusive there too. **The co-review features help when the graph is fresh relative to the query period; a stale graph hides them.**
- **Support:** the share of targets that are known is 0.347 at T0, 0.670 at T1 and 0.934 monthly (new: 0 / 0 / 0.80).
- **Verdict:** interesting. The T1 part holds, with gains larger than predicted. The monthly part is rejected.
- **Decision:** a mechanism follow-up with the same subagent, using hybrids. H1 = monthly with candidate_graph_known from T1; H2 = also degree from T1; H3 = T1 with knownness from monthly. Plus the B-training AUC of "unknown" and a logit probe flipping knownness. Hypothesis: knownness acts as a learned newness signal, so H1 recovers T1's new-item quality while keeping monthly's warm gain. **Why:** it explains why monthly refresh hurts, and it connects to the context arm, whose "new at period start" bucket encodes the same signal explicitly.

### 02:30 — F2 device-upgrade pool D (P-e/P-f) — ACCEPTED, small positive; leak concern
- **Run:** `results/analysis/20261106-device-upgrade`. Rules frozen at 02:07:55; a 512-row technical smoke ran first (only its checks were read, disclosed). Filter checks pass on all 199,507 changed lists.
- **Coverage:** 85.7% of events get upgrade candidates, but the candidate set is broad (median 1,107 items).
- **Recall@200, D − P1:**
  - overall **+0.001402 [0.001109, 0.001742]**: 751 targets gained, 474 lost;
  - new +0.00367;
  - warm +0.00026 (n.s.);
  - targets naming a newer same-family model (48,364 events): **+0.01387 [0.01278, 0.01502]**, a 24% relative gain.
- Append order leaves the top 10 unchanged by construction. Interleave order (descriptive): overall nDCG@10 −0.00007, newer-model subgroup +0.00023.
- **Verdict:** the hypothesis holds; the effect is real but small. The recency text score cannot single out the right newer-model item.
- **Leak concern:** crawl-time titles can name models released after the query (2383540 names the "iPhone 14 Plus"). This touches every text model.
- **Decision:** a last follow-up, after which this path closes:
  - audit how often top-10 slots and targets name a model that is still in the future at the query time, using a data-derived release proxy (frozen rule);
  - run D_strict, restricted to models already released.

  Hypothesis: more than 5% of top-10 slots name future models, and D_strict keeps most of the gain. **Why:** if titles leak future model names, that is a project-wide caveat for the text retriever and needs reporting.

### 02:45 — E2 freshness mechanism — ACCEPTED, interesting (mechanism identified)
- **Run:** `results/analysis/20261106-freshness-mechanism`. The hypothesis was written before computing; checks pass; 7 tests.
- **Training data (T0 snapshot):**
  - 40.5% of B positives are "unknown" (no reviews before the snapshot), against 17.7% of negatives.
  - As a score, "unknown" has AUC 0.614. **"Unknown" was a learned positive cue.**
  - On Val, its AUC is 0.635 at T0, 0.626 at T1 and 0.540 monthly.
- **Probe:** flipping only candidate_graph_known 0→1 lowers the logit by −1.30 (bce_text; within-list SD 1.39), −1.17 (bce_coreview), −0.39 (pairwise_text) and −0.51 (pairwise_coreview). BCE relies on the cue about 3× more than pairwise.
- **Hybrids:**
  - **H3** (T1 with knownness from monthly) loses −0.0010 overall vs T1, almost all on new products (−0.0047).
  - **H1** (monthly with knownness from T1) beats monthly by +0.0008 to +0.0018, but its warm quality does not keep monthly's gain.
  - **H2** (also degree from T1): text models return to about T1; co-review models keep +0.0003 to +0.0008 over T1, so fresher co-review paths carry real information.
  - Co-review − text: +0.0009 to +0.0013 under H1/H2.
- **Verdict:** partly supported. The cue exists and dominates new-item quality. Knownness also helps warm targets by demoting newly known competitors. The H1/H2 hybrids are off-distribution, which is a caveat.
- **Interpretation (lead):**
  - The train/serve-consistent setting is "features at the start of the query's period": B was trained at T0, and Val should be scored at T1.
  - The published pipeline scored Val with year-old T0 features, a stale-serving mismatch.
  - Consistent T1 serving gives bce_text 0.009432 vs 0.008141 (+16%), and pairwise_coreview 0.010221 vs 0.007735 (+32%).
- **Decision:** a rolling-snapshot TRAINING study (same subagent): monthly snapshots for B queries in training, and monthly on Val; bce_text and bce_coreview, seeds 0/1/2; same initial states and draws. Hypothesis: rolling train+serve beats T0-trained models served at T1, and co-review − text is clearly positive. **Why:** it tests whether consistently fresh features beat period-start freshness, which is the strongest lead on the professor's context question (P-b). Features are built on CPU now; the GPU runs after the pool fits.
- **Note for Track B:** the refit-draw models will also be scored at T1, so co-review's robustness is judged in the consistent setting.

### 02:50 — F3 title-leak audit — ACCEPTED; path F CLOSED (dataset limitation)
- **Run:** `results/analysis/20261106-title-leak`. Frozen at 02:17:26; 12 tests pass.
- **The frozen data-derived release proxy is invalid.** It dates the iPhone 13/14, Galaxy S22 and Pixel 6 years too early, because parent listings group older variants whose reviews predate the crawl-time title. The frozen test therefore says nothing either way:
  - 0.89% of slots name a future model;
  - D_strict ≈ D (+0.001398 vs +0.001402).
- **Post-hoc check, labelled as such** (external release dates for 4 models; a lower bound; v2 corrects a counting slip in v1):
  - **8.5% of recency top-10 slots, 20.9% of two-tower top-10 slots and 13.4% of Val targets** name one of these models before its release.
  - 96% of items naming the iPhone 14 have first reviews more than 90 days before its release.
  - 173 of D's 751 rescues (23%) rely only on such names.
- **Verdict:** crawl-time titles and parent-ASIN grouping are a leakage channel that the first_ts release filter cannot stop. Its direction and size are unmeasurable without period-correct titles: future names are about as common among hits (12.6%) as among all targets (13.4%).
- **Decision:** close the device/case path. Report this as a dataset limitation next to the 5-core caveat. It affects every text model, new-item claims and Test. The device-upgrade gain is partly built on leaked names, so it is illustrative only.
- **Launched B refit-draws:** code now, GPU after the pool fits. Each fit is evaluated with both T0 and T1 features. Hypothesis: the pooled co-review gain is n.s. at T0 and positive at T1; draw SD ≈ 0.0002.

### 02:40 — A1 P3t gate FAILED — lead error; P3t dropped
- **Run:** `results/analysis/20261106-pool-ranking/interim_P3t.md`, `p3t_gate.json`.
- My P3t definition was inconsistent, for two reasons:
  - RRF order differs from interleave order even without ties: items found by both sources move up.
  - The 1e-6 × position offset also reorders distinct RRF values.
- Epoch-0 Val nDCG@10: RRF pessimistic 0.006555; P3t 0.007044; lexicographic tie-break 0.007041; interleave order 0.007408. The subagent correctly stopped instead of improvising.
- **Decision:** drop P3t to save GPU time; P3 (registered RRF) and P2 continue. The lexicographic variant runs only if GPU time remains at the end.
- **Lesson for the write-up:** with a residual ranker, the skip score defines the starting order. RRF's exact ties and its "both sources" boost make P3's starting order (0.006555) worse than plain interleaving (0.007408).

### 13:35 — Resumed after a rate-limit pause (≈03:00–13:30)
- The subagents stopped at about 03:00. Their background chains kept running and finished: pools P3/P2 fitted and evaluated by 04:04; refit-draws seed-0 draws 1–3 fitted by 05:01; the context report done at 03:15. The rolling-ranking prepare had not finished.
- Brandon asked to continue past the original 11:30 limit ("continue working on it until you use up the limits again"). That supersedes the time limits; all other rules stand.
- **D context-arm** (`results/analysis/20261106-context/report/report.md`), Val nDCG@10:

  | Arm | Overall | New | Warm |
  |---|---|---|---|
  | bce_context | **0.010870** | **0.018075** | 0.007332 |
  | bce_text | 0.008141 | 0.008666 | 0.007887 |

  - Context − text: +0.002730 [0.002478, 0.002988]; new +0.009410; warm −0.000555.
  - The T0-statistics arm still gains +0.000952 [0.000780, 0.001136]: about 35% of the gain survives stale statistics, so it is not only freshness.
  - **Decision:** accept provisionally, but the new-item gain is very large, so require three no-training follow-ups first: attribution by group, a leak re-check on the driving group, and fully period-start-consistent scoring (common/graph features at T1 as well).
- **A1 pools** (seed-level Val nDCG@10 from the logs): P3 ranker 0.01085 / 0.01111 / 0.01109; P2 ranker 0.00995 / 0.00999 / 0.01030. P1 ranker (rebuilt bce_text) is 0.008141.
  - **Decision:** the subagent finishes the checks and contrasts, plus a decomposition of "better candidates" vs "more training positives".
- **B refit-draws:** evaluate the seed-0 draws under T0 and T1 and report before seeds 1–2.
- **Rolling:** resume the prepare.

### 14:00 — B refit-draws, seed-0 interim — ACCEPTED, interesting
- **Run:** `results/analysis/20261106-refit-draws/interim_seed0.md`.
- **Checks:**
  - draw 0 regenerates all saved plans exactly, and the initial states match;
  - CPU scoring reproduces the freshness study's T0/T1 ranks;
  - 4 tests pass.
- **Co-review − text, seed 0, pooled over draws 1–3:**
  - **T0: +0.000196 [0.000072, 0.000325]**; T1: **+0.000387 [0.000244, 0.000536]**.
  - Warm: +0.00038 / +0.00072; new: −0.00018 / −0.00041.
  - All three T1 draws have CIs above zero.
- **Draw SD:** about 0.0001 overall, half the predicted 0.0002; 0.0004–0.0007 on new targets.
- **Reading:** the rebuild's three-seed loss of the gain comes mostly from seeds 1–2 (seed 1 stopped early at epoch 5), not from draw noise on seed 0.
- **Decision:** fit seeds 1–2 × draws 1–3 (12 fits, ≈ 1.8 h GPU), then the final 3-seed analysis. Pre-stated: T1 gain CI above zero; T0 smaller, possibly including zero. **Why:** this settles how the poster states co-review.

### 14:05 — Duplicate refit chain stopped
- I launched the seeds 1–2 chain at 14:02, thinking the subagent had stopped; the subagent then launched its own. I killed mine (pid 52403) while it was still waiting for the GPU, so no fit had started. The subagent's chain (pid 55199) continues and runs the final analysis.

### 14:15 — D context follow-ups — ACCEPTED; path D CLOSED
- **Run:** `results/analysis/20261106-context/followup/`. The hypothesis was written first; 10 tests pass. CPU rescoring matches the MPS ranks (one non-top-10 event differs).
- **Attribution:** neutralising review_count + rating + item_age together drops −0.00413 overall (new −0.0133), leaving 0.006738, below bce_text. Individually: item_age −0.00179, review_count −0.00166, rating −0.00068, price −0.00059; time gap and history length are negligible. **Driver: a "new at period start" prior.**
- **Leak re-check: passes.**
  - Every new candidate gets the same period-column row, whatever its gap to the query.
  - The target rate falls with that gap (656/656 at the same timestamp; 0.23% at ≤ 1 day down to 0.08% at > 90 days), and no input can see the gradient.
  - On the 68,518 new events with gap > 0, the gain is +0.0092.
- **Fully period-start-consistent scoring** (common and context both at T1): vs bce_text(T1), **+0.001273 [0.001052, 0.001501]** (new +0.0023, warm +0.0006). Of the +0.0027 headline over bce_text(T0), about half is the control's own staleness.
- **Caveat:** the 5-core filter is computed over the whole timeline, so every new product is guaranteed later reviews. That likely inflates any newness prior; this is the known §4.4 limitation.
- **Verdict:** interesting. Leak-safe context adds a modest, real gain once the comparison is fair.
- **Next:** launched C random-mix (P-a). CPU diagnostics now; it waits behind refit/rolling for the GPU and may fit on CPU if that takes under 25 minutes per fit.

### 14:25 — A1 pool-ranking — ACCEPTED, the main result of the night
- **Run:** `results/analysis/20261106-pool-ranking` (tables.md, README.md).
- **Checks:**
  - archives match P1 events, users, targets and histories;
  - the feature builder reproduces P1 exactly;
  - initial states match;
  - epoch 0 equals each pool's own retrieval order;
  - 347 tests pass.
- **Ranker levels (nDCG@10):** P1 0.008141; **P3 0.011019**; P2 0.010081. Selected epochs: P3 7/8/9, P2 5/10/9.
- **Contrasts:**
  - **P3 − P1: +0.002878 [0.002544, 0.003211]** (warm +0.0048, new −0.0017).
  - P2 − P1: +0.001940 (new −0.0050).
  - P3 − P2: +0.000938, all on new targets.
  - **Each ranker's lift over its own pool order:** P1 +0.0019, P2 +0.0044, P3 +0.0045 (vs RRF order) / +0.0036 (vs interleave order).
- **Decomposition (no training):** P1 checkpoints scored on P3 lists with their own cosine skip (bit-exact on overlapping slots).
  - Better candidates alone: +0.000670.
  - Training on P3 lists: +0.002208. This also changes the skip and the scaler.
  - **The retriever shapes what the ranker learns:** most of the gain needs ranker training on the new pool.
- **Same-candidate subset** (targets in both P1 and P3): the P3 ranker loses −0.0426 to the P1 ranker. The gain comes entirely from the 9,394 targets only P3 retrieves; longer mixed lists add strong distractors. This subset is selected on retrieval outcomes, which is a caveat.
- **Surprise:** P2 retrieval is worse than P1 (−0.0006), yet the P2 ranker beats the P1 ranker (+0.0019). Retrieval-order quality does not predict ranker quality.
- **Decision:** an additivity test.
  - Arm 1, no training: P3 checkpoints scored at T1.
  - Arm 2: P3 + context, 3 fits, evaluated at T0 and T1.

  Pre-stated: gains are mostly additive, and context recovers part of P3's new-item loss. Labelled additivity, not selection.

### 14:35 — C random-mix diagnostics (before training) — H2 supported at the input level
- **Run:** `results/analysis/20261106-random-mix/diagnostics_auc.csv`.
- **AUC of each input against the list's negatives:**

  | Input | Natural positives | Inserted positives |
  |---|---|---|
  | Skip score | 0.684 | 0.501 (randomisation works on the score) |
  | Text cosine to history | 0.587 | **0.103** |
  | Score − cosine | 0.611 | **0.889** |
  | Leaf match | | 0.295 |
  | Store match | | 0.411 |

- **Reading:** a missed target is by construction textually dissimilar to the history. Giving it a random rank's score therefore creates a visible score/content mismatch. Random mixing cannot hide contamination when other inputs encode retrieval similarity.
- Fits M2 (natural lists + 20,676 mixed lists) are running; M1 follows.
- **Rolling:** all 6 fits are done (by about 14:30), and the analysis phase is running.

### 14:45 — E3 rolling-snapshot training — ACCEPTED; path E CLOSED
- **Run:** `results/analysis/20261106-rolling-ranking`. 14 B snapshots, with cut checks passing; initial states and plans match; the analysis reruns byte-identically.
- **bce_coreview:**
  - rolling monthly **0.010613**, vs existing at T1 **+0.000811 [0.000614, 0.001002]** (new −0.0005, warm +0.0015), vs existing at T0 +0.002445;
  - co-review − text under rolling: **+0.001308 [0.001129, 0.001483]**.
- **bce_text:** rolling monthly vs existing T1 −0.000128 (n.s.): new −0.0030, warm +0.0013.
- **Mechanism:** "unknown" = released this month (target rate ratio 6.1×, against 3.1× at T0).
- **Verdict:** partly supported. Supported for co-review; not for text.
- **Decision:** close the freshness path. Answer to P-b: relational context (co-review) helps substantially when it is fresh in both training and serving; stale graphs hide it. Results file updated.
- **One subagent edit:** the rolling subagent fixed a missing-directory bug in its own new `rolling_ranking.py` (not an existing module).

### 15:10 — C random-mix — ACCEPTED; path C CLOSED
- **Run:** `results/analysis/20261106-random-mix`. Checks pass and 6 tests pass. M2 was fitted (3 seeds); M1 was not run, at an estimated 25–70 min per fit.
- **Val:** M2 − bce_text **−0.001080 [−0.001278, −0.000881]** (new −0.0014, warm −0.0009). Every metric's CI is below zero. On new targets M2 is not above retrieval.
- **Diagnostics:** inserted-positive cosine AUC 0.103, score − cosine 0.889. **H2 holds and H1 is rejected.**
- **Verdict:** a clean answer to P-a. Random insertion turns the position/score shortcut into a content-similarity shortcut.
- **Coordination issue reported:** `refit_draws_chain.sh` `busy()` misses running fits (the process name is capital-P "Python"), so one refit-draws fit overlapped with M2's evaluate on mps. Correctness is not affected (separate processes and outputs), only speed and MPS contention. No action needed: the remaining jobs are the refit chain and pool-combo, and pool-combo checks for itself.

### 15:55 — Pool-combo arm 1 (no training), from the chain log
- P3 checkpoints scored at T1 give Val nDCG@10 0.011701 / 0.011412 / 0.011240 per seed (mean ≈ 0.01145), against 0.01102 at T0: about +0.0004. Freshness on P1 bce_text was +0.0013, so the effect looks **sub-additive**; the formal CIs are pending the subagent's report.
- CPU vs saved MPS: 1 rank mismatch, no top-10 flips.
- Arm 2 (P3 + context): prepared at 14:24; its fits wait behind the refit chain (seed 2, about 6 fits left).

### 20:00 — Resumed after a second rate-limit pause (≈16:40–19:58). B final and pool-combo — ACCEPTED
- Both chains finished at 16:48 and 16:55, before the subagents were cut off.
- **B final** (3 seeds × draws 0–3), co-review − text:
  - **T0: +0.000157 [0.000065, 0.000255]**;
  - **T1: +0.000492 [0.000375, 0.000609]** (warm +0.0008, new −0.0002);
  - draw SD 0.00011–0.00017.
  - The hypothesis holds (T1 positive; T0 smaller). Path B CLOSED.
- **Pool-combo:**
  - P3 + context, consistent T1: **0.012167**, +0.004026 [0.003658, 0.004415] over P1 bce_text(T0); the sum of single gains is about +0.0055, so the gains are **sub-additive**.
  - Freshness on P3: +0.000432 only.
  - Context on P3: +0.000716 (new +0.0041).
  - The mixed T0-common/T1-context arm scores 0.012697; it is a staleness artefact, not recommended.
  - Path A CLOSED.
- Results file updated.
- **Decision on Track G** (both gates met: P3 clearly beats P1, and freshness is confirmed on Val): run **G2 first**. G2 retrains Two-tower (text) seed 0 on all pre-T1 data and uses it only to serve fresher Val candidates (P2'/P3') to the existing frozen P3 rankers; train/serve differences are disclosed.
  - **Why G2 before G1:** freshness is the night's dominant theme. A sample DIN-style attention study (E3) was null, and with the pool's distractors on same-candidate events a richer ranker has unclear upside.
  - G1 runs only if usage allows after G2.

### 20:45 — G2 interim, from the training log
- The fresh Two-tower (text) seed 0 is trained on all pre-T1 data with the registered recipe, at about 75 s per epoch.
- Val-subset (fixed 20K) Recall@200: **0.1129 at epoch 26**, against the frozen pre-T0 model's best of about **0.0722** (epochs 28–30 in its train_log.json). That is about +57%. Val-subset R@100 is 0.0757 vs 0.0475.
- **Caveat:** the subset is the selection set for both runs, so the full-Val pool numbers from the subagent are the real comparison.
- Training continues (patience 5).

### 21:16 — G2: fresh retriever, full-Val evaluation (from the training log)
- Best epoch 39 (training time 63 min).
- Full Val, macro: R@50 0.0511, R@100 0.0765, **R@200 0.1113**, R@500 0.1797, **nDCG@10 0.0100**. The frozen pre-T0 two-tower had R@200 0.0687 and nDCG@10 0.0056.
- **A fresh retriever alone beats the current pipeline's ranker (bce_text 0.008141).**
- Caveats: the registered recipe early-stops on a Val subset, as the original did; new/warm splits are pending.
- Pools and features are being built; the first features attempt failed and the second is running, so the reason needs checking in the report.

### 22:00 — G2 fresh two-tower — ACCEPTED; path G2 CLOSED
- **Run:** `results/analysis/20261106-fresh-two-tower`. All reproduction checks are exact.
- **Retriever, full-Val R@200:** 0.0687 → **0.1113** (+62%); the [T0, T1) cohort gains ×2.4; new-in-Val only +21%.
- **Retrieval order:** P2' nDCG@10 0.010035, +0.0044 over P2.
- **Frozen rankers:** P3' − P3 **+0.000365 [0.000160, 0.000569]** (new −0.0008, warm +0.0009); P2' − P2 −0.000149 (n.s.).
- **Verdict:** partly supported. Retriever freshness is the largest pool gain, but stale-list-trained rankers pass on only about 15% of it.
- **Subagent housekeeping:** a 1.7 GB failed partial feature folder (`results/local_experiments/20261106-fresh-two-tower/features_failed_attempt1_P3f/`) was kept, not deleted. Deleting it is Brandon's call.
- **Decision on G1 (DIN-style ranker): skip.** The sample attention study was null, and the ranker-side differences are small next to retrieval and freshness effects.
- **Instead:** launched P-f meso-segment tables (no training): winners by history length, new/warm, warm-popularity terciles, category and activity gap, across P1, P3, P3+context(T1), rolling co-review and the fresh P2' retrieval. **Why:** it is the one professor item only partly answered.

### 22:20 — P-f meso-segments — ACCEPTED. Loop STOPPED: all tracks complete
- **Run:** `results/analysis/20261106-meso-segments`. All 6 systems reproduce to 1e-9; 5 tests pass.
- **Winners by segment:**
  - popularity separates the systems;
  - fresh P2' wins the most-reviewed warm third (0.0453);
  - rolling co-review wins the middle third and Cases/Holsters;
  - P1 bce_text ties best on the least-reviewed third;
  - P3+context T1 wins overall and every history bucket.
- Full suite: 359 passed.
- `docs/evaluations.md` section written; results file marked complete.
- **Closed or not run:** G1 (DIN-style ranker), skipped by decision; M1 (all mixed lists), too slow; P3t, dropped after its gate failed.
- **Left for Brandon:**
  - review the results file;
  - decide on deleting the 1.7 GB failed partial features folder;
  - commit when ready (nothing committed).

### 7 Nov 00:56 — Phase H authorised (loop continues until 08:00)
- Brandon asked to continue until 08:00, waiting through credit limits.
- **H1:** P3 + rolling co-review (6 fits incl. text control in H3).
- **H2:** fresh two-tower seeds 1–2 (robustness of +62% R@200).
- **Why:** H1 combines the two largest consistent effects (the fused pool, and fresh co-review). H2 checks the single-seed retriever result before the team leans on it.
- Instructions: scratchpad `loop_phase_H_2026-11-07.md`.

### 01:05 — Resource decision
- macOS swap reached about 13 GB with the H2 retriever training and the H1 prepare running together. Free disk fell 27 → 22 GB, against a 20 GB floor.
- **Decision:** serialise. The H1 prepare is stopped and re-armed as a chain that waits for the H2 chain to finish (about 03:00), then checks disk ≥ 24 GB and memory ≥ 5 GB. Nothing is lost: the bulk folder was empty.

### 01:20 — H1 chain re-armed (pid 92069). The pre-build disk gate is now 22 GB: the estimate is 0.96 GB, since features are stored only for B hit queries. Other gates are unchanged. It waits for the H2 chain.

### 01:45 — Status
- H2 seed 1 is at epoch 29: val-subset R@200 0.1116, tracking seed 0 (0.1129 at epoch 26).
- Free disk is 21 GB, with swap at 14.6 GB (old pages). The H1 gate is 22 GB, with a 20-minute retry once H2 ends.
- If the gate fails around 03:00, I re-decide then; prepare's own 20.5 GB refusal protects the floor.

### 02:20 — H2 seed 1 done; seed 2 stopped; H1 margin lowered
- **H2 seed 1:** best epoch 35. Full-Val **R@200 0.1108**, R@100 0.0753, nDCG@10 0.0100. Seed 0 had 0.1113 / 0.0765 / 0.0100, so it **replicates**.
- **Seed 2** stopped by the lead at epoch 8. Free disk was at the 20 GB floor (swap 15–16 GB held by other apps); the partial run folder is 1.1 MB and kept. H2 reports two fresh seeds.
- **Decision on H1:** the 20 GB floor was the lead's own margin. With H1 writing about 1.5 GB in total, it is lowered for H1 only to a pre-build gate of 19.5 GB and a hard floor of 18.5 GB. Free disk would still stay at least 18.5 GB.
- Freeing space would need deleting files (e.g. the 1.7 GB failed-features folder), which waits for Brandon.

### 02:30 — H2 closed
- Frozen seeds 0/1/2 R@200: 0.0687 / 0.0697 / 0.0675 (SD ≈ 0.0011). Fresh seeds 0/1: 0.1113 / 0.1108.
- The +62% gain is robust, about 40× the seed spread. Results file §7 updated.
- H1 prepare has been running since 02:16 (chain pid 51445).

### 02:35 — H2 subagent comparison accepted
- `results/analysis/20261107-fresh-tt-seeds`: R@200 +0.042460 [0.041310, 0.043569]; nDCG@10 +0.004321 [0.004039, 0.004597]. Mostly warm; new-in-Val +20% R@200.
- The partial seed-2 run folder is kept; deleting it is Brandon's call. Disk is back to 24 GB.

### 11:16 — H1/H3 accepted; loop STOPPED (past 08:00)
- The H1/H3 chain finished at 03:23. The lead was not woken between about 03:42 and 11:16 (session idle or rate limit). No further work was launched, per the 08:00 stop.
- **P3 rolling bce_coreview monthly: 0.013084**, against P3 bce_text (T1) +0.001633 [0.001391, 0.001870]. Co-review − text within P3 rolling +0.001539; interaction −0.000407. The hypothesis holds.
- Docs updated: results §9, `evaluations.md`.
- **Not done:** a P3 + rolling co-review + context combination; fresh-retriever seed 2.

## 7 Nov (after the loop): size-matched P3 control (user-requested)
- Why: the P3 ranker trained on 1.9x more natural B positives than P1 (38,861 vs 20,676), confounding "pool distribution" with "training-set size".
- Setup: `src/cascade/pipelines/pool_size_control.py` reuses the P3 study (candidates, features, scaler, split; `pool_ranking.fit/evaluate` unchanged, verified by function identity) with a fixed subset (seed 20261107) at P1's prefix/tune natural counts. Seeds 0/1/2, Val only.
- Result: size-matched 0.010387; vs P1 ranker +0.002246 [0.001901, 0.002602]; full P3 − size-matched +0.000632. About 78% of the P3 gain survives at equal training size. All seeds hit the 10-epoch cap.
- Outputs: `results/analysis/20261107-pool-size-control/`; bulk `results/local_experiments/20261107-pool-size-control-P3/`; log `results/logs/20261107/pool-size-control.log`. Tests: `tests/test_pool_size_control.py`.

## 7 Nov (evening): statistics-timing diagnostics (user-requested, Val only)
- Column swap T0 -> T1 and age curve (fixed cut date; fixed lag 0–12 months) on the frozen B1/B4/B5/B6 checkpoints, natural-hit Val rows, no training.
- Hypotheses were written at 12:25 UTC, before any arm was built. All five held (see `results/analysis/20261107-statistics-timing/README.md`).
- Code: `src/cascade/analysis/statistics_timing.py` (tests: `tests/test_statistics_timing.py`). Bulk: `results/local_experiments/20261107-statistics-timing/`. Log: `results/logs/20261107/statistics-timing.log`.
