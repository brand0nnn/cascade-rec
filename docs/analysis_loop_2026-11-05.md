# Overnight rebuild and analysis loop — 5 Nov 2026 (Val only, exploratory)

Instructions: the loop prompt (Track R rebuild, Track N current-pipeline analyses, Track L laptop-local evidence). Hard stop: no new task after 09:30, results file `docs/overnight_results_2026-11-05.md` by 10:45, loop stops by 11:00. Authorisation recorded in `docs/decisions.md` (2026-11-05).

Start: 04:40, git `d890988` + `.gitignore` edit (uncommitted), 287 tests pass.

## Status

| Task | Subagent | State |
|---|---|---|
| R0–R4 rebuild + D1 diagnostic | rebuild | **accepted 08:00** |
| L1 U-shape by popularity, L2 HR@3/MRR archived | local | **accepted 04:52** |
| N1 oracle + blend, N2 HR@3/MRR current | n12 | **accepted 08:15** |
| N3 case study | n3 | **accepted 07:20** |

## Log

- **04:45 — Track R.** R0/R1 done. The MiniLM revision (1110a243) matches the recorded one. The regenerated `item_text_emb.npy` sha256 (2e0e2330…) differs from the recorded e494249a…. The rebuild agent reports float-level noise (max abs diff 1.6e-7 against an older local copy). The real check is whether the text_recency90 Val metrics and K200 target ranks match. Approved running `full_data_ranking --phase report` and `full_data_ranking_audit` on the regenerated study: I checked that both write only inside the gitignored `results/local_experiments/20261103-full-promising-ranking/`, and `objective_coreview` prepare needs them. Publish phases stay forbidden, and a failed audit check may not be bypassed.
- **04:52 — L1 U-shape by popularity (accepted).**
  - Question: is the forced-list U-shape explained by preferring unpopular products, by the score shortcut, or both?
  - Hypothesis (written before the run): arm A shows both effects; arm C shows neither.
  - Run: `results/analysis/20261105-ushape-popularity`. 50,000 random natural Val lists (seed 0); policy arms A and C, seeds 0–2; candidate groups defined before T1.
  - Alignment check: recomputed ranks match each checkpoint's saved `prefix_ranks` exactly.
  - Result:
    - **Score shortcut:** A's centred logit rises at low retrieval ranks in *every* popularity group and seed (decile 10 − decile 6: head +0.32 to +0.38, tail +0.42 to +0.52, new +0.44 to +0.54). C's curves fall monotonically (−0.04 to −0.11).
    - **Popularity:** tail is not preferred overall (tail − head −0.07 to −0.13 in A). New products are preferred in *both* arms (new − head: A +0.67 to +0.89, C +0.27 to +0.37), so A's preference is about 2.4× C's.
    - **Composition:** holding group shares fixed changes the curve by ≤ 0.03, so composition explains essentially none of the U-shape.
  - Verdict: **interesting; hypothesis partly held.** The U-shape is the score/rank shortcut acting within every popularity group, not a popularity preference. The new-product preference exists in both arms and is stronger in A. Figure `fig_ushape_popularity.png` reviewed. Path closed.
- **04:52 — L2 HR@3 / MRR@10 for the archived pipelines (accepted).**
  - Run: `results/analysis/20261105-secondary-metrics-archived-r2`. The earlier `-archived` folder is a crashed first attempt holding only config and git hash.
  - Reproduction check: all 33 saved nDCG@10 / Recall@10 values reproduce (max diff 8e-17).
  - Result:
    - Same system ordering as nDCG@10.
    - Natural DIN K_train 500 − forced MLP: HR@3 +0.00328 [+0.00297, +0.00361], MRR@10 +0.00278 [+0.00255, +0.00302]. Head gains; new is now inconclusive (HR@3 +0.00002 [−0.00031, +0.00037]); tail is a small loss.
    - Natural MLP − forced MLP: loses on new (HR@3 −0.00055 [−0.00087, −0.00020]).
  - Verdict: **confirms nDCG@10 conclusions; historical tables only.** It closes the archived part of the "Hit@3/MRR@10 unverified" status row. Path closed.
- pytest after accepting Track L code: 289 passed.
- **04:56 — R2 reproduced exactly.**
  - text_recency90 Val matches the published `metrics.csv` to full precision: nDCG@10 0.006283596988513183, Recall@10 0.011506106114173385, Recall@200 0.054746324742468065. Popularity and text_mean also match.
  - The independent 100-query check shows max rank difference 0.
  - The embedding hash difference is float noise (max 1.56e-7), as the R2 metrics confirm.
  - R3 sample smoke passed, and the full R3 prepare passed the frozen-reference K200 check. Its feature build is running (~12 min for B, then Val).
  - Verdict: **rebuild trustworthy so far**.
- **05:25 — R3 status.**
  - Full prepare done (~24 min). Val natural hits 14,916 (matches). One ranker-training (B) positive differs: prefix natural 16,819 vs 16,818, all-B 20,676 vs 20,675. This is likely a K200-boundary tie moved by float noise. As a result, negative plans change and the fits cannot be bit-identical.
  - inst/user/target/history/split arrays hash-match the published ones.
  - bce_text seed 0: initial state hash matches. It selected epoch 8 vs the published 10, with inner nDCG 0.010740 vs 0.010780 on a flat curve.
  - Fits take ~8.5 min each, so all 12 should finish ~07:00, then evaluate (~07:15), then R4.
  - Evaluate refuses to run until all 12 fits exist, and it locks selection before outer scoring. I kept that protocol: no early Val scoring.
  - N1 may freeze its blend weights on the inner tuning sets (B) before any Val score exists. N3 may freeze buckets and draw the retrieval-only bucket now.
- **05:38 — bce_text fits done (3/12).** Regenerated selected epochs 8/8/7 vs published 10/6/6; inner nDCG@10 0.010740/0.010591/0.010709 vs published 0.010780/0.010514/0.010494. Within seed noise. pairwise_text is running.
- **05:59 — pairwise_text fits done (6/12).**
  - Regenerated selected epochs 3/3/5 vs published 2/3/3; inner nDCG@10 0.010186/0.009696/0.009937 vs 0.009452/0.010058/0.009941.
  - Seed means: 0.009940 vs 0.009817.
  - Individual pairwise seeds swing by ~±0.0007, larger than for BCE, but the seed means agree. This is consistent with the one-positive difference in the negative plans, not a recipe difference.
  - pairwise_coreview is running. The N1 blend agent has frozen its config.
- **06:20 — pairwise_coreview fits done (9/12).**
  - Selected epochs 7/5/5 vs published 10/8/5; inner seed-mean 0.010381 vs 0.010317.
  - Reviewed the N1 config, registered at 04:58 before any score existed:
    - 7×7 grid; the shared (a, b) is chosen on the inner tuning queries (ranker-training period), so Val is untouched.
    - New-candidate flag = no review before the period start. At Val query time this is known (zero pre-T1 reviews), so it is serving-time valid.
    - The oracle is reported as an upper bound only.
  - Accepted as designed.
- **06:41 — all 12 fits done; selection locked; Val evaluate running.** pairwise_combined: selected epochs 6/6/7 vs published 10/6/5; inner seed-mean 0.010461 vs 0.010351.
- **06:58 — R3 Val evaluate complete; reproduction check** (`results/logs/20261105/R3e_compare_published.txt`).
  - Retrieval reproduces exactly.
  - Reproduced (seed-mean nDCG@10, regenerated vs published): bce_text 0.008141 vs 0.008115 (+0.000026); pairwise_text 0.007787 vs 0.007767 (+0.000019); pairwise_combined 0.007917 vs 0.007930 (−0.000013).
  - **Not reproduced: pairwise_coreview 0.007735 vs 0.008065 (−0.000331).** It is low in every seed (ours 0.00761/0.00782/0.00777, published 0.00804/0.00805/0.00810). Seed 2 has the same selected epoch in both, so this looks systematic.
  - Consequence: the published co-review gain over pairwise text (+0.000298) does not appear here (regenerated −0.000052).
  - Also: the pairwise new-item advantage over BCE shrinks (regenerated new nDCG 0.008754 vs 0.008666, against published 0.009152 vs 0.008697).
  - Hypothesis (before investigating): the co-review/graph feature columns differ from the teammates' (graph build, T0 boundary or scaler), not training noise.
  - Action: the rebuild agent runs report + audit (feature-reference replay), compares graph and feature hashes and edge counts, then starts R4 (time-boxed 25 min). N2 flags pairwise_coreview as not reproduced.
- **07:15 — co-review investigation (rebuild agent).** No feature bug found.
  - The local report + audit pass (96 feature-reference queries, 106 inner / 12 outer replays, 675 metric reconstructions).
  - The batched graph features equal the reference builder (max err 7.5e-9). The pre-T0 graph has 1,877,548 edges (matches).
  - The leaf/store metadata, protocol and config hashes match. Feature-array hashes differ, as expected from candidate tie order. No published feature values exist to compare co-review columns directly; our scaler means/SDs are recorded.
  - **The inner (B tuning) co-review gain reproduces: +0.000441 vs published +0.000500. The Val gain does not: −0.000052 vs +0.000298.** Per-seed published-vs-regenerated Val gaps for the other 9 fits have SD ≈ 0.00014; co-review's −0.00043/−0.00023/−0.00033 are ≈ 4 SE out.
  - Refit plans are entirely different draws (eligible rows 20,676 vs 20,675).
  - Verdict: **unresolved**: either the all-B refit is fragile or the Val features differ.
  - Follow-up approved (one dig): score the inner-selected checkpoints on Val (post-lock, exploratory, no selection). Hypothesis: if inner checkpoints keep coreview > text on Val, the reversal comes from the refit. R4 (bce_coreview vs bce_text) gives a second, independent look at the co-review columns. R4 started 07:02.
- **07:20 — N3 verified case study (accepted).**
  - Question: what do individual retrieval misses, demotions and rescues look like on the current pipeline?
  - Frozen rule: config written 04:52, before any case was viewed; cases drawn at 07:01 after evaluate completed.
  - Run: `results/analysis/20261105-case-study`. 28 events, 7 per bucket, seeded draw over sorted IDs, seed-0 bce_text / pairwise_text shown, ranks recorded for all seeds.
  - Eligible counts:
    - B1, new-target retrieval miss: 70,715;
    - B2, retrieval top-10 hit that BCE demotes and pairwise keeps: 342;
    - B3, BCE rescue from retrieval ranks 11–200: 2,515;
    - B4, counterexample where pairwise demotes and BCE keeps: 305.
  - Checks: alignment, bucket rules, and an independent float64 retrieval-rank reconstruction for the 21 K200 cases. pytest 301 passed.
  - Result:
    - **Only 3/21 BCE-vs-pairwise contrast cases meet their bucket rule in all three seeds** (0/7 for B2). So event-level objective differences are largely seed-dependent.
    - Contrast cases are mostly substitutes (17/21).
    - In 6/7 B1 misses, the retrieval top 3 echo the last product's store (recency text-mean). 6/7 B1 targets are unrelated to the last product, e.g. a phone case after a Fitbit charger.
    - Stable rescues are same-type products for a newer phone model.
    - Counterexample 5991: pairwise's top 3 are Pixel 4a cases whose titles say "NOT Compatible with Pixel 4a 5G", for a Pixel 4a 5G user. This is from titles only.
  - Coding is model-assisted and needs human review; there is no device audit.
  - Verdict: **interesting (illustrative)**. Seed instability matches the small aggregate objective differences. Path closed.
- **07:35 — R4 bce_coreview done; the co-review question is largely resolved.**
  - Val nDCG@10 0.008167 vs published 0.008409 (−0.000242; outside the 0.0002 tolerance). Initial states match.
  - **Seed 0 selected the same epoch (8) as published and reproduces almost exactly: 0.008357 vs 0.008364.** Seed 1 stopped at epoch 5 vs 9 (0.007985 vs 0.008617). Seed 2: epoch 7 vs 8.
  - So the co-review Val features are not systematically different. The gap is early-stop / refit variance from the different negative-plan draw.
  - Regenerated factorial (published in brackets):

    | Co-review effect | Overall | New | Warm |
    |---|---|---|---|
    | BCE | +0.000026 [+0.000295] | −0.000416 [−0.000354] | +0.000218 [+0.000538] |
    | Pairwise | −0.000052 [+0.000298] | −0.000179 [−0.000279] | −0.000005 [+0.000468] |

    Interaction −0.000078 [+0.000003].
  - **Conclusion: the co-review new-item loss reproduces under both objectives. The overall and warm gains are within training-recipe variance and shrink to about zero in this regeneration.** This is a robustness finding about the teammates' §13–14 results, not a bug.
  - Next: the inner-checkpoint diagnostic runs on MPS as a second check of the refit explanation; the objective_coreview local report and audit run on CPU. N1/N2 have been told about the bce_coreview archives.
- **08:00 — Track R final + D1 inner-checkpoint diagnostic (accepted).**
  - All of R0–R4 ran. Both local audits PASS. Initial-state hashes match for all 15 fits.
  - Regenerated vs published Val overall nDCG@10:
    - reproduced: bce_text +0.000026, pairwise_text +0.000019, pairwise_combined −0.000013;
    - not reproduced: pairwise_coreview −0.000331, bce_coreview −0.000242.
  - Regenerated contrasts (published in brackets):
    - pairwise − BCE new **+0.000089, not significant** [+0.000455, significant];
    - pairwise − BCE overall −0.000354 [−0.000347];
    - novelty interaction contrast +0.000459 [0.000125, 0.000782] [+0.000146, inconclusive].
  - D1 (`results/analysis/20261105-inner-checkpoints-on-val`; post-lock, exploratory): the inner-selected pre-refit checkpoints scored on Val give
    - co-review − text, pairwise: **+0.000280 [0.000152, 0.000415]** (new +0.000377, warm +0.000135);
    - co-review − text, BCE: **+0.000214 [0.000102, 0.000335]** (new +0.000486);
    - after the all-B refit, the same contrasts are −0.000052 [−0.000180, 0.000075] and +0.000026 [−0.000093, 0.000151], with BCE new −0.000416 [−0.000597, −0.000252].
  - Verdict: **interesting; hypothesis held.**
    - The co-review Val features are fine. The gain (even on new items) exists before the refit and disappears at the all-B refit.
    - The refit-draw variance is confounded with the prefix-vs-all-B data difference.
    - The co-review gain and the pairwise new-item advantage are therefore **not robust to the training draw**; BCE's overall lead and every ranker's gain over retrieval are.
  - Path closed. Disk: ~17 GB of rebuilt studies under the gitignored `results/local_experiments/`.
- **08:15 — N1 oracle and blend + N2 current-pipeline secondary metrics (accepted).**
  - Checks: verified that `choose()` reads only inner tuning logits (`val_logits` is reachable only from replay/run/diagnose). The freeze was at 06:56:38, before the first Val read at 06:57:32; pair 2 froze at 07:30, before its run at 07:50. The (0,0)/(1,1) blend ranks equal the components exactly.
  - Pair BCE text + pairwise text (a = 1/6, b = 1/2):
    - blend − BCE: all +0.000037 [−0.000029, 0.000101], new −0.000402 [−0.000566, −0.000246], warm +0.000236 [0.000182, 0.000294];
    - event oracle − BCE: +0.001963 [0.001864, 0.002070];
    - group oracle − BCE: +0.000056 [−0.000032, 0.000144].
  - Pair BCE + co-review + pairwise text: blend − BCE + co-review +0.000199 [0.000122, 0.000282]. This is tentative because of BCE + co-review's seed variance.
  - Mechanism: z-mixing compresses new candidates' spread, so the new top-10 share falls (0.179 → 0.169). Tuning-period "new" fills 57% of top-10 slots vs 18% on Val, so the tuned direction does not transfer.
  - N2: same ordering as nDCG@10. Pairwise − BCE on new targets includes zero at nDCG, HR@3 and MRR.
  - Verdict: **N1 hypothesis not supported for the main pair** (the oracle-gap part held). **N2 mostly supported.** Paths closed.
  - Noted for the team: 656 Val candidates whose first review equals the query time are all the target (§4.6 admission rule), so any query-relative item-age feature would leak the label.
- **08:20 — all tasks complete; finishing documents and stopping the loop early** (well before the 09:30 cut-off; no further authorised task remains).
