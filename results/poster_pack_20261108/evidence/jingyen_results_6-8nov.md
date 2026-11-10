## Cluster policy replication, meso, oracle and micro analyses (Jing Yen, 6–7 November 2026)

Validation only; no Test predictions. Appended at merge time. The cluster A/B/C/D run below uses the **original** frozen text seed-0 candidates (hash-checked) on an H100, independently of the 29 October laptop run on rebuilt candidates; both reach the same conclusion (C−A +0.00169 here versus +0.00187 there; no arm qualifies). Sections are kept in chronological order as originally written.


### 2 November: meeting follow-up and development execution

The supplied meeting summary prioritises natural-list performance and head/tail/new-item trade-offs. Keep forced-list effects as pipeline diagnostics; the summary alone does not establish that held-out Test labels entered training. The existing A/B/C/D study is the immediate next experiment. Reuse the completed segment/funnel analysis, then extend representative cases to 20–30 systematically selected items. An oracle comparison is a diagnostic upper bound, pending an explicit choice of the two systems; DIN here is a ranker, not a retriever. Randomising candidate positions alone does not remove score/item-distribution differences associated with force-added targets. No new model or label policy was introduced.

Local execution on source commit `7e10c06`:

- `.venv/bin/python -m pytest -q`: **140 passed**.
- Ranker preflight: **ready**, no missing inputs; `val.npz` and `meta.json` match the frozen candidate identities. Record: `results/targeted_study/policy-20261102-preflight.json`. This supersedes the older missing-local-input inventory below for this checkout only; the cluster still needs its own check.
- `policy-20261102-development_sample`: all four arms completed, seed 0, one epoch, fixed 2% full-data user sample, CPU. Validation has 4,775 events / 2,553 users; A/B train on 6,224 lists and C/D on 730 natural-hit lists. All arms have the same initial-parameter hash. K200 checkpoint macro nDCG reproduces exactly in all four arms; C has one target-rank difference with no headline metric difference, retained in its anchor diagnostics.
- The report completed at `results/targeted_study/policy-20261102-development-report/report.md`, including group metrics, training support, funnel, expansion and score diagnostics. These are **development checks, not research results**; do not select a policy from their scores. No Test predictions were loaded. Original candidates and historical results were preserved.
- This session exposes neither CUDA nor MPS. Full-data pilot/replication have not run; cluster submission remains user-managed. The existing Slurm wrapper passes `bash -n`.

Next, on the cluster after syncing this source and checking its original inputs, run from the repository root:

```bash
.venv/bin/python -m cascade.pipelines.study_preflight --study-kind ranker --require-ready \
  --output results/targeted_study/policy-20261102-cluster-preflight.json
mkdir -p results/logs
sbatch --export=ALL,STUDY_KIND=ranker,STAGE=pilot,RUN_ID=policy-20261102-pilot scripts/slurm/targeted_study.sbatch
```

Use a fresh RUN_ID if that directory already exists on the cluster. The pilot trains A/B/C/D seed 0 and automatically writes `results/targeted_study/policy-20261102-pilot-pilot-report/report.md`. Review validity and runtime, then complete seeds 1/2 for all valid arms under the fixed protocol below. Final Test stays paused.

#### 6 November: cluster pilot launched through xlogin

The user explicitly authorised SSH via `xlogin`, superseding the earlier user-managed-SSH restriction for this work. An isolated checkout of source commit `7e10c06` is at `/home/t/tjingyen/cp4285-policy-20261106/cascade-rec`. All 13 copied processed-input/candidate files match local SHA-256 hashes; the pinned environment was installed with Python 3.11.18 and PyTorch 2.5.1+cu124. No training recipe, split, filtering or metric was changed.

- **Active Slurm job: `1080925`**, running on `xgpi17`, NVIDIA H100 NVL. Requested one `h100-47` GPU, four CPUs, 32 GB host RAM, six hours in `gpu-long`. NVIDIA reports 95,830 MiB. A pending A100 request (`1080918`) was cancelled before execution; a generic-GPU attempt (`1080920`, `xgpe2`) failed its CUDA check before any model training. Its outputs are not research evidence.
- The active job passed **127 tests, with 13 sample-data-dependent skips**. The full local suite previously passed all 140. CUDA is confirmed usable on the H100.
- GPU development run `policy-20261106-gpu-development_sample` completed all four arms and reporting. Copied report: `results/targeted_study/policy-20261106-gpu-development-development-report/report.md`. These remain development-only checks, not full-data research findings.
- Full pilot `policy-20261106-pilot` passed preflight and entered arm A epoch 1 at approximately **00:13 SGT on 6 November**. At this observation its status was `running`, with no completed arms. No full-pilot result or policy selection is claimed yet. Seeds 1/2 were not submitted; Test remains paused.
- The wrapper `/home/t/tjingyen/cp4285-policy-20261106/run-pilot.sh` runs tests, GPU development/reporting, then the pilot/reporting sequentially with failure stopping the job. It does not submit another job. Environment versions and input hashes are recorded beside the checkout. Preflight's static `cluster_execution` string predates this authorised SSH execution; the actual job history is recorded here.

Inspect the active job and log:

```bash
ssh xlogin 'squeue -j 1080925'
ssh xlogin 'tail -n 40 /home/t/tjingyen/cp4285-policy-20261106/cascade-rec/results/logs/slurm-cascade-policy-pilot-1080925.out'
```

On successful completion, the full report will be at `/home/t/tjingyen/cp4285-policy-20261106/cascade-rec/results/targeted_study/policy-20261106-pilot-pilot-report/report.md`. Check all four arms and output identities, review runtime, then launch the separate seeds-1/2 phase on this same source/input snapshot. Do not rerun completed fits or infer a three-seed remedy from the pilot alone.

#### 6 November: full-data seed-0 pilot complete

Slurm job `1080925` completed with exit code `0:0` in **1:16:49**. All four arms and reporting finished on the H100. Checkpoints, selected natural-Val prefix predictions and provenance were copied locally to `results/ranker_policy/policy-20261106-pilot/`; the report and CSV evidence are at `results/targeted_study/policy-20261106-pilot-pilot-report/`. All **26 declared arm-artifact hashes** match after transfer. K200 selected-checkpoint nDCG reproduces exactly in every arm, with zero changed target ranks. No Test predictions were loaded.

Full Val: 232,904 events / 124,416 users. Same natural frozen-retriever K200 lists and user-macro aggregation throughout:

| Arm | Training population / score input | nDCG@10 | Recall@10 | New nDCG@10 | Warm-head nDCG@10 | Warm-tail nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| A | All lists (forced when missed), score enabled | 0.007076 | 0.014252 | 0.003541 | 0.011637 | 0.001244 |
| B | All lists, score masked | 0.004216 | 0.009691 | 0.002404 | 0.006649 | 0.001029 |
| C | Natural-hit-only, score enabled | 0.008617 | 0.016233 | 0.002104 | 0.015925 | 0.000891 |
| D | Natural-hit-only, score masked | 0.007780 | 0.014961 | 0.002496 | 0.014043 | 0.000735 |

**Measured pilot trade-off:** C versus fresh A improves overall nDCG by **0.001541** (21.8%), paired-user 95% CI **[0.001244, 0.001819]**, while new-item nDCG falls by **0.001436**, CI **[-0.001769, -0.001106]** (40.6% relative loss). Warm-head improves by 0.004287, CI [0.003747, 0.004791]; warm-tail falls by 0.000353, CI [-0.000612, -0.000097]. B-A is -0.002860, CI [-0.003147, -0.002591]; masking score alone is not a remedy in this pilot. D-C is -0.000837, CI [-0.001053, -0.000609]. These are exploratory user-sampling intervals conditional on one trained seed, not training-seed replication.

**Expansion and score response:** A drops from K200 nDCG 0.007076 to K500 0.002400; C drops from 0.008617 to 0.007863. A's added-position-201–500 top-ten slot share on the common 1,000-event diagnostic cohort is 93.7%, versus C's 37.6%. Mean score/logit Spearman changes from -0.401 to +0.156. In the fixed-feature score probe, mean logit(z=-1) minus logit(z=0) changes from +2.672 to -0.207. These accompany the recipe change, but are not a unique causal explanation; interventions may be off the natural feature distribution. C still loses quality under K expansion.

**Support and interpretation:** A/B use 315,114 training lists (158,634 users); C/D use 38,793 lists (32,829 users). New-in-ranker-period target lists fall from 118,392 to 5,303. Natural-only changes supervised support and optimizer exposure as well as removing forced targets. The matched within-seed initialization checks pass. Selected epochs A/B/C/D are 11/15/5/8, respectively. Do not present C-A as an isolated contamination effect.

**Decision:** no policy qualifies for integration. All three seeds are required, and C/D also show a new-item loss against A in this pilot. Keep the reference unchanged. Next is seeds 1/2 for **all four valid arms**, without pruning negative arms or introducing another model. Replication has not been submitted in this status check; final Test remains paused.
As of 2 November, this checkout contains `data/processed/item_text_emb.npy`, `item_text_emb.json`, and `data/processed/candidates/two_tower_text_seed0/{ranker_train.npz,val.npz,val_target_score.npz,meta.json}`. Ranker preflight and the four-arm development run pass; checked validation/metadata hashes match the frozen source. The earlier missing-file diagnosis is historical. Verify these original inputs independently on the cluster before full fits. User handles SSH and cluster commands; no teammate allocation is assumed available.
Candidate-slot amplification and the original MLP/DIN top-three products still require their respective candidate-level analyses/replay; recovering inputs alone does not complete them. Complete the locked annotation before device claims. Final Test and index/cost work stay paused. The ablation runners have not produced full-data results; the updated submission script runs blocking input checks and exports comparison reports. Cluster execution has not been verified here. The 2 November test/development evidence above verifies the current source; older implementation-review/test counts remain historical.

#### 7 November: seeds 1/2 replication submitted

The user authorised submission. Slurm job `1089445` (`STAGE=replication`, `RUN_ID=policy-20261107-replication`) was submitted on the same checkout/inputs as the pilot, with the pilot's resources (gpu-long, h100-47, 4 CPU, 32G, 6h). It covers all four arms × seeds 1/2 (eight fits). At submission it was pending (Priority). No result is claimed; Test remains paused. After completion, run the all-seeds report with `--studies results/ranker_policy/policy-20261106-pilot results/ranker_policy/policy-20261107-replication`.

#### 7 November: E3 candidate-count curve per policy arm (pilot seed 0, no training)

Report `results/targeted_study/e3-20261107-pilot-kcurve/` (`cascade.eval.ranker_policy_extension_report`, 1,000 user resamples). Overall macro nDCG@10 at K50/100/200/500 and paired gap of each K versus that arm's best observed K:

| Arm | K50 | K100 | K200 | K500 | Best | K200−best CI | K500−best CI |
|---|---:|---:|---:|---:|---:|---|---|
| A | 0.008026 | 0.008370 | 0.007076 | 0.002400 | 100 | [-0.001524, -0.001053] | [-0.006317, -0.005631] |
| B | 0.006598 | 0.006023 | 0.004216 | 0.001711 | 50 | [-0.002683, -0.002052] | [-0.005227, -0.004540] |
| C | 0.008406 | 0.008684 | 0.008617 | 0.007863 | 100 | [-0.000234, +0.000113] | [-0.001031, -0.000579] |
| D | 0.007677 | 0.007932 | 0.007780 | 0.006937 | 100 | [-0.000326, +0.000035] | [-0.001245, -0.000731] |

Under natural-only training (C/D) the K200 interval versus K100 includes zero and the K500 loss is about 7× smaller than A's. C's 200→500 competition loss is −0.0021 versus A's −0.0065 with similar new-target gain (+0.0013 / +0.0018). The earlier RQ2 conclusion that larger K₁ strongly hurts is therefore largely conditional on the forced-positive training recipe. Single seed; interval excludes training-seed variance; no K₁ selection is made here (no δ recorded).

#### 7 November: extension arms authorised (pre-registered before any outcome)

The user authorised additional controlled arms after seeing the seed-0 pilot. They are therefore **post-hoc and exploratory**; any qualification is not confirmatory. Same frozen candidates, feature contract, normalisation, initialisation per seed, BCE/50 negatives, selector and K prefixes as A–D:

- **E (score imputation):** all lists; each forced target's retrieval score replaced by the score of a uniformly drawn retrieved candidate from the same list (RNG keyed by instance ID, seed-independent). Natural lists and all other cells unchanged (verified: 276,320 cell changes over 276,321 forced lists; mean forced-target score −0.342 → −0.180, natural targets −0.154). Tests whether removing the score shortcut while keeping forced supervision recovers new-item quality.
- **F (longer natural-only schedule):** C with max 100 epochs and patience 10. Tests whether C's new-item loss is an exposure artefact (C stopped at epoch 5 with 4,850 selected steps versus A's 86,658).
- **G / H (partial forced lists):** natural lists plus a fixed 10% / 30% subsample of forced lists (RNG 20261107 over the full universe): 66,578 / 121,948 lists. Tests an intermediate point on the overall-versus-new-item trade-off.
- Fresh A and C controls are retrained inside each extension run; seeds 0/1/2 run as three independent jobs. Report with `cascade.eval.ranker_policy_extension_report`; the same guard (3 seeds, overall K200 CI lower bound > 0, no new-item loss versus A) is applied, labelled exploratory.
- Development check `ext-20261107-localdev_sample` (2% users, 1 epoch, CPU) completed all six arms with identical initial parameters. Development values are not reported.
- Cluster execution (7 Nov, user-authorised via xlogin): separate checkout `/home/t/tjingyen/cp4285-ext-20261107/cascade-rec` (copied from the pilot checkout so job 1089445's source-hash guard is untouched), plus the three changed files (SHA-256 in `staging/sha256.txt`). A cluster-side driver (`staging/ext_driver.sh`, log `driver.log`) waits for the copy, verifies hashes and that `cascade` imports from the new checkout, then submits `scripts/slurm/policy_extension.sbatch` for seeds 0/1/2 as `ext-20261107-s{0,1,2}`. Check `driver.log` for job IDs; if the import assertion fails no job is submitted. Afterwards: `python -m cascade.eval.ranker_policy_extension_report --studies results/ranker_policy/ext-20261107-s0 results/ranker_policy/ext-20261107-s1 results/ranker_policy/ext-20261107-s2 --output results/targeted_study/ext-20261107-all-seeds`.
- Correction (7 Nov 15:40 SGT): the full checkout copy was too slow (many small `.venv`/data files), so the driver was disabled (its checksum file replaced so it exits before submitting; original hashes kept in `staging/sha256.ok`). The extension instead runs from `/home/t/tjingyen/cp4285-ext-20261107/fast/cascade-rec`: code/results copied, `data` symlinked read-only to the pilot checkout, the pilot checkout's Python environment used via `PY=...` with `PYTHONPATH` pointing at the new `src` (verified `cascade` imports from the new checkout). Submitted Slurm jobs `1089821/1089822/1089823` = seeds 0/1/2 (`ext-20261107-s{0,1,2}`). Outputs land in that checkout's `results/ranker_policy/`.
- Jobs 1089821–23 failed at import within 5 s (no run directories created): the lightweight copy's `--exclude data` also dropped `src/cascade/data`. Re-synced with root-anchored excludes; `diff -r` against the pilot `src/` and `configs/` now shows only the two intended files. Resubmitted as **1090001/1090002/1090003** (seeds 0/1/2, same RUN_IDs).

#### 7 November: A–D replicated over three seeds

Replication job `1089445` completed (exit 0, 1:47:06; 8 fits). Reports were generated on the cluster as CPU job `1090722` (login-node memory cap too small) and copied back: `results/targeted_study/policy-all-seeds-20261107/` (factorial) and `results/targeted_study/e3-20261107-all-seeds/` (K-curve). The local copy of the replication archive was incomplete at report time because of SSH drops; the reports read the complete cluster copy.

K200 macro nDCG@10, mean over seeds 0/1/2 (seed SD): A 0.006887 (0.000316), B 0.004200 (0.000269), C 0.008572 (0.000124), D 0.007625 (0.000421). Seed-mean paired user-bootstrap differences:

| Comparison | Overall | New-in-period | Warm-head | Warm-tail |
|---|---|---|---|---|
| C−A | +0.001685 [0.001445, 0.001920] | −0.000757 [−0.001030, −0.000489] | +0.004222 [0.003809, 0.004643] | −0.000469 [−0.000674, −0.000263] |
| B−A | −0.002687 [−0.002896, −0.002478] | −0.000599 [−0.000863, −0.000364] | −0.004939 | −0.000170 [−0.000370, 0.000010] |
| D−C | −0.000946 [−0.001106, −0.000787] | +0.000124 [−0.000016, 0.000262] | −0.002026 | −0.000005 |

C beats A in every seed (overall 0.008617/0.008431/0.008667 versus 0.007076/0.007063/0.006522) and has the lowest seed SD. The pilot's new-item loss also replicates but is smaller in the mean (−0.000757, about −24% relative, versus −40.6% at seed 0). Seed-to-seed new-item values overlap (A 0.0028–0.0035; C 0.0021–0.0026). **No arm qualifies** under the pre-registered guard because of the new-item loss; A remains the integration reference without a claim that it is optimal.

Three-seed K-curve: C's best observed K is 100 (0.008609). K200 versus K100 is −0.000037 [−0.000194, 0.000118]; K500 versus K100 is −0.000686 [−0.000870, −0.000477]. A's K500 loss versus its best K is −0.005971, so C's is about 8.7× smaller. The large-K collapse is largely tied to the forced-positive recipe, replicated over seeds.

#### 7 November: meso secondary metrics and per-event oracles (saved predictions, no training)

`scripts/analysis_meso_oracle.py` → `results/targeted_study/meso-oracle-20261107/`. K200 natural Val, user-macro, seed-mean (policy A/B/C and R2/R3 three seeds; D seed 0; R1 deterministic). Oracle = per event and seed, the better target rank of two same-seed systems; an upper bound for a perfect per-event selector, not an achievable system.

| System | nDCG@10 | Hit@3 | MRR | Head nDCG | Tail nDCG | New nDCG |
|---|---:|---:|---:|---:|---:|---:|
| R1 retrieval order | 0.005642 | 0.004818 | 0.005390 | 0.011290 | 0.000166 | 0.000361 |
| A (forced, MLP) | 0.006887 | 0.005415 | 0.006130 | 0.011438 | 0.001316 | 0.003147 |
| C (natural-only) | 0.008572 | 0.007328 | 0.007676 | 0.015660 | 0.000847 | 0.002390 |
| R3 DIN (forced) | 0.006436 | 0.004708 | 0.005799 | 0.009865 | 0.001398 | 0.004074 |
| Oracle(A, C) | 0.011497 | 0.010005 | 0.009987 | 0.019907 | 0.001755 | 0.004511 |
| Oracle(MLP, DIN) | 0.009728 | 0.007775 | 0.008401 | 0.015513 | 0.001980 | 0.005210 |

- C−A Hit@3 +35% versus nDCG@10 +24%: natural-only gains are concentrated at the very top.
- Oracle(A, C) exceeds C by +0.002925 [0.002782, 0.003069] (+34%); oracle(MLP, DIN) exceeds MLP by +0.002885 [0.002757, 0.003021] (+42%). Seed-0 top-ten hits overlap only 2,117 of 5,319 (A/C) and 2,022 of 4,634 (MLP/DIN): the systems succeed on substantially different events.
- DIN−MLP: new-in-period +0.001254 [0.001013, 0.001501] (+44%), warm-head −0.001673 [−0.001961, −0.001345]. Learned rankers raise new-item nDCG 6.6–11× over retrieval order, whose new/tail values are near zero.
- An achievable router would need a selector using pre-event information only (for example candidate item-age); none is trained or claimed here.

#### 7 November: 24 micro cases (seed 0)

`scripts/analysis_micro_cases.py` → `results/targeted_study/micro-cases-20261107/{cases.md,cases.json}`. Four prespecified categories, 6 events each drawn with RNG 0 from all eligible natural-K200 events: C rescues warm-head target missed by A (pool 1,606); A keeps new-in-period target missed by C (387); DIN rescues new target missed by MLP (342); retrieval miss at K500 (204,329). Ranker top-3 products were recomputed from the saved A/C/DIN seed-0 checkpoints; all recomputed target ranks equal the saved ranks. Illustrative only: six cases per category do not estimate prevalence.

Observed patterns (to be quantified before being stated as general):
- **C concentrates on a few bestsellers.** Across the 24 lists, C's 72 top-3 slots contain 38 distinct products versus 59 for A and 52 for DIN; one smartwatch case appears 9 times and one wireless charger 7 times. C's warm-head rescues are mostly purchases of those same products, with the target's leaf category in the recent history in only 1 of 6 cases. This suggests the head gain partly reflects popularity concentration (a diversity cost), not personal relevance.
- **A/DIN keep new items through brand and model continuity.** Examples: TORRAS iPhone 12 case → TORRAS iPhone 13 case (DIN rank 1 and 4); Moto G6 owner → Motorola One 5G; three single-ear headsets → another headset; Anker → Anker stand; BELONGME iPhone case → BELONGME watch protector. Target store appears in the five-item history in 3/6 DIN-rescue cases versus 0/6 C-rescue cases.
- **Many new-in-period targets are iPhone 13 launch accessories** (first seen 13–25 Sep 2021), shortly after Val start (11 Aug 2021): the new-item segment is partly a product-launch wave.
- **A/DIN top-3 come from deep retrieval positions** (median retrieval rank 107 / 140.5 versus 53 for C), consistent at micro level with the low-score preference.
- **Retrieval misses are often device or category switches** (iPhone case buyer → Apple Watch band; Fitbit accessories → iPhone screen protector; renewed Samsung watch), with full-catalogue ranks of 649–28,699. No ranker can recover them.
- Next check: full-validation top-10 catalogue coverage and popularity concentration per arm, to test the diversity hypothesis.

#### 7 November: top-10 diversity on full Val (seed-0 models)

`scripts/analysis_top10_diversity.py` → `results/targeted_study/top10-diversity-20261107/` (all 232,904 natural K200 Val lists; top-1% by counts before T2 per CLAUDE.md §5; "new" = first seen at/after T1).

| System | nDCG@10 (3-seed) | Top-1% slot share | New-product slot share | Distinct products in any top-10 | Median pre-T1 reviews | Lists containing the single most-shown product |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval order | 0.005642 | 70.5% | 8.9% | 2,889 | 735 | 55.0% |
| Forced MLP (A) | 0.006887 | 41.3% | 26.2% | 16,394 | 66 | 36.0% |
| Natural MLP (C) | 0.008572 | 56.2% | 17.0% | 10,749 | 291 | 54.3% |
| DIN (forced) | 0.006436 | 31.3% | 34.3% | 16,863 | 13 | 33.2% |

The micro-case concentration pattern holds on full Val: natural-only training raises accuracy while recommending more bestsellers, fewer new products and about 35% fewer distinct products than forced A. Brandon's 28 October analysis also found 71% top-1% share for retrieval. Natural-list DIN composition was not measured here (its checkpoints are not local). Popularity and accuracy co-move across these four systems; this is descriptive, not a causal popularity effect.

Poster figures (`scripts/poster_figures_20261107.py` → `results/figures/poster_20261107/`): fig1 K-curve A vs C; fig2 winners/losers grid by item group; fig3 accuracy versus bestseller and new-product share; fig4 MLP/DIN under forced versus natural lists (natural-list values from `docs/report_2026-10-28.md`).

#### 7 November: last-item hybrid retrieval, launch titles and brand loyalty (saved ranks, no training)

`scripts/analysis_last_item_hybrid.py` → `results/targeted_study/last-item-hybrid-20261107/`. Inputs: Brandon's saved full-catalogue Val ranks for the frozen two-tower and the untrained last-item scorer (`results/analysis/20261028-last-item/aligned_val_ranks.npz`), seed-0 ranker ranks. User-macro, 1,000-resample paired user bootstrap. A hybrid pool of 200 = top-a two-tower + top-(200−a) last-item; a target counts only if it lies within either source's allotted top-a/top-(200−a). Deduplication would let a real pool reach deeper, so hybrid recall is a **lower bound**.

| Pool (two-tower + last-item) | Recall@200 | vs two-tower 200 | Head | Tail | New |
|---|---:|---|---:|---:|---:|
| 200 + 0 (current) | 0.0687 | — | 0.1180 | 0.0153 | 0.0261 |
| 180 + 20 | 0.0744 | +0.0056 [0.0052, 0.0062] | 0.1186 (ns) | 0.0257 | 0.0369 |
| 150 + 50 | 0.0746 | +0.0059 [0.0051, 0.0066] | 0.1136 | 0.0318 | 0.0418 |
| 100 + 100 | 0.0707 | +0.0020 [0.0011, 0.0030] | 0.1013 | 0.0388 | 0.0454 |

- Twenty last-item slots raise overall Recall@200 by ≥8% with no detectable head loss, tail +68% and new +41%. 150+50 maximises overall (+8.5%) with new +60% and a small head cost (−0.0044).
- Retrieval-only top 10 (no ranker), 7+3 slots: Recall@10 0.0128 vs 0.0104 (+22%, CI [0.0020, 0.0027]); head unchanged (CI includes zero), new 0.0060 vs 0.0009. Rankers are trained only on two-tower pools; end-to-end ranking of a hybrid pool requires new candidate lists and retraining (not done).
- **Launch waves verified from titles:** 1,995 of 8,475 Val-new products (24%) mention iPhone 13 and 1,014 mention S22. In the week of 20 Sep 2021, 186 of 346 new products mention iPhone 13; in the week of 21 Feb 2022, 194 of 330 mention S22. iPhone 13 announced 14 Sep, on sale 24 Sep 2021 (Apple newsroom); Galaxy S22 Unpacked 9 Feb, on sale 25 Feb 2022 in most regions.
- **Brand loyalty:** only 9.0% of Val purchases are from a store already in the user's history, yet they carry most top-10 hits (rankers' Recall@10 6.4–7.1% versus 0.9–1.1% for new-store purchases). On same-store purchases the untrained last-item order reaches Recall@10 0.0906, above every learned ranker (natural MLP 0.0707, forced 0.0646, DIN 0.0639) and far above two-tower order (0.0262). For same-store **new** products: last-item 0.0842, DIN 0.0459, forced MLP 0.0393, natural MLP 0.0289. Learned rankers dominate new-store purchases. Seed 0; descriptive, not causal.

#### 7 November: extension arms E–H, three seeds (jobs 1090001–1090003)

All 18 fits completed (exit 0, 1:17–1:23 each on H100). Report `results/targeted_study/ext-20261107-all-seeds/` (CPU job 1091436 on the cluster). Fresh A/C controls inside the same runs. K200 seed-mean nDCG@10 and paired user-bootstrap differences versus fresh A:

| Arm | Recipe | nDCG@10 | vs A overall | vs A new-in-period | K500 |
|---|---|---:|---|---|---:|
| A | all lists, forced | 0.006902 | — | — | 0.002307 |
| C | natural only | 0.008519 | +0.001617 [0.001386, 0.001839] | −0.000817 [−0.001071, −0.000548] | 0.007934 |
| E | forced kept, forced-target score imputed | 0.004882 | −0.002020 [−0.002235, −0.001820] | +0.000172 [−0.000061, 0.000404] | 0.001847 |
| F | natural only, max 100 epochs, patience 10 | identical to C | identical | identical | identical |
| G | natural + 10% forced | 0.008167 | +0.001265 [0.001072, 0.001442] | −0.000237 [−0.000470, 0.000003] | 0.005500 |
| H | natural + 30% forced | 0.007816 | +0.000914 [0.000751, 0.001074] | −0.000080 [−0.000253, 0.000100] | 0.003886 |

- **F = C exactly:** in every seed F selected the same epoch (9/4/6) after 7–10 further epochs without improvement. C's new-item loss is not an under-training artefact.
- **E fails:** imputing a typical score onto forced targets removes the score shortcut but leaves overall quality below A (and below B-like levels); new-item quality is unchanged within noise. Forced positives appear harmful beyond their score; which other inputs carry it is not identified.
- **G/H form a dose-response:** adding forced lists back trades overall quality for new-item quality and reintroduces the K500 collapse in proportion (K500 0.0079 → 0.0055 → 0.0039 → 0.0023 for 0/10/30/100% forced). H shows no detectable new-item loss versus A (CI includes zero) with +13% overall, but its point estimate is negative, so under the pre-registered guard nothing qualifies (`qualified: []`). These arms were added after the pilot; exploratory.

#### 7 November: list-merge router (saved seed-0 top-10 lists, no training)

`scripts/analysis_list_merge.py` → `results/targeted_study/list-merge-20261107/`. Final top 10 = first k of natural MLP, then fill from forced-list DIN (deduplicated); no target information used. nDCG@10×100 (overall / new / tail / head): natural 0.862 / 0.211 / 0.089 / 1.593; natural8+din2 0.845 / 0.237 / 0.100 / 1.534 (overall −0.00017 [−0.00025, −0.00009]); natural7+din3 0.835 / 0.250 / 0.108 / 1.501. A history-length router (natural for ≤5 prior purchases, else DIN) loses more overall (−0.00081). Merging is a tunable dial (two DIN slots: −2% overall, +12% new and tail), not a free gain. Natural-list DIN checkpoints are not local; with them the dial may be better.

#### 7 November: hybrid-pool and last-item-feature study submitted

User-authorised follow-up to the findings doc's experiments 2 and 3. Validation only; exploratory (designed after the meso/micro results).
- **Hybrid lists** (`scripts/build_hybrid_candidates.py`): per instance, top-180 two-tower + top-20 last-item (cosine of frozen item vectors to the most recent product), deduplicated, both after §4.6 filtering; every candidate keeps its two-tower score and lists are sorted by it. A 3,000-list local check reproduced the original top-180 exactly, gave 20 last-item additions per list and no duplicates. Output `data/processed/candidates/hybrid_tt180_li20_seed0/` (ranker_train forced as usual; val unforced plus target scores).
- **Last-item feature:** optional fifth numeric column cos(candidate text, last product text), via `RankerFeatureSpec.extra_numeric` (default 0, so existing checkpoints and the four-column contract are unchanged; 287 tests pass). The ranker already had same-store-in-history and mean-history cosine inputs.
- **Study** (`scripts/run_hybrid_study.py`): natural-only training, retrieval+matching features, the policy-study recipe; arms mlp_tt (control), mlp_hyb, mlp_tt_li, din_tt (control), din_hyb; normalisation fitted on the original full training lists. Local 2%-user, one-epoch check completed for the three original-pool arms.
- Cluster: build job `1091477`, then seeds 0/1/2 as `1091478`–`1091480` (`afterok` dependency), checkout `/home/t/tjingyen/cp4285-ext-20261107/fast/cascade-rec`, outputs `results/hybrid_study/hybrid-20261107-s{0,1,2}/`. No result yet.

#### 7 November: hybrid-pool and last-item-feature results (3 seeds)

Jobs 1091477 (build, 13 min) and 1091478–1091480 (5 arms each, 16–19 min) completed with exit 0. Hybrid Val Recall@200 (micro) 0.0804 vs 0.0716 for the original pool; 3,068 Val targets are found only through the 20 last-item slots. Ranks copied to `results/hybrid_study/` (checkpoints remain on the cluster); analysis `results/targeted_study/hybrid-study-20261107/`. End-to-end K₁=200 nDCG@10, seed means; user-bootstrap 95% CIs on seed-mean per-event values.

| Arm | nDCG@10 | Popular | Niche | New | Recall@10 | Hit@3 |
|---|---:|---:|---:|---:|---:|---:|
| MLP natural, original pool | 0.00852 | 0.01566 | 0.00081 | 0.00228 | 0.01584 | 0.00730 |
| MLP natural + last-item feature | 0.00865 | 0.01582 | 0.00088 | 0.00245 | 0.01615 | 0.00731 |
| MLP natural, hybrid pool | 0.00981 | 0.01598 | 0.00253 | 0.00470 | 0.01833 | 0.00825 |
| DIN natural, original pool | 0.00994 | 0.01793 | 0.00095 | 0.00303 | 0.01837 | 0.00857 |
| DIN natural, hybrid pool | **0.01072** | 0.01662 | 0.00295 | 0.00630 | 0.02010 | 0.00911 |

- **Hybrid pool removes the MLP trade-off:** +0.00129 [0.00110, 0.00149] overall (+15%), new +0.00242 [0.00213, 0.00270] (+106%), niche +0.00172 (×3), popular +0.00032 [−0.00001, 0.00063] (no loss).
- **DIN control reproduces Brandon's natural DIN** (0.00994 vs 0.00982). DIN on the hybrid pool: +0.00078 [0.00058, 0.00097] overall, new ×2.1, niche ×3.1, popular −0.00131 [−0.00162, −0.00099] (−7%).
- **Last-item feature alone is small:** +0.00013 [0.00002, 0.00024] overall (+1.5%), new +0.00017 [0.00008, 0.00026]. Candidate supply matters far more than adding the signal as an input.
- **Versus the original forced-list MLP (A, 3 seeds from the policy runs, same 232,904 events):** MLP hybrid +0.00293 [0.00267, 0.00320] (+42%); DIN hybrid +0.00383 [0.00357, 0.00410] (+55%); both positive in popular, niche and new groups (lowest CI bound: MLP hybrid niche +0.00086). Both pass the pre-registered numerical guard (3 seeds, overall CI > 0, no new-item loss). Caveats: designed after seeing earlier results (exploratory, not confirmatory); a pipeline change (retrieval pool and training population), not a ranker-only fix; A comes from separate runs evaluated on the same events; Val only, Test untouched.

### Our rankers on Brandon's mixed candidate list (Jing Yen, 8 November 2026)

- **Setup.**
  - Rebuilt Brandon's mixed list (P3): recency and two-tower retrievers alternate, skipping duplicates, until 200 products. Script: `scripts/build_p3_candidates.py`.
  - The rebuilt list contains the purchase for 8.8% of Val events, against Brandon's 8.0%. The recency retriever alone finds 6.4%, against his 5.5%.
  - MLP and DIN were trained on natural lists only, seeds 0/1/2, with the hybrid-study recipe. Script: `scripts/run_hybrid_study.py`.
  - The candidate score input is the two-tower score.
- **Analysis.** Seed-averaged per-purchase nDCG@10, per-user macro average, 1,000 paired user resamples. Script: `scripts/analysis_p3_groups.py`; outputs in `results/targeted_study/p3-groups-20261108/`. Validation only.
- **Levels (×10⁻³):**
  - MLP on the mixed list: 9.49. Popular 14.9, niche 3.38, new 4.94.
  - DIN on the mixed list: 10.64. Popular 16.0, niche 3.80, new 6.47.
  - Brandon's ranker (retrieval score + MLP correction) on his list: 11.0.
- **Mixed list vs two-tower list:**
  - MLP: +0.00097 [0.00071, 0.00123] (+11%).
  - DIN: +0.00069 [0.00043, 0.00095] (+7%).
  - Niche roughly ×4, new ×2.1, popular −5% for MLP and −11% for DIN.
- **DIN vs MLP on the mixed list:** +0.00115 [0.00094, 0.00135] (+12%). Positive in every group: popular +7%, niche +13%, new +31%.
- **Adding last-item products to the mixed list:** no detectable change.
  - MLP: +0.00002 [−0.00011, 0.00014].
  - DIN: −0.00001 [−0.00014, 0.00013].
- **DIN, mixed list vs two-tower + last-item list:** no detectable overall difference (−0.00008 [−0.00031, 0.00013]). The two-tower + last-item list wins on popular products; the mixed list wins on niche ones.

### Brandon's combined retrieval score and score-plus-correction design on our rankers (Jing Yen, 8 November 2026)

- **Setup.**
  - Same mixed (P3) lists as above, but each candidate carries Brandon's combined score: reciprocal-rank fusion over both retrievers, 1/(60 + recency rank) + 1/(60 + two-tower rank), using full-catalogue ranks. Built with `scripts/build_p3_candidates.py --rrf`, written to `p3rrf_seed0`.
  - Score standardisation is refitted on those training lists.
  - The `_res` arms score as a·(standardised retrieval score) + correction. The correction's last layer starts at zero and a starts at 1, so the untrained model reproduces the retrieval order. Option `residual_score` in `MLPRanker`/`DINRanker`.
  - Seeds 0/1/2, natural lists only, same recipe as above.
  - Analysis: `scripts/analysis_p3_groups.py`; outputs in `results/targeted_study/p3-rrf-groups-20261108/`. Validation only.
- **Levels (×10⁻³), overall / popular / niche / new:**

  | Ranker | Overall | Popular | Niche | New |
  |---|---:|---:|---:|---:|
  | DIN with the combined score as input | 11.24 | 16.0 | 4.72 | 7.63 |
  | DIN, score plus correction | 11.09 | 16.5 | 4.27 | 6.90 |
  | MLP with the combined score as input | 9.63 | 14.4 | 3.78 | 5.88 |
  | MLP, score plus correction | 9.34 | 14.3 | 3.77 | 5.31 |

  Brandon's ranker (score plus MLP correction, his features, on his P3): 11.0, seed mean; per-purchase results unavailable, so no CI.
- **Combined score vs two-tower score as input:**
  - DIN: +0.00061 [0.00043, 0.00081] (+6%). Niche +24%, new +18%, popular unchanged.
  - MLP: +0.00015 [−0.00004, 0.00034], not detectable.
- **DIN vs MLP, both with the combined score:** +0.00161 [0.00142, 0.00182] (+17%). Positive in every group: popular +11%, niche +25%, new +30%.
- **Score plus correction vs combined score as input:**
  - DIN: −0.00015 [−0.00027, −0.00003].
  - MLP: −0.00029 [−0.00040, −0.00020].
  - The design does not help our rankers; it loses mostly on new and niche products.
- **Best system vs our original MLP on two-tower lists:** DIN on the mixed list with the combined score scores +0.00272 [0.00242, 0.00304] (+32%). Niche ×5.8, new ×3.3, popular unchanged [−0.00007, 0.00087].

### New meso cuts and micro cases for the final systems (Jing Yen, 8 November 2026)

Scripts: `scripts/analysis_meso_segments_rrf.py` and `scripts/analysis_micro_cases_rrf.py`. Outputs: `results/targeted_study/meso-segments-rrf-20261108/` (`segments.csv`, `labels.parquet`, `micro_cases.csv`).

Systems compared: original = MLP on two-tower lists; best = DIN on mixed lists with the combined score. Labels come from titles and categories only. Validation only.

**Results (nDCG@10 ×10⁻³, original → best; difference with 95% interval):**

| Group | Share | Original → best | Difference |
|---|---:|---:|---:|
| Cases | 33% | 1.74 → 6.85 (×3.9) | +5.10 [4.68, 5.52] |
| Screen protectors | 18% | 28.1 → 26.1 | −1.98 [−2.91, −1.10] |
| Phones | 1% | 31.4 → 23.7 | −7.78 [−11.67, −3.95] |
| Same platform as history | 59% | 10.2 → 13.3 | |
| Platform switch | 11% | 2.5 → 3.6 | DIN vs MLP −0.14 [−0.45, 0.18] |
| Same iPhone model as history | 9% | 15.6 → 25.9 | +10.3 [8.95, 11.7] |
| Newer iPhone model than history | 14% | 7.7 → 10.4 | DIN vs MLP +0.15 [−0.30, 0.60] |
| iPhone 13 launch month (14 Sep to 14 Oct 2021) | 9% | 10.1 → 13.8 | |

- **Screen protectors and phones** are the easiest categories, and the only ones where the best system loses.
- **Platform switchers** are the hardest group. Coverage of the purchase in the 200 candidates is only 2–3%.
- **Upgraders to a newer iPhone model:** DIN shows no gain over MLP.
- **iPhone 13 launch month:** every system scores higher than in other weeks.

**Caveat: titles are current metadata, not as of purchase.** Some purchases before the launch have titles naming later models. For example, a case bought on 29 Sep 2021 is titled "iPhone 14 Pro", and a Galaxy S23 product appears in a 2021 history. Sellers update titles to add new models, so the model and platform labels, and the earlier launch-title count, are approximate.

### Micro event cohorts (Jing Yen, 8 November 2026)

Script: `scripts/analysis_micro_cohorts_rrf.py`. Output: `results/targeted_study/micro-cohorts-rrf-20261108/cohorts.csv`.

Cohorts are defined by purchase date plus a title pattern. Systems are seed-averaged: original (MLP on two-tower lists) and best (DIN on mixed lists with the combined score). Validation only.

**iPhone 13 items over time (nDCG@10 ×10⁻³, original → best):**

| Window | Purchases | nDCG@10 | Hit@10 |
|---|---:|---:|---:|
| Launch fortnight (14–30 Sep 2021) | 1,415 | 1.97 → 19.2 | 0.6% → 4.8% |
| Oct–Dec 2021 | 9,699 | 2.65 → 10.7 | |
| 2022 | 18,666 | 1.60 → 7.9 | |

- In the launch fortnight, the share of purchases found among the candidates goes from 4.9% to 12.6%.
- The best system's advantage shrinks as the launch ages.

**Other launches and product lines (nDCG@10 ×10⁻³, original → best):**

| Cohort | Purchases | nDCG@10 |
|---|---:|---:|
| Galaxy S22 launch | 3,110 | 4.2 → 9.7 |
| Pixel 6 | 4,391 | 1.4 → 8.0 |
| MagSafe accessories | 8,149 | 4.4 → 12.4 |
| iPhone SE 2022 | 1,585 | 5.7 → 12.3 |

**The exception: Apple Watch 41/45 mm bands at the Series 7 launch** (2,362 purchases). Here the best system is worse: 15.1 → 12.3, a difference of −2.79 [−5.09, −0.72].

**Title-edit check:** 770 purchases made before the iPhone 13 announcement (14 Sep 2021) have "iPhone 13" in the product title. This confirms that product titles are edited after purchase.

All cohort differences have 95% intervals excluding 0.

### User-behaviour failure modes (Jing Yen, 8 November 2026)

Script: `scripts/analysis_behaviour_failures.py`. Outputs: `results/targeted_study/behaviour-failures-20261108/`.

Each Val purchase is labelled by comparing the phone models its title mentions (all of them; titles often list several) with those in the user's last 50 purchases. Labels are approximate because titles are current metadata.

Failure types:
- **Retrieval miss:** the purchase is not in the 200 candidates.
- **Ranking miss:** the purchase is in the 200 but ranked below 10.
- **Hit:** the purchase is in the top 10.

Systems are seed-averaged: original = MLP on two-tower lists; best = DIN on mixed lists with the combined score.

| Behaviour | Share | Retrieval miss (orig → best) | Hit@10 (orig → best) | nDCG@10 ×10⁻³ | Best − orig, 95% interval |
|---|---:|---:|---:|---:|---:|
| Same device as history | 23% | 88.2% → 83.5% | 2.6% → 4.5% | 13.9 → 23.1 | [8.33, 10.07] |
| Upgrade (newer model, same family) | 20% | 94.5% → 94.1% | 1.3% → 1.8% | 6.4 → 8.7 | [1.77, 2.87] |
| Device-agnostic (cables, chargers, mounts) | 37% | 93.4% → 92.2% | 1.7% → 2.0% | 8.9 → 9.3 | [−0.09, 0.74] |
| Other model, same family | 5% | | | 9.1 → 11.4 | |
| Older model than history | 3% | | | 6.2 → 6.6 | not detectable |
| Other device, same platform | 4% | 98% | | ≈2.4 | |
| Platform switch | 4% | 98.5% | | 0.8 → 1.2 | not detectable |

**Dose-response with prior purchases for the same device (nDCG@10 ×10⁻³, original → best):**

| Prior purchases | nDCG@10 |
|---|---:|
| 0 | 7.0 → 8.2 |
| 1 | 12.5 → 22.0 |
| 2 | 16.1 → 27.6 |
| 3–4 | 18.7 → 29.2 |

Owning two or more devices over time (68% of purchases) makes little difference.

**Interpretation:**
- Retrieval misses dominate every behaviour: 83–98% of purchases never reach the ranker.
- The gains concentrate on repeat accessorising of a phone the user already owns.
- Upgrades, switches and device-agnostic purchases are barely helped.

### Single-query latency of the final systems (Jing Yen, 8 November 2026)

Script: `scripts/latency_final_systems.py`. Output: `results/latency/20261108-final-systems/latency.json`.

Setup:
- CPU Apple M1 Max, single thread, 50 warm-up + 1,000 Val queries, K₁ = 200, exact FAISS.
- Rankers are freshly initialised with the trained architectures. Timing does not depend on weight values.

Median (p95) total latency:

| Pipeline | Ranker | Median ms | p95 ms |
|---|---|---:|---:|
| Original | MLP | 4.0 | 4.4 |
| Original | DIN | 9.6 | 10.3 |
| Best | DIN | 32.9 | 38.8 |
| Best | MLP | 25.4 | 32.0 |

**Best + DIN, median per phase (ms):**

| Phase | Median ms |
|---|---:|
| User tower | 0.8 |
| Two-tower search and filter | 1.2 |
| Recency query and catalogue scan | 4.4 |
| Recency filter and top 200 | 1.4 |
| Merge and combined score | 15.9 |
| Ranker features | 1.3 |
| DIN forward | 6.9 |

The combined score dominates: this implementation sorts the full catalogue for both retrievers to get exact ranks. Ranks within each retriever's own top list, the usual reciprocal-rank-fusion practice, would remove most of it; this has not been measured. DIN costs about 5.5 ms more than the MLP for 200 candidates.

### Final Test result (Jing Yen, opened once 8 November 2026)

Setup:
- Scripts: `scripts/build_p3_candidates.py --splits test --rrf`, `scripts/score_test_final.py`, `scripts/analysis_test_final.py`.
- Outputs: `results/test_final_ranker/20261108/`.
- Predictions were written in advance in `predictions.md`. Frozen Val-selected checkpoints, seeds 0/1/2, no tuning on Test.
- Test has 297,390 purchases; 44% are new products.
- Share of purchases in the 200 candidates: 3.9% for two-tower lists, 6.4% for mixed lists.

**Test nDCG@10 ×10⁻³ (all / popular / niche / new):**

| System | All | Popular | Niche | New |
|---|---:|---:|---:|---:|
| MLP, two-tower lists (original) | 5.19 | 11.6 | 1.31 | 1.33 |
| DIN, two-tower lists | 6.37 | 14.2 | 1.13 | 2.16 |
| MLP, mixed lists + combined score | 7.31 | 11.0 | 3.58 | 6.87 |
| **DIN, mixed lists + combined score (best)** | **8.27** | 12.6 | 3.95 | 7.66 |
| DIN, score plus correction | 8.06 | 12.9 | 3.58 | 6.95 |

**Comparisons (95% intervals):**
- **Best vs original:** +0.00308 [0.00281, 0.00336] (+59%). New ×5.8 (+0.00634), niche ×3.0, popular +9%.
- **DIN vs MLP, both on mixed lists:** +0.00097 [0.00080, 0.00114] (+13%). Positive in every group.
- **Combined score as input vs score plus correction:** +0.00021 [0.00011, 0.00032].

**Predictions:**
1. Held.
2. Held.
3. Partly held. The mixed-list gain on new products is larger on Test than on Val (×3.5 vs ×2.5 for DIN), but on niche products it is smaller (×3.5 vs ×5).
4. Held.
5. Held. 8.27 is within +5% of Brandon's ranker on his mixed lists (D1, Test 7.9; no interval possible). His best system, which adds monthly statistics and co-review features, reaches 10.4 on Test.

### Upgrade hypothesis check (Jing Yen, 8 November 2026)

Script: `scripts/analysis_upgrader_candidates.py`. Output: `results/targeted_study/behaviour-failures-20261108/upgrader_candidates.csv`.

Method: for 2,000 random Val upgrade purchases, every candidate in each retriever's top 200 is classified by the phone models its title names.

| Source of the 200 candidates | Old phone | Purchased (new) model |
|---|---:|---:|
| Recency retriever | 37.2% | 3.1% |
| Two-tower retriever | 29.0% | 7.8% |
| Released catalogue at that time | 8.0% | 2.0% |

- The recency retriever's list is about 12 times richer in the old phone than in the new one, and 4.6 times richer in the old phone than the catalogue itself.
- The two-tower list carries 2.5 times more new-model candidates than the recency list.
- For comparison, same-device purchases get 28.6% target-model candidates from recency.
- This confirms the hypothesis: history-anchored retrieval fills the list with accessories for the phone the user already owns.

### Faster combined score: latency only (Jing Yen, 8 November 2026)

Using each candidate's position in each retriever's own top-200 list, instead of exact full-catalogue ranks, cuts the merge step from 15.5 ms to 0.24 ms. The best system's median latency falls from 31.8 ms to 17.6 ms (p95 28.9 ms).

Script: `scripts/latency_final_systems.py` (best_fast). The accuracy of this variant is not measured; the trained rankers saw exact ranks.

### Product-side archetypes (Jing Yen, 8 November 2026)

Script: `scripts/analysis_product_archetypes.py`. Output: `results/targeted_study/product-archetypes-20261108/`. Validation; original → best, seed-averaged.

| Product group | Share | nDCG@10 ×10⁻³ | Hit@10 | Retrieval miss | Best − orig, 95% interval |
|---|---:|---:|---:|---:|---:|
| Same store as user's history | 9% | 38.3 → 79.9 | 6.6% → 14.7% | 83% → 67% | |
| New store | 91% | 5.90 → 5.37 | | | [−0.72, −0.34] |
| Product under 30 days old | 10% | 2.4 → 10.7 | | | |
| Product over 2 years old | 28% | 21.8 → 20.8 | | | [−1.66, −0.34] |
| Stores with 1–9 products | 28% | 1.2 → 3.0 | | 97% → 95% | |
| Stores with 100–499 products | 18% | 24.0 → 23.9 | | | |
| Under $10 or over $50 | | | | | not detectable |

- **Same store as history:** the whole overall gain comes from these purchases. For new-store purchases the best system is slightly worse.
- **Product age:** gains shrink steadily as product age at purchase grows; products over 2 years old do slightly worse.
- **Stores with 1–9 products:** almost never retrieved.
- **Stores with 100–499 products:** already well served; no change.
- **Price:** no detectable change for products under $10 or over $50.

### Dissecting the retriever list and the ranker's top 10 (Jing Yen, 8 November 2026)

Scripts: `scripts/export_top10_lists.py` (cluster job 1095733), `scripts/analysis_list_dissection.py`. Output: `results/targeted_study/list-dissection-20261108/dissection.csv`.

Method:
- 20,000 random Val purchases (RNG 0); seed-0 original (MLP, two-tower lists) and best (DIN, mixed lists + combined score).
- Every candidate is classified by the phone models its title names, relative to the purchase:
  - **bought phone:** a model of the purchased product;
  - **owned phone:** a model in the user's history only;
  - **other phone;**
  - **no phone.**
- Labels are approximate (titles are current metadata).

**Results (list of 200 → top 10, %; best system unless stated):**

| Purchase type | Bought-phone share, list → top 10 | Top 10 holds ≥1 bought-phone product | Same product type in top 10 | Exact purchase in top 10 |
|---|---:|---:|---:|---:|
| Same phone as before | 26 → 33 (original 19 → 24) | 81% (original 72%) | 35% | 4.6% |
| Upgrade | 6 → 9 | 48% | 27% | 2.0% |
| Device-agnostic | — | — | 22% | 1.8% |
| Platform switch | | 16% | | 0.4% |

**Notes:**
- **Same phone as before:** the right phone is usually there, but the exact product rarely is.
- **Upgrade:** owned-phone products fill 30% of the top 10 and 36% of the list. The two-tower list carries more new-phone candidates than the mixed list: 8.7% vs 6.2%.
- **Device-agnostic:** 29% of the top 10 are accessories for the user's own phone, and 40% name no phone.
- **Platform switch:** the top 10 is for the old platform or other phones.
- **The ranker sharpens toward the bought phone** for same-phone and upgrade purchases. It cannot add products the list lacks.

### Faster combined score: accuracy (Jing Yen, 8 November 2026)

- **Method:** DIN on mixed lists with the combined score computed from positions within each retriever's own top-200 list (absent = 201), instead of exact full-catalogue ranks (`din_rrfl`). Trained and selected on Validation like `din_rrf`, seeds 0/1/2, then scored once on Test. Prediction 6 was written before training.
- **Validation:** 11.11 vs 11.24 (×10⁻³).
- **Test:** 8.23 vs 8.27, difference −0.00004 [−0.00012, +0.00005]. No detectable loss; prediction 6 held.
- **Latency:** median 17.6 ms vs 31.8 ms.
- **Where:** cluster jobs 1096571–5; outputs in `results/test_final_ranker/20261108/din_rrfl_*`.
