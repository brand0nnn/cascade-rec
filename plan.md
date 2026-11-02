# Local experimentation plan

## Current status — local plan complete, 2 November 2026

**The authorised local programme and its evidence/reporting checklist are complete.** Sections 2–5 retain the original protocol; §§6–9 record completed work and approved extensions. The consolidated deliverable is `results/local_experiments/20261102-plan-completion/report-final/report.md`, with its completion audit and evidence bundle. New experiment ideas and the explicit later decisions below are outside this completed local programme.

| Original programme / approved extension | Status |
|---|---|
| Original RecBole weights, strict-negative ID/content/content+ID controls and popularity | Complete; standalone weights/mappings and audits saved |
| E1 multiple interests; E2 metadata neighbours; E3 mean/attention rankers; E4 actual candidate fusion | Three-seed runs and complete comparisons done; fusion improves graph but not clearly content, while E1–E3 show no clear improvement over their matched controls |
| History truncation, uniform/fixed-recency frozen text and twenty fixed query examples | Complete for the original sample comparison |
| Full archived-Val monthly/new-warm/category/store/history audit | Complete, including monthly history intersections and clipped catalogue-introduction counts |
| Full-catalogue frozen-text validation | Complete; no recommender training |
| Frozen signal diagnosis and subsequently approved candidate-normalised training | Complete; see §§7–8, including the positive normalisation result |

### Completed evidence checklist

- [x] **Temporal/history reporting:** all ten local retrieval methods have monthly/history/new-warm metrics and paired intervals; the full archived retrieval/MLP/DIN audit has the same intersections. Support partitions reproduce all 4,825 sample / 232,904 full events, and prior full monthly nDCG estimates reproduce within 1e-12. Sample/full tables remain separate, with partial-month/seasonality caveats.
- [x] **Local ranking failure analysis:** per-seed and subgroup retrieval misses, ranking misses, hits, rescues, demotions and retained hits, plus all event transitions. Funnel partitions and rescue/demotion identities pass exactly.
- [x] **E4's two-source comparison:** paired fusion-minus-content **and fusion-minus-graph** intervals from actual candidate IDs/scores under a unique 200-item budget. Positive graph-only gains and inconclusive content gains are both retained.
- [x] **Score evidence required by §4:** six sample/full frozen-control top-200 archives and twelve selected-ranker logit archives (both periods, two models, three seeds) recovered from frozen models. Every regenerated Val rank and native top-ten list reproduces exactly. Tie/display differences are explicitly recorded, and previous artifacts remain intact.
- [x] **Consolidated local comparison:** one final report includes all experiments, the approved normalisation follow-up, uncertainty, original baseline references, temporal/support/composition/concentration tables, twenty fixed examples, figures, costs and source/hash audits. See §9.

### Decisions after that completion

- The suggested pooling comparison on the normalised model and a geometry-preserving encoder are **additional proposed experiments**, not unfinished E1–E4 runs. Design/approve them explicitly before changing training again.
- Nominate at most two variants for any full-data learned-model fits; this remains a separate decision under §5.
- Nomination is recorded in §9: a matched native content control and candidate-normalised treatment, with frozen recency as the existing full-data reference. No full-data learned fit has been launched.
- Ranking on a changed candidate pool, final pipeline selection, serving/cascade timing and held-out Test are later work. The sample rerankers used the original content-seed-0 pool; improved retrieval has not automatically supplied new ranking results.

## 1. Goal and working approach

Create an independent local experiment track, starting from a simple baseline we understand. Test how changes to retrieval, history modelling, graph information and ranking alter recommendations, especially for future products.

Our contribution will combine **full-data analysis of existing predictions** with **new small-scale model experiments**. Small-data results establish feasibility and interesting behaviour; they will not be presented as improvements over the existing full-catalogue results.

The root `plan.md` is the local experiment record. Keep the teammates’ existing models, configurations and experiments intact. Use the original RecBole SASRec as the first reference, with fresh local weights rather than a new architecture as our initial baseline.

### Completed baseline preparation — 2 November 2026

The original `SASRecBCE` seed-0 small-data runs are complete, trained on MPS with the unchanged original early-stopping recipe. The retriever-period run selected epoch **8**, and the all-training run selected epoch **13** (one-based). Both weight files reload strictly and match their selected RecBole checkpoint on 32 validation histories across the 30,389-item catalogue, with zero score difference. Held-out Test was not evaluated.

- Matched-period reference: `results/baselines/local/20261102-sasrec-retriever-sample-seed0/model.pth`.
- All-training reference: `results/baselines/local/20261102-sasrec-all-train-sample-seed0/model.pth`.
- Each run also includes `id_mappings.npz`, `config.yaml`, `provenance.json`, `weights_verification.json`, validation ranks and its original RecBole checkpoint. The mappings are required because RecBole's internal item IDs differ from processed-data IDs.
- These are fresh sample-model weights. No full-data checkpoint was recovered or trained. The matched content extension and its frozen text vectors are now complete; see §6.

## 2. Establish our own baseline

### Prepare isolated inputs

- Use the existing 7,619-user sample, containing 30,389 products and 4,825 validation events, as the initial development population.
- Copy its structure into a fresh study directory. Preserve item/user mappings and histories.
- For original sample weight recovery, retain its existing **T0=1596679591746**, **T1=1628643414042**, **T2=1658002729837**. All subsequent sample controls must use those same boundaries. Full-data experiments use canonical **T0=1595800947930**; sample and full-data outcomes are not interchangeable. Do not relabel the original sample while calling it a baseline reproduction.
- Original ID-based SASRec needs no text vectors and can run immediately. Recreate MiniLM vectors only for later content variants; record encoder revision, text hashes and embedding hashes. Newly generated vectors are not claimed reproductions of historical inputs.
- Prioritise Apple MPS for training, with CPU export and final evaluation. Verify GPU access with a real tensor operation. Benchmark one epoch before estimating larger runs; report unsupported operations explicitly rather than silently moving all training to CPU.

### Original RecBole retrieval baseline

First train the existing `SASRecBCE` recipe on the small dataset, seed 0:

- RecBole 1.2.1 SASRec, 2 transformer blocks, 1 head, hidden/FFN width 64, last-50 histories, dropout 0.2.
- Original BCE override with one uniform negative per prefix–target row; Adam at 0.001, training batches of 512, max 200 epochs, early-stopping patience 5 on RecBole validation Recall@100.
- Run one technical-smoke epoch in its own output directory, then complete fresh small-data runs for `retriever_train` and `all_train`. The former is the reference for matched-period content/retrieval experiments; the latter preserves the stronger original baseline setting.
- Use MPS for model training. Evaluate the selected checkpoint through the project harness on validation only, including every eligible event.
- Save standalone selected weights, RecBole's checkpoint, item/user token mappings, input/source hashes, selected epoch and config. Reload the standalone state and verify scores against the selected checkpoint before declaring weights reusable.
- Use unique local run IDs; never overwrite historical results or reuse cached exports with another split. These are new sample fits, not recovered historical or full-data weights.

### Approved future-item control — 2 November 2026

Use **pre-T0-only training negatives for both fresh ID and content controls**, as selected after the sampler review. The original RecBole sampler draws from the complete item vocabulary; exposing later products' content as negatives would test a known-catalogue setting rather than an unseen-product setting. The sample restricted pool contains **20,740 items**, excluding **9,649** later-observed products. Preserve the original checkpoints as historical references. Keep the sequence backbone, BCE, exported training pairs, user-positive exclusions, selector and full eligible validation catalogue matched. Record the allowed pool, actual negative samples and matching backbone initialisation; later-product content must not affect training gradients. Precomputed future vectors may only be used for no-gradient validation/inference.

### Controlled content extension

After the original baseline works, replace SASRec's learned item-ID lookup with a shared content projection from frozen MiniLM 384 → 256 → 64. Keep its sequence backbone, dot-product scoring, original BCE loss, training pairs and selector matched. This is a new explicit variant, not a stock RecBole model. It allows future items to obtain vectors from content without an item-specific learned parameter. Add a content+ID variant only after the content-only comparison is working; unseen items receive no ID residual.

Use the completed content-only model as the matched reference for the graph and multiple-interest experiments below. Preserve the ID-based RecBole model as an independent anchor. Include training-period popularity as an inexpensive control.

### Ranking baseline

Generate our own natural top-200 candidate lists from the completed baseline retriever.

Train a small content-based MLP using candidate text, mean history text and known category/store relationships. Use shared 32-dimensional text projections and a **128 → 64 → 1** scoring head.

- Train only on lists where the target was naturally retrieved.
- Never append a missed target or supply a target-only feature.
- Use BCE with 50 uniformly sampled alternatives per positive, Adam at 0.001, and batches of 32 lists.
- Run at most 10 epochs with patience 2, selecting by natural-hit conditional user-macro nDCG@10.
- Report excluded training support and end-to-end validation performance, including retrieval misses.
- Preserve retrieval order as a ranking control.

## 3. Experiment programme

Run individual changes against their matched baseline before combining them.

| Experiment | Fixed implementation | Hypothesis and comparison |
|---|---|---|
| **E1: Multiple interests** | Replace the content-SASRec last-state summary with two learned, padding-masked attention pools over its causal sequence states. Train and score using the maximum candidate similarity across the two interests; training pools must remain causal for every prediction position. | Earlier interests may survive better than in one averaged vector. Compare overall retrieval, returning-category targets, mixed histories and top-10 changes. Record head usage and similarity to detect collapse. |
| **E2: Content-and-metadata graph** | Extend the item encoder with separate mean-content representations of same-leaf and same-store neighbours. Use up to 16 deterministic, seeded neighbours per relation, drawn exclusively from products seen before T0. Concatenate self-content and neighbour representations, then project to 64 dimensions. Empty relations contribute zero with explicit availability flags. | Metadata relationships may improve representations of new products without requiring their own review edges or learned item IDs. Compare against the content-only retriever on identical populations. |
| **E3: Semantic attention ranker** | Replace the ranking baseline’s mean history pooling with candidate-conditioned attention over projected history text. Keep candidate inputs, metadata features, training lists, loss and scoring head matched. Label this a DIN-inspired adaptation, not a reproduction of the full DIN architecture. | Candidate-specific selection may help returning interests and heterogeneous histories. Compare with the mean-pooling ranker on the same baseline candidate lists. |
| **E4: Actual candidate fusion** | Combine E2 and baseline retrieval lists using equal-weight reciprocal-rank fusion with constant 60. Deduplicate, apply the shared filters and retain exactly 200 candidates where available. | Different representations may retrieve complementary products. Measure actual merged-list coverage rather than inferring gains from target groups or separate target ranks. |

For E2, new products may connect to the frozen training-product reference through their metadata. Training representations must never aggregate future-product neighbours.

For E4, compare against both constituent retrievers at the same total candidate budget. Ranking a changed pool is a separate follow-up; do not claim a complete improved cascade from retrieval coverage alone.

Also run inexpensive frozen-model probes:

- Last **5 / 10 / 50** history items.
- A separate content-history mean versus exponentially weighted mean with a **90-day half-life**; this pooling probe is not the original SASRec baseline.
- History truncation is an inference intervention on SASRec; content pooling is an additional reference. Report both separately from retrained models.

Use seed 0 for initial pilots. Complete seeds 1 and 2 for the baseline and technically successful model variants before treating directions as stable. Report incomplete or negative experiments rather than quietly dropping them.

## 4. Analyse recommendations and time effects

### Full-data analysis available immediately

Using existing aligned validation predictions, produce monthly tables for:

- Events, users, newly introduced products and category/store mix.
- Retrieval coverage and retrieval/MLP/DIN nDCG@10 and Recall@10.
- Model differences within new/warm and available history groups.

Annotate the September 2021 phone-launch period and November–December shopping period as exploratory context. Report the partial boundary months. Review timestamps are imperfect proxies for purchasing dates; one validation year cannot establish recurring seasonality.

Do not load new Test predictions or change the canonical split.

### Evidence saved for every local experiment

Save aligned target ranks, actual candidate IDs and scores, top-10 lists, configuration, seeds, checkpoints, input hashes and runtime.

Report:

- Recall@200, nDCG@10 and Recall@10, including all retrieval misses.
- New-in-period versus warm targets, unseen-before-T0 targets, category continuity and monthly performance.
- Ranking rescues and demotions where applicable.
- Top-10 overlap, category/store concentration and catalogue coverage. Diversity measures describe list composition, not known relevance.
- Paired user-clustered 95% bootstrap intervals: **1,000 resamples, seed 0**. Average event metrics across completed training seeds before bootstrapping.

Choose **20 validation query IDs**, sorted then sampled with seed 0, before inspecting experiment outputs. Reuse them for recommendation comparisons; retain unchanged lists as well as striking changes.

Evaluate future-item performance with frozen weights and permitted catalogue availability. Record unchanged parameter hashes across temporal evaluation. Distinguish encoding/index updates from retraining, and disclose that catalogue text is a crawl-time snapshot.

## 5. Implementation, checks and progression

Add an isolated entry point, `python -m cascade.pipelines.local_experiments`, with phases **prepare, baseline, pilot, report** and explicit study ID, experiment, seed and device options. Store outputs under `results/local_experiments/<study>/`; reject existing run destinations.

Reuse the shared filtering, metric and bootstrap code. New model constructors must be opt-in and must preserve existing checkpoint loading and teammate workflows.

Required checks:

- Strict temporal boundaries and causal histories, including graph neighbours.
- No target insertion; no label-dependent features; misses receive zero end-to-end credit.
- Missing categories/stores and padding never create semantic matches.
- Two-interest scoring matches a hand-computed example; padded histories remain finite.
- Fusion produces unique, filtered candidates within the shared budget.
- Attention and mean-pooling rankers receive identical information and candidate populations.
- Saved ranks and recommendation lists align with recorded event IDs.
- Future-item encoding leaves learned weights unchanged.

Run functionality checks first, then small-scale training and replication. Produce one comparison report containing results, uncertainty, recommendation examples, failure cases and measured scaling costs.

**Full-data training is a subsequent decision:** nominate at most two variants based on stable behaviour, scientific interest and measured feasibility. No automatic full-data launch follows a good pilot. Existing frozen-insertion diagnostics remain supporting evidence; this track’s main deliverable is a controlled account of how our own design changes affect recommendations.

## 6. Completed local experiments — 2 November 2026

Study: `results/local_experiments/20261102-sasrec-pre-t0/`. All **15 retriever fits** (ID, content, content+ID, graph and two-interest × seeds 0/1/2) and **six ranker fits** (mean and semantic attention × seeds 0/1/2) completed on MPS. Technical smoke runs are separate. Standalone weights reload with identical scores. The pinned frozen MiniLM revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; its vectors are new artifacts, not recovered historical embeddings.

| Retrieval method | Recall@200 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| ID SASRec, strict negative pool | 0.023005 | 0.002401 | 0.001298 |
| Content SASRec | 0.056750 | 0.003862 | 0.001836 |
| Content + warm-item ID residual | 0.043602 | 0.004994 | 0.002225 |
| Content + metadata neighbours | 0.053690 | 0.003040 | 0.001248 |
| Two interests | 0.052636 | 0.003269 | 0.001727 |
| Actual content/graph RRF fusion | 0.057858 | 0.004006 | 0.001931 |
| Pre-T0 popularity | 0.022277 | 0.002494 | 0.000842 |
| Frozen text mean | 0.071513 | 0.013313 | 0.006961 |
| Frozen text, fixed 90-day half-life | **0.100990** | **0.022309** | **0.012489** |

These are retrieval outputs: top-ten quality uses the retriever's own score order, without a reranker. Learned entries average three seeds; popularity and frozen text controls have no task-specific training. Content improves Recall@200 over matched ID by **+0.033746 [0.026206, 0.041296]**, but its nDCG@10 difference is inconclusive. Content+ID reduces Recall@200 relative to content; its top-ten gain is inconclusive. Graph, two-interest and fusion show no clear gain over content. Intervals are unadjusted paired-user 95% bootstrap intervals, B=1000, seed 0, on seed-mean event metrics.

The 90-day frozen-text control beats uniform frozen text by **+0.005527 [0.003225, 0.008073] nDCG@10** and **+0.029476 [0.021533, 0.038459] Recall@200**. Its half-life was fixed in this plan before pilot outcomes. An independent CPU float64 reconstruction on 100 fixed queries matches top-10/top-200 hit outcomes; one target rank differs by one due to numerical precision. Histories were strictly prior and excluded the target. This sample has **3,240/4,825 new-in-Val events (67.2%)**, so numerical gains cannot be projected onto full data.

Natural-only rankers trained on **458/6,306 lists**, excluding 5,848 retrieval misses. End-to-end nDCG@10 is approximately **0.000827 mean / 0.000858 attention**, versus **0.001557** from the fixed content-seed-0 retrieval order. Neither improves clearly. The attention-minus-mean estimate is tiny; it does not establish scientific value for a more complex ranker. History-5/10 interventions are inconclusive. The two learned interest heads have mean cosine **0.958**; both win slots, but distinct semantic interests are not established.

Completed evidence:

- `summary-final/report.md`, `retrieval_comparison.csv`, `ranker_groups.csv`, `list_composition.csv`, `frozen_probes.csv`, `cases.md` and `cases.json`: one combined comparison, groups and the 20 queries fixed before outcomes.
- `final_audit.json`: matching backbone initialisation/initial negative traces, strict pool, portable weights/source snapshots, aligned archives, graph temporal checks and unique/available/history-free candidate lists all pass.
- `runs/<arm>/seed<seed>/model.pth` plus mappings/provenance; `rankers/<arm>/seed<seed>/model.pth`; actual top-200 candidate IDs/scores under `retrieved/` and `fusion/`.
- `results/local_experiments/20261102-full-val-calendar/`: all 232,904 archived Val events partitioned into 12 UTC months, with new/warm groups and category/store mix. August 2021/July 2022 are partial. Ageing, turnover and annual events remain confounded; no seasonal causal claim.

Validation: **154 passed, 4 skipped** after the sample additions. Four legacy candidate tests require the conventional sample embedding path; the isolated experiment vectors live inside the study. Generated weights, data and reports remain ignored by Git. No new Test evaluation or full-data training.

### Completed full-catalogue frozen-control check

The user resumed after the sample report. **Frozen text mean versus the unchanged 90-day mean** completed on all 232,904 Val events / 111,480 products, with canonical T0 and the same frozen encoder. Pre-T0 popularity and existing three-seed two-tower ranks are references. Entry point: `cascade.pipelines.frozen_text_study`; output `results/local_experiments/20261102-full-frozen-text/report-final/`. No recommender training or half-life tuning. The full Val population overlaps pilot users; this is follow-up validation, not independent untouched holdout evidence.

| Full retrieval | Recall@200 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| Pre-T0 popularity | 0.026985 | 0.004155 | 0.001655 |
| Frozen text mean | 0.034183 | 0.005675 | 0.002920 |
| Frozen text, 90-day recency | 0.054746 | **0.011506** | **0.006284** |
| Archived two-tower text, three seeds | **0.068630** | 0.010550 | 0.005695 |

Recency minus archive has nDCG@10 **+0.000589 [0.000166, 0.000992]** but Recall@200 **-0.013883 [-0.015333, -0.012485]**. New-in-Val nDCG@10 is **0.007611 versus 0.000352**; warm-in-Val is **0.005606 versus 0.008098**. The archive differs in objective, architecture and encoder artifacts; this does not causally isolate recency within the trained model. New/warm groups use the true target and cannot directly define a deployable router. Fusion would need actual lists and shared-budget evaluation.

The 1,024-query benchmark completed; all 100 independently reconstructed float64 target ranks match exactly. Full audits pass strict histories, target exclusion, saved list filters, input/source/vector/rank hashes and unchanged encoder weights. Encoding took 84.6 s on MPS; offline evaluation took 22.1/40.2/41.4 s for popularity/uniform/recency. Final suite: **156 passed, 4 skipped**. The partial first report remains under `report/` after an example-export fix; use `report-final/`. Predictions were unaffected.

Reproduction phases (fresh study for preparation; installed encoder pins must match the recorded versions):

```bash
python -m cascade.pipelines.frozen_text_study --study YOUR_FRESH_STUDY --phase prepare --device mps
python -m cascade.pipelines.frozen_text_study --study YOUR_FRESH_STUDY --phase evaluate --device mps
python -m cascade.pipelines.frozen_text_study --study YOUR_FRESH_STUDY --phase report
```

Full-data model training remains a later decision. Do not select a new architecture or retrain teammates' models merely because this frozen-control check completes.

## 7. Frozen diagnosis of lost new-item signal

The user chose this next direction after the full-catalogue check. Use the saved **sample content-only seeds 0/1/2**, preserving every weight and the original Val population. Entry point: `cascade.pipelines.content_signal_diagnostic`. No training or checkpoint/half-life selection.

Before viewing outcomes, fix these substitutions:

- Recreate the initial 384→256→64 projection from its deterministic seed; compare initial versus final projected text under identical 90-day pooling and cosine scoring.
- Pool final projected item vectors with uniform/90-day weights, under dot/cosine scoring. Separately normalise each history item before pooling. These isolate candidate norm effects and history weighting in the frozen functions.
- Compare the original sequence output's dot order with cosine order. Compare its order with direct pooling of the same final item vectors. Bypassing the sequence model changes its inference representation/distribution; it does not identify the causal effect of training that component.
- Measure norm/popularity associations, random-pair cosine and centred covariance effective rank in fixed 4,096-item samples before/after T0. Compare 20-neighbour preservation for fixed 256 pre-T0 / 256 post-T0 probes, with pre-T0 references only and self excluded.
- Retain all probes, negative outcomes, new/warm and seen/unseen groups, paired-user intervals (1,000 draws, seed 0 after seed averaging), recommendation composition and the original twenty fixed query examples. Geometry/slot counts describe representations and lists, not known relevance of unobserved alternatives.

Fresh output: `results/local_experiments/20261102-sasrec-pre-t0/diagnostics/20261102-frozen-decomposition/`. Freeze the probe manifest before scoring; verify source/input hashes and unchanged model parameters. Any new objective or retraining remedy follows a separate discussion informed by these findings.

### Completed diagnosis

All 21 new frozen probe evaluations completed on MPS across the three saved seeds. No weights, checkpoints, loss functions or selectors changed. Native target ranks and top-ten lists reconstruct exactly on 128 queries per seed; all 21 saved archives pass identity, availability, uniqueness and full-history checks. Source snapshots, input hashes and unchanged model parameters pass. Final suite: **159 passed, 4 skipped**.

| Frozen function | Overall nDCG@10 | New-in-Val nDCG@10 |
|---|---:|---:|
| Raw text + recency/cosine | 0.012489 | 0.013470 |
| Initial projection + recency/cosine | 0.009796 | 0.010108 |
| Trained projection + recency/cosine | 0.006607 | 0.007400 |
| Trained projection + recency/dot | 0.001310 | 0.001172 |
| Trained sequence/dot | 0.001836 | 0.001112 |
| Trained sequence/cosine | 0.002575 | 0.001396 |

Findings:

- **Projection changes matter.** Under identical recency/cosine pooling, final-minus-initial projection nDCG@10 is **-0.003190 [-0.005300, -0.001180]**, and **-0.002708 [-0.005145, -0.000342]** for new-in-Val targets. Initial compression already loses signal relative to raw text. Post-T0 probe overlap with raw top-20 neighbours declines from **42.3% initial to 16.6% trained**; this is preservation of MiniLM neighbourhoods, not a ground-truth relevance metric. Centred effective rank in the 64-dimensional pre-T0 space falls from approximately **38.1 to 10.5**. This is a concentration of variation; random-pair cosines do not support a claim that all item directions collapse together.
- **Candidate magnitude changes rankings.** Holding projected recency pooling fixed, cosine-minus-dot nDCG@10 is **+0.005296 [0.003596, 0.007206]**. The top 1% of catalogue products by norm occupy **17.0% of dot slots versus 0.25% of cosine slots**. Dot lists cover approximately **3,398** distinct products versus **15,483** with cosine. Norm/review-count Spearman is only about **-0.05**, so a popularity-bias cause is not established.
- **The sequence model contributes useful signal within its own space.** Sequence/dot beats uniform projected-mean/dot by **+0.001053 [0.000280, 0.001930] nDCG@10**. Sequence/cosine versus projected-mean/cosine is inconclusive. Thus the overall raw-text gap cannot simply be assigned to the transformer.
- **A scoring-only fix is incomplete.** Native sequence cosine improves overall nDCG@10 by **+0.000739 [0.000049, 0.001464]**, but its new-in-Val difference **+0.000284 [-0.000293, 0.000834]** is inconclusive. Normalising history items before recency pooling also has no clear gain. These are frozen inference substitutions, not retrained model results.

Artifacts: `report/report.md`, `report/paired_differences.csv`, `report/metrics.csv`, `report/composition.csv`, `report/cases.json`, `vector_geometry.csv`, `neighbour_preservation.csv`, `audit/verification.json`, `audit/candidate_norm_slots.csv` and `figures/frozen_diagnosis.png`.

**Control proposed after diagnosis (subsequently approved and completed in §8):** train the same sample content-SASRec with `logit(q, item) = q · (e_item / max(||e_item||, 1e-12))` in BCE, native prediction/selection and shared final scoring. Keep the query vector unnormalised, history inputs unchanged and no new temperature parameter. Match sampler, seeds 0/1/2, backbone, pairs and selector against the existing control; begin with a separate one-epoch smoke. Compare both with native dot-trained scoring and its existing frozen cosine probe. This tests training/scoring consistency around candidate magnitude; it does not isolate the reason projection geometry changed. A geometry-preserving objective or frozen item encoder would be a separate experiment. The frozen diagnostics cannot attribute the changes specifically to BCE versus checkpoint selection or optimisation.

## 8. Approved matched candidate-normalisation pilot

After the frozen diagnosis, the user explicitly chose **run the matched sample experiment**, seeds 0/1/2. The §7 proposal is now approved for the sample. Opt-in arm `content_unit` uses `UnitCandidateSASRec`, with candidate normalisation applied in BCE, native `predict`/`full_sort_predict`, and `LocalSASRecScorer`. No new parameters; initial backbone/history forward and parameter keys match the original content model. Future-item features cannot affect loss/gradients on prior-only training pairs. These properties pass targeted tests before the MPS run.

The separate one-epoch smoke and all three full sample seeds completed on MPS. Outputs: `runs/content_unit/seed<seed>/`, `smoke/content_unit/seed0/`. `cascade.pipelines.unit_candidate_study` compares with the original dot-trained model and its frozen cosine probe, retaining raw recency as a reference. Matched native configs, atomic exports, mappings, initial backbones/negative traces, portable weights, temporal filters and unchanged parameters during analysis all pass. New geometry/neighbour diagnostics reuse the fixed query/item samples. No full-data learned-model training or Test evaluation.

| Sample method | Recall@200 | Recall@10 | nDCG@10 | New-in-Val nDCG@10 |
|---|---:|---:|---:|---:|
| Native dot-trained/scored content model | 0.056750 | 0.003862 | 0.001836 | 0.001112 |
| Same weights, frozen cosine scoring | 0.060244 | 0.005415 | 0.002575 | 0.001396 |
| Candidate-normalised training/scoring | **0.075097** | **0.008298** | **0.003978** | **0.002548** |
| Frozen raw-text recency reference | 0.100990 | 0.022309 | 0.012489 | 0.013470 |

New training minus native dot: overall nDCG@10 **+0.002142 [0.000814, 0.003608]**, Recall@200 **+0.018347 [0.011714, 0.024630]**; new-in-Val nDCG@10 **+0.001436 [0.000356, 0.002610]**, Recall@200 **+0.015421 [0.008113, 0.022083]**. It also beats the frozen cosine probe: overall nDCG@10 **+0.001403 [0.000153, 0.002757]**, new-in-Val **+0.001152 [0.000050, 0.002325]**. Intervals are exploratory/unadjusted paired-user 95% intervals, B=1000, seed 0, after event-metric seed averaging. The new-item interval versus frozen cosine has a small lower bound; retain the exploratory framing.

Post-T0 raw-neighbour preservation is **26.2%**, versus **16.6%** in the original trained projection and **42.3%** initially. Pre-T0 centred effective rank is approximately **17.4**, versus **10.5** in the original trained projection. These are descriptive geometry checks, not relevance/causal mediators. Training/selection normalisation is a controlled intervention on the pipeline; it does not isolate optimisation from selection or prove geometry is the mediator. Selected epochs are **33/38/24**, versus **12/14/24** originally (one-based); training took **77.0/86.6/58.7 s**, versus **32.4/37.2/55.9 s**. Do not claim equal update counts.

The remaining gap to frozen recency is substantial: overall nDCG@10 **-0.008511 [-0.011727, -0.005760]**, new-in-Val **-0.010922 [-0.014650, -0.007698]**. Thus candidate normalisation is useful sample evidence, not a complete cold-item remedy or a reason to launch full-data training automatically.

`unit_candidate_report/` contains the report, group metrics/CIs, geometry, runtimes, audit provenance and `candidate_audit.json`. Actual top-200 lists are saved under `retrieved/content_unit/seed<seed>/`; all are unique, available, history-free and aligned. Target membership matches pessimistic Recall@200 with zero boundary differences in all three seeds. Latest suite: **162 passed, 4 skipped**. Total strict-pool retrieval fits now 18, plus six local ranker fits; historical original-sampler fits remain separate.

Reproduce with fresh output destinations (the listed destinations already exist):

```bash
python -m cascade.pipelines.local_experiments --study STUDY --phase pilot --experiment content_unit --seed 0 --device mps --technical-smoke
python -m cascade.pipelines.local_experiments --study STUDY --phase pilot --experiment content_unit --seed 0 --device mps
# Repeat full fits with seeds 1 and 2, then assemble the matched report:
python -m cascade.pipelines.unit_candidate_study --study STUDY --device mps
```

The study must contain its matched controls and frozen diagnostic reference. Candidate export additionally uses `local_followups.candidate_outputs(..., arms=('content_unit',))` to preserve older exported lists.

## 9. Final evidence and report completion

Completed the five outstanding evidence tasks using only frozen selected models and saved validation artifacts. No new training, Test predictions, changes to splits, or overwriting original results.

Final bundle: **`results/local_experiments/20261102-plan-completion/`**. Use **`report-final/report.md`** as the consolidated report; `report/` is the earlier review version. The final bundle contains:

- `analysis/sample_monthly_history_metrics.csv`: **5,400 rows**, ten local retrieval methods × monthly/history/new-warm strata × three metrics, with paired intervals versus content. `full_monthly_history_metrics.csv`: **1,620 rows**, extending archived retrieval/MLP/DIN with the same history groups. Both support tables have 180 strata and exactly cover their populations under the stated partitions.
- `analysis/ranker_failure_funnel.csv` and `ranker_event_transitions.parquet`: 4,568 retrieval misses and 257 natural Val hits. Retrieval order has 14 top-ten hits. Each ranker demotes 13 on average; attention rescues 9, mean 8.67. Counts are per-seed averages, not independent unique-event totals. All funnel identities pass.
- `analysis/fusion_actual_budget_comparison.csv`: fusion-minus-graph Recall@200 **+0.004167 [0.000728, 0.007860]**, nDCG@10 **+0.000682 [0.000112, 0.001318]**. Fusion-minus-content remains inconclusive: Recall@200 **+0.001107 [-0.002628, 0.004783]**, nDCG@10 **+0.000095 [-0.000461, 0.000645]**.
- `exports/sample_raw/`, `exports/full_raw/` and `exports/ranker_logits/`: six raw top-200 archives and twelve full-logit archives. Every regenerated Val target rank and native top-ten display is exact. Raw popularity has 12 sample / 17 full membership-versus-pessimistic-boundary differences; native headline metrics are retained. topk(200)/topk(10) tie-order differences are recorded separately. All model parameters remain unchanged.
- `report-final/`: the full report, all twenty fixed cases with every retrieval method and both rerankers, composition including category/store HHI and dominant shares with missingness/support, catalogue-introduction and target-mix tables, costs, source comparison tables, two figures and `completion_audit.json`. Catalogue introductions are clipped to the observed Val portions of boundary months, excluding later Test-period introductions. Earlier baseline/diagnostic/normalisation/full-reference estimates are preserved and reconciled.

Normalisation's overall gain is heterogeneous: nDCG@10 change versus content is **+0.003624 [0.001615, 0.005795]** for history 3–5, **+0.001987 [0.000171, 0.004270]** for 11+, and inconclusive for 1–2 / 6–10. These are exploratory unadjusted subgroup results, not universal gains. Calendar, ageing, novelty and history mix remain confounded.

Latest validation: **168 passed, 4 skipped**. The four legacy skips require conventional-path sample embeddings; isolated study vectors and all completed score replays are separately verified. Figures were visually checked. `completion_audit.json` binds generated outputs and their source evidence; all seven completion checks pass.

Reproduce into a fresh output name in the project environment:

```bash
python -m cascade.pipelines.local_evidence_export --output FRESH_COMPLETION --device mps
python -m cascade.pipelines.local_plan_completion --output FRESH_COMPLETION
python -m cascade.pipelines.local_completion_report --output FRESH_COMPLETION --report-name report-final
```

Existing frozen study paths are defaults; every export/analysis/report destination refuses existing output. Generated arrays, figures and reports remain local/ignored; code and compact findings are committed.

**Later full-data nomination:** native content-SASRec control and candidate-normalised treatment, as one matched pair of learned configurations. Retain fixed-recency frozen text as the already-evaluated full-data reference. The current graph/interest/ranker evidence does not justify additional full-data fits. Execution, changed-pool ranking, final selection, serving/cascade timing and held-out Test remain explicit later decisions, not unfinished items in this local plan.
