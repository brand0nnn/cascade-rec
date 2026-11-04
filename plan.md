# Local experimentation plan

## Current status — registered studies complete, 4 November 2026

**The registered sample programme, full-data replication and objective/co-review factorial are complete.** Sections 2–12 retain their historical protocols and outcomes; §§13–14 record the subsequently authorised full-data studies. Compact deliverables are `results/full_data_replication/20261103-promising/` and `results/full_data_replication/20261104-objective-coreview/`; bulk evidence remains under `results/local_experiments/`. BCE+co-review leads overall quality within the matched factorial, while pairwise text leads new-item quality. Co-review improves warm quality and loses new-item quality under both objectives; direct objective and novelty interactions are inconclusive. Read `docs/evaluations.md` for the current assessment. The subsequently authorised §15 evaluation items are now complete, including locked Test; the user deferred narrative reporting. No further training, graph refresh, canonical integration or Test-driven tuning follows automatically.

| Original programme / approved extension | Status |
|---|---|
| Original RecBole weights, strict-negative ID/content/content+ID controls and popularity | Complete; standalone weights/mappings and audits saved |
| E1 multiple interests; E2 metadata neighbours; E3 mean/attention rankers; E4 actual candidate fusion | Three-seed runs and complete comparisons done; fusion improves graph but not clearly content, while E1–E3 show no clear improvement over their matched controls |
| History truncation, uniform/fixed-recency frozen text and twenty fixed query examples | Complete for the original sample comparison |
| Full archived-Val monthly/new-warm/category/store/history audit | Complete, including monthly history intersections and clipped catalogue-introduction counts |
| Full-catalogue frozen-text validation | Complete; no recommender training |
| Frozen signal diagnosis and subsequently approved candidate-normalised training | Complete; see §§7–8, including the positive normalisation result |
| Phase 2 graph/latent ranking, objective ablation and graph-family decomposition | Complete; see §§10–12, including unsuccessful trained BCE sample epochs and the later pairwise gains |
| Full-data promising-ranker replication | Complete; twelve fits, full-Val scoring, monthly/history/subgroups, primary/secondary/historical contrasts, checked figure and independent audit; see §13 |
| Objective x co-review factorial | Complete; three new BCE+co-review fits, nine frozen controls, direct objective/novelty interactions, checked figure and independent audit; see §14 |
| STePS evaluation items | Complete: full locked Test, popularity/composition, frozen-model K sweep and bounded FAISS/latency, with audits and checked figures. Narrative report deferred by the user; see §15 |

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
- The new sample ranking study is now specified in §10. Full-data execution, final selection, serving/cascade timing and Test remain later decisions.

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

## 10. Phase 2 — Graph- and latent-aware ranking

### Goal and fixed retrieval

Improve top-ten recommendations and new-item generalisation by adding relational and latent signals to ranking. Keep the strong frozen text path intact and explain which additions help which products. This is a new isolated sample study; Phase 1 and teammates' controls remain unchanged.

- Use **frozen MiniLM + 90-day recency retrieval, K=200**, for all arms. Preserve the sample split, query histories, availability/full-history filters and the same pinned encoder/vectors. No target insertion or retriever training.
- Read-only feasibility audit: this pool has **760/6,306 natural ranker-training positives**, versus 681 for normalised content seed 0 and 458 for the old pool. On Val it has **565 natural targets / 131 top-ten hits**, leaving 434 retrieved targets below ten. These are event counts, not macro accuracy or new ranker results.
- Use a fresh study directory under `results/local_experiments/`. Save candidate IDs/scores for all ranking-training and Val queries, their hashes and the exact natural-positive exclusions. All arms share these lists.

### Registered comparisons

| Arm | Addition to the common residual text ranker | Main question |
|---|---|---|
| R0 | Frozen retrieval order; no learned correction | Does reranking add value? |
| R1 | Text/history residual only, with common support controls | Can personalisation improve the strong starting order? |
| R2 | Explicit graph features | Do relationships add information beyond text and degree/support? |
| R3 | Frozen 64D BPR item/history factors | Does collaborative information help supported warm products? |
| R4 | Graph features + BPR factors | Are relational and collaborative signals complementary? |
| R5 | Frozen 64D PCA-text factors | Does simple inductive semantic compression help? |
| R6 | Learned self-only graph-encoder control | Does an additional text transformation explain graph-encoder gains? |
| R7 | One-hop, content-aware graph encoder | Does neighbour aggregation improve on R6, including new products? |

Use **score = frozen retrieval score + learned residual**. Zero-initialise the final residual layer so epoch 0 reproduces the candidate pool's retrieval scores/order. Keep the common head **128 → 64 → 1**, text projection width 32 and mean history pooling. Reserve identical optional input slots, zero ablated blocks, and share head/common initial weights within seed. Give every learned arm the same degree/knownness/support controls, so added factors do not merely reveal item age or graph coverage. Do not add a simultaneous attention or loss sweep.

### Graph and embedding inputs

- **Freeze collaborative inputs strictly before sample T0**, for both B-period training and Val. A pre-T1 graph includes B labels and must not be used as B-training features. Query histories remain strictly prior to each query.
- **Explicit graph features:** candidate-wise other-reviewer count, witness-path count, degree-normalised shared-reviewer strength, supported-history fraction and known category/store history relations. Use sparse pre-T0 user–item operations; exclude the query user's edges and padding/self paths. Compute features for **every candidate**, never from saved true-target profiles. Keep unknown metadata separate. No device/compatibility features.
- **BPR64:** fresh pre-T0 interactions, one uniform pre-T0 negative excluding that user's known positives; Adam .001, batch 4096, fixed 300 epochs, seeds 0/1/2. Benchmark a smoke epoch first. Export the final factors with no Val-based latent-model selection. Freeze factors for ranking; use candidate and known-history-pooled item factors, not a dependence on user-ID lookups. Unseen item/history factors are zero with explicit masks. Apply masks after any adapter so bias cannot invent a missing embedding.
- **PCA-text64:** fit mean/components on unweighted pre-T0 item text only, full SVD, no whitening. Freeze the transform and L2-normalise transformed vectors. Future products use the same transform without refitting; padding/missing text stays zero. This is a semantic compression control, not a collaborative model.
- **Learned graph branch:** frozen MiniLM self features plus separate same-leaf/store neighbour means and availability flags; up to 16 deterministic pre-T0 neighbours per relation, excluding self. Reuse the existing metadata-neighbour bank and a shared **1154 → 128 → 64** encoder. R6 uses identical encoder dimensions/parameter keys with neighbour blocks disabled; common support masks remain available to both arms. R7 uses actual neighbour blocks. Train this branch jointly with the residual head; keep MiniLM frozen. New products attach through their own text and permitted metadata, without item-specific learned IDs. Describe this as a GraphSAGE-style adaptation, not a paper reproduction. Plain PyTorch/MPS suffices; no new graph framework is required.
- Frozen 64D factor branches use the same **64 → 16** adapter and candidate/history/product/absolute-difference interactions. Fit numerical preprocessing on prefix-training candidate inputs only; freeze it for tuning/Val. Reset ranker RNG independently after factor fitting.

### Training, selection and reporting

- Define the chronological **80/20 inner split on all 6,306 B queries before natural-hit filtering**. Use the timestamp at sorted index `floor(0.8 * N)` as the cutoff: earlier queries fit, cutoff-and-later queries tune, keeping timestamp ties together. Record exact times/supports; do not assume 80% of the 760 positives.
- Retain BCE with 50 uniformly sampled alternatives per positive, Adam .001, batch 32 lists, maximum 10 epochs and patience 2. Pair eligible query order and negative columns across arms within seed. Use MPS for dense fitting; CPU handles sparse features/PCA.
- Select by **end-to-end per-user macro nDCG@10 on all inner-tuning queries**, including retrieval misses, with epoch 0 eligible. Break metric ties toward the earlier epoch. Report conditional metrics alongside. Do not use outer Val for checkpointing or hyperparameter search.
- Reinitialise each arm/seed and refit all B natural positives for its selected fixed epoch count; epoch 0 means retain the unchanged baseline. Reuse the frozen pre-T0 banks/preprocessing and paired RNG policy. Then perform one scheduled outer-Val comparison after all configurations are fixed.
- Run separate technical smoke checks, then all registered learned arms at seeds **0/1/2**, preserving negative and epoch-0 outcomes. Sparse support is a limitation, not a reason to drop an arm after seeing scores.
- Headline end-to-end nDCG@10/Recall@10; fixed Recall@200 ceiling; conditional ranking metrics; seed SD; 1,000 paired-user bootstrap draws, seed 0, after seed-mean event metrics. Compare R1–R0, R2/R3/R4/R5–R1, R4–R2/R3 and R7–R6. Report new/warm, pre-T0-known/unknown, graph-support/degree, category continuity and history groups, plus rescues/demotions, list composition, fixed cases, training/export cost and parameter sizes.
- Val has already been inspected extensively: this phase remains **exploratory**, not an untouched confirmatory test. Group differences do not prove a mechanism. A second retriever pool needs a fresh matched ranking replication; do not transfer an uncalibrated residual between raw-cosine and normalised-model score scales.

### Implementation order and acceptance checks

1. Export frozen candidates and the inner split/support manifest; lock the arm list before outcome inspection.
2. Build/audit pre-T0 graph features and frozen BPR/PCA factors.
3. Implement the common residual head and optional feature blocks; run R1–R5 with paired seeds.
4. Implement matched R6/R7; complete their three seeds under the same protocol.
5. Refit, score fixed outer Val, and produce one comparison report with aligned full logits, ranks/top tens, checkpoints, feature/config/source hashes and all registered outcomes. Update this plan and the evaluation record.

Required checks: target relabelling cannot change features; later edges cannot alter frozen banks; query-user-only graph paths disappear; unknown values never create matches; unseen factors stay zero after adapters; new metadata nodes remain finite; epoch-0 scoring is identical to retrieval; all arms share candidates/cohorts/negatives/common initial weights; misses receive zero credit; checkpoint reloads and failure-funnel identities reproduce. Keep all existing outputs and teammate workflows intact.

### Completed sample study — 3 November 2026

All five implementation steps above are complete in **`results/local_experiments/20261102-graph-latent-ranking/`**. The study was paused after fitting, then resumed for the scheduled outer-Val evaluation and reporting. Existing Phase 1 and teammate artifacts are preserved.

- Candidates and preprocessing are frozen. The chronological cutoff is **1621231348495**: **5,044 prefix / 1,262 tuning queries**, with **619 / 141 natural positives**. All-B refit eligibility is 760. The cutoff was defined before filtering for natural positives.
- PCA-text64 was fit on **20,740 pre-T0 items** and transforms future items without refitting. Three BPR64 banks each completed the fixed **300 epochs** on **37,506 pre-T0 interactions**, with exact reload and negative-sampler audits. BPR training took **18.52 / 17.49 / 16.82 s** on MPS; PCA fit/export took **0.178 s** on CPU.
- R1–R7 each completed seeds **0/1/2**, following separate R1/R7 technical smokes. All arms share candidates, negative plans, parameter keys/shapes and per-seed starting weights. Each model has **214,273 parameters**, including inactive reserved branches.
- **Every fit selected epoch 0.** Each actually trained for two epochs before patience stopped it. Tuning baseline nDCG@10 is **0.010169**. Trained epoch-1 scores range **0.000891–0.006426**, epoch-2 scores **0.003407–0.006943**. BCE fell from **0.898–0.996** in epoch 1 to **0.154–0.225** in epoch 2, without improving tuning ranking. Reinitialisation therefore used zero refit epochs for every arm.
- All 21 selected checkpoints were locked before one scheduled outer-Val comparison. Every saved score, target rank and top-ten list exactly reproduces frozen retrieval. Across **4,825 events**, there are **4,260 retrieval misses, 434 retrieved targets below ten and 131 top-ten hits**; rescues and demotions are both zero for every selected arm.

| Selected systems | nDCG@10 | Recall@10 | Fixed Recall@200 | New-in-Val nDCG@10 |
|---|---:|---:|---:|---:|
| R0 and every selected R1–R7 arm | 0.012489 | 0.022309 | 0.100990 | 0.013470 |

These identical outputs are a **no-change selection outcome**, not evidence that trained architectures are equivalent or that graph/latent signals cannot help. The recipe had only 619 fitting positives, 141 natural tuning positives and roughly 40 optimiser steps before stopping. The pointwise BCE objective, residual scale, limited support and optimisation are plausible explanations, not identified causes. No trained-epoch logits or nonselected trained weights were retained; the learning curves alone cannot measure how the residual changed candidate margins. For an unknown BPR candidate its candidate factor is zero, while known-history latent context can still enter the interaction block.

The primary deliverable is **`report/report.md`**, with `training_history.csv`, `metrics.csv`, `paired_differences.csv`, `graph_latent_interaction.csv`, `failure_funnel.csv`, `event_transitions.parquet`, `composition.csv`, all twenty original fixed cases, ranker/factor costs and `provenance.json`. Checkpoint selection replays the earliest tuning maximum; all 21 prediction archives replay exactly; filters, split/plan/feature/factor/source/checkpoint identities and unchanged evaluation-time weights pass. The outer scoring/export job took **29.454 s** including checks/loading/writing, not serving latency. Generated evidence remains local/ignored.

Reproduce into a **fresh study name**, preserving existing output directories, in the project environment:

```bash
python -m cascade.pipelines.graph_latent_study --study FRESH_STUDY --phase prepare --device mps
python -m cascade.pipelines.graph_latent_study --study FRESH_STUDY --phase bpr --seed 0 --device mps
# Repeat BPR seeds 1/2. Run separate --smoke fits for R1/R7, then all seven arms × three seeds:
python -m cascade.pipelines.graph_latent_study --study FRESH_STUDY --phase fit --arm R1 --seed 0 --device mps
# After every full fit is complete:
python -m cascade.pipelines.graph_latent_study --study FRESH_STUDY --phase evaluate --device mps
python -m cascade.pipelines.graph_latent_study --study FRESH_STUDY --phase report
```

**Question proposed after Phase 2, subsequently approved and completed in §11:** can a ranking-aligned objective learn a useful residual while preserving the same frozen retrieval starting point? Begin with a fresh matched R1 BCE-versus-pairwise pilot, retaining candidates, split, 50 alternatives, seeds and epoch-zero selector. Pairwise score differences cancel a common intercept; this is a reason to test, not a promised gain. Record trained-epoch residual margins/logits and gradients on a fixed inner-only probe. Register the exact loss and diagnostic policy before fitting, and decide on broader graph/latent replication from inner evidence before viewing outer Val. Do not launch a loss sweep or reinterpret the completed BCE study as that experiment.

Final verification: **192 tests passed, 4 legacy conventional-path sample-embedding tests skipped**. The study's isolated embeddings and all 21 prediction archives were verified separately. `completion_audit.json` verifies 192 report-input hashes, 15 report-output hashes, 1,200 group-metric rows, 2,100 paired-comparison rows and all registered fits/selection locks.

Full-data promotion, changed-pool replication, serving/cascade timing and final Test remain later decisions. The registered Phase 2 sample checklist is complete.

## 11. Approved objective ablation and conditional graph replication

On 3 November the user authorised continuing ablation/evaluation and updating the README and Markdown records. Fresh study: **`20261103-ranking-objective`**, referencing the immutable Phase 2 candidates/features/factors. Register the following before fitting:

- **Pilot:** R1 with original pointwise BCE versus **mean `softplus(score_negative - score_positive)`** over the same 50 alternatives; no temperature or extra weighting. Seeds 0/1/2, identical starting weights, row/negative plans, Adam .001, batches of 32 lists, maximum 10 epochs, patience 2. Separate one-epoch technical smokes for both losses. Keep the original chronological split, epoch-zero eligibility, end-to-end user-macro tuning nDCG@10 and earlier-tie selector. Reinitialise/refit all B natural positives for the selected epoch count.
- **Diagnostics:** save every inner-epoch checkpoint and all tuning logits, including epoch 0. Freeze a probe of the earliest 32 natural prefix and 32 natural tuning queries by timestamp/instance ID. Save probe logits, residual distributions, positive-versus-negative margins, top-ten boundary margins and per-module gradient norms. Diagnostics do not select checkpoints or tune hyperparameters. Retain exact curves from the fresh BCE control and compare with the original R1 fits. Record unavailable/inactive gradients separately from measured nonzero values.
- **Expansion gate, inner-only:** run pairwise R2–R7 at all three seeds only if R1 pairwise mean selected tuning nDCG@10 exceeds retrieval, at least two seeds individually exceed retrieval, and mean selected tuning nDCG@10 for targets introduced at/after T0 does not fall below retrieval. This is an exploratory point-estimate gate, not a confidence/noninferiority claim. If it fails, retain/report the six pilot fits without expanding. Save the decision before any outer-Val prediction.
- **Reporting:** lock the complete realised arm/seed set, selected epochs and refit weights before one outer-Val comparison. Retain all pilot arms regardless of outcome; if expanded, compare graph/factor additions against pairwise R1, R7–R6, R4–R2/R3 and the factorial graph/BPR contrast. Report all/new/warm/pre-T0-known/unknown/history/graph/category groups, paired-user 1,000-draw intervals (seed 0 after event-metric seed averaging), failure funnels, list composition, twenty fixed cases, training curves/diagnostics and offline costs. No Val-driven objective choice, second loss sweep, new retrieval pool, full-data fit or Test evaluation.
- **Integrity:** bind parent inputs, copied sampling plans, source snapshots, split/probe cohorts and retained checkpoint/logit files with hashes. Verify loss shift invariance, candidate/label alignment, gradient flow, epoch-zero retrieval equivalence, paired initial states/negatives, earliest-max selection, selected-state reloads, unchanged evaluation-time weights and exact saved-score replay. Preserve all older outputs. Existing pipeline defaults and frozen Phase 2 source files remain unchanged.

### Completed objective study — 3 November 2026

All registered work is complete on MPS: two technical smokes, six matched pilot fits and eighteen conditionally registered feature fits. The three fresh BCE learning curves exactly reproduce Phase 2 and select epoch 0. Pairwise R1 selects epochs **5/9/2**; its mean selected inner nDCG@10 is **0.017324 versus 0.010169** for retrieval, with all three seeds improving. Targets introduced at/after T0 have mean inner nDCG **0.024287 versus 0.007983**. The gate was independently replayed from saved inner predictions and fixed before expansion/outer evaluation; this inner cold group differs from new-in-Val reporting.

All twenty-four selected epochs/refit weights were locked before the scheduled outer-Val comparison. Fixed Recall@200 remains **0.100990** for every method. No retriever or factor model was retrained.

| Selected method | nDCG@10 | Recall@10 | New-in-Val nDCG@10 | Warm-in-Val nDCG@10 |
|---|---:|---:|---:|---:|
| Retrieval / BCE R1 | 0.012489 | 0.022309 | 0.013470 | 0.009736 |
| Pairwise R1, text residual | 0.015566 | 0.027348 | 0.020344 | 0.007234 |
| Pairwise R2, graph/history scalars | 0.018174 | 0.033548 | 0.022890 | 0.009333 |
| Pairwise R3, BPR64 | 0.016279 | 0.027682 | 0.020977 | 0.008060 |
| Pairwise R4, graph/history + BPR64 | 0.018337 | 0.033238 | 0.023190 | 0.009350 |
| Pairwise R5, PCA-text64 | 0.016195 | 0.029285 | 0.020864 | 0.008098 |
| Pairwise R6, self-only encoder | 0.016336 | 0.028904 | 0.020770 | 0.007392 |
| Pairwise R7, neighbour encoder | 0.013304 | 0.024214 | 0.016372 | 0.008164 |

Registered end-to-end nDCG@10 differences, with exploratory/unadjusted paired-user 95% intervals:

- **Objective:** pairwise R1 minus BCE R1 **+0.003078 [0.000501, 0.005719]**; new-in-Val **+0.006874 [0.003580, 0.010510]**. Warm difference **-0.002502 [-0.006700, 0.001523]** is inconclusive. This is the objective's total training/selection/refit recipe effect, not proof of a calibration cause or equal update counts.
- **Graph/history family:** R2−R1 **+0.002608 [0.001028, 0.004108]**; new **+0.002546 [0.000541, 0.004580]**, warm **+0.002099 [0.000238, 0.004063]**. R2 includes co-review counts/strength and category/store-history shares; no individual relation is isolated. Warm R2 remains slightly below retrieval in point estimate.
- **BPR:** R3−R1 **+0.000712 [0.000004, 0.001390]**, a small result with a near-zero lower bound and inconclusive new/warm effects. R4−R2 **+0.000163 [-0.000204, 0.000516]**; adding BPR to graph/history has no clear benefit. Factorial interaction **-0.000549 [-0.001348, 0.000259]** establishes no complementarity.
- **Compression/extra transform:** R5−R1 **+0.000629 [-0.000394, 0.001639]**, R6−R1 **+0.000770 [-0.000465, 0.001950]**; both inconclusive.
- **Neighbour encoder:** R7−R6 **-0.003033 [-0.004759, -0.001200]** overall and **-0.004398 [-0.006660, -0.002171]** new-in-Val. Warm differences are inconclusive; do not generalise the loss to every group or infer oversmoothing as its cause.

The R2 funnel averages **99.33 rescues / 53 demotions / 177.33 top-ten hits** per seed, versus retrieval's 131 hits. R4 averages 99.67 / 51.33 / 179.33. All retain **4,260 retrieval misses**. These are means of event counts across seeds, not counts of unique rescued events or relevance judgements about unobserved alternatives.

All **152 inner checkpoints/prediction sets**, including epoch zero, remain saved with **552 retained artifact hashes**. Diagnostics show active text/head gradients and intentionally inactive reserved branches. Pairwise final-bias gradients are near numerical zero, while query-mean shifts can still emerge through the network. A common offset alone cannot change ranking. Fixed chronological 32+32 natural-query probes are descriptive, not representative; candidate-dependent margins/spread are saved separately. Retention/diagnostic work adds offline fitting overhead, so timings are not directly comparable with lean Phase 2 fits.

Use **`results/local_experiments/20261103-ranking-objective/report/report.md`**, its metric/contrast/funnel/composition/case/diagnostic/cost tables, visually checked `objective_diagnostics.png` and source-bound provenance. `completion_audit.json` verifies **761 report-input hashes / 18 report-output hashes**, all checkpoint locks, gate replays and score/funnel identities. Suite: **207 passed, 4 legacy conventional-path embedding skips**; isolated study inputs and all real prediction replays pass separately. A compact figure is shared in `docs/figures/20261103-ranking-objective.png`; bulk artifacts remain local/ignored.

Reproduce with a fresh output name and the immutable parent study available:

```bash
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase prepare
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase smoke --device mps
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase pilot --device mps
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase decide
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase expand --device mps
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase evaluate --device mps
python -m cascade.pipelines.ranking_objective_study --study FRESH_STUDY --phase report
```

The simpler pairwise R2 is a useful candidate for further robustness work; its near-tie with R4 does not select a final system. A later decomposition should separate category/store-history inputs from co-review inputs to identify which family helps. Temporal replication, full-data fits, a new candidate pool, canonical integration, serving cost and Test remain separate decisions. No additional architecture/loss sweep follows automatically from these reused-Val outcomes.

## 12. Approved graph completion, then stop for assessment

The user approved implementing the bounded graph plan: twelve matched sample fits plus frozen R6/R7 diagnostics, then documentation/commit and no further experiment execution. Fresh output: **`20261103-graph-family-decomposition`**. Register before fitting:

- G0 disables the six optional graph/history columns; GM enables only `same_leaf_history_fraction` / `same_store_history_fraction`; GC enables only `log_other_reviewers`, `log_witness_paths`, `normalised_path_strength`, `supported_history_fraction`; GMC enables all six. Every arm uses the same existing R2 model and zeroes disabled columns **after standardisation**. Existing text/basic matching/nine common support controls remain present.
- Seeds 0/1/2, existing candidates/split/paired negative plans, pairwise softplus, Adam .001, batch32, max10/patience2, epoch-zero earliest-tie end-to-end inner selector and all-B fixed-epoch refit. Separate one-epoch GM/GC smokes, then all twelve full fits without pruning.
- Fresh G0/GMC must reproduce previous pairwise R1/R2 selected epochs and inner ranking curves before outer evaluation; loss comparison uses rtol=1e-6/atol=1e-8. Lock all twelve selected refit weights before the single outer report. Register all six pairwise arm comparisons and interaction GMC−GM−GC+G0.
- Reuse saved R6/R7 epoch2 and selected-inner checkpoints: score full prefix cohorts, reuse saved tuning scores and final Val predictions, and inspect residual/margin distributions, graph-input blocks, layer norms and gradient records. No new GNN fit or changed selection; diagnostic associations do not identify a causal mechanism.
- Preserve all inner checkpoint/logit/probe/gradient records; verify masks, initial states, candidates/negatives, epoch-zero equivalence, selected-state reloads, scores/ranks/top tens and funnels. Use existing all/new/warm/history/category/graph groups, conditional populations, fixed coverage, seed SD and 1,000 paired-user draws with seed0 after event-metric seed averaging.
- Deliver source-bound reports, checked comparison figure, README/CLAUDE/plan/evaluation/decision updates and a commit. Keep bulk arrays local/ignored. Then stop all experiment jobs for the deeper assessment; no automatic model sweep, temporal replication, full-data fitting, new candidate pool or Test evaluation.

### Completed graph handoff — 3 November 2026

Both technical smokes and all **12 full fits** completed on MPS. G0/GMC reproduce all six previous R1/R2 selectors, inner ranking curves and losses within the registered tolerance; their final Val logits/ranks/top tens also match exactly. All twelve selected epochs/refit weights were locked before one scheduled outer comparison. Fixed Recall@200 remains **0.100990**.

| Arm | Selected epochs 0/1/2 | Overall nDCG@10 | Recall@10 | New-in-Val nDCG@10 | Warm-in-Val nDCG@10 |
|---|---|---:|---:|---:|---:|
| G0 | 5 / 9 / 2 | 0.015566 | 0.027348 | 0.020344 | 0.007234 |
| GM | 2 / 9 / 2 | 0.016444 | 0.030213 | 0.020946 | 0.008222 |
| GC | 2 / 2 / 2 | 0.017728 | 0.032578 | 0.022505 | 0.009181 |
| GMC | 2 / 2 / 2 | 0.018174 | 0.033548 | 0.022890 | 0.009333 |

All six registered overall nDCG@10 contrasts, with exploratory/unadjusted paired-user 95% intervals:

- GM−G0 **+0.000877 [0.000033, 0.001793]**; GC−G0 **+0.002162 [0.000851, 0.003486]**; GMC−G0 **+0.002608 [0.001028, 0.004108]**.
- GC−GM **+0.001284 [0.000277, 0.002243]**; GMC−GM **+0.001730 [0.000729, 0.002682]**; GMC−GC **+0.000446 [-0.000282, 0.001193]**.
- Interaction GMC−GM−GC+G0 **-0.000431 [-0.001357, 0.000462]**: no complementarity established. Metadata-history adds no clear gain beyond co-review. GC improves new **+0.002161 [0.000329, 0.003990]** and warm **+0.001947 [0.000340, 0.003613]** versus G0. GM's new/warm intervals include zero; its overall lower bound is near zero. All arms retain existing matching/support controls, so this is not removal of all metadata/graph information or isolation of a single co-review feature. New products still have no pre-T0 review edges.

Mean-per-seed top-ten hits are **149.33 / 160.33 / 176.00 / 177.33** for G0/GM/GC/GMC. GC averages **98.67 rescues / 53.67 demotions**; GMC **99.33 / 53.00**. All keep **4,260 retrieval misses**. These are per-seed means, not unique rescued-event totals.

The frozen encoder diagnostic completed **12 checkpoint loads**: epoch2 and selected inner states for R6/R7 at all three seeds, scoring all 5,044 prefix queries and reusing 1,262 tuning-query predictions plus saved final Val predictions. Mean prefix/tune nDCG@10:

| Encoder | Inner checkpoint | Prefix | Tune |
|---|---|---:|---:|
| R6 self-only | Epoch 2 | 0.031686 | 0.014638 |
| R7 neighbour | Epoch 2 | 0.032823 | 0.016024 |
| R6 self-only | Selected | 0.035038 | 0.017119 |
| R7 neighbour | Selected | 0.038007 | 0.018435 |

Saved full-B-refit Val remains **0.016336 R6 / 0.013304 R7**. The neighbour branch learns on the inner cohorts; this does not identify why the final refit performs worse. Inner states and full-B refits are different weights/populations/update exposure. Full residual/margin distributions, input-block norms/cosines, initial-to-trained layer changes, active gradients and subgroup funnels are retained. No encoder training or reselection occurred, and no oversmoothing/optimisation cause is established.

Deliverables: **`results/local_experiments/20261103-graph-family-decomposition/report/report.md`**, **`encoder_diagnostic/report.md`**, all group/conditional/contrast/funnel/composition/case/cost tables and a checked shared figure in `docs/figures/20261103-graph-family-decomposition.png`. The outer score/export job took **12.309 s** and the frozen diagnostic **27.742 s**, including offline checks/loading/writes; neither is serving latency.

Final suite **226 passed / 4 legacy conventional-path embedding skips**. The completion audit verifies **75 retained inner epoch sets / 273 retained artifact hashes**, all twelve final-score and twelve frozen-prefix replays, six exact control reproductions, initial states/masks/plans/locks and funnel identities. Report provenance binds **836 inputs / 18 outputs**; encoder diagnostics bind **280 inputs / 23 outputs**. Existing studies and teammate workflows are unchanged; bulk artifacts remain local/ignored.

Reproduce only into a fresh study directory, with the immutable parents present, using `cascade.pipelines.graph_family_study` phases **prepare → smoke → fit → preflight → evaluate → diagnose → report**, with `--device mps` for fitting/evaluation/diagnosis. Existing destinations are refused. The retained diagnostics use saved checkpoints, not new fitting.

**Completion stop:** graph experiment work is finished. After the documentation/commit handoff, stop execution and leave the evidence for our deeper assessment. No additional sweep, temporal replication, candidate-pool change, full-data fit, canonical integration or Test evaluation is started.

## 13. Authorised full-data replication — 3 November 2026

After the deeper assessment, the user explicitly requested full-data testing of the most promising methods, integration of the findings and a commit. This resumes execution for this bounded isolated study and supersedes §12's stop for that scope. Register `configs/ranker/full_promising.yaml` before fitting:

- Frozen MiniLM/90-day-recency natural K200 retrieval; fresh BCE text and pairwise text/co-review/combined category-store+co-review rankers, all seeds 0/1/2. Preserve the historical teammate models and all sample studies. Do not add BPR, PCA, neighbour encoders or new retrieval sweeps.
- Canonical full-data T0/T1/T2, full catalogue and complete B/Val populations; strictly pre-T0 graph, unchanged 50-history/full-history availability filters, no target insertion. Chronological 80/20 B split defined before natural-hit filtering. Fit feature normalisation on all prefix candidate slots. Pair negatives and starting weights across arms. Same Adam .001, batch32, fifty alternatives, max10/patience2, epoch-zero earlier-tie all-inner-query user-macro nDCG selector; all-B refit for the selected fixed epoch count.
- Scale sparse graph extraction in batches, verify against the original builder, and keep bulk arrays on disk. First run the complete new path on the existing sample in a separate technical-smoke directory; do not present it as a new sample finding. Run tests before experiment batches.
- Lock all twelve selected epochs/refit weights before one full-Val comparison. Report overall/new/warm, monthly/history groups, conditional quality, fixed candidate ceiling, seed SD, paired-user intervals, exact rescues/demotions and offline costs. Reused Val remains exploratory; monthly slices do not constitute independent temporal retraining. Warm paired differences are a guardrail report, not equivalence without a margin.
- Add aligned historical context against archived MLP K200, the observed MLP K100 validation maximum and all-training SASRec, including paired new/warm uncertainty. These different-pipeline comparisons are exploratory context rather than matched causal interventions or a final K selection; they do not change any fit or selector.
- Complete exact score/checkpoint/filter/selection/provenance audits; add compact results and checked figures to README, evaluations, plan and decisions, then commit. Full-data validation is authorised; final held-out Test, canonical integration, index/serving studies and historical score-policy/dropout experiments retain their separate status.

### Completed full-data study — 4 November 2026

All twelve fits and one scheduled outer-Val comparison completed. Full B contains **315,114 queries / 20,675 natural positives**; chronological prefix/tune support is **252,091 / 63,023 queries** and **16,818 / 3,857 natural positives**. All **232,904 Val events / 124,416 users** are retained; **14,916 targets** enter the common K200 lists. The graph contains 1,877,548 strictly pre-T0 edges. All selected epochs/weights were locked before outer scoring.

| Method | Selected epochs | nDCG@10 | New | Warm |
|---|---|---:|---:|---:|
| Retrieval | — | 0.006284 | 0.007611 | 0.005606 |
| BCE text/common | 10/6/6 | 0.008115 | 0.008697 | 0.007846 |
| Pairwise text/common | 2/3/3 | 0.007767 | 0.009152 | 0.007103 |
| Pairwise co-review | 10/8/5 | 0.008065 | 0.008873 | 0.007571 |
| Pairwise combined | 10/6/5 | 0.007930 | 0.008261 | 0.007707 |

BCE's full-data learning and overall lead qualify the sample's objective finding; pairwise retains a new-item advantage. Co-review shifts its incremental benefit to warm products while losing new-item quality; extra metadata loses overall/new quality. The co-review-versus-BCE overall difference is inconclusive. BCE improves the frozen retrieval control by 29.1% in nDCG, but no new arm establishes superiority over the archived MLP K100 validation maximum. This is reused-Val exploratory evidence, with unadjusted user-bootstrap intervals conditional on these runs. Full primary and explicitly secondary/historical comparisons are in `docs/evaluations.md` and `results/full_data_replication/20261103-promising/`; no final system or causal mechanism is selected.

The separate sample path reproduced all twelve first-epoch ranking controls. Independent full audit: **548,018 candidate lists**, **96 feature-reference queries**, **104 inner score replays**, **12 outer replays / selected locks**, **675 metric reconstructions**, **18 independent main nDCG intervals**; all pass. Final suite **229 passed / 4 legacy skips**. Compact evidence and the checked figure are shared; bulk features/candidates/weights/logits remain ignored. The registered work is complete. No further training, changed-pool integration or held-out Test evaluation follows automatically.

## 14. Approved objective x co-review completion — 4 November 2026

The user supplied the original design, confirmed the controlled-comparison framing, and instructed proceeding with the missing BCE+co-review cell. This supersedes §13's completion stop only for this bounded extension. Register `configs/ranker/objective_coreview.yaml` before fitting:

- Reuse the exact full-data K200 candidates, pre-T0 features, prefix scalers, chronological split, paired negatives and three old cells from §13. Fit only BCE+four optional co-review columns, seeds 0/1/2. Keep basic matching/nine common controls and zero the optional metadata-history fractions after standardisation. The existing parent remains immutable.
- Same starting parameters, BCE, Adam .001, batch32, fifty alternatives, max10/patience2, earliest-tie epoch-zero inner user-macro selector and selected-fixed-epoch all-B refit. First complete a separate sample technical path, including an off-feature BCE control reproducing its old first-epoch curve. No new objective, graph snapshot or candidate-source sweep.
- Lock all twelve factorial cell/seed selections before scoring the three new full-Val outputs. Primary differences: BCE graph gain, pairwise graph gain, BCE-on versus pairwise-on, objective difference with co-review off, and new cell versus retrieval. Direct interaction is `(PW_on-PW_off)-(BCE_on-BCE_off)`; report all/new/warm user-paired intervals rather than comparing significance across cells.
- Test interaction_new minus interaction_warm using joint user-cluster resampling over the union of users, preserving each group's own qualifying-user macro denominator and covariance. This is a descriptive contrast between novelty populations, not a causal novelty intervention. Use B=1000, seed0 after averaging event metrics over training seeds. Retain monthly/history groups, fixed coverage, conditional metrics and funnels.
- Preserve all new inner checkpoints/logits, verify exact scoring/selection/initialisation/negative-plan controls, and share audited compact results plus a checked interaction figure. Update the project records and commit. Reused Val remains exploratory/unadjusted, and objective comparisons include selection/update-count differences. Graph retrieval, new GNNs, canonical integration, serving/index studies and final Test remain later decisions.

### Completed factorial study — 4 November 2026

Three new BCE+co-review seeds selected epochs **8/9/8**. All twelve cell/seed selections were locked before scoring the new cell on the same **232,904 Val events / 124,416 users**. The sample off-feature control reproduces the original BCE first epoch; all nine full-data controls remain immutable.

| Objective | Co-review | Overall nDCG@10 | New | Warm |
|---|---|---:|---:|---:|
| BCE | Off | 0.008115 | 0.008697 | 0.007846 |
| BCE | On | **0.008409** | 0.008342 | **0.008384** |
| Pairwise | Off | 0.007767 | **0.009152** | 0.007103 |
| Pairwise | On | 0.008065 | 0.008873 | 0.007571 |

BCE graph gain is **+0.000295 [0.000169, 0.000425]**, or **3.6%**, with new **−0.000354 [−0.000542, −0.000166]** and warm **+0.000538 [0.000375, 0.000706]**. The pairwise graph gain has the same directional trade-off. Direct objective interaction is **+0.000003 [−0.000159, 0.000163]** overall; the jointly resampled new-minus-warm interaction is **+0.000146 [−0.000162, 0.000511]**. Neither establishes objective dependence or equivalence. Fixed Recall@200 remains **0.054746**; all methods retain **217,988 retrieval misses**.

Compact evidence, the exact numerical-analysis source snapshot and hash-bound publication are under `results/full_data_replication/20261104-objective-coreview/`. The checked figure is `docs/figures/20261104-objective-coreview.png`; `docs/evaluations.md` records interpretation and reproduction. **233 tests pass / 4 legacy skips**. Independent checks reproduce 33 new inner score sets, three new outer outputs, all twelve locks, 675 metric rows, eighteen main intervals and the joint novelty interval. The source snapshot precedes a plot-label formatting adjustment; numerical outputs remain unchanged.

**Completion:** the bounded extension is complete. Val reuse and conditional, unadjusted intervals limit inference. Follow §15 below for the STePS evaluation tidy-up and completion programme.

## 15. STePS evaluation tidy-up and completion plan — 4 November 2026

This section provides a self-contained execution specification for agents and teammates preparing the final evaluation tables and figures for the CP4285 report and STePS poster.

### 15.1 Governing experimental principle (retiring force-inserted ranker claims)

- **Do not use historical force-inserted ranker runs (`20261024-feature-ladder`, `20261024-shortlist-repeat`, `20261025-rq3` MLP/DIN comparison, `ranker_k1_sweep`, or the post-hoc KG evaluation `20261101-kg-expanded-final` on those checkpoints) as primary scientific answers for RQ2 or RQ3.** In those historical runs, `87.69%` (`276,321 / 315,114`) of ranker-training lists had a missed target artificially appended at column 201 (`results/saved_evaluation/20261027-final-validation/saved_diagnostic_report.md`). The constructed training task has reversed raw retrieval-score AUC (`0.080` vs. `0.629` on natural validation hits). The saved MLP also shows inverse score associations (`Spearman about -0.40`) and deteriorates as K grows. These diagnostics establish a serious train/evaluation mismatch, but do not isolate its causal contribution to every historical failure.
- **Retain the score-inversion audit solely as a 1-box methodological post-mortem** illustrating the mismatch introduced by force-inserting retrieval misses into Stage-2 training.
- **Do not launch a new full-data GraphSAGE / GNN pipeline.** Inductive neighbour aggregation (`self + neighbour`) has already been evaluated in both stages under strict pre-$T_0$ controls:
  - **Retrieval (`content + metadata neighbours`, §6):** reduced `Recall@200` (`0.053690` vs. `0.056750` content-only) and `nDCG@10` (`0.001248` vs. `0.001836`).
  - **Ranking (`R7: learned neighbour encoder` vs. `R6: learned self-only encoder`, §§11–12):** fit inner prefix/tuning cohorts (`0.0380` / `0.0184` vs. `0.0350` / `0.0171`), but lost `-0.003033 [-0.004759, -0.001200]` `nDCG@10` on outer Val (`0.013304` vs. `0.016336`), far below explicit co-review scalars (`GC` `0.017728`, `R2` `0.018174`).
  - Frame `E2` (retrieval neighbour aggregation), `R6` vs. `R7` (ranker self vs. neighbour encoder), and `G0/GM/GC/GMC` (§12) + full-data `Objective x Co-Review` (§§13–14) as the project's controlled graph investigation: **in high-turnover chronological splits (`33%` Val / `44%` Test new items with zero pre-$T_0$ review edges), explicit co-review relational scalars performed better than the tested learned-neighbour recipes on the sample. This does not establish oversmoothing, universal graph-model superiority, or the cause of the final-refit loss.**

### 15.2 Headline thesis and canonical four-pillar narrative

> **Headline Thesis:** *In time-evolving product catalogues with high item turnover, representation, objective and graph-feature choices expose measurable trade-offs between **warm items** and **new items**. Candidate-normalised representations and explicit relational features are promising in our tested comparisons.*

1. **Pillar 1 — RQ1 Retriever Design & Representation Geometry (`results/rq1/summary.md`, §§6–8):**
   - Collaborative baselines (`Popularity`, `ItemKNN`, `BPR-MF`, `SASRec`) score $\approx 0$ on new-in-period items.
   - `Two-tower (text)` unlocks cold items (`Recall@200` `0.0686` overall, `0.0263` new-in-period); adding learned item IDs (`Two-tower (text + ID)`) improves warm co-reviewed items (`+5.4%` overall `Recall@200` to `0.0724`) at a `-28%` loss on new items (`0.0190`).
   - The tested dot-product-trained projection changes pretrained MiniLM neighbourhoods, and its scoring favours high-norm items (`17.0%` of top-10 slots go to top-1%-norm items); L2 candidate normalisation (`content_unit`, §8) more than doubles sample `nDCG@10` (`0.001836` $\rightarrow$ `0.003978`), while training-free `Frozen MiniLM + 90-day recency` (§6) reaches `0.006284` full-Val `nDCG@10` (`Recall@200` `0.054746`) by heavily boosting new-in-Val items (`0.007611` vs. `0.000352`).
2. **Pillar 2 — RQ3 Ranker Objectives & Graph Design on Natural Hits (§§11–14):**
   - On full-catalogue natural $K=200$ candidates (`232,904` Val events / `124,416` users), **`BCE + Co-Review` (`bce_coreview`)** has the highest overall point estimate in the matched full-data factorial (`nDCG@10` **`0.008409`**, `Recall@10` **`0.015845`**, `+33.8%` over retrieval `0.006284`, rescuing `2,505.7` targets vs. `1,515.7` demotions).
   - **Loss-objective recipe (`BCE` vs. `Pairwise`) shows a warm/new trade-off:** `Pairwise` softplus wins on **new items** (`0.009152` vs. `0.008697` `BCE`), while `BCE` wins on **warm items** (`0.007846` vs. `0.007103`) and overall (`0.008115` vs. `0.007767`) on full-data natural-positive support (`20,675` positives vs. `458` on the 2% sample).
   - **Explicit co-review graph features (`+3.6%` BCE / `+3.8%` pairwise overall)** consistently shift recommendations toward warm items (`+0.000538` BCE, `+0.000468` Pairwise) with a small new-item trade-off (`-0.000354` BCE, `-0.000279` Pairwise) and an inconclusive objective interaction (`+0.000003 [-0.000159, 0.000163]`). Extra category/store-history fractions (`pairwise_combined`) degrade overall (`-0.000135`) and new-item (`-0.000612`) quality.
3. **Pillar 3 — RQ2 Candidate Count $K$, FAISS Indexing & Cascade Latency (Phases 2 & 3 below):**
   - Replaces the invalidated force-inserted $K$-sweep with natural-hit ranker scaling across $K \in \{10, 50, 100, 200\}$ and benchmarks `Flat` vs. `IVF` vs. `HNSW` recall, end-to-end `nDCG@10`, and per-request latency.
4. **Pillar 4 — Methodological Callout (`saved_diagnostic_report.md`):**
   - Summarizes the force-insertion score-inversion mechanism as a concise design lesson.

### 15.3 Zero-retraining execution plan (Test prioritised)

The user approved the amendments discussed in chat: no delivery schedule; complete Validation saved-score checks, then evaluate locked Test before the bounded efficiency study. The latest instruction defers the narrative report: finish the items and retain numerical tables, figures and audits first. Use fresh bulk study `results/local_experiments/20261104-steps-final/` and compact publication `results/steps_final/20261104/`. The runner is `python -m cascade.pipelines.steps_final` with phases `prepare`, `validation`, `test`, `report`, `benchmark`, `publish`. Old frozen source files and studies remain immutable.

All four phases reuse immutable locked artifacts under `results/local_experiments/20261102-full-frozen-text/`, `results/local_experiments/20261103-full-promising-ranking/`, and `results/local_experiments/20261104-objective-coreview/`. Write new compact outputs under a fresh directory (`results/steps_final/` or a fresh study under `results/local_experiments/`); never overwrite existing study directories.

#### Phase 1 — Complete RQ3 baselines & beyond-accuracy composition on full Val ($K=200$)

- **Inputs:**
  - Candidate IDs and retrieval scores: `results/local_experiments/20261103-full-promising-ranking/candidates/val/` (`232,904 × 200`).
  - Locked outer-Val per-candidate score archives (seeds 0/1/2): `results/local_experiments/20261103-full-promising-ranking/evaluation/` (`bce_text`, `pairwise_text`, `pairwise_coreview`, `pairwise_combined`) and `results/local_experiments/20261104-objective-coreview/evaluation/` (`bce_coreview`).
- **Tasks:**
  1. **Add `R0: Popularity order` on the valid $K=200$ pool:** Sort the exact 200 retrieved candidates per query by strictly pre-$T_0$ positive review counts (using pessimistic tie-breaking for target ranks and stable candidate-ID tie-breaking for top-10 slot display). Compute overall, `New`, and `Warm` user-macro `nDCG@10`, `Recall@10`, conditional `nDCG@10`, five-way funnel counts, and paired-user 95% bootstrap intervals (`B=1000, seed=0`) against `Retrieval` and `bce_coreview`.
  2. **Compute Slide 10 "Beyond Accuracy" metrics across all full-Val methods (`Popularity order`, `Retrieval order`, `bce_text`, `pairwise_text`, `pairwise_coreview`, `pairwise_combined`, `bce_coreview`):**
     - Catalogue coverage (distinct products appearing in top-10 across all `232,904` Val queries).
     - Top-10 popularity concentration (mean pre-$T_0$ review count of recommended top-10 items, pre-$T_0$ head-item slot share, and unseen-after-$T_0$ slot share).
     - Top-10 category and store concentration (unique known leaf categories/stores per top-10 list and user-macro Herfindahl index, excluding unknown metadata slots with support recorded).
- **Outputs:** `results/steps_final/20261104/val_comparison.csv`, `results/steps_final/20261104/val_beyond_accuracy.csv`.

#### Phase 2 — RQ2 inference-budget sensitivity for K200-trained models ($K \in \{10, 50, 100, 200\}$) from saved logits

- **Inputs:** Saved `232,904 × 200` outer-Val candidate matrices (stored in exact retrieval rank order `1..200`) and the saved `232,904 × 200` logit matrices for `bce_text`, `bce_coreview`, `pairwise_text`, and `pairwise_coreview` (seeds 0/1/2).
- **Tasks:**
  1. This tests frozen models trained at K200, not independently trained or optimised per-K systems. Verify saved logits against fresh scoring and per-candidate score invariance. For each budget $K \in \{10, 50, 100, 200\}$, restrict each query's candidate pool and saved model logits to the first $K$ retrieval positions (`[:, :K]`).
  2. Compute candidate ceiling `Recall@K`, end-to-end user-macro `nDCG@10` and `Recall@10` (overall, `New`, and `Warm`), conditional `nDCG@10`, and five-way failure funnels per seed and seed-mean.
  3. Compute the additive $K$-expansion decomposition (`50 → 100` and `100 → 200`): separate `new-target entry nDCG gain` (targets entering in slots `K_prev+1 .. K_next`) from `existing-target competition change` (targets already in `1..K_prev` displaced or promoted when `K_prev+1 .. K_next` candidates are added), with paired-user 95% bootstrap intervals (`B=1000, seed=0`). Keep zero contributions for other events and the common full-population user weights so entry plus competition equals the observed change.
- **Outputs:** `results/steps_final/20261104/rq2_k_sweep.csv`, `results/steps_final/20261104/rq2_expansion_decomposition.csv`, and figure `docs/figures/steps_rq2_k_sweep.png`.

#### Phase 3 — RQ2 FAISS search settings (`Flat` vs. `IVF` vs. `HNSW`) & per-request cascade latency

- **Inputs:**
  - Full-catalogue L2-normalised MiniLM vectors: `results/local_experiments/20261102-full-frozen-text/item_text_emb.npy` (`111,480 × 384`).
  - Locked checkpoints and feature builders from `20261103-full-promising-ranking` (`bce_text`) and `20261104-objective-coreview` (`bce_coreview`).
- **Tasks:**
  1. Build FAISS indexes over the `111,480` product vectors:
     - `Flat`: `IndexFlatIP`, K200, inner-product reference.
     - `IVF`: `IndexIVFFlat` (`nlist=256`, trained on pre-$T_0$ items only, `nprobe=16`) at $K=200$.
     - `HNSW`: `IndexHNSWFlat` (`M=32`, `efConstruction=200`, `efSearch=max(256, fetched results)`) at $K=200$.
     - Account for post-retrieval availability and full prior-history exclusions by starting with 400 hits and doubling until 200 eligible candidates or the full catalogue is exhausted. The local index contract requires HNSW efSearch >= fetched results; report the effective value instead of the infeasible proposed efSearch64. Verify Flat against archived exact retrieval, disclose numeric ties, and label quality as a fixed benchmark-subset estimate.
  2. On a deterministic 1,000-query validation benchmark (after 50 warm-up queries, single-thread CPU, recording hardware/thread settings), measure per-query median and p95 latency (ms) for:
     - **Stage 1 Retrieval:** query recency pooling + FAISS search + availability/history filtering.
     - **Stage 2 Ranking:** feature extraction + scoring + top-10 sorting for `Popularity order`, `Retrieval order`, `bce_text`, and `bce_coreview`.
     - **End-to-End Total Latency:** Stage 1 + Stage 2.
  3. Record resulting candidate `Recall@K` and end-to-end `bce_coreview` `nDCG@10` across the index settings.
- **Outputs:** `results/steps_final/20261104/benchmark_latency_summary.csv` and Pareto figure `docs/figures/steps_rq2_faiss_latency.png`.

#### Phase 4 — Single locked held-out `Test` evaluation (`297,390` cases) & final consolidation

- **Prerequisites:** Complete and verify Phases 1–2 on Validation first; Test does not wait for FAISS. Retain K200 regardless of the sensitivity plot. Lock the final system list before touching `Test`: `F0: Popularity order`, `F1: Frozen 90d Recency Retrieval`, `F2: BCE Text Control`, `F3: Pairwise Text Control`, `F4: Pairwise + Co-Review`, and `F5: BCE + Co-Review`.
- **Tasks:**
  1. Generate natural $K=200$ Test candidate lists (`297,390` Test events from 16 Jul 2022) using `Frozen MiniLM + 90-day recency` with strict event-time availability and full prior-history exclusions.
  2. Extract Test features using the same strictly pre-$T_0$ graph and locked prefix feature scalers from `20261103-full-promising-ranking`. No graph refresh, training, target insertion or Test-driven model/K selection. Define new-in-Test relative to T2; catalogue eligibility and permitted histories still follow each query timestamp.
  3. Score the locked 3-seed checkpoints from `20261103-full-promising-ranking` and `20261104-objective-coreview` once on `Test` (no tuning, no epoch re-selection). Compute overall, `New`, and `Warm` user-macro `nDCG@10`, `Recall@10`, `Recall@200`, five-way funnels, and paired-user 95% bootstrap intervals (`B=1000, seed=0`).
  4. Save the numerical tables, checked figures and source/output hashes (`completion_audit.json`); update completion status and commit. **Do not write the narrative report yet**, per the user’s latest instruction.

### Completed evaluation items — 4 November 2026

- [x] Full-Val popularity ordering and composition (catalogue coverage, pre-T0 popularity/head/unseen slot shares, known leaf/store HHI and support).
- [x] Full-Val K10/50/100/200 inference sensitivity of K200-trained models, with additive entry/competition contributions and paired intervals. Fifteen full-K exports and sixty sliced forward checks pass.
- [x] Locked full Test: 297,390 events / 112,081 users, six systems / twelve previously trained checkpoints, unchanged K200, pre-T0 graph and prefix scalers. No retraining or Test-driven changes. Twelve rank/top-ten replays, 54 metric reconstructions and 27 independent nDCG intervals pass; all candidate/history rows and 24 independent feature references pass.
- [x] Fixed 1,000-query Validation Flat/IVF/HNSW benchmark, 50 disjoint warm-ups per index, single-thread CPU, per-request feature/scoring/filtering/sorting costs. Apple M4 Pro / 24 GiB. Full-probe IVF and common-only feature references pass. Flat has 21 float32-near-tied boundary exchanges across 19 queries (scores within 2e-6); these are retained explicitly.
- [x] Four checked figures, compact CSV/JSON evidence, phase source snapshots and publication hashes saved under `results/steps_final/20261104/`; bulk arrays remain under `results/local_experiments/20261104-steps-final/`.
- [x] 243 tests passed / 4 legacy skips.
- [ ] Narrative report — deliberately deferred by the user; no new report document created.

The `report` CLI phase writes numerical tables and independent checks only. Historical RQ1 Test summaries were previously exposed; this is a locked follow-up, not globally blind project-wide Test. No further experimental sweep, graph refresh, retraining or canonical integration is started.
