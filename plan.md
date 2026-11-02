# Local experimentation plan

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
