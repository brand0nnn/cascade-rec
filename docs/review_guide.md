# Recommendation experiment review guide

This project studies how retrieval and ranking interact in offline next-review prediction. The [poster evidence map](evidence_map.md) separates study families and evaluation periods. The submitted [poster image](../poster-companion/assets/poster.png) is preserved; the current companion reports corrections alongside it.

## Evaluation setup

The Amazon Reviews 2023 Cell Phones and Accessories data contain 111,480 products. Each query is an interaction with nonempty prior user history; all star ratings count as observed interactions. Models receive at most the latest 50 strictly earlier interactions. Earlier events within Validation or Test may enter a later query's history; future and same-timestamp events do not.

Search filters products in the user's complete earlier history and products whose first review is later than the query. First review is an availability proxy, not a verified launch date. Natural evaluation never inserts a missed target. Ranker training either force-inserts missed targets (historical controls) or retains only naturally retrieved positives, according to the study.

For one relevant target, Recall@K is 1 if its rank is at most K; nDCG@10 is `1 / log2(rank + 1)` within the top ten and 0 otherwise. Main scores average within each user, then across users. Ranker results normally average three trained seeds. Paired intervals resample users 1,000 times after aligning and averaging seed-level event metrics. These intervals are conditional on the fits and unadjusted for multiple comparisons.

Implementation: [instances](../src/cascade/data/instances.py), [filters and evaluation](../src/cascade/eval/harness.py), [metrics](../src/cascade/eval/metrics.py), [bootstrap](../src/cascade/eval/bootstrap.py).

## Chronological split and its interpretation

Dates below use the dataset's UTC labels. [configs/data.yaml](../configs/data.yaml) defines the split rule and T1/T2; [recorded statistics](../results/stats/stats.json) include the derived T0 and exact period counts.

| Period | Cutoff | Interactions | Eligible queries |
|---|---|---:|---:|
| A | Before T0, 26 July 2020 | 1,877,548 | 1,526,356 |
| B | T0 to T1, 11 August 2021 | 331,332 | 315,114 |
| Validation | T1 to T2, 16 July 2022 | 239,437 | 232,904 |
| Test | At or after T2 | 304,468 | 297,390 |

T0 is the 85th-percentile interaction timestamp within the pre-T1 training portion, so A/B is an 85%/15% interaction split. The diagram does not represent four equal data partitions. [Split code](../src/cascade/data/split.py), [recorded dataset statistics](../results/stats/stats.json).

Training the original retriever on A and generating ranker-training candidates in B excludes B targets from retriever gradient updates. It offers a controlled candidate generator, but disjoint training periods are an additional design choice. The implementation does not establish that this choice is required or optimal. Retriever checkpoints are selected using later Validation performance, so the offline protocol is not a fully prospective deployment at T0.

The fresh retriever trains on A+B, including updated logQ counts, strictly before T1. R1 then serves its outputs to rankers trained on the old A-only retriever's B lists. Training data now overlap, but the ranker has not adapted to the new generator. This tests transfer to changed inputs; it does not test a cascade trained consistently with that fresh generator.

The statistics-age finding is particularly conditional on feature preparation. The residual studies used T0 snapshots throughout B, making them about 6.1 months old on training natural-hit queries, then 18.0 months old on Validation if T0 was reused. The knownness cue and observed lag preference depend on this construction. The matching-only MLP/DIN family omits review-count inputs and must not inherit that explanation automatically.

Changing K and candidate sources are controlled input interventions for the fitted models. They demonstrate effects within this protocol, including competition and sensitivity to score inputs. Whether their magnitude persists under shared-period or rolling retriever/ranker training remains untested. Comparing the same input interventions under an alternative training protocol, with future evaluation queries fixed, is needed to identify that dependence.

## Study families

| Family | Training and selection | Evidence |
|---|---|---|
| Historical forced-list rankers | B lists with appended missed targets; Validation checkpoint selection | Early collapsed K sweep and feature ladder |
| Natural matching MLP and DIN for R1 | B natural lists from A-only two-tower; saved Validation-selected October checkpoints | Frozen K200→500, fresh retriever and mixed-list Test interventions |
| Matching MLP and DIN for RQ3 | Separate November hybrid-study fits; Validation selection; two-tower or mixed lists | Original and corrected 8 November Test comparison |
| Residual text and co-review rankers | Chronological prefix/tuning split within B, then fixed-epoch all-B refit | R2 size control, R3 timing and R4 combined system |

R1's original MLP/DIN Test values are about 5.18/6.24 ×10⁻³; RQ3's separate controls are about 5.19/6.37 ×10⁻³. These are different trained checkpoints. Macro recall and event-micro list coverage also have different denominators.

## Test exposure and corrections

- **4 November:** full Test and subsequent restoration studies, preceding the later confirmations.
- **7 November:** registered residual-family systems and timing addendum, each under its owning protocol.
- **8 November:** matching-feature hybrid systems and faster-score variant selected on Validation. The subsequent scoring correction reuses training-consistent score statistics, preserving original outputs.
- **9 November:** post-hoc frozen-ranker R1 follow-up, with predictions locked before this particular run. It is explicitly a third look relative to the 7/8 November studies, not a globally untouched Test.
- **10 November:** score-input swaps are Validation-only.

The normalisation audit reproduces twelve saved Validation arrays and twelve historical Test arrays exactly before changing the mixed-list Test scaler. Corrected DIN mixed-list nDCG is 0.008268565 versus 0.005193694 for the matching MLP two-tower reference. The absolute gain is 0.003074871, paired-user 95% interval [0.002812303, 0.003353031]. The +59.20% contrast bundles pool/scoring and ranker changes. [Audit](../results/test_final_ranker/20261108-normalization-corrected/audit.json), [comparisons](../results/test_final_ranker/20261108-normalization-corrected/comparisons.csv).

Original result tables, protocols, source snapshots and hashes remain historical evidence. Corrected outputs have separate paths. No new fitting or Test scoring is part of the review cleanup.

## Reproduction and artifact availability

| Level | Requirements | What can be checked |
|---|---|---|
| Inspect aggregates | Committed repository; Python standard library for companion evidence tests | Displayed numbers, relative changes, source hashes and interval endpoints |
| Recompute metrics from ranks | Owning per-event rank arrays and user alignment | Per-user macro values and paired bootstraps |
| Rescore selected weights | Exact checkpoint, processed IDs/histories, candidates, vectors, scaler and features | Prediction replay; owning runtime/device where ties matter |
| Reproduce training | All preceding inputs, pinned environment, config, seeds and selection protocol | New fit; identical hyperparameters alone do not guarantee identical weights |

The [253-artifact allowlist](../results/ranker_artifacts.json) supplies the older six-checkpoint study. It does not supply the fresh retriever's weights, R1's October natural rankers, November hybrid checkpoints or all residual-study bulk arrays. These remain in the original local/cluster run locations; no public download bundle is registered. The owning paths and missing prerequisites are identified in the evidence map. Aggregate inspection works without them; full rescoring does not.

The fresh retriever can be regenerated with [fresh_two_tower.py](../src/cascade/pipelines/fresh_two_tower.py); R1 scoring is [frozen_ranker_followup.py](../src/cascade/pipelines/frozen_ranker_followup.py). Both require the original processed inputs and checkpoint dependencies. Historical fixed run IDs, collision guards and source/protocol hashes must be accounted for before a fresh reproduction.

Run aggregate-only checks from the repository root:

```sh
python3 -m unittest discover -s poster-companion -p 'test_*.py' -v
```

For the tested Python 3.11 environment and full unit suite, follow the [root setup](../README.md#run-the-python-project). Data-dependent tests can skip in a fresh clone. Historical source snapshots are provenance; executable modules live under `src/cascade/`.

## Limits on the conclusions

The data are reviews rather than exposure logs or online satisfaction measures. Crawl-time titles may contain later device names, and all-time 5-core filtering uses later interaction survival. Natural-only training changes population and optimiser exposure. Pool comparisons can change score functions, scalers and available positives together. New products are defined by first review in the evaluation period. These conditions limit causal attribution and generalisation even when within-study comparisons and their hashes are correct.
