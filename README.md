# Retrieval and ranking study

CP4285 Group 7 · Brandon Kang, Tong Jing Yen, Phua Qing Rong · National University of Singapore

We study how retrieval candidates, ranker training and feature timing affect the ten products a recommender returns. The dataset is Amazon Reviews 2023, **Cell Phones and Accessories**: 111,480 products, 232,904 Validation queries and 297,390 Test queries. Reviews are observed interactions, not verified purchases.

**Main finding:** higher candidate recall does not guarantee higher end-to-end nDCG@10 for a frozen ranker. The effect depends on the ranker and the changed inputs. Mixed lists can help without retraining; the statistics-age shortcut is conditional on the snapshot and training protocol. A consistently updated retriever/ranker training protocol has not been compared with our original A/B split.

## Start the review

| Read or inspect | Purpose |
|---|---|
| [Review guide](docs/review_guide.md) | Methods, chronological split, study families, Test exposure and reproduction limits |
| [Poster evidence map](docs/evidence_map.md) | RQ1–RQ3 and R1–R4 mapped to exact tables, checkpoints and code |
| [Interactive companion](poster-companion/README.md) | Explore current aggregate results, paired intervals and the preserved poster |
| [Submitted poster image](poster-companion/assets/poster.png) | Original poster, unchanged; use the evidence map for corrections and qualifications |
| [Evaluation record](docs/evaluations.md) | Dated experiment history, including negative results and the normalisation correction |

The original poster and historical exports remain unchanged. The companion and review documents distinguish corrected numbers from the submitted versions. Earlier HTML/PDF drafts are under [results/poster_draft3](results/poster_draft3/).

## Findings and their scope

| Question | Supported result | Evaluation and source |
|---|---|---|
| Does better retrieval help a frozen ranker? | Fresh-retriever recall rises 4.385% → 6.989%; MLP nDCG changes −1.25% (inconclusive), DIN −4.58% (negative paired interval). Frozen rankers on mixed lists instead gain about 24% / 10%. | Test; three ranker seeds, one fresh-retriever seed. [R1 follow-up](results/analysis/20261109-frozen-ranker-followup/tables.md) |
| Does training on mixed lists help? | The residual MLP rises 0.008141 → 0.010387 with training-list count matched; full mixed training reaches 0.011019. Inputs, score function and optimisation exposure also change. | Validation. [R2 control](results/analysis/20261107-pool-size-control/tables.md) |
| Does feature timing matter? | The same residual recency-list ranker scores 0.006439 with T0 statistics and 0.007815 with Test-start statistics. | Test; fixed candidates and weights. [Timing diagnostics](results/test_final/20261107/timing/README.md) |
| How good is the combined system? | Mixed lists, monthly statistics and co-review inputs reach 0.010379 versus the matched recency/T0 reference 0.006439. This bundles several changes. | Test. [Residual-family comparison](results/test_final/20261107/tables.md) |
| Does the scoring correction change the matching-family headline? | Corrected DIN mixed-list nDCG is 0.008268565 versus MLP two-tower 0.005193694: +59.20%. Twelve Validation and twelve historical Test arrays replay exactly. | Test; separate matching-feature family. [Correction audit](results/test_final_ranker/20261108-normalization-corrected/audit.json) |

All nDCG values above are **unscaled, per-user macro nDCG@10**, with retrieval misses counted as zero. The first row's recall is also per-user macro; event-micro coverage differs. Paired-user intervals are conditional on the fitted seeds and unadjusted for the many comparisons. See the evidence map before combining rows from different study families.

## Inspect results without model execution

The committed CSV/JSON tables and companion require no dataset, weights, GPU or API keys.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory poster-companion
```

Open [the local companion](http://127.0.0.1:8765). It can also be opened directly from `poster-companion/index.html`.

Check aggregate values, source hashes and portable assets using the Python standard library:

```sh
python3 -m unittest discover -s poster-companion -p 'test_*.py' -v
```

Optional JavaScript checks need Node.js:

```sh
node --check poster-companion/app.js
node --test poster-companion/test_visuals.cjs
```

## Run the Python project

The tested environment is Python 3.11 with the pinned packages in `requirements.txt`.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
.venv/bin/python -m pytest -q
```

The package declares the same pinned dependencies as `requirements.txt`; Python 3.11 is the tested runtime. The core split, history, filtering and candidate tests always run on isolated synthetic reviews, with additional checks on the prepared 2% sample when available. Missing bulk/sample inputs skip only their optional checks. Passing unit tests does not reproduce training or establish the generality of a research claim.

With Node.js available, run all code and evidence checks through one command:

```sh
.venv/bin/python scripts/check_repo.py
```

For the aggregate-only check, use `python3 scripts/check_repo.py --evidence-only`. The command verifies export freshness without rewriting bundles, then checks evidence and JavaScript. [CI](.github/workflows/checks.yml) runs the same checks on Python 3.11 without research data. The [code review](docs/code_review.md) records completed fixes and the remaining structural work.

Rescoring and training require additional artifacts. The [review guide](docs/review_guide.md#reproduction-and-artifact-availability) distinguishes aggregate inspection, rank-based metric recomputation, checkpoint scoring and new training. The [older artifact allowlist](docs/ranker_artifacts.md) covers the historical six-checkpoint study; it does not supply every current poster checkpoint.

The [study prerequisite inventories](docs/reproduction.md) list inputs by study/phase and report missing files without executing an experiment. For example, `.venv/bin/python -m cascade.reproduction --study r1 --phase inspect --require-ready` checks R1's committed aggregate identities. Model-scoring phases need additional bulk inputs. Fresh destination checks and the refactored R1 module map are documented in that guide.

The [historical setup instructions](docs/history/README-20261107.md) describe data preparation and the Slurm workflow. Before running a study, check its owning protocol/configuration and input requirements in the evidence map. Use fresh output IDs; historical pipelines may reject existing destinations and changed source hashes. These review commands launch no experiment.

## Implementation map

| Component | Code |
|---|---|
| Periods and strict prior histories | [split.py](src/cascade/data/split.py), [instances.py](src/cascade/data/instances.py) |
| Filtering, evaluation and metrics | [harness.py](src/cascade/eval/harness.py), [metrics.py](src/cascade/eval/metrics.py), [bootstrap.py](src/cascade/eval/bootstrap.py) |
| Two-tower training and candidates | [train_retriever.py](src/cascade/pipelines/train_retriever.py), [build_candidates.py](src/cascade/pipelines/build_candidates.py) |
| Matching-feature MLP and DIN | [ranker_mlp.py](src/cascade/models/ranker_mlp.py), [ranker_din.py](src/cascade/models/ranker_din.py), [run_hybrid_study.py](scripts/run_hybrid_study.py) |
| Frozen-ranker interventions | [frozen_ranker_followup.py](src/cascade/pipelines/frozen_ranker_followup.py) |
| Residual ranker and pool controls | [full_data_ranking.py](src/cascade/pipelines/full_data_ranking.py), [pool_ranking.py](src/cascade/pipelines/pool_ranking.py), [pool_size_control.py](src/cascade/pipelines/pool_size_control.py) |
| Statistics timing and monthly training | [statistics_timing.py](src/cascade/analysis/statistics_timing.py), [statistics_timing_test.py](src/cascade/analysis/statistics_timing_test.py), [rolling_ranking.py](src/cascade/pipelines/rolling_ranking.py) |

## Historical material

[Research narrative](docs/research_story.md), [experiment overview](docs/experiment_overview.md), [decisions](docs/decisions.md), [initial plan](docs/initial_plan_20260929.md) and [local study plan](plan.md) retain the development history. Dated restrictions, counts and results apply to their stated studies. `CLAUDE.md` contains implementation guidance for coding agents; it is not the review entry point.
