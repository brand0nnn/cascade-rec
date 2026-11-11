# A Ranker in Plato’s Cave — interactive companion

A local, dependency-free website accompanying the unchanged CP4285 Group 07 poster. It follows its RQ1–RQ3 and R1–R4 labels, adds evaluation methodology, and lets readers explore saved results.

## Open it

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory poster-companion
```

Open **http://127.0.0.1:8765**. Stop the server with Ctrl+C. You can also open `index.html` directly: the bundled evidence loads without a server or internet connection. The local server is recommended for consistent navigation between sections.

No npm installation, Python packages, API keys, backend, model weights or dataset downloads are needed. The committed data bundle is sufficient for teammates to view everything.

## Repository review

The companion, the newer R1 follow-up and the scoring correction are reconciled on the local `codex/review-ready` branch. The [root README](../README.md), [review guide](../docs/review_guide.md) and [evidence map](../docs/evidence_map.md) are the review entry points.

## What is included

- **Guided story:** interactive pipeline and metrics, architecture explanations, retrieval groups, K sweep, same-pool ranker comparisons, frozen-ranker interventions, retraining controls, timing diagnostics and final comparisons.
- **Results explorer:** study-specific charts, exact values, available paired intervals, paired-difference plots and source references, all readable on the page. Visitor download links are intentionally omitted; aggregate CSV files remain in the repository for reproducibility.
- **Original poster:** an unchanged byte-for-byte copy in a keyboard-accessible zoom dialog.
- **Visual explanations:** candidate-pool composition, linked K trends, gain/competition waterfalls, retraining-controls matrix, review-cutoff timeline, co-review graph and paired uncertainty plots.
- **Methodology walkthrough:** query construction, natural-only training, end-to-end evaluation, per-user averaging, and paired bootstrap comparisons. Conceptual examples are explicitly labelled.
- **Local assets:** no external fonts, scripts, images, analytics or requests.

Direct links include `#rq1`, `#rq2`, `#rq3`, `#r1`, `#r2`, `#r3`, `#r4`, `#methods`, and `#explorer/timing`.

## Data and interpretation

`export_data.py` reads existing aggregate results and writes the small bundles in `data/`. It never trains, scores, changes evaluation definitions, or edits source artifacts. Each source has a repository-relative path and SHA-256 digest in `evidence.json`; only selected aggregate fields are packaged, not complete source configurations or per-user records.

The page keeps study families, Validation/Test, macro/micro averaging and training/scoring changes separate. Its illustrative metric example is explicitly labelled. Main chart labels use nDCG × 10⁻³; tables and CSVs retain unscaled source numbers.

Important qualifications:

- The poster’s R2 four-condition chart uses **Validation**. Its Test confirmation is a separate comparison.
- R1 now uses the committed 9 November Test follow-up, with paired intervals. The original poster prints 83% for a saved 82.41% K500 recall increase, and 838 displaced MLP successes where the saved count is 638. The poster image is preserved.
- RQ3 mixed-list scores use the completed training-consistent normalisation correction and recomputed paired intervals. Original scores remain in their historical folder; corrected aggregates have their own provenance.
- The companion’s original-pipeline RQ3 reference differs from R1’s October reference. A separate paired R4-vs-R1 comparison is in the evidence map. Established-product macro values are not inferred by averaging head/tail slices.
- Statistical intervals are paired user resamples, conditional on trained seeds and unadjusted. Seed standard deviations are not confidence intervals.

## Regenerate and check

The tracked source summaries must be present in their original repository locations:

```sh
python3 poster-companion/export_data.py
python3 -m unittest discover -s poster-companion -p 'test_*.py' -v
node --check poster-companion/app.js
node --test poster-companion/test_visuals.cjs
```

Regeneration is deterministic and fails when a required source is missing rather than substituting another study. The committed bundle can always be viewed without regeneration.

Website changes are under `poster-companion/`; the review also adds documentation and the separate rank-only aggregate report. The original poster, HTML poster drafts and historical evaluation data are preserved. The preview server binds only to the local computer; this cleanup does not publish the site.
