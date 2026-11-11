# A Ranker in Plato’s Cave — interactive companion

A local, dependency-free website accompanying the unchanged CP4285 Group 07 poster. It follows its RQ1–RQ3 and R1–R4 labels, adds evaluation methodology, and lets readers explore saved results.

## Open it

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory poster-companion
```

Open **http://127.0.0.1:8765**. Stop the server with Ctrl+C. You can also open `index.html` directly: the bundled evidence loads without a server or internet connection. CSV download behaviour can vary in file mode, so the local server is recommended.

No npm installation, Python packages, API keys, backend, model weights or dataset downloads are needed. The committed data bundle is sufficient for teammates to view everything.

## Pull the companion branch

```sh
git fetch origin
git switch --track origin/codex/poster-companion
```

If the branch already exists locally, use `git switch codex/poster-companion` and then `git pull --ff-only`.

## What is included

- **Guided story:** interactive pipeline and metrics, architecture explanations, retrieval groups, K sweep, same-pool ranker comparisons, frozen-ranker interventions, retraining controls, timing diagnostics and final comparisons.
- **Results explorer:** study-specific charts, exact values, available paired intervals, source references and aggregate CSV downloads.
- **Original poster:** an unchanged byte-for-byte copy in a keyboard-accessible zoom dialog.
- **Local assets:** no external fonts, scripts, images, analytics or requests.

Direct links include `#rq1`, `#rq2`, `#rq3`, `#r1`, `#r2`, `#r3`, `#r4`, `#methods`, and `#explorer/timing`.

## Data and interpretation

`export_data.py` reads existing aggregate results and writes the small bundles in `data/`. It never trains, scores, changes evaluation definitions, or edits source artifacts. Each source has a repository-relative path and SHA-256 digest in `evidence.json`; only selected aggregate fields are packaged, not complete source configurations or per-user records.

The page keeps study families, Validation/Test, macro/micro averaging and training/scoring changes separate. Its illustrative metric example is explicitly labelled. Main chart labels use nDCG × 10⁻³; tables and CSVs retain unscaled source numbers.

Important qualifications:

- The poster’s R2 four-condition chart uses **Validation**. Its Test confirmation is a separate comparison.
- Exact R1 poster percentages are a labelled transcription; their detailed run export is unavailable in this checkout.
- Current mixed-list training/Test code has an unresolved score-normalisation provenance discrepancy. Saved results are shown with that qualification, not declared incorrect.
- Cross-family final-system ratios have no supplied paired interval. A combined warm-product macro is unavailable for the original two-tower pipeline and is not inferred from head/tail slices.
- Statistical intervals are paired user resamples, conditional on trained seeds and unadjusted. Seed standard deviations are not confidence intervals.

## Regenerate and check

The tracked source summaries must be present in their original repository locations:

```sh
python3 poster-companion/export_data.py
python3 -m unittest discover -s poster-companion -p 'test_*.py' -v
node --check poster-companion/app.js
```

Regeneration is deterministic and fails when a required source is missing rather than substituting another study. The committed bundle can always be viewed without regeneration.

All website changes are isolated under `poster-companion/`. The original poster, existing HTML poster drafts, evaluation data and model code are not edited. Publication is deferred; the server binds only to the local computer.
