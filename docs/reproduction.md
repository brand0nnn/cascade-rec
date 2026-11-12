# Study prerequisites and safe report destinations

The input inventories under [configs/reproduction](../configs/reproduction/) cover RQ1–RQ3 and R1–R4. They distinguish committed aggregate inspection from selected saved-rank, model and candidate preparation phases. Paths identify the original study families and seeds; an absent seed is reported explicitly rather than skipped through a wildcard.

List the studies and phases, or check the committed R1 aggregate identities:

```sh
.venv/bin/python -m cascade.reproduction --list
.venv/bin/python -m cascade.reproduction --study r1 --phase inspect --require-ready
```

Inspect the files needed by the October matching-ranker R1 Validation phase:

```sh
.venv/bin/python -m cascade.reproduction --study r1 --phase din-val --require-ready
```

The JSON report lists each file, presence, size, any recorded checksum and mismatches. Missing inputs or mismatched identities cause exit code 2 with `--require-ready`; otherwise the inventory exits successfully and reports `not_ready`. No dataset is downloaded, no prediction array is loaded, no model executes, and no output directory is created. The commands do not open a new Test evaluation.

Use `--root /path/to/checkout` to inspect another local input tree. `--hash` additionally hashes present bulk files and can be expensive. Recorded checksums, including aggregate and protocol identities, are checked even without that flag. Unhashed weights and data are explicitly unverified; file presence alone does not establish that they are the original inputs.

## Output planning

Historical run IDs and CLI defaults remain unchanged. A preflight can check a proposed fresh destination without creating it:

```sh
.venv/bin/python -m cascade.reproduction --study r1 --phase inspect \
  --fresh-output results/review-r1/YOUR_FRESH_ID --require-ready
```

`--fresh-output` is a planning check. It does **not** redirect an experiment. An existing file, directory or symlink is reported as occupied. For the November matching study, the existing `--run-id` and `--out` arguments continue to own output selection and collision guards.

R1 now exposes an explicit context for a separate rank-only report destination. Researchers who already have the saved ranks can use the reporting module directly:

```python
from dataclasses import replace
from cascade.pipelines import frozen_ranker_followup as r1
from cascade.pipelines.frozen_followup import reporting

destination = r1.REPO_ROOT / "results/review-r1/YOUR_FRESH_ID"
destination.mkdir(parents=True, exist_ok=False)
context = replace(r1.study_context(), out=destination)
reporting.phase_report(context)
```

This retains the original bulk inputs, protocol, seeds and calculations. The report includes existing Test ranks if their original Test-open markers exist, exactly as the historical reporter did. It performs no scoring or fitting. The fresh-directory check prevents replacing an earlier report. This cleanup does not execute that example against the research data.

## Scope of an inventory

`ready` means the files listed for that inventory phase are present and its recorded hashes match. The inventories do not traverse every source/data dependency of every runner, certify tensor shapes, restore missing weights, or establish hardware equivalence. The original programs still enforce their source/selection locks, population alignment, feature contracts and scoring devices. See the [artifact availability guide](review_guide.md#reproduction-and-artifact-availability) before attempting a full replay.

The matching-family November recipe is in [matching_recipe.py](../src/cascade/pipelines/matching_recipe.py). R1 keeps its distinct October checkpoints. The original R1 CLI delegates to candidate, matching, residual, reporting, diagnostic and runtime modules under [frozen_followup](../src/cascade/pipelines/frozen_followup/). It retains its original flags, paths, devices, guards and phase order.

Original tables, protocols, source snapshots and recorded hashes are preserved. The [compatibility references](../tests/fixtures/refactor_20261112/manifest.json) archive the pre-refactor source from commit `48db428`, including the training/audit source matching the completed normalisation audit. Tests compare the moved operations and seeded model construction with these references. Existing provenance describes the original code at its recorded revision; it has not been rewritten to imply the historical results came from the new modules.
