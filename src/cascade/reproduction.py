"""Read-only inventory of the inputs needed by each poster study.

This command never imports an experiment runner, loads prediction arrays, creates
output directories, downloads inputs, or executes models. Original run guards and
protocol locks remain authoritative when a researcher later runs an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "configs/reproduction"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def input_paths(record: dict) -> list[str]:
    """Expand explicit seed/arm inventories; an absent match is never silently skipped."""
    expand = record.get("expand", {})
    keys = list(expand)
    result = []
    for values in product(*(expand[key] for key in keys)):
        variables = dict(zip(keys, values))
        base = record["path"].format(**variables)
        for filename in record.get("files", [""]):
            path = Path(base) / filename.format(**variables)
            if path.is_absolute() or ".." in path.parts or path == Path("."):
                raise ValueError(f"input must be a repository-relative file: {path}")
            result.append(path.as_posix())
    if not result:
        raise ValueError("an input expansion must contain at least one file")
    return result


def inspect_study(manifest: dict, phase: str, *, root: Path = ROOT,
                  hash_files: bool = False, fresh_output: Path | None = None) -> dict:
    """Check presence and supplied identities; this is not a certificate of score reproducibility."""
    if manifest.get("schema") != 1:
        raise ValueError("unsupported reproduction manifest schema")
    spec = manifest["phases"][phase]
    records = []
    seen = set()
    for group in spec["groups"]:
        for record in manifest["inputs"][group]:
            paths = input_paths(record)
            if record.get("sha256") and len(paths) != 1:
                raise ValueError("a recorded checksum must identify exactly one file")
            for name in paths:
                if name in seen:
                    continue
                seen.add(name)
                path = root / name
                present = path.is_file()
                expected = record.get("sha256")
                actual = digest(path) if present and (expected or hash_files) else None
                records.append({"path": name, "group": group, "present": present,
                                "bytes": path.stat().st_size if present else None,
                                "sha256": actual, "expected_sha256": expected,
                                "identity_matches": actual == expected if present and expected else None})
    missing = [row["path"] for row in records if not row["present"]]
    changed = [row["path"] for row in records if row["identity_matches"] is False]
    output = None
    if fresh_output is not None:
        path = fresh_output if fresh_output.is_absolute() else root / fresh_output
        occupied = path.exists() or path.is_symlink()
        ancestor = path.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        output = {"path": str(path), "available": not occupied and ancestor.is_dir()}
    available = not missing and not changed and (output is None or output["available"])
    return {"study": manifest["study"], "phase": phase, "status": "ready" if available else "not_ready",
            "scope": "file presence and recorded checksums; no model/array execution or transitive dependency audit",
            "entrypoint": spec["entrypoint"], "inputs": records, "missing": missing, "changed": changed,
            "historical_output_roots": manifest.get("historical_output_roots", []), "fresh_output": output,
            "models_executed": False, "prediction_arrays_loaded": False, "outputs_written": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(p.stem for p in MANIFESTS.glob("*.json")))
    parser.add_argument("--phase", default="inspect", help="inspect (aggregates), or a named manifest phase")
    parser.add_argument("--list", action="store_true", help="list registered studies and phases")
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout/input root; does not redirect a runner")
    parser.add_argument("--hash", action="store_true", help="also hash present bulk inputs; potentially expensive")
    parser.add_argument("--fresh-output", type=Path, help="check an unused destination; never creates or redirects it")
    parser.add_argument("--require-ready", action="store_true", help="exit 2 on absent/changed inputs or occupied output")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({p.stem: list(json.loads(p.read_text())["phases"])
                          for p in sorted(MANIFESTS.glob("*.json"))}, indent=2))
        return
    if args.study is None:
        parser.error("choose --study or --list")
    manifest = json.loads((MANIFESTS / (args.study + ".json")).read_text())
    if args.phase not in manifest["phases"]:
        parser.error("phase must be one of: " + ", ".join(manifest["phases"]))
    report = inspect_study(manifest, args.phase, root=args.root.resolve(), hash_files=args.hash,
                           fresh_output=args.fresh_output)
    print(json.dumps(report, indent=2))
    if args.require_ready and report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
