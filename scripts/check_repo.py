#!/usr/bin/env python3
"""Check code and committed evidence without preparing data or launching experiments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-only", action="store_true",
                        help="check companion sources/bundles with standard-library Python and Node.js")
    args = parser.parse_args()
    node = shutil.which("node")
    if node is None:
        parser.error("Node.js is required for JavaScript checks; Python-only commands are in README.md")
    commands = []
    if not args.evidence_only:
        commands.append([sys.executable, "-m", "pytest", "-q", "-ra"])
    commands += [
        [sys.executable, "poster-companion/export_data.py", "--check"],
        [sys.executable, "-m", "unittest", "discover", "-s", "poster-companion", "-p", "test_*.py", "-v"],
        [node, "--check", "poster-companion/app.js"],
        [node, "--check", "poster-companion/visuals.js"],
        [node, "--check", "poster-companion/data/evidence.js"],
        [node, "--test", "poster-companion/test_visuals.cjs"],
    ]
    env = {**os.environ, "MPLBACKEND": "Agg", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    print("Repository checks passed. No research run was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
