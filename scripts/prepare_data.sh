#!/usr/bin/env bash
# Usage: scripts/prepare_data.sh [--sample]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY -m cascade.data.download --config configs/data.yaml
$PY -m cascade.data.preprocess --config configs/data.yaml "$@"
$PY -m cascade.data.split --config configs/data.yaml "$@"
$PY -m cascade.data.instances --config configs/data.yaml "$@"
$PY -m cascade.data.stats --config configs/data.yaml "$@"
