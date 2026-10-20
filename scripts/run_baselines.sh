#!/usr/bin/env bash
# Train + evaluate all Phase-0 baselines. Usage: scripts/run_baselines.sh [--sample] [--device cuda]
# Deterministic baselines (Pop, ItemKNN) run seed 0 only; BPR/SASRec run seeds 0,1,2.
# Main setting trains on all training data; the retriever-training-period-only variant is also logged (CLAUDE.md §4.4).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export PYTHONWARNINGS=ignore
$PY -m pytest -q
for TR in all_train retriever_train; do
  for M in popularity itemknn; do
    $PY -m cascade.pipelines.train_baseline --config configs/baselines/$M.yaml --train-on $TR --seed 0 "$@"
  done
  for M in bpr sasrec; do
    for S in 0 1 2; do
      $PY -m cascade.pipelines.train_baseline --config configs/baselines/$M.yaml --train-on $TR --seed $S "$@"
    done
  done
done
$PY -m cascade.eval.report "$([[ " $* " == *" --sample "* ]] && echo --sample)"
