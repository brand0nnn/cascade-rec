#!/usr/bin/env bash
# Local launch of all full-data baselines (Apple Silicon): SASRec on MPS (2 queues), BPR/Pop/ItemKNN on CPU.
# Logs in results/logs/. Summary table: results/baselines/summary.md
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONWARNINGS=ignore
mkdir -p results/logs
run() { $PY -m cascade.pipelines.train_baseline --config configs/baselines/$1.yaml --train-on $2 --seed $3 --device $4 > results/logs/$1_$2_s$3.log 2>&1; }
( export OMP_NUM_THREADS=2; run sasrec AB 0 mps; run sasrec AB 2 mps; run sasrec A 1 mps ) &
( export OMP_NUM_THREADS=2; run sasrec AB 1 mps; run sasrec A 0 mps; run sasrec A 2 mps ) &
( export OMP_NUM_THREADS=4
  for S in 0 1 2; do run bpr AB $S cpu; done
  run pop AB 0 cpu; run itemknn AB 0 cpu; run pop A 0 cpu; run itemknn A 0 cpu
  for S in 0 1 2; do run bpr A $S cpu; done ) &
wait
$PY -m cascade.eval.report > results/logs/report.log 2>&1
