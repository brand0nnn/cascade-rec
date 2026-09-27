#!/usr/bin/env bash
# Local launch of all full-data baselines on a 16 GB Apple-Silicon laptop:
# one SASRec queue on MPS (each run needs ~3 GB) + one CPU queue for BPR/Pop/ItemKNN.
# Logs in results/logs/. Summary table: results/baselines/summary.md
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONWARNINGS=ignore
mkdir -p results/logs
run() { $PY -m cascade.pipelines.train_baseline --config configs/baselines/$1.yaml --train-on $2 --seed $3 --device $4 > results/logs/$1_$2_s$3.log 2>&1; }
( export OMP_NUM_THREADS=2
  for S in 0 1 2; do run sasrec AB $S mps; done
  for S in 0 1 2; do run sasrec A $S mps; done ) &
( export OMP_NUM_THREADS=4
  for S in 0 1 2; do run bpr AB $S cpu; done
  run pop AB 0 cpu; run itemknn AB 0 cpu; run pop A 0 cpu; run itemknn A 0 cpu
  for S in 0 1 2; do run bpr A $S cpu; done ) &
wait
$PY -m cascade.eval.report > results/logs/report.log 2>&1
