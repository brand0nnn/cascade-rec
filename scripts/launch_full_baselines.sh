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
  for S in 0 1 2; do run sasrec all_train $S mps; done
  for S in 0 1 2; do run sasrec retriever_train $S mps; done ) &
( export OMP_NUM_THREADS=4
  for S in 0 1 2; do run bpr all_train $S cpu; done
  run popularity all_train 0 cpu; run itemknn all_train 0 cpu; run popularity retriever_train 0 cpu; run itemknn retriever_train 0 cpu
  for S in 0 1 2; do run bpr retriever_train $S cpu; done ) &
wait
$PY -m cascade.eval.report > results/logs/report.log 2>&1
