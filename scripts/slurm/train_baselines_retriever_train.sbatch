#!/usr/bin/env bash
# Baselines trained on the retriever-training period only, for the RQ1 table (same training data as the retriever).
#   sbatch scripts/slurm/train_baselines_retriever_train.sbatch
# Popularity / ItemKNN are deterministic and already done.
#SBATCH --job-name=cascade-baselines-rt
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
source scripts/slurm/common.sh
RUNS=("sasrec 0" "sasrec 1" "sasrec 2" "bpr 0" "bpr 1" "bpr 2")
read -r MODEL SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
python -m cascade.pipelines.train_baseline --config configs/baselines/${MODEL}.yaml --train-on retriever_train --seed ${SEED} --device cuda
