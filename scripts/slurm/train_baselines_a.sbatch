#!/usr/bin/env bash
# A-only baselines for the RQ1 table (same training data as the retriever).   sbatch scripts/slurm/train_baselines_a.sbatch
# Pop / ItemKNN (A) are deterministic and already done.
#SBATCH --job-name=cascade-baselines-a
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
source scripts/slurm/common.sh
RUNS=("sasrec 0" "sasrec 1" "sasrec 2" "bpr 0" "bpr 1" "bpr 2")
read -r MODEL SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
python -m cascade.pipelines.train_baseline --config configs/baselines/${MODEL}.yaml --train-on A --seed ${SEED} --device cuda
