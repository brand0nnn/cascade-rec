#!/usr/bin/env bash
# RQ1 retriever runs as a job array: Two-tower (text) and Two-tower (text + ID), seeds 0-2.   sbatch scripts/slurm/train_retriever.sbatch
# Run a subset with e.g.  sbatch --array=3-5 scripts/slurm/train_retriever.sbatch
#SBATCH --job-name=cascade-retriever
#SBATCH --output=results/logs/slurm-%x-%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
source scripts/slurm/common.sh
RUNS=("two_tower_text 0" "two_tower_text 1" "two_tower_text 2" "two_tower_text_id 0" "two_tower_text_id 1" "two_tower_text_id 2")
read -r VARIANT SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
python -m cascade.pipelines.train_retriever --config configs/retriever/${VARIANT}.yaml --seed ${SEED} --device cuda
