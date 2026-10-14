#!/usr/bin/env bash
# Candidate lists from the frozen retriever (run after it is chosen and recorded in docs/decisions.md):
#   sbatch --export=ALL,RUN_DIR=results/retriever/v2_transformer/seed0 scripts/slurm/build_candidates.sbatch
#SBATCH --job-name=cascade-candidates
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source scripts/slurm/common.sh
: "${RUN_DIR:?set RUN_DIR to the frozen retriever run directory}"
python -m cascade.pipelines.build_candidates --run-dir "${RUN_DIR}" --device cuda
