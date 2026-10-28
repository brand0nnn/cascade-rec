#!/usr/bin/env bash
# Ranker policy study (docs/evaluations.md, "Authorised follow-up: controlled ablations"), one stage per job.
# User-submitted; no automatic next stage. Easiest entry point: scripts/slurm/submit_policy_study.sh <stage>.
# Set STAGE=development|pilot|replication and a fresh RUN_ID. Optional: SEEDS="1" (replication only, so seeds can
# run as separate jobs), CANDIDATE_DIR (the original frozen candidates), DATA_CONFIG.
# Ranker training is mostly CPU-bound feature building; the GPU partition matches the other cascade jobs.
#SBATCH --job-name=cascade-policy
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
source scripts/slurm/common.sh
: "${STAGE:?set STAGE to development, pilot or replication}"
: "${RUN_ID:?set a fresh RUN_ID}"
case "$STAGE" in development|pilot|replication) ;; *) echo "unknown STAGE: $STAGE" >&2; exit 2 ;; esac
extra=()
if [ -n "${SEEDS:-}" ]; then read -r -a seed_list <<< "$SEEDS"; extra+=(--seeds "${seed_list[@]}"); fi
if [ -n "${CANDIDATE_DIR:-}" ]; then extra+=(--candidate-dir "$CANDIDATE_DIR"); fi
python -m cascade.pipelines.ranker_policy_study --stage "$STAGE" --run-id "$RUN_ID" \
  --data-config "${DATA_CONFIG:-configs/data.yaml}" --device cuda ${extra[@]+"${extra[@]}"}
