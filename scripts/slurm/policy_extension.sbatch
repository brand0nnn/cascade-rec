#!/usr/bin/env bash
# Policy-study extension (E/F/G/H with fresh A/C controls) for ONE seed.
# Submit from the repository root:
#   sbatch --export=ALL,SEED=0,RUN_ID=ext-20261107-s0 scripts/slurm/policy_extension.sbatch
#SBATCH --job-name=cascade-policy-ext
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
set -euo pipefail
: "${SEED:?set SEED to 0, 1 or 2}"
: "${RUN_ID:?set a fresh RUN_ID}"
case "$RUN_ID" in .|..|*[!a-zA-Z0-9_-]*) echo "bad RUN_ID" >&2; exit 2 ;; esac
repo="$(pwd)"
py="${PY:-$repo/.venv/bin/python}"
export PYTHONPATH="$repo/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
echo "job ${SLURM_JOB_ID:-local} seed $SEED on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$py" -c 'import cascade, torch; print("cascade from", cascade.__file__); assert torch.cuda.is_available()'
"$py" -m cascade.pipelines.ranker_policy_study --stage extension --seeds "$SEED" \
  --arms A C E F G H --run-id "$RUN_ID" --device cuda
echo "Completed extension seed $SEED ($RUN_ID). No Test evaluation was run."
