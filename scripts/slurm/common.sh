# Sourced by every Slurm job: go to the repo, activate the venv, print job info.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source .venv/bin/activate
export PYTHONWARNINGS=ignore
mkdir -p results/logs
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
