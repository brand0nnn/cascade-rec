# Sourced by every Slurm job: go to the repo, activate the venv, print job info.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source .venv/bin/activate
export PYTHONWARNINGS=ignore
# fall back to a home-directory temp dir if the node's /tmp is full or not writable
if ! d=$(mktemp -d 2>/dev/null); then export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"; else rmdir "$d"; fi
mkdir -p results/logs
echo "job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) at $(date)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# fail fast if the environment is incomplete (e.g. an interrupted install), before any long step runs
python -c "import numpy, pandas, pyarrow, yaml, torch, recbole, faiss, sentence_transformers; print('env ok: torch', torch.__version__, '| cuda', torch.cuda.is_available())"
