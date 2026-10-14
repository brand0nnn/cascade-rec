#!/usr/bin/env bash
# One-off environment setup on the SoC cluster login node (no GPU needed):
#   bash scripts/slurm/setup_env.sh
# Installs uv into ~/.local/bin if missing, then creates .venv with Python 3.11 and the pinned requirements.
# Safe to re-run: an existing .venv is reused and already-installed packages are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
# the login node's /tmp has a small quota; keep temporary files (uv install, wheel builds) in the home directory
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
# the login node limits memory per user: install one package at a time with few threads
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 TOKIO_WORKER_THREADS=2
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
.venv/bin/python -c "import torch, recbole, faiss; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| recbole', recbole.__version__, '| faiss', faiss.__version__)"
