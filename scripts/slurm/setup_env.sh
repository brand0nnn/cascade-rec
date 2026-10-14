#!/usr/bin/env bash
# One-off environment setup on the SoC cluster login node (no GPU needed):
#   bash scripts/slurm/setup_env.sh
# Installs uv into ~/.local/bin if missing, then creates .venv with Python 3.11 and the pinned requirements.
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -e .
.venv/bin/python -c "import torch, recbole, faiss; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| recbole', recbole.__version__, '| faiss', faiss.__version__)"
