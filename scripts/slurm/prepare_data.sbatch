#!/usr/bin/env bash
# Download + preprocess the dataset, build toy data, encode item texts, run the tests.  sbatch scripts/slurm/prepare_data.sbatch
#SBATCH --job-name=cascade-prep
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source scripts/slurm/common.sh
scripts/prepare_data.sh
scripts/prepare_data.sh --toy
python -m cascade.text.encode --device cuda
python -m cascade.text.encode --toy
# raw files must be byte-identical to the ones used locally (see docs/decisions.md)
python - <<'PY'
import json
m = json.load(open("data/raw/MANIFEST.json"))
expected = {"Cell_Phones_and_Accessories.csv.gz": "63cfedbddeed0bf2c37a619f6712c4e3a330801933868fdd23128aabca9f8ce0",
            "meta_Cell_Phones_and_Accessories.jsonl.gz": "bfc3e7b93298a6c5591c091d532704b6c5bf32e2c8d34bad94a3b0fe93173430"}
for f, h in expected.items():
    assert m[f]["sha256"] == h, f"checksum mismatch for {f}"
print("raw data checksums match")
PY
python -m pytest -q
