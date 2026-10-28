#!/usr/bin/env bash
# Submit the ranker policy study one stage at a time. Run on the cluster login node; check each stage's log and
# status.json before submitting the next.
#   scripts/slurm/submit_policy_study.sh check        # inputs present + candidate hashes match the frozen study (no job)
#   scripts/slurm/submit_policy_study.sh development  # 4 arms, fixed 2% of users, 1 epoch (minutes)
#   scripts/slurm/submit_policy_study.sh pilot        # seed 0, arms A-D (~2-3.5 h)
#   scripts/slurm/submit_policy_study.sh replication  # seeds 1 and 2 as two parallel jobs (~2-3.5 h each)
#   scripts/slurm/submit_policy_study.sh report       # combine pilot + replication once all three are complete
# Env: CANDIDATE_DIR (default data/processed/candidates/two_tower_text_seed0; must hold the ORIGINAL frozen lists),
#      DATA_CONFIG (default configs/data.yaml), TAG (run-ID suffix, default today's date; reuse it for every stage).
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?usage: $0 check|development|pilot|replication|report}"
TAG="${TAG:-$(date +%Y%m%d)}"
export CANDIDATE_DIR="${CANDIDATE_DIR:-data/processed/candidates/two_tower_text_seed0}"
export DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
SCRIPT=scripts/slurm/policy_study.sbatch

submit() {  # submit <stage> <run-id> [seeds]
  if [ -e "results/ranker_policy/$2" ] || [ -e "results/ranker_policy/$2_sample" ]; then
    echo "results/ranker_policy/$2 already exists; choose another TAG" >&2; exit 1
  fi
  sbatch --export=ALL,STAGE="$1",RUN_ID="$2",SEEDS="${3:-}" "$SCRIPT"
  echo "submitted $1 as $2 (results/ranker_policy/$2${4:-})"
}

case "$STAGE" in
  check)
    source .venv/bin/activate
    python - <<'EOF'
import json, os
from cascade.pipelines.ranker_policy_study import _path, digest, required_input_paths
from cascade.utils.config import load_config
cfg = load_config(_path("configs/ranker/policy_study.yaml"))
cfg["candidate_dir"], cfg["data_config"] = os.environ["CANDIDATE_DIR"], os.environ["DATA_CONFIG"]
missing = [str(p) for p in required_input_paths(cfg) if not p.is_file()]
if missing:
    raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
frozen = json.loads(_path(cfg["reference_comparison"]).read_text())["candidate_source"]
cand = _path(cfg["candidate_dir"])
for name, key in (("meta.json", "meta_sha256"), ("val.npz", "val_sha256")):
    ok = digest(cand / name) == frozen[key]
    print(f"{name}: {'matches' if ok else 'DIFFERS from'} the frozen study")
    if not ok:
        raise SystemExit("use the original frozen candidate lists (set CANDIDATE_DIR); regenerated lists are refused")
print("all inputs present; candidate hashes match")
EOF
    ;;
  development) submit development "policy-development-$TAG" "" "_sample" ;;
  pilot)       submit pilot "policy-pilot-$TAG" ;;
  replication)
    submit replication "policy-replication-s1-$TAG" 1
    submit replication "policy-replication-s2-$TAG" 2
    ;;
  report)
    out="results/targeted_study/policy-report-$TAG"
    sbatch --job-name=cascade-policy-report --output=results/logs/slurm-%x-%j.out --time=01:00:00 \
      --cpus-per-task=4 --mem=32G --partition=gpu-long --gres=gpu:a100-40:1 \
      --wrap="source scripts/slurm/common.sh && python -m cascade.eval.ranker_policy_report --studies \
results/ranker_policy/policy-pilot-$TAG results/ranker_policy/policy-replication-s1-$TAG \
results/ranker_policy/policy-replication-s2-$TAG --output $out"
    echo "submitted report -> $out"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
