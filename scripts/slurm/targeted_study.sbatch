#!/usr/bin/env bash
# Submit from the repository root. One explicit phase; never submits another job.
# Override partition/GPU/time/account with your own sbatch options if required.
#SBATCH --job-name=cascade-targeted
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
set -euo pipefail
: "${STUDY_KIND:?set STUDY_KIND to ranker or retriever}"
: "${STAGE:?set STAGE to development, pilot or replication}"
: "${RUN_ID:?set a fresh RUN_ID}"
case "$RUN_ID" in .|..|*[!a-zA-Z0-9_-]*) printf 'RUN_ID must contain only letters, digits, _ or -\n' >&2; exit 2 ;; esac
case "$STUDY_KIND:$STAGE" in ranker:development|ranker:pilot|ranker:replication|retriever:pilot|retriever:replication) ;;
  *) printf 'Unsupported study/stage: %s/%s\n' "$STUDY_KIND" "$STAGE" >&2; exit 2 ;;
esac
source scripts/slurm/common.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for this job"'
study_data_config="${DATA_CONFIG:-configs/data.yaml}"
study_candidate_dir="${CANDIDATE_DIR:-data/processed/candidates/two_tower_text_seed0}"
study_text_reference="${TEXT_REFERENCE:-results/retriever/two_tower_text}"
study_check=(--study-kind "$STUDY_KIND" --data-config "$study_data_config" --require-ready
  --output "results/targeted_study/${RUN_ID}-${STAGE}-preflight.json")
if [ "$STUDY_KIND" = ranker ]; then
  study_check+=(--candidate-dir "$study_candidate_dir")
else
  study_check+=(--text-reference "$study_text_reference" --phase "$STAGE")
fi
python -m cascade.pipelines.study_preflight "${study_check[@]}"
case "$STUDY_KIND" in
  ranker)
    study_root="results/ranker_policy/$RUN_ID"
    if [ "$STAGE" = development ]; then study_root="${study_root}_sample"; fi
    python -m cascade.pipelines.ranker_policy_study --run-id "$RUN_ID" \
      --stage "$STAGE" --data-config "$study_data_config" --device cuda --candidate-dir "$study_candidate_dir"
    python -m cascade.eval.ranker_policy_report --studies "$study_root" \
      --output "results/targeted_study/${RUN_ID}-${STAGE}-report"
    ;;
  retriever)
    python -m cascade.pipelines.retriever_ablation --run-id "$RUN_ID" \
      --phase "$STAGE" --data-config "$study_data_config" --device cuda
    python -m cascade.pipelines.retriever_ablation --run-id "$RUN_ID" \
      --phase "$STAGE" --data-config "$study_data_config" --summarize-only --text-reference "$study_text_reference"
    ;;
esac
printf 'Completed %s %s (%s). No Test evaluation was run.\n' "$STUDY_KIND" "$STAGE" "$RUN_ID"
