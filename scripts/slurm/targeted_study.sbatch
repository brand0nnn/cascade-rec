#!/usr/bin/env bash
# User-submitted development/pilot/replication jobs. No SSH or automatic next stage.
# Set STUDY_KIND=ranker|retriever, STAGE=development|pilot|replication, RUN_ID=fresh-id.
#SBATCH --job-name=cascade-targeted
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
source scripts/slurm/common.sh
: "${STUDY_KIND:?set STUDY_KIND to ranker or retriever}"
: "${STAGE:?set STAGE to development, pilot or replication}"
: "${RUN_ID:?set a fresh RUN_ID}"
case "$STAGE" in development|pilot|replication) ;; *) exit 2 ;; esac
study_data_config="${DATA_CONFIG:-configs/data.yaml}"
case "$STUDY_KIND" in
  ranker)
    study_extra=()
    if [ -n "${CANDIDATE_DIR:-}" ]; then study_extra+=(--candidate-dir "$CANDIDATE_DIR"); fi
    python -m cascade.pipelines.ranker_policy_study --run-id "$RUN_ID" \
      --stage "$STAGE" --data-config "$study_data_config" --device cuda "${study_extra[@]}"
    ;;
  retriever)
    study_phase="$STAGE"
    study_extra=()
    if [ "$STAGE" = development ]; then study_phase=pilot; study_extra+=(--sample); fi
    python -m cascade.pipelines.retriever_ablation --run-id "$RUN_ID" \
      --phase "$study_phase" --data-config "$study_data_config" --device cuda "${study_extra[@]}"
    ;;
  *) printf 'Unknown STUDY_KIND: %s\n' "$STUDY_KIND" >&2; exit 2 ;;
esac
