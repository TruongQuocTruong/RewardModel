#!/bin/bash
#SBATCH --job-name=teacher_inspector_agreement
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/teacher_inspector_agreement_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/teacher_inspector_agreement_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.35gb:1

# Step 1 of the Stage 2 sample selection: score Unlabeled samples by the agreement
# CE(Inspector, Teacher), most certain Inspector first, Teacher in eval mode (dropout
# off); stop when 1008 samples have the same Teacher and Inspector label.
# Teacher = teacher_initialized.pt of Student-Stage1 batch 16 initialized from
# Inspector checkpoint-63 (default Stage 2 setting). Resumes if stopped.

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
CREATION_DIR="$BASE_DIR/TrainingRewardModel/TrainingDataCreation"
# Defaults = the 1008-pair setting; the 512-pair setting overrides them (AGREEMENT_*).
STUDENT_DIR="${AGREEMENT_STUDENT_DIR:-$PROJECT_DIR/output_training/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63}"
TEACHER_CHECKPOINT="$STUDENT_DIR/teacher_initialized.pt"
STUDENT_CHECKPOINT="$STUDENT_DIR/student_warmstart.pt"
INSPECTOR_CACHE="${AGREEMENT_INSPECTOR_CACHE:-$PROJECT_DIR/output_training/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/checkpoint-63/inspector_pseudo_labels_10k.jsonl}"
UNLABELED_DATASET="$CREATION_DIR/editreward_unlabeled_10k.jsonl"
BASE_MODEL_PATH="$BASE_DIR/Models/Qwen3-VL-Reranker-8B"
TARGET_COUNT="${AGREEMENT_TARGET_COUNT:-1008}"
# true = every accepted instruction differs (duplicates are skipped without scoring).
UNIQUE_ARGS=()
[ "${AGREEMENT_UNIQUE_INSTRUCTIONS:-false}" = true ] && UNIQUE_ARGS=(--unique-instructions)
# "LOW HIGH" = only samples whose largest Inspector probability is in [LOW, HIGH).
[ -n "${AGREEMENT_INSPECTOR_MAX_PROBABILITY_RANGE:-}" ] && UNIQUE_ARGS+=(--inspector-max-probability-range ${AGREEMENT_INSPECTOR_MAX_PROBABILITY_RANGE})
# "random" = seeded random visiting order inside the range (default: most certain Inspector first).
UNIQUE_ARGS+=(--scan-order "${AGREEMENT_SCAN_ORDER:-inspector_entropy}" --scan-seed "${AGREEMENT_SCAN_SEED:-42}")
OUTPUT_DIR="${AGREEMENT_OUTPUT_DIR:-$PROJECT_DIR/output_training/teacher_inspector_agreement_student_initialized_from_inspector_checkpoint63}"

for required_file in "$TEACHER_CHECKPOINT" "$STUDENT_CHECKPOINT" "$INSPECTOR_CACHE" "$UNLABELED_DATASET"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done

module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

ulimit -n 65537
cd "$PROJECT_DIR"

echo "======================================================================"
echo "TEACHER-INSPECTOR AGREEMENT SCORING"
echo "Teacher:        $TEACHER_CHECKPOINT"
echo "Inspector cache: $INSPECTOR_CACHE"
echo "Target count:   $TARGET_COUNT"
echo "Unique instructions: ${AGREEMENT_UNIQUE_INSTRUCTIONS:-false}"
echo "Inspector largest probability range: ${AGREEMENT_INSPECTOR_MAX_PROBABILITY_RANGE:-all}"
echo "Scan order:     ${AGREEMENT_SCAN_ORDER:-inspector_entropy} (seed ${AGREEMENT_SCAN_SEED:-42})"
echo "Output:         $OUTPUT_DIR"
echo "script md5:     $(md5sum "$PROJECT_DIR/scripts/score_teacher_inspector_agreement.py" | cut -d' ' -f1)"
echo "======================================================================"

python "$PROJECT_DIR/scripts/score_teacher_inspector_agreement.py" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --student-checkpoint "$STUDENT_CHECKPOINT" \
    --inspector-cache "$INSPECTOR_CACHE" \
    --unlabeled-dataset "$UNLABELED_DATASET" \
    --base-model-path "$BASE_MODEL_PATH" \
    --target-count "$TARGET_COUNT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    ${UNIQUE_ARGS[@]:+"${UNIQUE_ARGS[@]}"}

echo "Teacher-Inspector agreement scoring completed: $OUTPUT_DIR"
