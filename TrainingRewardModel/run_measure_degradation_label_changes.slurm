#!/bin/bash
#SBATCH --job-name=degradation_label_change
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/degradation_label_change_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/degradation_label_change_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.35gb:1

# How often the Teacher label changes per metric when the source, A and B images are
# degraded the same way (JPEG / blur / downscale / noise, 4 levels each), on 200 samples
# of the Teacher-Inspector agreement subset; chooses the degradation levels of the
# combined Student augmentation (2026-11-05). Teacher in eval mode. Resumes if stopped.

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
CREATION_DIR="$BASE_DIR/TrainingRewardModel/TrainingDataCreation"
STUDENT_DIR="$PROJECT_DIR/output_training/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63"
TEACHER_CHECKPOINT="$STUDENT_DIR/teacher_initialized.pt"
STUDENT_CHECKPOINT="$STUDENT_DIR/student_warmstart.pt"
SCENARIOS="$PROJECT_DIR/output_training/teacher_inspector_agreement_student_initialized_from_inspector_checkpoint63/augmentation_combined_scenarios_1008.json"
UNLABELED_DATASET="$CREATION_DIR/editreward_unlabeled_10k.jsonl"
BASE_MODEL_PATH="$BASE_DIR/Models/Qwen3-VL-Reranker-8B"
SAMPLE_COUNT="${DEGRADATION_SAMPLE_COUNT:-200}"
OUTPUT_DIR="$PROJECT_DIR/output_training/degradation_label_change_teacher_student_initialized_from_inspector_checkpoint63"

for required_file in "$TEACHER_CHECKPOINT" "$STUDENT_CHECKPOINT" "$SCENARIOS" "$UNLABELED_DATASET"; do
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
echo "DEGRADATION LABEL CHANGES (TEACHER, PER METRIC)"
echo "Teacher:        $TEACHER_CHECKPOINT"
echo "Scenarios:      $SCENARIOS"
echo "Sample count:   $SAMPLE_COUNT"
echo "Output:         $OUTPUT_DIR"
echo "script md5:     $(md5sum "$PROJECT_DIR/scripts/measure_degradation_label_changes.py" | cut -d' ' -f1)"
echo "======================================================================"

python "$PROJECT_DIR/scripts/measure_degradation_label_changes.py" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --student-checkpoint "$STUDENT_CHECKPOINT" \
    --scenarios "$SCENARIOS" \
    --unlabeled-dataset "$UNLABELED_DATASET" \
    --base-model-path "$BASE_MODEL_PATH" \
    --sample-count "$SAMPLE_COUNT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "Degradation label changes completed: $OUTPUT_DIR"
