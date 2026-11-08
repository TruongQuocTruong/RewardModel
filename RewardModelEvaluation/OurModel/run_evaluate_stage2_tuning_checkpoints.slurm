#!/bin/bash
#SBATCH --job-name=eval_stage2_tuning
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_stage2_tuning_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_stage2_tuning_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:2g.35gb:1

# Stage 2 tuning: evaluation on the Stage 2 training data (trained / untrained
# Unlabeled, 256 of the 1008 labeled pairs).
#
# STAGE2_EVAL_CHECKPOINTS       space-separated checkpoint files (may be empty when
#                               STAGE2_EVAL_LABELED_ALL_CHECKPOINTS is set)
# STAGE2_EVAL_LABELED_ALL_CHECKPOINTS  checkpoint files also scored on all 1008 labeled
#                               pairs (output <checkpoint>_labeled_all.json); usually step 0 / last
# STAGE2_EVAL_PLOT              true (default): run plot_stage2_tuning.py for the run at the end
# STAGE2_EVAL_RUN_DIR           run whose unlabeled_subset.json names the trained set
# STAGE2_EVAL_ALL_TRAINED_SETS  true: score every trained set (Student-Stage1 start)
# STAGE2_EVAL_TRAINED_SET_KEY   trained set of the sets file to score instead of the run's own
#                               subset key (its IDs must lie inside the run's subset)
# STAGE2_EVAL_SKIP_UNTRAINED    true: do not score the untrained Unlabeled set (default false)
# STAGE2_EVAL_ROLE              student (default) or teacher
# STAGE2_EVAL_SETS_RUN_DIRS     if set and the sets file is missing: build it first
#                               from these run directories (space-separated)
# STAGE2_EVAL_OUTPUT_DIR        default: evaluation_results/<train_loop|student>/<run>/stage2_tuning_diagnostics

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
CREATION_DIR="$BASE_DIR/TrainingRewardModel/TrainingDataCreation"
EVAL_DIR="$BASE_DIR/RewardModelEvaluation/OurModel"
RESULTS_DIR="$EVAL_DIR/evaluation_results"

CHECKPOINTS="${STAGE2_EVAL_CHECKPOINTS:-}"
LABELED_ALL_CHECKPOINTS="${STAGE2_EVAL_LABELED_ALL_CHECKPOINTS:-}"
PLOT="${STAGE2_EVAL_PLOT:-true}"
if [ -z "$CHECKPOINTS" ] && [ -z "$LABELED_ALL_CHECKPOINTS" ]; then
    echo "ERROR: set STAGE2_EVAL_CHECKPOINTS and/or STAGE2_EVAL_LABELED_ALL_CHECKPOINTS" >&2
    exit 1
fi
RUN_DIR="${STAGE2_EVAL_RUN_DIR:-}"
ALL_TRAINED_SETS="${STAGE2_EVAL_ALL_TRAINED_SETS:-false}"
ROLE="${STAGE2_EVAL_ROLE:-student}"
TRAINED_SET_KEY="${STAGE2_EVAL_TRAINED_SET_KEY:-}"
SKIP_UNTRAINED="${STAGE2_EVAL_SKIP_UNTRAINED:-false}"
SETS_FILE="${STAGE2_EVAL_SETS:-$RESULTS_DIR/train_loop/stage2_tuning_evaluation_sets.json}"
SETS_RUN_DIRS="${STAGE2_EVAL_SETS_RUN_DIRS:-}"
STUDENT_STAGE1="$PROJECT_DIR/output_training/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_head_fp32/student_warmstart.pt"
INSPECTOR_CACHE="$PROJECT_DIR/output_training/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/checkpoint-63/inspector_pseudo_labels_10k.jsonl"
HUMAN_DATASET="$CREATION_DIR/editreward_overall_humanGT_3k.jsonl"
GEMMA_DATASET="$CREATION_DIR/editreward_overall_finegrained_gemmaGT_3k.jsonl"
UNLABELED_DATASET="$CREATION_DIR/editreward_unlabeled_10k.jsonl"
BASE_MODEL_PATH="$BASE_DIR/Models/Qwen3-VL-Reranker-8B"

case "$ROLE" in student|teacher) ;; *) echo "ERROR: STAGE2_EVAL_ROLE must be student or teacher" >&2; exit 1 ;; esac
TRAINED_ARGS=()
case "$ALL_TRAINED_SETS" in
    true) TRAINED_ARGS=(--all-trained-sets) ;;
    false)
        if [ -z "$RUN_DIR" ]; then echo "ERROR: STAGE2_EVAL_RUN_DIR is required" >&2; exit 1; fi
        TRAINED_ARGS=(--run-dir "$RUN_DIR")
        [ -n "$TRAINED_SET_KEY" ] && TRAINED_ARGS+=(--trained-set-key "$TRAINED_SET_KEY") ;;
    *) echo "ERROR: STAGE2_EVAL_ALL_TRAINED_SETS must be true or false" >&2; exit 1 ;;
esac
case "$SKIP_UNTRAINED" in
    true) TRAINED_ARGS+=(--skip-untrained-set) ;;
    false) ;;
    *) echo "ERROR: STAGE2_EVAL_SKIP_UNTRAINED must be true or false" >&2; exit 1 ;;
esac
read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS"
read -r -a LABELED_ALL_LIST <<< "$LABELED_ALL_CHECKPOINTS"
ALL_LISTED=("${CHECKPOINT_LIST[@]}" "${LABELED_ALL_LIST[@]}")
FIRST_CHECKPOINT="${ALL_LISTED[0]}"
if [ -n "$RUN_DIR" ]; then
    RUN_NAME="$(basename "$RUN_DIR")"
else
    RUN_NAME="$(basename "$(dirname "$FIRST_CHECKPOINT")")"
fi
case "$RUN_NAME" in
    train_loop_*|stage2_tuning_*) RESULT_GROUP="train_loop" ;;
    *) RESULT_GROUP="student" ;;
esac
OUTPUT_DIR="${STAGE2_EVAL_OUTPUT_DIR:-$RESULTS_DIR/$RESULT_GROUP/$RUN_NAME/stage2_tuning_diagnostics}"

# Student-Stage1 (1008 pair IDs) is only read when the sets file is built here.
SETS_INPUTS=()
[ ! -f "$SETS_FILE" ] && SETS_INPUTS=("$STUDENT_STAGE1")
for required_file in "${ALL_LISTED[@]}" ${SETS_INPUTS[@]:+"${SETS_INPUTS[@]}"} "$INSPECTOR_CACHE" "$HUMAN_DATASET" "$GEMMA_DATASET" "$UNLABELED_DATASET"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done
if [ ! -f "$SETS_FILE" ] && [ -z "$SETS_RUN_DIRS" ]; then
    echo "ERROR: $SETS_FILE is missing and STAGE2_EVAL_SETS_RUN_DIRS is not set" >&2
    exit 1
fi

module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

ulimit -n 65537
cd "$PROJECT_DIR"

echo "======================================================================"
echo "STAGE 2 TUNING EVALUATION ON STAGE 2 TRAINING DATA"
echo "Role:              $ROLE"
echo "Checkpoints:       ${CHECKPOINTS:-none}"
echo "1008-pair labeled:  ${LABELED_ALL_CHECKPOINTS:-none}"
echo "Run directory:     ${RUN_DIR:-none}"
echo "Trained set key:   ${TRAINED_SET_KEY:-run subset}; skip untrained: $SKIP_UNTRAINED"
echo "All trained sets:  $ALL_TRAINED_SETS"
echo "Evaluation sets:   $SETS_FILE"
echo "Output:            $OUTPUT_DIR"
echo "======================================================================"

if [ ! -f "$SETS_FILE" ]; then
    read -r -a SETS_RUN_LIST <<< "$SETS_RUN_DIRS"
    python "$EVAL_DIR/evaluate_stage2_tuning_checkpoints.py" sets \
        --student-checkpoint "$STUDENT_STAGE1" \
        --run-dirs "${SETS_RUN_LIST[@]}" \
        --unlabeled-dataset "$UNLABELED_DATASET" \
        --inspector-cache "$INSPECTOR_CACHE" \
        --output "$SETS_FILE"
fi

run_evaluation() {
    python "$EVAL_DIR/evaluate_stage2_tuning_checkpoints.py" evaluate \
        --evaluation-sets "$SETS_FILE" \
        --role "$ROLE" \
        "${TRAINED_ARGS[@]}" \
        --human-dataset "$HUMAN_DATASET" \
        --gemma-dataset "$GEMMA_DATASET" \
        --unlabeled-dataset "$UNLABELED_DATASET" \
        --inspector-cache "$INSPECTOR_CACHE" \
        --base-model-path "$BASE_MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --device cuda \
        "$@"
}
if [ ${#CHECKPOINT_LIST[@]} -gt 0 ]; then
    run_evaluation --checkpoints "${CHECKPOINT_LIST[@]}"
fi
if [ ${#LABELED_ALL_LIST[@]} -gt 0 ]; then
    run_evaluation --labeled-all-pairs --checkpoints "${LABELED_ALL_LIST[@]}"
fi

echo "Stage 2 tuning evaluation completed: $OUTPUT_DIR"

# Plots (logs and evaluation files only; a failure here does not fail the evaluation).
if [ "$PLOT" = true ] && [ "$RESULT_GROUP" = train_loop ]; then
    python "$EVAL_DIR/plot_stage2_tuning.py" --versions "$RUN_NAME" \
        || echo "WARNING: plot_stage2_tuning.py failed for $RUN_NAME" >&2
fi
