#!/bin/bash
#SBATCH --job-name=eval_student_checkpoints
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_student_checkpoints_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_student_checkpoints_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
CREATION_DIR="$BASE_DIR/TrainingRewardModel/TrainingDataCreation"
EVAL_DATA_DIR="$BASE_DIR/RewardModelEvaluation/EvalDatasets"
EVAL_DIR="$BASE_DIR/RewardModelEvaluation/OurModel"

RUN_DIR="${STUDENT_EVAL_RUN_DIR:?STUDENT_EVAL_RUN_DIR must point to a Student training output directory}"
CHECKPOINT="${STUDENT_EVAL_CHECKPOINT:-$RUN_DIR/student_warmstart.pt}"
# Optional: an MS-Swift checkpoint-N directory of a Stage1 run; its adapter
# replaces the adapter of $CHECKPOINT and results go to <run>/checkpoint-N.
ADAPTER_DIR="${STUDENT_EVAL_ADAPTER_DIR:-}"
# Optional: without it only selected_equal (no fitted weights) is reported.
PREFERENCE_AGGREGATOR="${STUDENT_EVAL_PREFERENCE_AGGREGATOR:-}"
if [ -z "$PREFERENCE_AGGREGATOR" ] && [ -f "$RUN_DIR/preference_aggregator.json" ]; then
    PREFERENCE_AGGREGATOR="$RUN_DIR/preference_aggregator.json"
fi
RUN_NAME="$(basename "$RUN_DIR")"
case "$RUN_NAME" in
    train_loop_*) RESULT_GROUP="train_loop" ;;
    *) RESULT_GROUP="student" ;;
esac
OUTPUT_DIR="${STUDENT_EVAL_OUTPUT_DIR:-$BASE_DIR/RewardModelEvaluation/OurModel/evaluation_results/$RESULT_GROUP/$RUN_NAME${ADAPTER_DIR:+/$(basename "$ADAPTER_DIR")}}"
BATCH_SIZE="${STUDENT_EVAL_BATCH_SIZE:-1}"
ROLE="${STUDENT_EVAL_ROLE:-student}"
# all: EditReward-Data test set + EditReward-Bench; bench: EditReward-Bench only;
# test: EditReward-Data test set only (EditReward-Test 1k, Stage 2 ablation).
DATASETS="${STUDENT_EVAL_DATASETS:-all}"
case "$DATASETS" in
    all) FILE_NAME="evaluation.json" ;;
    bench) FILE_NAME="bench_evaluation.json" ;;
    test) FILE_NAME="test_evaluation.json" ;;
    *) echo "ERROR: STUDENT_EVAL_DATASETS must be all, bench or test" >&2; exit 1 ;;
esac
case "$ROLE" in
    student) OUTPUT_FILE="$FILE_NAME" ;;
    teacher) OUTPUT_FILE="teacher_$FILE_NAME" ;;
    *) echo "ERROR: STUDENT_EVAL_ROLE must be student or teacher" >&2; exit 1 ;;
esac
BASE_MODEL_PATH="${STUDENT_EVAL_BASE_MODEL_PATH:-$BASE_DIR/Models/Qwen3-VL-Reranker-8B}"
HUMAN_GT_DATA="$CREATION_DIR/editreward_overall_humanGT_3k.jsonl"
TEST_DATA="${STUDENT_EVAL_TEST_DATA:-$EVAL_DATA_DIR/EditReward/EditReward-Test_pairwise_1k.jsonl}"
EDITREWARD_BENCH="${STUDENT_EVAL_BENCH:-$EVAL_DATA_DIR/EditReward/EditReward-Bench_pairwise.json}"

for required_file in "$CHECKPOINT" ${ADAPTER_DIR:+"$ADAPTER_DIR/adapter_model.safetensors"} ${PREFERENCE_AGGREGATOR:+"$PREFERENCE_AGGREGATOR"} "$HUMAN_GT_DATA" "$TEST_DATA" "$EDITREWARD_BENCH"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL_PATH" >&2
    exit 1
fi
if [ -e "$OUTPUT_DIR/$OUTPUT_FILE" ]; then
    echo "ERROR: Refusing to overwrite existing evaluation: $OUTPUT_DIR/$OUTPUT_FILE" >&2
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
echo "STUDENT MODEL CHECKPOINT EVALUATION"
echo "Role:                  $ROLE"
echo "Checkpoint:            $CHECKPOINT"
echo "Adapter directory:     ${ADAPTER_DIR:-none (adapter of the checkpoint)}"
echo "Metrics Weights Aggregator: ${PREFERENCE_AGGREGATOR:-none (selected_equal only)}"
echo "Datasets:              $DATASETS"
echo "Test set:              $TEST_DATA"
echo "EditReward-Bench:      $EDITREWARD_BENCH"
echo "Output:                $OUTPUT_DIR"
echo "======================================================================"

python "$EVAL_DIR/evaluate_student_checkpoints.py" \
    --role "$ROLE" \
    --datasets "$DATASETS" \
    --checkpoint "$CHECKPOINT" \
    ${ADAPTER_DIR:+--adapter-dir "$ADAPTER_DIR"} \
    ${PREFERENCE_AGGREGATOR:+--preference-aggregator "$PREFERENCE_AGGREGATOR"} \
    --base-model-path "$BASE_MODEL_PATH" \
    --human-dataset "$HUMAN_GT_DATA" \
    --test-dataset "$TEST_DATA" \
    --editreward-root /datastore/shared/datasets/EditReward-Data \
    --editreward-bench "$EDITREWARD_BENCH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --device cuda

echo "Checkpoint evaluation completed: $OUTPUT_DIR/$OUTPUT_FILE"
