#!/bin/bash
#SBATCH --job-name=eval_inspector_checkpoints
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_inspector_checkpoints_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_inspector_checkpoints_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
EVAL_DATA_DIR="$BASE_DIR/RewardModelEvaluation/EvalDatasets"
EVAL_DIR="$BASE_DIR/RewardModelEvaluation/OurModel"

RUN_DIR="${INSPECTOR_EVAL_RUN_DIR:?INSPECTOR_EVAL_RUN_DIR must point to an Inspector training output directory}"
# Space-separated checkpoint directory names; empty means every checkpoint-*.
CHECKPOINTS="${INSPECTOR_EVAL_CHECKPOINTS:-}"
INCLUDE_BASE="${INSPECTOR_EVAL_INCLUDE_BASE:-1}"
# Empty keeps the processor defaults, as in MS-Swift training without --max_pixels.
MAX_PIXELS_VALUE="${INSPECTOR_EVAL_MAX_PIXELS:-}"
MIN_PIXELS_VALUE="${INSPECTOR_EVAL_MIN_PIXELS:-}"
# Results of different image resolutions never share a directory: a non-default max_pixels adds a suffix.
OUTPUT_DIR="${INSPECTOR_EVAL_OUTPUT_DIR:-$BASE_DIR/RewardModelEvaluation/OurModel/evaluation_results/inspector/$(basename "$RUN_DIR")${MAX_PIXELS_VALUE:+_maxpixels$MAX_PIXELS_VALUE}}"
BATCH_SIZE="${INSPECTOR_EVAL_BATCH_SIZE:-1}"
BASE_MODEL_PATH="${INSPECTOR_EVAL_BASE_MODEL_PATH:-$BASE_DIR/Models/Qwen3-VL-Reranker-8B}"
EDITREWARD_BENCH="$EVAL_DATA_DIR/EditReward-Bench_pairwise.json"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Inspector run directory not found: $RUN_DIR" >&2
    exit 1
fi
if [ ! -f "$EDITREWARD_BENCH" ]; then
    echo "ERROR: Required input not found: $EDITREWARD_BENCH" >&2
    exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL_PATH" >&2
    exit 1
fi

EXTRA_ARGS=()
if [ -n "$CHECKPOINTS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--checkpoints $CHECKPOINTS)
fi
if [ "$INCLUDE_BASE" = "1" ]; then
    EXTRA_ARGS+=(--include-base)
fi
if [ -n "$MIN_PIXELS_VALUE" ]; then
    EXTRA_ARGS+=(--min-pixels "$MIN_PIXELS_VALUE")
fi
if [ -n "$MAX_PIXELS_VALUE" ]; then
    EXTRA_ARGS+=(--max-pixels "$MAX_PIXELS_VALUE")
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
echo "OVERALL INSPECTOR MODEL CHECKPOINT EVALUATION"
echo "Run dir:          $RUN_DIR"
echo "Checkpoints:      ${CHECKPOINTS:-<all checkpoint-*>}"
echo "Zero-shot base:   $INCLUDE_BASE"
# EditReward-Data-Test is drawn from the Inspector training data, so it is not evaluated.
echo "Evaluation set:   EditReward-Bench only"
echo "Pixels:           min=${MIN_PIXELS_VALUE:-default} max=${MAX_PIXELS_VALUE:-default}"
echo "Output:           $OUTPUT_DIR"
echo "======================================================================"

python "$EVAL_DIR/evaluate_inspector_checkpoints.py" \
    "${EXTRA_ARGS[@]}" \
    --run-dir "$RUN_DIR" \
    --base-model-path "$BASE_MODEL_PATH" \
    --editreward-bench "$EDITREWARD_BENCH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --device cuda

echo "Inspector checkpoint evaluation completed: $OUTPUT_DIR/summary.json"
