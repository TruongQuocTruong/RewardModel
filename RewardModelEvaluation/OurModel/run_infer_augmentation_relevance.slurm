#!/bin/bash
#SBATCH --job-name=infer_augmentation_relevance
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/infer_augmentation_relevance_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/infer_augmentation_relevance_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2g.35gb:1

# Inference test of hypotheses (b), (d1), (d2), (f) of the Stage 2 iteration runs
# (infer_augmentation_relevance.py). No training.
#
# INFER_RUN_DIR       run folder (output_training/stage2_tuning_..._iterations_...)
# INFER_CHECKPOINTS   space-separated checkpoint-N.pt names of the run and/or "start"
#                     (Stage 2 start: Student-Stage1 + Teacher initialized)
# INFER_HOLD          true (default): hold the GPU and wait for a go file (submitted early to
#                     keep the slot, 2026-11-07):
#                       <hold dir>/<tag>.test  (content = any token): 4-sample test on the first
#                                               checkpoint, output ..._test_4_samples; a new token
#                                               runs the test again
#                       <hold dir>/<tag>.full  : full run once every checkpoint file exists
#                     false: full run at once
# INFER_TAG           name of the go files (default: run folder name)
# INFER_HOLD_MAX_SECONDS  give up after this long without the full run (default 18000 = 5 h)

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
CREATION_DIR="$BASE_DIR/TrainingRewardModel/TrainingDataCreation"
EVAL_DIR="$BASE_DIR/RewardModelEvaluation/OurModel"
RESULTS_DIR="$EVAL_DIR/evaluation_results"
HOLD_DIR="$BASE_DIR/RewardModelEvaluation/logs/infer_augmentation_relevance_hold"

RUN_DIR="${INFER_RUN_DIR:?set INFER_RUN_DIR}"
CHECKPOINTS="${INFER_CHECKPOINTS:?set INFER_CHECKPOINTS}"
HOLD="${INFER_HOLD:-true}"
RUN_NAME="$(basename "$RUN_DIR")"
TAG="${INFER_TAG:-$RUN_NAME}"
MAX_WAIT="${INFER_HOLD_MAX_SECONDS:-18000}"
OUTPUT_DIR="$RESULTS_DIR/train_loop/$RUN_NAME/augmentation_relevance"

START_DIR="$PROJECT_DIR/output_training/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63"
SCENARIOS="$PROJECT_DIR/output_training/teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63/augmentation_combined_scenarios_1008.json"
COMMON_ARGS=(
    --run-dir "$RUN_DIR"
    --start-student "$START_DIR/student_warmstart.pt"
    --start-teacher "$START_DIR/teacher_initialized.pt"
    --augmentation-operation composite+mask_content
    --augmentation-seed 42
    --augmentation-scenarios "$SCENARIOS"
    --unlabeled-dataset "$CREATION_DIR/editreward_unlabeled_10k.jsonl"
    --human-dataset "$CREATION_DIR/editreward_overall_humanGT_3k.jsonl"
    --gemma-dataset "$CREATION_DIR/editreward_overall_finegrained_gemmaGT_3k.jsonl"
    --base-model-path "$BASE_DIR/Models/Qwen3-VL-Reranker-8B"
)
read -r -a CHECKPOINT_NAMES <<< "$CHECKPOINTS"
CHECKPOINT_ARGS=()
for name in "${CHECKPOINT_NAMES[@]}"; do
    if [ "$name" = "start" ]; then CHECKPOINT_ARGS+=(start); else CHECKPOINT_ARGS+=("$RUN_DIR/$name"); fi
done

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
echo "INFERENCE TEST OF HYPOTHESES (b) (d1) (d2) (f) — $(date '+%F %T') — user $(whoami)"
echo "Run:          $RUN_DIR"
echo "Checkpoints:  $CHECKPOINTS"
echo "Hold:         $HOLD (go files $HOLD_DIR/$TAG.test / .full, max wait $MAX_WAIT s)"
echo "Output:       $OUTPUT_DIR"
echo "Script md5:   $(md5sum "$EVAL_DIR/infer_augmentation_relevance.py" | cut -d' ' -f1) (at job start)"
echo "======================================================================"

checkpoints_ready() {
    for name in "${CHECKPOINT_NAMES[@]}"; do
        [ "$name" = "start" ] && continue
        [ -f "$RUN_DIR/$name" ] || return 1
    done
    return 0
}

run_test() {
    local first="${CHECKPOINT_ARGS[0]}"
    local test_dir="${OUTPUT_DIR}_test_4_samples"
    rm -rf "$test_dir"
    echo "---- test on $first, script md5 $(md5sum "$EVAL_DIR/infer_augmentation_relevance.py" | cut -d' ' -f1)"
    if python -B "$EVAL_DIR/infer_augmentation_relevance.py" "${COMMON_ARGS[@]}" \
        --checkpoints "$first" --output-dir "$test_dir" \
        --unlabeled-samples 4 --labeled-samples 4 --draws 0 1 --dropout-passes 1; then
        echo "---- test finished OK $(date '+%F %T')"
    else
        echo "---- test FAILED $(date '+%F %T')"
    fi
}

run_full() {
    echo "---- full run, script md5 $(md5sum "$EVAL_DIR/infer_augmentation_relevance.py" | cut -d' ' -f1), $(date '+%F %T')"
    python -B "$EVAL_DIR/infer_augmentation_relevance.py" "${COMMON_ARGS[@]}" \
        --checkpoints "${CHECKPOINT_ARGS[@]}" --output-dir "$OUTPUT_DIR"
    echo "---- full run finished $(date '+%F %T')"
}

if [ "$HOLD" != "true" ]; then
    run_full
    exit 0
fi

mkdir -p "$HOLD_DIR" 2>/dev/null || true
WAITED=0
LAST_TOKEN=""
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
    if [ -f "$HOLD_DIR/$TAG.test" ]; then
        TOKEN="$(cat "$HOLD_DIR/$TAG.test")"
        if [ "$TOKEN" != "$LAST_TOKEN" ]; then
            LAST_TOKEN="$TOKEN"
            run_test
        fi
    fi
    if [ -f "$HOLD_DIR/$TAG.full" ]; then
        # A non-empty .full file replaces INFER_CHECKPOINTS (e.g. the real last step of iteration 3).
        if [ -s "$HOLD_DIR/$TAG.full" ]; then
            read -r -a CHECKPOINT_NAMES < "$HOLD_DIR/$TAG.full"
            CHECKPOINT_ARGS=()
            for name in "${CHECKPOINT_NAMES[@]}"; do
                if [ "$name" = "start" ]; then CHECKPOINT_ARGS+=(start); else CHECKPOINT_ARGS+=("$RUN_DIR/$name"); fi
            done
        fi
    fi
    if [ -f "$HOLD_DIR/$TAG.full" ] && checkpoints_ready; then
        echo "---- checkpoints: ${CHECKPOINT_NAMES[*]}"
        run_full
        exit 0
    fi
    sleep 60
    WAITED=$((WAITED + 60))
done
echo "ERROR: no full run after $MAX_WAIT s of holding" >&2
exit 1
