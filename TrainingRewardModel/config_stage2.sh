#!/bin/bash
# Stage 2 (Student-Teacher iteration training) configs of the three runs trained on 2026-11-07.
# Runs without Slurm: one GPU (the runs used one 35 GB MIG slice, about 20-30 h each).
#
# Usage:
#   bash config_stage2.sh <run> [--dry-run]
#   <run> = ema_step_0.99 | ema_step_0.99_teacher_kl_relevance_0.5 | ema_iteration_0.99 | ema_iteration_0.99_r0
#   --dry-run prints the python command (one argument per line) without training.
#
# The three runs share every argument below and differ only in:
#   run                                      --teacher-update   --teacher-kl-relevance-weight
#   ema_step_0.99                            optimizer_step     (not set = 0)
#   ema_step_0.99_teacher_kl_relevance_0.5   optimizer_step     0.5
#   ema_iteration_0.99                       after_round        (not set = 0)
#   ema_iteration_0.99_r0                    after_round        (not set = 0)
#
# ema_iteration_0.99_r0 is ema_iteration_0.99 plus three changes:
#   --restart-lr-each-round       cosine (with 5% warmup) restarts over each round's steps; the original
#                                 single cosine spans all five rounds, so rounds 4-5 get ~12% of the LR.
#   --replay-round-selections     trains the per-round Unlabeled IDs of the finished ema_iteration_0.99 run
#                                 (1008, 993, 992, 990, 990 -> 126+125+124+124+124 = 623 steps) instead of
#                                 re-running the agreement filter. Build the file once:
#                                   python scripts/build_round_schedule.py \
#                                     <output_training>/stage2_tuning_..._iterations_ema_iteration_0.99 \
#                                     "$ROUND_SCHEDULE"
#   Teacher export (code, always on with --teacher-update after_round): the round-end checkpoint and
#                                 best.pt are re-saved after the round's EMA update, so the exported
#                                 Teacher contains every update (the original files lag by one update).
# Output folder name = stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_<run>.
#
# Source: rendered from script_slurm/run_train_loop.slurm (md5 aeede9fa0ed7a73bd2f79829d3cd2006) and the
# config recorded in experiment_submissions.log (jobs 11199, 11200, 11216); train_loop.py md5
# ebb300922072aa7d8b89a2520adcd2fa. Only paths are replaced by the variables below.
#
# Inputs (checkpoints from the Drive zip files, unzipped into CHECKPOINT_DIR):
#   student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63.zip
#       student_warmstart.pt (Student-Stage1), teacher_initialized.pt (initial Teacher)
#   inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32.zip
#       checkpoint-63 (Inspector), checkpoint-63/inspector_pseudo_labels_10k.jsonl (Inspector labels of Unlabeled)
#   teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63.zip
#       selected_unlabeled_1008.json (Unlabeled subset), augmentation_combined_scenarios_1008.json
#   Data (in this repo, TrainingDataCreation/): editreward_overall_humanGT_3k.jsonl,
#       editreward_overall_finegrained_gemmaGT_3k.jsonl, editreward_unlabeled_10k.jsonl
#   Base model: Qwen3-VL-Reranker-8B (local folder).
#   Images: the paths stored inside the JSONL files must exist on your machine.
set -euo pipefail

# ---------------------------------------------------------------- paths (edit for your machine)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/Qwen3-VL-Embedding}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/TrainingDataCreation}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/output_training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output_training}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$SCRIPT_DIR/../Models/Qwen3-VL-Reranker-8B}"
PYTHON="${PYTHON:-python}"
ROUND_SCHEDULE="${ROUND_SCHEDULE:-$CHECKPOINT_DIR/round_schedule_ema_iteration_0.99.json}"

# ---------------------------------------------------------------- run selection
RUN="${1:-}"
DRY_RUN=false
[ "${2:-}" = "--dry-run" ] && DRY_RUN=true
case "$RUN" in
    ema_step_0.99)                          RUN_ARGS=(--teacher-update optimizer_step) ;;
    ema_step_0.99_teacher_kl_relevance_0.5) RUN_ARGS=(--teacher-update optimizer_step --teacher-kl-relevance-weight 0.5) ;;
    ema_iteration_0.99)                     RUN_ARGS=(--teacher-update after_round) ;;
    ema_iteration_0.99_r0)                  RUN_ARGS=(--teacher-update after_round --restart-lr-each-round
                                                      --replay-round-selections "$ROUND_SCHEDULE") ;;
    *) echo "usage: bash $0 <ema_step_0.99|ema_step_0.99_teacher_kl_relevance_0.5|ema_iteration_0.99|ema_iteration_0.99_r0> [--dry-run]" >&2; exit 1 ;;
esac
OUTPUT_DIR="$OUTPUT_ROOT/stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_$RUN"

# ---------------------------------------------------------------- arguments shared by the three runs
COMMON_ARGS=(
    --student-checkpoint "$CHECKPOINT_DIR/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/student_warmstart.pt"
    --base-model-path "$BASE_MODEL_PATH"
    --teacher-checkpoint "$CHECKPOINT_DIR/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/teacher_initialized.pt"
    --inspector-checkpoint "$CHECKPOINT_DIR/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/checkpoint-63"
    --inspector-cache "$CHECKPOINT_DIR/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/checkpoint-63/inspector_pseudo_labels_10k.jsonl"
    --human-dataset "$DATA_DIR/editreward_overall_humanGT_3k.jsonl"
    --gemma-dataset "$DATA_DIR/editreward_overall_finegrained_gemmaGT_3k.jsonl"
    --unlabeled-dataset "$DATA_DIR/editreward_unlabeled_10k.jsonl"
    --max-rounds 5
    --unlabeled-per-round 1008
    --diagnostic-probe-size 0
    --unlabeled-subset-size 1008
    --unlabeled-passes 1
    --gemma-monitor-pairs 32
    --labeled-source merged
    --labeled-pairs-per-step 8
    --loss-weighting per_pair
    --lambda-human 1.0
    --hard-replay-fraction 0.25
    --replay-strategy hard
    --unlabeled-sampling previous_iteration_teacher_inspector_agreement
    --loss-plateau-tolerance 0.01
    --gemma-targets majority_vote
    --overall-aggregation selected_equal
    --overall-probability linear
    --overall-loss three_way
    --exclude-probe-from-training false
    --human-split student_train_order
    --checkpoint-selection last
    --consistency-metrics selected
    --low-agreement-fraction 0.25
    --max-box-only-fraction 0.25
    --box-only-set-weight 0.25
    --min-round-fraction 0.50
    --min-supervised-fraction 0.10
    --high-inspector-confidence 0.80
    --screening-multiplier 4
    --max-replays-per-sample 2
    --patience-rounds 2
    --min-validation-improvement 0.002
    --eval-save-steps 0
    --eval-save-total-limit 10
    --human-batch-size 1
    --gemma-batch-size 1
    --unlabeled-batch-size 1
    --teacher-batch-size 1
    --eval-batch-size 1
    --gradient-accumulation-steps 8
    --learning-rate 1e-5
    --lr-scheduler-type cosine
    --warmup-ratio 0.05
    --unlabeled-objective inspector_soft_ema_consistency
    --lambda-consistency 1.0
    --selection-model student
    --lambda-set 1.0
    --set-target-contribution 0.10
    --set-target-projection-rate 0.35
    --set-scale-min 0.5
    --set-scale-max 4.0
    --set-scale-momentum 0.5
    --lambda-gemma 1.0
    --lambda-mask 1.0
    --set-ramp-steps 1
    --semantic-drift-threshold 0.02
    --ema-decay 0.99
    --ema-schedule constant
    --ema-decay-start 0.99
    --ema-decay-end 0.9999
    --mc-passes 1
    --teacher-dropout on
    --freeze-visual false
    --kappa 0.05
    --cache-slack 0.02
    --projection-iterations 40
    --logging-steps 1
    --seed 42
    --selection-seed 42
    --step-diagnostics
    --per-sample-log true
    --student-augmentation single_operation
    --unlabeled-subset-file "$CHECKPOINT_DIR/teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63/selected_unlabeled_1008.json"
    --augmentation-scenarios "$CHECKPOINT_DIR/teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63/augmentation_combined_scenarios_1008.json"
    --augmentation-operation composite+mask_content
    --inspector-target soft
    --augment-labeled
    --augmentation-examples 25
    --checkpoints-per-iteration 2
    --save-round-checkpoints first
)

CMD=("$PYTHON" scripts/train_loop.py "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}"
     --output "$OUTPUT_DIR/best.pt"
     --per-sample-log-path "$OUTPUT_DIR/per_sample_log.jsonl"
     --device cuda)

if $DRY_RUN; then
    printf '%s\n' "${CMD[@]}"
    exit 0
fi

if [ -e "$OUTPUT_DIR/best.pt" ]; then
    echo "ERROR: refusing to overwrite existing final checkpoint: $OUTPUT_DIR/best.pt" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Same environment as run_train_loop.slurm (HF offline: the base model is a local folder).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
ulimit -n 65537 2>/dev/null || true

cd "$PROJECT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train_loop_$(date +%Y%m%d_%H%M%S).log"
