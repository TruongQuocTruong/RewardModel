#!/bin/bash
#SBATCH --job-name=train_loop
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/train_loop_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/train_loop_%j.err
#SBATCH --account=uit
#SBATCH --partition=full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:7g.141gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
STUDENT_CHECKPOINT="${TRAIN_LOOP_STUDENT_CHECKPOINT:?Set TRAIN_LOOP_STUDENT_CHECKPOINT to the Student-Stage1 checkpoint}"
BASE_MODEL_PATH="${REFINEMENT_BASE_MODEL_PATH:-$BASE_DIR/Models/Qwen3-VL-Reranker-8B}"
TEACHER_CHECKPOINT="${TRAIN_LOOP_TEACHER_CHECKPOINT:?Set TRAIN_LOOP_TEACHER_CHECKPOINT to the initialized Teacher checkpoint}"
INSPECTOR_DIR="$PROJECT_DIR/output_training/overall_inspector_r32_llm_lr8e-5"
INSPECTOR_CHECKPOINT="${TRAIN_LOOP_INSPECTOR_CHECKPOINT:-$INSPECTOR_DIR/overall_inspector_checkpoint.pt}"
INSPECTOR_CACHE="${TRAIN_LOOP_INSPECTOR_CACHE:-$INSPECTOR_DIR/editreward_inspector_pseudo_labels_10k.jsonl}"
# Needed only when the fitted a_k or beta are used (see AGGREGATOR_ARGS below).
METRICS_WEIGHTS_AGGREGATOR="${TRAIN_LOOP_METRICS_WEIGHTS_AGGREGATOR:-}"
HUMAN_DATASET="$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_overall_humanGT_3k.jsonl"
GEMMA_DATASET="${TRAIN_LOOP_GEMMA_DATASET:-$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_overall_finegrained_gemmaGT_3k.jsonl}"
UNLABELED_DATASET="$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_unlabeled_10k.jsonl"
OUTPUT_DIR="${REFINEMENT_OUTPUT_DIR:?Set REFINEMENT_OUTPUT_DIR to the confirmed new output directory}"
OUTPUT="${REFINEMENT_OUTPUT:-$OUTPUT_DIR/best.pt}"
TRAINING_SCRIPT="scripts/train_loop.py"
export TRAIN_LOOP_MODE="${TRAIN_LOOP_MODE:-online_swift}"
case "$TRAIN_LOOP_MODE" in
    online_swift|supervised_control) ;;
    *) echo "ERROR: Unsupported TRAIN_LOOP_MODE=$TRAIN_LOOP_MODE" >&2; exit 1 ;;
esac

MAX_ROUNDS="${REFINEMENT_MAX_ROUNDS:-5}"
UNLABELED_PER_ROUND="${REFINEMENT_UNLABELED_PER_ROUND:-1500}"
DIAGNOSTIC_PROBE_SIZE="${REFINEMENT_DIAGNOSTIC_PROBE_SIZE:-128}"
# Stage 2 tuning: fixed Unlabeled subset (0 = all), passes over it (all_once only),
# and Gemma monitoring pairs taken from the training pairs (0 = all).
UNLABELED_SUBSET_SIZE="${REFINEMENT_UNLABELED_SUBSET_SIZE:-0}"
UNLABELED_PASSES="${REFINEMENT_UNLABELED_PASSES:-1}"
GEMMA_MONITOR_PAIRS="${REFINEMENT_GEMMA_MONITOR_PAIRS:-0}"
# Labeled source: separate (HumanGT and GemmaGT forwarded apart) or merged (one forward per
# labeled pair); per_pair loss weighting needs merged. MC passes: 1 = one deterministic Teacher pass.
LABELED_SOURCE="${REFINEMENT_LABELED_SOURCE:-separate}"
LABELED_PAIRS_PER_STEP="${REFINEMENT_LABELED_PAIRS_PER_STEP:-0}"
LOSS_WEIGHTING="${REFINEMENT_LOSS_WEIGHTING:-per_source}"
LAMBDA_HUMAN="${REFINEMENT_LAMBDA_HUMAN:-1.0}"
MC_PASSES="${REFINEMENT_MC_PASSES:-4}"
# Teacher dropout for one pass (auto = off for J=1) and Teacher update
# (optimizer_step = per-step EMA; after_round = fixed inside a loop, EMA once per loop).
TEACHER_DROPOUT="${REFINEMENT_TEACHER_DROPOUT:-auto}"
TEACHER_UPDATE="${REFINEMENT_TEACHER_UPDATE:-optimizer_step}"
# true = train only the LLM LoRA and metric_heads; ViT and aligner LoRA stay at Student-Stage1.
FREEZE_VISUAL="${REFINEMENT_FREEZE_VISUAL:-false}"
# Learning rate of metric_heads (unset = same as LoRA), schedule and warmup of the online run.
HEAD_LEARNING_RATE="${REFINEMENT_HEAD_LEARNING_RATE:-}"
HEAD_LEARNING_RATE_ARGS=()
if [ -n "$HEAD_LEARNING_RATE" ]; then
    HEAD_LEARNING_RATE_ARGS=(--head-learning-rate "$HEAD_LEARNING_RATE")
fi
LR_SCHEDULER_TYPE="${REFINEMENT_LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${REFINEMENT_WARMUP_RATIO:-0.05}"
HARD_REPLAY_FRACTION="${REFINEMENT_HARD_REPLAY_FRACTION:-0.25}"
LOW_AGREEMENT_FRACTION="${REFINEMENT_LOW_AGREEMENT_FRACTION:-0.25}"
MAX_BOX_ONLY_FRACTION="${REFINEMENT_MAX_BOX_ONLY_FRACTION:-0.25}"
BOX_ONLY_SET_WEIGHT="${REFINEMENT_BOX_ONLY_SET_WEIGHT:-0.25}"
MIN_ROUND_FRACTION="${REFINEMENT_MIN_ROUND_FRACTION:-0.50}"
MIN_SUPERVISED_FRACTION="${REFINEMENT_MIN_SUPERVISED_FRACTION:-0.15}"
HIGH_INSPECTOR_CONFIDENCE="${REFINEMENT_HIGH_INSPECTOR_CONFIDENCE:-0.80}"
SCREENING_MULTIPLIER="${REFINEMENT_SCREENING_MULTIPLIER:-4}"
MAX_REPLAYS_PER_SAMPLE="${REFINEMENT_MAX_REPLAYS_PER_SAMPLE:-2}"
PATIENCE_ROUNDS="${REFINEMENT_PATIENCE_ROUNDS:-2}"
MIN_VALIDATION_IMPROVEMENT="${REFINEMENT_MIN_VALIDATION_IMPROVEMENT:-0.002}"
EVAL_SAVE_STEPS="${REFINEMENT_EVAL_SAVE_STEPS:-200}"
EVAL_SAVE_TOTAL_LIMIT="${REFINEMENT_EVAL_SAVE_TOTAL_LIMIT:-5}"
LOGGING_STEPS="${REFINEMENT_LOGGING_STEPS:-10}"
PAIR_BATCH_PROFILE="${REFINEMENT_PAIR_BATCH_PROFILE:-16}"
case "$PAIR_BATCH_PROFILE" in
    16) DEFAULT_HUMAN_BATCH=5; DEFAULT_GEMMA_BATCH=5; DEFAULT_UNLABELED_BATCH=6 ;;
    32) DEFAULT_HUMAN_BATCH=10; DEFAULT_GEMMA_BATCH=10; DEFAULT_UNLABELED_BATCH=12 ;;
    *) echo "ERROR: REFINEMENT_PAIR_BATCH_PROFILE must be 16 or 32" >&2; exit 1 ;;
esac
HUMAN_BATCH_SIZE="${REFINEMENT_HUMAN_BATCH_SIZE:-$DEFAULT_HUMAN_BATCH}"
GEMMA_BATCH_SIZE="${REFINEMENT_GEMMA_BATCH_SIZE:-$DEFAULT_GEMMA_BATCH}"
UNLABELED_BATCH_SIZE="${REFINEMENT_UNLABELED_BATCH_SIZE:-$DEFAULT_UNLABELED_BATCH}"
TEACHER_BATCH_SIZE="${REFINEMENT_TEACHER_BATCH_SIZE:-6}"
EVAL_BATCH_SIZE="${REFINEMENT_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${REFINEMENT_GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${REFINEMENT_LEARNING_RATE:-1e-5}"
# theta0/kappa_raw of the margin-coupled Tie Head in their own group (no weight decay); empty = shared.
TIE_COUPLING_LEARNING_RATE="${REFINEMENT_TIE_COUPLING_LEARNING_RATE:-}"
TIE_COUPLING_ARGS=()
if [ -n "$TIE_COUPLING_LEARNING_RATE" ]; then
    TIE_COUPLING_ARGS=(--tie-coupling-learning-rate "$TIE_COUPLING_LEARNING_RATE")
fi
LAMBDA_SET="${REFINEMENT_LAMBDA_SET:-1.0}"
UNLABELED_OBJECTIVE="${REFINEMENT_UNLABELED_OBJECTIVE:-projection}"
REPLAY_STRATEGY="${REFINEMENT_REPLAY_STRATEGY:-hard}"
UNLABELED_SAMPLING="${REFINEMENT_UNLABELED_SAMPLING:-agreement_strata}"
LOSS_PLATEAU_TOLERANCE="${REFINEMENT_LOSS_PLATEAU_TOLERANCE:-0.01}"
GEMMA_TARGETS="${REFINEMENT_GEMMA_TARGETS:-majority_vote}"
OVERALL_AGGREGATION="${REFINEMENT_OVERALL_AGGREGATION:-all_metrics_fitted}"
OVERALL_PROBABILITY="${REFINEMENT_OVERALL_PROBABILITY:-sigmoid_beta}"
OVERALL_LOSS="${REFINEMENT_OVERALL_LOSS:-binary}"
HUMAN_SPLIT="${REFINEMENT_HUMAN_SPLIT:-inspector}"
CHECKPOINT_SELECTION="${REFINEMENT_CHECKPOINT_SELECTION:-validation}"
EXCLUDE_PROBE_FROM_TRAINING="${REFINEMENT_EXCLUDE_PROBE_FROM_TRAINING:-true}"
CONSISTENCY_METRICS="${REFINEMENT_CONSISTENCY_METRICS:-all}"
LAMBDA_CONSISTENCY="${REFINEMENT_LAMBDA_CONSISTENCY:-1.0}"
SELECTION_MODEL="${REFINEMENT_SELECTION_MODEL:-student}"
SEED="${REFINEMENT_SEED:-42}"
# Fixed data selections (probe, Unlabeled subset, Gemma monitoring pairs); default = SEED.
SELECTION_SEED="${REFINEMENT_SELECTION_SEED:-$SEED}"
# true: one 'step=N diagnostics {json}' log line per step (merged labeled source only).
STEP_DIAGNOSTICS="${REFINEMENT_STEP_DIAGNOSTICS:-false}"
STEP_DIAGNOSTICS_ARGS=()
case "$STEP_DIAGNOSTICS" in
    true) STEP_DIAGNOSTICS_ARGS=(--step-diagnostics) ;;
    false) ;;
    *) echo "ERROR: REFINEMENT_STEP_DIAGNOSTICS must be true or false, got '$STEP_DIAGNOSTICS'" >&2; exit 1 ;;
esac
# true: one JSON line per trained pair and step (Inspector / Teacher / Student probabilities)
# in <output dir>/per_sample_log_<job>.jsonl (merged labeled source only); the job id keeps a
# resumed run from refusing an existing file. logs/ holds only the Slurm logs.
PER_SAMPLE_LOG="${REFINEMENT_PER_SAMPLE_LOG:-false}"
PER_SAMPLE_LOG_ARGS=()
case "$PER_SAMPLE_LOG" in
    true) PER_SAMPLE_LOG_ARGS=(--per-sample-log true --per-sample-log-path "$OUTPUT_DIR/per_sample_log_${SLURM_JOB_ID:-manual}.jsonl") ;;
    false) ;;
    *) echo "ERROR: REFINEMENT_PER_SAMPLE_LOG must be true or false, got '$PER_SAMPLE_LOG'" >&2; exit 1 ;;
esac
# Fixed Unlabeled subset from a file (score_teacher_inspector_agreement.py) and online
# augmentation of the Student input (student_augmentation.py); empty / none = off.
UNLABELED_SUBSET_FILE="${REFINEMENT_UNLABELED_SUBSET_FILE:-}"
STUDENT_AUGMENTATION="${REFINEMENT_STUDENT_AUGMENTATION:-none}"
AUGMENTATION_SCENARIOS="${REFINEMENT_AUGMENTATION_SCENARIOS:-}"
AUGMENTATION_ARGS=(--student-augmentation "$STUDENT_AUGMENTATION")
[ -n "$UNLABELED_SUBSET_FILE" ] && AUGMENTATION_ARGS+=(--unlabeled-subset-file "$UNLABELED_SUBSET_FILE")
[ -n "$AUGMENTATION_SCENARIOS" ] && AUGMENTATION_ARGS+=(--augmentation-scenarios "$AUGMENTATION_SCENARIOS")
# single_operation: the one operation of student_augmentation.SINGLE_OPERATIONS (2026-11-05).
AUGMENTATION_OPERATION="${REFINEMENT_AUGMENTATION_OPERATION:-}"
[ -n "$AUGMENTATION_OPERATION" ] && AUGMENTATION_ARGS+=(--augmentation-operation "$AUGMENTATION_OPERATION")
# soft (default) or one_hot target of CE(Inspector, Student overall) (2026-11-06).
INSPECTOR_TARGET="${REFINEMENT_INSPECTOR_TARGET:-soft}"
AUGMENTATION_ARGS+=(--inspector-target "$INSPECTOR_TARGET")
# true = the labeled pairs (HumanGT + GemmaGT) get the same augmentation as the Unlabeled samples.
AUGMENT_LABELED="${REFINEMENT_AUGMENT_LABELED:-false}"
[ "$AUGMENT_LABELED" = true ] && AUGMENTATION_ARGS+=(--augment-labeled)
# Share w of KL_rel in the Teacher consistency, (1 - w) KL_pref + w KL_rel; 0 = previous loss (2026-11-07).
TEACHER_KL_RELEVANCE_WEIGHT="${REFINEMENT_TEACHER_KL_RELEVANCE_WEIGHT:-0}"
# K checkpoints per iteration instead of every EVAL_SAVE_STEPS (set REFINEMENT_EVAL_SAVE_STEPS=0); 0 = off.
CHECKPOINTS_PER_ITERATION="${REFINEMENT_CHECKPOINTS_PER_ITERATION:-0}"
KL_RELEVANCE_CHECKPOINT_ARGS=()
[ "$TEACHER_KL_RELEVANCE_WEIGHT" != 0 ] && KL_RELEVANCE_CHECKPOINT_ARGS+=(--teacher-kl-relevance-weight "$TEACHER_KL_RELEVANCE_WEIGHT")
[ "$CHECKPOINTS_PER_ITERATION" != 0 ] && KL_RELEVANCE_CHECKPOINT_ARGS+=(--checkpoints-per-iteration "$CHECKPOINTS_PER_ITERATION")
# round_<N>.pt at the end of every round (all, default), only round 1 (first) or none (2026-11-07).
SAVE_ROUND_CHECKPOINTS="${REFINEMENT_SAVE_ROUND_CHECKPOINTS:-all}"
KL_RELEVANCE_CHECKPOINT_ARGS+=(--save-round-checkpoints "$SAVE_ROUND_CHECKPOINTS")
HANDWRITTEN_INSTRUCTIONS="${REFINEMENT_HANDWRITTEN_INSTRUCTIONS:-}"
AUGMENTATION_EXAMPLES="${REFINEMENT_AUGMENTATION_EXAMPLES:-0}"
[ "$AUGMENTATION_EXAMPLES" != 0 ] && AUGMENTATION_ARGS+=(--augmentation-examples "$AUGMENTATION_EXAMPLES")
[ -n "$HANDWRITTEN_INSTRUCTIONS" ] && AUGMENTATION_ARGS+=(--handwritten-instructions "$HANDWRITTEN_INSTRUCTIONS")
for required_file in "$UNLABELED_SUBSET_FILE" "$AUGMENTATION_SCENARIOS" "$HANDWRITTEN_INSTRUCTIONS"; do
    if [ -n "$required_file" ] && [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done
SET_TARGET_CONTRIBUTION="${REFINEMENT_SET_TARGET_CONTRIBUTION:-0.10}"
SET_TARGET_PROJECTION_RATE="${REFINEMENT_SET_TARGET_PROJECTION_RATE:-0.35}"
SET_SCALE_MIN="${REFINEMENT_SET_SCALE_MIN:-0.5}"
SET_SCALE_MAX="${REFINEMENT_SET_SCALE_MAX:-4.0}"
SET_SCALE_MOMENTUM="${REFINEMENT_SET_SCALE_MOMENTUM:-0.5}"
LAMBDA_GEMMA="${REFINEMENT_LAMBDA_GEMMA:-1.0}"
LAMBDA_MASK="${REFINEMENT_LAMBDA_MASK:-1.0}"
SET_RAMP_STEPS="${REFINEMENT_SET_RAMP_STEPS:-0}"
SEMANTIC_DRIFT_THRESHOLD="${REFINEMENT_SEMANTIC_DRIFT_THRESHOLD:-0.02}"
EMA_DECAY="${REFINEMENT_EMA_DECAY:-0.99}"
EMA_SCHEDULE="${REFINEMENT_EMA_SCHEDULE:-constant}"
EMA_DECAY_START="${REFINEMENT_EMA_DECAY_START:-0.99}"
EMA_DECAY_END="${REFINEMENT_EMA_DECAY_END:-0.999}"
KAPPA="${REFINEMENT_KAPPA:-0.05}"
CACHE_SLACK="${REFINEMENT_CACHE_SLACK:-0.02}"
PROJECTION_ITERATIONS="${REFINEMENT_PROJECTION_ITERATIONS:-40}"

# linear + selected_equal without projection reads neither a_k nor beta.
AGGREGATOR_ARGS=(--metrics-weights-aggregator "$METRICS_WEIGHTS_AGGREGATOR")
if [ "$TRAIN_LOOP_MODE" = "online_swift" ] && [ "$OVERALL_PROBABILITY" = "linear" ] \
    && [ "$OVERALL_AGGREGATION" = "selected_equal" ] && [ "$UNLABELED_OBJECTIVE" != "projection" ]; then
    AGGREGATOR_ARGS=()
    METRICS_WEIGHTS_AGGREGATOR="not used"
fi
if [ -z "$METRICS_WEIGHTS_AGGREGATOR" ]; then
    echo "ERROR: Set TRAIN_LOOP_METRICS_WEIGHTS_AGGREGATOR; this configuration reads the fitted a_k or beta" >&2
    exit 1
fi

# With the train_order HumanGT split the Inspector is an MS-Swift checkpoint folder.
if [ "$HUMAN_SPLIT" = "student_train_order" ]; then
    if [ ! -f "$INSPECTOR_CHECKPOINT/adapter_model.safetensors" ]; then
        echo "ERROR: Inspector MS-Swift checkpoint not found: $INSPECTOR_CHECKPOINT" >&2
        exit 1
    fi
    INSPECTOR_FILE_CHECK=()
else
    INSPECTOR_FILE_CHECK=("$INSPECTOR_CHECKPOINT")
fi
for required_file in \
    "$STUDENT_CHECKPOINT" \
    "$TEACHER_CHECKPOINT" \
    ${INSPECTOR_FILE_CHECK[@]:+"${INSPECTOR_FILE_CHECK[@]}"} \
    "$INSPECTOR_CACHE" \
    ${AGGREGATOR_ARGS[@]:+"$METRICS_WEIGHTS_AGGREGATOR"} \
    "$HUMAN_DATASET" \
    "$GEMMA_DATASET" \
    "$UNLABELED_DATASET"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL_PATH" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Confirmed output directory not found: $OUTPUT_DIR" >&2
    exit 1
fi
if [ "$(realpath "$OUTPUT_DIR")" = "$(realpath "$PROJECT_DIR/output/training_loop")" ]; then
    echo "ERROR: The previous training_loop output must remain separate." >&2
    exit 1
fi
if [ ! -f "$PROJECT_DIR/$TRAINING_SCRIPT" ]; then
    echo "ERROR: Training script not found: $PROJECT_DIR/$TRAINING_SCRIPT" >&2
    exit 1
fi
if [ -e "$OUTPUT" ]; then
    echo "ERROR: Refusing to overwrite existing final checkpoint: $OUTPUT" >&2
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

ulimit -n 65537
cd "$PROJECT_DIR"

if [ "${TRAIN_LOOP_MODE:-online}" = "supervised_control" ]; then
    # Match the completed online run's actual optimizer steps and supervised
    # pair exposure. Do not guess the final quality-gated round length.
    if [ "$HUMAN_BATCH_SIZE" -le 0 ] || [ "$GEMMA_BATCH_SIZE" -le 0 ]; then
        echo "ERROR: supervised control needs positive H and G batch sizes." >&2
        exit 1
    fi
    export CONTROL_REFERENCE_DIR="${CONTROL_REFERENCE_DIR:-$PROJECT_DIR/output/training_loop_ms_swift_online}"
    CONTROL_MAX_STEPS="$(python -B -c 'import json,sys; from pathlib import Path; p=sorted(Path(sys.argv[1]).glob("round_*_summary.json")); assert p, "No completed online rounds"; print(json.loads(p[-1].read_text())["optimizer_step"])' "$CONTROL_REFERENCE_DIR")"
    if [ "$CONTROL_MAX_STEPS" -le 0 ]; then
        echo "ERROR: The online reference has no optimizer steps." >&2
        exit 1
    fi
    export CONTROL_STUDENT_CHECKPOINT="$STUDENT_CHECKPOINT"
    export CONTROL_INSPECTOR_CHECKPOINT="$INSPECTOR_CHECKPOINT"
    export CONTROL_AGGREGATOR="$METRICS_WEIGHTS_AGGREGATOR"
    export CONTROL_HUMAN_DATASET="$HUMAN_DATASET"
    export CONTROL_GEMMA_DATASET="$GEMMA_DATASET"
    export CONTROL_HUMAN_BATCH_SIZE="$HUMAN_BATCH_SIZE"
    export CONTROL_GEMMA_BATCH_SIZE="$GEMMA_BATCH_SIZE"
    export CONTROL_SCHEDULER_STEPS="$((MAX_ROUNDS * ((UNLABELED_PER_ROUND + UNLABELED_BATCH_SIZE - 1) / UNLABELED_BATCH_SIZE)))"

    echo "MS-Swift supervised control: $CONTROL_MAX_STEPS steps, H=$HUMAN_BATCH_SIZE, G=$GEMMA_BATCH_SIZE, U=0"
    echo "Reference: $CONTROL_REFERENCE_DIR"
    echo "Output: $OUTPUT_DIR"
    python "$TRAINING_SCRIPT" \
        --model "$BASE_MODEL_PATH" \
        --dataset "$GEMMA_DATASET" \
        --task_type generative_reranker \
        --template qwen3_vl_reranker \
        --loss_type factorized_metric_supervised_control \
        --target_modules all-linear \
        --tuner_type lora \
        --tuner_backend peft \
        --freeze_llm false \
        --freeze_vit false \
        --freeze_aligner false \
        --lora_rank 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --modules_to_save metric_heads \
        --learning_rate "$LEARNING_RATE" \
        --weight_decay 0.01 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.05 \
        --attn_impl sdpa \
        --torch_dtype bfloat16 \
        --bf16 true \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size "$((HUMAN_BATCH_SIZE + GEMMA_BATCH_SIZE))" \
        --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
        --gradient_accumulation_steps 1 \
        --max_steps "$CONTROL_MAX_STEPS" \
        --eval_strategy steps \
        --save_strategy steps \
        --eval_steps 250 \
        --save_steps 250 \
        --save_total_limit "$MAX_ROUNDS" \
        --save_only_model false \
        --load_best_model_at_end true \
        --metric_for_best_model accuracy \
        --greater_is_better true \
        --logging_steps 10 \
        --seed 42 \
        --data_seed 42 \
        --train_dataloader_shuffle false \
        --remove_unused_columns false \
        --packing false \
        --padding_free false \
        --sequence_parallel_size 1 \
        --max_length 16384 \
        --dataloader_num_workers 0 \
        --dataset_num_proc 1 \
        --load_from_cache_file false \
        --strict true \
        --gradient_checkpointing true \
        --add_version false
    echo "MS-Swift supervised control completed: $OUTPUT_DIR/best.pt"
    exit 0
fi

echo "======================================================================"
echo "BUDGETED ONLINE REFINEMENT ROUNDS"
echo "Training backend:     MS-Swift online reranker trainer"
echo "Student:             $STUDENT_CHECKPOINT"
echo "Base model:          $BASE_MODEL_PATH"
echo "Teacher:             $TEACHER_CHECKPOINT"
echo "Inspector cache:     $INSPECTOR_CACHE"
echo "Metrics weights:     $METRICS_WEIGHTS_AGGREGATOR"
echo "Human dataset:       $HUMAN_DATASET"
echo "Gemma dataset:       $GEMMA_DATASET"
echo "Unlabeled dataset:   $UNLABELED_DATASET"
echo "Final checkpoint:    $OUTPUT"
echo "Training script:     $TRAINING_SCRIPT"
echo "Maximum rounds:      $MAX_ROUNDS"
echo "Unlabeled per round: $UNLABELED_PER_ROUND"
echo "Fixed probe size:    $DIAGNOSTIC_PROBE_SIZE"
echo "Unlabeled subset / passes: $UNLABELED_SUBSET_SIZE / $UNLABELED_PASSES (0 = all Unlabeled)"
echo "Gemma monitoring pairs: $GEMMA_MONITOR_PAIRS (0 = all training pairs)"
echo "Sampling curriculum: round 1 high agreement only; later high/low/replay"
echo "Later-round targets: high=remainder, low=$LOW_AGREEMENT_FRACTION, replay=$HARD_REPLAY_FRACTION (soft)"
echo "Replay strategy:     $REPLAY_STRATEGY"
echo "Unlabeled sampling:  $UNLABELED_SAMPLING"
echo "Loss plateau tolerance (previous_iteration_teacher_inspector_agreement): $LOSS_PLATEAU_TOLERANCE"
echo "Gemma replay targets: $GEMMA_TARGETS (validation: soft)"
echo "Box-only cap/weight: $MAX_BOX_ONLY_FRACTION / $BOX_ONLY_SET_WEIGHT"
echo "Minimum round size:  $MIN_ROUND_FRACTION; minimum Human and Gemma share each: $MIN_SUPERVISED_FRACTION"
echo "High Inspector confidence: $HIGH_INSPECTOR_CONFIDENCE; screening multiplier: $SCREENING_MULTIPLIER"
echo "Patience / min gain: $PATIENCE_ROUNDS / $MIN_VALIDATION_IMPROVEMENT"
echo "Step evaluation:     every $EVAL_SAVE_STEPS steps, keep $EVAL_SAVE_TOTAL_LIMIT checkpoints plus best; logging every $LOGGING_STEPS"
echo "MC-Dropout passes:   4"
echo "Microbatches:        H=$HUMAN_BATCH_SIZE, G=$GEMMA_BATCH_SIZE, U=$UNLABELED_BATCH_SIZE, accumulation=$GRADIENT_ACCUMULATION_STEPS"
echo "Pair batch profile:  $PAIR_BATCH_PROFILE (total configured pairs=$((HUMAN_BATCH_SIZE + GEMMA_BATCH_SIZE + UNLABELED_BATCH_SIZE)))"
echo "Loss coefficients:   human=$LAMBDA_HUMAN, G=$LAMBDA_GEMMA, mask=$LAMBDA_MASK, set=$LAMBDA_SET"
echo "Labeled source: $LABELED_SOURCE (labeled pairs per step: $LABELED_PAIRS_PER_STEP); loss weighting: $LOSS_WEIGHTING; Teacher MC passes: $MC_PASSES"
echo "Teacher dropout: $TEACHER_DROPOUT; Teacher update: $TEACHER_UPDATE"
echo "ViT/aligner frozen:  $FREEZE_VISUAL"
echo "Per-sample log:      $PER_SAMPLE_LOG"
echo "Unlabeled subset file: ${UNLABELED_SUBSET_FILE:-none (random subset)}"
echo "Student augmentation:  $STUDENT_AUGMENTATION ${AUGMENTATION_SCENARIOS}"
echo "Augmentation operation: ${AUGMENTATION_OPERATION:-none}"
echo "Inspector target:       $INSPECTOR_TARGET"
echo "Augment labeled pairs:  $AUGMENT_LABELED"
echo "Teacher KL relevance weight: $TEACHER_KL_RELEVANCE_WEIGHT; checkpoints per iteration: $CHECKPOINTS_PER_ITERATION; round checkpoints: $SAVE_ROUND_CHECKPOINTS"
echo "Hand-written instructions: ${HANDWRITTEN_INSTRUCTIONS:-none}"
echo "Learning rate:       $LEARNING_RATE; heads: ${HEAD_LEARNING_RATE:-shared}; theta0/kappa: ${TIE_COUPLING_LEARNING_RATE:-shared}; schedule: $LR_SCHEDULER_TYPE, warmup ratio $WARMUP_RATIO"
echo "Unlabeled objective: $UNLABELED_OBJECTIVE (consistency=$LAMBDA_CONSISTENCY); selection by $SELECTION_MODEL; seed=$SEED; selection seed=$SELECTION_SEED"
echo "Step diagnostics:    $STEP_DIAGNOSTICS"
echo "Overall aggregation: $OVERALL_AGGREGATION; overall probability: $OVERALL_PROBABILITY; overall loss: $OVERALL_LOSS"
echo "Probe excluded from training: $EXCLUDE_PROBE_FROM_TRAINING"
echo "HumanGT split: $HUMAN_SPLIT; checkpoint selection: $CHECKPOINT_SELECTION"
echo "Consistency metrics: $CONSISTENCY_METRICS"
echo "Adaptive set target: contribution=$SET_TARGET_CONTRIBUTION projection=$SET_TARGET_PROJECTION_RATE scale=[$SET_SCALE_MIN,$SET_SCALE_MAX]"
echo "Set ramp / EMA:      $SET_RAMP_STEPS / $EMA_DECAY (schedule=$EMA_SCHEDULE, cosine $EMA_DECAY_START -> $EMA_DECAY_END)"
echo "Semantic drift threshold: $SEMANTIC_DRIFT_THRESHOLD"
echo "Kappa / cache slack: $KAPPA / $CACHE_SLACK"
echo "======================================================================"

RESUME_ARGS=()
if [ -n "${REFINEMENT_RESUME_CHECKPOINT:-}" ]; then
    RESUME_ARGS=(--resume-checkpoint "$REFINEMENT_RESUME_CHECKPOINT")
fi

python "$TRAINING_SCRIPT" "${RESUME_ARGS[@]}" \
    --student-checkpoint "$STUDENT_CHECKPOINT" \
    --base-model-path "$BASE_MODEL_PATH" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --inspector-checkpoint "$INSPECTOR_CHECKPOINT" \
    --inspector-cache "$INSPECTOR_CACHE" \
    "${AGGREGATOR_ARGS[@]}" \
    --human-dataset "$HUMAN_DATASET" \
    --gemma-dataset "$GEMMA_DATASET" \
    --unlabeled-dataset "$UNLABELED_DATASET" \
    --output "$OUTPUT" \
    --max-rounds "$MAX_ROUNDS" \
    --unlabeled-per-round "$UNLABELED_PER_ROUND" \
    --diagnostic-probe-size "$DIAGNOSTIC_PROBE_SIZE" \
    --unlabeled-subset-size "$UNLABELED_SUBSET_SIZE" \
    --unlabeled-passes "$UNLABELED_PASSES" \
    --gemma-monitor-pairs "$GEMMA_MONITOR_PAIRS" \
    --labeled-source "$LABELED_SOURCE" \
    --labeled-pairs-per-step "$LABELED_PAIRS_PER_STEP" \
    --loss-weighting "$LOSS_WEIGHTING" \
    --lambda-human "$LAMBDA_HUMAN" \
    --hard-replay-fraction "$HARD_REPLAY_FRACTION" \
    --replay-strategy "$REPLAY_STRATEGY" \
    --unlabeled-sampling "$UNLABELED_SAMPLING" \
    --loss-plateau-tolerance "$LOSS_PLATEAU_TOLERANCE" \
    --gemma-targets "$GEMMA_TARGETS" \
    --overall-aggregation "$OVERALL_AGGREGATION" \
    --overall-probability "$OVERALL_PROBABILITY" \
    --overall-loss "$OVERALL_LOSS" \
    --exclude-probe-from-training "$EXCLUDE_PROBE_FROM_TRAINING" \
    --human-split "$HUMAN_SPLIT" \
    --checkpoint-selection "$CHECKPOINT_SELECTION" \
    --consistency-metrics "$CONSISTENCY_METRICS" \
    --low-agreement-fraction "$LOW_AGREEMENT_FRACTION" \
    --max-box-only-fraction "$MAX_BOX_ONLY_FRACTION" \
    --box-only-set-weight "$BOX_ONLY_SET_WEIGHT" \
    --min-round-fraction "$MIN_ROUND_FRACTION" \
    --min-supervised-fraction "$MIN_SUPERVISED_FRACTION" \
    --high-inspector-confidence "$HIGH_INSPECTOR_CONFIDENCE" \
    --screening-multiplier "$SCREENING_MULTIPLIER" \
    --max-replays-per-sample "$MAX_REPLAYS_PER_SAMPLE" \
    --patience-rounds "$PATIENCE_ROUNDS" \
    --min-validation-improvement "$MIN_VALIDATION_IMPROVEMENT" \
    --eval-save-steps "$EVAL_SAVE_STEPS" \
    --eval-save-total-limit "$EVAL_SAVE_TOTAL_LIMIT" \
    --human-batch-size "$HUMAN_BATCH_SIZE" \
    --gemma-batch-size "$GEMMA_BATCH_SIZE" \
    --unlabeled-batch-size "$UNLABELED_BATCH_SIZE" \
    --teacher-batch-size "$TEACHER_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate "$LEARNING_RATE" \
    --lr-scheduler-type "$LR_SCHEDULER_TYPE" \
    --warmup-ratio "$WARMUP_RATIO" \
    ${HEAD_LEARNING_RATE_ARGS[@]+"${HEAD_LEARNING_RATE_ARGS[@]}"} \
    ${TIE_COUPLING_ARGS[@]:+"${TIE_COUPLING_ARGS[@]}"} \
    --unlabeled-objective "$UNLABELED_OBJECTIVE" \
    --lambda-consistency "$LAMBDA_CONSISTENCY" \
    --selection-model "$SELECTION_MODEL" \
    --lambda-set "$LAMBDA_SET" \
    --set-target-contribution "$SET_TARGET_CONTRIBUTION" \
    --set-target-projection-rate "$SET_TARGET_PROJECTION_RATE" \
    --set-scale-min "$SET_SCALE_MIN" \
    --set-scale-max "$SET_SCALE_MAX" \
    --set-scale-momentum "$SET_SCALE_MOMENTUM" \
    --lambda-gemma "$LAMBDA_GEMMA" \
    --lambda-mask "$LAMBDA_MASK" \
    --set-ramp-steps "$SET_RAMP_STEPS" \
    --semantic-drift-threshold "$SEMANTIC_DRIFT_THRESHOLD" \
    --ema-decay "$EMA_DECAY" \
    --ema-schedule "$EMA_SCHEDULE" \
    --ema-decay-start "$EMA_DECAY_START" \
    --ema-decay-end "$EMA_DECAY_END" \
    --mc-passes "$MC_PASSES" \
    --teacher-dropout "$TEACHER_DROPOUT" \
    --teacher-update "$TEACHER_UPDATE" \
    --freeze-visual "$FREEZE_VISUAL" \
    --kappa "$KAPPA" \
    --cache-slack "$CACHE_SLACK" \
    --projection-iterations "$PROJECTION_ITERATIONS" \
    --logging-steps "$LOGGING_STEPS" \
    --seed "$SEED" \
    --selection-seed "$SELECTION_SEED" \
    "${STEP_DIAGNOSTICS_ARGS[@]}" \
    ${PER_SAMPLE_LOG_ARGS[@]+"${PER_SAMPLE_LOG_ARGS[@]}"} \
    "${AUGMENTATION_ARGS[@]}" \
    ${KL_RELEVANCE_CHECKPOINT_ARGS[@]+"${KL_RELEVANCE_CHECKPOINT_ARGS[@]}"} \
    --device cuda

echo "Refinement training completed successfully."
echo "Final checkpoint: $OUTPUT"
