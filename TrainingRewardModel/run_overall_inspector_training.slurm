#!/bin/bash
#SBATCH --job-name=overall_inspector
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/overall_inspector_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/overall_inspector_%j.err
#SBATCH --account=uit
#SBATCH --partition=full
#SBATCH --qos=mig7-141
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:7g.141gb:1

# Three-way Overall Inspector (P(A), P(B), P(Tie)).  Defaults are the
# supervised baseline on the whole EditReward-Data: one pass, no validation,
# batch 4 x 4, flash attention, no gradient checkpointing.  For a job under
# 35 GB submit with: sbatch -p normal --qos=normal --gres=gpu:2g.35gb:1 ...

set -euo pipefail

PAIR_BATCH_SIZE="${INSPECTOR_PAIR_BATCH_SIZE:-4}"
EVAL_PAIR_BATCH_SIZE="${INSPECTOR_EVAL_PAIR_BATCH_SIZE:-4}"
GRAD_ACCUMULATION_STEPS="${INSPECTOR_GRAD_ACCUMULATION_STEPS:-4}"
LORA_RANK="${INSPECTOR_LORA_RANK:-32}"
LORA_ALPHA=$((2 * LORA_RANK))
LEARNING_RATE="${INSPECTOR_LEARNING_RATE:-4e-5}"
LR_SCHEDULER="${INSPECTOR_LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${INSPECTOR_WARMUP_RATIO:-0.05}"
DATA_PASSES="${OVERALL_INSPECTOR_DATA_PASSES:-1}"
FREEZE_VISUAL="${INSPECTOR_FREEZE_VISUAL:-true}"
GRADIENT_CHECKPOINTING="${INSPECTOR_GRADIENT_CHECKPOINTING:-false}"
ATTN_IMPL="${INSPECTOR_ATTN_IMPL:-flash_attn}"
VALIDATION_RATIO="${INSPECTOR_VALIDATION_RATIO:-0}"
SAVE_STEPS="${INSPECTOR_SAVE_STEPS:-1000}"
EVAL_STEPS="${INSPECTOR_EVAL_STEPS:-10}"
# Checkpoints kept when a validation split exists; -1 keeps all (Inspector Baseline).
SAVE_TOTAL_LIMIT="${INSPECTOR_SAVE_TOTAL_LIMIT:-5}"
MAX_STEPS="${INSPECTOR_MAX_STEPS:--1}"
NUM_WORKERS="${INSPECTOR_DATALOADER_NUM_WORKERS:-8}"

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
INSPECTOR_DATA="${OVERALL_INSPECTOR_DATA:-/datastore/shared/datasets/EditReward-Data/extracted_data/metadata.jsonl}"
OUTPUT_DIR="${INSPECTOR_OUTPUT_DIR:?Set INSPECTOR_OUTPUT_DIR to the confirmed output directory}"
MODEL_PATH="${OVERALL_INSPECTOR_MODEL_PATH:-/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Reranker-8B}"
RESUME_FROM_CHECKPOINT="${OVERALL_INSPECTOR_RESUME_FROM_CHECKPOINT:-}"
export OVERALL_INSPECTOR_SPLIT="${OVERALL_INSPECTOR_SPLIT:-}"
# JSONL of pairs kept out of the validation split and trained first.
export OVERALL_INSPECTOR_TRAIN_PREFIX="${OVERALL_INSPECTOR_TRAIN_PREFIX:-}"
export OVERALL_INSPECTOR_DATA_PASSES="$DATA_PASSES"
export OVERALL_INSPECTOR_SAVE_PAIR_MILESTONES="${OVERALL_INSPECTOR_SAVE_PAIR_MILESTONES:-}"
# 1: also score the validation split at each pair milestone and the last step.
export OVERALL_INSPECTOR_EVAL_AT_PAIR_MILESTONES="${OVERALL_INSPECTOR_EVAL_AT_PAIR_MILESTONES:-0}"

if [ -e "$OUTPUT_DIR/overall_inspector_checkpoint.pt" ]; then
    echo "ERROR: Inspector training is already finalized; reuse the existing checkpoint: $OUTPUT_DIR/overall_inspector_checkpoint.pt" >&2
    exit 1
fi
if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null && [ -z "$RESUME_FROM_CHECKPOINT" ]; then
    echo "ERROR: An Inspector training checkpoint already exists. Refusing to restart without an explicit resume checkpoint." >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Base model not found: $MODEL_PATH" >&2
    exit 1
fi
for shard in \
    model-00001-of-00004.safetensors \
    model-00002-of-00004.safetensors \
    model-00003-of-00004.safetensors \
    model-00004-of-00004.safetensors; do
    if [ ! -s "$MODEL_PATH/$shard" ]; then
        echo "ERROR: Shared base model is incomplete; missing shard: $MODEL_PATH/$shard" >&2
        exit 1
    fi
done
if [ ! -f "$INSPECTOR_DATA" ]; then
    echo "ERROR: Inspector data not found: $INSPECTOR_DATA" >&2
    exit 1
fi
if [ -n "$OVERALL_INSPECTOR_SPLIT" ] && [ ! -f "$OVERALL_INSPECTOR_SPLIT" ]; then
    echo "ERROR: Split file not found: $OVERALL_INSPECTOR_SPLIT" >&2
    exit 1
fi

# MS-Swift disables evaluation when --split_dataset_ratio is 0 and no
# --val_dataset is given.  With a split file the ratio is never used for the
# split itself (the file wins), so a nonzero placeholder keeps evaluation on.
SPLIT_RATIO_ARG="$VALIDATION_RATIO"
if [ -n "$OVERALL_INSPECTOR_SPLIT" ] && [ "$VALIDATION_RATIO" = "0" ]; then
    SPLIT_RATIO_ARG="0.01"
fi

EVAL_ARGS=(--eval_strategy no)
if [ -n "$OVERALL_INSPECTOR_SPLIT" ] || [ "$VALIDATION_RATIO" != "0" ]; then
    EVAL_ARGS=(
        --eval_strategy steps
        --eval_steps "$EVAL_STEPS"
        --metric_for_best_model loss
    )
    if [ "$SAVE_TOTAL_LIMIT" != "-1" ]; then
        EVAL_ARGS+=(--save_total_limit "$SAVE_TOTAL_LIMIT")
    fi
fi

RESUME_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    if [ ! -s "$RESUME_FROM_CHECKPOINT/trainer_state.json" ] \
        || [ ! -s "$RESUME_FROM_CHECKPOINT/optimizer.pt" ] \
        || [ ! -s "$RESUME_FROM_CHECKPOINT/scheduler.pt" ]; then
        echo "ERROR: Resume checkpoint is incomplete: $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    fi
    # The trainer restores optimizer and schedule itself (torch < 2.6).
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl_flash
mkdir -p "$OUTPUT_DIR"

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export MIN_PIXELS=4096
export INSPECTOR_EDITREWARD_ROOT="/datastore/shared/datasets/EditReward-Data"
# Tie Head chosen 26/10/2026; set TIE_HEAD_MODE=independent explicitly for the old head.
export TIE_HEAD_MODE="${TIE_HEAD_MODE:-margin_coupled}"

ulimit -n 65537
cd "$PROJECT_DIR"

echo "======================================================================"
echo "TRAINING THREE-WAY OVERALL INSPECTOR MODEL WITH MS-SWIFT"
echo "Model:              $MODEL_PATH"
echo "Data:               $INSPECTOR_DATA"
echo "Split file:         ${OVERALL_INSPECTOR_SPLIT:-none}"
echo "Train prefix:       ${OVERALL_INSPECTOR_TRAIN_PREFIX:-none}"
echo "Validation ratio:   $VALIDATION_RATIO"
echo "Eval / save steps:  $EVAL_STEPS / $SAVE_STEPS (kept checkpoints: $SAVE_TOTAL_LIMIT)"
echo "Output:             $OUTPUT_DIR"
echo "Loss:               three-way cross-entropy, ties kept"
echo "LoRA:               rank $LORA_RANK, alpha $LORA_ALPHA, dropout 0.05, ViT/aligner frozen=$FREEZE_VISUAL"
echo "Learning rate:      $LEARNING_RATE, $LR_SCHEDULER, warmup $WARMUP_RATIO, data passes $DATA_PASSES"
echo "Batch:              $PAIR_BATCH_SIZE x $GRAD_ACCUMULATION_STEPS = $((PAIR_BATCH_SIZE * GRAD_ACCUMULATION_STEPS)) pairs"
echo "Attention:          $ATTN_IMPL, gradient checkpointing $GRADIENT_CHECKPOINTING"
echo "Tie Head:           $TIE_HEAD_MODE"
echo "Pair milestones:    ${OVERALL_INSPECTOR_SAVE_PAIR_MILESTONES:-none} (validation at milestones: $OVERALL_INSPECTOR_EVAL_AT_PAIR_MILESTONES)"
echo "Resume checkpoint:  ${RESUME_FROM_CHECKPOINT:-none}"
echo "======================================================================"

python scripts/train_overall_inspector.py \
    "${RESUME_ARGS[@]}" \
    --model "$MODEL_PATH" \
    --dataset "$INSPECTOR_DATA" \
    --task_type generative_reranker \
    --template qwen3_vl_reranker \
    --loss_type overall_inspector \
    --tuner_type lora \
    --tuner_backend peft \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit "$FREEZE_VISUAL" \
    --freeze_aligner "$FREEZE_VISUAL" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --attn_impl "$ATTN_IMPL" \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --add_version false \
    --per_device_train_batch_size "$PAIR_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_PAIR_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --packing false \
    --padding_free false \
    --sequence_parallel_size 1 \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --max_length 16384 \
    "${EVAL_ARGS[@]}" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 5 \
    --split_dataset_ratio "$SPLIT_RATIO_ARG" \
    --train_dataloader_shuffle false \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --dataloader_persistent_workers true \
    --dataloader_prefetch_factor 2 \
    --dataset_num_proc 8 \
    --load_from_cache_file false \
    --lazy_tokenize true \
    --strict true \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --seed 42 \
    --data_seed 42

echo "Overall Inspector MS-Swift training completed successfully."
echo "Consolidated checkpoint: $OUTPUT_DIR/overall_inspector_checkpoint.pt"
