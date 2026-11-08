#!/bin/bash
#SBATCH --job-name=student_hardlabel_gemma
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/student_hardlabel_gemma_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/student_hardlabel_gemma_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:3g.71gb:1

set -euo pipefail

PAIR_BATCH_SIZE=1
EVAL_PAIR_BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS="${WARMSTART_GRAD_ACCUMULATION_STEPS:-8}"

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"

MODEL_PATH="${STUDENT_MODEL_PATH:-/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Reranker-8B}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Shared base model not found on $(hostname): $MODEL_PATH" >&2
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

GEMMA_GT_DATA="${WARMSTART_GEMMA_GT_DATA:-$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_overall_finegrained_gemmaGT_3k.jsonl}"
OUTPUT_DIR="${WARMSTART_OUTPUT_DIR:?Set WARMSTART_OUTPUT_DIR to the confirmed output directory}"
LORA_RANK="${WARMSTART_LORA_RANK:-64}"
LORA_ALPHA=$((2 * LORA_RANK))
LEARNING_RATE="${WARMSTART_LEARNING_RATE:-4e-5}"
NUM_TRAIN_EPOCHS="${WARMSTART_NUM_TRAIN_EPOCHS:-2}"
VALIDATION_RATIO="${WARMSTART_VALIDATION_RATIO:-0.15}"
EVAL_STEPS="${WARMSTART_EVAL_STEPS:-50}"
SAVE_STEPS="${WARMSTART_SAVE_STEPS:-$EVAL_STEPS}"
SAVE_TOTAL_LIMIT="${WARMSTART_SAVE_TOTAL_LIMIT:-5}"
# Only change vs. Student warm-start: training targets are majority-vote hard GemmaGT labels
# (validation keeps the soft three-run targets).
# Optional split shared by every pipeline stage (scripts/backups/create_shared_source_split.py, no longer used).
export STUDENT_WARMSTART_SHARED_SPLIT="${WARMSTART_SHARED_SPLIT:-}"
if [ -n "$STUDENT_WARMSTART_SHARED_SPLIT" ] && [ ! -f "$STUDENT_WARMSTART_SHARED_SPLIT" ]; then
    echo "ERROR: Shared split not found: $STUDENT_WARMSTART_SHARED_SPLIT" >&2
    exit 1
fi
# Optional train-order split: train on the first WARMSTART_TRAIN_PAIRS pairs of an Inspector
# train_order.jsonl. Validation pairs come only from WARMSTART_VALIDATION_IDS (JSONL with
# sample_id); without it there is no validation and the last checkpoint is selected.
export STUDENT_WARMSTART_TRAIN_ORDER="${WARMSTART_TRAIN_ORDER:-}"
export STUDENT_WARMSTART_TRAIN_PAIRS="${WARMSTART_TRAIN_PAIRS:-}"
export STUDENT_WARMSTART_VALIDATION_IDS="${WARMSTART_VALIDATION_IDS:-}"
EVAL_STRATEGY=steps
SPLIT_DESCRIPTION="${STUDENT_WARMSTART_SHARED_SPLIT:-<random pair split, ratio $VALIDATION_RATIO>}"
if [ -n "$STUDENT_WARMSTART_TRAIN_ORDER" ]; then
    if [ ! -f "$STUDENT_WARMSTART_TRAIN_ORDER" ] || [ -z "$STUDENT_WARMSTART_TRAIN_PAIRS" ]; then
        echo "ERROR: WARMSTART_TRAIN_ORDER must exist and WARMSTART_TRAIN_PAIRS must be set." >&2
        exit 1
    fi
    if [ -n "$STUDENT_WARMSTART_VALIDATION_IDS" ]; then
        if [ ! -f "$STUDENT_WARMSTART_VALIDATION_IDS" ]; then
            echo "ERROR: Validation pairs file not found: $STUDENT_WARMSTART_VALIDATION_IDS" >&2
            exit 1
        fi
        # MS-Swift only enables validation when split_dataset_ratio > 0; the ratio itself is unused.
        SPLIT_DESCRIPTION="first $STUDENT_WARMSTART_TRAIN_PAIRS pairs of $STUDENT_WARMSTART_TRAIN_ORDER; validation: $STUDENT_WARMSTART_VALIDATION_IDS"
    else
        VALIDATION_RATIO=0
        EVAL_STRATEGY=no
        SPLIT_DESCRIPTION="first $STUDENT_WARMSTART_TRAIN_PAIRS pairs of $STUDENT_WARMSTART_TRAIN_ORDER; validation: none (last checkpoint selected)"
    fi
fi

# Optional Inspector initialization: copy the Overall Inspector LLM LoRA (MS-Swift checkpoint
# directory) into the leading ranks of the Student LoRA before training; ViT/aligner LoRA and
# metric_heads keep the Student initialization.
export STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT="${WARMSTART_INIT_INSPECTOR_CHECKPOINT:-}"
if [ -n "$STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT" ] && [ ! -s "$STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT/adapter_model.safetensors" ]; then
    echo "ERROR: Inspector checkpoint for initialization not found: $STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT" >&2
    exit 1
fi

# Optional finalize-only mode: rebuild student_warmstart.pt from an existing checkpoint
# (no optimization). It replaces the current student_warmstart.pt.
export STUDENT_WARMSTART_FINALIZE_CHECKPOINT="${WARMSTART_FINALIZE_CHECKPOINT:-}"
if [ -n "$STUDENT_WARMSTART_FINALIZE_CHECKPOINT" ]; then
    if [ ! -s "$STUDENT_WARMSTART_FINALIZE_CHECKPOINT/adapter_model.safetensors" ]; then
        echo "ERROR: Checkpoint to finalize not found: $STUDENT_WARMSTART_FINALIZE_CHECKPOINT" >&2
        exit 1
    fi
    echo "Finalize-only: student_warmstart.pt will be rebuilt from $STUDENT_WARMSTART_FINALIZE_CHECKPOINT"
elif [ -e "$OUTPUT_DIR/student_warmstart.pt" ]; then
    echo "ERROR: Student warm-start is already finalized; reuse the existing checkpoint: $OUTPUT_DIR/student_warmstart.pt" >&2
    exit 1
fi
if [ -z "$STUDENT_WARMSTART_FINALIZE_CHECKPOINT" ] && [ -e "$OUTPUT_DIR/last-checkpoint" ]; then
    echo "ERROR: A Student training checkpoint already exists. Refusing to restart warm-start and overwrite it." >&2
    exit 1
fi

# ----------------------------------------------------------------------
# 1. ENVIRONMENT
# ----------------------------------------------------------------------
module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

mkdir -p "$BASE_DIR/TrainingRewardModel/logs"

# ----------------------------------------------------------------------
# 2. GPU & ENVIRONMENT SETUP
# ----------------------------------------------------------------------
# Bỏ qua phần check GPU thủ công vì Slurm đã tự động cấp phát MIG instance và gán CUDA_VISIBLE_DEVICES

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MASTER_PORT=29995
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ulimit -n 65537

# ----------------------------------------------------------------------
# 3. FACTORIZED WARM-START LOSS CONFIGURATION
# ----------------------------------------------------------------------
export WARMSTART_RELEVANCE_WEIGHT=1.0
# Tie Head chosen 26/10/2026; set TIE_HEAD_MODE=independent explicitly for the old head.
export TIE_HEAD_MODE="${TIE_HEAD_MODE:-margin_coupled}"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "STARTING SIX-METRIC STUDENT WARM-START WITH MAJORITY-VOTE HARD GEMMAGT LABELS"
echo "Train targets: majority vote (no majority -> Tie); validation targets: soft"
echo "GemmaGT Data: $GEMMA_GT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Split: $SPLIT_DESCRIPTION"
echo "LoRA rank/alpha: $LORA_RANK/$LORA_ALPHA (LLM + ViT + aligner); learning rate: $LEARNING_RATE; epochs: $NUM_TRAIN_EPOCHS"
echo "Loss: Tie BCE + conditional-winner BCE + relevance BCE"
echo "Microbatch: $PAIR_BATCH_SIZE pairs ($((3 * PAIR_BATCH_SIZE)) A/B/source sequences)"
echo "Effective batch: $((PAIR_BATCH_SIZE * GRAD_ACCUMULATION_STEPS)) pairs per optimizer step"
echo "Eval / save steps: $EVAL_STEPS / $SAVE_STEPS (eval strategy: $EVAL_STRATEGY, kept checkpoints: $SAVE_TOTAL_LIMIT)"
echo "Tie Head: $TIE_HEAD_MODE"
echo "Student initialization: ${STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT:-base model with fresh LoRA}"
echo "======================================================================"

python scripts/train_student_hardlabel.py \
    --model "$MODEL_PATH" \
    --dataset "$GEMMA_GT_DATA" \
    --task_type generative_reranker \
    --template qwen3_vl_reranker \
    --loss_type factorized_metric_warmstart \
    --target_modules all-linear \
    --tuner_type lora \
    --tuner_backend peft \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --learning_rate "$LEARNING_RATE" \
    --attn_impl sdpa \
    --torch_dtype bfloat16 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --load_from_cache_file false \
    --metric_for_best_model loss \
    --modules_to_save metric_heads \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PAIR_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_PAIR_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --packing false \
    --padding_free false \
    --sequence_parallel_size 1 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_length 16384 \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps 5 \
    --seed 42 \
    --data_seed 42 \
    --split_dataset_ratio "$VALIDATION_RATIO" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataloader_persistent_workers false \
    --dataset_num_proc 1 \
    --gradient_checkpointing true \
    --add_version false

echo "Student hard-label warm-start completed successfully."
echo "Consolidated checkpoint: $OUTPUT_DIR/student_warmstart.pt"
