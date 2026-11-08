#!/bin/bash
#SBATCH --job-name=student_inspector_transfer
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/student_inspector_transfer_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/student_inspector_transfer_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

PAIR_BATCH_SIZE=1
EVAL_PAIR_BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS=8

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
# Only change vs. Student warm-start: Student LoRA starts from the Overall Inspector Model LoRA.
export STUDENT_INSPECTOR_TRANSFER_CHECKPOINT="${INSPECTOR_TRANSFER_CHECKPOINT:?Set INSPECTOR_TRANSFER_CHECKPOINT to the Inspector checkpoint directory}"
if [ ! -s "$STUDENT_INSPECTOR_TRANSFER_CHECKPOINT/adapter_model.safetensors" ]; then
    echo "ERROR: Overall Inspector Model adapter not found: $STUDENT_INSPECTOR_TRANSFER_CHECKPOINT" >&2
    exit 1
fi
# Optional split shared by every pipeline stage (scripts/backups/create_shared_source_split.py, no longer used).
export STUDENT_WARMSTART_SHARED_SPLIT="${WARMSTART_SHARED_SPLIT:-}"
if [ -n "$STUDENT_WARMSTART_SHARED_SPLIT" ] && [ ! -f "$STUDENT_WARMSTART_SHARED_SPLIT" ]; then
    echo "ERROR: Shared split not found: $STUDENT_WARMSTART_SHARED_SPLIT" >&2
    exit 1
fi

if [ -e "$OUTPUT_DIR/student_warmstart.pt" ]; then
    echo "ERROR: Student warm-start is already finalized; reuse the existing checkpoint: $OUTPUT_DIR/student_warmstart.pt" >&2
    exit 1
fi
if [ -e "$OUTPUT_DIR/last-checkpoint" ]; then
    echo "ERROR: A Student training checkpoint already exists. Refusing to restart warm-start and overwrite it." >&2
    exit 1
fi

# ----------------------------------------------------------------------
# 1. ENVIRONMENT
# ----------------------------------------------------------------------
module clear -f
# Cập nhật đường dẫn tới môi trường python/conda của bạn tại server mới
# Ví dụ nếu dùng venv:
# source /datastore/$USER/venv/bin/activate
# Hoặc nếu dùng conda (cần đảm bảo conda đã được cài đặt):
source /datastore/$USER/miniconda3/etc/profile.d/conda.sh
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
export MAX_PIXELS=200704
export MIN_PIXELS=4096
ulimit -n 65537

# ----------------------------------------------------------------------
# 3. FACTORIZED WARM-START LOSS CONFIGURATION
# ----------------------------------------------------------------------
export WARMSTART_RELEVANCE_WEIGHT=1.0

cd "$PROJECT_DIR"

echo "======================================================================"
echo "STARTING SIX-METRIC STUDENT WARM-START INITIALIZED FROM OVERALL INSPECTOR LORA"
echo "Inspector adapter: $STUDENT_INSPECTOR_TRANSFER_CHECKPOINT"
echo "GemmaGT Data: $GEMMA_GT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Shared split: ${STUDENT_WARMSTART_SHARED_SPLIT:-<none: own source-grouped split>}"
echo "Loss: Tie BCE + conditional-winner BCE + relevance BCE"
echo "Microbatch: $PAIR_BATCH_SIZE pairs ($((3 * PAIR_BATCH_SIZE)) A/B/source sequences)"
echo "Effective batch: $((PAIR_BATCH_SIZE * GRAD_ACCUMULATION_STEPS)) pairs per optimizer step"
echo "======================================================================"

python scripts/train_student_inspector_transfer.py \
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
    --lora_rank 128 \
    --lora_alpha 256 \
    --learning_rate 1e-5 \
    --attn_impl sdpa \
    --torch_dtype bfloat16 \
    --max_pixels 200704 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --load_from_cache_file false \
    --metric_for_best_model loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --modules_to_save metric_heads \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PAIR_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_PAIR_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --packing false \
    --padding_free false \
    --sequence_parallel_size 1 \
    --num_train_epochs 4 \
    --max_length 16384 \
    --eval_strategy steps \
    --save_strategy best \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 3 \
    --save_only_model false \
    --logging_steps 5 \
    --seed 42 \
    --data_seed 42 \
    --split_dataset_ratio 0.10 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataloader_persistent_workers false \
    --dataset_num_proc 1 \
    --gradient_checkpointing true \
    --add_version false

echo "Student Inspector-transfer warm-start completed successfully."
echo "Consolidated checkpoint: $OUTPUT_DIR/student_warmstart.pt"
