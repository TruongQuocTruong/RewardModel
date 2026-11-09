#!/bin/bash
# Inspector Baseline (three-way Overall Inspector, full EditReward-Data, one pass) config of the run
#   inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32   (job 6955, finished 2026-10-30)
# Stage 1 of the pipeline: checkpoint-63 (= first 1008 pairs) is the Inspector used by Student-Stage1 and Stage 2.
# Runs without Slurm: one GPU with about 120 GB (the run used one 141 GB MIG slice, peak 116.49 GiB, 1 day 2 h).
#
# Usage:
#   bash config_inspector_baseline.sh [--dry-run]
#   --dry-run prints the environment variables and the python command (one argument per line) without training.
#
# What the run does:
#   - trains first on the pairs of editreward_overall_finegrained_gemmaGT_3k.jsonl (train prefix, in its file order),
#     then on the rest of metadata.jsonl; 3% of the pairs outside the prefix are the validation split;
#   - batch 2 x 8 = 16 pairs per optimizer step, 10359 steps;
#   - saves checkpoints at the pair milestones below (rounded up to multiples of 16 pairs), e.g. 1008 pairs -> checkpoint-63,
#     and scores the validation split at each milestone;
#   - writes train_order.jsonl (order of trained pairs; Student-Stage1 reads its first 1008 pairs) and
#     overall_inspector_checkpoint.pt (last step).
#
# Source: rendered from run_overall_inspector_training.slurm (md5 7d8a99b4c8ca937b68d266bf3d1905eb) with the settings
# printed in logs/overall_inspector_6955.out and stored in <output>/args.json; train_overall_inspector.py md5
# 4d54def8ef8a2edbc2487a97b7c43d70 (current code; job 12080 reran the same settings with it). Only paths are replaced
# by the variables below.
#
# Inputs:
#   EditReward-Data (metadata.jsonl and its images) under EDITREWARD_ROOT.
#   Data (in this repo, TrainingDataCreation/): editreward_overall_finegrained_gemmaGT_3k.jsonl (train prefix).
#   Base model: Qwen3-VL-Reranker-8B (local folder).
#   Python environment with flash attention (the run used conda env qwen_vl_flash).
set -euo pipefail

# ---------------------------------------------------------------- paths (edit for your machine)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/Qwen3-VL-Embedding}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/TrainingDataCreation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output_training}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$SCRIPT_DIR/../Models/Qwen3-VL-Reranker-8B}"
EDITREWARD_ROOT="${EDITREWARD_ROOT:-/datastore/shared/datasets/EditReward-Data}"
PYTHON="${PYTHON:-python}"

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true
OUTPUT_DIR="$OUTPUT_ROOT/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32"

# ---------------------------------------------------------------- environment read by train_overall_inspector.py
export OVERALL_INSPECTOR_SPLIT=""
export OVERALL_INSPECTOR_TRAIN_PREFIX="$DATA_DIR/editreward_overall_finegrained_gemmaGT_3k.jsonl"
export OVERALL_INSPECTOR_DATA_PASSES=1
export OVERALL_INSPECTOR_SAVE_PAIR_MILESTONES="50,100,200,300,500,1000,2000,3000,5000,8000,10000,15000,30000,50000,70000,100000,130000,150000"
export OVERALL_INSPECTOR_EVAL_AT_PAIR_MILESTONES=1
export INSPECTOR_EDITREWARD_ROOT="$EDITREWARD_ROOT"
export MIN_PIXELS=4096
export TIE_HEAD_MODE=margin_coupled

ARGS=(
    --model "$BASE_MODEL_PATH"
    --dataset "$EDITREWARD_ROOT/extracted_data/metadata.jsonl"
    --task_type generative_reranker
    --template qwen3_vl_reranker
    --loss_type overall_inspector
    --tuner_type lora
    --tuner_backend peft
    --target_modules all-linear
    --freeze_llm false
    --freeze_vit true
    --freeze_aligner true
    --lora_rank 32
    --lora_alpha 64
    --lora_dropout 0.05
    --learning_rate 5e-5
    --lr_scheduler_type constant
    --weight_decay 0.01
    --max_grad_norm 1.0
    --attn_impl flash_attn
    --torch_dtype bfloat16
    --output_dir "$OUTPUT_DIR"
    --add_version false
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 8
    --packing false
    --padding_free false
    --sequence_parallel_size 1
    --num_train_epochs 1
    --max_steps -1
    --max_length 16384
    --eval_strategy steps
    --eval_steps 100000
    --metric_for_best_model loss
    --save_strategy steps
    --save_steps 100000
    --logging_steps 5
    --split_dataset_ratio 0.03
    --train_dataloader_shuffle false
    --warmup_ratio 0
    --dataloader_num_workers 8
    --dataloader_persistent_workers true
    --dataloader_prefetch_factor 2
    --dataset_num_proc 8
    --load_from_cache_file false
    --lazy_tokenize true
    --strict true
    --gradient_checkpointing false
    --seed 42
    --data_seed 42
)
# Every checkpoint is kept (INSPECTOR_SAVE_TOTAL_LIMIT=-1 in the launcher, so --save_total_limit is not passed).

CMD=("$PYTHON" scripts/train_overall_inspector.py "${ARGS[@]}")

if $DRY_RUN; then
    env | grep -E '^(OVERALL_INSPECTOR_|INSPECTOR_EDITREWARD_ROOT|MIN_PIXELS|TIE_HEAD_MODE)' | sort
    printf '%s\n' "${CMD[@]}"
    exit 0
fi

if [ -e "$OUTPUT_DIR/overall_inspector_checkpoint.pt" ]; then
    echo "ERROR: refusing to overwrite existing final checkpoint: $OUTPUT_DIR/overall_inspector_checkpoint.pt" >&2
    exit 1
fi
if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
    echo "ERROR: checkpoints already exist in $OUTPUT_DIR; refusing to restart over them." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Same environment as run_overall_inspector_training.slurm (HF offline: the base model is a local folder).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
ulimit -n 65537 2>/dev/null || true

cd "$PROJECT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/inspector_training_$(date +%Y%m%d_%H%M%S).log"
