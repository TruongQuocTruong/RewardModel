#!/bin/bash
# Student-Stage1 (six-metric Student warm-start on majority-vote hard GemmaGT labels) config of the run
#   student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63
#   (job 8580, 2026-11-02)
# Runs without Slurm: one GPU (the run used one 35 GB MIG slice, peak 30.31 GiB, about 2 h 26 min).
#
# Usage:
#   bash config_student_stage1.sh [--dry-run]
#   --dry-run prints the environment variables and the python command (one argument per line) without training.
#
# What the run does:
#   - trains on the first 1008 pairs of the Inspector Baseline train_order.jsonl, labels from
#     editreward_overall_finegrained_gemmaGT_3k.jsonl (majority vote of the Gemma runs, no majority -> Tie);
#   - no validation; the last checkpoint is selected; 2 passes x 63 steps = 126 steps, batch 1 x 16 = 16 pairs;
#   - before training, the Inspector checkpoint-63 LLM LoRA is copied into the leading ranks of the Student LoRA
#     (ViT/aligner LoRA and metric heads keep the Student initialization);
#   - saves checkpoint-21, 42, ..., 126 and writes student_warmstart.pt (Student-Stage1 used by Stage 2).
#   The initial Teacher (teacher_initialized.pt in the same folder) is made by a separate job (8932), not by this config.
#
# Source: rendered from run_student_hardlabel_training.slurm (md5 2c5d2ada41b2d7129c3460051b84c509) with the settings
# printed in logs/student_hardlabel_gemma_8580.out and stored in <output>/args.json; current code of
# scripts/train_student_hardlabel.py and scripts/train_student_warmstart.py. Only paths are replaced by the variables below.
#
# Inputs:
#   inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32 (from the Drive zip, unzipped into CHECKPOINT_DIR):
#       checkpoint-63 (Inspector used for the initialization), train_order.jsonl (pair order)
#   Data (in this repo, TrainingDataCreation/): editreward_overall_finegrained_gemmaGT_3k.jsonl
#   Base model: Qwen3-VL-Reranker-8B (local folder).
#   Images: the paths stored inside the JSONL file must exist on your machine.
set -euo pipefail

# ---------------------------------------------------------------- paths (edit for your machine)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/Qwen3-VL-Embedding}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/TrainingDataCreation}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/output_training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output_training}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$SCRIPT_DIR/../Models/Qwen3-VL-Reranker-8B}"
PYTHON="${PYTHON:-python}"

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true
INSPECTOR_DIR="$CHECKPOINT_DIR/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32"
OUTPUT_DIR="$OUTPUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63"

# ---------------------------------------------------------------- environment read by the training scripts
export STUDENT_WARMSTART_SHARED_SPLIT=""
export STUDENT_WARMSTART_TRAIN_ORDER="$INSPECTOR_DIR/train_order.jsonl"
export STUDENT_WARMSTART_TRAIN_PAIRS=1008
export STUDENT_WARMSTART_VALIDATION_IDS=""
export STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT="$INSPECTOR_DIR/checkpoint-63"
export STUDENT_WARMSTART_FINALIZE_CHECKPOINT=""
export WARMSTART_RELEVANCE_WEIGHT=1.0
export TIE_HEAD_MODE=margin_coupled

ARGS=(
    --model "$BASE_MODEL_PATH"
    --dataset "$DATA_DIR/editreward_overall_finegrained_gemmaGT_3k.jsonl"
    --task_type generative_reranker
    --template qwen3_vl_reranker
    --loss_type factorized_metric_warmstart
    --target_modules all-linear
    --tuner_type lora
    --tuner_backend peft
    --freeze_llm false
    --freeze_vit false
    --freeze_aligner false
    --lora_rank 64
    --lora_alpha 128
    --learning_rate 4e-5
    --attn_impl sdpa
    --torch_dtype bfloat16
    --lora_dropout 0.05
    --weight_decay 0.01
    --load_from_cache_file false
    --metric_for_best_model loss
    --modules_to_save metric_heads
    --output_dir "$OUTPUT_DIR"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 16
    --packing false
    --padding_free false
    --sequence_parallel_size 1
    --num_train_epochs 2
    --max_length 16384
    --eval_strategy no
    --save_strategy steps
    --save_steps 21
    --eval_steps 50
    --save_total_limit -1
    --logging_steps 5
    --seed 42
    --data_seed 42
    --split_dataset_ratio 0
    --warmup_ratio 0.05
    --dataloader_num_workers 0
    --dataloader_persistent_workers false
    --dataset_num_proc 1
    --gradient_checkpointing true
    --add_version false
)

CMD=("$PYTHON" scripts/train_student_hardlabel.py "${ARGS[@]}")

if $DRY_RUN; then
    env | grep -E '^(STUDENT_WARMSTART_|WARMSTART_RELEVANCE_WEIGHT|TIE_HEAD_MODE)' | sort
    printf '%s\n' "${CMD[@]}"
    exit 0
fi

if [ ! -s "$STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT/adapter_model.safetensors" ] || [ ! -f "$STUDENT_WARMSTART_TRAIN_ORDER" ]; then
    echo "ERROR: Inspector checkpoint-63 or train_order.jsonl not found under $INSPECTOR_DIR" >&2
    exit 1
fi
if [ -e "$OUTPUT_DIR/student_warmstart.pt" ] || [ -e "$OUTPUT_DIR/last-checkpoint" ]; then
    echo "ERROR: refusing to overwrite existing Student-Stage1 output: $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Same environment as run_student_hardlabel_training.slurm (HF offline: the base model is a local folder).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ulimit -n 65537 2>/dev/null || true

cd "$PROJECT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/student_stage1_$(date +%Y%m%d_%H%M%S).log"
