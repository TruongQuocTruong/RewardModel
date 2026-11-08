#!/bin/bash
#SBATCH --job-name=inspector_inference
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/inspector_inference_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/inspector_inference_%j.err
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
INSPECTOR_DIR="$PROJECT_DIR/output_training/overall_inspector_r32_llm_lr8e-5"
CHECKPOINT="${INSPECTOR_CHECKPOINT:-$INSPECTOR_DIR/overall_inspector_checkpoint.pt}"
UNLABELED_DATA="$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_unlabeled_10k.jsonl"
DATASET_ROOT="/datastore/shared/datasets/EditReward-Data"
OUTPUT="${INSPECTOR_CACHE_OUTPUT:-$INSPECTOR_DIR/editreward_inspector_pseudo_labels_10k.jsonl}"

# Consolidated .pt file or MS-Swift checkpoint-<step> folder.
if [ ! -f "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT/adapter_model.safetensors" ]; then
    echo "ERROR: Inspector checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$UNLABELED_DATA" ]; then
    echo "ERROR: Unlabeled Data not found: $UNLABELED_DATA" >&2
    exit 1
fi
if [ -e "$OUTPUT" ]; then
    echo "ERROR: Refusing to overwrite existing Inspector cache: $OUTPUT" >&2
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

echo "======================================================================"
echo "RUNNING FROZEN OVERALL INSPECTOR OFFLINE INFERENCE"
echo "Checkpoint: $CHECKPOINT"
echo "Unlabeled Data: $UNLABELED_DATA"
echo "Output:     $OUTPUT"
echo "======================================================================"

python scripts/infer_overall_inspector.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$UNLABELED_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --output "$OUTPUT" \
    --batch-size 1 \
    --temperature 1.0 \
    --device cuda

echo "Overall Inspector inference completed successfully."
