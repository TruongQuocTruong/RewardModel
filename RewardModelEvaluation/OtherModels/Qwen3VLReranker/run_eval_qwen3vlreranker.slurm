#!/bin/bash
#SBATCH --job-name=eval_qwen3vlreranker
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/Qwen3VLReranker/logs/eval_qwen3vlreranker_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/Qwen3VLReranker/logs/eval_qwen3vlreranker_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.35gb:1

# Original Qwen3-VL-Reranker-8B, official unmodified source code, zero-shot.
# Expected GPU memory: about 18-20 GB (bf16 weights about 16.5 GB, batch 1, images up to 1280 tokens).

set -euo pipefail

ROOT="/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation"
PYTHON="/datastore/truongtq/miniconda3/envs/qwen_vl/bin/python"
SCRIPT="$ROOT/OtherModels/Qwen3VLReranker/EvalQwen3VLReranker.py"
OUTPUT_DIR="$ROOT/OtherModels/evaluation_results/qwen3_vl_reranker_original"

JOB_TMP="${SLURM_TMPDIR:-/tmp}/eval_qwen3vlreranker_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP/matplotlib" "$JOB_TMP/hf_modules" "$OUTPUT_DIR"

export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export HF_MODULES_CACHE="$JOB_TMP/hf_modules"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi || true
"$PYTHON" "$SCRIPT" --output-dir "$OUTPUT_DIR"
