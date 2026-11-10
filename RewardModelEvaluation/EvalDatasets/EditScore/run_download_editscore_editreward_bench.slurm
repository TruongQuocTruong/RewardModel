#!/bin/bash
#SBATCH --job-name=download_editscore_editreward_bench
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/%x_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/%x_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2g.35gb:1

# Downloads EditReward-Bench of the EditScore paper (about 5.5 GB) into
# EvalDatasets/EditScore/Raw and checks file sizes, rows and columns.
# No GPU is used; a GPU slice is requested only because the partition needs one.

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
EVAL_DATASETS_DIR="$BASE_DIR/RewardModelEvaluation/EvalDatasets/EditScore"
OUTPUT_DIR="$EVAL_DATASETS_DIR/Raw"

source /datastore/$USER/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

umask 002
echo "Host: $(hostname)  Start: $(date)"
python -I "$EVAL_DATASETS_DIR/download_editscore_editreward_bench.py" \
    --output-dir "$OUTPUT_DIR" \
    --max-workers 4
echo "End: $(date)"
du -sh "$OUTPUT_DIR"
