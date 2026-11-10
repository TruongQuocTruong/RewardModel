#!/bin/bash
#SBATCH --job-name=convert_editscore_editreward_bench
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/%x_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/%x_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2g.35gb:1

# Converts the overall rows of EditReward-Bench (EditScore paper) in EvalDatasets/EditScore/Raw
# to our pairwise JSON format; images go to EvalDatasets/EditScore/Extracted/images.
# No GPU is used; a GPU slice is requested only because the partition needs one.

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
EVAL_DATASETS_DIR="$BASE_DIR/RewardModelEvaluation/EvalDatasets/EditScore"

source /datastore/$USER/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

umask 002
echo "Host: $(hostname)  Start: $(date)"
python -I "$EVAL_DATASETS_DIR/convert_editscore_editreward_bench.py" \
    --raw-dir "$EVAL_DATASETS_DIR/Raw" \
    --extracted-dir "$EVAL_DATASETS_DIR/Extracted" \
    --output-prefix "$EVAL_DATASETS_DIR/EditScore-EditReward-Bench" \
    --dimensions overall
echo "End: $(date)"
du -sh "$EVAL_DATASETS_DIR/Extracted"
