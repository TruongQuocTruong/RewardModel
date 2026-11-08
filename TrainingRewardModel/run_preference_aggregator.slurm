#!/bin/bash
#SBATCH --job-name=preference_aggregator
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/preference_aggregator_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/preference_aggregator_%j.err
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
STUDENT_CHECKPOINT="${AGGREGATOR_STUDENT_CHECKPOINT:?Set AGGREGATOR_STUDENT_CHECKPOINT to the Student checkpoint}"
HUMAN_GT_DATA="$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_overall_humanGT_3k.jsonl"
OUTPUT="${AGGREGATOR_OUTPUT:?Set AGGREGATOR_OUTPUT to the confirmed output file}"
# Optional split shared by every pipeline stage (scripts/backups/create_shared_source_split.py, no longer used).
SHARED_SPLIT="${AGGREGATOR_SHARED_SPLIT:-}"
if [ -n "$SHARED_SPLIT" ] && [ ! -f "$SHARED_SPLIT" ]; then
    echo "ERROR: Shared split not found: $SHARED_SPLIT" >&2
    exit 1
fi

FEATURE_DEFINITION="${PREFERENCE_AGGREGATOR_FEATURE_DEFINITION:-signed_evidence}"
BOOTSTRAP_REPLICATES="${PREFERENCE_AGGREGATOR_BOOTSTRAP_REPLICATES:-100}"
L2_PENALTY="${PREFERENCE_AGGREGATOR_L2_PENALTY:-0.01}"

for required_file in "$STUDENT_CHECKPOINT" "$HUMAN_GT_DATA"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required file not found: $required_file" >&2
        exit 1
    fi
done
if [ -e "$OUTPUT" ]; then
    echo "ERROR: Refusing to overwrite Preference Aggregator output: $OUTPUT" >&2
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
echo "FITTING THE SIX-METRIC PREFERENCE AGGREGATOR"
echo "Student checkpoint:  $STUDENT_CHECKPOINT"
echo "HumanGT Data:        $HUMAN_GT_DATA"
echo "Output:              $OUTPUT"
echo "Feature definition:  $FEATURE_DEFINITION"
echo "Bootstrap runs:      $BOOTSTRAP_REPLICATES"
echo "L2 penalty:          $L2_PENALTY"
echo "Shared split:        ${SHARED_SPLIT:-<none: own source-grouped split>}"
echo "======================================================================"

python scripts/fit_preference_aggregator.py \
    --student-checkpoint "$STUDENT_CHECKPOINT" \
    --dataset "$HUMAN_GT_DATA" \
    --output "$OUTPUT" \
    --feature-definition "$FEATURE_DEFINITION" \
    --batch-size 1 \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --l2-penalty "$L2_PENALTY" \
    --validation-ratio 0.20 \
    ${SHARED_SPLIT:+--shared-split "$SHARED_SPLIT"} \
    --max-iterations 1000 \
    --seed 42 \
    --device cuda

echo "Preference Aggregator fitting completed successfully."
