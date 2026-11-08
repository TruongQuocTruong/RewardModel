#!/bin/bash
#SBATCH --job-name=build_augmentation_scenarios
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/build_augmentation_scenarios_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/build_augmentation_scenarios_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --qos=mig1-18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1g.18gb:1

# Student augmentation scenarios of the selected Unlabeled samples (text only, no image
# is opened; the GPU is not used): build_augmentation_scenarios.py on
# selected_unlabeled_<N>.json of score_teacher_inspector_agreement.py, written next to it.
# Runs as a Slurm job so that it can wait for the selection job (--dependency).

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
SELECTED="${SCENARIOS_SELECTED:?Set SCENARIOS_SELECTED to selected_unlabeled_<N>.json}"
MODE="${SCENARIOS_MODE:-combined}"
OUTPUT_DIR="$(dirname "$SELECTED")"
UNLABELED_DATASET="$BASE_DIR/TrainingRewardModel/TrainingDataCreation/editreward_unlabeled_10k.jsonl"

for required_file in "$SELECTED" "$UNLABELED_DATASET"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done

module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "STUDENT AUGMENTATION SCENARIOS"
echo "Selected:  $SELECTED"
echo "Mode:      $MODE"
echo "Output:    $OUTPUT_DIR"
echo "script md5: $(md5sum "$PROJECT_DIR/scripts/build_augmentation_scenarios.py" | cut -d' ' -f1)"
echo "======================================================================"

python "$PROJECT_DIR/scripts/build_augmentation_scenarios.py" \
    --selected "$SELECTED" \
    --unlabeled-dataset "$UNLABELED_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --mode "$MODE"

echo "Augmentation scenarios completed: $OUTPUT_DIR"
