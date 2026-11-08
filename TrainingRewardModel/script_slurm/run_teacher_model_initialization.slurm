#!/bin/bash
#SBATCH --job-name=teacher_init
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/teacher_init_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/teacher_init_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --qos=mig1-18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
STUDENT_CHECKPOINT="${TEACHER_STUDENT_CHECKPOINT:?Set TEACHER_STUDENT_CHECKPOINT to the Student checkpoint}"
TEACHER_CHECKPOINT="${TEACHER_OUTPUT:?Set TEACHER_OUTPUT to the confirmed output file}"

if [ ! -f "$STUDENT_CHECKPOINT" ]; then
    echo "ERROR: Student checkpoint not found: $STUDENT_CHECKPOINT" >&2
    exit 1
fi
if [ -e "$TEACHER_CHECKPOINT" ]; then
    echo "ERROR: Refusing to overwrite Teacher checkpoint: $TEACHER_CHECKPOINT" >&2
    exit 1
fi

module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"

echo "======================================================================"
echo "INITIALIZING EMA TEACHER FROM STUDENT WARM-START"
echo "Student checkpoint: $STUDENT_CHECKPOINT"
echo "Teacher checkpoint: $TEACHER_CHECKPOINT"
echo "Copy rule: exact independent tensor clone"
echo "======================================================================"

python scripts/initialize_teacher_model.py \
    --student-checkpoint "$STUDENT_CHECKPOINT" \
    --output "$TEACHER_CHECKPOINT"

echo "Teacher initialization completed successfully."
