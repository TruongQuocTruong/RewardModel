#!/bin/bash
#SBATCH --job-name=diagnose_gradient_components
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/diagnose_gradient_components_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/logs/diagnose_gradient_components_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:2g.35gb:1

# Gradient of every Stage 2 loss component, replayed without training
# (scripts/diagnose_gradient_components.py). No weight is updated.
#
# DIAG_RUN_DIR          Stage 2 run folder (round_01.pt, screening_round_01.pt, round_01_summary.json)
# DIAG_STUDENT_WEIGHTS  stage1 (the run's Student-Stage1) or step_checkpoint
# DIAG_STEP_CHECKPOINT  checkpoint-N.pt of the run (for step_checkpoint)
# DIAG_STEPS            replay the data of steps 1..DIAG_STEPS
# DIAG_VERIFY_STEPS     steps with the sum-of-components check (one extra backward), default 1
# DIAG_REFERENCE_LOG    training log; step-1 losses must match it (Student-Stage1 only)
# DIAG_OUTPUT           output JSON (must not exist)

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
RUN_DIR="${DIAG_RUN_DIR:?Set DIAG_RUN_DIR}"
STUDENT_WEIGHTS="${DIAG_STUDENT_WEIGHTS:?Set DIAG_STUDENT_WEIGHTS to stage1 or step_checkpoint}"
STEPS="${DIAG_STEPS:?Set DIAG_STEPS}"
VERIFY_STEPS="${DIAG_VERIFY_STEPS:-1}"
OUTPUT="${DIAG_OUTPUT:?Set DIAG_OUTPUT}"

EXTRA_ARGS=()
case "$STUDENT_WEIGHTS" in
    stage1) ;;
    step_checkpoint) EXTRA_ARGS+=(--step-checkpoint "${DIAG_STEP_CHECKPOINT:?Set DIAG_STEP_CHECKPOINT}") ;;
    *) echo "ERROR: DIAG_STUDENT_WEIGHTS must be stage1 or step_checkpoint" >&2; exit 1 ;;
esac
[ -n "${DIAG_REFERENCE_LOG:-}" ] && EXTRA_ARGS+=(--reference-log "$DIAG_REFERENCE_LOG")
for required in "$RUN_DIR/round_01.pt" "$RUN_DIR/screening_round_01.pt" "$RUN_DIR/round_01_summary.json"; do
    [ -f "$required" ] || { echo "ERROR: missing $required" >&2; exit 1; }
done
[ -e "$OUTPUT" ] && { echo "ERROR: $OUTPUT exists" >&2; exit 1; }

module clear -f
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

ulimit -n 65537
cd "$PROJECT_DIR"

echo "======================================================================"
echo "GRADIENT COMPONENT DIAGNOSTIC (no training)"
echo "Run:              $RUN_DIR"
echo "Student weights:  $STUDENT_WEIGHTS ${DIAG_STEP_CHECKPOINT:-}"
echo "Steps replayed:   $STEPS (verify on first $VERIFY_STEPS)"
echo "Reference log:    ${DIAG_REFERENCE_LOG:-none}"
echo "Output:           $OUTPUT"
echo "Script md5:       $(md5sum scripts/diagnose_gradient_components.py | cut -d' ' -f1)"
echo "train_loop md5:   $(md5sum scripts/train_loop.py | cut -d' ' -f1)"
echo "======================================================================"

python scripts/diagnose_gradient_components.py \
    --run-dir "$RUN_DIR" \
    --student-weights "$STUDENT_WEIGHTS" \
    --steps "$STEPS" \
    --verify-total-steps "$VERIFY_STEPS" \
    --output "$OUTPUT" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
