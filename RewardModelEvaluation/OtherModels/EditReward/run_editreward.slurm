#!/bin/bash
#SBATCH --job-name=eval_editreward
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/EditReward/logs/eval_editreward_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/EditReward/logs/eval_editreward_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

# all | editreward_data_test | editreward_bench | editreward_bench_reannotation
BENCHMARK="${1:-all}"
BATCH_SIZE="${2:-2}"

ROOT="/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation"
SOURCE_ROOT="$ROOT/OtherModels/EditReward/EditReward"
MODEL_DIR="$SOURCE_ROOT/pretrained"
BENCHMARK_DIR="$ROOT/EvalDatasets"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/OtherModels/EditReward/evaluation_results}"

PYTHON="/datastore/truongtq/miniconda3/envs/edit_reward/bin/python"
SCRIPT="$ROOT/OtherModels/EditReward/EvalEditReward.py"

JOB_TMP="${SLURM_TMPDIR:-/tmp}/eval_editreward_${SLURM_JOB_ID:-manual}"

mkdir -p "$JOB_TMP/matplotlib" "$JOB_TMP/hf_modules"
mkdir -p "$OUTPUT_DIR"

export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export HF_MODULES_CACHE="$JOB_TMP/hf_modules"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$ROOT/OtherModels:$SOURCE_ROOT:${PYTHONPATH:-}"

cleanup() {
    rm -rf "$JOB_TMP"
}
trap cleanup EXIT

module clear -f

# Check required paths
for path in "$PYTHON" "$SCRIPT"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: File not found: $path"
        exit 1
    fi
done

for path in "$SOURCE_ROOT" "$MODEL_DIR" "$BENCHMARK_DIR"; do
    if [ ! -d "$path" ]; then
        echo "ERROR: Directory not found: $path"
        exit 1
    fi
done

cd "$(dirname "$SCRIPT")"

echo "======================================================================"
echo "RUNNING EDITREWARD INFERENCE + EVALUATION"
echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "Script: $SCRIPT"
echo "Model: $MODEL_DIR"
echo "Benchmark: $BENCHMARK"
echo "Batch size: $BATCH_SIZE"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "======================================================================"

"$PYTHON" "$SCRIPT" \
    --benchmark "$BENCHMARK" \
    --batch_size "$BATCH_SIZE" \
    --source_root "$SOURCE_ROOT" \
    --model_dir "$MODEL_DIR" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --output_dir "$OUTPUT_DIR" ${EXTRA_ARGS:-}

echo "======================================================================"
echo "EditReward inference and evaluation finished successfully!"
echo "Results: $OUTPUT_DIR"
echo "======================================================================"
