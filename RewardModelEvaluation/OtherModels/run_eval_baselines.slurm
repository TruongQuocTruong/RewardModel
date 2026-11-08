#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ModelEvaluation/logs/eval_baselines_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ModelEvaluation/logs/eval_baselines_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

MODEL="${1:-all}"          # options: edithf | editreward | all
BENCHMARK="${2:-all}"      # options: all | editreward_bench | edithf_test

BASE_DIR="/datastore/truongtq/TrainingRewardModel/ModelEvaluation/Baselines"
PYTHON="/datastore/raccoon/miniconda3/envs/edit_reward/bin/python"
SOURCE_ROOT="/datastore/raccoon/truongtq/ImageEditingBenchmark/TRAINING_REWARD_MODEL/ModelEvaluation/EditReward"

JOB_TMP="${SLURM_TMPDIR:-/tmp}/eval_baselines_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP/matplotlib" "$JOB_TMP/hf_modules"
export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export HF_MODULES_CACHE="$JOB_TMP/hf_modules"
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$SOURCE_ROOT:${PYTHONPATH:-}"

cleanup() {
    rm -rf "$JOB_TMP"
}
trap cleanup EXIT

module clear -f
cd "$BASE_DIR"

echo "======================================================================"
echo "RUNNING BASELINE EVALUATION: Model=$MODEL, Benchmark=$BENCHMARK"
echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "======================================================================"

if [ "$MODEL" = "edithf" ] || [ "$MODEL" = "all" ]; then
    echo -e "\n>>> Starting EditHF Evaluation..."
    "$PYTHON" EvalEditHF.py --benchmark "$BENCHMARK"
fi

if [ "$MODEL" = "editreward" ] || [ "$MODEL" = "all" ]; then
    echo -e "\n>>> Starting EditReward Evaluation..."
    "$PYTHON" EvalEditReward.py --benchmark "$BENCHMARK"
fi

echo -e "\n======================================================================"
echo "Evaluation finished successfully!"
echo "======================================================================"
