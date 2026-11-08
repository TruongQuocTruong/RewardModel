#!/bin/bash
#SBATCH --job-name=eval_editreward_ourbench
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_editreward_ourbench_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_editreward_ourbench_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels"
PYTHON="/datastore/raccoon/miniconda3/envs/edit_reward/bin/python"

JOB_TMP="${SLURM_TMPDIR:-/tmp}/eval_editreward_ourbench_${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP/matplotlib" "$JOB_TMP/hf_modules"

export MPLCONFIGDIR="$JOB_TMP/matplotlib"
export HF_MODULES_CACHE="$JOB_TMP/hf_modules"
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() {
    rm -rf "$JOB_TMP"
}
trap cleanup EXIT

module clear -f || true

cd "$BASE_DIR"

echo "======================================================================"
echo "RUNNING EDITREWARD EVALUATION ON OURBENCH"
echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "Start time: $(date)"
echo "======================================================================"

"$PYTHON" \
    "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/EvalEditReward_OurBench.py" \
    --benchmark_path "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/EvalDatasets/OurBench.json" \
    --output_dir "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/evaluation_results/ourbench/editreward" \
    --batch_size 2

echo -e "\n======================================================================"
echo "EditReward evaluation finished successfully at $(date)!"
echo "======================================================================"