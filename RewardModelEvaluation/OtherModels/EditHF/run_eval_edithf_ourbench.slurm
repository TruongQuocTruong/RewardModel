#!/bin/bash
#SBATCH --job-name=eval_edithf_ourbench
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_edithf_ourbench_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/eval_edithf_ourbench_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels"
PYTHON="/datastore/raccoon/miniconda3/envs/edit_reward/bin/python"

JOB_TMP="${SLURM_TMPDIR:-/tmp}/eval_edithf_ourbench_${SLURM_JOB_ID:-$$}"
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
echo "RUNNING EDITHF EVALUATION ON OURBENCH"
echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "Start time: $(date)"
echo "======================================================================"

"$PYTHON" EvalEditHF_OurBench.py \
    --benchmark_path "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/EvalDatasets/OurBench.json" \
    --output_dir "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/evaluation_results/ourbench" \
    --device "cuda"

echo -e "\n======================================================================"
echo "Evaluation finished successfully at $(date)!"
echo "======================================================================"
