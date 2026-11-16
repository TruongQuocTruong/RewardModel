#!/bin/bash
# Judge every rollout image (300 samples x 9 seeds) of ImgEdit with Qwen2.5-VL-72B-Instruct-AWQ; resumable (resubmit to continue).
#SBATCH --job-name=judge_qwen72b_awq_imgedit
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ComparisonRewardGuidedBenchs/logs/%x_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ComparisonRewardGuidedBenchs/logs/%x_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --qos=mig3-71
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:3g.71gb:1

set -euo pipefail
umask 002
PROJECT_DIR=/datastore/truongtq/TrainingRewardModel/ComparisonRewardGuidedBenchs
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/JudgeScores/Qwen2.5-VL-72B-Instruct-AWQ/ImgEdit}"
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl_awq
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami) started $(date '+%F %T')"
python -c 'import transformers, torch; print("transformers", transformers.__version__, "torch", torch.__version__)'
nvidia-smi --query-gpu=name,memory.total --format=csv || true
md5sum "$0" "$PROJECT_DIR"/scripts/{bench_protocols,judges,judge_bench}.py
cd "$PROJECT_DIR"
python -B scripts/judge_bench.py --bench ImgEdit --judge Qwen2.5-VL-72B-Instruct-AWQ \
    --model-path /datastore/shared/models/Qwen2.5-VL-72B-Instruct-AWQ --output-dir "$OUTPUT_DIR" "$@"
echo "finished $(date '+%F %T')"
