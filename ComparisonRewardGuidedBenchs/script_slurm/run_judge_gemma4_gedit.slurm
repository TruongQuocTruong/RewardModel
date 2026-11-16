#!/bin/bash
# Judge every rollout image (300 samples x 9 seeds) of GEdit-Bench-EN with gemma-4-31B-it; resumable (resubmit to continue).
#SBATCH --job-name=judge_gemma4_gedit
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
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/JudgeScores/gemma-4-31B-it/GEdit-Bench-EN}"
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate gemma4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami) started $(date '+%F %T')"
python -c 'import transformers, torch; print("transformers", transformers.__version__, "torch", torch.__version__)'
nvidia-smi --query-gpu=name,memory.total --format=csv || true
md5sum "$0" "$PROJECT_DIR"/scripts/{bench_protocols,judges,judge_bench}.py
cd "$PROJECT_DIR"
python -B scripts/judge_bench.py --bench GEdit-Bench-EN --judge gemma-4-31B-it \
    --model-path /datastore/shared/models/gemma-4-31B-it --output-dir "$OUTPUT_DIR" "$@"
echo "finished $(date '+%F %T')"
