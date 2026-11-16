#!/bin/bash
# Score the 9 rollouts of every sample (3 benches x 300) with one selector, then write its Best-of-N selections.
#   SELECTOR="FIRE-Reward-Stage2-Teacher" sbatch script_slurm/run_score_selector.slurm          (normal, about 30 GiB)
#   SELECTOR="CLIP" sbatch script_slurm/run_score_selector.slurm; SELECTOR="ImageReward" CONDA_ENV=image_reward sbatch script_slurm/run_score_selector.slurm  (all on normal)
#   SELECTOR="FIRE-Reward-Stage2-Teacher" RUN_NAME=<new name> sbatch script_slurm/run_score_selector.slurm --model-path <checkpoint.pt>
#   SELECTOR="EditScore" CONDA_ENV=omnigen2-rl sbatch --partition=full --qos=mig7-141 --gres=gpu:7g.141gb:1 --time=3-00:00:00 script_slurm/run_score_selector.slurm   (Qwen3-VL-8B LoRA, up to ~16.8 Mpx per image -> full; only normal or full are used)
#   SELECTOR="EditReward" CONDA_ENV=edit_reward sbatch script_slurm/run_score_selector.slurm  (Qwen2.5-VL-7B)
# Optional: CONDA_ENV (default qwen_vl), RUN_NAME (output folder, default = SELECTOR),
#           FIRE_SCORE_RULE (aesthetic_fidelity | selected_equal), extra args passed to score_selector.py
#SBATCH --job-name=score_selector
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ComparisonRewardGuidedBenchs/logs/%x_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ComparisonRewardGuidedBenchs/logs/%x_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
umask 002
SELECTOR="${SELECTOR:?Set SELECTOR=Qwen2.5-VL-7B|ImageReward|CLIP|FIRE-Reward-Stage2-Teacher|EditScore|EditReward}"
RUN_NAME="${RUN_NAME:-$SELECTOR}"
PROJECT_DIR=/datastore/truongtq/TrainingRewardModel/ComparisonRewardGuidedBenchs
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-qwen_vl}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTORCH_ALLOC_CONF=expandable_segments:True
echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami); selector $SELECTOR; run name $RUN_NAME; started $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd "$PROJECT_DIR"
md5sum scripts/{bench_protocols,best_of_n_selectors,score_selector,select_best,judge_bench}.py
python -B scripts/score_selector.py --selector "$SELECTOR" --run-name "$RUN_NAME" "$@"
python -B scripts/select_best.py --selector "$SELECTOR" --run-name "$RUN_NAME" --fire-score-rule "${FIRE_SCORE_RULE:-aesthetic_fidelity}"
python -B scripts/select_best.py --selector FLUX.2-Klein-9B
echo "finished $(date '+%F %T')"
