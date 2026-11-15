#!/bin/bash
# Score AURA-Test with EditHF / InternVL3.5-8B (scripts/score_edithf.py). Works from any account in Domain Users.
#
# Environment variables:
#   One job scores the WHOLE bench (1561 pairs) with the three LoRA sets loaded one after another; never split.
#   PER_METRIC_LIMIT  smoke test: only the first N rows of each metric (default: all rows)
#   OUTPUT_DIR        default results/EditHF_InternVL3.5-8B
# Conda env (installed under truongtq, shared read-only): edit_reward.
# Example:
#   umask 000 && sbatch script_slurm/run_score_edithf.slurm
#
#SBATCH --job-name=score_edithf
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_edithf_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_edithf_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
umask 000   # results must stay readable and writable by every account that shares the work

ROOT="/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA"
SCRIPT="$ROOT/scripts/score_edithf.py"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/EditHF_InternVL3.5-8B}"
CONDA_ENV=edit_reward

source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

ARGS=(--output-dir "$OUTPUT_DIR")
if [ -n "${PER_METRIC_LIMIT:-}" ]; then ARGS+=(--per-metric-limit "$PER_METRIC_LIMIT"); fi

# One line per job in submissions.log (written at job start, from whichever account runs it).
printf '%s\taccount=%s\tjob=%s\tpartition=%s\tgres=%s\tmodel=EditHF\tpretrained=%s\tconda_env=%s\tper_metric_limit=%s\toutput=%s\tlauncher_md5=%s\tscript_md5=%s\n' \
    "$(date '+%F %T')" "$(whoami)" "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-none}" "gpu:2g.35gb:1" \
    "InternVL3.5-8B" "$CONDA_ENV" "${PER_METRIC_LIMIT:-all}" "$OUTPUT_DIR" \
    "$(md5sum "$ROOT/script_slurm/run_score_edithf.slurm" | cut -d' ' -f1)" "$(md5sum "$SCRIPT" | cut -d' ' -f1)" \
    >> "$ROOT/submissions.log"

echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami), env $CONDA_ENV, started $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT/scripts"
status=0
python -u "$SCRIPT" "${ARGS[@]}" || status=$?
chmod -R a+rwX "$OUTPUT_DIR" 2>/dev/null || true
echo "finished $(date '+%F %T') with status $status"
exit $status
