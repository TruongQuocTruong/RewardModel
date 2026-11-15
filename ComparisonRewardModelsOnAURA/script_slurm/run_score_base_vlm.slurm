#!/bin/bash
# Score AURA-Test with a base VLM judge (scripts/score_base_vlm.py). Works from any account in Domain Users.
#
# Environment variables:
#   MODEL             Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3.5-9B                 (required)
#   One job scores the WHOLE bench (1561 pairs) for one model; the bench is never split.
#   PER_METRIC_LIMIT  smoke test: only the first N rows of each metric (default: all rows)
#   OUTPUT_DIR        default results/<MODEL>
# Conda env (installed under truongtq, shared read-only): qwen_vl; Qwen3.5-9B uses gemma4 (transformers 5.x).
# Example:
#   umask 000 && MODEL=Qwen3-VL-8B sbatch script_slurm/run_score_base_vlm.slurm
#
#SBATCH --job-name=score_base_vlm
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_base_vlm_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_base_vlm_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
umask 000   # results must stay readable and writable by every account that shares the work

ROOT="/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA"
SCRIPT="$ROOT/scripts/score_base_vlm.py"
MODEL="${MODEL:?Set MODEL to Qwen2.5-VL-7B, Qwen3-VL-8B or Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/${MODEL}}"
case "$MODEL" in
    Qwen2.5-VL-7B|Qwen3-VL-8B) CONDA_ENV=qwen_vl ;;
    Qwen3.5-9B) CONDA_ENV=gemma4 ;;
    *) echo "unknown MODEL $MODEL" >&2; exit 2 ;;
esac

source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

ARGS=(--model "$MODEL" --output-dir "$OUTPUT_DIR")
if [ -n "${PER_METRIC_LIMIT:-}" ]; then ARGS+=(--per-metric-limit "$PER_METRIC_LIMIT"); fi

# One line per job in submissions.log (written at job start, from whichever account runs it).
printf '%s\taccount=%s\tjob=%s\tpartition=%s\tgres=%s\tmodel=BaseVLM\tbackbone=%s\tconda_env=%s\tper_metric_limit=%s\toutput=%s\tlauncher_md5=%s\tscript_md5=%s\n' \
    "$(date '+%F %T')" "$(whoami)" "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-none}" "gpu:2g.35gb:1" \
    "$MODEL" "$CONDA_ENV" "${PER_METRIC_LIMIT:-all}" "$OUTPUT_DIR" \
    "$(md5sum "$ROOT/script_slurm/run_score_base_vlm.slurm" | cut -d' ' -f1)" "$(md5sum "$SCRIPT" | cut -d' ' -f1)" \
    >> "$ROOT/submissions.log"

echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami), env $CONDA_ENV, started $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT/scripts"
status=0
python -u "$SCRIPT" "${ARGS[@]}" || status=$?
chmod -R a+rwX "$OUTPUT_DIR" 2>/dev/null || true
echo "finished $(date '+%F %T') with status $status"
exit $status
