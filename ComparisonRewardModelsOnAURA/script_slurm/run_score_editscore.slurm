#!/bin/bash
# Score AURA-Test with EditScore (scripts/score_editscore.py). Works from any account in Domain Users.
#
# Environment variables:
#   PRETRAINED        Qwen2.5-VL-7B | Qwen3-VL-4B | Qwen3-VL-8B           (required)
#   One job scores the WHOLE bench (1561 pairs) for one variant; do not split (job slots are limited).
#   PER_METRIC_LIMIT  smoke test: only the first N rows of each metric (default: all rows)
#   OUTPUT_DIR        default results/EditScore_<PRETRAINED>; smoke tests use results/EditScore_<PRETRAINED>_test_smoke
# Example (smoke test, 1 row per metric):
#   umask 000 && PRETRAINED=Qwen3-VL-4B PER_METRIC_LIMIT=1 OUTPUT_DIR=$ROOT/results/EditScore_Qwen3-VL-4B_test_smoke \
#     sbatch --time=02:00:00 script_slurm/run_score_editscore.slurm
#
#SBATCH --job-name=score_editscore
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_editscore_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_editscore_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
umask 000   # results must stay readable and writable by every account that shares the work

ROOT="/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA"
SCRIPT="$ROOT/scripts/score_editscore.py"
PRETRAINED="${PRETRAINED:?Set PRETRAINED to Qwen2.5-VL-7B, Qwen3-VL-4B or Qwen3-VL-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/EditScore_${PRETRAINED}}"

source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate omnigen2-rl   # peft 0.17.1 reads the EditScore LoRA configs; qwen_vl has peft 0.12 (corda_config error, jobs 12992-12994)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

ARGS=(--pretrained "$PRETRAINED" --output-dir "$OUTPUT_DIR")
if [ -n "${PER_METRIC_LIMIT:-}" ]; then ARGS+=(--per-metric-limit "$PER_METRIC_LIMIT"); fi

# One line per job in submissions.log (written at job start, from whichever account runs it).
printf '%s\taccount=%s\tjob=%s\tpartition=%s\tgres=%s\tmodel=EditScore\tpretrained=%s\tper_metric_limit=%s\toutput=%s\tlauncher_md5=%s\tscript_md5=%s\n' \
    "$(date '+%F %T')" "$(whoami)" "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-none}" "gpu:2g.35gb:1" \
    "$PRETRAINED" "${PER_METRIC_LIMIT:-all}" "$OUTPUT_DIR" \
    "$(md5sum "$ROOT/script_slurm/run_score_editscore.slurm" | cut -d' ' -f1)" "$(md5sum "$SCRIPT" | cut -d' ' -f1)" \
    >> "$ROOT/submissions.log"

echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami), started $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT/scripts"
status=0
python -u "$SCRIPT" "${ARGS[@]}" || status=$?
chmod -R a+rwX "$OUTPUT_DIR" 2>/dev/null || true
echo "finished $(date '+%F %T') with status $status"
exit $status
