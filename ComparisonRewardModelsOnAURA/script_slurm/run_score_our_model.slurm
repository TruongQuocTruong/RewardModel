#!/bin/bash
# Score AURA-Test with our Student checkpoints (scripts/score_our_model.py).
#
# Environment variables:
#   VARIANT           student_stage1 | student_stage2 | teacher_stage2         (required)
#                     (Inspector Baseline is not evaluated on AURA-Test: overall-only model)
#                     student_stage1     = Student-Stage1 student_warmstart.pt (= checkpoint-126, 1008 pairs)
#                     student_stage2     = Student Model of Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5/checkpoint-927.pt
#                     teacher_stage2     = Teacher Model of Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5/checkpoint-927.pt
#   One job scores the WHOLE bench (1561 pairs) for one variant; the bench is never split.
#   PER_METRIC_LIMIT  smoke test: only the first N rows of each metric (default: all rows)
#   OUTPUT_DIR        default results/<Student-Stage1|Student-Stage2|Teacher-Stage2>_Qwen3-VL-Reranker-8B
# Conda env: qwen_vl (the env of training and of RewardModelEvaluation/OurModel).
# VRAM: Student evaluation at batch 1 needs about 20-30 GiB -> normal (2g.35gb).
# Example:
#   umask 000 && VARIANT=student_stage1 sbatch script_slurm/run_score_our_model.slurm
#
#SBATCH --job-name=score_our_model
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_our_model_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/ComparisonRewardModelsOnAURA/logs/score_our_model_%j.err
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
SCRIPT="$ROOT/scripts/score_our_model.py"
VARIANT="${VARIANT:?Set VARIANT to student_stage1, student_stage2 or teacher_stage2}"
case "$VARIANT" in
    student_stage1)     RESULT_FOLDER=Student-Stage1_Qwen3-VL-Reranker-8B; MODEL_ROW="Student-Stage1" ;;
    student_stage2)     RESULT_FOLDER=Student-Stage2_Qwen3-VL-Reranker-8B; MODEL_ROW="Student-Stage2" ;;
    teacher_stage2)     RESULT_FOLDER=Teacher-Stage2_Qwen3-VL-Reranker-8B; MODEL_ROW="Teacher-Stage2" ;;
    *) echo "ERROR: VARIANT must be student_stage1, student_stage2 or teacher_stage2, got $VARIANT" >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/$RESULT_FOLDER}"
CONDA_ENV=qwen_vl
# Actual GPU type of the job (the #SBATCH default can be overridden on the sbatch command line):
# the typed entry gres/gpu:<type>[=count or :count] of scontrol, e.g. "2g.35gb"; never the untyped "gres/gpu=1".
GRES="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'gres/gpu:[^ ,=]*' | head -1 | sed 's#^gres/gpu:##; s#:[0-9]*$##' || true)"

source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

ARGS=(--variant "$VARIANT" --output-dir "$OUTPUT_DIR")
if [ -n "${PER_METRIC_LIMIT:-}" ]; then ARGS+=(--per-metric-limit "$PER_METRIC_LIMIT"); fi

# One line per job in submissions.log (written at job start, from whichever account runs it).
printf '%s\taccount=%s\tjob=%s\tpartition=%s\tgres=%s\tmodel=%s\tvariant=%s\tconda_env=%s\tper_metric_limit=%s\toutput=%s\tlauncher_md5=%s\tscript_md5=%s\n' \
    "$(date '+%F %T')" "$(whoami)" "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-none}" "${GRES:-unknown}" \
    "$MODEL_ROW" "$VARIANT" "$CONDA_ENV" "${PER_METRIC_LIMIT:-all}" "$OUTPUT_DIR" \
    "$(md5sum "$ROOT/script_slurm/run_score_our_model.slurm" | cut -d' ' -f1)" "$(md5sum "$SCRIPT" | cut -d' ' -f1)" \
    >> "$ROOT/submissions.log"

echo "job ${SLURM_JOB_ID:-none} on $(hostname) as $(whoami), env $CONDA_ENV, variant $VARIANT, started $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT/scripts"
status=0
python -u "$SCRIPT" "${ARGS[@]}" || status=$?
chmod -R a+rwX "$OUTPUT_DIR" 2>/dev/null || true
echo "finished $(date '+%F %T') with status $status"
exit $status
