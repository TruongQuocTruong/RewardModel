#!/bin/bash
#SBATCH --job-name=eval_reward_model
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/%x_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/logs/%x_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

BASE_DIR="/datastore/truongtq/TrainingRewardModel"
PROJECT_DIR="$BASE_DIR/TrainingRewardModel/Qwen3-VL-Embedding"
EVAL_DIR="$BASE_DIR/RewardModelEvaluation/OurModel"
RESULT_DIR="$EVAL_DIR/evaluation_results"
PREFERENCE_AGGREGATOR="${EVAL_PREFERENCE_AGGREGATOR:-$PROJECT_DIR/output/student_warmstart/preference_aggregator.json}"
EDITHF_TEST="$BASE_DIR/RewardModelEvaluation/EvalDatasets/EditHF-Test_pairwise_200.json"
EDITREWARD_BENCH="$BASE_DIR/RewardModelEvaluation/EvalDatasets/EditReward-Bench_pairwise.json"
ROLE="${EVAL_ROLE:?EVAL_ROLE must be inspector, student, or teacher}"
BASE_MODEL_PATH="${EVAL_BASE_MODEL_PATH:-$BASE_DIR/Models/Qwen3-VL-Reranker-8B}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

case "$ROLE" in
    inspector)
        CHECKPOINT="$PROJECT_DIR/output/overall_inspector/overall_inspector_checkpoint.pt"
        OUTPUT="$RESULT_DIR/overall_inspector_evaluation.json"
        ;;
    student)
        CHECKPOINT="$PROJECT_DIR/output/training_loop/best.pt"
        OUTPUT="$RESULT_DIR/student_model_evaluation.json"
        ;;
    teacher)
        CHECKPOINT="$PROJECT_DIR/output/training_loop/best.pt"
        OUTPUT="$RESULT_DIR/teacher_model_evaluation.json"
        ;;
    *)
        echo "ERROR: Unsupported EVAL_ROLE: $ROLE" >&2
        exit 1
        ;;
esac

CHECKPOINT="${EVAL_CHECKPOINT:-$CHECKPOINT}"
OUTPUT="${EVAL_OUTPUT:-$OUTPUT}"

for required_file in "$CHECKPOINT" "$PREFERENCE_AGGREGATOR" "$EDITHF_TEST" "$EDITREWARD_BENCH"; do
    if [ ! -f "$required_file" ]; then
        echo "ERROR: Required input not found: $required_file" >&2
        exit 1
    fi
done
if [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: Confirmed result directory not found: $RESULT_DIR" >&2
    exit 1
fi
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: Base model not found: $BASE_MODEL_PATH" >&2
    exit 1
fi
if [ -e "$OUTPUT" ]; then
    echo "ERROR: Refusing to overwrite existing evaluation output: $OUTPUT" >&2
    exit 1
fi

module clear -f
source /datastore/$USER/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl

export HF_HOME="$BASE_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export MAX_PIXELS=200704
export MIN_PIXELS=4096

ulimit -n 65537
cd "$EVAL_DIR"

echo "======================================================================"
echo "REWARD MODEL EVALUATION"
echo "Role:             $ROLE"
echo "Checkpoint:       $CHECKPOINT"
echo "Base model:       $BASE_MODEL_PATH"
echo "EditHF-Test:      $EDITHF_TEST"
echo "EditReward-Bench: $EDITREWARD_BENCH"
echo "Output:           $OUTPUT"
echo "======================================================================"

python evaluate_reward_model.py \
    --role "$ROLE" \
    --checkpoint "$CHECKPOINT" \
    --preference-aggregator "$PREFERENCE_AGGREGATOR" \
    --edithf-test "$EDITHF_TEST" \
    --editreward-bench "$EDITREWARD_BENCH" \
    --output "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --base-model-path "$BASE_MODEL_PATH" \
    --min-pixels 4096 \
    --max-pixels 200704 \
    --device cuda

echo "Evaluation completed: $OUTPUT"
