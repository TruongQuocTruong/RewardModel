#!/bin/bash
#SBATCH --job-name=omnigen2_qwen3
#SBATCH --output=logs/omnigen2_qwen3_%j.out
#SBATCH --error=logs/omnigen2_qwen3_%j.err
#SBATCH --account=uit
#SBATCH --partition=full
#SBATCH --qos=mig7-141
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:7g.141gb:1

set -euo pipefail
BUNDLE_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the running directory}"
test -f "$BUNDLE_DIR/run_job.py" || { echo 'ERROR: cd to running before sbatch' >&2; exit 1; }
mapfile -t ENV_PATHS < <(python3 - "$BUNDLE_DIR/settings.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as stream:
    config = json.load(stream)
for key in ('conda_sh', 'train_env', 'reward_env'):
    value = config[key]
    if not isinstance(value, str) or not value or '\n' in value or '\r' in value:
        raise ValueError(f'Invalid {key} in settings.json')
    print(value)
PY
)
test "${#ENV_PATHS[@]}" = 3 || { echo 'ERROR: failed to read Conda paths from settings.json' >&2; exit 1; }
source "${ENV_PATHS[0]}"
conda activate "${ENV_PATHS[1]}"
TRAIN_PY="${ENV_PATHS[1]}/bin/python"
REWARD_PY="${ENV_PATHS[2]}/bin/python"
test -x "$TRAIN_PY"
test -x "$REWARD_PY"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="$BUNDLE_DIR/outputs/hf_cache"
export HF_DATASETS_CACHE="$BUNDLE_DIR/outputs/datasets_cache"
export MPLCONFIGDIR="$BUNDLE_DIR/outputs/matplotlib"
unset RANK WORLD_SIZE LOCAL_RANK ACCELERATE_USE_FSDP ACCELERATE_USE_DEEPSPEED
mkdir -p "$BUNDLE_DIR/outputs" "$BUNDLE_DIR/logs" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
nvidia-smi
"$TRAIN_PY" -B -c 'from transformers import Qwen2_5_VLModel; import accelerate, peft, diffusers, yaml, timm; print("Policy dependencies OK", flush=True)'
"$REWARD_PY" -B -c 'from transformers import Qwen3VLForConditionalGeneration; import peft, swift, qwen_vl_utils; print("Reward dependencies OK", flush=True)'
exec "$TRAIN_PY" -B "$BUNDLE_DIR/run_job.py" --bundle "$BUNDLE_DIR" \
    --run-dir "$BUNDLE_DIR/outputs/job_${SLURM_JOB_ID}"
