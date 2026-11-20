#!/bin/bash
set -euo pipefail
MODE="${1:?Usage: bash run_inference.sh base|checkpoint [extra Python arguments]}"
shift
INFERENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$MODE" in
    base) SCRIPT=generate_omnigen2_gedit.py ;;
    checkpoint) SCRIPT=generate_omnigen2_gedit_fireward.py ;;
    *) echo 'ERROR: mode must be base or checkpoint' >&2; exit 1 ;;
esac
mapfile -t ENV_PATHS < <(python3 - "$INFERENCE_DIR/settings.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as stream:
    settings = json.load(stream)
for key in ('conda_sh', 'train_env'):
    value = settings[key]
    if not isinstance(value, str) or not value or '\n' in value or '\r' in value:
        raise ValueError(f'Invalid {key} in settings.json')
    print(value)
PY
)
test "${#ENV_PATHS[@]}" = 2 || { echo 'ERROR: failed to read environment paths from settings.json' >&2; exit 1; }
source "${ENV_PATHS[0]}"
conda activate "${ENV_PATHS[1]}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$INFERENCE_DIR/cache/huggingface"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$INFERENCE_DIR/logs" "$INFERENCE_DIR/cache"
cd "$INFERENCE_DIR"
nvidia-smi
exec python -B "$INFERENCE_DIR/$SCRIPT" --settings "$INFERENCE_DIR/settings.json" "$@"
