#!/bin/bash
#SBATCH --job-name=omnigen_prepare
#SBATCH --output=logs/omnigen_prepare_%j.out
#SBATCH --error=logs/omnigen_prepare_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
# Submit from running: mkdir -p logs; sbatch prepare_pretrained.slurm
BUNDLE_DIR="${SLURM_SUBMIT_DIR:?Submit from running}"
test -f "$BUNDLE_DIR/settings.json" || { echo 'ERROR: cd to running before sbatch' >&2; exit 1; }
cd "$BUNDLE_DIR"

mapfile -t ENV_PATHS < <(python3 - <<'PY'
import json
from pathlib import Path

settings = json.loads(Path('settings.json').read_text(encoding='utf-8'))
for field in ('conda_sh', 'train_env'):
    value = settings[field]
    if not isinstance(value, str) or not value or '\n' in value or '\r' in value:
        raise ValueError(f'Invalid {field} in settings.json')
    print(value)
PY
)
test "${#ENV_PATHS[@]}" = 2 || { echo 'ERROR: failed to read Conda paths from settings.json' >&2; exit 1; }
source "${ENV_PATHS[0]}"
conda activate "${ENV_PATHS[1]}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi

"${ENV_PATHS[1]}/bin/python" -B - <<'PY'
from pathlib import Path
import fcntl
import json
import os
import sys
import torch

bundle = Path.cwd()
settings = json.loads((bundle / 'settings.json').read_text(encoding='utf-8'))
configured = os.environ.get('OMNIGEN_DIR') or settings['omnigen_transformer_bin']
if not configured:
    raise ValueError('Set omnigen_transformer_bin in settings.json or export OMNIGEN_DIR')
transformer_dir = (bundle / Path(configured).expanduser()).resolve()
if transformer_dir.name == 'pytorch_model.bin':
    transformer_dir = transformer_dir.parent
elif (transformer_dir / 'transformer').is_dir():
    transformer_dir = transformer_dir / 'transformer'
destination = transformer_dir / 'pytorch_model.bin'
temporary = transformer_dir / 'pytorch_model.bin.tmp'
if not (transformer_dir / 'config.json').is_file():
    raise FileNotFoundError(f'Missing downloaded transformer config: {transformer_dir / "config.json"}')

with (transformer_dir / 'conversion.lock').open('a') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if destination.is_file():
        existing = torch.load(destination, map_location='cpu', mmap=True, weights_only=True)
        if not isinstance(existing, dict) or not existing or not all(isinstance(value, torch.Tensor) for value in existing.values()):
            raise ValueError(f'Existing file is not a valid transformer state dict: {destination}')
        print(f'Already present: {destination}; tensors={len(existing)}; bytes={destination.stat().st_size:,}', flush=True)
        sys.exit(0)

    from prepare_run import resolve_repo
    _, rl = resolve_repo(bundle, settings)
    sys.path.insert(0, str(rl))
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    print(f'Loading local transformer: {transformer_dir}', flush=True)
    model, loading = OmniGen2Transformer2DModel.from_pretrained(
        str(transformer_dir), local_files_only=True,
        torch_dtype=torch.bfloat16, output_loading_info=True,
    )
    for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(field):
            raise RuntimeError(f'Checkpoint loading failed: {field}={loading[field]}')
    state = model.state_dict()
    if not state or any(value.is_meta for value in state.values()):
        raise ValueError('Transformer state is empty or contains uninitialized meta tensors')
    print(f'Saving BF16 state: tensors={len(state)} -> {destination}', flush=True)
    try:
        torch.save(state, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'OK: {destination}; bytes={destination.stat().st_size:,}', flush=True)
PY
