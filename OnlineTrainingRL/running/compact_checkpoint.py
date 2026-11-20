"""Single-process LoRA-only model checkpoints with Accelerate optimizer/RNG state.

Save/load pre-hooks replace only model serialization. Frozen pretrained weights
are deliberately absent and must be loaded again before restoring training.
"""
import hashlib
import json
import os
from pathlib import Path

FORMAT = 'omnigen2-lora-accelerate-v1'
MARKER = 'omnigen2_checkpoint.json'
COMPLETE = 'checkpoint_complete.json'


def _json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def export_lora(accelerator, model, directory):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    raw = accelerator.unwrap_model(model)
    state = get_peft_model_state_dict(raw, adapter_name='default')
    if not state or not all('lora_' in key for key in state):
        raise RuntimeError('Expected a nonempty LoRA-only state; refusing an incomplete export')
    state = {(key if key.startswith('base_model.model.') else 'base_model.model.' + key):
             value.detach().cpu().contiguous().clone() for key, value in state.items()}
    output = Path(directory) / 'lora_adapter'
    output.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output / 'adapter_model.safetensors'))
    raw.peft_config['default'].save_pretrained(str(output))
    return output


def _adapter_contract(model):
    config = model.peft_config['default']
    return {'r': int(config.r), 'lora_alpha': float(config.lora_alpha),
            'target_modules': sorted(config.target_modules),
            'bias': getattr(config, 'bias', 'none'),
            'use_rslora': bool(getattr(config, 'use_rslora', False)),
            'use_dora': bool(getattr(config, 'use_dora', False))}


def register_checkpoint_hooks(accelerator, model, *, base_info=None, scheduler=None, mode='lora'):
    """Call once after accelerator.prepare and before any save/load_state calls."""
    if mode not in ('lora', 'full'):
        raise ValueError('train.checkpoint_format must be lora or full')
    if accelerator.num_processes != 1:
        raise ValueError('These checkpoint hooks support exactly one training process')
    raw = accelerator.unwrap_model(model)
    trainable = sorted(name for name, value in raw.named_parameters() if value.requires_grad)
    if not trainable or any('lora_' not in name for name in trainable):
        raise ValueError('Compact checkpoint requires LoRA-only trainable parameters')
    accelerator._omnigen_checkpoint_format = mode
    if mode == 'full':
        return
    contract = _adapter_contract(raw)
    base_info = dict(base_info or {})

    def save_hook(models, weights, directory):
        if len(models) != 1 or len(weights) != 1:
            raise RuntimeError('Expected exactly one model/state in the single-GPU checkpoint')
        adapter = export_lora(accelerator, models[0], directory)
        marker = {'format': FORMAT, 'adapter_contract': contract,
                  'trainable_parameter_names': trainable, 'base_info': base_info,
                  'adapter_config_sha256': hashlib.sha256((adapter / 'adapter_config.json').read_bytes()).hexdigest()}
        if scheduler is not None:
            import torch
            torch.save(scheduler.state_dict(), Path(directory) / 'lora_scheduler.pt')
        _json(Path(directory) / MARKER, marker)
        # Accelerate will still save optimizer/scaler/RNG, but no model.safetensors.
        weights.clear()

    def load_hook(models, directory):
        import torch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        directory = Path(directory)
        marker_path = directory / MARKER
        if not marker_path.exists():
            # Legacy full-state checkpoints use Accelerate's normal model loader.
            return
        if not (directory / COMPLETE).is_file():
            raise ValueError(f'Checkpoint is incomplete: {directory}')
        marker = json.loads(marker_path.read_text())
        if marker.get('format') != FORMAT or marker.get('adapter_contract') != contract:
            raise ValueError('Compact checkpoint LoRA architecture does not match this model')
        if marker.get('trainable_parameter_names') != trainable:
            raise ValueError('Trainable parameter ordering/names differ from the checkpoint')
        if marker.get('base_info') != base_info:
            raise ValueError('Compact checkpoint requires the same frozen pretrained paths')
        if len(models) != 1:
            raise RuntimeError('Expected one model when restoring compact checkpoint')
        adapter = directory / 'lora_adapter'
        config_hash = hashlib.sha256((adapter / 'adapter_config.json').read_bytes()).hexdigest()
        if config_hash != marker['adapter_config_sha256']:
            raise ValueError('Saved adapter configuration has changed')
        portable = load_file(str(adapter / 'adapter_model.safetensors'), device='cpu')
        state = {key.removeprefix('base_model.model.'): value for key, value in portable.items()}
        if not state or any('lora_' not in key for key in state):
            raise ValueError('Invalid LoRA-only checkpoint')
        current = accelerator.unwrap_model(models[0])
        result = set_peft_model_state_dict(current, state, adapter_name='default')
        if result.unexpected_keys:
            raise RuntimeError(f'Unexpected adapter keys: {result.unexpected_keys[:10]}')
        restored = get_peft_model_state_dict(current, adapter_name='default')
        if set(restored) != set(state):
            raise RuntimeError('Checkpoint adapter key set does not match the current model')
        for key, value in state.items():
            actual = restored[key].detach().cpu()
            if actual.shape != value.shape or not torch.equal(actual, value.to(dtype=actual.dtype)):
                raise RuntimeError(f'LoRA tensor was not restored: {key}')
        if scheduler is not None:
            scheduler.load_state_dict(torch.load(directory / 'lora_scheduler.pt', map_location='cpu', weights_only=True))
        # Models are loaded above; leave optimizer/RNG loading to Accelerate.
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def save_checkpoint(accelerator, model, path):
    """Publish only fully saved checkpoints; retain failed staging for inspection."""
    destination = Path(path)
    staging = destination.with_name('.' + destination.name + '.incomplete')
    if destination.exists() or staging.exists():
        raise FileExistsError(f'Checkpoint destination/staging already exists: {destination}')
    accelerator.save_state(str(staging))
    mode = getattr(accelerator, '_omnigen_checkpoint_format', 'full')
    if mode == 'full':
        export_lora(accelerator, model, staging)
    if mode == 'lora' and any(staging.glob('model*.safetensors')):
        raise RuntimeError('Compact save unexpectedly wrote a full model')
    # Catch deferred filesystem/quota errors before publishing a checkpoint.
    for artifact in staging.rglob('*'):
        if artifact.is_file():
            if artifact.stat().st_size == 0:
                raise OSError(f'Checkpoint artifact is empty: {artifact}')
            with artifact.open('r+b') as stream:
                os.fsync(stream.fileno())
    _json(staging / COMPLETE, {'format': FORMAT if mode == 'lora' else 'full', 'checkpoint': destination.name})
    os.replace(staging, destination)
    total = sum(item.stat().st_size for item in destination.rglob('*') if item.is_file())
    print(f'Saved checkpoint format={mode}; size={total / 1024**2:.1f} MiB; path={destination}', flush=True)
