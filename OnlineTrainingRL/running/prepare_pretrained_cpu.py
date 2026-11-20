#!/usr/bin/env python3
"""Pack the local OmniGen2 Safetensors shards into a BF16 PyTorch state dict."""

import fcntl
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    bundle = Path(__file__).resolve().parent
    settings = json.loads((bundle / 'settings.json').read_text(encoding='utf-8'))
    folder = (bundle / Path(settings['omnigen_transformer_bin']).expanduser()).resolve()
    if folder.name == 'pytorch_model.bin':
        folder = folder.parent
    if (folder / 'transformer').is_dir():
        folder = folder / 'transformer'

    index = folder / 'diffusion_pytorch_model.safetensors.index.json'
    weights = json.loads(index.read_text(encoding='utf-8'))['weight_map']
    destination = folder / 'pytorch_model.bin'
    temporary = folder / 'pytorch_model.bin.tmp'
    with (folder / 'conversion.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.is_file():
            state = torch.load(destination, map_location='cpu', mmap=True, weights_only=True)
            if state.keys() != weights.keys():
                raise ValueError(f'Existing file has unexpected tensor keys: {destination}')
            print(f'Already present: {destination}', flush=True)
            return

        state = {}
        for shard in sorted(set(weights.values())):
            print(f'Reading {shard}', flush=True)
            with safe_open(folder / shard, framework='pt', device='cpu') as source:
                expected = {key for key, filename in weights.items() if filename == shard}
                if set(source.keys()) != expected:
                    raise ValueError(f'Tensor keys differ from index: {shard}')
                for key in source.keys():
                    tensor = source.get_tensor(key)
                    state[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

        print(f'Saving {len(state)} tensors to {destination}', flush=True)
        try:
            torch.save(state, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'OK: {destination} ({destination.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    main()
