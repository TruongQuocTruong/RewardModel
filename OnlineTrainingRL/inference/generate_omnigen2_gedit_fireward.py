#!/usr/bin/env python3
"""Generate GEdit edited images with an OmniGen2 Fireward post-training checkpoint.

Uses this repo's OmniGen2 classes and separate local transformer/text encoder/VAE.
Pretrained paths are explicit. A PEFT export needs no training YAML; a full
Accelerate state requires its original LoRA rank/alpha config.
Output: OmniGen2-Fireward/GEdit-Bench-EN/<task_type>/<source_filename>.png.
Loads an explicit LoRA/full training checkpoint. Only edited PNGs are saved to output_root.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

from inference_config import configure_parser, load_settings, resolve_repo

ROOT = Path(__file__).resolve().parent
BENCH = Path('/datastore/anhndt/Truongtq/OtherEditingBenchmark')


def prepare_source(image, max_long_edge):
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image)
    if image.mode == 'P' and 'transparency' in image.info:
        image = image.convert('RGBA')
    if image.mode in ('RGBA', 'LA'):
        foreground = image.convert('RGBA')
        background = Image.new('RGBA', foreground.size, 'white')
        image = Image.alpha_composite(background, foreground)
    image = image.convert('RGB')
    scale = min(1.0, max_long_edge / max(image.size))
    content = tuple(max(1, round(value * scale)) for value in image.size)
    if image.size != content:
        image = image.resize(content, Image.Resampling.LANCZOS)
    width, height = content
    padded = (math.ceil(width / 16) * 16, math.ceil(height / 16) * 16)
    canvas = Image.new('RGB', padded)
    canvas.paste(image, (0, 0))
    if padded[0] > width:
        canvas.paste(image.crop((width - 1, 0, width, height)).resize((padded[0] - width, height)), (width, 0))
    if padded[1] > height:
        canvas.paste(canvas.crop((0, height - 1, padded[0], height)).resize((padded[0], padded[1] - height)), (0, height))
    return canvas, content


def valid_png(path, size):
    from PIL import Image
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            valid = image.format == 'PNG' and image.size == tuple(size)
            image.verify()
        with Image.open(path) as image:
            image.load()
        return valid
    except (OSError, ValueError):
        return False


def read_pairs(metadata, data_root, output_root):
    rows, ids, targets = [], {}, {}
    with metadata.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ('key', 'task_type', 'image', 'instruction'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'{metadata}:{number}: missing {field}')
            for value in (row['key'], row['task_type']):
                if value in ('.', '..') or any(char in value for char in '/\\:\x00'):
                    raise ValueError(f'Unsafe sample identity: {value!r}')
            source = Path(row['image'])
            source = (source if source.is_absolute() else data_root / source).resolve()
            identity = (str(source), row['instruction'])
            sample = (row['task_type'], row['key'])
            if sample in ids:
                if ids[sample] != identity:
                    raise ValueError(f'Conflicting metadata: {sample}')
                continue
            ids[sample] = identity
            if not source.is_file() or not source.stem:
                raise FileNotFoundError(source)
            target = output_root / row['task_type'] / (source.stem + '.png')
            if target in targets:
                raise ValueError(f'Two distinct samples map to the same edited image: {target}')
            targets[target] = sample
            rows.append({**row, 'source': source, 'target': target})
    if not rows:
        raise ValueError('No GEdit samples')
    return rows


def pretrained_paths(settings_file=None, *, transformer=None, text_encoder=None, vae_model=None,
                     pretrained_root=None):
    settings = load_settings(settings_file) if settings_file else {}
    root = Path(pretrained_root).expanduser().resolve() if pretrained_root else None
    paths = {}
    for field, explicit, folder in (('text_encoder', text_encoder, 'Qwen2.5-VL-3B-Instruct'),
                                     ('vae_model', vae_model, 'FLUX.1-dev')):
        value = explicit or (root / folder if root else None) or settings.get(field)
        if not value:
            raise ValueError(f'Pass --{field.replace("_", "-")} or --pretrained-root; optional --settings is also supported')
        paths[field] = Path(value).expanduser().resolve()
    candidate = transformer or (root / 'OmniGen2' if root else None) or settings.get('omnigen_transformer_bin')
    if not candidate:
        raise ValueError('Pass --transformer-path or --pretrained-root')
    candidate = Path(candidate).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f'Transformer path does not exist: {candidate}')
    if candidate.is_file():
        if candidate.suffix != '.bin':
            raise ValueError('Use a pytorch_model.bin file or a local Diffusers transformer directory')
        paths['transformer_bin'] = candidate
        paths['transformer_dir'] = candidate.parent
    else:
        directory = candidate / 'transformer' if (candidate / 'transformer').is_dir() else candidate
        paths['transformer_dir'] = directory
        paths['transformer_bin'] = directory / 'pytorch_model.bin' if (directory / 'pytorch_model.bin').is_file() else None
    if paths['transformer_bin'] and not paths['transformer_bin'].stat().st_size:
        raise ValueError(f'Transformer .bin is empty: {paths["transformer_bin"]}')
    if not (paths['transformer_dir'] / 'config.json').is_file():
        raise FileNotFoundError(f'Transformer config.json must be beside its weights: {paths["transformer_dir"]}')
    for path in (paths['text_encoder'] / 'config.json', paths['vae_model'] / 'vae/config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def resolve_checkpoint(path, lora_config=None):
    """Prefer this trainer's portable LoRA export; never guess the latest step."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Fireward checkpoint missing: {path}')
    if path.is_dir():
        for adapter in (path / 'lora_adapter', path):
            config = adapter / 'adapter_config.json'
            if not config.is_file():
                continue
            for name in ('adapter_model.safetensors', 'adapter_model.bin'):
                weight = adapter / name
                if weight.is_file() and weight.stat().st_size:
                    return {'kind': 'peft', 'path': str(weight), 'config': str(config),
                            'size': weight.stat().st_size, 'mtime_ns': weight.stat().st_mtime_ns,
                            'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}
            raise FileNotFoundError(f'Adapter config exists but its weights are missing/empty: {adapter}')
        candidates = [path / name for name in ('model.safetensors', 'pytorch_model.bin')
                      if (path / name).is_file()]
        if len(candidates) != 1:
            raise ValueError(f'Choose an exact checkpoint-N directory with lora_adapter/ or model.safetensors: {path}')
        path = candidates[0]
    if not path.stat().st_size:
        raise ValueError(f'Checkpoint file is empty: {path}')
    if path.name not in ('model.safetensors', 'pytorch_model.bin'):
        raise ValueError('For a PEFT checkpoint pass its directory; full state accepts model.safetensors/pytorch_model.bin')
    if lora_config:
        config = Path(lora_config).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
    else:
        candidates = [path.parent / 'adapter_config.json', path.parent.parent / 'train.yml',
                      path.parent.parent.parent / 'train.yml']
        config = next((candidate for candidate in candidates if candidate.is_file()), None)
        if config is None:
            raise FileNotFoundError('Full state needs its original training YAML/adapter config to recover rank/alpha. Pass --lora-config.')
    return {'kind': 'full', 'path': str(path), 'config': str(config),
            'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
            'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}


def canonical_lora_state(state):
    """Accept exported PEFT prefixes and older direct-Diffusers LoRA prefixes."""
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint state is empty or is not a state dictionary')
    result = {}
    for key, value in state.items():
        if not isinstance(key, str) or 'lora_' not in key:
            raise ValueError(f'Expected LoRA-only adapter state, received {key!r}')
        if key.startswith('module.'):
            key = key[len('module.'):]
        if not key.startswith('base_model.model.'):
            key = 'base_model.model.' + key
        key = key.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')
        if key in result:
            raise ValueError(f'Duplicate normalized LoRA tensor: {key}')
        result[key] = value
    return result


def apply_checkpoint(transformer, checkpoint):
    """Restore all trained tensors, then explicitly verify the LoRA state."""
    import torch
    from peft import LoraConfig, PeftConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    weight = Path(checkpoint['path'])
    if weight.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(weight), device='cpu')
    else:
        state = torch.load(weight, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not any('lora_' in key for key in state):
        raise ValueError('Checkpoint has no trained LoRA tensors; refusing pretrained-only fallback')
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint has non-tensor or nonfinite weights')
    if checkpoint['kind'] == 'peft':
        config = PeftConfig.from_pretrained(str(Path(checkpoint['config']).parent), local_files_only=True)
        config.inference_mode = True
        transformer = get_peft_model(transformer, config)
        expected = canonical_lora_state(state)
        result = set_peft_model_state_dict(transformer, expected, adapter_name='default')
        if result.unexpected_keys:
            raise RuntimeError(f'Unexpected LoRA keys: {result.unexpected_keys[:10]}')
        actual = get_peft_model_state_dict(transformer, adapter_name='default')
    else:
        config_path = Path(checkpoint['config'])
        if config_path.suffix == '.json':
            config = LoraConfig.from_pretrained(str(config_path.parent), local_files_only=True)
        else:
            import yaml
            train = yaml.safe_load(config_path.read_text())['train']
            if not train.get('lora_ft'):
                raise ValueError('Training config does not describe LoRA fine-tuning')
            config = LoraConfig(r=int(train['lora_rank']), lora_alpha=int(train['lora_alpha']),
                                lora_dropout=float(train.get('lora_dropout', 0)),
                                target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'], inference_mode=True)
        transformer.add_adapter(config)
        direct = {key[len('module.'):] if key.startswith('module.') else key: value for key, value in state.items()}
        transformer.load_state_dict(direct, strict=True)
        expected = canonical_lora_state({key: value for key, value in direct.items() if 'lora_' in key})
        actual = canonical_lora_state(get_peft_model_state_dict(transformer, adapter_name='default'))
    if set(actual) != set(expected):
        raise RuntimeError(f'LoRA keys differ: missing={sorted(set(expected)-set(actual))[:5]}, extra={sorted(set(actual)-set(expected))[:5]}')
    for key, value in expected.items():
        restored = actual[key].detach().cpu()
        if restored.shape != value.shape or not torch.equal(restored, value.to(dtype=restored.dtype)):
            raise RuntimeError(f'LoRA tensor was not restored correctly: {key}')
    if checkpoint['kind'] == 'peft':
        # Keep the injected/loaded LoRA layers, but give the pipeline the original
        # OmniGen2 module. Its predict() inspects forward's explicit parameters;
        # a generic PeftModel.forward(*args, **kwargs) makes it drop source latents.
        # This does NOT merge, disable or unload the adapter.
        transformer = transformer.get_base_model()
    if 'ref_image_hidden_states' not in inspect.signature(transformer.forward).parameters:
        raise RuntimeError('OmniGen2 forward must expose ref_image_hidden_states for the editing pipeline')
    transformer.eval().requires_grad_(False)
    print(f'Loaded Fireward checkpoint: {weight}; format={checkpoint["kind"]}; verified LoRA tensors={len(expected)}', flush=True)
    return transformer


def load_pipeline(args):
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL
    from transformers import AutoProcessor, Qwen2_5_VLModel
    sys.path.insert(0, str(args.repo_rl))
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    paths = args.pretrained
    weight_path = paths['transformer_bin']
    if weight_path and weight_path.is_file():
        config = json.loads((paths['transformer_dir'] / 'config.json').read_text())
        with init_empty_weights():
            transformer = OmniGen2Transformer2DModel.from_config(config)
        state = torch.load(weight_path, map_location='cpu', mmap=True, weights_only=True)
        transformer.load_state_dict(state, assign=True, strict=True)
        del state
    else:
        transformer, loading = OmniGen2Transformer2DModel.from_pretrained(
            str(paths['transformer_dir']), torch_dtype=torch.bfloat16,
            local_files_only=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError(f'Transformer loading mismatch: {loading}')
    transformer = apply_checkpoint(transformer, args.checkpoint_info)
    processor = AutoProcessor.from_pretrained(str(paths['text_encoder']), local_files_only=True)
    processor.tokenizer.padding_side = 'right'
    text_encoder = Qwen2_5_VLModel.from_pretrained(
        str(paths['text_encoder']), torch_dtype=torch.bfloat16, local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(str(paths['vae_model']), subfolder='vae',
                                       torch_dtype=torch.bfloat16, local_files_only=True)
    for component in (transformer, text_encoder, vae):
        component.eval().requires_grad_(False)
    pipe = OmniGen2Pipeline(
        transformer=transformer, vae=vae, mllm=text_encoder, processor=processor,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=args.steps,
                                                   time_shift_base_res=args.time_shift_base_res),
    ).to(dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
    else:
        pipe.to('cuda:0')
    pipe.set_progress_bar_config(disable=True)
    # Same empty negative-prompt template as EditScore's GEdit evaluation launcher.
    negative = processor.tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': ''},
    ], tokenize=False, add_generation_prompt=False)
    with torch.inference_mode():
        negative_embeds, negative_mask = pipe._get_qwen2_prompt_embeds(
            prompt=negative, device=torch.device('cuda:0'), max_sequence_length=1024,
        )
    return pipe, negative_embeds, negative_mask


@contextmanager
def output_lock(args):
    import fcntl
    key = hashlib.sha256(str(args.output_root).encode()).hexdigest()[:20]
    directory = ROOT / 'state'
    directory.mkdir(parents=True, exist_ok=True)
    signature = {
        'output_root': str(args.output_root), 'metadata_sha256': hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        'data_root': str(args.data_root), 'pretrained': {key: str(value) for key, value in args.pretrained.items()},
        'seed': args.seed, 'steps': args.steps, 'text_guidance': args.text_guidance,
        'image_guidance': args.image_guidance, 'cfg_range': [0.0, 1.0],
        'time_shift_base_res': args.time_shift_base_res, 'max_long_edge': args.max_long_edge,
        'policy': 'fireward_checkpoint', 'scheduler': 'Euler',
        'checkpoint': args.checkpoint_info,
    }
    with (directory / (key + '.lock')).open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another job is writing to {args.output_root}') from exc
        record = directory / (key + '.json')
        if record.exists() and json.loads(record.read_text()) != signature:
            raise ValueError('Inference config differs from this output folder. Use a different --output-root.')
        temporary = record.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(signature, indent=2) + '\n')
        os.replace(temporary, record)
        yield


def generate(args):
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU: submit run_generate_omnigen2_gedit_fireward.slurm')
    started = time.perf_counter()
    generated = skipped = 0
    inference_seconds = 0.0
    pipe = None
    with output_lock(args):
        for index, row in enumerate(args.rows, args.start_index + 1):
            with Image.open(row['source']) as image:
                source, content_size = prepare_source(image, args.max_long_edge)
            target = row['target']
            if not args.overwrite and valid_png(target, content_size):
                skipped += 1
                print(f'[{index}] Skip completed: {target}', flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if pipe is None:
                before = time.perf_counter()
                pipe, negative_embeds, negative_mask = load_pipeline(args)
                torch.cuda.synchronize()
                print(f'Loaded pretrained + Fireward checkpoint in {time.perf_counter()-before:.2f}s', flush=True)
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = pipe(
                    prompt=[row['instruction']], input_images=[[source]], size=[source.size],
                    max_pixels=args.max_long_edge ** 2, max_input_image_side_length=args.max_long_edge,
                    align_res=True, max_sequence_length=1024, num_images_per_prompt=1,
                    num_inference_steps=args.steps, text_guidance_scale=args.text_guidance,
                    image_guidance_scale=args.image_guidance, cfg_range=(0.0, 1.0),
                    negative_prompt_embeds=negative_embeds, negative_prompt_attention_mask=negative_mask,
                    generator=torch.Generator(device='cuda:0').manual_seed(args.seed), output_type='pil',
                ).images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            if result.size != source.size:
                raise ValueError(f'Unexpected generated size {result.size}, expected {source.size}')
            result = result.crop((0, 0, *content_size)).convert('RGB')
            temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
            try:
                with temporary.open('wb') as stream:
                    result.save(stream, format='PNG')
                    stream.flush()
                    os.fsync(stream.fileno())
                if not valid_png(temporary, content_size):
                    raise OSError(f'Generated PNG failed verification: {temporary}')
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            generated += 1
            inference_seconds += elapsed
            print(f'[{index}] {row["task_type"]}/{row["key"]} seed={args.seed} size={content_size} '
                  f'seconds={elapsed:.2f} generated={generated} skipped={skipped} -> {target}', flush=True)
    print(json.dumps({'generated': generated, 'skipped': skipped, 'inference_seconds': inference_seconds,
                      'mean_inference_seconds': inference_seconds / generated if generated else None,
                      'wall_seconds': time.perf_counter()-started, 'output_root': str(args.output_root)}, indent=2), flush=True)


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--editscore-repo', type=Path, help='Path to the external EditScore repository')
    parser.add_argument('--settings', type=Path, help='Optional fallback settings; explicit pretrained arguments take precedence')
    parser.add_argument('--checkpoint', type=Path, help='checkpoint-N, its lora_adapter directory, or full model.safetensors')
    parser.add_argument('--lora-config', type=Path, help='Optional training YAML/PEFT JSON for a full-state checkpoint')
    parser.add_argument('--pretrained-root', type=Path, help='Parent containing OmniGen2/, Qwen2.5-VL-3B-Instruct/, FLUX.1-dev/')
    parser.add_argument('--transformer-path', type=Path, help='Local OmniGen2 root, transformer directory, or pytorch_model.bin')
    parser.add_argument('--text-encoder', type=Path, help='Local Qwen2.5-VL-3B-Instruct directory')
    parser.add_argument('--vae-model', type=Path, help='Local FLUX.1-dev directory containing vae/config.json')
    parser.add_argument('--metadata', type=Path, default=BENCH / 'GEdit-Bench-EN/metadata.jsonl')
    parser.add_argument('--data-root', type=Path, default=BENCH / 'GEdit-Bench-EN')
    parser.add_argument('--output-root', type=Path, default=BENCH / 'OmniGen2-Fireward/GEdit-Bench-EN')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--text-guidance', type=float, default=5.0)
    parser.add_argument('--image-guidance', type=float, default=1.5)
    parser.add_argument('--time-shift-base-res', type=int, default=168)
    parser.add_argument('--max-long-edge', type=int, default=1024)
    parser.add_argument('--cpu-offload', action=argparse.BooleanOptionalAction)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0, help='Number of metadata samples to process; default 0 means all')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    configure_parser(parser, ROOT, checkpoint=True)
    args = parser.parse_args()
    args.repo_rl = resolve_repo(ROOT, args.settings, args.editscore_repo)
    if args.steps < 1 or args.seed < 0 or args.start_index < 0 or args.limit < 0:
        parser.error('Invalid steps, seed, start-index or limit')
    if args.max_long_edge < 16 or args.max_long_edge % 16:
        parser.error('max-long-edge must be >=16 and divisible by 16')
    if not math.isfinite(args.text_guidance) or not math.isfinite(args.image_guidance) or args.text_guidance <= 1 or args.image_guidance <= 1:
        parser.error('This GEdit preset requires both guidance scales > 1')
    for field in ('settings', 'metadata', 'data_root', 'output_root'):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    if args.checkpoint is None:
        parser.error('Set checkpoint in settings.json or pass --checkpoint')
    args.checkpoint_info = resolve_checkpoint(args.checkpoint, args.lora_config)
    args.pretrained = pretrained_paths(args.settings, transformer=args.transformer_path,
        text_encoder=args.text_encoder, vae_model=args.vae_model, pretrained_root=args.pretrained_root)
    all_rows = read_pairs(args.metadata, args.data_root, args.output_root)
    args.rows = all_rows[args.start_index:args.start_index + args.limit if args.limit else None]
    if not args.rows:
        parser.error('No selected samples')
    for index, row in enumerate(args.rows, 1):
        with Image.open(row['source']) as image:
            image.verify()
        with Image.open(row['source']) as image:
            source, content = prepare_source(image, args.max_long_edge)
        if index % 100 == 0 or index == len(args.rows):
            print(f'Preflight images {index}/{len(args.rows)}', flush=True)
    print(json.dumps({'samples': len(args.rows), 'seed': args.seed, 'steps': args.steps,
                      'text_guidance': args.text_guidance, 'image_guidance': args.image_guidance,
                      'output': str(args.output_root), 'pretrained_only': False, 'checkpoint': args.checkpoint_info}, indent=2), flush=True)
    if not args.dry_run:
        generate(args)


if __name__ == '__main__':
    main()
