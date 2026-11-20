#!/usr/bin/env python3
"""Validate unique editing conditions and freeze a job-specific configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def probe_write_capacity(directory, mib):
    """An actual write/fsync catches quota errors that df cannot detect."""
    if mib < 0:
        raise ValueError('write-probe-mib must be nonnegative')
    if not mib:
        return
    path = directory / '.checkpoint_write_probe'
    print(f'Checking {mib} MiB of actual checkpoint write capacity...', flush=True)
    try:
        chunk = b'\0' * (1024 * 1024)
        with path.open('xb') as stream:
            for _ in range(mib):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise OSError(exc.errno, f'Checkpoint quota/write probe failed before loading models: {exc}') from exc
    finally:
        if path.exists():
            path.unlink()
    print('Write probe OK; temporary probe file removed. Future quota usage can still change.', flush=True)


def prepare_rows(metadata, data_root, verify=True):
    from PIL import Image
    rows, seen_pairs, seen_keys = [], set(), set()
    with metadata.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            instruction = item.get('instruction', item.get('prompt'))
            src = item.get('source_img', item.get('source_image', item.get('src_image')))
            if not isinstance(instruction, str) or not instruction.strip() or not isinstance(src, str):
                raise ValueError(f'{metadata}:{number}: expected instruction and source_img/source_image/src_image')
            source = Path(src)
            source = (source if source.is_absolute() else data_root / source).resolve()
            identity = (str(source), instruction.strip())
            if identity in seen_pairs:
                continue
            seen_pairs.add(identity)
            key = str(item.get('unique_key') or item.get('key') or hashlib.sha256(repr(identity).encode()).hexdigest())
            if key in seen_keys:
                key += '_' + hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
            if key in seen_keys:
                raise ValueError(f'Duplicate sample key: {key}')
            seen_keys.add(key)
            if not source.is_file():
                raise FileNotFoundError(source)
            if verify:
                with Image.open(source) as image:
                    image.verify()
                with Image.open(source) as image:
                    image.load()
            meta = {'sample_key': key, 'instruction': instruction.strip(), 'source_image': str(source)}
            rows.append({'instruction': instruction.strip(), 'input_images': [str(source)],
                         'task_type': 'image_edit', 'meta_data': json.dumps(meta, ensure_ascii=False)})
            if len(rows) % 500 == 0:
                print(f'Validated {len(rows)} unique source/instruction pairs', flush=True)
    if not rows:
        raise ValueError('No training pairs')
    return rows


def resolve_model(explicit, repo, marker, roots):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / marker).is_file():
            raise FileNotFoundError(f'Missing {marker}: {path}')
        return path
    folder = 'models--' + repo.replace('/', '--')
    for root in roots:
        root = Path(root).expanduser()
        for base in (root / folder, root / 'hub' / folder):
            ref = base / 'refs/main'
            if ref.is_file():
                candidate = base / 'snapshots' / ref.read_text().strip()
                if (candidate / marker).is_file():
                    return candidate.resolve()
            candidates = sorted(p.parent for p in (base / 'snapshots').glob('*/' + marker))
            if len(candidates) == 1:
                return candidates[0].resolve()
    raise FileNotFoundError(f'{repo} not found offline. Set its local path in settings.json; searched {roots}')


def resolve_repo(bundle, settings):
    value = settings.get('editscore_repo')
    if not value:
        raise ValueError('Set editscore_repo in running/settings.json')
    repo = Path(value).expanduser()
    repo = (repo if repo.is_absolute() else bundle / repo).resolve()
    rl = repo / 'examples/OmniGen2-RL'
    for relative in ('omnigen2/models/transformers/transformer_omnigen2.py',
                     'omnigen2/grpo/utils.py', 'omnigen2/grpo/reward_client_edit.py',
                     'omnigen2/pipelines/omnigen2/pipeline_omnigen2.py'):
        path = rl / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/empty EditScore source: {path}')
    return repo, rl


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--write-probe-mib', type=int, default=0)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    settings = json.loads((bundle / 'settings.json').read_text(encoding='utf-8'))
    repo, rl = resolve_repo(bundle, settings)
    settings['editscore_repo'] = str(repo)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    probe_write_capacity(run, args.write_probe_mib)
    settings['reward_project'] = str(Path(settings['reward_project']).resolve())
    for path in (Path(settings['reward_project']) / 'src/models/custom_qwen3_vl_reranker.py',
                 Path(settings['reward_pretrained']) / 'config.json', Path(settings['reward_checkpoint'])):
        if not path.exists() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f'Missing/unreadable: {path}')
    config = yaml.safe_load((bundle / 'train.yml').read_text())
    roots = settings['hf_cache_roots']
    transformer = Path(settings['omnigen_transformer_bin']) if settings['omnigen_transformer_bin'] else rl / 'pretrained_models/OmniGen2/transformer/pytorch_model.bin'
    if not (transformer.is_file() or (transformer / 'config.json').is_file()):
        for root in roots:
            candidate = Path(root) / 'OmniGen2/transformer/pytorch_model.bin'
            if candidate.is_file():
                transformer = candidate
                break
    if not (transformer.is_file() or (transformer / 'config.json').is_file()):
        raise FileNotFoundError(f'Set omnigen_transformer_bin to a local transformer directory or pytorch_model.bin: {transformer}')
    config['model']['pretrained_model_path'] = str(transformer.resolve())
    config['model']['pretrained_text_encoder_model_name_or_path'] = str(resolve_model(settings['text_encoder'], 'Qwen/Qwen2.5-VL-3B-Instruct', 'config.json', roots))
    config['model']['pretrained_vae_model_name_or_path'] = str(resolve_model(settings['vae_model'], 'black-forest-labs/FLUX.1-dev', 'vae/config.json', roots))
    settings['metadata'] = str((bundle / Path(settings['metadata']).expanduser()).resolve())
    rows = prepare_rows(Path(settings['metadata']), Path(settings['data_root']))
    data = run / 'training_pairs.jsonl'
    with data.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    data_config = run / 'data.yml'
    data_config.write_text(yaml.safe_dump({'data': [{'path': str(data), 'ratio': 1}]}), encoding='utf-8')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        settings['port'] = sock.getsockname()[1]
    settings['run_id'] = str(run)
    config['data']['data_path'] = str(data_config)
    config['name'] = run.name
    config['output_dir'] = str(run / 'checkpoints')
    config['reward_server_config'] = str(run / 'reward_server.yml')
    (run / 'reward_server.yml').write_text(yaml.safe_dump({'server': {'hosts': ['127.0.0.1'], 'proxy_port': settings['port']}}))
    (run / 'train.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    (run / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    # Freeze every executable and loader used by this job, so later edits do not change a running job.
    import shutil
    for name in ('train.py', 'compact_checkpoint.py', 'reward_server.py', 'reward_runtime.py'):
        shutil.copy2(bundle / name, run / name)
    (run / 'repo_rl_path.txt').write_text(str(rl))
    import subprocess
    try:
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    (run / 'source.json').write_text(json.dumps({'editscore_repo': str(repo), 'commit': revision}, indent=2))
    print(json.dumps({'run': str(run), 'unique_pairs': len(rows), 'model': config['model'],
                      'train': config['train']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
