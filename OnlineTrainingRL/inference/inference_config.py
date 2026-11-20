"""Settings and external EditScore source resolution for both inference policies."""
import argparse
import json
from pathlib import Path


PATH_FIELDS = ('editscore_repo', 'omnigen_transformer_bin', 'text_encoder', 'vae_model',
               'metadata', 'data_root', 'output_base', 'output_checkpoint', 'checkpoint', 'lora_config')


def load_settings(path):
    path = Path(path).expanduser().resolve()
    settings = json.loads(path.read_text(encoding='utf-8-sig'))
    for field in PATH_FIELDS:
        value = settings.get(field)
        if value:
            target = Path(value).expanduser()
            settings[field] = str((target if target.is_absolute() else path.parent / target).resolve())
    return settings


def configure_parser(parser, root, *, checkpoint=False):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--settings', type=Path, default=root / 'settings.json')
    early, _ = bootstrap.parse_known_args()
    settings = load_settings(early.settings)
    defaults = {'settings': early.settings}
    for field in ('metadata', 'data_root', 'seed', 'steps', 'text_guidance', 'image_guidance',
                  'time_shift_base_res', 'max_long_edge', 'cpu_offload', 'limit', 'start_index', 'editscore_repo'):
        if field in settings:
            defaults[field] = Path(settings[field]) if field in ('metadata', 'data_root', 'editscore_repo') else settings[field]
    output = settings.get('output_checkpoint' if checkpoint else 'output_base')
    if output:
        defaults['output_root'] = Path(output)
    if checkpoint:
        for field in ('checkpoint', 'lora_config'):
            if settings.get(field):
                defaults[field] = Path(settings[field])
    parser.set_defaults(**defaults)


def resolve_repo(root, settings_file=None, explicit=None):
    config = load_settings(settings_file) if settings_file else {}
    repo = Path(explicit or config.get('editscore_repo') or root.parent / 'EditScore').expanduser().resolve()
    rl = repo / 'examples/OmniGen2-RL'
    for name in ('models/transformers/transformer_omnigen2.py', 'pipelines/omnigen2/pipeline_omnigen2.py'):
        path = rl / 'omnigen2' / name
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f'Missing/empty EditScore source: {path}; edit editscore_repo in settings.json')
    return rl
