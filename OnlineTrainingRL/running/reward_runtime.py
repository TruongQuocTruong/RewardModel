#!/usr/bin/env python3
"""Frozen Student loader copied from the existing Stage2 inference contract.
No dependency on an external scoring script. Architecture/prompt source is read
from the configured Qwen3 project, using the exact checkpoint head shapes.
"""
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import sys

def checkpoint_files(path):
    """Resolve an explicit training file, PEFT export, or unambiguous directory."""
    if path.is_file():
        if path.suffix in ('.pt', '.pth', ''):
            return path
        raise ValueError(f'Use a training .pt/.pth file or a PEFT directory: {path}')
    if not path.is_dir():
        raise FileNotFoundError(f'Checkpoint missing or unreadable: {path}')
    # The training loop writes the validation-selected Student to best.pt.
    if (path / 'best.pt').is_file():
        return path / 'best.pt'
    if (path / 'adapter_config.json').is_file():
        for name in ('adapter_model.safetensors', 'adapter_model.bin'):
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f'PEFT weights missing: {path}')
    candidates = sorted(set(p.resolve() for p in itertools.chain(
        path.glob('*.pt'), path.glob('*.pth'),
        path.glob('*/adapter_config.json'), path.glob('*/*.pt'), path.glob('*/*.pth'),
    ) if p.name not in ('optimizer.pt', 'scheduler.pt', 'scaler.pt')))
    if len(candidates) == 1:
        candidate = candidates[0]
        return checkpoint_files(candidate.parent if candidate.name == 'adapter_config.json' else candidate)
    raise ValueError(
        f'Cannot uniquely select checkpoint under {path}. '
        f'Pass --checkpoint with the exact training file or PEFT directory. Candidates: {candidates}'
    )

def student_checkpoint_payload(checkpoint, metric_names):
    """Same Student role/state contract as evaluate_reward_model.load_refined_model."""
    role = checkpoint.get('model_role')
    allowed = ('six_metric_refinement_student_teacher', 'six_metric_student_warmstart',
               'six_metric_supervised_control_student')
    if role not in allowed:
        raise ValueError(f'Checkpoint role {role!r} is incompatible with student')
    if list(checkpoint.get('metric_order', [])) != list(metric_names):
        raise ValueError('Checkpoint metric order is incompatible')
    key = ('adapter_state_dict' if role == 'six_metric_student_warmstart'
           else 'student_adapter_state_dict')
    state = checkpoint.get(key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Checkpoint is missing {key}')
    if not isinstance(checkpoint.get('peft_config'), dict):
        raise ValueError('Checkpoint is missing peft_config')
    if not isinstance(checkpoint.get('prompt'), dict):
        raise ValueError('Checkpoint is missing the prompt contract')
    return state, checkpoint['prompt']

def canonical_head_state(state):
    heads = {}
    for name, tensor in state.items():
        if 'metric_heads.' not in name:
            continue
        suffix = name.split('metric_heads.', 1)[1]
        if suffix.startswith('original_module.'):
            continue
        for prefix in ('modules_to_save.default.', 'default.'):
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix):]
        if suffix in heads:
            raise ValueError(f'Duplicate checkpoint head tensor: {suffix}')
        heads[suffix] = tensor
    required = ('score_head.pretrained_projection.weight', 'score_head.hidden.weight',
                'tie_head.projection.weight', 'relevance_head.pretrained_projection.weight')
    if not all(name in heads for name in required):
        raise ValueError('Checkpoint does not contain the trained six-metric score/tie/relevance heads; refusing random-head inference')
    return heads

def load_custom_module(args):
    sys.path.insert(0, str(args.project_root.resolve()))
    module_path = args.custom_model.resolve()
    spec = importlib.util.spec_from_file_location('_student_reward_architecture', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from src import prompt_config
    if Path(prompt_config.__file__).resolve() != (args.project_root / 'src/prompt_config.py').resolve():
        raise RuntimeError('A different project src.prompt_config was imported')
    return module, prompt_config

def build_reward_heads(module, head_state):
    """Support both the original independent Tie Head and newer coupled heads.

    Older custom modules have neither tie_head_mode_from_state_keys nor a
    tie_head_mode constructor argument. Never drop checkpoint tensors to make
    an incompatible architecture appear to load successfully.
    """
    import torch
    mode = ('margin_coupled' if any(key.startswith('tie_coupling.') for key in head_state)
            else 'independent')
    constructor = module.MultiMetricRewardHeads
    parameters = inspect.signature(constructor).parameters
    kwargs = {
        'num_metrics': 6,
        'hidden_dim': int(head_state['score_head.hidden.weight'].shape[0]),
        'tie_projection_dim': int(head_state['tie_head.projection.weight'].shape[0]),
        'dropout_p': getattr(module, 'DEFAULT_HEAD_DROPOUT', 0.05),
    }
    if 'tie_head_mode' in parameters:
        kwargs['tie_head_mode'] = mode
    elif mode != 'independent':
        raise ValueError(
            'Checkpoint contains tie_coupling weights, but this custom model '
            'has no tie_head_mode support. Use the custom model source from '
            'checkpoint training; coupled weights cannot be discarded.'
        )
    heads = constructor(
        int(head_state['score_head.pretrained_projection.weight'].shape[1]), **kwargs,
    ).to(dtype=head_state['score_head.hidden.weight'].dtype)
    heads.load_state_dict(head_state, strict=True)
    return heads, mode

def load_reward(args):
    import torch
    from peft import PeftModel, get_peft_model_state_dict, LoraConfig, PeftConfig, set_peft_model_state_dict
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run scoring through Slurm')
    weights = checkpoint_files(args.checkpoint)
    module, prompt_config = load_custom_module(args)
    names = list(module.METRIC_NAMES)
    checkpoint_prompt = None
    checkpoint = None
    if weights.name == 'adapter_model.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(weights), device='cpu')
    elif weights.name != 'adapter_model.bin':
        # The user's training checkpoints contain metadata as well as tensors.
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        state, checkpoint_prompt = student_checkpoint_payload(checkpoint, names)
    else:
        state = torch.load(weights, map_location='cpu', weights_only=True)
    if not any('lora_' in name for name in state):
        raise ValueError('Checkpoint has no LoRA tensors')
    head_state = canonical_head_state(state)
    if len(names) != 6 or head_state['score_head.pretrained_projection.weight'].shape[0] != 6:
        raise ValueError('Expected six reward metrics')
    # Validate the actual custom head architecture before loading the 8B backbone.
    heads, mode = build_reward_heads(module, head_state)
    pixel_kwargs = {key: value for key, value in (('min_pixels', args.min_pixels), ('max_pixels', args.max_pixels)) if value is not None}
    processor = AutoProcessor.from_pretrained(str(args.pretrained_dir), local_files_only=True,
                                             padding_side='left', **pixel_kwargs)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.pretrained_dir), torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True,
    )
    base.metric_heads = heads
    if checkpoint is None:
        model = PeftModel.from_pretrained(base, str(args.checkpoint), is_trainable=False, local_files_only=True)
    else:
        # Preserve the adapter wrapping used by the supplied reference loader.
        try:
            from swift.tuners import Swift
        except ImportError as exc:
            raise ImportError('Loading training .pt checkpoints requires ms-swift: python -m pip install ms-swift') from exc
        supported = set(inspect.signature(LoraConfig).parameters)
        config = {key: value for key, value in checkpoint['peft_config'].items() if key in supported}
        config['inference_mode'] = False
        model = Swift.prepare_model(base, PeftConfig.from_peft_type(**config))
        result = set_peft_model_state_dict(model, state, adapter_name='default')
        if result.unexpected_keys:
            raise RuntimeError(f'Unexpected checkpoint keys: {result.unexpected_keys[:10]}')
    restored = get_peft_model_state_dict(model, adapter_name='default')
    verified = 0
    for name, tensor in state.items():
        if 'metric_heads.' in name:
            continue
        if name not in restored or restored[name].shape != tensor.shape:
            raise RuntimeError(f'Adapter tensor was not restored: {name}')
        actual = restored[name].detach().cpu()
        if not torch.equal(actual, tensor.to(dtype=actual.dtype)):
            raise RuntimeError(f'Adapter tensor differs after loading: {name}')
        verified += 1
    # Explicit strict head restoration also protects against PEFT head-prefix differences.
    active = model.get_base_model().metric_heads
    if hasattr(active, 'modules_to_save'):
        active = active.modules_to_save['default']
    active.load_state_dict(head_state, strict=True)
    model.eval()
    model.requires_grad_(False)
    model.to(args.device)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if checkpoint_prompt is not None:
        prompt = {key: checkpoint_prompt[key] for key in ('system_message', 'evaluation_instruction')}
    else:
        prompt = {'system_message': prompt_config.DEFAULT_SYSTEM_MESSAGE,
                  'evaluation_instruction': prompt_config.DEFAULT_EVALUATION_INSTRUCTION}
    if args.prompt_file:
        payload = json.loads(args.prompt_file.read_text(encoding='utf-8'))
        if 'profiles' in payload:
            payload = payload['profiles']['six_metric']
            prompt = {'system_message': payload['system_prompt'], 'evaluation_instruction': payload['evaluation_instruction']}
        else:
            prompt = {key: payload[key] for key in ('system_message', 'evaluation_instruction')}
    if not all(isinstance(text, str) and text.strip() for text in prompt.values()):
        raise ValueError('Invalid prompt contract')
    print(f'Loaded checkpoint: {weights}; verified adapter tensors={verified}; strict head tensors={len(head_state)}; tie mode={mode}', flush=True)
    return model, processor, module, names, prompt

def pair_conversations(source, instruction, a, b, prompt):
    def conversation(candidate):
        content = [
            {'type': 'text', 'text': f"<Instruct>: {prompt['evaluation_instruction']}\n<Query>:"},
            {'type': 'image', 'image': f'file://{source}'},
            {'type': 'text', 'text': f' {instruction}' + ('\n<Document>:' if candidate else '')},
        ]
        if candidate:
            content.append({'type': 'image', 'image': f'file://{candidate}'})
        return [{'role': 'system', 'content': prompt['system_message']}, {'role': 'user', 'content': content}]
    return [conversation(a), conversation(b), conversation(None)]

def forward_pair(model, processor, module, prompt, source, instruction, a, b, device):
    import torch
    from qwen_vl_utils import process_vision_info
    conversations = pair_conversations(source, instruction, a, b, prompt)
    texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(conversations)
    inputs = processor(text=texts, images=images, videos=videos, padding=True, return_tensors='pt')
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    with torch.inference_mode():
        hidden = module._pool_qwen_last_hidden(model, inputs, inputs.get('attention_mask'))
        output = model.get_base_model().metric_heads(hidden[0:1], hidden[1:2], hidden[2:3])
    result = {}
    fields = ('scores_A', 'scores_B', 'relevance_probs')
    for field in fields:
        tensor = getattr(output, field).detach().float().cpu()
        expected = (1, 6)
        if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid reward output: {field}')
        result[field] = tensor[0].tolist()
    return result
