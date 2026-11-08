#!/usr/bin/env python3
"""Official EditReward inference + cached pairwise evaluation on three local label sets.

Benchmarks (the same pairs as the proposed reward model):
- editreward_data_test: EditReward-Data-Test, 1000 HumanGT pairs (jsonl; leftvote = A wins).
- editreward_bench: EditReward-Bench 2-pair, 133 pairs, 18 ties excluded from Accuracy/AUC.
- editreward_bench_reannotation: same pairs/images as editreward_bench, reannotated labels;
  shares the editreward_bench inference cache.

Metrics follow TrainingRewardModel/AUC-Margin.md: margin = score_A - score_B, Accuracy =
sign(margin) == label (margin 0 is wrong), symmetric within-pair margin AUC.
``evaluation.json`` stores per-pair predictions in the format read by
OurModel/recompute_margin_auc.py (--evaluation EditReward=...) for CIs and paired tests.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

ROOT = Path('/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation')
REPO = ROOT / 'OtherModels/EditReward/EditReward'
DATA = ROOT / 'EvalDatasets'
EDITREWARD_ROOT = Path('/datastore/shared/datasets/EditReward-Data')
MODEL_NAME = 'EditReward-Qwen2.5-VL-7B'
SCORE_DEFINITION = ('overall = mean of head_0 (instruction-following prompt) and head_1 '
                    '(visual-quality prompt), reward()[i][0]')
FILES = {
    'editreward_data_test': 'EditReward-Test_pairwise_1k.jsonl',
    'editreward_bench': 'EditReward-Bench_pairwise.json',
    'editreward_bench_reannotation': 'EditReward-Bench_pairwise_Reannotation.json',
}
# Dataset names used by OurModel evaluation files and recompute_margin_auc.py.
REPORT_NAMES = {
    'editreward_data_test': 'EditReward-Data-Test',
    'editreward_bench': 'EditReward-Bench',
    'editreward_bench_reannotation': 'EditReward-Bench-Reannotation',
}
CACHE_GROUPS = {
    'editreward_data_test': 'editreward_data_test',
    'editreward_bench': 'editreward_bench',
    'editreward_bench_reannotation': 'editreward_bench',
}
VOTE_TO_RANKING = {'leftvote': 'A>B', 'rightvote': 'B>A', 'tie': 'A=B', 'tievote': 'A=B'}

def read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def resolve_image(raw, editreward_root):
    """Absolute path, or relative to the EditReward-Data root (or its extracted_data/)."""
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [editreward_root/path, editreward_root/'extracted_data'/path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f'Image not found: {raw}')

def jsonl_samples(path, editreward_root):
    """EditReward-Data-Test rows -> 2-pair samples (A = left image, B = right image)."""
    samples = []
    with open(path, encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            vote = str(row['vote_type']).strip().lower()
            if vote not in VOTE_TO_RANKING:
                raise ValueError(f'{path}:{line_number}: unsupported vote_type {vote!r}')
            samples.append({
                'sample_id': row['unique_key'],
                'source_image': resolve_image(row['source_img'], editreward_root),
                'candidate_1': resolve_image(row['left_output_image'], editreward_root),
                'candidate_2': resolve_image(row['right_output_image'], editreward_root),
                'edit_instruction': row['instruction'],
                'metadata_ranking': VOTE_TO_RANKING[vote],
            })
    return samples

def samples_of(path, editreward_root):
    if path.suffix == '.jsonl':
        samples = jsonl_samples(path, editreward_root)
    else:
        samples = read(path)
        if not isinstance(samples, list):
            raise ValueError(f'Expected JSON list: {path}')
    ids = [s['sample_id'] for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate sample_id in {path}')
    return samples

def identity(sample):
    return tuple(str(sample[k]) for k in ('sample_id', 'source_image', 'candidate_1', 'candidate_2', 'edit_instruction'))

def label_of(sample):
    """+1 if A wins, -1 if B wins, 0 for a tie."""
    raw = str(sample['metadata_ranking']).strip().upper().replace(' ', '')
    if raw == 'A>B':
        return 1
    if raw in ('B>A', 'A<B'):
        return -1
    if raw in ('A=B', 'TIE'):
        return 0
    raise ValueError(f"Unsupported metadata_ranking {sample['metadata_ranking']!r} for {sample.get('sample_id')}")

def compute_margin_metrics(samples, predictions):
    """Accuracy and symmetric within-pair margin AUC (AUC-Margin.md); ties excluded."""
    labels, margins = [], []
    for sample, pred in zip(samples, predictions):
        label = label_of(sample)
        if label == 0:
            continue
        margin = float(pred['score_a']) - float(pred['score_b'])
        if not np.isfinite(margin):
            raise ValueError(f"Non-finite reward margin for {sample.get('sample_id')}")
        labels.append(label)
        margins.append(margin)
    if not labels:
        raise ValueError('No non-tie pairs available for margin AUC')
    y = np.asarray(labels, dtype=np.float64)
    m = np.asarray(margins, dtype=np.float64)
    d = y * m
    # Equivalent to ROC AUC of positive d versus negative -d, with average ranks.
    values = np.concatenate((d, -d))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    ranks = (last_rank - (counts - 1) / 2.0)[inverse]
    n = len(d)
    auc = (ranks[:n].sum() - n * (n + 1) / 2.0) / (n * n)
    return {
        'accuracy_percent': round(float(np.mean(np.sign(m) == y) * 100), 6),
        'margin_auc_percent': round(float(auc * 100), 6),
        'non_tie_pairs': n,
        'label_a_wins': int((y > 0).sum()),
        'label_b_wins': int((y < 0).sum()),
        'excluded_tie_pairs': len(samples) - n,
        'zero_margin_pairs': int(np.count_nonzero(m == 0)),
        'margin_definition': 'score_a - score_b, score = overall reward()[i][0] (mean of head_0 and head_1)',
        'auc_definition': 'ROC AUC of signed winner-minus-loser margins d versus -d; average ranks for ties',
    }

def evaluate(samples, predictions, name, source, output):
    from evaluation_metrics import evaluate_pairwise
    if len(samples) != len(predictions):
        raise ValueError('Prediction/sample length mismatch')
    for s, p in zip(samples, predictions):
        if identity(s) != tuple(p['identity']):
            raise ValueError(f'Prediction identity mismatch at {s.get("sample_id")}')
    labels = [label_of(s) for s in samples]
    a = [p['score_a'] for p in predictions]
    b = [p['score_b'] for p in predictions]

    from collections import Counter

    print(f"Benchmark: {name}")
    print(f"Number of samples: {len(labels)}")
    print(f"Label distribution: {Counter(labels)}")

    summary = evaluate_pairwise(labels, a, b)
    summary['symmetric_margin_evaluation'] = compute_margin_metrics(samples, predictions)
    summary.update(model=MODEL_NAME, benchmark=name,
                   benchmark_path=str(source), score_definition=SCORE_DEFINITION)
    details = [{**{k:v for k,v in p.items() if k != 'identity'},
                'ground_truth': s['metadata_ranking'], 'selected_metrics': s.get('selected_metrics'),
                'margin': p['score_a']-p['score_b'],
                'tie_confidence': -abs(p['score_a']-p['score_b'])}
               for s,p in zip(samples,predictions)]
    write(output/name/'summary.json', summary)
    write(output/name/'detailed_samples.json', details)
    return summary

def prediction_rows(samples, predictions):
    """Non-tie rows in the OurModel evaluation format (recompute_margin_auc.py)."""
    rows = []
    for s, p in zip(samples, predictions):
        label = label_of(s)
        if label == 0:
            continue
        rows.append({
            'sample_id': s['sample_id'], 'label': label,
            'source_image': s['source_image'], 'edit_instruction': s['edit_instruction'],
            'image_a': s['candidate_1'], 'image_b': s['candidate_2'],
            'score_A': float(p['score_a']), 'score_B': float(p['score_b']),
            'margin_A_minus_B': float(p['score_a']) - float(p['score_b']),
        })
    return rows

def infer(samples, inferencer, cache, batch_size):
    predictions = read(cache) if cache.exists() else []
    if not isinstance(predictions, list) or len(predictions) > len(samples):
        raise ValueError(f'Invalid cache: {cache}')
    for i,p in enumerate(predictions):
        if tuple(p['identity']) != identity(samples[i]):
            raise ValueError(f'Cache mismatch at index {i}; use --reset_cache')
    for start in tqdm(range(len(predictions), len(samples), batch_size), desc=cache.parent.name):
        batch = samples[start:start+batch_size]
        prompts = [s['edit_instruction'] for s in batch for _ in range(2)]
        sources = [s['source_image'] for s in batch for _ in range(2)]
        targets = [path for s in batch for path in (s['candidate_1'], s['candidate_2'])]
        for path in sources+targets:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        with torch.inference_mode():
            rewards = inferencer.reward(prompts=prompts, image_src=sources, image_paths=targets)
        if isinstance(rewards, torch.Tensor):
            scores = rewards[:, 0].detach().float().cpu().tolist()
        else:
            scores = [float(r[0].item() if hasattr(r[0], 'item') else r[0]) for r in rewards]
        if len(scores) != 2*len(batch):
            raise ValueError('Unexpected reward output length')
        for j,s in enumerate(batch):
            predictions.append(dict(sample_id=s['sample_id'], identity=list(identity(s)),
                                    score_a=float(scores[2*j]), score_b=float(scores[2*j+1])))
        write(cache, predictions)
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['all', *FILES], default='all')
    parser.add_argument('--benchmark_dir', type=Path, default=DATA)
    parser.add_argument('--editreward_root', type=Path, default=EDITREWARD_ROOT,
                        help='Root for relative image paths of EditReward-Data-Test')
    parser.add_argument('--source_root', type=Path, default=REPO)
    parser.add_argument('--model_dir', type=Path, default=REPO/'pretrained')
    parser.add_argument('--output_dir', type=Path, default=ROOT/'OtherModels/EditReward/evaluation_results')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of PAIRS per batch')
    parser.add_argument('--reset_cache', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch_size must be >= 1')
    if args.benchmark == 'all':
        targets = list(FILES)
    elif args.benchmark in ('editreward_bench', 'editreward_bench_reannotation'):
        targets = ['editreward_bench', 'editreward_bench_reannotation']
    else:
        targets = [args.benchmark]
    # Load and resolve every input before the model is loaded, so path errors fail fast.
    datasets = {name: samples_of(args.benchmark_dir/FILES[name], args.editreward_root) for name in targets}
    for name, samples in datasets.items():
        print(f'Loaded {name}: {len(samples)} samples from {args.benchmark_dir/FILES[name]}', flush=True)
    # Reannotations must match all inference inputs, not merely sample IDs.
    if 'editreward_bench_reannotation' in datasets:
        original = datasets['editreward_bench']
        if [identity(s) for s in original] != [identity(s) for s in datasets['editreward_bench_reannotation']]:
            raise ValueError('Reannotation has different samples/order; cannot share cache')
    cache_paths = {name: args.output_dir/CACHE_GROUPS[name]/'predictions.json' for name in targets}
    if args.reset_cache:
        for p in set(cache_paths.values()):
            p.unlink(missing_ok=True)
    needs_inference = any(not p.exists() or len(read(p)) < len(datasets[name]) for name,p in cache_paths.items())
    inferencer = None
    if needs_inference:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for inference')
        sys.path.insert(0, str(args.source_root))
        # Same inferencer as the earlier slide run (eval_editreward_3pair.py). Its create_model_and_processor
        # passes aggregation_threshold, which the Qwen2.5 reward model class does not accept, so drop it.
        from transformers import PreTrainedModel
        original_from_pretrained = PreTrainedModel.from_pretrained
        @classmethod
        def from_pretrained_without_aggregation_threshold(cls, *model_args, **kwargs):
            kwargs.pop('aggregation_threshold', None)
            return original_from_pretrained.__func__(cls, *model_args, **kwargs)
        PreTrainedModel.from_pretrained = from_pretrained_without_aggregation_threshold
        from EditReward.inference_vl_edit import EditRewardVLInferencer
        config = args.source_root/'EditReward/config/EditReward-Qwen2.5-7B-VL.yaml'
        if not config.is_file():
            raise FileNotFoundError(config)
        if not args.model_dir.is_dir():
            raise FileNotFoundError(args.model_dir)
        # Preserve all official YAML hyperparameters; only redirect the backbone path.
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config)
        cfg.model_name_or_path = str(args.model_dir)
        cfg.output_dir = str(args.output_dir/'runtime_output')
        runtime = args.output_dir/'runtime_config.yaml'
        runtime.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, runtime)
        # Official released setting: rm_head_type='ranknet_multi_head' ignores reward_dim and scores head_0
        # (instruction-following prompt) and head_1 (visual-quality prompt), fused by pooling_strategy 'mean'.
        inferencer = EditRewardVLInferencer(config_path=str(runtime), checkpoint_path=str(args.model_dir),
                                           device='cuda', reward_dim='overall_detail',
                                           rm_head_type='ranknet_multi_head')
    summaries, rows = {}, {}
    for name in targets:
        samples = datasets[name]
        cache = cache_paths[name]
        predictions = infer(samples, inferencer, cache, args.batch_size) if inferencer is not None and (not cache.exists() or len(read(cache)) < len(samples)) else read(cache)
        if len(predictions) != len(samples):
            raise ValueError(f'Incomplete predictions for {name}: {len(predictions)}/{len(samples)}')
        summaries[name] = evaluate(samples, predictions, name, args.benchmark_dir/FILES[name], args.output_dir)
        rows[REPORT_NAMES[name]] = prediction_rows(samples, predictions)
        print(f'Finished {name}: {len(samples)} samples', flush=True)
    write(args.output_dir/'overall_summary.json', summaries)
    if args.benchmark == 'all':
        write(args.output_dir/'evaluation.json', {
            'format_version': 1,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_role': 'other_reward_model',
            'model': MODEL_NAME,
            'checkpoint': str(args.model_dir),
            'score_definition': SCORE_DEFINITION + '; margin_A_minus_B = score_A - score_B',
            'image_pixels': 'official config: min_pixels = max_pixels = 200704',
            'benchmarks': {REPORT_NAMES[n]: str(args.benchmark_dir/FILES[n]) for n in targets},
            'predictions': rows,
        })
    for name, summary in summaries.items():
        metrics = summary['symmetric_margin_evaluation']
        print(f"{name}: Accuracy={metrics['accuracy_percent']:.2f}% | Margin AUC={metrics['margin_auc_percent']:.2f}% | non-tie={metrics['non_tie_pairs']} | excluded ties={metrics['excluded_tie_pairs']} | zero margins={metrics['zero_margin_pairs']}")
    print('Saved to', args.output_dir)

if __name__ == '__main__':
    main()
