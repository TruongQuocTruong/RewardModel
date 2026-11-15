#!/usr/bin/env python3
"""Evaluate one Stage 2 pipeline ablation checkpoint (Student or Teacher) on EditReward-Test 1k5.

Copy of ``RewardModelEvaluation/OurModel/evaluate_student_checkpoints.py`` reduced to the
setting used for the pipeline ablation evaluations (``--datasets test``, no Metrics Weights
Aggregator, no ``--adapter-dir``, no GemmaGT validation, batch size 1, 1000 bootstrap
replicates). The output file and its JSON layout are the same as that script's
(``test_evaluation.json`` for the Student, ``teacher_test_evaluation.json`` for the Teacher).

EditReward-Test 1k5 = 1000 non-tie pairs + 500 tie pairs. Accuracy and margin AUC of
``selected_equal`` use the non-tie pairs; the three-way metrics (Tie Head AUC, three-class
accuracy) use every pair. ``selected_equal`` averages over the metrics the Relevance Head
selects, S = {k : relevance_probs_k >= 0.5} (the most relevant metric when none passes):
margin = mean_{k in S} signed_evidence_k.

Model code is imported from the main repository (this folder sits at its root):
``TrainingRewardModel/Qwen3-VL-Embedding`` and ``RewardModelEvaluation/OurModel``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent  # ComparisonPipelineAblation
_REPO_ROOT = _THIS_DIR.parent
_PROJECT_ROOT = _REPO_ROOT / "TrainingRewardModel" / "Qwen3-VL-Embedding"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_EVAL_DIR = _REPO_ROOT / "RewardModelEvaluation" / "OurModel"
for path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_inspector_checkpoints import (  # noqa: E402
    _margin_auc,
    _pooled_auc,
    _three_way_metrics,
)
from evaluate_reward_model import BenchmarkRecord, load_refined_model  # noqa: E402
from train_loop import PairRecord, forward_pairs  # noqa: E402
from train_overall_inspector import load_inspector_records  # noqa: E402
from src.models.custom_qwen3_vl_reranker import METRIC_NAMES, NUM_METRICS  # noqa: E402

TEST_DATASET = "EditReward-Data-Test"
VOTE_TO_LABEL = {"A": 1, "B": -1, "Tie": 0}
RELEVANCE_THRESHOLD = 0.5
ROLES = ("student", "teacher")
DEFAULT_TEST_DATASET = _THIS_DIR / "data" / "EditReward-Test_pairwise_1k5.jsonl"
# Only used to check that no Test pair or edited image is in HumanGT Data (as in the original runs).
DEFAULT_HUMAN_DATASET = (
    _REPO_ROOT / "TrainingRewardModel" / "TrainingDataCreation" / "editreward_overall_humanGT_3k.jsonl"
)


def evaluation_file_name(role: str) -> str:
    """test_evaluation.json for the Student, teacher_test_evaluation.json for the Teacher."""
    if role not in ROLES:
        raise ValueError(f"Unsupported --role: {role}")
    return "test_evaluation.json" if role == "student" else "teacher_test_evaluation.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # A Stage 2 checkpoint-N.pt holds both adapters; teacher scores the EMA Teacher Model.
    parser.add_argument("--role", choices=ROLES, default="student")
    parser.add_argument("--checkpoint", required=True, help="Stage 2 checkpoint-N.pt")
    parser.add_argument("--base-model-path", required=True, help="Qwen3-VL-Reranker-8B folder")
    parser.add_argument(
        "--editreward-root", required=True,
        help="EditReward-Data root; image paths in the Test file are relative to it.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-dataset", default=str(DEFAULT_TEST_DATASET))
    parser.add_argument("--human-dataset", default=str(DEFAULT_HUMAN_DATASET))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    # Fields of the original output "configuration", so result files stay comparable.
    args.datasets = "test"
    return args


def _to_benchmark(records) -> List[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            sample_id=record.sample_id,
            source_image=record.source_image,
            edit_instruction=record.edit_instruction,
            image_a=record.image_a,
            image_b=record.image_b,
            label=VOTE_TO_LABEL[record.vote],
        )
        for record in records
    ]


def load_test_records(args: argparse.Namespace) -> List[BenchmarkRecord]:
    """EditReward-Test 1k5 with its tie pairs; refuses a Test pair or edited image found in HumanGT Data."""
    test_records, _ = load_inspector_records(args.test_dataset, args.editreward_root)
    if not test_records:
        raise ValueError(f"No pairs in {args.test_dataset}")
    human_records, _ = load_inspector_records(args.human_dataset, args.editreward_root)
    if {r.sample_id for r in test_records} & {r.sample_id for r in human_records}:
        raise AssertionError("EditReward-Data test set shares a pair with HumanGT Data.")
    human_images = {img for r in human_records for img in (r.image_a, r.image_b)}
    if {img for r in test_records for img in (r.image_a, r.image_b)} & human_images:
        raise AssertionError("EditReward-Data test set shares an edited image with HumanGT Data.")
    return _to_benchmark(test_records)


def extract_outputs(model, processor, prompt, records: Sequence[BenchmarkRecord], batch_size, device):
    import torch

    fields = ("scores_A", "scores_B", "signed_evidence", "tie_probs", "relevance_probs")
    outputs: Dict[str, List[List[float]]] = {field: [] for field in fields}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            pairs = [
                PairRecord(
                    sample_id=r.sample_id,
                    source_image=r.source_image,
                    edit_instruction=r.edit_instruction,
                    image_a=r.image_a,
                    image_b=r.image_b,
                )
                for r in batch
            ]
            output = forward_pairs(model, processor, pairs, prompt, device)
            for field in fields:
                outputs[field].extend(getattr(output, field).float().cpu().tolist())
            done = start + len(batch)
            if done % 100 < batch_size or done == len(records):
                print(f"Scored {done}/{len(records)}", flush=True)
    arrays = {field: np.asarray(values, dtype=np.float64) for field, values in outputs.items()}
    check_outputs(arrays, len(records))
    return arrays


def check_outputs(arrays: Dict[str, np.ndarray], count: int) -> None:
    for field, array in arrays.items():
        if array.shape != (count, NUM_METRICS) or not np.isfinite(array).all():
            raise ValueError(f"Invalid model output {field}: shape {array.shape}")


def selection_mask(relevance: np.ndarray) -> np.ndarray:
    mask = relevance >= RELEVANCE_THRESHOLD
    empty = ~mask.any(axis=1)
    mask[empty, relevance[empty].argmax(axis=1)] = True
    return mask


def selected_equal_three_way(records, arrays, mask):
    """Three-way metrics over every pair (ties included) for ``selected_equal``."""
    tie = arrays["tie_probs"]
    cond_a = 1.0 / (1.0 + np.exp(-(arrays["scores_A"] - arrays["scores_B"])))
    per_metric = np.stack([(1.0 - tie) * cond_a, (1.0 - tie) * (1.0 - cond_a), tie], axis=-1)
    count = mask.sum(axis=1, keepdims=True)
    probabilities = (mask[:, :, None] * per_metric).sum(axis=1) / count
    score_margins = (mask * (arrays["scores_A"] - arrays["scores_B"])).sum(axis=1) / count[:, 0]
    report = _three_way_metrics(records, probabilities.tolist(), score_margins.tolist())
    report["definition"] = (
        "P(A), P(B), P(Tie) = mean over selected metrics of (1-t_k)w_k, (1-t_k)(1-w_k), t_k; "
        "absolute-margin baseline uses mean_{k in S} d_k"
    )
    return report, probabilities


def bootstrap_intervals(labels, margins, image_scores, replicates, seed) -> Dict[str, Any]:
    """Percentile intervals over pairs resampled with replacement."""
    rng = np.random.default_rng(seed)
    winner_is_a = labels == 1
    positive = np.where(winner_is_a, image_scores[:, 0], image_scores[:, 1])
    negative = np.where(winner_is_a, image_scores[:, 1], image_scores[:, 0])
    correct = np.sign(margins) == labels
    accuracy, auc = [], []
    for _ in range(replicates):
        index = rng.integers(0, len(labels), len(labels))
        accuracy.append(correct[index].mean())
        pos, neg = positive[index], negative[index]
        greater = (pos[:, None] > neg[None, :]).sum()
        equal = (pos[:, None] == neg[None, :]).sum()
        auc.append((greater + 0.5 * equal) / (pos.size * neg.size))

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return [round(100.0 * float(low), 2), round(100.0 * float(high), 2)]

    return {
        "replicates": replicates,
        "sampling_unit": "pair",
        "accuracy_percent_95": interval(accuracy),
        "pooled_auc_percent_95": interval(auc),
    }


def evaluate_selected_equal(records, arrays, mask, args):
    """Non-tie pairs: margin = mean of signed_evidence over the selected metrics (AUC-Margin.md section 2)."""
    weights = mask.astype(np.float64)
    labels = np.asarray([r.label for r in records])
    normalizer = weights.sum(axis=1)
    margins = (weights * arrays["signed_evidence"]).sum(axis=1) / normalizer
    image_scores = np.stack(
        [(weights * arrays["scores_A"]).sum(axis=1) / normalizer,
         (weights * arrays["scores_B"]).sum(axis=1) / normalizer],
        axis=1,
    )
    correct = int((np.sign(margins) == labels).sum())
    report: Dict[str, Any] = {
        "accuracy": {
            "scope": "non_tie_samples",
            "correct": correct,
            "total": int(labels.size),
            "value": correct / labels.size,
            "percent": round(100.0 * correct / labels.size, 2),
        },
        "pooled_auc": _pooled_auc(records, image_scores.tolist()),
    }
    report["pooled_auc"]["definition"] = (
        "ROC AUC of sum_k w_k * score_k(image) / sum_k w_k over the selected metrics, "
        "winner images positive vs loser images negative, pooled"
    )
    report["bootstrap"] = bootstrap_intervals(
        labels, margins, image_scores, args.bootstrap_replicates, args.seed
    )
    report["margin_auc"] = _margin_auc(records, margins.tolist())
    report["selection"] = {
        "rule": f"relevance_probs >= {RELEVANCE_THRESHOLD}, else the most relevant metric",
        "selected_metric_fraction": mask.mean(axis=0).round(4).tolist(),
        "mean_selected_metrics": round(float(mask.sum(axis=1).mean()), 4),
    }
    return report, margins, image_scores


def evaluate_records(all_records, all_arrays, args):
    """Metrics and per-pair rows of EditReward-Test 1k5 from the model outputs (no model needed)."""
    if not any(r.label != 0 for r in all_records):
        raise ValueError("The Test set has no non-tie pair.")
    metrics: Dict[str, Any] = {}
    all_mask = selection_mask(all_arrays["relevance_probs"])
    three_way_probs = None
    if any(r.label == 0 for r in all_records):
        three_way, three_way_probs = selected_equal_three_way(all_records, all_arrays, all_mask)
        metrics["selected_equal_three_way"] = three_way
        print(
            f"[selected_equal three-way] three_class={three_way['three_class_accuracy_percent']}%, "
            f"tie_auc(t)={three_way['tie_from_tie_head'].get('roc_auc_percent')}%, "
            f"tie_auc(|d|)={three_way['tie_from_absolute_margin'].get('roc_auc_percent')}%",
            flush=True,
        )
    keep = np.asarray([r.label != 0 for r in all_records])
    records = [r for r, kept in zip(all_records, keep) if kept]
    arrays = {field: array[keep] for field, array in all_arrays.items()}
    mask = all_mask[keep]
    report, margins, image_scores = evaluate_selected_equal(records, arrays, mask, args)
    metrics["selected_equal"] = report
    print(
        f"[selected_equal] accuracy={report['accuracy']['percent']}% "
        f"({report['accuracy']['correct']}/{report['accuracy']['total']}), "
        f"margin_auc={report['margin_auc']['percent']}%",
        flush=True,
    )
    rows = []
    for i, r in enumerate(records):
        rows.append({
            "sample_id": r.sample_id, "label": r.label,
            "source_image": r.source_image, "edit_instruction": r.edit_instruction,
            "image_a": r.image_a, "image_b": r.image_b,
            **{field: arrays[field][i].tolist() for field in arrays},
            "per_metric_margin": (arrays["scores_A"][i] - arrays["scores_B"][i]).tolist(),
            "selected_metrics": [METRIC_NAMES[k] for k in np.flatnonzero(mask[i])],
            "selected_equal_margin": float(margins[i]),
            "selected_equal_image_scores": image_scores[i].tolist(),
        })
    for i, r in enumerate(all_records):
        if r.label != 0:
            continue
        rows.append({
            "sample_id": r.sample_id, "label": 0,
            "source_image": r.source_image, "edit_instruction": r.edit_instruction,
            "image_a": r.image_a, "image_b": r.image_b,
            **{field: all_arrays[field][i].tolist() for field in all_arrays},
            "selected_metrics": [METRIC_NAMES[k] for k in np.flatnonzero(all_mask[i])],
            "selected_equal_three_way_probabilities": three_way_probs[i].tolist(),
        })
    return metrics, rows


def main(argv: Sequence[str] | None = None) -> None:
    import torch

    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = output_dir / evaluation_file_name(args.role)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output_path}")
    for path in (checkpoint_path, Path(args.test_dataset), Path(args.human_dataset)):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    records = load_test_records(args)
    print({TEST_DATASET: len(records)}, flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    prompt = checkpoint.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Checkpoint is missing the prompt contract.")
    device = torch.device(args.device)
    model, processor = load_refined_model(
        checkpoint, args.role, device, args.min_pixels, args.max_pixels,
        base_model_override=Path(args.base_model_path).expanduser().resolve(),
    )

    print(f"Evaluating {TEST_DATASET} ({len(records)} pairs), role {args.role}", flush=True)
    arrays = extract_outputs(model, processor, prompt, records, args.batch_size, device)
    dataset_metrics, rows = evaluate_records(records, arrays, args)

    result = {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": args.role,
        "checkpoint": str(checkpoint_path),
        "adapter_dir": None,
        "ms_swift_best_checkpoint": checkpoint.get("ms_swift_best_checkpoint"),
        "selection": checkpoint.get("selection"),
        "preference_aggregator": None,
        "tie_head_mode": checkpoint.get("tie_head_mode", "independent"),
        "gemma_validation_tie": None,
        "metric_order": list(METRIC_NAMES),
        "dropout_enabled": False,
        "configuration": vars(args),
        "metrics": {TEST_DATASET: dataset_metrics},
        "predictions": {TEST_DATASET: rows},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Evaluation output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
