#!/usr/bin/env python3
"""Evaluate the Overall Inspector or a refined six-metric Student/Teacher."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, PeftConfig, set_peft_model_state_dict
from swift.tuners import Swift
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parents[1]
_PROJECT_ROOT = _WORKSPACE / "TrainingRewardModel" / "Qwen3-VL-Embedding"
_SCRIPTS = _PROJECT_ROOT / "scripts"
for path in (_PROJECT_ROOT, _SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from infer_overall_inspector import infer_batch, load_frozen_inspector  # noqa: E402
from train_loop import PairRecord, forward_pairs  # noqa: E402
from src.models.custom_qwen3_vl_reranker import (  # noqa: E402
    METRIC_NAMES,
    NUM_METRICS,
    install_multi_metric_heads,
    tie_head_mode_from_state_keys,
)


REFINEMENT_ROLE = "six_metric_refinement_student_teacher"
SUPERVISED_CONTROL_ROLE = "six_metric_supervised_control_student"
INSPECTOR_ROLE = "overall_inspector"
STUDENT_WARMSTART_ROLE = "six_metric_student_warmstart"
TEACHER_INITIALIZED_ROLE = "six_metric_ema_teacher_initialized"


@dataclass(frozen=True)
class BenchmarkRecord:
    sample_id: str
    source_image: str
    edit_instruction: str
    image_a: str
    image_b: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inspector", "student", "teacher"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preference-aggregator", required=True)
    parser.add_argument("--edithf-test", required=True)
    parser.add_argument("--editreward-bench", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Local copy of the checkpoint's base model; checkpoint weights remain unchanged.",
    )
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def required_input(raw_path: str, description: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def required_output(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {path}")
    return path


def parse_ranking(value: Any) -> int:
    normalized = str(value).strip().upper().replace(" ", "")
    if normalized == "A>B":
        return 1
    if normalized in {"B>A", "A<B"}:
        return -1
    if normalized in {"A=B", "TIE"}:
        return 0
    raise ValueError(f"Unsupported ranking label: {value!r}")


def resolve_image(raw_path: Any, sample_id: str, field: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Sample {sample_id}: missing {field}")
    value = raw_path.strip()
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sample {sample_id}: {field} not found: {path}")
    return str(path)


def load_benchmark(path: Path) -> List[BenchmarkRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Benchmark must be a non-empty JSON list: {path}")
    records: List[BenchmarkRecord] = []
    seen = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} row {index}: expected an object")
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{path.name} row {index}: missing sample_id")
        if sample_id in seen:
            raise ValueError(f"{path.name}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
        instruction = row.get("edit_instruction", row.get("instruction"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Sample {sample_id}: missing edit instruction")
        records.append(
            BenchmarkRecord(
                sample_id=sample_id,
                source_image=resolve_image(row.get("source_image"), sample_id, "source_image"),
                edit_instruction=instruction.strip(),
                image_a=resolve_image(row.get("candidate_1"), sample_id, "candidate_1"),
                image_b=resolve_image(row.get("candidate_2"), sample_id, "candidate_2"),
                label=parse_ranking(row.get("metadata_ranking", row.get("label"))),
            )
        )
    return records


def batches(records: Sequence[BenchmarkRecord], size: int) -> Iterable[Sequence[BenchmarkRecord]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def tie_ranking_metrics(labels: np.ndarray, confidence: np.ndarray) -> Dict[str, Any]:
    if labels.ndim != 1 or labels.shape != confidence.shape:
        raise ValueError("Tie metrics require aligned one-dimensional arrays")
    if not np.isfinite(confidence).all():
        raise ValueError("Tie confidence contains a non-finite value")
    positives = labels == 0
    num_positive = int(positives.sum())
    num_negative = int((~positives).sum())
    result: Dict[str, Any] = {
        "tie_samples": num_positive,
        "non_tie_samples": num_negative,
        "random_ap_baseline": float(num_positive / len(labels)),
    }
    if num_positive == 0 or num_negative == 0:
        result.update(
            {
                "evaluable": False,
                "reason": "AP and ROC-AUC require both tie and non-tie samples",
            }
        )
        return result

    order = np.argsort(-confidence, kind="mergesort")
    sorted_scores = confidence[order]
    sorted_targets = positives[order]
    cumulative_positive = 0
    average_precision = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_positive = int(sorted_targets[start:end].sum())
        cumulative_positive += group_positive
        if group_positive:
            average_precision += (
                group_positive / num_positive
            ) * (cumulative_positive / end)
        start = end

    positive_scores = confidence[positives]
    negative_scores = confidence[~positives]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    roc_auc = float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
        / comparisons.size
    )
    result.update(
        {
            "evaluable": True,
            "average_precision": float(average_precision),
            "average_precision_percent": round(100.0 * average_precision, 2),
            "roc_auc": roc_auc,
            "roc_auc_percent": round(100.0 * roc_auc, 2),
        }
    )
    return result


def summarize(
    records: Sequence[BenchmarkRecord],
    margins: Sequence[float],
    native_tie_confidence: Sequence[float] | None,
) -> Dict[str, Any]:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    margin_array = np.asarray(margins, dtype=np.float64)
    if margin_array.shape != labels.shape or not np.isfinite(margin_array).all():
        raise ValueError("Model margins are invalid")
    non_tie = labels != 0
    correct = ((margin_array > 0) & (labels == 1)) | (
        (margin_array < 0) & (labels == -1)
    )
    correct_count = int((correct & non_tie).sum())
    total = int(non_tie.sum())
    report: Dict[str, Any] = {
        "total_samples": len(records),
        "non_tie_samples": total,
        "tie_samples": int((labels == 0).sum()),
        "accuracy": {
            "scope": "non_tie_samples",
            "correct": correct_count,
            "total": total,
            "value": float(correct_count / total) if total else None,
            "percent": round(100.0 * correct_count / total, 2) if total else None,
        },
        "tie_from_absolute_margin": {
            "confidence_definition": "-|overall_margin|",
            **tie_ranking_metrics(labels, -np.abs(margin_array)),
        },
    }
    if native_tie_confidence is not None:
        native = np.asarray(native_tie_confidence, dtype=np.float64)
        if native.shape != labels.shape:
            raise ValueError("Native tie confidence shape is invalid")
        report["tie_from_tie_head"] = {
            "confidence_definition": "sum_k aggregator_weight[k] * tie_prob[k]",
            **tie_ranking_metrics(labels, native),
        }
    return report


def _build_peft_config(config_dict: Mapping[str, Any]) -> PeftConfig:
    supported = set(inspect.signature(LoraConfig).parameters)
    filtered = {key: value for key, value in config_dict.items() if key in supported}
    filtered["inference_mode"] = False
    return PeftConfig.from_peft_type(**filtered)


def load_refined_model(
    checkpoint: Mapping[str, Any],
    role: str,
    device: torch.device,
    min_pixels: int | None,
    max_pixels: int | None,
    base_model_override: Path | None = None,
):
    checkpoint_role = checkpoint.get("model_role")
    expected_role = (
        (REFINEMENT_ROLE, STUDENT_WARMSTART_ROLE, SUPERVISED_CONTROL_ROLE)
        if role == "student"
        else (REFINEMENT_ROLE, TEACHER_INITIALIZED_ROLE)
    )
    if checkpoint_role not in expected_role:
        raise ValueError(f"Checkpoint role {checkpoint_role!r} is incompatible with {role}")
    if list(checkpoint.get("metric_order", [])) != list(METRIC_NAMES):
        raise ValueError("Checkpoint metric order is incompatible")
    state_key = (
        f"{role}_adapter_state_dict"
        if checkpoint_role in {REFINEMENT_ROLE, SUPERVISED_CONTROL_ROLE}
        else "adapter_state_dict"
    )
    state = checkpoint.get(state_key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint is missing {state_key}")
    base_path = (
        str(base_model_override)
        if base_model_override is not None else checkpoint.get("base_model_path")
    )
    if not isinstance(base_path, str) or not Path(base_path).is_dir():
        raise FileNotFoundError(f"Base model is unavailable: {base_path}")
    pixel_kwargs = {
        key: value
        for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels))
        if value is not None
    }
    processor = AutoProcessor.from_pretrained(
        base_path,
        trust_remote_code=True,
        padding_side="left",
        **pixel_kwargs,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Head structure follows the saved weights (independent or margin-coupled Tie Head).
    install_multi_metric_heads(
        base_model, processor.tokenizer, tie_head_mode=tie_head_mode_from_state_keys(state)
    )
    model = Swift.prepare_model(
        base_model, _build_peft_config(dict(checkpoint["peft_config"]))
    )
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys[:10]}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    if any(module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)):
        raise RuntimeError("Dropout must be disabled during evaluation")
    return model, processor


def read_preference_aggregator(
    checkpoint: Mapping[str, Any], aggregator_path: Path
) -> Tuple[torch.Tensor, float]:
    aggregator_file = json.loads(aggregator_path.read_text(encoding="utf-8"))
    embedded = checkpoint.get("preference_aggregator")
    file_weights = np.asarray(aggregator_file.get("weights"), dtype=np.float64)
    file_beta = float(aggregator_file.get("beta"))
    if (
        file_weights.shape != (NUM_METRICS,)
        or not np.isfinite(file_weights).all()
        or np.any(file_weights < 0)
        or not math.isclose(float(file_weights.sum()), 1.0, abs_tol=1e-6)
        or not math.isfinite(file_beta)
        or file_beta <= 0
    ):
        raise ValueError("Preference Aggregator file has invalid weights or beta")
    if checkpoint.get("model_role") in {REFINEMENT_ROLE, SUPERVISED_CONTROL_ROLE}:
        if not isinstance(embedded, dict):
            raise ValueError("Refinement checkpoint is missing its frozen Preference Aggregator")
        embedded_weights = np.asarray(embedded.get("weights"), dtype=np.float64)
        embedded_beta = float(embedded.get("beta"))
        if not np.allclose(file_weights, embedded_weights, atol=1e-7, rtol=0.0):
            raise ValueError("Preference Aggregator file and refinement checkpoint weights differ")
        if not math.isclose(file_beta, embedded_beta, abs_tol=1e-7, rel_tol=0.0):
            raise ValueError("Preference Aggregator file and refinement checkpoint beta differ")
    return torch.tensor(file_weights, dtype=torch.float32), file_beta


@torch.inference_mode()
def evaluate_inspector_dataset(
    model,
    processor,
    checkpoint: Mapping[str, Any],
    records: Sequence[BenchmarkRecord],
    batch_size: int,
    device: torch.device,
) -> Tuple[List[float], None, List[Dict[str, Any]]]:
    margins: List[float] = []
    predictions: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(batches(records, batch_size), start=1):
        compatible = [
            {
                "sample_id": record.sample_id,
                "source_image": record.source_image,
                "edit_instruction": record.edit_instruction,
                "edited_image_A": record.image_a,
                "edited_image_B": record.image_b,
            }
            for record in batch
        ]
        output = infer_batch(
            model, processor, compatible, dict(checkpoint), device, temperature=1.0
        )
        for record, prediction in zip(batch, output):
            margin = float(prediction["margin_A_minus_B"])
            margins.append(margin)
            predictions.append(
                {
                    "sample_id": record.sample_id,
                    "label": record.label,
                    "overall_margin": margin,
                    "probability_A": float(prediction["probability_A"]),
                }
            )
        if batch_index % 25 == 0 or batch_index * batch_size >= len(records):
            print(
                f"Inspector evaluated {min(batch_index * batch_size, len(records))}/{len(records)}",
                flush=True,
            )
    return margins, None, predictions


@torch.inference_mode()
def evaluate_metric_dataset(
    model,
    processor,
    checkpoint: Mapping[str, Any],
    records: Sequence[BenchmarkRecord],
    batch_size: int,
    device: torch.device,
    aggregator_weights: torch.Tensor,
    aggregator_beta: float,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    margins: List[float] = []
    tie_confidences: List[float] = []
    predictions: List[Dict[str, Any]] = []
    prompt = checkpoint.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Refinement checkpoint is missing the prompt contract")
    weights = aggregator_weights.to(device)
    for batch_index, batch in enumerate(batches(records, batch_size), start=1):
        pair_records = [
            PairRecord(
                sample_id=record.sample_id,
                source_image=record.source_image,
                edit_instruction=record.edit_instruction,
                image_a=record.image_a,
                image_b=record.image_b,
            )
            for record in batch
        ]
        output = forward_pairs(model, processor, pair_records, prompt, device)
        aggregator_margin = aggregator_beta * (
            output.signed_evidence * weights
        ).sum(dim=-1)
        tie_confidence = (output.tie_probs * weights).sum(dim=-1)
        for row, record in enumerate(batch):
            margin = float(aggregator_margin[row].float().cpu())
            tie_score = float(tie_confidence[row].float().cpu())
            margins.append(margin)
            tie_confidences.append(tie_score)
            predictions.append(
                {
                    "sample_id": record.sample_id,
                    "label": record.label,
                    "overall_margin": margin,
                    "probability_A": float(
                        torch.sigmoid(aggregator_margin[row]).float().cpu()
                    ),
                    "tie_head_confidence": tie_score,
                    "signed_evidence": output.signed_evidence[row].float().cpu().tolist(),
                    "tie_probs": output.tie_probs[row].float().cpu().tolist(),
                    "relevance_probs": output.relevance_probs[row].float().cpu().tolist(),
                }
            )
        if batch_index % 25 == 0 or batch_index * batch_size >= len(records):
            print(
                f"Aspect model evaluated {min(batch_index * batch_size, len(records))}/{len(records)}",
                flush=True,
            )
    return margins, tie_confidences, predictions


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint_path = required_input(args.checkpoint, "Checkpoint")
    aggregator_path = required_input(
        args.preference_aggregator, "Preference Aggregator"
    )
    edithf_path = required_input(args.edithf_test, "EditHF-Test")
    editreward_path = required_input(args.editreward_bench, "EditReward-Bench")
    output_path = required_output(args.output)
    base_model_override = (
        Path(args.base_model_path).expanduser().resolve()
        if args.base_model_path is not None else None
    )
    if base_model_override is not None and not base_model_override.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {base_model_override}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")

    datasets = {
        "EditHF-Test": load_benchmark(edithf_path),
        "EditReward-Bench": load_benchmark(editreward_path),
    }
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if args.role == "inspector":
        if checkpoint.get("model_role") != INSPECTOR_ROLE:
            raise ValueError("Checkpoint is not an Overall Inspector checkpoint")
        model, processor, checkpoint = load_frozen_inspector(
            checkpoint_path, device, args.min_pixels, args.max_pixels,
            base_model_override=base_model_override,
        )
        evaluator = lambda records: evaluate_inspector_dataset(
            model, processor, checkpoint, records, args.batch_size, device
        )
        score_definition = "raw scalar score_A - raw scalar score_B"
    else:
        model, processor = load_refined_model(
            checkpoint, args.role, device, args.min_pixels, args.max_pixels,
            base_model_override=base_model_override,
        )
        aggregator_weights, aggregator_beta = read_preference_aggregator(
            checkpoint, aggregator_path
        )
        evaluator = lambda records: evaluate_metric_dataset(
            model,
            processor,
            checkpoint,
            records,
            args.batch_size,
            device,
            aggregator_weights,
            aggregator_beta,
        )
        score_definition = "beta * sum_k aggregator_weight[k] * signed_evidence[k]"

    reports: Dict[str, Any] = {}
    predictions: Dict[str, Any] = {}
    for dataset_name, records in datasets.items():
        print(f"Evaluating {args.role} on {dataset_name} ({len(records)} samples)", flush=True)
        margins, native_ties, details = evaluator(records)
        reports[dataset_name] = summarize(records, margins, native_ties)
        predictions[dataset_name] = details
        print(
            f"{dataset_name}: accuracy={reports[dataset_name]['accuracy']['percent']}%",
            flush=True,
        )

    result = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": args.role,
        "checkpoint_model_role": checkpoint.get("model_role"),
        "base_model_loaded_from": str(base_model_override or checkpoint.get("base_model_path")),
        "checkpoint": str(checkpoint_path),
        "score_definition": score_definition,
        "dropout_enabled": False,
        "metric_order": list(METRIC_NAMES) if args.role != "inspector" else None,
        "datasets": {
            "EditHF-Test": str(edithf_path),
            "EditReward-Bench": str(editreward_path),
        },
        "metrics": reports,
        "predictions": predictions,
        "configuration": vars(args),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Evaluation output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
