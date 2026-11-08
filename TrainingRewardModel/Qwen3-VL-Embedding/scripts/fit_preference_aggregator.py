#!/usr/bin/env python3
"""Extract Student evidence from HumanGT Data and fit Preference Aggregator."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, PeftConfig, get_peft_model, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from scipy.optimize import minimize
from scipy.special import expit
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.custom_qwen3_vl_reranker import (  # noqa: E402
    METRIC_NAMES,
    _pool_qwen_last_hidden,
    install_multi_metric_heads,
    tie_head_mode_from_state_keys,
)


DEFAULT_EDITREWARD_ROOT = (
    "/datastore/shared/datasets/EditReward-Data"
)
STUDENT_ROLE = "six_metric_student_warmstart"
VOTE_TO_LABEL = {"leftvote": 1.0, "rightvote": 0.0}
TIE_VOTES = {"tie", "tievote"}


@dataclass(frozen=True)
class HumanRecord:
    sample_id: str
    source_image: str
    edit_instruction: str
    image_a: str
    image_b: str
    label_a: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--feature-definition",
        choices=("signed_evidence", "relevance_weighted_signed_evidence"),
        default="signed_evidence",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--l2-penalty", type=float, default=0.01)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument(
        "--shared-split",
        default=None,
        help="Split shared by every pipeline stage (scripts/backups/create_shared_source_split.py, no longer used); "
        "overrides --validation-ratio.",
    )
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: Dict[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: missing or empty {field!r}.")
    return value.strip()


def resolve_image_path(raw_path: str, dataset_root: Path) -> str:
    value = raw_path[7:] if raw_path.startswith("file://") else raw_path
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Image does not exist: {resolved}")
        return str(resolved)
    candidates = [dataset_root / path]
    if (dataset_root / "extracted_data").is_dir():
        candidates.append(dataset_root / "extracted_data" / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"Relative image path escapes dataset root: {raw_path}") from exc
        if resolved.is_file():
            return str(resolved)
    raise FileNotFoundError(
        f"Image does not exist below dataset root {dataset_root}: {raw_path}"
    )


def load_human_records(dataset_path: Path, dataset_root: Path) -> List[HumanRecord]:
    records: List[HumanRecord] = []
    seen_ids = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            vote = _required_text(row, "vote_type", line_number).lower()
            if vote in TIE_VOTES:
                continue
            if vote not in VOTE_TO_LABEL:
                raise ValueError(f"Line {line_number}: unsupported vote_type {vote!r}.")
            sample_id = _required_text(row, "unique_key", line_number)
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate non-tie sample ID: {sample_id}")
            seen_ids.add(sample_id)
            source_image = resolve_image_path(
                _required_text(row, "source_img", line_number), dataset_root
            )
            image_a = resolve_image_path(
                _required_text(row, "left_output_image", line_number), dataset_root
            )
            image_b = resolve_image_path(
                _required_text(row, "right_output_image", line_number), dataset_root
            )
            if image_a == image_b:
                raise ValueError(f"Line {line_number}: candidates A and B are identical paths.")
            records.append(
                HumanRecord(
                    sample_id=sample_id,
                    source_image=source_image,
                    edit_instruction=_required_text(row, "instruction", line_number),
                    image_a=image_a,
                    image_b=image_b,
                    label_a=VOTE_TO_LABEL[vote],
                )
            )
    if not records:
        raise ValueError("No non-tie human preference records were found.")
    return records


def source_grouped_split(
    records: Sequence[HumanRecord],
    validation_ratio: float,
    seed: int,
) -> Tuple[List[HumanRecord], List[HumanRecord]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be strictly between 0 and 1.")
    grouped: Dict[str, List[HumanRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_image].append(record)
    source_groups = sorted(grouped)
    random.Random(seed).shuffle(source_groups)
    validation_count = max(1, round(len(source_groups) * validation_ratio))
    if validation_count >= len(source_groups):
        raise RuntimeError("Not enough source groups for Preference Aggregator split.")
    validation_sources = set(source_groups[:validation_count])
    train = [r for r in records if r.source_image not in validation_sources]
    validation = [r for r in records if r.source_image in validation_sources]
    if not train or not validation:
        raise ValueError("Preference Aggregator split produced an empty partition.")
    train_sources = {record.source_image for record in train}
    validation_sources = {record.source_image for record in validation}
    if train_sources & validation_sources:
        raise ValueError("Source leakage exists in the Preference Aggregator split.")
    for name, partition in (("train", train), ("validation", validation)):
        if {record.label_a for record in partition} != {0.0, 1.0}:
            raise ValueError(f"{name} split must contain both preference classes.")
    return train, validation


def shared_source_split(
    records: Sequence[HumanRecord],
    split_path: Path,
) -> Tuple[List[HumanRecord], List[HumanRecord], Dict[str, Any]]:
    """Fit on the shared train partition, report on the shared validation partition.

    The calibration partition is reserved for the Overall Inspector Model.
    """
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    partition = payload.get("sample_partition")
    if not isinstance(partition, dict) or not partition:
        raise ValueError(f"Shared split has no sample_partition: {split_path}")
    missing = [r.sample_id for r in records if r.sample_id not in partition]
    if missing:
        raise ValueError(
            f"{len(missing)} HumanGT samples are missing from the shared split."
        )
    train = [r for r in records if partition[r.sample_id] == "train"]
    validation = [r for r in records if partition[r.sample_id] == "validation"]
    if {r.source_image for r in train} & {r.source_image for r in validation}:
        raise ValueError("Source leakage exists in the shared split.")
    for name, part in (("train", train), ("validation", validation)):
        if {record.label_a for record in part} != {0.0, 1.0}:
            raise ValueError(f"{name} split must contain both preference classes.")
    metadata = {
        "path": str(split_path),
        "split_method": payload.get("split_method"),
        "seed": payload.get("seed"),
        "validation_ratio": payload.get("validation_ratio"),
        "calibration_ratio": payload.get("calibration_ratio"),
    }
    return train, validation, metadata


def _build_peft_config(config_dict: Dict[str, Any]) -> PeftConfig:
    supported = set(inspect.signature(LoraConfig).parameters)
    filtered = {key: value for key, value in config_dict.items() if key in supported}
    filtered["inference_mode"] = True
    return PeftConfig.from_peft_type(**filtered)


def load_student_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
    min_pixels: int | None,
    max_pixels: int | None,
):
    if checkpoint.get("model_role") != STUDENT_ROLE:
        raise ValueError("The supplied Student checkpoint has an incompatible role.")
    if list(checkpoint.get("metric_order", [])) != list(METRIC_NAMES):
        raise ValueError("Student checkpoint metric order is incompatible.")
    base_model_path = checkpoint.get("base_model_path")
    if not isinstance(base_model_path, str) or not Path(base_model_path).is_dir():
        raise FileNotFoundError(f"Student base model is unavailable: {base_model_path}")
    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Student checkpoint contains no adapter weights.")
    if not any("metric_heads" in key for key in state):
        raise ValueError("Student checkpoint contains no metric-head weights.")

    pixel_kwargs = {
        key: value
        for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels))
        if value is not None
    }
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
        **pixel_kwargs,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Head structure follows the saved weights (independent or margin-coupled Tie Head).
    install_multi_metric_heads(
        base_model, processor.tokenizer, tie_head_mode=tie_head_mode_from_state_keys(state)
    )
    model = get_peft_model(base_model, _build_peft_config(checkpoint["peft_config"]))
    load_result = set_peft_model_state_dict(model, state, adapter_name="default")
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Student checkpoint keys: {load_result.unexpected_keys[:10]}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, processor


def _format_candidate(
    record: HumanRecord,
    candidate_image: str,
    system_message: str,
    evaluation_instruction: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<Instruct>: {evaluation_instruction}\n<Query>:",
                },
                {"type": "image", "image": f"file://{record.source_image}"},
                {
                    "type": "text",
                    "text": f" {record.edit_instruction}\n<Document>:",
                },
                {"type": "image", "image": f"file://{candidate_image}"},
            ],
        },
    ]


def _format_source(
    record: HumanRecord,
    system_message: str,
    evaluation_instruction: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<Instruct>: {evaluation_instruction}\n<Query>:",
                },
                {"type": "image", "image": f"file://{record.source_image}"},
                {"type": "text", "text": f" {record.edit_instruction}"},
            ],
        },
    ]


def batched(records: Sequence[HumanRecord], batch_size: int) -> Iterable[Sequence[HumanRecord]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


@torch.inference_mode()
def extract_features(
    model,
    processor,
    records: Sequence[HumanRecord],
    prompt: Dict[str, Any],
    feature_definition: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    system_message = prompt.get("system_message")
    evaluation_instruction = prompt.get("evaluation_instruction")
    if not isinstance(system_message, str) or not isinstance(
        evaluation_instruction, str
    ):
        raise ValueError("Student checkpoint is missing its prompt text.")
    feature_batches: List[np.ndarray] = []
    labels: List[float] = []
    processed = 0
    for batch in batched(records, batch_size):
        conversations: List[List[Dict[str, Any]]] = []
        for record in batch:
            conversations.extend(
                [
                    _format_candidate(
                        record,
                        record.image_a,
                        system_message,
                        evaluation_instruction,
                    ),
                    _format_candidate(
                        record,
                        record.image_b,
                        system_message,
                        evaluation_instruction,
                    ),
                    _format_source(
                        record,
                        system_message,
                        evaluation_instruction,
                    ),
                ]
            )
        texts = processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hidden = _pool_qwen_last_hidden(model, inputs, inputs.get("attention_mask"))
        starts = torch.arange(len(batch), device=device, dtype=torch.long) * 3
        heads = model.get_base_model().metric_heads
        output = heads(hidden[starts], hidden[starts + 1], hidden[starts + 2])
        features = output.signed_evidence
        if feature_definition == "relevance_weighted_signed_evidence":
            features = output.relevance_probs * features
        feature_batches.append(features.float().cpu().numpy().astype(np.float64))
        labels.extend(record.label_a for record in batch)
        processed += len(batch)
        print(f"Extracted {processed}/{len(records)}", flush=True)
    matrix = np.concatenate(feature_batches, axis=0)
    target = np.asarray(labels, dtype=np.float64)
    if matrix.shape != (len(records), len(METRIC_NAMES)):
        raise RuntimeError(f"Unexpected Preference Aggregator feature shape: {matrix.shape}")
    if not np.isfinite(matrix).all() or np.any(matrix < -1.0) or np.any(matrix > 1.0):
        raise RuntimeError("Preference Aggregator features must be finite and in [-1, 1].")
    if not np.isin(target, (0.0, 1.0)).all():
        raise RuntimeError("Preference Aggregator labels must contain only 0 and 1.")
    return matrix, target


def logistic_objective(
    gamma: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    l2_penalty: float,
) -> Tuple[float, np.ndarray]:
    logits = features @ gamma
    loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
    gradient = features.T @ (expit(logits) - labels) / labels.size
    if l2_penalty:
        loss += 0.5 * l2_penalty * float(gamma @ gamma)
        gradient = gradient + l2_penalty * gamma
    return float(loss), gradient.astype(np.float64, copy=False)


def fit_nonnegative_aggregator(
    features: np.ndarray,
    labels: np.ndarray,
    l2_penalty: float,
    max_iterations: int,
    initial_gamma: np.ndarray | None = None,
) -> Dict[str, Any]:
    if features.ndim != 2 or features.shape[1] != len(METRIC_NAMES):
        raise ValueError("Preference Aggregator features must have shape [N, 6].")
    if labels.shape != (features.shape[0],):
        raise ValueError("Preference Aggregator labels must have shape [N].")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Preference Aggregator fitting requires both human preference classes.")
    start = (
        np.full(len(METRIC_NAMES), 1.0 / len(METRIC_NAMES), dtype=np.float64)
        if initial_gamma is None
        else np.asarray(initial_gamma, dtype=np.float64)
    )
    result = minimize(
        fun=lambda gamma: logistic_objective(
            gamma, features, labels, l2_penalty
        ),
        x0=start,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(METRIC_NAMES),
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-9},
    )
    gamma = np.asarray(result.x, dtype=np.float64)
    if not result.success or not np.isfinite(gamma).all() or np.any(gamma < -1e-10):
        raise RuntimeError(
            f"Preference Aggregator optimization failed: success={result.success}, message={result.message}"
        )
    gamma = np.maximum(gamma, 0.0)
    beta = float(gamma.sum())
    if not math.isfinite(beta) or beta <= 1e-10:
        raise RuntimeError("Preference Aggregator beta is zero; no usable human-preference signal was fitted.")
    weights = gamma / beta
    return {
        "gamma": gamma,
        "weights": weights,
        "beta": beta,
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "gradient_max_abs": float(np.abs(result.jac).max()),
        "optimizer_message": str(result.message),
    }


def binary_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    gamma: np.ndarray,
) -> Dict[str, float]:
    logits = features @ gamma
    probabilities = expit(logits)
    predictions = probabilities >= 0.5
    eps = np.finfo(np.float64).eps
    bce = -np.mean(
        labels * np.log(np.clip(probabilities, eps, 1.0))
        + (1.0 - labels) * np.log(np.clip(1.0 - probabilities, eps, 1.0))
    )
    class_accuracy = []
    for label in (0.0, 1.0):
        mask = labels == label
        class_accuracy.append(float(np.mean(predictions[mask] == bool(label))))
    swap_error = float(np.max(np.abs(expit(-logits) - (1.0 - probabilities))))
    return {
        "samples": int(labels.size),
        "bce": float(bce),
        "accuracy": float(np.mean(predictions == labels.astype(bool))),
        "balanced_accuracy": float(np.mean(class_accuracy)),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "mean_probability_A": float(np.mean(probabilities)),
        "swap_probability_max_error": swap_error,
    }


def fit_equal_weight_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    max_iterations: int,
) -> Dict[str, Any]:
    equal_weights = np.full(len(METRIC_NAMES), 1.0 / len(METRIC_NAMES))
    aggregate = features @ equal_weights

    def objective(beta_array: np.ndarray) -> Tuple[float, np.ndarray]:
        beta = float(beta_array[0])
        logits = beta * aggregate
        loss = np.mean(np.logaddexp(0.0, logits) - labels * logits)
        gradient = np.asarray(
            [np.mean((expit(logits) - labels) * aggregate)], dtype=np.float64
        )
        return float(loss), gradient

    result = minimize(
        objective,
        x0=np.asarray([1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)],
        options={"maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(f"Equal-weight baseline fitting failed: {result.message}")
    beta = float(result.x[0])
    gamma = beta * equal_weights
    return {"beta": beta, "weights": equal_weights, "gamma": gamma}


def bootstrap_by_source(
    records: Sequence[HumanRecord],
    features: np.ndarray,
    labels: np.ndarray,
    replicates: int,
    seed: int,
    l2_penalty: float,
    max_iterations: int,
    initial_gamma: np.ndarray,
) -> Dict[str, Any]:
    if replicates == 0:
        return {"replicates_requested": 0, "replicates_completed": 0}
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.source_image].append(index)
    groups = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    failed = 0
    for replicate in range(replicates):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.asarray(
            [index for group in sampled_groups for index in grouped[str(group)]],
            dtype=np.int64,
        )
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            failed += 1
            continue
        try:
            fitted = fit_nonnegative_aggregator(
                features[indices],
                sampled_labels,
                l2_penalty,
                max_iterations,
                initial_gamma=initial_gamma,
            )
        except RuntimeError:
            failed += 1
            continue
        estimates.append(
            {
                "replicate": replicate,
                "weights": fitted["weights"].tolist(),
                "beta": fitted["beta"],
                "gamma": fitted["gamma"].tolist(),
            }
        )
    if not estimates:
        raise RuntimeError("Every source-grouped bootstrap fit failed.")
    weight_matrix = np.asarray([item["weights"] for item in estimates])
    beta_values = np.asarray([item["beta"] for item in estimates])
    return {
        "replicates_requested": replicates,
        "replicates_completed": len(estimates),
        "replicates_failed": failed,
        "sampling_unit": "source_image",
        "seed": seed,
        "weights_mean": weight_matrix.mean(axis=0).tolist(),
        "weights_std": weight_matrix.std(axis=0, ddof=0).tolist(),
        "weights_percentile_2_5": np.percentile(weight_matrix, 2.5, axis=0).tolist(),
        "weights_percentile_97_5": np.percentile(weight_matrix, 97.5, axis=0).tolist(),
        "beta_mean": float(beta_values.mean()),
        "beta_std": float(beta_values.std(ddof=0)),
        "runs": estimates,
    }


def class_counts(labels: np.ndarray) -> Dict[str, int]:
    counts = Counter(int(value) for value in labels)
    return {"B_wins": counts[0], "A_wins": counts[1]}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative.")
    if not math.isfinite(args.l2_penalty) or args.l2_penalty < 0.0:
        raise ValueError("--l2-penalty must be finite and non-negative.")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be strictly between 0 and 1.")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")

    student_path = Path(args.student_checkpoint).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for description, path in (
        ("Student checkpoint", student_path),
        ("HumanGT Data", dataset_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Preference Aggregator output: {output_path}"
        )

    student_checkpoint = torch.load(
        student_path, map_location="cpu", weights_only=False
    )
    if not isinstance(student_checkpoint, dict):
        raise TypeError("Student checkpoint must contain a dictionary.")
    dataset_root = Path(DEFAULT_EDITREWARD_ROOT).expanduser().resolve()
    records = load_human_records(dataset_path, dataset_root)
    if args.shared_split:
        split_path = Path(args.shared_split).expanduser().resolve()
        train_records, validation_records, shared_split = shared_source_split(
            records, split_path
        )
        print(f"Using shared split: {split_path}")
    else:
        train_records, validation_records = source_grouped_split(
            records, args.validation_ratio, args.seed
        )
        shared_split = None
    print(
        f"Preference Aggregator split: train={len(train_records)} {class_counts(np.asarray([r.label_a for r in train_records]))}, "
        f"validation={len(validation_records)} {class_counts(np.asarray([r.label_a for r in validation_records]))}"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model, processor = load_student_model(
        student_checkpoint,
        device,
        args.min_pixels,
        args.max_pixels,
    )
    all_records = train_records + validation_records
    all_features, all_labels = extract_features(
        model,
        processor,
        all_records,
        student_checkpoint["prompt"],
        args.feature_definition,
        args.batch_size,
        device,
    )
    train_count = len(train_records)
    train_features, validation_features = (
        all_features[:train_count],
        all_features[train_count:],
    )
    train_labels, validation_labels = (
        all_labels[:train_count],
        all_labels[train_count:],
    )

    fitted = fit_nonnegative_aggregator(
        train_features,
        train_labels,
        args.l2_penalty,
        args.max_iterations,
    )
    baseline = fit_equal_weight_baseline(
        train_features, train_labels, args.max_iterations
    )
    bootstrap = bootstrap_by_source(
        train_records,
        train_features,
        train_labels,
        args.bootstrap_replicates,
        args.seed,
        args.l2_penalty,
        args.max_iterations,
        fitted["gamma"],
    )
    train_metrics = binary_metrics(train_features, train_labels, fitted["gamma"])
    validation_metrics = binary_metrics(
        validation_features, validation_labels, fitted["gamma"]
    )
    baseline_train_metrics = binary_metrics(
        train_features, train_labels, baseline["gamma"]
    )
    baseline_validation_metrics = binary_metrics(
        validation_features, validation_labels, baseline["gamma"]
    )
    weights = fitted["weights"]
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise RuntimeError("Preference Aggregator weights violate the simplex constraint.")
    if validation_metrics["swap_probability_max_error"] > 1e-12:
        raise RuntimeError("Preference Aggregator prediction violates A/B swap symmetry.")

    output = {
        "format_version": 2,
        "model_role": "preference_aggregator",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "metric_order": list(METRIC_NAMES),
        "feature_definition": args.feature_definition,
        "optimization": {
            "parameterization": "gamma_k = beta * a_k",
            "method": "L-BFGS-B",
            "constraints": "gamma_k >= 0; a_k >= 0; sum(a_k) = 1; beta > 0",
            "no_intercept": True,
            "l2_penalty": args.l2_penalty,
            "max_iterations": args.max_iterations,
            "iterations": fitted["iterations"],
            "objective": fitted["objective"],
            "gradient_max_abs": fitted["gradient_max_abs"],
            "message": fitted["optimizer_message"],
        },
        "weights": weights.tolist(),
        "beta": fitted["beta"],
        "gamma": fitted["gamma"].tolist(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "equal_weight_baseline": {
            "weights": baseline["weights"].tolist(),
            "beta": baseline["beta"],
            "train_metrics": baseline_train_metrics,
            "validation_metrics": baseline_validation_metrics,
        },
        "bootstrap": bootstrap,
        "student_checkpoint": {
            "path": str(student_path),
            "sha256": file_sha256(student_path),
        },
        "data": {
            "dataset_path": str(dataset_path),
            "dataset_root": str(dataset_root),
            "split_method": (
                shared_split["split_method"] if shared_split else "source_grouped_random_v1"
            ),
            "split_seed": shared_split["seed"] if shared_split else args.seed,
            "validation_ratio": (
                shared_split["validation_ratio"] if shared_split else args.validation_ratio
            ),
            "shared_split": shared_split,
            "ties_excluded": True,
            "train_samples": len(train_records),
            "validation_samples": len(validation_records),
            "train_source_groups": len({r.source_image for r in train_records}),
            "validation_source_groups": len(
                {r.source_image for r in validation_records}
            ),
            "train_class_counts": class_counts(train_labels),
            "validation_class_counts": class_counts(validation_labels),
        },
    }
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    print(f"Preference Aggregator output: {output_path}")
    print(f"Weights: {output['weights']}")
    print(f"Beta: {output['beta']}")
    print(f"Validation metrics: {validation_metrics}")


if __name__ == "__main__":
    main()
