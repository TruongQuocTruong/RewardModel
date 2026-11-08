#!/usr/bin/env python3
"""MS-Swift warm-start training for the six-metric Student model.

The input is GemmaGT Data with three annotation runs. Validation is a random pair-level
split, as MS-Swift ``--split_dataset_ratio`` does, so a source image can appear
in both partitions with different edit instructions and edited images. It derives
the empirical three-way targets in memory, and optimizes the structured Score,
Tie, and Relevance heads with the factorized warm-start objective.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from datasets import Dataset as HfDataset
from peft import (
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file as load_safetensors

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from swift.dataset import RowPreprocessor  # noqa: E402
from swift.loss.base import BaseLoss  # noqa: E402
from swift.loss.mapping import loss_map  # noqa: E402
from swift.pipelines import SwiftSft  # noqa: E402
from swift.template.register import TEMPLATE_MAPPING, register_template  # noqa: E402
from swift.template.template_inputs import TemplateInputs  # noqa: E402
from swift.template.templates.qwen import Qwen3VLRerankerTemplate  # noqa: E402
from swift.trainers import RerankerTrainer  # noqa: E402
from swift.trainers.trainer_factory import TrainerFactory  # noqa: E402
from swift.utils import get_logger  # noqa: E402

from src.models.custom_qwen3_vl_reranker import (  # noqa: E402
    METRIC_NAMES,
    NUM_METRICS,
    MultiMetricRewardOutput,
    FactorizedWarmStartLoss,
    _pool_qwen_last_hidden,
    install_multi_metric_heads,
    resolve_tie_head_mode,
)
from src.prompt_config import (  # noqa: E402
    DEFAULT_EVALUATION_INSTRUCTION,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_SYSTEM_MESSAGE,
)

logger = get_logger()

PAIRWISE_TEMPLATE_TYPE = "qwen3_vl_metric_warmstart_pairwise"
CHECKPOINT_FILENAME = "student_warmstart.pt"
LABEL_WIDTH = 4 * NUM_METRICS
CHOICES = ("A", "B", "Tie")

_SYSTEM_MESSAGE = {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}
_INSTRUCT_PREFIX = f"<Instruct>: {DEFAULT_EVALUATION_INSTRUCTION}\n<Query>:"


def _clone_tensor_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _restore_peft_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Restore selected PEFT weights into an already prepared Student model."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Student PEFT weights are missing or empty: {weights_path}"
        )
    if not isinstance(model, PeftModel):
        raise TypeError("Student checkpoint restoration requires a PEFT model.")
    state = load_safetensors(str(weights_path), device="cpu")
    load_result = set_peft_model_state_dict(model, state, adapter_name="default")
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            "Unexpected keys while restoring the Student checkpoint: "
            f"{unexpected[:10]}"
        )
    logger.info("Restored selected Student weights from: %s", checkpoint_dir)


def _initialize_lora_from_inspector(
    model: torch.nn.Module, checkpoint_dir: Path
) -> Dict[str, Any]:
    """Copy the Overall Inspector LLM LoRA into the leading ranks of the Student LoRA.

    A_student[:r_I] = A_I and B_student[:, :r_I] = B_I, while B_student[:, r_I:] stays
    zero, so scale_S * B_S A_S equals scale_I * B_I A_I on every Inspector module.
    Student modules without Inspector LoRA (ViT, aligner) and metric_heads keep their
    own initialization; the Inspector head is not used.
    """
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    config_path = checkpoint_dir / "adapter_config.json"
    for path in (weights_path, config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Inspector adapter file is missing or empty: {path}")
    if not isinstance(model, PeftModel):
        raise TypeError("Inspector initialization requires a PEFT Student model.")
    with config_path.open("r", encoding="utf-8") as handle:
        inspector_config = json.load(handle)
    if inspector_config.get("use_rslora") or inspector_config.get("rank_pattern") or inspector_config.get("alpha_pattern"):
        raise ValueError("Inspector adapter uses rslora or per-module rank/alpha patterns.")
    inspector_rank = int(inspector_config["r"])
    inspector_scale = float(inspector_config["lora_alpha"]) / inspector_rank

    inspector_state = load_safetensors(str(weights_path), device="cpu")
    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in inspector_state.items():
        for part in ("lora_A", "lora_B"):
            marker = f".{part}.weight"
            if key.endswith(marker):
                lora_pairs[key[: -len(marker)]][part] = value
    if not lora_pairs:
        raise RuntimeError(f"No LoRA tensors found in Inspector adapter: {weights_path}")

    modules = dict(model.named_modules())
    max_relative_error = 0.0
    with torch.no_grad():
        for prefix, pair in sorted(lora_pairs.items()):
            if set(pair) != {"lora_A", "lora_B"}:
                raise RuntimeError(f"Inspector LoRA pair is incomplete: {prefix}")
            module = modules.get(prefix)
            if module is None or not hasattr(module, "lora_A") or "default" not in module.lora_A:
                raise RuntimeError(f"Student has no LoRA module for Inspector key: {prefix}")
            student_a = module.lora_A["default"].weight
            student_b = module.lora_B["default"].weight
            student_rank = student_a.shape[0]
            student_scale = float(module.scaling["default"])
            inspector_a, inspector_b = pair["lora_A"], pair["lora_B"]
            if inspector_rank > student_rank or inspector_a.shape[0] != inspector_rank:
                raise ValueError(f"Inspector rank {inspector_rank} does not fit Student rank {student_rank}: {prefix}")
            if inspector_a.shape[1] != student_a.shape[1] or inspector_b.shape[0] != student_b.shape[0]:
                raise ValueError(f"Inspector and Student LoRA shapes differ: {prefix}")
            if abs(student_scale - inspector_scale) > 1e-12:
                raise ValueError(f"LoRA scale differs (Student {student_scale}, Inspector {inspector_scale}): {prefix}")
            if torch.count_nonzero(student_b).item() != 0:
                raise RuntimeError(f"Student lora_B is not zero-initialized: {prefix}")
            student_a[:inspector_rank].copy_(inspector_a.to(student_a.dtype))
            student_b[:, :inspector_rank].copy_(inspector_b.to(student_b.dtype))

            device = student_a.device
            reference = inspector_scale * (
                inspector_b.to(device, torch.float32) @ inspector_a.to(device, torch.float32)
            )
            delta = student_scale * (student_b.float() @ student_a.float())
            relative_error = (
                (delta - reference).norm() / reference.norm().clamp_min(1e-12)
            ).item()
            max_relative_error = max(max_relative_error, relative_error)
    if max_relative_error > 1e-2:
        raise RuntimeError(
            f"Student LoRA update differs from the Inspector update (relative error {max_relative_error:.3e})."
        )
    summary = {
        "inspector_checkpoint": str(checkpoint_dir),
        "inspector_lora_rank": inspector_rank,
        "inspector_lora_scale": inspector_scale,
        "initialized_modules": len(lora_pairs),
        "max_relative_error_of_lora_update": max_relative_error,
        "rule": "A_S[:r_I]=A_I, B_S[:, :r_I]=B_I, B_S[:, r_I:]=0; ViT/aligner LoRA and metric_heads keep Student initialization",
    }
    logger.info("Initialized Student LoRA from the Overall Inspector: %s", summary)
    return summary


def _format_candidate(
    source_image: str,
    edit_instruction: str,
    candidate_image: str,
) -> List[Dict[str, Any]]:
    return [
        dict(_SYSTEM_MESSAGE),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _INSTRUCT_PREFIX},
                {"type": "image", "image": os.path.abspath(source_image)},
                {"type": "text", "text": f" {edit_instruction}\n<Document>:"},
                {"type": "image", "image": os.path.abspath(candidate_image)},
            ],
        },
    ]


def _format_source(
    source_image: str,
    edit_instruction: str,
) -> List[Dict[str, Any]]:
    """Create a candidate-independent source/instruction representation."""
    return [
        dict(_SYSTEM_MESSAGE),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _INSTRUCT_PREFIX},
                {"type": "image", "image": os.path.abspath(source_image)},
                {"type": "text", "text": f" {edit_instruction}"},
            ],
        },
    ]


def _validate_and_aggregate_runs(
    runs: Any,
    sample_id: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    if not isinstance(runs, list) or len(runs) != 3:
        raise ValueError(f"Sample {sample_id}: runs must contain exactly three items.")
    by_index: Dict[int, Dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"Sample {sample_id}: every run must be an object.")
        run_index = run.get("run_index")
        if type(run_index) is not int or run_index not in (1, 2, 3):
            raise ValueError(f"Sample {sample_id}: run_index must be 1, 2, or 3.")
        if run_index in by_index:
            raise ValueError(f"Sample {sample_id}: duplicate run_index {run_index}.")
        by_index[run_index] = run
    if set(by_index) != {1, 2, 3}:
        raise ValueError(f"Sample {sample_id}: runs 1, 2, and 3 are required.")

    g_a: List[float] = []
    g_b: List[float] = []
    g_t: List[float] = []
    relevance: List[float] = []
    for metric in METRIC_NAMES:
        counts: Counter[str] = Counter()
        applicable = 0
        for run_index in (1, 2, 3):
            metrics = by_index[run_index].get("metric_groundtruth")
            if not isinstance(metrics, dict) or list(metrics) != list(METRIC_NAMES):
                raise ValueError(
                    f"Sample {sample_id}, run {run_index}: metric order does not match {METRIC_NAMES}."
                )
            annotation = metrics[metric]
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"Sample {sample_id}, run {run_index}, {metric}: invalid annotation."
                )
            targeted = annotation.get("targeted_metric")
            choice = annotation.get("choice")
            if type(targeted) is not int or targeted not in (0, 1):
                raise ValueError(
                    f"Sample {sample_id}, run {run_index}, {metric}: targeted_metric must be 0 or 1."
                )
            if choice not in CHOICES:
                raise ValueError(
                    f"Sample {sample_id}, run {run_index}, {metric}: invalid choice {choice!r}."
                )
            if targeted == 0 and choice != "Tie":
                raise ValueError(
                    f"Sample {sample_id}, run {run_index}, {metric}: inactive metrics must be Tie."
                )
            if targeted:
                counts[choice] += 1
                applicable += 1
        if applicable:
            g_a.append(counts["A"] / applicable)
            g_b.append(counts["B"] / applicable)
            g_t.append(counts["Tie"] / applicable)
        else:
            g_a.append(0.0)
            g_b.append(0.0)
            g_t.append(0.0)
        relevance.append(applicable / 3.0)
    return g_a, g_b, g_t, relevance


class GemmaWarmStartPreprocessor(RowPreprocessor):
    """Convert one raw three-run record into one A/B/source training item."""

    _RAW_FIELDS = {
        "sample_id",
        "source_image",
        "edit_instruction",
        "edited_image_A",
        "edited_image_B",
        "runs",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cache_signature = (
            "gemma_three_run_factorized_v1",
            DEFAULT_SYSTEM_MESSAGE,
            DEFAULT_EVALUATION_INSTRUCTION,
            tuple(METRIC_NAMES),
        )

    def prepare_dataset(self, dataset):
        missing = self._RAW_FIELDS - set(dataset.features)
        if missing:
            raise ValueError(
                f"Gemma warm-start dataset is missing fields: {sorted(missing)}."
            )
        return dataset

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(row.get("sample_id", "")).strip()
        source_image = str(row.get("source_image", "")).strip()
        edit_instruction = str(row.get("edit_instruction", "")).strip()
        image_a = str(row.get("edited_image_A", "")).strip()
        image_b = str(row.get("edited_image_B", "")).strip()
        if not all((sample_id, source_image, edit_instruction, image_a, image_b)):
            raise ValueError("Gemma warm-start record contains an empty required field.")
        for field, path in (
            ("source_image", source_image),
            ("edited_image_A", image_a),
            ("edited_image_B", image_b),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Sample {sample_id}: {field} does not exist: {path}")
        if image_a == image_b:
            raise ValueError(f"Sample {sample_id}: candidates A and B are identical.")

        g_a, g_b, g_t, relevance = _validate_and_aggregate_runs(
            row.get("runs"), sample_id
        )
        labels = g_a + g_b + g_t + relevance
        return {
            "messages": _format_candidate(source_image, edit_instruction, image_a),
            "rejected_messages": _format_candidate(
                source_image, edit_instruction, image_b
            ),
            # A third MS-Swift document transports the source-only sequence; it
            # is not interpreted as a ranking positive by this custom template.
            "positive_messages": [_format_source(source_image, edit_instruction)],
            "label": labels,
        }


class MetricWarmStartTemplate(Qwen3VLRerankerTemplate):
    """Encode adjacent A/B/source rows and preserve one label per pair."""

    def _preprocess_inputs(self, inputs: Any) -> None:
        return super(Qwen3VLRerankerTemplate, self)._preprocess_inputs(inputs)

    def _reranker_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
        if inputs.rejected is None or len(inputs.positive) != 1 or inputs.negative:
            raise ValueError(
                "Each warm-start item requires exactly A, B, and one source representation."
            )
        labels = inputs.chosen.label
        if labels is None or len(labels) != LABEL_WIDTH:
            raise ValueError("Each warm-start item requires 24 factorized targets.")
        candidate_a = self._encode_truncated(inputs.chosen)
        candidate_b = self._encode_truncated(inputs.rejected)
        source = self._encode_truncated(inputs.positive[0])
        return {
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "source": source,
            "labels": labels,
            "length": [
                candidate_a["length"],
                candidate_b["length"],
                source["length"],
            ],
        }

    def data_collator(
        self,
        batch: List[Dict[str, Any]],
        *,
        padding_to=None,
    ) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty warm-start batch.")
        if self.padding_free or self.sequence_parallel_size > 1:
            raise ValueError(
                "Warm-start batches require padding_free=False and sequence_parallel_size=1."
            )
        rows: List[Dict[str, Any]] = []
        labels: List[List[float]] = []
        for item in batch:
            if not all(
                key in item
                for key in ("candidate_a", "candidate_b", "source", "labels")
            ):
                raise ValueError("Encoded warm-start item lost A/B/source alignment.")
            labels.append(item["labels"])
            for key in ("candidate_a", "candidate_b", "source"):
                encoded = dict(item[key])
                encoded.pop("labels", None)
                encoded.pop("loss_scale", None)
                rows.append(encoded)
        model_inputs = self._data_collator(rows, padding_to=padding_to)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        if tuple(label_tensor.shape) != (len(batch), LABEL_WIDTH):
            raise ValueError("Warm-start labels must have shape [B, 24].")
        if model_inputs["input_ids"].size(0) != 3 * len(batch):
            raise ValueError("Every warm-start pair must produce exactly three backbone rows.")
        starts = torch.arange(len(batch), dtype=torch.long) * 3
        return {
            "labels": label_tensor,
            "pair_indices": torch.stack((starts, starts + 1), dim=1),
            "source_indices": starts + 2,
            **model_inputs,
        }

    def print_inputs(self, inputs: Dict[str, Any]) -> None:
        for key in ("candidate_a", "candidate_b", "source"):
            logger.info("Warm-start %s:", key)
            super().print_inputs(dict(inputs[key]))


_template_meta = deepcopy(TEMPLATE_MAPPING["qwen3_vl_reranker"])
_template_meta.template_type = PAIRWISE_TEMPLATE_TYPE
_template_meta.template_cls = MetricWarmStartTemplate
register_template(_template_meta, exist_ok=True)


def _student_forward(
    self,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    pair_indices: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    **kwargs: Any,
) -> MultiMetricRewardOutput:
    del labels
    model_inputs = {
        key: value
        for key, value in {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            **kwargs,
        }.items()
        if value is not None
    }
    hidden = _pool_qwen_last_hidden(self, model_inputs, attention_mask)
    if (
        not isinstance(pair_indices, torch.Tensor)
        or pair_indices.ndim != 2
        or pair_indices.size(1) != 2
        or pair_indices.size(0) == 0
    ):
        raise ValueError("pair_indices must have non-empty shape [B, 2].")
    if (
        not isinstance(source_indices, torch.Tensor)
        or source_indices.ndim != 1
        or source_indices.size(0) != pair_indices.size(0)
    ):
        raise ValueError("source_indices must have shape [B].")
    if pair_indices.dtype not in (torch.int32, torch.int64) or source_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Pair/source indices must be integer tensors.")
    pair_indices = pair_indices.to(device=hidden.device, dtype=torch.long)
    source_indices = source_indices.to(device=hidden.device, dtype=torch.long)
    used_rows = torch.cat((pair_indices.flatten(), source_indices)).sort().values
    if not torch.equal(used_rows, torch.arange(hidden.size(0), device=hidden.device)):
        raise ValueError("Pair/source indices must use every backbone row exactly once.")
    rows_a, rows_b = pair_indices.unbind(dim=1)
    return self.metric_heads(hidden[rows_a], hidden[rows_b], hidden[source_indices])


class FactorizedWarmStartLossWrapper(BaseLoss):
    def __init__(self, args, trainer) -> None:
        super().__init__(args, trainer)
        relevance_weight = float(os.environ.get("WARMSTART_RELEVANCE_WEIGHT", "1.0"))
        self.loss_fn = FactorizedWarmStartLoss(relevance_weight=relevance_weight)
        logger.info("Warm-start relevance loss weight: %s", relevance_weight)

    def __call__(
        self,
        outputs: MultiMetricRewardOutput,
        labels: Union[torch.Tensor, list],
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        target = torch.as_tensor(
            labels, device=outputs.tie_probs.device, dtype=torch.float32
        )
        if target.ndim != 2 or target.size(1) != LABEL_WIDTH:
            raise ValueError("Warm-start labels must have shape [B, 24].")
        return self.loss_fn(
            outputs,
            target[:, 0:NUM_METRICS],
            target[:, NUM_METRICS : 2 * NUM_METRICS],
            target[:, 2 * NUM_METRICS : 3 * NUM_METRICS],
            target[:, 3 * NUM_METRICS : 4 * NUM_METRICS],
        )


loss_map["factorized_metric_warmstart"] = FactorizedWarmStartLossWrapper


class MetricWarmStartTrainer(RerankerTrainer):
    def create_loss_and_eval_metric(self, args):
        if args.loss_type != "factorized_metric_warmstart":
            raise ValueError("Student warm-start requires factorized_metric_warmstart loss.")
        return {"compute_loss_func": FactorizedWarmStartLossWrapper(args, self)}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factorized_loss_func = self.compute_loss_func
        self.compute_loss_func = None
        self.model_accepts_loss_kwargs = False
        self.label_names = ["labels"]

    def evaluation_loop(self, *args, **kwargs):
        self.gather_function = self.accelerator.gather_for_metrics
        return super().evaluation_loop(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Warm-start batch is missing labels.")
        outputs = model(**model_inputs)
        loss = self.factorized_loss_func(outputs, labels)
        if not return_outputs:
            return loss
        prediction = torch.cat(
            (
                outputs.three_way_probs[:, :, 0],
                outputs.three_way_probs[:, :, 1],
                outputs.three_way_probs[:, :, 2],
                outputs.relevance_probs,
            ),
            dim=1,
        )
        return loss, {"logits": prediction}


TrainerFactory.TRAINER_MAPPING["generative_reranker"] = (
    f"{MetricWarmStartTrainer.__module__}.{MetricWarmStartTrainer.__qualname__}"
)
_original_get_trainer_cls = TrainerFactory.get_trainer_cls.__func__


@classmethod
def _get_trainer_cls(cls, args):
    if (
        getattr(args, "task_type", None) == "generative_reranker"
        and getattr(args, "loss_type", None) == "factorized_metric_warmstart"
    ):
        return MetricWarmStartTrainer
    return _original_get_trainer_cls(cls, args)


TrainerFactory.get_trainer_cls = _get_trainer_cls


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            sample_id = str(record.get("sample_id", "")).strip()
            source_image = str(record.get("source_image", "")).strip()
            if not sample_id or not source_image:
                raise ValueError(f"Line {line_number}: sample_id/source_image is empty.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            records.append(record)
    if not records:
        raise ValueError("Warm-start dataset is empty.")
    return records


def random_pair_split(
    records: Sequence[Dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Random split over pairs, as MS-Swift ``--split_dataset_ratio``."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("split_dataset_ratio must be strictly between 0 and 1.")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    validation_count = max(1, int(len(records) * validation_ratio))
    if validation_count >= len(records):
        raise RuntimeError("Random pair split produced an empty partition.")
    validation_index = set(order[:validation_count])
    train = [r for i, r in enumerate(records) if i not in validation_index]
    validation = [r for i, r in enumerate(records) if i in validation_index]
    return train, validation


def source_grouped_split(
    records: Sequence[Dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("split_dataset_ratio must be strictly between 0 and 1.")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_image"])].append(record)
    sources = sorted(grouped)
    random.Random(seed).shuffle(sources)
    validation_count = max(1, round(len(sources) * validation_ratio))
    if validation_count >= len(sources):
        validation_count = len(sources) - 1
    validation_sources = set(sources[:validation_count])
    train = [r for r in records if r["source_image"] not in validation_sources]
    validation = [r for r in records if r["source_image"] in validation_sources]
    if not train or not validation:
        raise RuntimeError("Source-grouped split produced an empty partition.")
    if {r["source_image"] for r in train} & {r["source_image"] for r in validation}:
        raise AssertionError("Source leakage detected between train and validation.")
    return train, validation


def shared_source_split(
    records: Sequence[Dict[str, Any]],
    split_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Use the split shared by every pipeline stage.

    Calibration samples are excluded from Student training and validation:
    they stay unseen for the Overall Inspector Model acceptance calibration.
    """
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    partition = payload.get("sample_partition")
    if not isinstance(partition, dict) or not partition:
        raise ValueError(f"Shared split has no sample_partition: {split_path}")
    sample_ids = {str(r["sample_id"]) for r in records}
    if sample_ids != set(partition):
        raise ValueError("Shared split sample identifiers differ from the GemmaGT dataset.")
    train = [r for r in records if partition[str(r["sample_id"])] == "train"]
    validation = [r for r in records if partition[str(r["sample_id"])] == "validation"]
    if not train or not validation:
        raise RuntimeError("Shared split produced an empty Student partition.")
    if {r["source_image"] for r in train} & {r["source_image"] for r in validation}:
        raise AssertionError("Source leakage detected between train and validation.")
    metadata = {
        "path": str(split_path),
        "split_method": payload.get("split_method"),
        "seed": payload.get("seed"),
        "validation_ratio": payload.get("validation_ratio"),
        "calibration_ratio": payload.get("calibration_ratio"),
        "excluded_calibration_samples": sum(
            1 for value in partition.values() if value == "calibration"
        ),
    }
    return train, validation, metadata


def _read_sample_ids(path: Path) -> List[str]:
    """Ordered ``sample_id`` (or ``unique_key``) values of a JSONL file."""
    sample_ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("unique_key") or "").strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: sample_id/unique_key is empty.")
            sample_ids.append(sample_id)
    return sample_ids


def train_order_split(
    records: Sequence[Dict[str, Any]],
    train_order_path: Path,
    num_pairs: int,
    validation_path: Path | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Train on the first ``num_pairs`` pairs of an Inspector ``train_order.jsonl``.

    Validation is empty unless ``validation_path`` lists the validation pairs,
    which must not overlap the training pairs.
    """
    if num_pairs <= 0:
        raise ValueError("The number of training pairs must be positive.")
    by_id = {str(r["sample_id"]): r for r in records}
    train_ids: List[str] = []
    for sample_id in _read_sample_ids(train_order_path):
        if sample_id not in train_ids:
            train_ids.append(sample_id)
        if len(train_ids) == num_pairs:
            break
    if len(train_ids) < num_pairs:
        raise ValueError(f"{train_order_path} has only {len(train_ids)} distinct pairs.")
    missing = [sample_id for sample_id in train_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} training pairs are missing from the dataset: {missing[:5]}")
    train = [by_id[sample_id] for sample_id in train_ids]

    validation: List[Dict[str, Any]] = []
    if validation_path is not None:
        validation_ids = list(dict.fromkeys(_read_sample_ids(validation_path)))
        overlap = set(validation_ids) & set(train_ids)
        if overlap:
            raise AssertionError(f"{len(overlap)} validation pairs are also training pairs.")
        missing = [sample_id for sample_id in validation_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} validation pairs are missing from the dataset: {missing[:5]}")
        validation = [by_id[sample_id] for sample_id in validation_ids]
        shared_sources = {r["source_image"] for r in train} & {r["source_image"] for r in validation}
        logger.info("Validation pairs sharing a source image with train: %d sources", len(shared_sources))

    metadata = {
        "path": str(train_order_path),
        "split_method": "train_order_prefix_v1",
        "train_pairs": num_pairs,
        "validation_path": str(validation_path) if validation_path is not None else None,
    }
    return train, validation, metadata


class MetricWarmStartSwiftSft(SwiftSft):
    def _prepare_model_tokenizer(self, **kwargs):
        super()._prepare_model_tokenizer(**kwargs)
        if self.model is None:
            return
        if self.args.task_type != "generative_reranker":
            raise ValueError("Student warm-start requires task_type=generative_reranker.")
        tokenizer = (
            self.processor
            if hasattr(self.processor, "convert_tokens_to_ids")
            else getattr(self.processor, "tokenizer", self.processor)
        )
        install_multi_metric_heads(self.model, tokenizer)
        self.model.forward = MethodType(_student_forward, self.model)
        if "metric_heads" not in self.args.modules_to_save:
            self.args.modules_to_save.append("metric_heads")

        output_embedding = self.model.get_output_embeddings()
        target_modules = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and module is not output_embedding
            and not name.startswith("metric_heads")
        ]
        if not target_modules:
            raise RuntimeError("No Qwen Linear modules were found for Student LoRA.")
        self.args.target_modules = target_modules
        logger.info("Student exact LoRA Linear targets: %d", len(target_modules))

    def _prepare_template(self):
        args = self.args
        if args.packing or args.padding_free or args.sequence_parallel_size > 1:
            raise ValueError(
                "Student warm-start requires packing=False, padding_free=False, and sequence_parallel_size=1."
            )
        if args.template not in {"qwen3_vl_reranker", PAIRWISE_TEMPLATE_TYPE}:
            raise ValueError("Student warm-start requires the Qwen3-VL reranker template.")
        args.template = PAIRWISE_TEMPLATE_TYPE
        args.template_meta = TEMPLATE_MAPPING[PAIRWISE_TEMPLATE_TYPE]
        super()._prepare_template()
        logger.info("Warm-start batch sizes count pairs; each pair becomes A/B/source.")

    def _get_dataset(self):
        if self.args.val_dataset:
            raise ValueError("Do not pass --val_dataset; validation is split from --dataset.")
        specs = list(self.args.dataset or [])
        if len(specs) != 1:
            raise ValueError("Student warm-start requires exactly one local JSONL dataset.")
        dataset_path = Path(str(specs[0])).expanduser().resolve()
        records = _load_jsonl(dataset_path)
        shared_split_raw = os.environ.get("STUDENT_WARMSTART_SHARED_SPLIT", "").strip()
        train_order_raw = os.environ.get("STUDENT_WARMSTART_TRAIN_ORDER", "").strip()
        if train_order_raw and shared_split_raw:
            raise ValueError("Set only one of STUDENT_WARMSTART_TRAIN_ORDER and STUDENT_WARMSTART_SHARED_SPLIT.")
        if train_order_raw:
            validation_raw = os.environ.get("STUDENT_WARMSTART_VALIDATION_IDS", "").strip()
            train_records, validation_records, self._shared_split = train_order_split(
                records,
                Path(train_order_raw).expanduser().resolve(),
                int(os.environ["STUDENT_WARMSTART_TRAIN_PAIRS"]),
                Path(validation_raw).expanduser().resolve() if validation_raw else None,
            )
        elif shared_split_raw:
            shared_split_path = Path(shared_split_raw).expanduser().resolve()
            train_records, validation_records, self._shared_split = (
                shared_source_split(records, shared_split_path)
            )
        else:
            train_records, validation_records = random_pair_split(
                records,
                float(self.args.split_dataset_ratio),
                int(self.args.data_seed),
            )
            self._shared_split = None
        self._dataset_path = str(dataset_path)
        self._train_records = train_records
        self._validation_records = validation_records
        logger.info(
            "Warm-start split: train=%d (%d sources), validation=%d (%d sources)",
            len(train_records),
            len({r["source_image"] for r in train_records}),
            len(validation_records),
            len({r["source_image"] for r in validation_records}),
        )
        preprocessor = GemmaWarmStartPreprocessor()
        preprocess_kwargs = {
            "num_proc": int(self.args.dataset_num_proc),
            "load_from_cache_file": bool(self.args.load_from_cache_file),
            "strict": True,
        }
        train_dataset = preprocessor(
            HfDataset.from_list(train_records), **preprocess_kwargs
        )
        if not validation_records:
            # Train-order split without validation: the last checkpoint is selected.
            return train_dataset, None
        validation_dataset = preprocessor(
            HfDataset.from_list(validation_records), **preprocess_kwargs
        )
        return train_dataset, validation_dataset

    def train(self, trainer):
        finalize_checkpoint_raw = os.environ.get(
            "STUDENT_WARMSTART_FINALIZE_CHECKPOINT", ""
        ).strip()
        if finalize_checkpoint_raw:
            selected_checkpoint = Path(finalize_checkpoint_raw).expanduser().resolve()
            trainer_state_path = selected_checkpoint / "trainer_state.json"
            if not trainer_state_path.is_file() or trainer_state_path.stat().st_size == 0:
                raise FileNotFoundError(
                    "Cannot finalize the Student because trainer_state.json is "
                    f"missing or empty: {trainer_state_path}"
                )
            with trainer_state_path.open("r", encoding="utf-8") as handle:
                saved_trainer_state = json.load(handle)
            trainer.state.global_step = int(saved_trainer_state["global_step"])
            best_metric = saved_trainer_state.get("best_metric")
            # Runs without validation have no best_metric; the given checkpoint is selected.
            trainer.state.best_metric = None if best_metric is None else float(best_metric)
            trainer.state.best_model_checkpoint = str(selected_checkpoint)
            trainer.state.last_model_checkpoint = str(selected_checkpoint)
            result = {
                "finalize_only": True,
                "selected_checkpoint": str(selected_checkpoint),
            }
            logger.info(
                "Skipping optimization and finalizing existing Student checkpoint: %s",
                selected_checkpoint,
            )
        else:
            init_checkpoint_raw = os.environ.get(
                "STUDENT_WARMSTART_INIT_INSPECTOR_CHECKPOINT", ""
            ).strip()
            if init_checkpoint_raw:
                self._inspector_initialization = _initialize_lora_from_inspector(
                    trainer.accelerator.unwrap_model(trainer.model),
                    Path(init_checkpoint_raw),
                )
            result = super().train(trainer)
            selected_checkpoint_raw = (
                trainer.state.best_model_checkpoint
                or getattr(trainer.state, "last_model_checkpoint", None)
            )
            if not selected_checkpoint_raw:
                raise RuntimeError(
                    "Training finished without a selected Student checkpoint."
                )
            selected_checkpoint = Path(selected_checkpoint_raw).expanduser().resolve()
            # With SaveStrategy.BEST, transformers tracks best_metric but can
            # leave best_model_checkpoint unset. MS-Swift's sole retained
            # checkpoint is the latest validation improvement and is selected.
            trainer.state.best_model_checkpoint = str(selected_checkpoint)

        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        _restore_peft_checkpoint(unwrapped, selected_checkpoint)
        if not isinstance(unwrapped, PeftModel):
            raise TypeError("MS-Swift Student checkpoint must be a PEFT model.")
        adapter_state = _clone_tensor_dict(get_peft_model_state_dict(unwrapped))
        if not any("metric_heads" in key for key in adapter_state):
            raise RuntimeError("PEFT checkpoint is missing metric_heads.")
        if trainer.is_world_process_zero():
            checkpoint_path = Path(self.args.output_dir) / CHECKPOINT_FILENAME
            checkpoint = {
                "format_version": 1,
                "model_role": "six_metric_student_warmstart",
                "training_framework": "ms-swift",
                "tie_head_mode": resolve_tie_head_mode(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_model_path": str(Path(self.args.model).resolve()),
                "ms_swift_best_checkpoint": trainer.state.best_model_checkpoint,
                "adapter_state_dict": adapter_state,
                "peft_config": unwrapped.peft_config["default"].to_dict(),
                "metric_order": list(METRIC_NAMES),
                "loss": {
                    "name": "factorized_gemma_warmstart",
                    "formula": "m*[BCE(g_T,t)+(g_A+g_B)BCE(g_A/(g_A+g_B),w)]+lambda_M*BCE(m,mu)",
                    "preference_target_scope": "applicable_gemma_runs_only",
                    "relevance_weight": float(
                        os.environ.get("WARMSTART_RELEVANCE_WEIGHT", "1.0")
                    ),
                },
                "prompt": {
                    "config_path": str(
                        _PROJECT_ROOT / "prompts" / "reward_model_template.json"
                    ),
                    "profile": DEFAULT_PROMPT_CONFIG.profile,
                    "system_message": DEFAULT_SYSTEM_MESSAGE,
                    "evaluation_instruction": DEFAULT_EVALUATION_INSTRUCTION,
                },
                "data": {
                    "dataset_path": self._dataset_path,
                    "shared_split": self._shared_split,
                    "split_method": (
                        self._shared_split["split_method"]
                        if self._shared_split
                        else "random_pair_v1"
                    ),
                    "split_seed": int(self.args.data_seed),
                    "validation_ratio": float(self.args.split_dataset_ratio),
                    "train_sample_ids": [r["sample_id"] for r in self._train_records],
                    "validation_sample_ids": [
                        r["sample_id"] for r in self._validation_records
                    ],
                },
                "selection": {
                    "criterion": self.args.metric_for_best_model,
                    "best_metric": trainer.state.best_metric,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                },
                "teacher_initialization": "Clone this checkpoint in memory after warm-start.",
                "student_initialization": getattr(
                    self, "_inspector_initialization", None
                )
                or "base model with fresh LoRA",
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved consolidated Student checkpoint: %s", checkpoint_path)
        return result


def main():
    return MetricWarmStartSwiftSft().main()


if __name__ == "__main__":
    main()
