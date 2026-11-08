#!/usr/bin/env python3
"""MS-Swift training integration for the three-way Overall Inspector Model.

The Inspector reuses the Student Score Head and Symmetric Tie Head with one
overall output and no Relevance Head:
``P(A) = (1 - t) w``, ``P(B) = (1 - t)(1 - w)``, ``P(Tie) = t``.  The loss is
three-way cross-entropy on the human overall vote, so tie pairs are kept.

Each pair is encoded as two sequences (candidate A, candidate B); there is no
source-only sequence.  The training order is fixed by this script and written
to ``train_order.jsonl``; the dataloader does not reshuffle it
(``--train_dataloader_shuffle false``).  Every training pair carries its
position in that order, so ``progressive_loss.jsonl`` records the loss of each
pair at the step it was first seen, before the optimizer updates on it.

Splits:
- ``OVERALL_INSPECTOR_SPLIT``: JSON file with ``sample_partition``
  ``{sample_id: "train" | "validation" | "unused"}``; only train and
  validation pairs are used.
- otherwise ``--split_dataset_ratio > 0``: MS-Swift random split.
- otherwise: no validation; every pair is trained once per data pass.

``OVERALL_INSPECTOR_DATA_PASSES`` repeats the training pairs, each pass in its
own seeded order.  ``--num_train_epochs`` must be 1 so that the learning-rate
schedule and the recorded order cover exactly the same sequence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HfDataset
from peft import (
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file as load_safetensors
from transformers import TrainerCallback


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from swift.dataset import DatasetLoader, RowPreprocessor  # noqa: E402
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
    MultiMetricRewardOutput,
    OverallInspectorLoss,
    _pool_qwen_last_hidden,
    install_overall_inspector_head,
    resolve_tie_head_mode,
)
from src.prompt_config import (  # noqa: E402
    OVERALL_INSPECTOR_EVALUATION_INSTRUCTION,
    OVERALL_INSPECTOR_PROMPT_CONFIG,
    OVERALL_INSPECTOR_SYSTEM_MESSAGE,
)


logger = get_logger()

DEFAULT_EDITREWARD_ROOT = (
    "/datastore/shared/datasets/EditReward-Data"
)
CHECKPOINT_FILENAME = "overall_inspector_checkpoint.pt"
TRAIN_ORDER_FILENAME = "train_order.jsonl"
PROGRESSIVE_LOSS_FILENAME = "progressive_loss.jsonl"
PAIRWISE_TEMPLATE_TYPE = "qwen3_vl_overall_inspector_pairwise"
LOSS_TYPE = "overall_inspector"
VOTE_TO_CLASS = {"leftvote": "A", "rightvote": "B", "tie": "Tie", "tievote": "Tie"}
CLASS_TO_TARGET = {"A": (1.0, 0.0, 0.0), "B": (0.0, 1.0, 0.0), "Tie": (0.0, 0.0, 1.0)}
# Columns 0-2 are (y_A, y_B, y_T); column 3 is the position in the training
# order (-1 for validation pairs).  float32 represents positions exactly below 2**24.
LABEL_WIDTH = 4


@dataclass(frozen=True)
class InspectorRecord:
    sample_id: str
    source_image: str
    edit_instruction: str
    image_a: str
    image_b: str
    vote: str


def resolve_image_path(raw_path: Any, dataset_root: str) -> str:
    """Resolve an absolute path or a path relative to the EditReward root."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Image path must be a non-empty string.")
    value = raw_path.strip()
    if value.startswith("file://"):
        value = value[7:]
    path = Path(value).expanduser()
    root = Path(dataset_root).expanduser().resolve()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Image does not exist: {resolved}")
        return str(resolved)
    candidates = [root / path]
    if (root / "extracted_data").is_dir():
        candidates.append(root / "extracted_data" / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Relative image path escapes the EditReward root: {raw_path}"
            ) from exc
        if resolved.is_file():
            return str(resolved)
    raise FileNotFoundError(
        f"Image does not exist below EditReward root {root}: {raw_path}"
    )


def _required_text(row: Dict[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: missing or empty field {field!r}.")
    return value.strip()


def load_inspector_records(
    dataset_path: str,
    editreward_root: str,
) -> Tuple[List[InspectorRecord], Dict[str, int]]:
    """Read HumanGT-format rows (EditReward metadata fields), ties included."""
    rows: List[Tuple[int, Dict[str, Any]]] = []
    seen_ids = set()
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc.msg}."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: JSON value must be an object.")
            sample_id = _required_text(row, "unique_key", line_number)
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate unique_key in Inspector data: {sample_id}")
            seen_ids.add(sample_id)
            rows.append((line_number, row))

    def build(item: Tuple[int, Dict[str, Any]]) -> InspectorRecord:
        line_number, row = item
        vote = _required_text(row, "vote_type", line_number).lower()
        if vote not in VOTE_TO_CLASS:
            raise ValueError(f"Line {line_number}: unsupported vote_type {vote!r}.")
        image_a = resolve_image_path(
            _required_text(row, "left_output_image", line_number), editreward_root
        )
        image_b = resolve_image_path(
            _required_text(row, "right_output_image", line_number), editreward_root
        )
        if image_a == image_b:
            raise ValueError(f"Line {line_number}: candidates A and B are identical.")
        return InspectorRecord(
            sample_id=_required_text(row, "unique_key", line_number),
            source_image=resolve_image_path(
                _required_text(row, "source_img", line_number), editreward_root
            ),
            edit_instruction=_required_text(row, "instruction", line_number),
            image_a=image_a,
            image_b=image_b,
            vote=VOTE_TO_CLASS[vote],
        )

    # Image existence checks are I/O bound on the shared file system.
    with ThreadPoolExecutor(max_workers=32) as pool:
        records = list(pool.map(build, rows, chunksize=256))
    if not records:
        raise ValueError("No Inspector records were found.")
    stats: Dict[str, int] = {"input_rows": len(records)}
    stats.update({f"vote_{key}": value for key, value in count_votes(records).items()})
    stats["source_groups"] = len({record.source_image for record in records})
    return records, stats


def count_votes(records: Sequence[InspectorRecord]) -> Dict[str, int]:
    counts = Counter(record.vote for record in records)
    return {"A": counts["A"], "B": counts["B"], "Tie": counts["Tie"]}


def build_training_order(
    records: Sequence[InspectorRecord],
    passes: int,
    seed: int,
    prefix_groups: Sequence[Sequence[str]] = (),
) -> List[Tuple[int, InspectorRecord]]:
    """Concatenate ``passes`` seeded permutations; returns (pass, record).

    With ``prefix_groups`` the first pass trains each group in turn, then the
    remaining pairs, each part in its own seeded order.
    """
    order: List[Tuple[int, InspectorRecord]] = []
    ranked = sorted(records, key=lambda record: record.sample_id)
    by_id = {record.sample_id: record for record in ranked}
    prefixed = [sample_id for group in prefix_groups for sample_id in group]
    if len(set(prefixed)) != len(prefixed):
        raise ValueError("Training prefix groups overlap.")
    missing = [sample_id for sample_id in prefixed if sample_id not in by_id]
    if missing:
        raise ValueError(f"Training prefix pairs are not in train: {missing[:5]}")
    for data_pass in range(passes):
        if data_pass == 0 and prefix_groups:
            parts = [[by_id[sample_id] for sample_id in sorted(group)] for group in prefix_groups]
            prefixed_set = set(prefixed)
            parts.append([r for r in ranked if r.sample_id not in prefixed_set])
        else:
            parts = [list(ranked)]
        for permutation in parts:
            random.Random(seed + data_pass).shuffle(permutation)
            order.extend((data_pass, record) for record in permutation)
    return order


def _format_candidate(record: InspectorRecord, candidate_image: str) -> List[Dict]:
    return [
        {"role": "system", "content": OVERALL_INSPECTOR_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<Instruct>: "
                        f"{OVERALL_INSPECTOR_EVALUATION_INSTRUCTION}\n<Query>:"
                    ),
                },
                {"type": "image", "image": record.source_image},
                {
                    "type": "text",
                    "text": f" {record.edit_instruction}\n<Document>:",
                },
                {"type": "image", "image": candidate_image},
            ],
        },
    ]


class OverallInspectorPairwisePreprocessor(RowPreprocessor):
    """Build one A/B pair row through MS-Swift's Arrow-safe pipeline."""

    _RAW_FIELDS = {
        "sample_id",
        "source_image",
        "edit_instruction",
        "image_a",
        "image_b",
        "vote",
        "position",
    }

    def prepare_dataset(self, dataset):
        missing = self._RAW_FIELDS - set(dataset.features)
        if missing:
            raise ValueError(
                "Overall Inspector dataset is missing raw fields: "
                f"{sorted(missing)}"
            )
        return dataset

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        record = InspectorRecord(
            sample_id=str(row["sample_id"]),
            source_image=str(row["source_image"]),
            edit_instruction=str(row["edit_instruction"]),
            image_a=str(row["image_a"]),
            image_b=str(row["image_b"]),
            vote=str(row["vote"]),
        )
        return {
            "messages": _format_candidate(record, record.image_a),
            "rejected_messages": _format_candidate(record, record.image_b),
            "label": [*CLASS_TO_TARGET[record.vote], float(row["position"])],
        }


def _records_to_rows(
    items: Sequence[Tuple[int, InspectorRecord]],
) -> List[Dict[str, Any]]:
    return [{**asdict(record), "position": position} for position, record in items]


class OverallInspectorPairwiseTemplate(Qwen3VLRerankerTemplate):
    """Encode one pair-row into adjacent candidate A / candidate B rows."""

    def _preprocess_inputs(self, inputs: Any) -> None:
        return super(Qwen3VLRerankerTemplate, self)._preprocess_inputs(inputs)

    def _reranker_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
        if inputs.rejected is None or inputs.positive or inputs.negative:
            raise ValueError("Each Inspector row must contain exactly candidates A and B.")
        labels = inputs.chosen.label
        if labels is None or len(labels) != LABEL_WIDTH:
            raise ValueError(f"Each Inspector row requires {LABEL_WIDTH} label values.")
        candidate_a = self._encode_truncated(inputs.chosen)
        candidate_b = self._encode_truncated(inputs.rejected)
        return {
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "labels": labels,
            "length": [candidate_a["length"], candidate_b["length"]],
        }

    def data_collator(
        self,
        batch: List[Dict[str, Any]],
        *,
        padding_to=None,
    ) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty Inspector batch.")
        if self.padding_free or self.sequence_parallel_size > 1:
            raise ValueError(
                "Inspector pairs require padding_free=False and sequence_parallel_size=1."
            )
        rows: List[Dict[str, Any]] = []
        labels: List[List[float]] = []
        for item in batch:
            if not all(key in item for key in ("candidate_a", "candidate_b", "labels")):
                raise ValueError("Encoded Inspector row lost its A/B alignment.")
            labels.append(item["labels"])
            for key in ("candidate_a", "candidate_b"):
                encoded = dict(item[key])
                encoded.pop("labels", None)
                encoded.pop("loss_scale", None)
                rows.append(encoded)
        model_inputs = self._data_collator(rows, padding_to=padding_to)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        if tuple(label_tensor.shape) != (len(batch), LABEL_WIDTH):
            raise ValueError(f"Inspector labels must have shape [B, {LABEL_WIDTH}].")
        if model_inputs["input_ids"].size(0) != 2 * len(batch):
            raise ValueError("Each Inspector pair must produce exactly two sequences.")
        starts = torch.arange(len(batch), dtype=torch.long) * 2
        return {
            "labels": label_tensor,
            "pair_indices": torch.stack((starts, starts + 1), dim=1),
            **model_inputs,
        }

    def print_inputs(self, inputs: Dict[str, Any]) -> None:
        for key in ("candidate_a", "candidate_b"):
            logger.info("Overall Inspector %s:", key)
            super().print_inputs(dict(inputs[key]))


_inspector_template_meta = deepcopy(TEMPLATE_MAPPING["qwen3_vl_reranker"])
_inspector_template_meta.template_type = PAIRWISE_TEMPLATE_TYPE
_inspector_template_meta.template_cls = OverallInspectorPairwiseTemplate
register_template(_inspector_template_meta, exist_ok=True)


def _inspector_forward(
    self,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    pair_indices: torch.Tensor | None = None,
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
    if pair_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("pair_indices must contain integer row indices.")
    pair_indices = pair_indices.to(device=hidden.device, dtype=torch.long)
    used_rows = pair_indices.flatten().sort().values
    if not torch.equal(used_rows, torch.arange(hidden.size(0), device=hidden.device)):
        raise ValueError("Inspector pair_indices must use every candidate row exactly once.")
    rows_a, rows_b = pair_indices.unbind(dim=1)
    return self.overall_inspector_head(hidden[rows_a], hidden[rows_b])


class OverallInspectorLossWrapper(BaseLoss):
    def __init__(self, args, trainer) -> None:
        super().__init__(args, trainer)
        self.loss_fn = OverallInspectorLoss()

    def __call__(
        self,
        outputs: MultiMetricRewardOutput,
        labels: Union[torch.Tensor, list],
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        target = torch.as_tensor(labels, device=outputs.margins.device, dtype=torch.float32)
        return self.loss_fn(outputs, target[:, 0:3])


loss_map[LOSS_TYPE] = OverallInspectorLossWrapper


class OverallInspectorTrainer(RerankerTrainer):
    def create_loss_and_eval_metric(self, args):
        if args.loss_type != LOSS_TYPE:
            raise ValueError(f"Overall Inspector requires loss_type={LOSS_TYPE}.")
        return {"compute_loss_func": OverallInspectorLossWrapper(args, self)}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inspector_loss_func = self.compute_loss_func
        self.compute_loss_func = None
        self.model_accepts_loss_kwargs = False
        self.label_names = ["labels"]
        self.progressive_loss_path: Path | None = None
        self.training_order: List[Dict[str, Any]] | None = None

    def evaluation_loop(self, *args, **kwargs):
        self.gather_function = self.accelerator.gather_for_metrics
        return super().evaluation_loop(*args, **kwargs)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Resume optimizer and schedule state on torch < 2.6.

        Transformers refuses ``torch.load`` below torch 2.6 (CVE-2025-32434)
        even with ``weights_only=True``.  These files were written by this
        training run, so they are loaded directly with ``weights_only=True``.
        """
        if checkpoint is None:
            return
        optimizer_path = Path(checkpoint) / "optimizer.pt"
        scheduler_path = Path(checkpoint) / "scheduler.pt"
        if not optimizer_path.is_file() or not scheduler_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint lacks optimizer.pt or scheduler.pt: {checkpoint}"
            )
        self.optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        self.lr_scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=True)
        )
        logger.info("Restored optimizer and learning-rate schedule from: %s", checkpoint)

    def _load_rng_state(self, checkpoint):
        # RNG state only affects dropout masks; it is not restored on torch < 2.6.
        logger.info("RNG state is not restored from: %s", checkpoint)

    def _record_progressive_loss(
        self,
        labels: torch.Tensor,
        outputs: MultiMetricRewardOutput,
        per_pair: torch.Tensor,
    ) -> None:
        if self.progressive_loss_path is None or self.training_order is None:
            return
        if not self.is_world_process_zero():
            return
        positions = labels[:, 3].detach().round().long().cpu().tolist()
        probs = outputs.three_way_probs[:, 0, :].detach().float().cpu().tolist()
        losses = per_pair.detach().float().cpu().tolist()
        targets = labels[:, 0:3].detach().cpu().tolist()
        step = int(self.state.global_step) + 1
        lines = []
        for position, prob, loss, target in zip(positions, probs, losses, targets):
            expected = self.training_order[position]
            if list(CLASS_TO_TARGET[expected["vote"]]) != target:
                raise RuntimeError(
                    f"Training batch is misaligned with {TRAIN_ORDER_FILENAME} at position {position}."
                )
            lines.append(
                json.dumps(
                    {
                        "step": step,
                        "position": position,
                        "sample_id": expected["sample_id"],
                        "vote": expected["vote"],
                        "loss": loss,
                        "prob_A": prob[0],
                        "prob_B": prob[1],
                        "prob_Tie": prob[2],
                    },
                    ensure_ascii=False,
                )
            )
        with self.progressive_loss_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

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
            raise ValueError("Inspector batch is missing labels.")
        outputs = model(**model_inputs)
        target = torch.as_tensor(labels, device=outputs.margins.device, dtype=torch.float32)
        per_pair = self.inspector_loss_func.loss_fn.per_pair(outputs, target[:, 0:3])
        loss = per_pair.mean()
        if model.training:
            self._record_progressive_loss(target, outputs, per_pair)
        if not return_outputs:
            return loss
        return loss, {"logits": outputs.three_way_probs[:, 0, :]}


TrainerFactory.TRAINER_MAPPING["generative_reranker"] = (
    f"{OverallInspectorTrainer.__module__}.{OverallInspectorTrainer.__qualname__}"
)
_original_get_trainer_cls = TrainerFactory.get_trainer_cls.__func__


@classmethod
def _get_trainer_cls(cls, args):
    if (
        getattr(args, "task_type", None) == "generative_reranker"
        and getattr(args, "loss_type", None) == LOSS_TYPE
    ):
        return OverallInspectorTrainer
    return _original_get_trainer_cls(cls, args)


TrainerFactory.get_trainer_cls = _get_trainer_cls


class PairMilestoneSaveCallback(TrainerCallback):
    """Save at the steps where given numbers of training pairs have been seen.

    With ``evaluate`` the validation split is also scored at those steps and
    at the last step.
    """

    def __init__(self, milestone_steps: Sequence[int], evaluate: bool = False) -> None:
        self.milestone_steps = set(int(step) for step in milestone_steps)
        self.evaluate = evaluate

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.milestone_steps:
            control.should_save = True
            control.should_evaluate = control.should_evaluate or self.evaluate
        elif self.evaluate and state.global_step == state.max_steps:
            control.should_save = True
            control.should_evaluate = True
        return control


def _clone_tensor_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _restore_peft_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Restore selected PEFT weights into an already prepared Inspector model."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Inspector PEFT weights are missing or empty: {weights_path}"
        )
    if not isinstance(model, PeftModel):
        raise TypeError("Inspector checkpoint restoration requires a PEFT model.")
    state = load_safetensors(str(weights_path), device="cpu")
    load_result = set_peft_model_state_dict(model, state, adapter_name="default")
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            "Unexpected keys while restoring the Inspector checkpoint: "
            f"{unexpected[:10]}"
        )
    logger.info("Restored selected Inspector weights from: %s", checkpoint_dir)


def _load_split(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    partition = payload.get("sample_partition")
    if not isinstance(partition, dict) or not partition:
        raise ValueError(f"Split file has no sample_partition: {path}")
    # "unused" pairs (e.g. outside the 300-pair setting) are listed but never loaded.
    invalid = {
        value for value in partition.values() if value not in ("train", "validation", "unused")
    }
    if invalid:
        raise ValueError(f"Split file contains unsupported partitions: {sorted(invalid)}")
    return {str(key): value for key, value in partition.items() if value != "unused"}


def _load_prefix_ids(path: Path) -> List[str]:
    """Pair ids (``sample_id`` or ``unique_key``) of a JSONL file, trained first."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ids.append(str(row.get("sample_id") or row["unique_key"]).strip())
    return ids


def _parse_pair_milestones(raw: str) -> List[int]:
    values = sorted({int(value) for value in raw.replace(" ", "").split(",") if value})
    if any(value <= 0 for value in values):
        raise ValueError("Pair milestones must be positive integers.")
    return values


class OverallInspectorSwiftSft(SwiftSft):
    """MS-Swift pipeline for the three-way Overall Inspector Model."""

    def _prepare_model_tokenizer(self, **kwargs):
        super()._prepare_model_tokenizer(**kwargs)
        if self.model is None:
            return
        if self.args.task_type != "generative_reranker":
            raise ValueError("Overall Inspector requires task_type=generative_reranker.")
        if self.args.loss_type != LOSS_TYPE:
            raise ValueError(f"Overall Inspector requires loss_type={LOSS_TYPE}.")
        if self.args.freeze_vit != self.args.freeze_aligner:
            raise ValueError("Inspector requires freeze_vit and freeze_aligner to match.")
        tokenizer = (
            self.processor
            if hasattr(self.processor, "convert_tokens_to_ids")
            else getattr(self.processor, "tokenizer", self.processor)
        )
        install_overall_inspector_head(self.model, tokenizer)
        self.model.forward = MethodType(_inspector_forward, self.model)
        if "overall_inspector_head" not in self.args.modules_to_save:
            self.args.modules_to_save.append("overall_inspector_head")

        output_embedding = self.model.get_output_embeddings()
        target_modules = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and module is not output_embedding
            and not name.startswith("overall_inspector_head")
            and not (self.args.freeze_vit and name.startswith("model.visual."))
        ]
        if not target_modules:
            raise RuntimeError("No Qwen Linear modules were found for Inspector LoRA.")
        # Exact names avoid PEFT 0.12 matching Conv3d through suffix ``proj``.
        self.args.target_modules = target_modules
        logger.info("Overall Inspector exact LoRA Linear targets: %d", len(target_modules))

    def _prepare_template(self):
        args = self.args
        if args.packing or args.padding_free or args.sequence_parallel_size > 1:
            raise ValueError(
                "Overall Inspector requires packing=False, padding_free=False, "
                "and sequence_parallel_size=1."
            )
        if args.template not in {"qwen3_vl_reranker", PAIRWISE_TEMPLATE_TYPE}:
            raise ValueError("Overall Inspector requires the Qwen3-VL reranker template.")
        args.template = PAIRWISE_TEMPLATE_TYPE
        args.template_meta = TEMPLATE_MAPPING[PAIRWISE_TEMPLATE_TYPE]
        super()._prepare_template()
        logger.info("Inspector batch sizes count pairs; each pair becomes [A, B].")

    def _get_dataset(self):
        args = self.args
        if args.val_dataset:
            raise ValueError("Do not pass --val_dataset; validation comes from --dataset.")
        if float(args.num_train_epochs) != 1.0:
            raise ValueError(
                "Use --num_train_epochs 1 and OVERALL_INSPECTOR_DATA_PASSES for repeated passes."
            )
        if args.train_dataloader_shuffle:
            raise ValueError("Use --train_dataloader_shuffle false; this script fixes the order.")
        dataset_specs = list(args.dataset or [])
        if len(dataset_specs) != 1:
            raise ValueError("Overall Inspector requires exactly one local JSONL dataset.")
        dataset_path = Path(str(dataset_specs[0])).expanduser().resolve()
        editreward_root = os.environ.get(
            "INSPECTOR_EDITREWARD_ROOT", DEFAULT_EDITREWARD_ROOT
        )
        records, stats = load_inspector_records(str(dataset_path), editreward_root)

        split_raw = os.environ.get("OVERALL_INSPECTOR_SPLIT", "").strip()
        split_path = None
        # OVERALL_INSPECTOR_TRAIN_PREFIX: JSONL of pairs that stay in train and
        # come first in the first data pass.
        prefix_raw = os.environ.get("OVERALL_INSPECTOR_TRAIN_PREFIX", "").strip()
        prefix_ids = set(_load_prefix_ids(Path(prefix_raw))) if prefix_raw else set()
        prefix_groups: List[List[str]] = [sorted(prefix_ids)] if prefix_ids else []
        if split_raw:
            split_path = Path(split_raw).expanduser().resolve()
            partition = _load_split(split_path)
            by_id = {record.sample_id: record for record in records}
            missing = sorted(set(partition) - set(by_id))
            if missing:
                raise ValueError(f"Split ids are missing from the dataset: {missing[:5]}")
            train_records = [r for r in records if partition.get(r.sample_id) == "train"]
            validation_records = [
                r for r in records if partition.get(r.sample_id) == "validation"
            ]
            split_method = "split_file"
        elif float(args.split_dataset_ratio) > 0.0:
            # Prefix pairs are kept out of the random split.
            train_rows, validation_rows = DatasetLoader.post_process(
                HfDataset.from_list(
                    [asdict(record) for record in records if record.sample_id not in prefix_ids]
                ),
                split_dataset_ratio=float(args.split_dataset_ratio),
                shuffle=bool(args.dataset_shuffle),
                random_state=np.random.RandomState(int(args.data_seed)),
            )
            by_id = {record.sample_id: record for record in records}
            train_records = [by_id[sample_id] for sample_id in train_rows["sample_id"]]
            train_records += [r for r in records if r.sample_id in prefix_ids]
            validation_records = [
                by_id[sample_id] for sample_id in validation_rows["sample_id"]
            ]
            split_method = "ms_swift_split_dataset_ratio"
        else:
            train_records, validation_records = list(records), []
            split_method = "none"
        if not train_records:
            raise RuntimeError("The Inspector split has no training pairs.")
        if split_method == "split_file" and (
            {r.source_image for r in train_records} & {r.source_image for r in validation_records}
        ):
            raise AssertionError("Source leakage detected between train and validation.")

        passes = int(os.environ.get("OVERALL_INSPECTOR_DATA_PASSES", "1"))
        if passes < 1:
            raise ValueError("OVERALL_INSPECTOR_DATA_PASSES must be at least 1.")
        order = build_training_order(
            train_records, passes, int(args.data_seed), prefix_groups
        )
        order_rows = [
            {
                "position": position,
                "pass": data_pass,
                "sample_id": record.sample_id,
                "vote": record.vote,
            }
            for position, (data_pass, record) in enumerate(order)
        ]
        order_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in order_rows
        )
        order_digest = hashlib.sha256(order_text.encode("utf-8")).hexdigest()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        order_path = output_dir / TRAIN_ORDER_FILENAME
        if order_path.exists():
            existing = hashlib.sha256(order_path.read_bytes()).hexdigest()
            if existing != order_digest:
                raise RuntimeError(f"{order_path} differs from the recomputed training order.")
        else:
            tmp_path = order_path.with_suffix(".jsonl.tmp")
            tmp_path.write_text(order_text, encoding="utf-8")
            os.replace(tmp_path, order_path)

        self._inspector_dataset_path = str(dataset_path)
        self._inspector_editreward_root = str(Path(editreward_root).resolve())
        self._inspector_data_stats = stats
        self._inspector_split = {
            "method": split_method,
            "path": str(split_path) if split_path else None,
            "train_prefix": prefix_raw or None,
            "train_prefix_pairs": len(prefix_ids),
            "seed": int(args.data_seed),
            "validation_ratio": (
                len(validation_records) / (len(train_records) + len(validation_records))
                if split_method == "split_file"
                else float(args.split_dataset_ratio)
            ),
        }
        self._inspector_passes = passes
        self._inspector_train_records = train_records
        self._inspector_validation_records = validation_records
        self._inspector_order_rows = order_rows
        self._inspector_order_digest = order_digest
        logger.info("Inspector data stats: %s", stats)
        logger.info(
            "Inspector split (%s): train=%d %s, validation=%d %s; passes=%d, ordered pairs=%d",
            split_method,
            len(train_records),
            count_votes(train_records),
            len(validation_records),
            count_votes(validation_records),
            passes,
            len(order_rows),
        )

        preprocessor = OverallInspectorPairwisePreprocessor()
        preprocess_kwargs = {
            "num_proc": int(args.dataset_num_proc),
            "load_from_cache_file": bool(args.load_from_cache_file),
            "strict": True,
        }
        train_dataset = preprocessor(
            HfDataset.from_list(
                _records_to_rows([(position, record) for position, (_, record) in enumerate(order)])
            ),
            **preprocess_kwargs,
        )
        validation_dataset = None
        if validation_records:
            validation_dataset = preprocessor(
                HfDataset.from_list(
                    _records_to_rows([(-1, record) for record in validation_records])
                ),
                **preprocess_kwargs,
            )
        return train_dataset, validation_dataset

    def _prepare_progressive_loss(self, trainer, resume_step: int) -> None:
        path = Path(self.args.output_dir) / PROGRESSIVE_LOSS_FILENAME
        if trainer.is_world_process_zero() and path.exists():
            # Drop rows written after the resumed checkpoint; they are re-trained.
            kept = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["step"] <= resume_step
            ]
            tmp_path = path.with_suffix(".jsonl.tmp")
            tmp_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            os.replace(tmp_path, path)
        trainer.progressive_loss_path = path
        trainer.training_order = self._inspector_order_rows

    def _milestone_steps(self) -> List[int]:
        raw = os.environ.get("OVERALL_INSPECTOR_SAVE_PAIR_MILESTONES", "").strip()
        if not raw:
            return []
        args = self.args
        pairs_per_step = (
            int(args.per_device_train_batch_size)
            * int(args.gradient_accumulation_steps)
            * max(1, int(getattr(args, "world_size", 1) or 1))
        )
        return [math.ceil(pairs / pairs_per_step) for pairs in _parse_pair_milestones(raw)]

    def train(self, trainer):
        resume_raw = getattr(self.args, "resume_from_checkpoint", None)
        resume_step = 0
        if resume_raw:
            state_path = Path(str(resume_raw)) / "trainer_state.json"
            resume_step = int(json.loads(state_path.read_text(encoding="utf-8"))["global_step"])
        self._prepare_progressive_loss(trainer, resume_step)
        milestone_steps = self._milestone_steps()
        if milestone_steps:
            evaluate = os.environ.get("OVERALL_INSPECTOR_EVAL_AT_PAIR_MILESTONES", "0") == "1"
            trainer.add_callback(
                PairMilestoneSaveCallback(
                    milestone_steps, evaluate=evaluate and trainer.eval_dataset is not None
                )
            )
            logger.info("Extra checkpoint steps for pair milestones: %s", milestone_steps)

        result = super().train(trainer)
        selected_checkpoint_raw = trainer.state.best_model_checkpoint or getattr(
            trainer.state, "last_model_checkpoint", None
        )
        if not selected_checkpoint_raw:
            last = Path(self.args.output_dir) / f"checkpoint-{trainer.state.global_step}"
            if not last.is_dir():
                raise RuntimeError("Training finished without a selected Inspector checkpoint.")
            selected_checkpoint_raw = str(last)
        selected_checkpoint = Path(selected_checkpoint_raw).expanduser().resolve()
        trainer.state.best_model_checkpoint = str(selected_checkpoint)

        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        _restore_peft_checkpoint(unwrapped, selected_checkpoint)
        if not isinstance(unwrapped, PeftModel):
            raise TypeError("MS-Swift Inspector checkpoint must be a PEFT model.")
        adapter_state = _clone_tensor_dict(get_peft_model_state_dict(unwrapped))
        if not any("overall_inspector_head" in key for key in adapter_state):
            raise RuntimeError("PEFT checkpoint is missing overall_inspector_head.")
        checkpoint_path = Path(self.args.output_dir) / CHECKPOINT_FILENAME
        if trainer.is_world_process_zero():
            checkpoint = {
                "format_version": 3,
                "model_role": "overall_inspector",
                "training_framework": "ms-swift",
                "loss_type": LOSS_TYPE,
                "outputs": ["P(A)", "P(B)", "P(Tie)"],
                "head": "score_head + symmetric_tie_head, one overall output",
                "tie_head_mode": resolve_tie_head_mode(),
                "loss": {
                    "name": "three_way_cross_entropy",
                    "formula": "BCE(y_T,t)+(y_A+y_B)BCE(y_A/(y_A+y_B),w)",
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_model_path": str(Path(self.args.model).resolve()),
                "ms_swift_best_checkpoint": trainer.state.best_model_checkpoint,
                "adapter_state_dict": adapter_state,
                "peft_config": unwrapped.peft_config["default"].to_dict(),
                "prompt": {
                    "config_path": str(_PROJECT_ROOT / "prompts" / "reward_model_template.json"),
                    "profile": OVERALL_INSPECTOR_PROMPT_CONFIG.profile,
                    "system_message": OVERALL_INSPECTOR_SYSTEM_MESSAGE,
                    "evaluation_instruction": OVERALL_INSPECTOR_EVALUATION_INSTRUCTION,
                },
                "data": {
                    "dataset_path": self._inspector_dataset_path,
                    "editreward_root": self._inspector_editreward_root,
                    "stats": self._inspector_data_stats,
                    "split": self._inspector_split,
                    "data_passes": self._inspector_passes,
                    "train_order_file": TRAIN_ORDER_FILENAME,
                    "train_order_sha256": self._inspector_order_digest,
                    "train_votes": count_votes(self._inspector_train_records),
                    "validation_votes": count_votes(self._inspector_validation_records),
                    "train_sample_ids": [r.sample_id for r in self._inspector_train_records],
                    "validation_sample_ids": [
                        r.sample_id for r in self._inspector_validation_records
                    ],
                },
                "selection": {
                    "criterion": (
                        self.args.metric_for_best_model
                        if self._inspector_validation_records
                        else "last_checkpoint"
                    ),
                    "best_metric": trainer.state.best_metric,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                },
                "frozen_for_inference": True,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved consolidated Inspector checkpoint: %s", checkpoint_path)
        for parameter in unwrapped.parameters():
            parameter.requires_grad_(False)
        unwrapped.eval()
        trainer.accelerator.wait_for_everyone()
        return result


def main():
    pipeline = OverallInspectorSwiftSft()
    return pipeline.main()


if __name__ == "__main__":
    main()
