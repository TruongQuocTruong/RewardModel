#!/usr/bin/env python3
"""Run the frozen three-way Overall Inspector once on Unlabeled Data and cache pseudo-labels.

Each row stores P(A), P(B), P(Tie) and the decision with the largest
probability. How Stage 2 consumes the Tie decision is decided separately.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import LoraConfig, PeftConfig, get_peft_model, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.custom_qwen3_vl_reranker import (  # noqa: E402
    _pool_qwen_last_hidden,
    install_overall_inspector_head,
    tie_head_mode_from_state_keys,
)
from src.prompt_config import (  # noqa: E402
    OVERALL_INSPECTOR_EVALUATION_INSTRUCTION,
    OVERALL_INSPECTOR_SYSTEM_MESSAGE,
)

CONSOLIDATED_CHECKPOINT_FILENAME = "overall_inspector_checkpoint.pt"

DEFAULT_EDITREWARD_ROOT = (
    "/datastore/shared/datasets/EditReward-Data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_EDITREWARD_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional consistency check; inference always uses checkpoint temperature.",
    )
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _required_text(row: Dict[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: missing or empty {field!r}.")
    return value.strip()


def _resolve_image_path(raw_path: str, dataset_root: Path) -> str:
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


def load_records(path: Path, dataset_root: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            normalized_schema = "sample_id" in row
            fields = (
                ("sample_id", "source_image", "edit_instruction", "edited_image_A", "edited_image_B")
                if normalized_schema
                else ("unique_key", "source_img", "instruction", "left_output_image", "right_output_image")
            )
            sample_field, source_field, instruction_field, a_field, b_field = fields
            record = {
                "sample_id": _required_text(row, sample_field, line_number),
                "source_image": _resolve_image_path(
                    _required_text(row, source_field, line_number), dataset_root
                ),
                "edit_instruction": _required_text(row, instruction_field, line_number),
                "edited_image_A": _resolve_image_path(
                    _required_text(row, a_field, line_number), dataset_root
                ),
                "edited_image_B": _resolve_image_path(
                    _required_text(row, b_field, line_number), dataset_root
                ),
            }
            if record["sample_id"] in seen_ids:
                raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
            seen_ids.add(record["sample_id"])
            if record["edited_image_A"] == record["edited_image_B"]:
                raise ValueError(
                    f"Line {line_number}: candidates A and B are identical."
                )
            records.append(record)
    if not records:
        raise ValueError("Unlabeled Data is empty.")
    return records


def batched(records: List[Dict[str, str]], size: int) -> Iterable[List[Dict[str, str]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _format_candidate(
    record: Dict[str, str],
    candidate_field: str,
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
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(record['source_image'])}",
                },
                {
                    "type": "text",
                    "text": f" {record['edit_instruction']}\n<Document>:",
                },
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(record[candidate_field])}",
                },
            ],
        },
    ]


def _build_peft_config(config_dict: Dict[str, Any]) -> PeftConfig:
    # MS-Swift adds a few optional LoRA fields newer than the installed PEFT
    # constructor. Retain every field supported by this runtime.
    supported = set(inspect.signature(LoraConfig).parameters)
    filtered = {key: value for key, value in config_dict.items() if key in supported}
    filtered["inference_mode"] = True
    return PeftConfig.from_peft_type(**filtered)


def load_checkpoint_directory(checkpoint_dir: Path) -> Dict[str, Any]:
    """Build the checkpoint contract from an MS-Swift checkpoint-<step> folder.

    The prompt comes from the run's consolidated checkpoint, which belongs to
    the same training run; src.prompt_config is the fallback.
    """
    weights = checkpoint_dir / "adapter_model.safetensors"
    config_path = checkpoint_dir / "adapter_config.json"
    for path in (weights, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"MS-Swift checkpoint file is missing: {path}")
    adapter_state = load_safetensors(str(weights), device="cpu")
    if not any("overall_inspector_head" in key for key in adapter_state):
        raise RuntimeError(f"Checkpoint has no trained overall_inspector_head: {weights}")
    peft_config = json.loads(config_path.read_text(encoding="utf-8"))

    prompt: Dict[str, Any] = {
        "system_message": OVERALL_INSPECTOR_SYSTEM_MESSAGE,
        "evaluation_instruction": OVERALL_INSPECTOR_EVALUATION_INSTRUCTION,
        "source": "src.prompt_config",
    }
    consolidated = checkpoint_dir.parent / CONSOLIDATED_CHECKPOINT_FILENAME
    if consolidated.is_file():
        run_checkpoint = torch.load(
            consolidated, map_location="cpu", mmap=True, weights_only=False
        )
        run_prompt = run_checkpoint.get("prompt", {})
        if isinstance(run_prompt.get("system_message"), str):
            prompt = {**run_prompt, "source": str(consolidated)}
    return {
        "model_role": "overall_inspector",
        "frozen_for_inference": True,
        "base_model_path": peft_config.get("base_model_name_or_path"),
        "ms_swift_checkpoint": str(checkpoint_dir),
        "adapter_state_dict": adapter_state,
        "peft_config": peft_config,
        "prompt": prompt,
    }


def load_frozen_inspector(
    checkpoint_path: Path,
    device: torch.device,
    min_pixels: int | None,
    max_pixels: int | None,
    base_model_override: Path | None = None,
):
    if checkpoint_path.is_dir():
        checkpoint = load_checkpoint_directory(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_role") != "overall_inspector":
        raise ValueError("Checkpoint is not an Overall Inspector checkpoint.")
    if not checkpoint.get("frozen_for_inference"):
        raise ValueError("Inspector checkpoint is not marked frozen_for_inference.")
    base_model_path = (
        str(base_model_override.resolve())
        if base_model_override is not None
        else checkpoint.get("base_model_path")
    )
    if not isinstance(base_model_path, str) or not Path(base_model_path).is_dir():
        raise FileNotFoundError(f"Inspector base model is unavailable: {base_model_path}")

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
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    install_overall_inspector_head(
        base_model,
        processor.tokenizer,
        tie_head_mode=tie_head_mode_from_state_keys(checkpoint["adapter_state_dict"]),
    )
    peft_config = _build_peft_config(checkpoint["peft_config"])
    model = get_peft_model(base_model, peft_config)
    load_result = set_peft_model_state_dict(
        model, checkpoint["adapter_state_dict"], adapter_name="default"
    )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected Inspector checkpoint keys: {load_result.unexpected_keys[:10]}"
        )
    # Base weights and the frozen original head are expected to be missing; a
    # missing LoRA or head key would leave its initialization in place.
    missing = [
        key for key in load_result.missing_keys
        if "lora_" in key or "modules_to_save" in key
    ]
    if missing:
        raise RuntimeError(f"Missing Inspector checkpoint keys: {missing[:10]}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, processor, checkpoint


@torch.inference_mode()
def infer_batch(
    model,
    processor,
    records: List[Dict[str, str]],
    checkpoint: Dict[str, Any],
    device: torch.device,
    temperature: float,
) -> List[Dict[str, Any]]:
    prompt = checkpoint.get("prompt", {})
    system_message = prompt.get("system_message")
    evaluation_instruction = prompt.get("evaluation_instruction")
    if not isinstance(system_message, str) or not isinstance(
        evaluation_instruction, str
    ):
        raise ValueError("Inspector checkpoint is missing its prompt contract.")
    conversations: List[List[Dict[str, Any]]] = []
    for record in records:
        conversations.append(
            _format_candidate(
                record,
                "edited_image_A",
                system_message,
                evaluation_instruction,
            )
        )
        conversations.append(
            _format_candidate(
                record,
                "edited_image_B",
                system_message,
                evaluation_instruction,
            )
        )
    texts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )
    # Same pre-resize as MS-Swift training: qwen_vl_utils with the model patch size.
    images, videos = process_vision_info(
        conversations, image_patch_size=processor.image_processor.patch_size
    )
    inputs = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    attention_mask = inputs.get("attention_mask")
    hidden = _pool_qwen_last_hidden(model, inputs, attention_mask)
    raw_model = model.get_base_model()
    # Rows alternate A, B for every pair, as in training.
    head_output = raw_model.overall_inspector_head(hidden[0::2], hidden[1::2])
    scores_a = head_output.scores_A[:, 0].float()
    scores_b = head_output.scores_B[:, 0].float()
    margins = scores_a - scores_b
    # Temperature only rescales the A-versus-B margin; t is used as trained.
    cond_win_prob_a = torch.sigmoid(margins / temperature)
    tie_probs = head_output.tie_probs[:, 0].float()
    prob_a = (1.0 - tie_probs) * cond_win_prob_a
    prob_b = (1.0 - tie_probs) * (1.0 - cond_win_prob_a)
    three_way = torch.stack((prob_a, prob_b, tie_probs), dim=-1).cpu()

    output: List[Dict[str, Any]] = []
    decisions = ("A", "B", "Tie")
    for index, record in enumerate(records):
        probabilities = three_way[index]
        decision = decisions[int(probabilities.argmax().item())]
        output.append(
            {
                "sample_id": record["sample_id"],
                "pseudo_label": {"A": 1, "B": -1, "Tie": 0}[decision],
                "decision": decision,
                "probability_A": float(probabilities[0].item()),
                "probability_B": float(probabilities[1].item()),
                "probability_Tie": float(probabilities[2].item()),
                "cond_win_probability_A": float(cond_win_prob_a[index].item()),
                "signed_evidence": float((probabilities[0] - probabilities[1]).item()),
                "margin_A_minus_B": float(margins[index].item()),
                "score_A": float(scores_a[index].item()),
                "score_B": float(scores_b[index].item()),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.temperature is not None and args.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    # Either a consolidated .pt file or an MS-Swift checkpoint-<step> folder.
    if not (checkpoint_path.is_file() or checkpoint_path.is_dir()):
        raise FileNotFoundError(f"Inspector checkpoint does not exist: {checkpoint_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Unlabeled Data does not exist: {dataset_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Inspector cache: {output_path}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    records = load_records(dataset_path, dataset_root)
    model, processor, checkpoint = load_frozen_inspector(
        checkpoint_path,
        device,
        args.min_pixels,
        args.max_pixels,
    )
    checkpoint_temperature = float(checkpoint.get("probability_temperature", 1.0))
    if checkpoint_temperature <= 0.0:
        raise ValueError("Inspector checkpoint has an invalid probability temperature.")
    if args.temperature is not None and not torch.isclose(
        torch.tensor(args.temperature),
        torch.tensor(checkpoint_temperature),
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "--temperature does not match the temperature used for Inspector "
            f"calibration ({checkpoint_temperature})."
        )
    counts: Counter[str] = Counter()
    written = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for batch in batched(records, args.batch_size):
            results = infer_batch(
                model,
                processor,
                batch,
                checkpoint,
                device,
                checkpoint_temperature,
            )
            for result in results:
                handle.write(
                    json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                counts[result["decision"]] += 1
                written += 1
            handle.flush()
            print(f"Processed {written}/{len(records)}", flush=True)
    if written != len(records):
        raise RuntimeError("Inspector cache row count does not match the input dataset.")
    print(f"Overall Pseudo-label Cache: {output_path}")
    print(f"Decision counts: {dict(counts)}")


if __name__ == "__main__":
    main()
