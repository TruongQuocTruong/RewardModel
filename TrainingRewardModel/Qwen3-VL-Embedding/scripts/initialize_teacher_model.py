#!/usr/bin/env python3
"""Clone a completed Student warm-start checkpoint into an EMA Teacher checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch


STUDENT_ROLE = "six_metric_student_warmstart"
TEACHER_ROLE = "six_metric_ema_teacher_initialized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_student_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if checkpoint.get("model_role") != STUDENT_ROLE:
        raise ValueError(
            f"Expected model_role={STUDENT_ROLE!r}, got {checkpoint.get('model_role')!r}."
        )
    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Student checkpoint has no adapter_state_dict.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("adapter_state_dict must map string names to tensors.")
    if not any("lora_" in key for key in state):
        raise ValueError("Student checkpoint contains no LoRA tensors.")
    if not any("metric_heads" in key for key in state):
        raise ValueError("Student checkpoint contains no metric-head tensors.")
    for key, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Student tensor contains non-finite values: {key}")
    required = ("base_model_path", "peft_config", "metric_order", "prompt", "data")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Student checkpoint is missing required metadata: {missing}")
    return state


def clone_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def assert_exact_state_match(
    student_state: Dict[str, torch.Tensor],
    teacher_state: Dict[str, torch.Tensor],
) -> None:
    if student_state.keys() != teacher_state.keys():
        raise AssertionError("Student and Teacher parameter names differ.")
    for name, student_tensor in student_state.items():
        teacher_tensor = teacher_state[name]
        if student_tensor.shape != teacher_tensor.shape:
            raise AssertionError(f"Shape mismatch for {name}.")
        if student_tensor.dtype != teacher_tensor.dtype:
            raise AssertionError(f"Dtype mismatch for {name}.")
        if student_tensor.data_ptr() == teacher_tensor.data_ptr():
            raise AssertionError(f"Teacher tensor still shares storage with Student: {name}")
        if not torch.equal(student_tensor.cpu(), teacher_tensor):
            raise AssertionError(f"Teacher tensor differs from Student: {name}")


def main() -> None:
    args = parse_args()
    student_path = Path(args.student_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not student_path.is_file():
        raise FileNotFoundError(f"Student checkpoint does not exist: {student_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Teacher checkpoint: {output_path}")

    student = torch.load(student_path, map_location="cpu", weights_only=False)
    if not isinstance(student, dict):
        raise TypeError("Student checkpoint must contain a dictionary.")
    student_state = validate_student_checkpoint(student)
    teacher_state = clone_state_dict(student_state)
    assert_exact_state_match(student_state, teacher_state)

    # Preserve every Student checkpoint field. Only the role/state-transition
    # fields are replaced, while the independently cloned tensors are stored
    # under the same adapter_state_dict contract.
    teacher = {
        key: value
        for key, value in student.items()
        if key != "adapter_state_dict"
    }
    teacher.update({
        "model_role": TEACHER_ROLE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initialized_from": {
            "student_checkpoint": str(student_path),
            "student_checkpoint_sha256": file_sha256(student_path),
            "student_created_at_utc": student.get("created_at_utc"),
            "copy_rule": "exact_tensor_clone",
        },
        "adapter_state_dict": teacher_state,
        "teacher_initialization": "completed_exact_tensor_clone",
        "ema": {
            "step": 0,
            "decay": None,
            "updated_parameters": ["lora", "metric_heads"],
            "receives_gradients": False,
            "has_optimizer": False,
        },
        "frozen_for_gradient_updates": True,
    })
    torch.save(teacher, output_path)

    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    reloaded_state = reloaded.get("adapter_state_dict")
    if not isinstance(reloaded_state, dict):
        raise RuntimeError("Saved Teacher checkpoint cannot be reloaded.")
    if teacher_state.keys() != reloaded_state.keys():
        raise RuntimeError("Reloaded Teacher checkpoint has different parameter names.")
    for name, expected in teacher_state.items():
        actual = reloaded_state[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise RuntimeError(f"Reloaded Teacher tensor metadata differs: {name}")
        if not torch.equal(expected, actual):
            raise RuntimeError(f"Reloaded Teacher tensor value differs: {name}")

    lora_tensors = sum("lora_" in name for name in teacher_state)
    head_tensors = sum("metric_heads" in name for name in teacher_state)
    tensor_values = sum(tensor.numel() for tensor in teacher_state.values())
    print(f"Student checkpoint: {student_path}")
    print(f"Teacher checkpoint: {output_path}")
    print(f"Copied tensors: {len(teacher_state)}")
    print(f"LoRA tensors: {lora_tensors}")
    print(f"Metric-head tensors: {head_tensors}")
    print(f"Copied tensor values: {tensor_values}")
    print("Exact tensor equality after reload: passed")


if __name__ == "__main__":
    main()
