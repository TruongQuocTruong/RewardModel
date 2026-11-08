#!/usr/bin/env python3
"""Student warm-start initialized from the Overall Inspector Model LoRA weights.

Identical to ``train_student_warmstart.py`` (GemmaGT Data, factorized
warm-start loss, same split and hyperparameters) except for one change: before
optimization, every LoRA matrix of the Student Model is overwritten with the
corresponding LoRA matrix of the Overall Inspector Model, which was supervised
on HumanGT Data. The Inspector's ``overall_inspector_head`` is discarded; the
Student ``metric_heads`` keep their usual initialization.

The Inspector checkpoint directory is read from
``STUDENT_INSPECTOR_TRANSFER_CHECKPOINT`` (a PEFT directory containing
``adapter_model.safetensors``).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors

from train_student_warmstart import CHECKPOINT_FILENAME, MetricWarmStartSwiftSft, logger

INSPECTOR_HEAD_PREFIX = "base_model.model.overall_inspector_head."


def _load_inspector_lora(model: torch.nn.Module, checkpoint_dir: Path) -> int:
    """Copy all Overall Inspector Model LoRA matrices into the Student Model."""
    if not isinstance(model, PeftModel):
        raise TypeError("Inspector transfer requires a PEFT Student model.")
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise FileNotFoundError(f"Inspector PEFT weights are missing or empty: {weights_path}")
    state = load_safetensors(str(weights_path), device="cpu")
    lora_state = {key: value for key, value in state.items() if "lora_" in key}
    dropped = sorted(set(state) - set(lora_state))
    if any(not key.startswith(INSPECTOR_HEAD_PREFIX) for key in dropped):
        raise RuntimeError(f"Unexpected non-LoRA Inspector keys: {dropped[:10]}")

    student_lora_keys = {
        key for key in get_peft_model_state_dict(model) if "lora_" in key
    }
    if set(lora_state) != student_lora_keys:
        raise RuntimeError(
            "Inspector and Student LoRA layouts differ: "
            f"only Inspector={sorted(set(lora_state) - student_lora_keys)[:5]}, "
            f"only Student={sorted(student_lora_keys - set(lora_state))[:5]}"
        )
    load_result = set_peft_model_state_dict(model, lora_state, adapter_name="default")
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading Inspector LoRA: {unexpected[:10]}")

    loaded = get_peft_model_state_dict(model)
    for key, value in lora_state.items():
        if not torch.equal(loaded[key].detach().cpu().to(value.dtype), value):
            raise RuntimeError(f"Inspector LoRA tensor was not copied: {key}")
    logger.info(
        "Initialized %d Student LoRA tensors from the Overall Inspector Model (%s); "
        "dropped Inspector head tensors: %s",
        len(lora_state),
        checkpoint_dir,
        dropped,
    )
    return len(lora_state)


class StudentInspectorTransferSwiftSft(MetricWarmStartSwiftSft):
    def train(self, trainer):
        if os.environ.get("STUDENT_WARMSTART_FINALIZE_CHECKPOINT", "").strip():
            raise ValueError("Inspector transfer does not support finalize-only mode.")
        checkpoint_raw = os.environ.get("STUDENT_INSPECTOR_TRANSFER_CHECKPOINT", "").strip()
        if not checkpoint_raw:
            raise ValueError("Set STUDENT_INSPECTOR_TRANSFER_CHECKPOINT to the Inspector PEFT directory.")
        inspector_dir = Path(checkpoint_raw).expanduser().resolve()
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        num_tensors = _load_inspector_lora(unwrapped, inspector_dir)

        result = super().train(trainer)

        if trainer.is_world_process_zero():
            checkpoint_path = Path(self.args.output_dir) / CHECKPOINT_FILENAME
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["student_initialization"] = {
                "source": "overall_inspector_lora",
                "inspector_checkpoint": str(inspector_dir),
                "num_lora_tensors": num_tensors,
                "metric_heads": "default initialization",
            }
            tmp_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, checkpoint_path)
            logger.info("Recorded Student initialization in: %s", checkpoint_path)
        return result


def main():
    return StudentInspectorTransferSwiftSft().main()


if __name__ == "__main__":
    main()
