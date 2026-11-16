"""Local judge models. Each judge maps a judge call content [("text", str) | ("image", path, resize)] to text.

gemma-4-31B-it          env gemma4 (transformers >= 5.10), bf16, greedy, enable_thinking=False
Qwen2.5-VL-72B-AWQ      env with AWQ support; fp16, device_map auto, greedy + seed 42 before each call
                        (Step1X-Edit/GEdit-Bench/viescore/mllm_tools/qwen25vl_eval.py)
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from bench_protocols import gedit_resize_size

JUDGES = ("gemma-4-31B-it", "Qwen2.5-VL-72B-Instruct-AWQ")


def load_image(path: str, resize: str | None):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
    if resize == "gedit_512_area":
        image = image.resize(gedit_resize_size(image.width, image.height))  # Pillow default, as GEdit
    elif resize is not None:
        raise ValueError(f"Unknown resize rule: {resize!r}")
    return image


def to_messages(content: Sequence[Tuple]) -> List[dict]:
    parts = []
    for entry in content:
        if entry[0] == "text":
            parts.append({"type": "text", "text": entry[1]})
        elif entry[0] == "image":
            parts.append({"type": "image", "image": load_image(entry[1], entry[2])})
        else:
            raise ValueError(f"Unknown content entry: {entry[0]!r}")
    return [{"role": "user", "content": parts}]


class GemmaJudge:
    name = "gemma-4-31B-it"

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: submit the Slurm job")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
        ).eval().requires_grad_(False).to("cuda")

    def generate(self, content, max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            to_messages(content), tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        ).to("cuda")
        for key, value in inputs.items():
            if self.torch.is_floating_point(value):
                inputs[key] = value.to(dtype=self.torch.bfloat16)
        length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        return self.processor.decode(out[0, length:], skip_special_tokens=True)


def patch_transformers_for_autoawq() -> None:
    """autoawq 0.2.9 (last release) imports PytorchGELUTanh, renamed GELUTanh in transformers 4.57 (same tanh GELU)."""
    import transformers.activations as activations
    if not hasattr(activations, "PytorchGELUTanh"):
        activations.PytorchGELUTanh = activations.GELUTanh


class QwenAwqJudge:
    name = "Qwen2.5-VL-72B-Instruct-AWQ"

    def __init__(self, model_path: str):
        import numpy as np
        import torch
        from qwen_vl_utils import process_vision_info
        patch_transformers_for_autoawq()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: submit the Slurm job")
        self.torch, self.np, self.process_vision_info = torch, np, process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    def generate(self, content, max_new_tokens: int) -> str:
        random.seed(42)
        self.np.random.seed(42)
        self.torch.manual_seed(42)
        self.torch.cuda.manual_seed_all(42)
        messages = to_messages(content)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True,
                                return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1,
                                      do_sample=False, temperature=0.1, top_p=None)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        return decoded[0] if decoded else ""


def load_judge(name: str, model_path: str):
    if name == GemmaJudge.name:
        return GemmaJudge(model_path)
    if name == QwenAwqJudge.name:
        return QwenAwqJudge(model_path)
    raise ValueError(f"Unknown judge: {name!r}")
