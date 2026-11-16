#!/usr/bin/env python
"""Score AURA-Test pairs with a base VLM judge (Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3.5-9B, Qwen3-VL-Reranker-8B or
Gemma-4-31B, no fine-tuning).

Prompts: EvaluationConfig/base_vlm_metric_prompts.json (system prompt + one user prompt per metric).
Every metric gets the source image, then the edited image, then the prompt with the instruction.
Each edited image of a pair is scored on its own: one forward pass, no text generation; the score is the
expected value of the digits 1..9 under the next-token probabilities, renormalised over the nine digits.
Images are kept at their original size (processor defaults, no resizing, no max_pixels).
Qwen3.5-9B and Gemma-4-31B run with thinking disabled (enable_thinking=False), so the first generated token is the digit.

One job scores the whole bench (all 1561 pairs) for one model; the bench is never split.
Output (in --output-dir, default results/<model>): scores.jsonl, run_config.json.
Re-running the same command resumes (already scored sample_ids are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import bench_inputs as bi

DEFAULT_PROMPTS = bi.COMPARISON_ROOT / "EvaluationConfig" / "base_vlm_metric_prompts.json"
DIGITS = [str(d) for d in range(1, 10)]
VARIANTS = {
    "Qwen2.5-VL-7B": {"model_path": bi.MODELS_DIR / "Qwen2.5-VL-7B", "conda_env": "qwen_vl", "chat_template_kwargs": {}},
    "Qwen3-VL-8B": {"model_path": bi.MODELS_DIR / "Qwen3-VL-8B", "conda_env": "qwen_vl", "chat_template_kwargs": {}},
    "Qwen3.5-9B": {"model_path": bi.MODELS_DIR / "Qwen3.5-9B", "conda_env": "gemma4",
                   "chat_template_kwargs": {"enable_thinking": False}},
    # Backbone of our pipeline, zero-shot (no LoRA, no heads); its lm_head is the full vocabulary.
    "Qwen3-VL-Reranker-8B": {"model_path": bi.COMPARISON_ROOT.parent / "Models" / "Qwen3-VL-Reranker-8B",
                             "conda_env": "qwen_vl", "chat_template_kwargs": {}},
    # Gemma4 needs transformers 5.x (env gemma4); 31B in bf16 (about 59 GB of weights) runs on the full partition.
    "Gemma-4-31B": {"model_path": Path("/datastore/shared/models/gemma-4-31B-it"), "conda_env": "gemma4",
                    "chat_template_kwargs": {"enable_thinking": False}},
}
# End of the rendered prompt when thinking is disabled: Qwen closes an empty think block, Gemma an empty thought channel.
THINKING_CLOSED_SUFFIXES = ("</think>", "<channel|>")
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")


def load_prompts(path: Path = DEFAULT_PROMPTS) -> dict:
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, entry in prompts["metrics"].items():
        if entry.get("images") != ["source", "edited"]:
            raise ValueError(f"prompt {key!r}: expected images ['source', 'edited'], got {entry.get('images')}")
        if "{instruction}" not in entry["user_prompt"]:
            raise ValueError(f"prompt {key!r} has no {{instruction}} placeholder")
    return prompts


def build_messages(prompts: dict, prompt_key: str, instruction: str, source_image, edited_image) -> List[dict]:
    """Chat messages: system prompt, then user turn = [source image, edited image, metric prompt]."""
    if prompt_key not in prompts["metrics"]:
        raise ValueError(f"no base VLM prompt for {prompt_key!r}")
    text = prompts["metrics"][prompt_key]["user_prompt"].replace("{instruction}", instruction)
    return [
        {"role": "system", "content": [{"type": "text", "text": prompts["system_prompt"]}]},
        {"role": "user", "content": [{"type": "image", "image": source_image},
                                     {"type": "image", "image": edited_image},
                                     {"type": "text", "text": text}]},
    ]


def expected_digit_score(digit_probs: Sequence[float]) -> dict:
    """Expected value over 1..9 of the digit probabilities renormalised to sum 1; mass = their raw sum."""
    if len(digit_probs) != len(DIGITS):
        raise ValueError(f"need {len(DIGITS)} digit probabilities, got {len(digit_probs)}")
    probs = [float(p) for p in digit_probs]
    if any(p != p or p < 0 for p in probs):
        raise ValueError(f"digit probabilities must be non-negative numbers: {probs}")
    mass = sum(probs)
    if mass <= 0:
        raise ValueError("all digit probabilities are zero")
    normalised = [p / mass for p in probs]
    return {"score": sum(n * (i + 1) for i, n in enumerate(normalised)), "digit_probs": normalised, "digit_mass": mass}


def score_rows(next_token: Callable, prompts: dict, rows, prompt_key_of_metric: Dict[str, str], score_path: Path,
               load_images=None, log=print) -> dict:
    """Score both edited images of every row; append records to score_path; resume-safe.

    next_token(messages) -> {"digit_probs": [9 raw probabilities], "top_token": str, "num_input_tokens": int}.
    """
    load_images = load_images or (lambda row: load_pair_images(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    counts = {"scored": 0, "skipped_done": 0, "not_applicable": 0}
    with open(score_path, "a", encoding="utf-8") as score_file:
        for row in rows:
            prompt_key = prompt_key_of_metric.get(row["metric"])
            if prompt_key is None:
                counts["not_applicable"] += 1
                continue
            if prompt_key not in prompts["metrics"]:
                raise ValueError(f"mapping gives {prompt_key!r} for {row['metric']!r}, which has no base VLM prompt")
            if (row["sample_id"], prompt_key) in done:
                counts["skipped_done"] += 1
                continue
            started = time.time()
            source, edited = load_images(row)
            results = []
            for image in edited:
                output = next_token(build_messages(prompts, prompt_key, row["prompt"], source, image))
                results.append(dict(output, **expected_digit_score(output["digit_probs"])))
            seconds = round(time.time() - started, 2)
            record = {"sample_id": row["sample_id"], "dimension": prompt_key,
                      "score_a": results[0]["score"], "score_b": results[1]["score"],
                      "margin": results[0]["score"] - results[1]["score"]}
            record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
            for side, result in zip("ab", results):
                record.update({f"digit_probs_{side}": result["digit_probs"], f"digit_mass_{side}": result["digit_mass"],
                               f"top_token_{side}": result["top_token"],
                               f"num_input_tokens_{side}": result["num_input_tokens"]})
            record.update({"image_size": list(getattr(source, "size", ())) or None,
                           "edited_image_sizes": [list(getattr(im, "size", ())) or None for im in edited],
                           "seconds": seconds})
            score_file.write(json.dumps(record) + "\n")
            score_file.flush()
            counts["scored"] += 1
            log(f"[{counts['scored']}] {row['sample_id']} {prompt_key}: "
                f"{results[0]['score']:.3f} vs {results[1]['score']:.3f} "
                f"(digit mass {results[0]['digit_mass']:.3f}/{results[1]['digit_mass']:.3f}, {seconds}s)")
    return counts


def load_pair_images(row: dict, source_root: Path, edited_root: Path):
    """PIL RGB images at their original size (no resizing)."""
    from PIL import Image

    source_path, edited_paths = bi.image_paths(row, source_root, edited_root)
    return Image.open(source_path).convert("RGB"), [Image.open(p).convert("RGB") for p in edited_paths]


def digit_token_ids(tokenizer) -> List[int]:
    """Token id of each digit 1..9; raises if a digit is not exactly one token."""
    ids = []
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"digit {digit!r} is {len(encoded)} tokens, expected 1")
        ids.append(encoded[0])
    return ids


class NextTokenJudge:
    """Loads one base VLM and returns the next-token digit probabilities for a chat."""

    def __init__(self, model_path: Path, chat_template_kwargs: dict):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.chat_template_kwargs = dict(chat_template_kwargs)
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
        self.digit_ids = digit_token_ids(self.processor.tokenizer)

    def rendered_prompt(self, messages) -> str:
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.chat_template_kwargs)

    def __call__(self, messages) -> dict:
        torch = self.torch
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt",
                                                    **self.chat_template_kwargs).to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top_id = int(probs.argmax())
        return {"digit_probs": probs[self.digit_ids].tolist(),
                "top_token": self.processor.tokenizer.decode([top_id]),
                "num_input_tokens": int(inputs["input_ids"].shape[1])}


def check_thinking_disabled(rendered: str, chat_template_kwargs: dict) -> None:
    """With enable_thinking=False the prompt must end with a closed, empty thinking block, so the next token is the digit."""
    if chat_template_kwargs.get("enable_thinking") is False and not rendered.rstrip().endswith(THINKING_CLOSED_SUFFIXES):
        raise RuntimeError("enable_thinking=False did not close the think block; the digit would not come first")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/<model>")
    parser.add_argument("--per-metric-limit", type=int, default=None, help="smoke test: first N rows per metric")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    prompts = load_prompts(args.prompts)
    rows = bi.load_bench_rows(args.bench or Path(mapping["bench"]))
    prompt_key_of_metric = bi.dimension_by_bench_metric(mapping, args.model, "-")
    selected = bi.select_rows(rows, per_metric_limit=args.per_metric_limit)
    variant = VARIANTS[args.model]
    output_dir = args.output_dir or bi.DEFAULT_RESULTS / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(variant["model_path"]).is_dir():
        raise FileNotFoundError(f"missing local weights: {variant['model_path']}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers

    config = {
        "model": args.model, "model_path": str(variant["model_path"]),
        "chat_template_kwargs": variant["chat_template_kwargs"], "scoring": "expected digit 1..9, next-token probabilities",
        "image_resizing": "none (processor defaults)", "bench": str(args.bench or mapping["bench"]),
        "rows_selected": len(selected), "per_metric_limit": args.per_metric_limit,
        "prompt_key_of_metric": prompt_key_of_metric, "prompts_file": str(args.prompts),
        "prompts_md5": bi.file_md5(args.prompts), "host": socket.gethostname(), "user": os.environ.get("USER"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_md5": bi.file_md5(Path(__file__)), "torch": torch.__version__, "transformers": transformers.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    load_started = time.time()
    judge = NextTokenJudge(variant["model_path"], variant["chat_template_kwargs"])
    config["load_seconds"] = round(time.time() - load_started, 1)
    config["digit_token_ids"] = judge.digit_ids
    sample = judge.rendered_prompt(build_messages(prompts, "Fidelity", "<instruction>", None, None))
    check_thinking_disabled(sample, variant["chat_template_kwargs"])
    config["rendered_prompt_tail"] = sample[-200:]
    counts = score_rows(judge, prompts, selected, prompt_key_of_metric, output_dir / "scores.jsonl",
                        log=lambda message: print(message, flush=True))
    config.update(counts)
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        config["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
