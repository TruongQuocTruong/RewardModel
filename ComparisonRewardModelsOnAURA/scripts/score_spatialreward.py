#!/usr/bin/env python
"""Score AURA-Test pairs with SpatialReward-8B (Qwen3-VL-8B, full fine-tune) using transformers generate.

The original evaluation (SpatialReward/eval/EditReward-Bench, not modified) runs vLLM; here the same steps run
with transformers (no vLLM env): prompts SC_PROMPT_TEMPLATE / PQ_PROMPT and parse_output imported from the repo
dataset.py, the same messages (SC: source image, edited image, prompt; PQ: edited image, prompt),
processor.apply_chat_template + qwen_vl_utils.process_vision_info, AutoProcessor(max_pixels, min_pixels) and
the run.sh sampling settings: temperature 0.0 (greedy), max 4096 new tokens, seed 42.
Image size: run.sh sets MAX_PIXELS=589824 / MIN_PIXELS=3136, but the original code only gives them to
AutoProcessor, not to process_vision_info; here they are also put on each image message so they really apply
(see IMAGE_PIXELS_IN_MESSAGES). Greedy decoding with transformers can differ slightly from vLLM numerically.

Only the prompt needed for the row's metric is generated (metric_mapping.json), "min" aggregation of dataset.py:
    Aesthetic, Realism         -> Perceptual Quality   = min(naturalness, artifacts) / 2.5
    Fidelity, BC, FC, SC       -> Semantic Consistency = min(editing success, overediting) / 2.5
Each image of the pair is scored separately. When the output cannot be parsed, the original parse_output returns
[12.5, 12.5]; such images are flagged "parse_failed" (and "partial_json" when only the score list was recovered).

One job scores the whole bench (all 1561 pairs); the bench is never split.
Output (in --output-dir, default results/SpatialReward_Qwen3-VL-8B): scores.jsonl, raw_outputs.jsonl, run_config.json.
Re-running the same command resumes (already scored sample_ids are skipped).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import bench_inputs as bi

MODEL_NAME = "SpatialReward"
PRETRAINED = "Qwen3-VL-8B"
SPATIALREWARD_REPO = bi.COMPARISON_ROOT / "SpatialReward"
DATASET_MODULE = SPATIALREWARD_REPO / "eval" / "EditReward-Bench" / "dataset.py"
MODEL_PATH = bi.MODELS_DIR / "SpatialReward-8B"
# run.sh of eval/EditReward-Bench
MAX_PIXELS = 589824
MIN_PIXELS = 3136
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 4096
SEED = 42
SCORE_RANGE = 25
IMAGE_PIXELS_IN_MESSAGES = True
DIMENSIONS = ("Semantic Consistency", "Perceptual Quality")
SUB_SCORE_NAMES = {"Semantic Consistency": ["editing success", "overediting"],
                   "Perceptual Quality": ["naturalness", "artifacts"]}
PARSE_FAILED_REASONING = "Failed to parse output"
PARTIAL_JSON_REASONING = "Parsed from incomplete JSON"
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")


def load_dataset_module():
    """SpatialReward eval/EditReward-Bench/dataset.py imported under a unique name (prompts, parse_output)."""
    spec = importlib.util.spec_from_file_location("spatialreward_dataset", DATASET_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_entry(image, pixels_in_messages: bool = IMAGE_PIXELS_IN_MESSAGES) -> dict:
    entry = {"type": "image", "image": image}
    if pixels_in_messages:
        entry.update({"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS})
    return entry


def build_messages(module, dimension: str, instruction: str, source_image, edited_image,
                   pixels_in_messages: bool = IMAGE_PIXELS_IN_MESSAGES) -> list:
    """Messages exactly as EditScoreBatchDataset.__getitem__ builds them for the SC or the PQ prompt."""
    if dimension == "Semantic Consistency":
        content = [image_entry(source_image, pixels_in_messages), image_entry(edited_image, pixels_in_messages),
                   {"type": "text", "text": module.SC_PROMPT_TEMPLATE.format(instruction=instruction)}]
    elif dimension == "Perceptual Quality":
        content = [image_entry(edited_image, pixels_in_messages), {"type": "text", "text": module.PQ_PROMPT}]
    else:
        raise ValueError(f"unknown SpatialReward dimension {dimension!r}")
    return [{"role": "user", "content": content}]


def parse_status(result: dict) -> str:
    reasoning = str(result.get("reasoning", ""))
    if reasoning == PARSE_FAILED_REASONING:
        return "parse_failed"
    if reasoning == PARTIAL_JSON_REASONING:
        return "partial_json"
    return "parsed"


def dimension_score(sub_scores, score_range: int = SCORE_RANGE) -> float:
    """'min' aggregation of dataset.calculate_scores: min of the two sub-scores, on a 0-10 scale."""
    values = [float(s) for s in sub_scores]
    if len(values) != 2 or any(v != v for v in values):
        raise ValueError(f"expected two finite sub-scores, got {sub_scores!r}")
    return min(values) / (score_range / 10)


def score_image(generate, module, source_image, edited_image, instruction: str, dimension: str) -> dict:
    """One SpatialReward prompt on one edited image: generate, original parse_output, min / 2.5."""
    messages = build_messages(module, dimension, instruction, source_image, edited_image)
    output = generate(messages)
    text = output["text"].strip()  # post_process strips the vLLM text
    result = module.EditScoreDataset.parse_output(None, text, with_region=dimension == "Semantic Consistency")
    return {"score": dimension_score(result["score"]), "sub_scores": [float(s) for s in result["score"]],
            "status": parse_status(result), "reasoning": str(result.get("reasoning", "")),
            "edit_region": result.get("edit_region"), "text": text,
            "num_input_tokens": output.get("num_input_tokens"), "num_output_tokens": output.get("num_output_tokens")}


def load_pair_images(row: dict, source_root: Path, edited_root: Path):
    """PIL RGB images as dataset.py opens them (Image.open(path).convert('RGB'))."""
    from PIL import Image

    source_path, edited_paths = bi.image_paths(row, source_root, edited_root)
    return Image.open(source_path).convert("RGB"), [Image.open(p).convert("RGB") for p in edited_paths]


def score_rows(generate, module, rows, dimension_of_metric, score_path: Path, raw_path: Path,
               load_images=None, log=print) -> dict:
    """Score every row whose metric has a dimension; append to score_path / raw_path; resume-safe."""
    load_images = load_images or (lambda row: load_pair_images(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    counts = {"scored": 0, "skipped_done": 0, "not_applicable": 0, "parse_failed_images": 0, "partial_json_images": 0}
    with open(score_path, "a", encoding="utf-8") as score_file, open(raw_path, "a", encoding="utf-8") as raw_file:
        for row in rows:
            dimension = dimension_of_metric.get(row["metric"])
            if dimension is None:
                counts["not_applicable"] += 1
                continue
            if dimension not in DIMENSIONS:
                raise ValueError(f"mapping gives {dimension!r} for {row['metric']!r}, not a SpatialReward dimension")
            if (row["sample_id"], dimension) in done:
                counts["skipped_done"] += 1
                continue
            started = time.time()
            source, edited = load_images(row)
            results = [score_image(generate, module, source, image, row["prompt"], dimension) for image in edited]
            seconds = round(time.time() - started, 2)
            a, b = results
            record = {"sample_id": row["sample_id"], "dimension": dimension,
                      "score_a": a["score"], "score_b": b["score"], "margin": a["score"] - b["score"]}
            record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
            record.update({
                "sub_scores_a": a["sub_scores"], "sub_scores_b": b["sub_scores"],
                "sub_score_names": SUB_SCORE_NAMES[dimension],
                "status_a": a["status"], "status_b": b["status"],
                "reasoning_a": a["reasoning"], "reasoning_b": b["reasoning"],
                "edit_region_a": a["edit_region"], "edit_region_b": b["edit_region"],
                "num_input_tokens_a": a["num_input_tokens"], "num_input_tokens_b": b["num_input_tokens"],
                "num_output_tokens_a": a["num_output_tokens"], "num_output_tokens_b": b["num_output_tokens"],
                "image_size": list(getattr(source, "size", ())) or None, "seconds": seconds,
            })
            score_file.write(json.dumps(record) + "\n")
            raw_file.write(json.dumps({"sample_id": row["sample_id"], "dimension": dimension,
                                       "output_a": a["text"], "output_b": b["text"]}) + "\n")
            score_file.flush()
            raw_file.flush()
            counts["scored"] += 1
            counts["parse_failed_images"] += sum(r["status"] == "parse_failed" for r in results)
            counts["partial_json_images"] += sum(r["status"] == "partial_json" for r in results)
            log(f"[{counts['scored']}] {row['sample_id']} {dimension}: {a['score']:.2f} vs {b['score']:.2f} ({seconds}s)")
    return counts


class SpatialRewardGenerator:
    """SpatialReward-8B with transformers; __call__(messages) -> {'text', 'num_input_tokens', 'num_output_tokens'}."""

    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cuda"):
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch, self.process_vision_info = torch, process_vision_info
        self.processor = AutoProcessor.from_pretrained(str(model_path), max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()

    def __call__(self, messages) -> dict:
        torch = self.torch
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = self.process_vision_info(messages, image_patch_size=self.processor.image_processor.patch_size)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                            max_new_tokens=MAX_NEW_TOKENS)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        return {"text": self.processor.decode(new_tokens, skip_special_tokens=True),
                "num_input_tokens": int(inputs["input_ids"].shape[1]), "num_output_tokens": int(new_tokens.numel())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/SpatialReward_Qwen3-VL-8B")
    parser.add_argument("--per-metric-limit", type=int, default=None, help="smoke test: first N rows per metric")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows = bi.load_bench_rows(args.bench or Path(mapping["bench"]))
    dimension_of_metric = bi.dimension_by_bench_metric(mapping, MODEL_NAME, PRETRAINED)
    selected = bi.select_rows(rows, per_metric_limit=args.per_metric_limit)
    output_dir = args.output_dir or bi.DEFAULT_RESULTS / f"{MODEL_NAME}_{PRETRAINED}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"missing local weights: {MODEL_PATH}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers

    module = load_dataset_module()
    config = {
        "model": MODEL_NAME, "pretrained": PRETRAINED, "model_path": str(MODEL_PATH), "backend": "transformers generate",
        "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "image_pixels_in_messages": IMAGE_PIXELS_IN_MESSAGES,
        "temperature": TEMPERATURE, "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
        "score_aggregation": "min", "score_range": SCORE_RANGE,
        "bench": str(args.bench or mapping["bench"]), "rows_selected": len(selected),
        "per_metric_limit": args.per_metric_limit, "dimension_of_metric": dimension_of_metric,
        "host": socket.gethostname(), "user": os.environ.get("USER"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "script_md5": bi.file_md5(Path(__file__)),
        "dataset_module_md5": bi.file_md5(DATASET_MODULE),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    load_started = time.time()
    generator = SpatialRewardGenerator()
    config["load_seconds"] = round(time.time() - load_started, 1)
    counts = score_rows(generator, module, selected, dimension_of_metric, output_dir / "scores.jsonl",
                        output_dir / "raw_outputs.jsonl", log=lambda message: print(message, flush=True))
    config.update(counts)
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        config["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
