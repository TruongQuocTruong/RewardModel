#!/usr/bin/env python
"""Score AURA-Test pairs with LMM4Edit (Qwen3-VL-8B-Instruct + LoRA + MLP score head per mode).

Uses the original LMM4Edit/inference.py (not modified), the forward part of infer_single: load_model_bundle,
resolve_image (images larger than 1048576 pixels are shrunk, the repo default), build_processor_inputs
(source image, edited image, build_query(instruction, mode)), forward_scores (last token of the last hidden
state -> ScoreRegressor), score = prediction * score_output_scale (100). The model is loaded once per mode
instead of once per image, and the local backbone path is passed so the repo never downloads from ModelScope
(regression_config.json points to /mnt/sda/..., which does not exist here).

Mode (weights folder) per dimension, from metric_mapping.json:
    Perceptual Quality     -> visual       (weights/visual)        Aesthetic, Realism
    Editing Alignment      -> alignment    (weights/alignment)     Fidelity
    Attribute Preservation -> preservation (weights/preservation)  BC, FC, SC
One job loads the backbone with one LoRA set, scores all rows of that mode, frees it, then loads the next.
Each edited image of a pair is scored on its own (as infer_single).

One job scores the whole bench (all 1561 pairs); the bench is never split.
Output (in --output-dir, default results/LMM4Edit_Qwen3-VL-8B): scores.jsonl, run_config.json.
Re-running the same command resumes (already scored sample_ids are skipped).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import bench_inputs as bi

MODEL_NAME = "LMM4Edit"
PRETRAINED = "Qwen3-VL-8B"
LMM4EDIT_REPO = bi.COMPARISON_ROOT / "LMM4Edit"
BACKBONE_PATH = bi.MODELS_DIR / "Qwen3-VL-8B"
MAX_PIXELS_PER_IMAGE = 1048576  # default of inference.py
MODES = {  # dimension -> (mode of build_query, weights folder); dict order = scoring order
    "Perceptual Quality": ("visual", LMM4EDIT_REPO / "weights" / "visual"),
    "Editing Alignment": ("alignment", LMM4EDIT_REPO / "weights" / "alignment"),
    "Attribute Preservation": ("preservation", LMM4EDIT_REPO / "weights" / "preservation"),
}
WEIGHT_FILES = ("adapter_config.json", "adapter_model.safetensors", "regression_config.json", "score_head.pth")
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")


def load_lmm4edit_module():
    """LMM4Edit/inference.py imported under a unique name (it is not a package)."""
    spec = importlib.util.spec_from_file_location("lmm4edit_inference", LMM4EDIT_REPO / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_score(prediction: float, score_output_scale: float) -> float:
    """infer_single: float(pred.item() * score_output_scale)."""
    value = float(prediction) * float(score_output_scale)
    if value != value:
        raise ValueError("LMM4Edit score is NaN")
    return value


def check_checkpoint_mode(checkpoint_dimension, mode: str) -> None:
    """infer_single raises when regression_config.json 'dimension' differs from --mode."""
    if checkpoint_dimension and checkpoint_dimension != mode:
        raise ValueError(f"checkpoint dimension mismatch: checkpoint is {checkpoint_dimension!r}, mode is {mode!r}")


def rows_by_dimension(rows, dimension_of_metric):
    """{dimension: [rows]} in MODES order (rows keep file order); unmapped metrics are counted, unknown raise."""
    grouped, not_applicable = {d: [] for d in MODES}, 0
    for row in rows:
        dimension = dimension_of_metric.get(row["metric"])
        if dimension is None:
            not_applicable += 1
        elif dimension not in MODES:
            raise ValueError(f"mapping gives {dimension!r} for {row['metric']!r}, not an LMM4Edit dimension")
        else:
            grouped[dimension].append(row)
    return grouped, not_applicable


def score_rows(load_scorer, rows, dimension_of_metric, score_path: Path, image_paths=None, log=print) -> dict:
    """For each mode with rows left: scorer = load_scorer(dimension); score both images of its rows; release it.

    scorer(source_path, edited_path, instruction) -> float; scorer.release() frees the model.
    """
    image_paths = image_paths or (lambda row: bi.image_paths(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    grouped, not_applicable = rows_by_dimension(rows, dimension_of_metric)
    counts = {"scored": 0, "skipped_done": 0, "not_applicable": not_applicable, "modes_loaded": []}
    with open(score_path, "a", encoding="utf-8") as score_file:
        for dimension, dimension_rows in grouped.items():
            todo = [r for r in dimension_rows if (r["sample_id"], dimension) not in done]
            counts["skipped_done"] += len(dimension_rows) - len(todo)
            if not todo:
                continue
            mode, peft_dir = MODES[dimension]
            started_load = time.time()
            scorer = load_scorer(dimension)
            counts["modes_loaded"].append({"dimension": dimension, "mode": mode, "peft_dir": str(peft_dir),
                                           "rows": len(todo), "load_seconds": round(time.time() - started_load, 1)})
            try:
                for row in todo:
                    started = time.time()
                    source, edited = image_paths(row)
                    score_a, score_b = (scorer(str(source), str(path), row["prompt"]) for path in edited)
                    seconds = round(time.time() - started, 2)
                    record = {"sample_id": row["sample_id"], "dimension": dimension,
                              "score_a": score_a, "score_b": score_b, "margin": score_a - score_b}
                    record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
                    record.update({"mode": mode, "peft_dir": str(peft_dir), "seconds": seconds})
                    score_file.write(json.dumps(record) + "\n")
                    score_file.flush()
                    counts["scored"] += 1
                    log(f"[{counts['scored']}] {row['sample_id']} {dimension}: {score_a:.3f} vs {score_b:.3f} ({seconds}s)")
            finally:
                scorer.release()
    return counts


class LMM4EditScorer:
    """Backbone + LoRA + score head of one mode, loaded once with load_model_bundle; __call__ = infer_single forward."""

    def __init__(self, lmm4edit, dimension: str, model_path: Path = BACKBONE_PATH, device: str = "cuda",
                 max_pixels_per_image: int = MAX_PIXELS_PER_IMAGE):
        import torch

        self.torch, self.lmm4edit, self.max_pixels_per_image = torch, lmm4edit, max_pixels_per_image
        self.mode, peft_dir = MODES[dimension]
        self.device = torch.device(device)
        self.processor, self.model, self.score_head, regression_config, checkpoint_dimension = \
            lmm4edit.load_model_bundle(model_path=str(model_path), peft_dir=str(peft_dir), device=self.device,
                                       merge_lora=False)
        check_checkpoint_mode(checkpoint_dimension, self.mode)
        self.score_output_scale = float(regression_config.get("score_output_scale", 100.0))

    def __call__(self, source_path: str, edited_path: str, instruction: str) -> float:
        source = self.lmm4edit.resolve_image(source_path, max_pixels_per_image=self.max_pixels_per_image)
        edited = self.lmm4edit.resolve_image(edited_path, max_pixels_per_image=self.max_pixels_per_image)
        inputs = self.lmm4edit.build_processor_inputs(processor=self.processor, source_image=source,
                                                      edited_image=edited, instruction=instruction, mode=self.mode)
        batch = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        for key in ("pixel_values", "image_grid_thw"):
            if key in inputs:
                batch[key] = inputs[key]
        prediction = self.lmm4edit.forward_scores(model=self.model, score_head=self.score_head, batch=batch,
                                                  device=self.device)
        return final_score(prediction.item(), self.score_output_scale)

    def release(self):
        del self.model, self.score_head, self.processor
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/LMM4Edit_Qwen3-VL-8B")
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
    if not BACKBONE_PATH.is_dir():
        raise FileNotFoundError(f"missing local backbone: {BACKBONE_PATH}")
    for _, peft_dir in MODES.values():
        for name in WEIGHT_FILES:
            if not (peft_dir / name).is_file():
                raise FileNotFoundError(f"missing LMM4Edit weights: {peft_dir / name}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import peft
    import torch
    import transformers

    lmm4edit = load_lmm4edit_module()
    config = {
        "model": MODEL_NAME, "pretrained": PRETRAINED, "backbone_path": str(BACKBONE_PATH),
        "modes": {d: {"mode": m, "peft_dir": str(p)} for d, (m, p) in MODES.items()},
        "max_pixels_per_image": MAX_PIXELS_PER_IMAGE, "merge_lora": False,
        "score": "ScoreRegressor(last token of the last hidden state) * score_output_scale, as LMM4Edit infer_single",
        "bench": str(args.bench or mapping["bench"]), "rows_selected": len(selected),
        "per_metric_limit": args.per_metric_limit, "dimension_of_metric": dimension_of_metric,
        "host": socket.gethostname(), "user": os.environ.get("USER"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "script_md5": bi.file_md5(Path(__file__)),
        "lmm4edit_inference_md5": bi.file_md5(LMM4EDIT_REPO / "inference.py"),
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    counts = score_rows(lambda dimension: LMM4EditScorer(lmm4edit, dimension), selected, dimension_of_metric,
                        output_dir / "scores.jsonl", log=lambda message: print(message, flush=True))
    config.update(counts)
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        config["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in counts.items() if k != "modes_loaded"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
