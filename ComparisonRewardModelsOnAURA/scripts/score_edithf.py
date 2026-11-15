#!/usr/bin/env python
"""Score AURA-Test pairs with EditHF (InternVL3.5-8B + vision/LLM LoRA + MLP head per mode).

Follows EditHF/inference.py infer_single (repo not modified), but loads the model once per mode instead of
once per image: source and edited image resized to 448x448 (build_transform), query = build_query(instruction,
mode) + " The quality score is 00.00" with the <image> placeholders expanded, last hidden state at position -6,
score = |MLP(feature)| * 100. Only the local backbone path replaces the hard-coded "OpenGVLab/InternVL3_5-8B".

Mode (LoRA folder) per dimension, from metric_mapping.json:
    Visual Quality         -> visual       (lora_checkpoints_visual)        Aesthetic, Realism
    Instruction Alignment  -> alignment    (lora_checkpoints_editing)       Fidelity
    Attribute Preservation -> preservation (lora_checkpoints_preservation)  BC, FC, SC
One job loads the backbone with one LoRA set, scores all rows of that mode, frees it, then loads the next.
Each edited image of a pair is scored on its own (as infer_single).

One job scores the whole bench (all 1561 pairs); the bench is never split.
Output (in --output-dir, default results/EditHF_InternVL3.5-8B): scores.jsonl, run_config.json.
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

MODEL_NAME = "EditHF"
PRETRAINED = "InternVL3.5-8B"
EDITHF_REPO = bi.COMPARISON_ROOT / "EditHF"
BACKBONE_PATH = bi.MODELS_DIR / "InternVL3.5-8B"
INPUT_SIZE = 448  # default of inference.py
FEATURE_POSITION = -6  # hidden_states[-1][:, -6, :] in infer_single
QUERY_SUFFIX = " The quality score is 00.00"
MODES = {  # dimension -> (mode of build_query, LoRA folder); dict order = scoring order
    "Visual Quality": ("visual", EDITHF_REPO / "lora_checkpoints_visual"),
    "Instruction Alignment": ("alignment", EDITHF_REPO / "lora_checkpoints_editing"),
    "Attribute Preservation": ("preservation", EDITHF_REPO / "lora_checkpoints_preservation"),
}
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")


def load_edithf_module():
    """EditHF/inference.py imported under a unique name (it is not a package)."""
    spec = importlib.util.spec_from_file_location("edithf_inference", EDITHF_REPO / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand_query(edithf, instruction: str, mode: str, num_image_token: int, num_patches_list) -> str:
    """Query text exactly as infer_single builds it."""
    query = edithf.build_query(instruction, mode) + QUERY_SUFFIX
    for num_patches in num_patches_list:
        image_tokens = (edithf.IMG_START_TOKEN + edithf.IMG_CONTEXT_TOKEN * num_image_token * num_patches
                        + edithf.IMG_END_TOKEN)
        query = query.replace("<image>", image_tokens, 1)
    if "<image>" in query:
        raise ValueError("query still has an <image> placeholder")
    return query


def regression_score(mlp_output: float) -> float:
    """infer_single: abs(mlp(feats)) * 1e2."""
    value = float(mlp_output)
    if value != value:
        raise ValueError("MLP output is NaN")
    return abs(value) * 1e2


def rows_by_dimension(rows, dimension_of_metric):
    """{dimension: [rows]} in MODES order (rows keep file order); unmapped metrics are counted, unknown raise."""
    grouped, not_applicable = {d: [] for d in MODES}, 0
    for row in rows:
        dimension = dimension_of_metric.get(row["metric"])
        if dimension is None:
            not_applicable += 1
        elif dimension not in MODES:
            raise ValueError(f"mapping gives {dimension!r} for {row['metric']!r}, not an EditHF dimension")
        else:
            grouped[dimension].append(row)
    return grouped, not_applicable


def score_rows(load_scorer, rows, dimension_of_metric, score_path: Path, load_images=None, log=print) -> dict:
    """For each mode with rows left: scorer = load_scorer(dimension); score both images of its rows; release it.

    scorer(source_image, edited_image, instruction) -> float; scorer.release() frees the model.
    """
    load_images = load_images or (lambda row: load_pair_images(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    grouped, not_applicable = rows_by_dimension(rows, dimension_of_metric)
    counts = {"scored": 0, "skipped_done": 0, "not_applicable": not_applicable, "modes_loaded": []}
    with open(score_path, "a", encoding="utf-8") as score_file:
        for dimension, dimension_rows in grouped.items():
            todo = [r for r in dimension_rows if (r["sample_id"], dimension) not in done]
            counts["skipped_done"] += len(dimension_rows) - len(todo)
            if not todo:
                continue
            mode, lora_dir = MODES[dimension]
            started_load = time.time()
            scorer = load_scorer(dimension)
            counts["modes_loaded"].append({"dimension": dimension, "mode": mode, "lora_dir": str(lora_dir),
                                           "rows": len(todo), "load_seconds": round(time.time() - started_load, 1)})
            try:
                for row in todo:
                    started = time.time()
                    source, edited = load_images(row)
                    score_a, score_b = (scorer(source, image, row["prompt"]) for image in edited)
                    seconds = round(time.time() - started, 2)
                    record = {"sample_id": row["sample_id"], "dimension": dimension,
                              "score_a": score_a, "score_b": score_b, "margin": score_a - score_b}
                    record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
                    record.update({"mode": mode, "lora_dir": str(lora_dir),
                                   "image_size": list(getattr(source, "size", ())) or None,
                                   "edited_image_sizes": [list(getattr(im, "size", ())) or None for im in edited],
                                   "seconds": seconds})
                    score_file.write(json.dumps(record) + "\n")
                    score_file.flush()
                    counts["scored"] += 1
                    log(f"[{counts['scored']}] {row['sample_id']} {dimension}: {score_a:.3f} vs {score_b:.3f} ({seconds}s)")
            finally:
                scorer.release()
    return counts


def load_pair_images(row: dict, source_root: Path, edited_root: Path):
    """PIL RGB images as infer_single opens them (resizing happens in build_transform)."""
    from PIL import Image

    source_path, edited_paths = bi.image_paths(row, source_root, edited_root)
    return Image.open(source_path).convert("RGB"), [Image.open(p).convert("RGB") for p in edited_paths]


class EditHFScorer:
    """Backbone + LoRA + MLP of one mode, loaded once; __call__ follows the forward part of infer_single."""

    def __init__(self, edithf, dimension: str, model_path: Path = BACKBONE_PATH, device: str = "cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch, self.edithf = torch, edithf
        self.mode, peft_dir = MODES[dimension]
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        self.transform = edithf.build_transform(INPUT_SIZE)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(edithf.IMG_CONTEXT_TOKEN)
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        model = AutoModel.from_pretrained(str(model_path), torch_dtype=self.dtype, trust_remote_code=True)
        model.img_context_token_id = img_context_token_id
        model.vision_model = PeftModel.from_pretrained(model.vision_model, f"{peft_dir}/vision_lora", torch_dtype=self.dtype)
        model.language_model = PeftModel.from_pretrained(model.language_model, f"{peft_dir}/llm_lora", torch_dtype=self.dtype)
        self.model = model.to(self.device).eval()
        hidden_size = getattr(model.config, "hidden_size", 4096)
        self.mlp = edithf.MLPRegressor(input_dim=hidden_size, hidden_dim=hidden_size // 2, hidden_dim2=128,
                                       output_dim=1).to(self.device)
        self.mlp.load_state_dict(torch.load(os.path.join(peft_dir, "mlp_head.pth"), map_location=self.device))
        self.mlp.eval()

    def __call__(self, source_image, edited_image, instruction: str) -> float:
        torch = self.torch
        pixel_values1 = self.transform(source_image).unsqueeze(0).to(self.dtype).to(self.device)
        pixel_values2 = self.transform(edited_image).unsqueeze(0).to(self.dtype).to(self.device)
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
        query = expand_query(self.edithf, instruction, self.mode, self.num_image_token, num_patches_list)
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        image_flags = torch.ones((len(num_patches_list), num_patches_list[-1]), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values, input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"], image_flags=image_flags,
                                 output_hidden_states=True)
            feats = outputs.hidden_states[-1][:, FEATURE_POSITION, :].detach().to(torch.float32)
            return regression_score(self.mlp(feats).squeeze(1).item())

    def release(self):
        del self.model, self.mlp
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/EditHF_InternVL3.5-8B")
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
    for path in [BACKBONE_PATH] + [lora for _, lora in MODES.values()]:
        if not Path(path).is_dir():
            raise FileNotFoundError(f"missing local weights: {path}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import peft
    import torch
    import transformers

    edithf = load_edithf_module()
    config = {
        "model": MODEL_NAME, "pretrained": PRETRAINED, "backbone_path": str(BACKBONE_PATH),
        "modes": {d: {"mode": m, "lora_dir": str(p)} for d, (m, p) in MODES.items()},
        "input_size": INPUT_SIZE, "feature_position": FEATURE_POSITION, "query_suffix": QUERY_SUFFIX,
        "score": "abs(MLP(last hidden state at -6)) * 100, as EditHF/inference.py",
        "bench": str(args.bench or mapping["bench"]), "rows_selected": len(selected),
        "per_metric_limit": args.per_metric_limit, "dimension_of_metric": dimension_of_metric,
        "host": socket.gethostname(), "user": os.environ.get("USER"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "script_md5": bi.file_md5(Path(__file__)),
        "edithf_inference_md5": bi.file_md5(EDITHF_REPO / "inference.py"),
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    counts = score_rows(lambda dimension: EditHFScorer(edithf, dimension), selected, dimension_of_metric,
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
