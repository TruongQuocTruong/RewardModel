#!/usr/bin/env python
"""Score AURA-Test pairs with EditReward (Qwen2.5-VL-7B or MiMo-VL-7B backbone, full fine-tuned checkpoint).

Uses the original EditReward code (EditReward/EditReward, not modified): the main inferencer of the README,
EditRewardInferencer (inference_edit.py, default config EditReward-Qwen2.5-7B-VL.yaml), for both backbones, with
its prompts and its image preprocessing (256*28*28 pixels per image). The optional "unified" EditRewardVLInferencer
is not used: it passes aggregation_threshold, which the Qwen2.5-VL reward model class does not accept (job 13000).
The checkpoint has two heads ("ranknet_multi_head", output [mu, log sigma]):
    head_0 <- prompt INSTRUCTION_EDIT_FOLLOWING -> "Instruction Following" (column Fidelity)
    head_1 <- prompt INSTRUCTION_EDIT_QUALITY   -> "Visual Quality"        (columns Aesthetic, Realism)
The official inferencer averages the two heads; here pooling_strategy is set to None on the loaded object so
each head is read on its own, and only the head needed for the row's metric is run. Score = mu.
Both edited images of a pair go in one batch of two, as in the official pairwise example.

The yaml of the repo names the backbone by its Hugging Face id; a copy with the local backbone path is written
to --output-dir/editreward_config.yaml (the repo file is not changed).

One job scores the whole bench (all 1561 pairs) for one variant; the bench is never split.
Output (in --output-dir): scores.jsonl, run_config.json, editreward_config.yaml.
Re-running the same command resumes (already scored sample_ids are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

import bench_inputs as bi

MODEL_NAME = "EditReward"
EDITREWARD_REPO = bi.COMPARISON_ROOT / "EditReward"
CONFIG_DIR = EDITREWARD_REPO / "EditReward" / "config"
VARIANTS = {
    "Qwen2.5-VL-7B": {"module": "EditReward.inference_edit", "inferencer": "EditRewardInferencer",
                      "config": CONFIG_DIR / "EditReward-Qwen2.5-7B-VL.yaml",
                      "backbone_path": bi.MODELS_DIR / "Qwen2.5-VL-7B",
                      "checkpoint_path": bi.MODELS_DIR / "EditReward-Qwen2.5-VL-7B"},
    "MiMo-VL-7B": {"module": "EditReward.inference_edit", "inferencer": "EditRewardInferencer",
                   "config": CONFIG_DIR / "EditReward-MiMo-VL-7B-SFT-2508.yaml",
                   "backbone_path": bi.MODELS_DIR / "MiMo-VL-7B-SFT-2508",
                   "checkpoint_path": bi.MODELS_DIR / "EditReward-MiMo-VL-7B"},
}
# dimension -> (per-head batch key of the EditReward model, index of the head output)
HEAD_OF_DIMENSION = {"Instruction Following": ("batch_dim1", 0), "Visual Quality": ("batch_dim2", 1)}
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")


def local_config_text(yaml_text: str, backbone_path: Path, output_dir: Path) -> str:
    """Repo yaml with model_name_or_path set to the local backbone and output_dir set; nothing else changes."""
    replacements = {"model_name_or_path": f'"{backbone_path}"', "output_dir": f'"{output_dir}"'}
    for key, value in replacements.items():
        pattern = re.compile(rf"^{key}:.*$", re.MULTILINE)
        if len(pattern.findall(yaml_text)) != 1:
            raise ValueError(f"expected exactly one '{key}:' line in the EditReward config")
        yaml_text = pattern.sub(f"{key}: {value}", yaml_text)
    return yaml_text


def pair_scores(run_head, source_path: str, edited_paths, instruction: str, dimension: str) -> list:
    """[(mu, log sigma) of image A, (mu, log sigma) of image B] from the head of `dimension`.

    run_head(prompts, sources, edited, batch_key) -> two rows [mu, log sigma], one per edited image.
    """
    if dimension not in HEAD_OF_DIMENSION:
        raise ValueError(f"unknown EditReward dimension {dimension!r}")
    batch_key, _ = HEAD_OF_DIMENSION[dimension]
    outputs = run_head([instruction, instruction], [source_path, source_path], list(edited_paths), batch_key)
    if len(outputs) != 2:
        raise ValueError(f"expected two outputs (images A and B), got {len(outputs)}")
    result = []
    for output in outputs:
        values = [float(v) for v in output]
        if len(values) != 2 or any(v != v for v in values):
            raise ValueError(f"expected [mu, log sigma] without NaN, got {values}")
        result.append(tuple(values))
    return result


def score_rows(run_head, rows, dimension_of_metric, score_path: Path, image_paths=None, log=print) -> dict:
    """Score every row whose metric has a dimension; append records to score_path; resume-safe."""
    image_paths = image_paths or (lambda row: bi.image_paths(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    counts = {"scored": 0, "skipped_done": 0, "not_applicable": 0}
    with open(score_path, "a", encoding="utf-8") as score_file:
        for row in rows:
            dimension = dimension_of_metric.get(row["metric"])
            if dimension is None:
                counts["not_applicable"] += 1
                continue
            if dimension not in HEAD_OF_DIMENSION:
                raise ValueError(f"mapping gives {dimension!r} for {row['metric']!r}, not an EditReward dimension")
            if (row["sample_id"], dimension) in done:
                counts["skipped_done"] += 1
                continue
            started = time.time()
            source, edited = image_paths(row)
            (mu_a, log_sigma_a), (mu_b, log_sigma_b) = pair_scores(
                run_head, str(source), [str(p) for p in edited], row["prompt"], dimension)
            seconds = round(time.time() - started, 2)
            batch_key, head_index = HEAD_OF_DIMENSION[dimension]
            record = {"sample_id": row["sample_id"], "dimension": dimension,
                      "score_a": mu_a, "score_b": mu_b, "margin": mu_a - mu_b}
            record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
            record.update({"log_sigma_a": log_sigma_a, "log_sigma_b": log_sigma_b,
                           "head": f"head_{head_index}", "batch_key": batch_key, "seconds": seconds})
            score_file.write(json.dumps(record) + "\n")
            score_file.flush()
            counts["scored"] += 1
            log(f"[{counts['scored']}] {row['sample_id']} {dimension}: {mu_a:.4f} vs {mu_b:.4f} ({seconds}s)")
    return counts


def make_run_head(inferencer, torch):
    """run_head for score_rows: official prepare_batch, then only the needed head of the model."""
    inferencer.model.pooling_strategy = None  # return one output per head instead of their mean

    def run_head(prompts, sources, edited, batch_key):
        batch = inferencer.prepare_batch(sources, edited, prompts)
        with torch.no_grad():
            outputs = inferencer.model(return_dict=True, **{batch_key: batch[batch_key]})
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError(f"expected the output of one head, got {type(outputs).__name__}")
        return outputs[0].float().cpu().tolist()

    return run_head


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/EditReward_<pretrained>")
    parser.add_argument("--per-metric-limit", type=int, default=None, help="smoke test: first N rows per metric")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows = bi.load_bench_rows(args.bench or Path(mapping["bench"]))
    dimension_of_metric = bi.dimension_by_bench_metric(mapping, MODEL_NAME, args.pretrained)
    selected = bi.select_rows(rows, per_metric_limit=args.per_metric_limit)
    variant = VARIANTS[args.pretrained]
    output_dir = args.output_dir or bi.DEFAULT_RESULTS / f"{MODEL_NAME}_{args.pretrained}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (variant["backbone_path"], variant["checkpoint_path"]):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"missing local weights: {path}")
    config_path = output_dir / "editreward_config.yaml"
    config_path.write_text(local_config_text(Path(variant["config"]).read_text(encoding="utf-8"),
                                             variant["backbone_path"], output_dir), encoding="utf-8")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(EDITREWARD_REPO))
    import importlib

    import torch
    import transformers

    inferencer_cls = getattr(importlib.import_module(variant["module"]), variant["inferencer"])
    config = {
        "model": MODEL_NAME, "pretrained": args.pretrained, "inferencer": f"{variant['module']}.{variant['inferencer']}",
        "repo_config": str(variant["config"]), "local_config": str(config_path),
        "backbone_path": str(variant["backbone_path"]), "checkpoint_path": str(variant["checkpoint_path"]),
        "rm_head_type": "ranknet_multi_head", "score": "mu of the head of the dimension",
        "image_preprocessing": "EditReward prepare_batch (min_pixels = max_pixels = 256*28*28)",
        "bench": str(args.bench or mapping["bench"]), "rows_selected": len(selected),
        "per_metric_limit": args.per_metric_limit, "dimension_of_metric": dimension_of_metric,
        "host": socket.gethostname(), "user": os.environ.get("USER"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "script_md5": bi.file_md5(Path(__file__)),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    load_started = time.time()
    inferencer = inferencer_cls(config_path=str(config_path), checkpoint_path=str(variant["checkpoint_path"]),
                                device="cuda", rm_head_type="ranknet_multi_head")
    config["load_seconds"] = round(time.time() - load_started, 1)
    counts = score_rows(make_run_head(inferencer, torch), selected, dimension_of_metric, output_dir / "scores.jsonl",
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
