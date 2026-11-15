#!/usr/bin/env python
"""Score AURA-Test pairs with EditScore (Qwen2.5-VL-7B, Qwen3-VL-4B or Qwen3-VL-8B LoRA).

Uses the original EditScore code (EditScore/editscore, not modified): the same prompts, sampling
(temperature 0.7, top_p 0.9, top_k 20, seed 42, max 512 new tokens), parsing and retry rule as
EditScore.evaluate, and the official evaluation.py preprocessing (edited image resized to the source size).

Only the prompt needed for the row's metric is generated (metric_mapping.json):
    Aesthetic, Realism                    -> Perceptual Quality   = min(naturalness, artifacts) / 2.5
    Fidelity, BC, FC, SC                  -> Semantic Consistency = min(success, overediting) / 2.5
Each image of the pair is scored separately with the same seed. When the output cannot be parsed after
the original retries, EditScore itself guesses a random score; such cases are flagged as "guessed".

One job scores the whole bench (all 1561 pairs) for one variant; the bench is never split.
Output (in --output-dir): scores.jsonl, raw_outputs.jsonl, run_config.json.
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

import bench_inputs as bi

MODEL_NAME = "EditScore"
EDITSCORE_REPO = bi.COMPARISON_ROOT / "EditScore"
VARIANTS = {
    "Qwen2.5-VL-7B": {"backbone": "qwen25vl", "model_path": bi.MODELS_DIR / "Qwen2.5-VL-7B",
                      "lora_path": EDITSCORE_REPO / "weights" / "EditScore-7B"},
    "Qwen3-VL-4B": {"backbone": "qwen3vl", "model_path": bi.MODELS_DIR / "Qwen3-VL-4B",
                    "lora_path": EDITSCORE_REPO / "weights" / "EditScore-Qwen3-VL-4B-Instruct"},
    "Qwen3-VL-8B": {"backbone": "qwen3vl", "model_path": bi.MODELS_DIR / "Qwen3-VL-8B",
                    "lora_path": EDITSCORE_REPO / "weights" / "EditScore-Qwen3-VL-8B-Instruct"},
}
SCORE_RANGE = 25
TEMPERATURE = 0.7
SEED = 42
MAX_TRIES = 2  # same as EditScore.evaluate
REFUSAL = "I'm sorry, but I can't assist with that request."
DIMENSIONS = ("Semantic Consistency", "Perceptual Quality")
SUB_SCORE_NAMES = {"Semantic Consistency": ["editing success", "overediting"],
                   "Perceptual Quality": ["naturalness", "artifacts"]}
# Copied into every score record so per-pair files are self-contained for later analyses.
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")


def score_image(scorer, parse_output, source_image, edited_image, instruction: str, dimension: str) -> dict:
    """One EditScore prompt on one edited image, following EditScore.evaluate for that prompt."""
    if dimension == "Semantic Consistency":
        prompt = scorer.SC_prompt.replace("<instruction>", instruction)
        inputs = scorer.model.prepare_input([source_image, edited_image], prompt)
    elif dimension == "Perceptual Quality":
        inputs = scorer.model.prepare_input(edited_image, scorer.PQ_prompt)
    else:
        raise ValueError(f"unknown EditScore dimension {dimension!r}")

    tries, result, texts = 0, False, []
    while result is False:
        tries += 1
        give_up = tries > MAX_TRIES
        text = scorer.model.inference(inputs, seed=scorer.seed)
        texts.append(text)
        if text == REFUSAL:
            give_up = True
        result = parse_output(text, give_up_parsing=give_up, text_prompt=instruction, score_range=scorer.score_range)
    if result == "rate_limit_exceeded":
        raise RuntimeError("rate_limit_exceeded")
    sub_scores = [float(s) for s in result["score"]]
    if not sub_scores:
        raise ValueError(f"EditScore returned no score: {texts[-1]!r}")
    guessed = str(result.get("reasoning", "")).startswith("guess_if_cannot_parse")
    return {"score": min(sub_scores) / (scorer.score_range / 10), "sub_scores": sub_scores,
            "status": "guessed" if guessed else "parsed", "tries": tries, "texts": texts,
            "reasoning": str(result.get("reasoning", ""))}


def load_pair_images(row: dict, source_root: Path, edited_root: Path):
    """PIL images as in EditScore/evaluation.py: RGB, edited image resized to the source size."""
    from PIL import Image

    source_path, edited_paths = bi.image_paths(row, source_root, edited_root)
    source = Image.open(source_path).convert("RGB")
    edited = [Image.open(p).convert("RGB").resize(source.size) for p in edited_paths]
    return source, edited


def score_rows(scorer, parse_output, rows, dimension_of_metric, score_path: Path, raw_path: Path,
               load_images=None, log=print) -> dict:
    """Score every row whose metric has a dimension; append to score_path / raw_path; resume-safe."""
    load_images = load_images or (lambda row: load_pair_images(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    counts = {"scored": 0, "skipped_done": 0, "not_applicable": 0, "guessed_images": 0}
    with open(score_path, "a", encoding="utf-8") as score_file, open(raw_path, "a", encoding="utf-8") as raw_file:
        for row in rows:
            dimension = dimension_of_metric.get(row["metric"])
            if dimension is None:
                counts["not_applicable"] += 1
                continue
            if dimension not in DIMENSIONS:
                raise ValueError(f"mapping gives {dimension!r} for {row['metric']!r}, not an EditScore dimension")
            if (row["sample_id"], dimension) in done:
                counts["skipped_done"] += 1
                continue
            started = time.time()
            source, edited = load_images(row)
            results = [score_image(scorer, parse_output, source, image, row["prompt"], dimension) for image in edited]
            seconds = round(time.time() - started, 2)
            record = {"sample_id": row["sample_id"], "dimension": dimension,
                      "score_a": results[0]["score"], "score_b": results[1]["score"],
                      "margin": results[0]["score"] - results[1]["score"]}
            record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
            record.update({
                "sub_scores_a": results[0]["sub_scores"], "sub_scores_b": results[1]["sub_scores"],
                "sub_score_names": SUB_SCORE_NAMES[dimension],
                "status_a": results[0]["status"], "status_b": results[1]["status"],
                "tries_a": results[0]["tries"], "tries_b": results[1]["tries"],
                "reasoning_a": results[0]["reasoning"], "reasoning_b": results[1]["reasoning"],
                "image_size": list(getattr(source, "size", ())) or None, "seconds": seconds,
            })
            score_file.write(json.dumps(record) + "\n")
            raw_file.write(json.dumps({"sample_id": row["sample_id"], "dimension": dimension,
                                       "outputs_a": results[0]["texts"], "outputs_b": results[1]["texts"]}) + "\n")
            score_file.flush()
            raw_file.flush()
            counts["scored"] += 1
            counts["guessed_images"] += sum(r["status"] == "guessed" for r in results)
            log(f"[{counts['scored']}] {row['sample_id']} {dimension}: "
                f"{results[0]['score']:.2f} vs {results[1]['score']:.2f} ({seconds}s)")
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: results/EditScore_<pretrained>")
    parser.add_argument("--per-metric-limit", type=int, default=None, help="smoke test: first N rows per metric")
    # Splitting was removed (2026-11-12). Jobs queued before that (12876-12878, 12992-12994) keep a copy of the
    # old launcher, which passes "--num-shards 1 --shard-index 0"; only those values are accepted, hidden.
    parser.add_argument("--num-shards", type=int, choices=[1], default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, choices=[0], default=0, help=argparse.SUPPRESS)
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
    score_path, raw_path = output_dir / "scores.jsonl", output_dir / "raw_outputs.jsonl"
    for path in (variant["model_path"], variant["lora_path"]):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"missing local weights: {path}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(EDITSCORE_REPO))
    import torch
    from editscore import EditScore
    from editscore.utils import mllm_output_to_dict

    config = {
        "model": MODEL_NAME, "pretrained": args.pretrained, "backbone": variant["backbone"],
        "model_path": str(variant["model_path"]), "lora_path": str(variant["lora_path"]),
        "score_range": SCORE_RANGE, "temperature": TEMPERATURE, "seed": SEED, "num_pass": 1,
        "bench": str(args.bench or mapping["bench"]), "rows_selected": len(selected),
        "per_metric_limit": args.per_metric_limit,
        "dimension_of_metric": dimension_of_metric, "host": socket.gethostname(), "user": os.environ.get("USER"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_md5": bi.file_md5(Path(__file__)), "torch": torch.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    load_started = time.time()
    scorer = EditScore(backbone=variant["backbone"], model_name_or_path=str(variant["model_path"]),
                       lora_path=str(variant["lora_path"]), score_range=SCORE_RANGE, temperature=TEMPERATURE,
                       seed=SEED, num_pass=1)
    config["load_seconds"] = round(time.time() - load_started, 1)
    counts = score_rows(scorer, mllm_output_to_dict, selected, dimension_of_metric, score_path, raw_path,
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
