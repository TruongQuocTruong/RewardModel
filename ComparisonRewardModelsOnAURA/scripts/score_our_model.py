#!/usr/bin/env python
"""Score AURA-Test pairs with our Student checkpoints (Qwen3-VL-Reranker-8B + LoRA + six metric heads).

Variants (fixed checkpoints, never chosen on AURA-Test):
    student_stage1      Student Model after Stage 1 (student_warmstart.pt = MS-Swift checkpoint-126, 1008 pairs).
                        The Teacher Model of Stage 1 is a copy of this Student, so it also stands for Teacher-Stage1.
    student_stage2      Student Model of the final Stage 2 checkpoint (version Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5, checkpoint-927.pt).
    teacher_stage2      Teacher Model (EMA) of the same checkpoint-927.pt (teacher_adapter_state_dict).
The Inspector Baseline is not evaluated on AURA-Test (user decision 2026-11-13): it is trained and prompted
for the overall preference only, while every AURA-Test pair is labelled for one specific metric.

Each column reads the head of its own metric (the bench row gives the metric, so the Relevance Head does not
select anything; its probabilities are only saved). Model code and prompt are those of the training repository
(TrainingRewardModel/Qwen3-VL-Embedding): evaluate_reward_model.load_refined_model + train_loop.forward_pairs.
Images use the processor defaults, as in training (no min/max pixels).

Per pair and head: score_a / score_b = Score Head output of edited_images[0] / [1] (margin = score_a - score_b),
probabilities = [P(A), P(B), P(Tie)] of the margin-coupled Tie Head (three-class accuracy = argmax),
signed_evidence = P(A) - P(B) (kept for analysis). Student records also keep all six heads and the
Relevance Head probabilities.

One job scores the whole bench (all 1561 pairs) for one variant; the bench is never split.
Output (in --output-dir): scores.jsonl, run_config.json. Re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import bench_inputs as bi

REPO_ROOT = bi.COMPARISON_ROOT.parent
PROJECT_DIR = REPO_ROOT / "TrainingRewardModel" / "Qwen3-VL-Embedding"
OUR_EVAL_DIR = REPO_ROOT / "RewardModelEvaluation" / "OurModel"
OUTPUT_TRAINING = PROJECT_DIR / "output_training"
BASE_MODEL_PATH = REPO_ROOT / "Models" / "Qwen3-VL-Reranker-8B"
STUDENT_RUN = OUTPUT_TRAINING / ("student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32"
                                 "_initialized_from_inspector_checkpoint63")
STAGE2_CHECKPOINT = OUTPUT_TRAINING / "Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5" / "checkpoint-927.pt"

# Student head order (src/models/custom_qwen3_vl_reranker.py METRIC_NAMES).
STUDENT_HEADS = ("aesthetic", "fidelity", "realism", "background_consistency", "foreground_consistency",
                 "structure_consistency")
STUDENT_HEAD_OF_METRIC = {
    "Aesthetic": "aesthetic", "Realism": "realism", "Fidelity": "fidelity",
    "Background Consistency": "background_consistency", "Foreground Consistency": "foreground_consistency",
    "Structure Consistency": "structure_consistency",
}
# kind = head layout (six metric heads, shared by Student and Teacher); role = which adapter of the checkpoint.
VARIANTS = {
    "student_stage1": {"kind": "student", "role": "student", "run_dir": STUDENT_RUN,
                       "model": "Student-Stage1", "checkpoint": STUDENT_RUN / "student_warmstart.pt", "pairs_trained": 1008,
                       "result_folder": "Student-Stage1_Qwen3-VL-Reranker-8B"},
    "student_stage2": {"kind": "student", "role": "student", "run_dir": STAGE2_CHECKPOINT.parent,
                       "model": "Student-Stage2", "checkpoint": STAGE2_CHECKPOINT, "pairs_trained": 1008,
                       "result_folder": "Student-Stage2_Qwen3-VL-Reranker-8B"},
    "teacher_stage2": {"kind": "student", "role": "teacher", "run_dir": STAGE2_CHECKPOINT.parent,
                       "model": "Teacher-Stage2", "checkpoint": STAGE2_CHECKPOINT, "pairs_trained": 1008,
                       "result_folder": "Teacher-Stage2_Qwen3-VL-Reranker-8B"},
}
ROLES = ("student", "teacher")
BENCH_FIELDS_TO_KEEP = ("metric", "image_id", "edit_type", "prompt", "src_image", "edited_images", "models",
                        "num_raters", "votes", "observed_agreement", "human_label", "preferred_model")
PROBABILITY_TOLERANCE = 1e-4


def head_of_metric(kind: str, metric: str) -> str:
    """Head that scores one bench metric: the Student head of that metric."""
    if metric not in STUDENT_HEAD_OF_METRIC:
        raise ValueError(f"unknown AURA-Test metric {metric!r}")
    if kind == "student":
        return STUDENT_HEAD_OF_METRIC[metric]
    raise ValueError(f"unknown variant kind {kind!r}")


def _finite(values, name: str, length: int) -> list:
    values = [float(v) for v in values]
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name}: expected {length} finite values, got {values}")
    return values


def head_record(output: dict, heads, head: str) -> dict:
    """score_a, score_b, margin, probabilities and signed_evidence of one head of one pair's output.

    output: {"scores_a": [H], "scores_b": [H], "three_way_probs": [H x 3]}, heads in `heads` order.
    """
    if head not in heads:
        raise ValueError(f"head {head!r} is not one of {list(heads)}")
    count = len(heads)
    scores_a = _finite(output["scores_a"], "scores_a", count)
    scores_b = _finite(output["scores_b"], "scores_b", count)
    three_way = output["three_way_probs"]
    if len(three_way) != count:
        raise ValueError(f"three_way_probs: expected {count} rows, got {len(three_way)}")
    index = list(heads).index(head)
    probabilities = _finite(three_way[index], "three_way_probs", 3)
    if any(p < 0 for p in probabilities) or abs(sum(probabilities) - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"three_way_probs of {head!r} must be non-negative and sum to 1: {probabilities}")
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    return {"score_a": scores_a[index], "score_b": scores_b[index], "margin": scores_a[index] - scores_b[index],
            "probabilities": probabilities, "signed_evidence": probabilities[0] - probabilities[1]}


def score_rows(run_pair, kind: str, rows, score_path: Path, image_paths=None, log=print) -> dict:
    """Score every row with the head of its metric; append records to score_path; resume-safe.

    run_pair(source, [edited A, edited B], instruction) -> {"heads": [...], "scores_a": [...], "scores_b": [...],
    "three_way_probs": [[pA, pB, pTie], ...], optional "relevance_probs": [...]}.
    """
    image_paths = image_paths or (lambda row: bi.image_paths(row, bi.SOURCE_IMAGE_ROOT, bi.EDITED_IMAGE_ROOT))
    done = bi.already_scored(score_path)
    counts = {"scored": 0, "skipped_done": 0}
    with open(score_path, "a", encoding="utf-8") as score_file:
        for row in rows:
            head = head_of_metric(kind, row["metric"])
            if (row["sample_id"], head) in done:
                counts["skipped_done"] += 1
                continue
            started = time.time()
            source, edited = image_paths(row)
            output = run_pair(str(source), [str(p) for p in edited], row["prompt"])
            heads = list(output["heads"])
            record = {"sample_id": row["sample_id"], "dimension": head, **head_record(output, heads, head)}
            record.update({field: row.get(field) for field in BENCH_FIELDS_TO_KEEP})
            if kind == "student":
                record["all_heads"] = {
                    name: {key: value for key, value in head_record(output, heads, name).items()}
                    for name in heads}
                if output.get("relevance_probs") is not None:
                    record["relevance_probs"] = dict(zip(heads, _finite(output["relevance_probs"],
                                                                        "relevance_probs", len(heads))))
            record["seconds"] = round(time.time() - started, 2)
            score_file.write(json.dumps(record) + "\n")
            score_file.flush()
            counts["scored"] += 1
            log(f"[{counts['scored']}] {row['sample_id']} {head}: {record['score_a']:.4f} vs "
                f"{record['score_b']:.4f}, P(Tie)={record['probabilities'][2]:.3f} ({record['seconds']}s)")
    return counts


def _import_training_code():
    for path in (PROJECT_DIR, PROJECT_DIR / "scripts", OUR_EVAL_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def make_student_runner(variant: dict, device):
    """Student or Teacher Model (variant["role"]) of the checkpoint, loaded exactly as
    evaluate_student_checkpoints.py does (STUDENT_EVAL_ROLE=student / teacher)."""
    role = variant.get("role", "student")
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    _import_training_code()
    import torch
    from evaluate_reward_model import load_refined_model
    from src.models.custom_qwen3_vl_reranker import METRIC_NAMES
    from train_loop import PairRecord, forward_pairs

    if tuple(METRIC_NAMES) != STUDENT_HEADS:
        raise ValueError(f"Student head order changed: {METRIC_NAMES}")
    checkpoint = torch.load(variant["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    prompt = checkpoint.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Student checkpoint is missing the prompt contract")
    model, processor = load_refined_model(checkpoint, role, device, None, None,
                                          base_model_override=BASE_MODEL_PATH)

    @torch.inference_mode()
    def run_pair(source, edited, instruction):
        output = forward_pairs(model, processor, [PairRecord(sample_id="aura", source_image=source,
                                                             edit_instruction=instruction, image_a=edited[0],
                                                             image_b=edited[1])], prompt, device)
        return {"heads": list(STUDENT_HEADS), "scores_a": output.scores_A[0].float().cpu().tolist(),
                "scores_b": output.scores_B[0].float().cpu().tolist(),
                "three_way_probs": output.three_way_probs[0].float().cpu().tolist(),
                "relevance_probs": output.relevance_probs[0].float().cpu().tolist()}

    return run_pair, {"tie_head_mode": checkpoint.get("tie_head_mode"), "prompt": prompt,
                      "ms_swift_checkpoint": checkpoint.get("ms_swift_best_checkpoint"),
                      "model_role": checkpoint.get("model_role"), "role": role,
                      "optimizer_step": checkpoint.get("optimizer_step")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--mapping", type=Path, default=bi.DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: results/<result_folder of the variant>")
    parser.add_argument("--per-metric-limit", type=int, default=None, help="smoke test: first N rows per metric")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    bench_path = args.bench or Path(mapping["bench"])
    rows = bi.load_bench_rows(bench_path)
    selected = bi.select_rows(rows, per_metric_limit=args.per_metric_limit)
    variant = VARIANTS[args.variant]
    output_dir = args.output_dir or bi.DEFAULT_RESULTS / variant["result_folder"]
    for path in (BASE_MODEL_PATH, variant["checkpoint"]):
        if not Path(path).exists():
            raise FileNotFoundError(f"missing local weights: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers

    device = torch.device(args.device)
    config = {
        "model": variant["model"], "variant": args.variant, "kind": variant["kind"], "role": variant["role"],
        "checkpoint": str(variant["checkpoint"]), "pairs_trained": variant["pairs_trained"],
        "base_model_path": str(BASE_MODEL_PATH),
        "head_of_metric": {m: head_of_metric(variant["kind"], m) for m in STUDENT_HEAD_OF_METRIC},
        "score": "Score Head output per image; margin = score_a - score_b",
        "probabilities": "[P(A), P(B), P(Tie)] of the head; three-class prediction = argmax",
        "image_preprocessing": "processor defaults (as in training, no min/max pixels)",
        "bench": str(bench_path), "rows_selected": len(selected), "per_metric_limit": args.per_metric_limit,
        "host": socket.gethostname(), "user": os.environ.get("USER"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "script_md5": bi.file_md5(Path(__file__)),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    print(json.dumps(config, indent=2), flush=True)
    load_started = time.time()
    run_pair, loaded = make_student_runner(variant, device)
    config.update(loaded)
    config["load_seconds"] = round(time.time() - load_started, 1)
    counts = score_rows(run_pair, variant["kind"], selected, output_dir / "scores.jsonl",
                        log=lambda message: print(message, flush=True))
    config.update(counts)
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if torch.cuda.is_available():
        config["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
