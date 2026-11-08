"""Choose Unlabeled samples on which the Teacher and the Inspector agree.

Step 1 of the Stage 2 sample selection (user, 2026-11-04). The agreement of one
sample is the Unlabeled loss term between Inspector and Student with the Student
replaced by the Teacher:

    CE(Inspector, Teacher) = -sum_c P_Inspector(c) * log P_Teacher(c),  c in (A, B, Tie)

P_Teacher is the overall three-way probability of the training loss: selected_equal
mean of the per-metric three_way_probs over the metrics the Teacher relevance selects.
The Teacher runs in eval mode (dropout off), one pair at a time, so every score is
deterministic.

Order and stopping rule: the Inspector cache already holds every Unlabeled sample, so
samples are visited from the most certain Inspector (lowest entropy, the floor of the
CE) to the least certain. A visited sample is accepted when the Teacher label (argmax)
equals the Inspector label. The scan stops as soon as --target-count samples are
accepted; the rest of the pool is never scored.

Outputs (--output-dir):
  agreement_scored.jsonl         one row per scored sample, in scan order (appended and
                                 flushed, so a stopped job resumes where it ended)
  selected_unlabeled_<N>.json    the accepted sample IDs (sorted by CE) and the rule
  selection_summary.json         counts, acceptance rate, label counts, CE / KL stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent  # Qwen3-VL-Embedding
_EVAL_DIR = _PROJECT_ROOT.parents[1] / "RewardModelEvaluation" / "OurModel"
for path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_reward_model import load_refined_model  # noqa: E402
from train_loop import (  # noqa: E402
    DEFAULT_EDITREWARD_ROOT,
    PROBABILITY_EPS,
    THREE_WAY_CLASSES,
    forward_pairs,
    index_unlabeled_pool,
    load_inspector_cache,
    materialize_unlabeled_round,
    overall_three_way_probs,
    three_way_cross_entropy,
)

SCORED_FILE = "agreement_scored.jsonl"
SUMMARY_FILE = "selection_summary.json"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-checkpoint", required=True, help="teacher_initialized.pt of Student-Stage1")
    parser.add_argument("--student-checkpoint", required=True, help="student_warmstart.pt (prompt contract)")
    parser.add_argument("--inspector-cache", required=True)
    parser.add_argument("--unlabeled-dataset", required=True)
    parser.add_argument("--editreward-root", default=str(DEFAULT_EDITREWARD_ROOT))
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--target-count", type=int, default=1008)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inspector-max-probability-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
        help="Only samples whose largest Inspector three-way probability is in [LOW, HIGH) "
             "are visited (same order, most certain first); the others are never scored.")
    parser.add_argument(
        "--unique-instructions", action="store_true",
        help="Skip (without scoring) a sample whose instruction (case, spaces and final "
             "punctuation ignored) is already accepted, so every accepted instruction differs.")
    parser.add_argument(
        "--scan-order", choices=("inspector_entropy", "random"), default="inspector_entropy",
        help="inspector_entropy: most certain Inspector first (default, earlier subsets); "
             "random: seeded random order, so the accepted samples cover the whole "
             "--inspector-max-probability-range evenly (user, 2026-11-05).")
    parser.add_argument("--scan-seed", type=int, default=42, help="Seed of --scan-order random.")
    return parser.parse_args()


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities.clamp_min(PROBABILITY_EPS))).sum(dim=-1)


def stats(values) -> dict:
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]  # noqa: E731
    return {"mean": round(sum(values) / len(values), 6), "min": round(values[0], 6),
            "median": round(pick(0.5), 6), "p90": round(pick(0.9), 6), "max": round(values[-1], 6)}


def instruction_key(instruction: str) -> str:
    """Instruction compared for --unique-instructions: lower case, single spaces, no final . ! ?"""
    return " ".join(instruction.split()).lower().rstrip(".!? ")


def main() -> None:
    args = parse_args()
    if args.target_count < 1:
        raise ValueError("--target-count must be positive.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / SCORED_FILE

    root = Path(args.editreward_root).expanduser().resolve()
    pool = index_unlabeled_pool(Path(args.unlabeled_dataset), root)
    inspector = load_inspector_cache(Path(args.inspector_cache), pool.sample_ids)
    three_way = inspector.three_way.double()
    if not bool(torch.all(torch.isfinite(three_way))):
        raise ValueError("The Inspector cache must hold three-way probabilities.")
    inspector_entropy = entropy(three_way)
    # Most certain Inspector first; ties keep the pool order.
    order = sorted(range(len(pool.sample_ids)), key=lambda i: (float(inspector_entropy[i]), i))
    if args.inspector_max_probability_range:
        low, high = args.inspector_max_probability_range
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("--inspector-max-probability-range needs 0 <= LOW < HIGH <= 1.")
        largest = three_way.max(dim=1).values
        order = [i for i in order if low <= float(largest[i]) < high]
        print(f"Inspector largest probability in [{low}, {high}): {len(order)} samples", flush=True)
    if args.scan_order == "random":
        random.Random(args.scan_seed).shuffle(order)
        print(f"Scan order: random (seed {args.scan_seed})", flush=True)
    print(f"Unlabeled pool: {len(order)}; target accepted: {args.target_count}", flush=True)

    # Resume: rows already scored keep their scan positions.
    done = []
    if scored_path.exists():
        done = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = [pool.sample_ids[i] for i in order[: len(done)]]
        if [row["sample_id"] for row in done] != expected:
            raise RuntimeError(f"{scored_path} does not follow the scan order; refusing to resume.")
        print(f"Resuming after {len(done)} scored samples", flush=True)
    accepted = sum(1 for row in done if row["accepted"])
    accepted_instructions = {row["instruction_key"] for row in done if row["accepted"] and "instruction_key" in row}

    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    prompt = teacher_checkpoint.get("prompt")
    if not isinstance(prompt, dict):
        student_checkpoint = torch.load(args.student_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        prompt = student_checkpoint.get("prompt")
        del student_checkpoint
    if not isinstance(prompt, dict):
        raise ValueError("No prompt contract in the Teacher or Student-Stage1 checkpoint.")
    model = processor = None
    if accepted < args.target_count:
        # load_refined_model puts the model in eval mode and fails if any dropout is on.
        model, processor = load_refined_model(
            teacher_checkpoint, "teacher", device, None, None,
            base_model_override=Path(args.base_model_path).expanduser().resolve(),
        )
    del teacher_checkpoint

    position = len(done)
    with scored_path.open("a", encoding="utf-8") as handle:
        while accepted < args.target_count and position < len(order):
            index = order[position]
            record = materialize_unlabeled_round(pool, [index])[index]
            key = instruction_key(record.edit_instruction)
            if args.unique_instructions and key in accepted_instructions:
                row = {"scan_position": position, "sample_id": pool.sample_ids[index],
                       "instruction_key": key, "skipped_duplicate_instruction": True,
                       "accepted": False}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done.append(row)
                position += 1
                continue
            with torch.no_grad():
                output = forward_pairs(model, processor, [record], prompt, device)
            teacher = overall_three_way_probs(output, None, "selected_equal")[0].double().cpu()
            relevance = output.relevance_probs.float()[0].cpu()
            target = three_way[index]
            cross_entropy = float(three_way_cross_entropy(teacher.unsqueeze(0), target.unsqueeze(0))[0])
            target_entropy = float(inspector_entropy[index])
            teacher_label = THREE_WAY_CLASSES[int(teacher.argmax())]
            inspector_label = THREE_WAY_CLASSES[int(target.argmax())]
            same = teacher_label == inspector_label
            accepted += int(same)
            if same:
                accepted_instructions.add(key)
            row = {
                "scan_position": position,
                "sample_id": pool.sample_ids[index],
                "inspector": [round(float(v), 6) for v in target],
                "teacher": [round(float(v), 6) for v in teacher],
                "inspector_label": inspector_label,
                "teacher_label": teacher_label,
                "teacher_same_as_inspector": same,
                "accepted": same,
                "inspector_teacher_cross_entropy": round(cross_entropy, 6),
                "inspector_entropy": round(target_entropy, 6),
                "inspector_teacher_kl": round(cross_entropy - target_entropy, 6),
                "teacher_selected_metrics": int(max(1, int((relevance >= 0.5).sum()))),
                "instruction_key": key,
            }
            if not all(math.isfinite(v) for v in row["teacher"]) or not math.isfinite(cross_entropy):
                raise RuntimeError(f"Non-finite Teacher output for {row['sample_id']}")
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done.append(row)
            position += 1
            if position % 50 == 0 or accepted == args.target_count:
                print(f"Scored {position}, accepted {accepted}/{args.target_count}, "
                      f"acceptance {accepted / position:.3f}", flush=True)

    chosen = sorted((row for row in done if row["accepted"]),
                    key=lambda row: (row["inspector_teacher_cross_entropy"], row["scan_position"]))
    if len(chosen) < args.target_count:
        raise RuntimeError(f"Only {len(chosen)} accepted after scanning the whole pool.")
    visit = ("visit Unlabeled samples by ascending Inspector entropy" if args.scan_order == "inspector_entropy"
             else f"visit Unlabeled samples in a random order (seed {args.scan_seed})")
    rule = (visit + "; accept when the Teacher "
            "label (argmax of selected_equal three-way, eval mode, dropout off) equals the "
            "Inspector label; stop at the target count; IDs sorted by CE(Inspector, Teacher)")
    if args.inspector_max_probability_range:
        rule += ("; only samples whose largest Inspector probability is in "
                 f"[{args.inspector_max_probability_range[0]}, {args.inspector_max_probability_range[1]})")
    if args.unique_instructions:
        rule += ("; a sample whose instruction is already accepted is skipped without scoring "
                 "(every accepted instruction differs)")
    skipped = [row for row in done if row.get("skipped_duplicate_instruction")]
    done = [row for row in done if not row.get("skipped_duplicate_instruction")]
    selected = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": rule,
        "subset_size": len(chosen),
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "inspector_cache": str(Path(args.inspector_cache).resolve()),
        "inspector_cache_md5": file_md5(Path(args.inspector_cache)),
        "unlabeled_dataset": str(Path(args.unlabeled_dataset).resolve()),
        "sample_ids": [row["sample_id"] for row in chosen],
    }
    (output_dir / f"selected_unlabeled_{args.target_count}.json").write_text(
        json.dumps(selected, indent=1), encoding="utf-8")
    rejected = [row for row in done if not row["accepted"]]
    summary = {
        "rule": rule,
        "pool": len(order),
        "scanned": len(done) + len(skipped),
        "scored": len(done),
        "skipped_duplicate_instruction": len(skipped),
        "unique_accepted_instructions": len({row["instruction_key"] for row in chosen if "instruction_key" in row}),
        "accepted": len(chosen),
        "acceptance_rate": round(len(chosen) / len(done), 6),  # among scored samples
        "last_scanned_inspector_entropy": done[-1]["inspector_entropy"],
        "last_scanned_inspector_max_probability": max(done[-1]["inspector"]),
        "selected_label_counts": dict(Counter(row["inspector_label"] for row in chosen)),
        "rejected_transitions": dict(Counter(
            f"Inspector {row['inspector_label']} -> Teacher {row['teacher_label']}" for row in rejected
        ).most_common()),
        "selected_cross_entropy": stats(row["inspector_teacher_cross_entropy"] for row in chosen),
        "selected_kl": stats(row["inspector_teacher_kl"] for row in chosen),
        "selected_inspector_max_probability": stats(max(row["inspector"]) for row in chosen),
        "selected_teacher_max_probability": stats(max(row["teacher"]) for row in chosen),
        "rejected_cross_entropy": stats(row["inspector_teacher_cross_entropy"] for row in rejected),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    print(f"Written {output_dir}", flush=True)


if __name__ == "__main__":
    main()
