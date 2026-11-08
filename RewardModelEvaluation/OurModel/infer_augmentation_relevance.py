"""Inference test of hypotheses (b), (d1), (d2), (f) of the Stage 2 iteration runs (2026-11-07).

Why: in the iteration runs, S Student (Student relevance on the augmented input) differs more and
more from S Teacher (Teacher relevance on the original input). Candidate causes:
  (b)  the augmentation is drawn again at every iteration (same weights, new input -> new S)
  (d1) the Relevance Head reacts to the augmentation itself (e.g. realism on the composite image)
  (d2) the Student moves away from the Teacher because it learns GemmaGT relevance (L_relevance)
  (f)  measurement noise: Teacher (and Student) run with dropout during training
No training: one model, adapters swapped, every forward in eval mode except the dropout passes.

Per checkpoint (Student and Teacher adapters of checkpoint-N.pt, or the Stage 2 start =
Student-Stage1 student_warmstart.pt + teacher_initialized.pt):
  Unlabeled (256 of the run's 1008-sample subset, fixed seed):
    student/original, student/draw_<k> (augmentation exactly as iteration k+1 of the run:
    same sample, iteration index, seed; k = 0, 1, 2 used in iterations 1-3, k = 9 never used),
    student/original_dropout_<j>, teacher/original, teacher/original_dropout_<j>
  Labeled (256 of the run's 1008 labeled pairs, fixed seed): student/original, student/draw_0,
    teacher/original; GemmaGT relevance kept for comparison.
Draws 0-2 are checked against the run's per-sample log (instruction, position, distractor) so
the reproduced inputs are exactly the ones the Student saw.

Output (--output-dir): predictions_<checkpoint>.jsonl (one row per set, sample, role, input),
summary_<checkpoint>.json, summary.csv (all checkpoints, per metric), draws_check.json.
--summarize-only rebuilds the summaries from the predictions files (no GPU).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

_EVAL_DIR = Path(__file__).resolve().parent  # RewardModelEvaluation/OurModel
_PROJECT_ROOT = _EVAL_DIR.parents[1] / "TrainingRewardModel" / "Qwen3-VL-Embedding"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for _path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

METRICS = ("aesthetic", "fidelity", "realism", "background_consistency",
           "foreground_consistency", "structure_consistency")
THRESHOLD = 0.5
UNUSED_DRAW = 9  # iteration index never reached by a 5-iteration run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="checkpoint-N.pt files of the run, and/or the word 'start' (Stage 2 start)")
    parser.add_argument("--start-student", default="")
    parser.add_argument("--start-teacher", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unlabeled-samples", type=int, default=256)
    parser.add_argument("--labeled-samples", type=int, default=256)
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2, UNUSED_DRAW])
    parser.add_argument("--dropout-passes", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--augmentation-seed", type=int, default=42, help="REFINEMENT_SEED of the run")
    parser.add_argument("--augmentation-operation", default="composite+mask_content")
    parser.add_argument("--augmentation-scenarios", default="")
    parser.add_argument("--unlabeled-dataset", default="")
    parser.add_argument("--human-dataset", default="")
    parser.add_argument("--gemma-dataset", default="")
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


# ----------------------------------------------------------------------------- inputs
def choose_ids(ids: Sequence[str], count: int, seed: int) -> List[str]:
    ids = sorted(ids)
    return sorted(random.Random(seed).sample(ids, min(count, len(ids))))


def logged_draws(run_dir: Path) -> Dict[tuple, Dict[str, Any]]:
    """(sample_id, iteration index) -> logged augmentation of the Unlabeled Student input."""
    logs = sorted(run_dir.glob("per_sample_log_*.jsonl"))
    if len(logs) != 1:
        raise RuntimeError(f"Expected one per_sample_log in {run_dir}, found {logs}")
    out = {}
    with logs[0].open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "unlabeled" and row.get("augmentation_round"):
                out[(row["sample_id"], int(row["augmentation_round"]) - 1)] = row
    return out


def build_inputs(args: argparse.Namespace, work_dir: Path) -> Dict[str, Any]:
    import train_loop as tl
    from scripts import student_augmentation as augmentation

    run_dir = Path(args.run_dir)
    root = Path(tl.DEFAULT_EDITREWARD_ROOT).expanduser().resolve()
    operations = args.augmentation_operation.split("+")
    scenarios_file = json.loads(Path(args.augmentation_scenarios).read_text(encoding="utf-8"))
    unlabeled_scenarios, distractors = scenarios_file["samples"], scenarios_file["distractors"]
    labeled_scenarios = json.loads((run_dir / "augmentation_labeled_scenarios.json").read_text(encoding="utf-8"))
    if labeled_scenarios.get("operation") != args.augmentation_operation or labeled_scenarios.get("seed") != args.augmentation_seed:
        raise ValueError("augmentation_labeled_scenarios.json differs from --augmentation-operation / --augmentation-seed")
    labeled_scenarios = labeled_scenarios["samples"]

    def distractor_path(index: int) -> str:
        return tl.resolve_image_path(distractors[index]["source_img"], root)

    subset = json.loads((run_dir / "unlabeled_subset.json").read_text(encoding="utf-8"))
    unlabeled_ids = choose_ids(subset["sample_ids"], args.unlabeled_samples, args.selection_seed)
    pool = tl.index_unlabeled_pool(Path(args.unlabeled_dataset), root)
    index_of = {sample_id: index for index, sample_id in enumerate(pool.sample_ids)}
    by_index = tl.materialize_unlabeled_round(pool, [index_of[i] for i in unlabeled_ids])
    unlabeled = {sample_id: by_index[index_of[sample_id]] for sample_id in unlabeled_ids}

    labeled_ids = choose_ids(list(labeled_scenarios), args.labeled_samples, args.selection_seed)
    human = {r.sample_id: r for r in tl.load_human_records(Path(args.human_dataset), root, keep_ties=True)}
    gemma = {r.sample_id: r for r in tl.load_gemma_records(Path(args.gemma_dataset))}
    missing = [i for i in labeled_ids if i not in human or i not in gemma]
    if missing:
        raise ValueError(f"{len(missing)} labeled pairs are missing in HumanGT / GemmaGT, e.g. {missing[:3]}")

    def augmented(record, scenario, sample_id, draw, tag, allow_swap):
        choice = augmentation.choose_combined_operations(
            operations, scenario, sample_id, draw, args.augmentation_seed, distractors)
        if choice["swap"] and not allow_swap:
            raise RuntimeError("Labeled augmentation must not swap A and B.")
        distractor = distractor_path(choice["distractor_index"]) if choice.get("distractor_index") is not None else None
        source, image_a, image_b = augmentation.combined_operations_images(
            choice, record.source_image, record.image_a, record.image_b, distractor, work_dir, tag)
        if choice["swap"]:
            image_a, image_b = image_b, image_a
        new = dataclasses.replace(record, source_image=source, edit_instruction=choice["instruction"],
                                  image_a=image_a, image_b=image_b)
        info = {"draw": draw, "swap": bool(choice["swap"]), "applied_operations": choice.get("applied_operations"),
                "position": choice.get("position"), "instruction": choice["instruction"],
                "distractor": distractors[choice["distractor_index"]]["sample_id"]
                if choice.get("distractor_index") is not None else None}
        return new, info

    # Check draws 0-2 against the run's per-sample log (exact reproduction of the Student input).
    log = logged_draws(run_dir)
    checked, mismatches = 0, []
    inputs: Dict[str, List[Dict[str, Any]]] = {"unlabeled": [], "labeled": []}
    for sample_id in unlabeled_ids:
        record = unlabeled[sample_id]
        inputs["unlabeled"].append({"sample_id": sample_id, "input": "original", "record": record, "info": None})
        for draw in args.draws:
            new, info = augmented(record, unlabeled_scenarios[sample_id], sample_id, draw,
                                  f"unlabeled_{sample_id}_draw{draw}", True)
            logged = log.get((sample_id, draw))
            if logged is not None:
                checked += 1
                expected = (logged.get("student_instruction"), logged.get("augmentation_position"),
                            logged.get("augmentation_distractor"), bool(logged.get("augmentation_swap")))
                got = (info["instruction"], info["position"], info["distractor"], info["swap"])
                if expected != got:
                    mismatches.append({"sample_id": sample_id, "draw": draw, "logged": expected, "reproduced": got})
            inputs["unlabeled"].append({"sample_id": sample_id, "input": f"draw_{draw}", "record": new, "info": info})
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} reproduced augmentations differ from the run log, e.g. {mismatches[:2]}")
    for sample_id in labeled_ids:
        record = human[sample_id]
        extra = {"gemma_relevance": list(gemma[sample_id].relevance)}
        inputs["labeled"].append({"sample_id": sample_id, "input": "original", "record": record, "info": None, **extra})
        new, info = augmented(record, labeled_scenarios[sample_id], sample_id, 0,
                              f"labeled_{sample_id}_draw0", False)
        inputs["labeled"].append({"sample_id": sample_id, "input": "draw_0", "record": new, "info": info, **extra})
    return {"inputs": inputs, "draws_check": {"checked_against_log": checked, "mismatches": 0,
                                              "unlabeled_samples": len(unlabeled_ids),
                                              "labeled_samples": len(labeled_ids)}}


# ----------------------------------------------------------------------------- scoring
def score(model, processor, prompt, entry, device, swap: bool) -> Dict[str, Any]:
    import train_loop as tl

    with torch.no_grad():
        output = tl.forward_pairs(model, processor, [entry["record"]], prompt, device)
        if swap:
            output = tl.unswap_student_output(output, [True])
        overall = tl.overall_three_way_probs(output, None, "selected_equal")
    return {"relevance": output.relevance_probs.float()[0].cpu().tolist(),
            "metric_three_way": output.three_way_probs.float()[0].cpu().tolist(),
            "overall_three_way": overall.float()[0].cpu().tolist()}


def score_checkpoint(model, processor, prompt, states, inputs, args, device, path: Path) -> None:
    import train_loop as tl
    from peft import set_peft_model_state_dict

    started = time.time()
    with path.open("x", encoding="utf-8") as handle:
        for role in ("student", "teacher"):
            result = set_peft_model_state_dict(model, states[role], adapter_name="default")
            if result.unexpected_keys:
                raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys[:5]}")
            plans = []  # (set, entry, input name, dropout pass or None)
            for set_name, entries in inputs.items():
                for entry in entries:
                    if role == "teacher" and entry["input"] != "original":
                        continue
                    if role == "student" and set_name == "labeled" and entry["input"] not in ("original", "draw_0"):
                        continue
                    plans.append((set_name, entry, entry["input"], None))
                    if entry["input"] == "original" and set_name == "unlabeled":
                        plans += [(set_name, entry, f"original_dropout_{j}", j) for j in range(args.dropout_passes)]
            for number, (set_name, entry, input_name, dropout_pass) in enumerate(plans, start=1):
                if dropout_pass is None:
                    model.eval()
                else:
                    tl.enable_selected_mc_dropout(model)
                    torch.manual_seed(1000 * (dropout_pass + 1) + number)
                swap = bool(entry["info"] and entry["info"]["swap"])
                row = {"set": set_name, "sample_id": entry["sample_id"], "role": role, "input": input_name,
                       **score(model, processor, prompt, entry, device, swap)}
                if entry["info"]:
                    row["augmentation"] = entry["info"]
                if "gemma_relevance" in entry:
                    row["gemma_relevance"] = entry["gemma_relevance"]
                handle.write(json.dumps(row) + "\n")
                if number % 100 == 0 or number == len(plans):
                    print(f"{path.stem} {role}: {number}/{len(plans)} ({time.time() - started:.0f} s)", flush=True)
            model.eval()


# ----------------------------------------------------------------------------- summary
def selected(relevance: Sequence[float]) -> np.ndarray:
    r = np.asarray(relevance)
    s = r >= THRESHOLD
    if not s.any():
        s = np.zeros_like(s)
        s[int(r.argmax())] = True
    return s


def label(three_way: Sequence[float]) -> int:
    return int(np.argmax(three_way))


def compare(a_rows: Dict[str, Dict], b_rows: Dict[str, Dict]) -> Dict[str, Any]:
    ids = sorted(set(a_rows) & set(b_rows))
    if not ids:
        return {}
    ra = np.asarray([a_rows[i]["relevance"] for i in ids])
    rb = np.asarray([b_rows[i]["relevance"] for i in ids])
    sa = np.asarray([selected(a_rows[i]["relevance"]) for i in ids])
    sb = np.asarray([selected(b_rows[i]["relevance"]) for i in ids])
    out = {"samples": len(ids),
           "s_same_rate": float((sa == sb).all(axis=1).mean()),
           "overall_label_same_rate": float(np.mean([label(a_rows[i]["overall_three_way"]) == label(b_rows[i]["overall_three_way"]) for i in ids])),
           "relevance_absolute_difference_mean": float(np.abs(ra - rb).mean())}
    for m, metric in enumerate(METRICS):
        out[f"{metric}_difference_mean (first - second)"] = float((ra[:, m] - rb[:, m]).mean())
        out[f"{metric}_selection_mismatch_rate"] = float((sa[:, m] != sb[:, m]).mean())
    return out


def gemma_report(rows: Dict[str, Dict]) -> Dict[str, Any]:
    ids = sorted(rows)
    if not ids:
        return {}
    r = np.clip(np.asarray([rows[i]["relevance"] for i in ids]), 1e-6, 1 - 1e-6)
    g = np.asarray([rows[i]["gemma_relevance"] for i in ids], dtype=float)
    bce = -(g * np.log(r) + (1 - g) * np.log(1 - r))
    out = {"samples": len(ids), "bce_mean": float(bce.mean()),
           "selection_agreement_rate": float(((r >= THRESHOLD) == (g >= THRESHOLD)).mean())}
    for m, metric in enumerate(METRICS):
        out[f"{metric}_bce"] = float(bce[:, m].mean())
        out[f"{metric}_difference_mean (model - GemmaGT)"] = float((r[:, m] - g[:, m]).mean())
    return out


def summarize(path: Path) -> Dict[str, Any]:
    rows: Dict[tuple, Dict[str, Dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["set"], row["role"], row["input"])][row["sample_id"]] = row
    draws = sorted(key[2] for key in rows if key[:2] == ("unlabeled", "student") and key[2].startswith("draw_"))
    dropout = sorted(key[2] for key in rows if key[:2] == ("unlabeled", "teacher") and "dropout" in key[2])
    student_dropout = sorted(key[2] for key in rows if key[:2] == ("unlabeled", "student") and "dropout" in key[2])
    u = lambda role, name: rows.get(("unlabeled", role, name), {})  # noqa: E731
    out: Dict[str, Any] = {"checkpoint": path.stem.replace("predictions_", ""), "comparisons": {}}
    c = out["comparisons"]
    # (b) same weights, different augmentation draws
    for a, b in itertools.combinations(draws, 2):
        c[f"b_student_{a}_vs_student_{b}"] = compare(u("student", a), u("student", b))
    # (d1) augmentation effect, same weights
    for d in draws:
        c[f"d1_student_{d}_vs_student_original"] = compare(u("student", d), u("student", "original"))
    # (d2) weights: Student vs Teacher on the same original input
    c["d2_student_original_vs_teacher_original"] = compare(u("student", "original"), u("teacher", "original"))
    # what the training log measures (Student augmented vs Teacher original, with / without dropout)
    for d in draws:
        c[f"log_student_{d}_vs_teacher_original"] = compare(u("student", d), u("teacher", "original"))
        if dropout:
            c[f"log_student_{d}_vs_teacher_{dropout[0]}"] = compare(u("student", d), u("teacher", dropout[0]))
    # (f) dropout noise
    for role, names in (("teacher", dropout), ("student", student_dropout)):
        for a, b in itertools.combinations(names, 2):
            c[f"f_{role}_{a}_vs_{role}_{b}"] = compare(u(role, a), u(role, b))
        for name in names:
            c[f"f_{role}_{name}_vs_{role}_original"] = compare(u(role, name), u(role, "original"))
    # (d2) labeled pairs: relevance against GemmaGT
    for role, name in (("student", "original"), ("student", "draw_0"), ("teacher", "original")):
        c[f"d2_labeled_{role}_{name}_vs_gemmaGT"] = gemma_report(rows.get(("labeled", role, name), {}))
    c["d1_labeled_student_draw_0_vs_student_original"] = compare(
        rows.get(("labeled", "student", "draw_0"), {}), rows.get(("labeled", "student", "original"), {}))
    return out


def write_summaries(output_dir: Path) -> None:
    table = []
    for path in sorted(output_dir.glob("predictions_*.jsonl")):
        summary = summarize(path)
        (output_dir / f"summary_{summary['checkpoint']}.json").write_text(json.dumps(summary, indent=1))
        for name, values in summary["comparisons"].items():
            if values:
                table.append({"checkpoint": summary["checkpoint"], "comparison": name, **values})
    if table:
        columns: List[str] = []
        for row in table:
            columns.extend(k for k in row if k not in columns)
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in table:
                writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    print(f"Summaries written to {output_dir}", flush=True)


# ----------------------------------------------------------------------------- main
def checkpoint_states(name: str, args: argparse.Namespace) -> tuple:
    if name == "start":
        student = torch.load(args.start_student, map_location="cpu", mmap=True, weights_only=False)
        teacher = torch.load(args.start_teacher, map_location="cpu", mmap=True, weights_only=False)
        if student.get("prompt") != teacher.get("prompt"):
            raise ValueError("Start Student and Teacher prompts differ.")
        return "start", student, {"student": student["adapter_state_dict"], "teacher": teacher["adapter_state_dict"]}
    checkpoint = torch.load(name, map_location="cpu", mmap=True, weights_only=False)
    states = {role: checkpoint.get(f"{role}_adapter_state_dict") for role in ("student", "teacher")}
    if not all(isinstance(s, dict) and s for s in states.values()):
        raise ValueError(f"{name} lacks the Student or Teacher adapter.")
    return Path(name).stem, checkpoint, states


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.summarize_only:
        write_summaries(output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [n if n == "start" else str(Path(n).resolve()) for n in args.checkpoints]
    if not names:
        raise ValueError("--checkpoints is empty")
    existing = [p for p in (output_dir / f"predictions_{'start' if n == 'start' else Path(n).stem}.jsonl" for n in names) if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {existing}")
    work_dir = output_dir / "augmentation_tmp"
    work_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    built = build_inputs(args, work_dir)
    built["draws_check"]["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    built["draws_check"]["arguments"] = vars(args)
    (output_dir / "draws_check.json").write_text(json.dumps(built["draws_check"], indent=1))
    print(f"Inputs: {json.dumps({k: v for k, v in built['draws_check'].items() if k != 'arguments'})}", flush=True)

    from evaluate_reward_model import load_refined_model

    model = processor = prompt = None
    for name in names:
        stem, checkpoint, states = checkpoint_states(name, args)
        if model is None:
            # checkpoint-N.pt (refinement role) or student_warmstart.pt; both adapters are set later.
            model, processor = load_refined_model(
                checkpoint, "student", device, None, None,
                base_model_override=Path(args.base_model_path).expanduser().resolve())
            prompt = checkpoint.get("prompt")
        elif checkpoint.get("prompt") != prompt:
            raise ValueError(f"{name}: prompt differs from the first checkpoint.")
        del checkpoint
        print(f"Scoring {stem}", flush=True)
        score_checkpoint(model, processor, prompt, states, built["inputs"], args, device,
                         output_dir / f"predictions_{stem}.jsonl")
    shutil.rmtree(work_dir, ignore_errors=True)  # composed images (quota); predictions keep the draw info
    write_summaries(output_dir)


if __name__ == "__main__":
    main()
