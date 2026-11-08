"""Stage 2 tuning: offline evaluation of checkpoints on the Stage 2 training data.

Step 2 of the tuning evaluation order (training-log analysis, then this, then
EditReward-Bench). No label outside the 1008 labeled pairs is used.

Two commands:

``sets``      chooses the fixed evaluation sets once and writes them to JSON:
              - trained Unlabeled: 256 pairs of each Unlabeled subset the tuning
                runs trained on (all of a 256 subset);
              - untrained Unlabeled: 256 pairs outside every trained subset and
                outside the diagnostic probe, preferably from source images that
                no trained pair uses;
              - labeled: 256 of the 1008 HumanGT + GemmaGT training pairs.
``evaluate``  loads the model once and scores each checkpoint (Student or
              Teacher adapter) on the run's trained set, the untrained set and the
              labeled set; with ``--all-trained-sets`` (the Student-Stage1 start)
              on every trained set.

The overall three-way probabilities use the training definition:
selected_equal mean of the per-metric three_way_probs over the metrics with
relevance >= 0.5 (else the most relevant one). Per-pair outputs are saved, so
pair-level comparisons with the start (changed predictions, |delta margin|)
are computed later without the GPU.
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
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from peft import set_peft_model_state_dict
from scipy.stats import spearmanr

_EVAL_DIR = Path(__file__).resolve().parent  # RewardModelEvaluation/OurModel
_PROJECT_ROOT = _EVAL_DIR.parents[1] / "TrainingRewardModel" / "Qwen3-VL-Embedding"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_reward_model import load_refined_model  # noqa: E402
from train_loop import (  # noqa: E402
    DEFAULT_EDITREWARD_ROOT,
    PROBABILITY_EPS,
    FactorizedWarmStartLoss,
    forward_pairs,
    gemma_replay_losses,
    human_three_way_targets,
    index_unlabeled_pool,
    load_gemma_records,
    load_human_records,
    load_inspector_cache,
    majority_vote_gemma_record,
    materialize_unlabeled_round,
    overall_three_way_probs,
    select_diagnostic_probe,
    three_way_cross_entropy,
)
from src.models.custom_qwen3_vl_reranker import NUM_METRICS  # noqa: E402

SETS_FORMAT_VERSION = 1
SET_SIZE = 256
CONFIDENT_INSPECTOR = 0.7
UNTRAINED_KEY = "untrained_unlabeled"
LABELED_KEY = "labeled"
LABELED_ALL_KEY = "labeled_all"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_key(subset: Mapping[str, Any]) -> str:
    return f"trained_unlabeled_seed{subset['seed']}_size{subset['subset_size']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sets = commands.add_parser("sets", help="Choose and write the fixed evaluation sets.")
    sets.add_argument("--student-checkpoint", required=True,
                      help="Student-Stage1 student_warmstart.pt (supplies the 1008 pair IDs).")
    sets.add_argument("--run-dirs", nargs="+", required=True,
                      help="Tuning run directories; their unlabeled_subset.json are the trained subsets.")
    sets.add_argument("--unlabeled-dataset", required=True)
    sets.add_argument("--inspector-cache", required=True)
    sets.add_argument("--editreward-root", default=DEFAULT_EDITREWARD_ROOT)
    sets.add_argument("--diagnostic-probe-size", type=int, default=32)
    sets.add_argument("--selection-seed", type=int, default=42)
    sets.add_argument("--seed", type=int, default=20261102)
    sets.add_argument("--output", required=True)

    evaluate = commands.add_parser("evaluate", help="Score checkpoints on the fixed sets.")
    evaluate.add_argument("--evaluation-sets", required=True)
    evaluate.add_argument("--checkpoints", nargs="+", required=True,
                          help="train_loop checkpoint-N.pt files, or a student_warmstart.pt.")
    evaluate.add_argument("--role", choices=("student", "teacher"), default="student")
    evaluate.add_argument("--run-dir", default=None,
                          help="Run directory whose unlabeled_subset.json names the trained set.")
    evaluate.add_argument("--trained-set-key", default=None,
                          help="Use this trained set of the sets file instead of the run's own subset "
                               "key; every ID must lie inside the run's unlabeled_subset.json.")
    evaluate.add_argument("--skip-untrained-set", action="store_true",
                          help="Do not score the untrained Unlabeled set.")
    evaluate.add_argument("--all-trained-sets", action="store_true",
                          help="Score every trained set (for the Student-Stage1 start).")
    evaluate.add_argument("--labeled-all-pairs", action="store_true",
                          help="Score all 1008 Student-Stage1 pairs (labeled_all); the 256-pair "
                               "labeled report is then taken from the same rows.")
    evaluate.add_argument("--human-dataset", required=True)
    evaluate.add_argument("--gemma-dataset", required=True)
    evaluate.add_argument("--unlabeled-dataset", required=True)
    evaluate.add_argument("--inspector-cache", required=True)
    evaluate.add_argument("--editreward-root", default=DEFAULT_EDITREWARD_ROOT)
    evaluate.add_argument("--base-model-path", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------- sets


def build_sets(args: argparse.Namespace) -> None:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation sets: {output}")
    student = torch.load(args.student_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    labeled_ids = list((student.get("data") or {}).get("train_sample_ids") or [])
    if len(labeled_ids) != 1008 or len(set(labeled_ids)) != len(labeled_ids):
        raise ValueError(f"Expected the 1008 Student-Stage1 pair IDs, found {len(labeled_ids)}.")
    rng = random.Random(args.seed)
    labeled = rng.sample(labeled_ids, SET_SIZE)

    root = Path(args.editreward_root).expanduser().resolve()
    pool = index_unlabeled_pool(Path(args.unlabeled_dataset), root)
    inspector = load_inspector_cache(Path(args.inspector_cache), pool.sample_ids)
    probe = select_diagnostic_probe(
        inspector, pool.source_groups, args.diagnostic_probe_size, args.selection_seed + 40_000
    )
    index_of = {sample_id: index for index, sample_id in enumerate(pool.sample_ids)}

    trained: Dict[str, Dict[str, Any]] = {}
    for run_dir in args.run_dirs:
        path = Path(run_dir) / "unlabeled_subset.json"
        subset = json.loads(path.read_text(encoding="utf-8"))
        key = subset_key(subset)
        ids = list(subset["sample_ids"])
        if key in trained:
            if trained[key]["subset_ids"] != ids:
                raise ValueError(f"{path} differs from another subset with the same seed and size.")
            trained[key]["runs"].append(str(Path(run_dir).resolve()))
            continue
        missing = [sample_id for sample_id in ids if sample_id not in index_of]
        if missing:
            raise ValueError(f"{path}: {len(missing)} IDs are not in the Unlabeled dataset.")
        trained[key] = {"subset_ids": ids, "runs": [str(Path(run_dir).resolve())]}
    for key in sorted(trained):
        ids = trained[key]["subset_ids"]
        trained[key]["sample_ids"] = (
            list(ids) if len(ids) <= SET_SIZE else rng.sample(ids, SET_SIZE)
        )

    trained_indices = {index_of[i] for entry in trained.values() for i in entry["subset_ids"]}
    trained_groups = {pool.source_groups[index] for index in trained_indices}
    probe_groups = {pool.source_groups[index] for index in probe}
    candidates = [
        index for index in range(len(pool.sample_ids))
        if index not in trained_indices
        and pool.source_groups[index] not in trained_groups
        and pool.source_groups[index] not in probe_groups
    ]
    rule = "not trained, source image unused by trained pairs and the probe"
    if len(candidates) < SET_SIZE:
        candidates = [
            index for index in range(len(pool.sample_ids))
            if index not in trained_indices and pool.source_groups[index] not in probe_groups
        ]
        rule = "not trained, source image unused by the probe (may share trained source images)"
    untrained = [pool.sample_ids[index] for index in sorted(rng.sample(candidates, SET_SIZE))]

    result = {
        "format_version": SETS_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "inputs": {
            "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
            "unlabeled_dataset": str(Path(args.unlabeled_dataset).resolve()),
            "unlabeled_dataset_md5": file_md5(Path(args.unlabeled_dataset)),
            "inspector_cache": str(Path(args.inspector_cache).resolve()),
            "inspector_cache_md5": file_md5(Path(args.inspector_cache)),
            "diagnostic_probe_size": args.diagnostic_probe_size,
            "selection_seed": args.selection_seed,
        },
        "trained": {
            key: {"runs": entry["runs"], "subset_size": len(entry["subset_ids"]),
                  "sample_ids": entry["sample_ids"]}
            for key, entry in sorted(trained.items())
        },
        UNTRAINED_KEY: {"rule": rule, "candidates": len(candidates), "sample_ids": untrained},
        LABELED_KEY: {"rule": f"{SET_SIZE} of the 1008 Student-Stage1 training pairs",
                      "sample_ids": labeled},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Evaluation sets: trained={ {k: len(v['sample_ids']) for k, v in result['trained'].items()} }, "
        f"untrained={len(untrained)} ({rule}; {len(candidates)} candidates), labeled={len(labeled)}",
        flush=True,
    )
    print(f"Evaluation sets output: {output}", flush=True)


# ---------------------------------------------------------------- evaluate


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(np.clip(probabilities, PROBABILITY_EPS, None))).sum(axis=-1)


def _mean(values) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return round(float(values.mean()), 6) if values.size else None


def unlabeled_report(student: np.ndarray, inspector: np.ndarray, selected: np.ndarray) -> Dict[str, Any]:
    """student / inspector: [N, 3] (P(A), P(B), P(Tie)); selected: [N] metric counts."""
    cross_entropy = -(inspector * np.log(np.clip(student, PROBABILITY_EPS, None))).sum(axis=-1)
    target_entropy = _entropy(inspector)
    agree = student.argmax(axis=-1) == inspector.argmax(axis=-1)
    confident = inspector.max(axis=-1) >= CONFIDENT_INSPECTOR
    student_margin = student[:, 0] - student[:, 1]
    inspector_margin = inspector[:, 0] - inspector[:, 1]
    rho = spearmanr(student_margin, inspector_margin).statistic
    return {
        "pairs": int(len(student)),
        "inspector_cross_entropy": _mean(cross_entropy),
        "inspector_target_entropy": _mean(target_entropy),
        "inspector_kl": _mean(cross_entropy - target_entropy),
        "argmax_agreement": _mean(agree),
        "argmax_agreement_confident_inspector": _mean(agree[confident]),
        "argmax_agreement_unconfident_inspector": _mean(agree[~confident]),
        "confident_inspector_pairs": int(confident.sum()),
        "margin_spearman": round(float(rho), 6) if math.isfinite(rho) else None,
        "same_margin_sign": _mean(np.sign(student_margin) == np.sign(inspector_margin)),
        "student_mean_absolute_margin": _mean(np.abs(student_margin)),
        "inspector_mean_absolute_margin": _mean(np.abs(inspector_margin)),
        "student_mean_tie_probability": _mean(student[:, 2]),
        "inspector_mean_tie_probability": _mean(inspector[:, 2]),
        "student_mean_entropy": _mean(_entropy(student)),
        "student_mean_max_probability": _mean(student.max(axis=-1)),
        "inspector_mean_max_probability": _mean(inspector.max(axis=-1)),
        "student_argmax_share": {
            name: _mean(student.argmax(axis=-1) == k) for k, name in enumerate(("A", "B", "Tie"))
        },
        "selected_metric_count": {
            str(k): int(v) for k, v in sorted(Counter(selected.astype(int).tolist()).items())
        },
    }


def labeled_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    probabilities = np.asarray([row["overall_three_way"] for row in rows])
    targets = np.asarray([row["human_target"] for row in rows])
    margin = probabilities[:, 0] - probabilities[:, 1]
    non_tie = targets[:, 2] == 0
    human_sign = np.where(targets[:, 0] == 1, 1.0, -1.0)
    report = {
        "pairs": len(rows),
        "human_tie_pairs": int((~non_tie).sum()),
        "human_three_way_cross_entropy": _mean([row["human_cross_entropy"] for row in rows]),
        "human_three_class_accuracy": _mean(probabilities.argmax(axis=-1) == targets.argmax(axis=-1)),
        "human_non_tie_accuracy": _mean((np.sign(margin) == human_sign)[non_tie]),
        "student_mean_absolute_margin": _mean(np.abs(margin)),
        "student_mean_tie_probability": _mean(probabilities[:, 2]),
        "student_mean_entropy": _mean(_entropy(probabilities)),
    }
    for name in ("gemma_preference_loss_majority_vote", "gemma_relevance_loss_majority_vote",
                 "gemma_preference_loss_soft", "gemma_relevance_loss_soft"):
        report[name] = _mean([row[name] for row in rows])
    tie_error = np.concatenate([row["gemma_tie_squared_error"] for row in rows])
    report["gemma_tie_brier_soft"] = _mean(tie_error) if tie_error.size else None
    return report


def score_pairs(model, processor, prompt, records, device, extra=None):
    """Forward one pair at a time; returns per-pair rows (``extra`` adds labeled losses)."""
    rows = []
    for position, record in enumerate(records):
        # Same forward as evaluate_student_checkpoints.py (no autocast).
        with torch.no_grad():
            output = forward_pairs(model, processor, [record], prompt, device)
        three_way = overall_three_way_probs(output, None, "selected_equal")
        relevance = output.relevance_probs.float()
        row = {
            "sample_id": record.sample_id,
            "overall_three_way": three_way[0].cpu().tolist(),
            "selected_metric_count": int(
                max(1, int((relevance[0] >= 0.5).sum()))
            ),
            "tie_probs": output.tie_probs.float()[0].cpu().tolist(),
            "cond_win_prob_A": output.cond_win_prob_A.float()[0].cpu().tolist(),
            "relevance_probs": relevance[0].cpu().tolist(),
        }
        if extra is not None:
            row.update(extra(output, three_way, position))
        rows.append(row)
        if (position + 1) % 64 == 0 or position + 1 == len(records):
            print(f"Scored {position + 1}/{len(records)}", flush=True)
    return rows


def checkpoint_state(checkpoint: Mapping[str, Any], role: str) -> Dict[str, torch.Tensor]:
    key = f"{role}_adapter_state_dict" if f"{role}_adapter_state_dict" in checkpoint else "adapter_state_dict"
    if role == "teacher" and key != "teacher_adapter_state_dict":
        raise ValueError("Teacher evaluation needs a train_loop checkpoint.")
    state = checkpoint.get(key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint is missing {key}")
    return state


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    sets = json.loads(Path(args.evaluation_sets).read_text(encoding="utf-8"))
    if sets.get("format_version") != SETS_FORMAT_VERSION:
        raise ValueError("Unsupported evaluation sets file.")
    if file_md5(Path(args.inspector_cache)) != sets["inputs"]["inspector_cache_md5"]:
        raise ValueError("Inspector cache differs from the one the evaluation sets were built with.")
    if args.all_trained_sets:
        trained_keys = sorted(sets["trained"])
    else:
        if not args.run_dir:
            raise ValueError("--run-dir is required without --all-trained-sets.")
        subset = json.loads((Path(args.run_dir) / "unlabeled_subset.json").read_text(encoding="utf-8"))
        key = args.trained_set_key or subset_key(subset)
        if key not in sets["trained"]:
            raise ValueError(f"{key} is not in the evaluation sets; rebuild them with this run.")
        outside = set(sets["trained"][key]["sample_ids"]) - set(subset["sample_ids"])
        if outside:
            raise ValueError(f"{len(outside)} IDs of {key} are not in the run's trained subset.")
        trained_keys = [key]
    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = "" if args.role == "student" else f"{args.role}_"
    suffix = "_labeled_all" if args.labeled_all_pairs else ""
    outputs = [output_dir / f"{prefix}{Path(path).stem}{suffix}.json" for path in args.checkpoints]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing evaluations: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.editreward_root).expanduser().resolve()
    pool = index_unlabeled_pool(Path(args.unlabeled_dataset), root)
    inspector = load_inspector_cache(Path(args.inspector_cache), pool.sample_ids)
    index_of = {sample_id: index for index, sample_id in enumerate(pool.sample_ids)}
    unlabeled_sets = {key: sets["trained"][key]["sample_ids"] for key in trained_keys}
    if not args.skip_untrained_set:
        unlabeled_sets[UNTRAINED_KEY] = sets[UNTRAINED_KEY]["sample_ids"]
    unlabeled_records = {}
    for name, ids in unlabeled_sets.items():
        indices = [index_of[sample_id] for sample_id in ids]
        by_index = materialize_unlabeled_round(pool, indices)
        unlabeled_records[name] = (
            [by_index[index] for index in indices],
            inspector.three_way[torch.tensor(indices)].double().numpy(),
        )

    labeled_ids = list(sets[LABELED_KEY]["sample_ids"])
    if args.labeled_all_pairs:
        stage1 = torch.load(sets["inputs"]["student_checkpoint"], map_location="cpu", mmap=True,
                            weights_only=False)
        all_ids = list((stage1.get("data") or {}).get("train_sample_ids") or [])
        del stage1
        if len(all_ids) != 1008 or not set(labeled_ids) <= set(all_ids):
            raise ValueError("Student-Stage1 pair IDs do not contain the 256-pair labeled set.")
        labeled_subset = set(labeled_ids)
        labeled_ids = labeled_ids + [sample_id for sample_id in all_ids if sample_id not in labeled_subset]
    # Labeled pairs drawn in Stage 2 (round summaries written after 2026-11-02 only).
    stage2_labeled: set[str] = set()
    if args.run_dir:
        for summary_path in sorted(Path(args.run_dir).glob("round_*_summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            stage2_labeled.update((summary.get("trained_labeled_sample_ids") or {}).keys())
    human_by_id = {
        record.sample_id: record
        for record in load_human_records(Path(args.human_dataset), root, keep_ties=True)
    }
    gemma_by_id = {record.sample_id: record for record in load_gemma_records(Path(args.gemma_dataset))}
    human_records = [human_by_id[sample_id] for sample_id in labeled_ids]
    gemma_soft = [gemma_by_id[sample_id] for sample_id in labeled_ids]
    gemma_hard = [majority_vote_gemma_record(record) for record in gemma_soft]
    for human, gemma in zip(human_records, gemma_soft):
        if (human.image_a, human.image_b, human.source_image) != (gemma.image_a, gemma.image_b, gemma.source_image):
            raise ValueError(f"HumanGT and GemmaGT differ for {human.sample_id}")
    objective = FactorizedWarmStartLoss()

    def labeled_extra(output, three_way, position):
        labels = torch.tensor([human_records[position].label_a], device=device)
        target = human_three_way_targets(labels)
        soft = gemma_replay_losses(output, [gemma_soft[position]], objective, device)
        hard = gemma_replay_losses(output, [gemma_hard[position]], objective, device)
        record = gemma_soft[position]
        applicable = torch.tensor(record.relevance, device=device) > 0.0
        tie_error = (output.tie_probs.float()[0] - torch.tensor(record.g_t, device=device)) ** 2
        return {
            "human_label_a": human_records[position].label_a,
            "human_target": target[0].cpu().tolist(),
            "human_cross_entropy": float(three_way_cross_entropy(three_way, target)[0]),
            "gemma_preference_loss_majority_vote": float(hard["preference_loss"]),
            "gemma_relevance_loss_majority_vote": float(hard["relevance_loss"]),
            "gemma_preference_loss_soft": float(soft["preference_loss"]),
            "gemma_relevance_loss_soft": float(soft["relevance_loss"]),
            "gemma_tie_squared_error": tie_error[applicable].cpu().tolist(),
        }

    model = processor = None
    for checkpoint_path, output_path in zip(args.checkpoints, outputs):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        prompt = checkpoint.get("prompt")
        if not isinstance(prompt, dict):
            raise ValueError(f"{checkpoint_path} is missing the prompt contract.")
        if model is None:
            model, processor = load_refined_model(
                checkpoint, args.role, device, None, None,
                base_model_override=Path(args.base_model_path).expanduser().resolve(),
            )
        else:
            result = set_peft_model_state_dict(model, checkpoint_state(checkpoint, args.role),
                                               adapter_name="default")
            if result.unexpected_keys:
                raise RuntimeError(f"Unexpected keys in {checkpoint_path}: {result.unexpected_keys[:10]}")
        model.eval()
        print(f"Evaluating {args.role} of {checkpoint_path}", flush=True)
        reports: Dict[str, Any] = {}
        predictions: Dict[str, Any] = {}
        for name, (records, targets) in unlabeled_records.items():
            print(f"{name}: {len(records)} pairs", flush=True)
            rows = score_pairs(model, processor, prompt, records, device)
            student = np.asarray([row["overall_three_way"] for row in rows], dtype=np.float64)
            selected = np.asarray([row["selected_metric_count"] for row in rows])
            reports[name] = unlabeled_report(student, targets, selected)
            for row, target in zip(rows, targets):
                row["inspector_three_way"] = target.tolist()
            predictions[name] = rows
            print(f"[{name}] {json.dumps(reports[name])}", flush=True)
        print(f"{LABELED_KEY}: {len(human_records)} pairs", flush=True)
        rows = score_pairs(model, processor, prompt, human_records, device, labeled_extra)
        for row, sample_id in zip(rows, labeled_ids):
            row.setdefault("sample_id", sample_id)
            row["trained_in_stage2"] = sample_id in stage2_labeled if stage2_labeled else None
        reports[LABELED_KEY] = labeled_report(rows[:SET_SIZE])
        predictions[LABELED_KEY] = rows[:SET_SIZE]
        if args.labeled_all_pairs:
            reports[LABELED_ALL_KEY] = labeled_report(rows)
            predictions[LABELED_ALL_KEY] = rows[SET_SIZE:]
            if stage2_labeled:
                for flag, name in ((True, "trained_in_stage2"), (False, "not_trained_in_stage2")):
                    part = [row for row in rows if row["trained_in_stage2"] is flag]
                    if part:
                        reports[f"{LABELED_ALL_KEY}_{name}"] = labeled_report(part)
        for name in (LABELED_KEY, LABELED_ALL_KEY):
            if name in reports:
                print(f"[{name}] {json.dumps(reports[name])}", flush=True)
        result = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_role": args.role,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "optimizer_step": checkpoint.get("optimizer_step", 0),
            "evaluation_sets": str(Path(args.evaluation_sets).resolve()),
            "overall_aggregation": "selected_equal",
            "configuration": vars(args),
            "metrics": reports,
            "predictions": predictions,
        }
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Evaluation output: {output_path}", flush=True)
        del checkpoint


def main() -> None:
    args = parse_args()
    if args.command == "sets":
        build_sets(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
