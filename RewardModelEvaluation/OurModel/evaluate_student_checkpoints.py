#!/usr/bin/env python3
"""Evaluate a six-metric Student Model checkpoint on EditReward-Data-Test and EditReward-Bench.

Pairwise metrics use the non-tie pairs, exactly as the Overall Inspector Model
checkpoint evaluation (``evaluate_inspector_checkpoints.py``). EditReward-Bench
keeps its tie pairs for the three-way ``selected_equal`` metrics: per pair,
P(A), P(B), P(Tie) are the means of the per-metric three-way probabilities
over the selected metrics S; Tie AUC is reported from that P(Tie) and from
-|mean_{k in S} d_k| (d_k = score_A_k - score_B_k).

With ``--gemma-dataset`` (default: the dataset recorded in the checkpoint) the
Student's GemmaGT validation pairs are also scored per metric on the
majority-vote cells (relevant when >= 2 of 3 runs mark the metric; A/B/Tie by
strict majority, else Tie): Tie AUC from t_k and from -|d_k|, Brier on Tie and
non-Tie cells.

The six metrics are combined into one overall preference in two ways, both
read from the Student's Metrics Weights Aggregator file:

- ``equal_weight``: a_k = 1/6, with the beta fitted for equal weights,
- ``fitted``: the fitted a_k and beta.

Both need ``--preference-aggregator``; without it only the selected
aggregations that need no fitted weights (``selected_equal``) are reported.

Per pair, ``margin = beta * sum_k a_k * signed_evidence_k`` gives the accuracy
and sigmoid(margin) the Brier score and log loss. Pooled AUC needs one score
per image, so it uses the Score Head: ``score(x) = sum_k a_k * score_k(x)``.

Two more aggregations use only the metrics the Relevance Head selects for the
pair, S = {k : relevance_probs_k >= 0.5} (the most relevant metric when none
passes), exactly as the model's inference wrapper:

- ``selected_equal``: w_k = 1 for k in S (primary result),
- ``selected_fitted``: w_k = a_k for k in S.

Their margin is sum_k w_k * signed_evidence_k / sum_k w_k (the mean over S for
``selected_equal``), used for accuracy and the symmetric margin AUC; pooled AUC
uses the image score sum_k w_k * score_k / sum_k w_k. No beta is fitted for
them, so no Brier score or log loss.

The raw per-metric outputs are saved, so other aggregations can be computed
later without the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

_EVAL_DIR = Path(__file__).resolve().parent  # RewardModelEvaluation/OurModel
_PROJECT_ROOT = _EVAL_DIR.parents[1] / "TrainingRewardModel" / "Qwen3-VL-Embedding"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_inspector_checkpoints import (  # noqa: E402
    BENCH_DATASET,
    _margin_auc,
    _pooled_auc,
    _probability_metrics,
    _three_way_metrics,
)
from evaluate_reward_model import (  # noqa: E402
    BenchmarkRecord,
    load_benchmark,
    load_refined_model,
    summarize,
    tie_ranking_metrics,
)
from train_loop import PairRecord, forward_pairs  # noqa: E402
from train_overall_inspector import load_inspector_records  # noqa: E402
from src.models.custom_qwen3_vl_reranker import METRIC_NAMES, NUM_METRICS  # noqa: E402

TEST_DATASET = "EditReward-Data-Test"
VOTE_TO_LABEL = {"A": 1, "B": -1, "Tie": 0}
AGGREGATIONS = ("equal_weight", "fitted")
SELECTED_AGGREGATIONS = ("selected_equal", "selected_fitted")
RELEVANCE_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The refinement checkpoint holds both adapters; teacher scores the EMA Teacher Model.
    parser.add_argument("--role", choices=("student", "teacher"), default="student")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="MS-Swift checkpoint-N directory of a Stage1 warm-start run: its "
        "adapter_model.safetensors replaces the adapter of --checkpoint, which "
        "still supplies the prompt, PEFT config and Tie Head mode.",
    )
    parser.add_argument("--preference-aggregator", default=None)
    parser.add_argument(
        "--gemma-dataset",
        default=None,
        help="GemmaGT Data for the per-metric Tie evaluation; default: from the checkpoint.",
    )
    parser.add_argument("--skip-gemma-validation", action="store_true")
    parser.add_argument(
        "--datasets",
        choices=("all", "bench"),
        default="all",
        help="all: EditReward-Data test set and EditReward-Bench; bench: "
        "EditReward-Bench only, written to bench_evaluation.json.",
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--human-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--editreward-root", required=True)
    parser.add_argument("--editreward-bench", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _to_benchmark(records) -> List[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            sample_id=record.sample_id,
            source_image=record.source_image,
            edit_instruction=record.edit_instruction,
            image_a=record.image_a,
            image_b=record.image_b,
            label=VOTE_TO_LABEL[record.vote],
        )
        for record in records
    ]


def load_datasets(args: argparse.Namespace) -> Dict[str, List[BenchmarkRecord]]:
    if args.datasets == "bench":
        # Ties are kept: pairwise metrics skip them, three-way metrics use them.
        return {BENCH_DATASET: load_benchmark(Path(args.editreward_bench))}
    test_records, _ = load_inspector_records(args.test_dataset, args.editreward_root)
    if any(record.vote == "Tie" for record in test_records):
        raise ValueError("The EditReward-Data test set must not contain ties.")
    human_records, _ = load_inspector_records(args.human_dataset, args.editreward_root)
    # Like Unlabeled Data, test pairs may share a source image and
    # instruction with HumanGT Data, but never a pair or an edited image.
    if {r.sample_id for r in test_records} & {r.sample_id for r in human_records}:
        raise AssertionError("EditReward-Data test set shares a pair with HumanGT Data.")
    human_images = {img for r in human_records for img in (r.image_a, r.image_b)}
    if {img for r in test_records for img in (r.image_a, r.image_b)} & human_images:
        raise AssertionError("EditReward-Data test set shares an edited image with HumanGT Data.")
    return {
        TEST_DATASET: _to_benchmark(test_records),
        # Ties are kept: pairwise metrics skip them, three-way metrics use them.
        BENCH_DATASET: load_benchmark(Path(args.editreward_bench)),
    }


def read_aggregations(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if list(payload.get("metric_order", [])) != list(METRIC_NAMES):
        raise ValueError("Metrics Weights Aggregator metric order is incompatible.")
    if payload.get("feature_definition") != "signed_evidence":
        raise ValueError("Only the signed_evidence Metrics Weights Aggregator is supported.")
    result = {}
    for name, source in (("equal_weight", payload["equal_weight_baseline"]), ("fitted", payload)):
        weights = np.asarray(source["weights"], dtype=np.float64)
        beta = float(source["beta"])
        if (
            weights.shape != (NUM_METRICS,)
            or np.any(weights < 0)
            or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-6)
            or not math.isfinite(beta)
            or beta <= 0
        ):
            raise ValueError(f"Invalid {name} weights or beta in {path}")
        result[name] = {"weights": weights, "beta": beta}
    return result


@torch.inference_mode()
def extract_outputs(model, processor, prompt, records: Sequence[BenchmarkRecord], batch_size, device):
    fields = ("scores_A", "scores_B", "signed_evidence", "tie_probs", "relevance_probs")
    outputs: Dict[str, List[List[float]]] = {field: [] for field in fields}
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        pairs = [
            PairRecord(
                sample_id=r.sample_id,
                source_image=r.source_image,
                edit_instruction=r.edit_instruction,
                image_a=r.image_a,
                image_b=r.image_b,
            )
            for r in batch
        ]
        output = forward_pairs(model, processor, pairs, prompt, device)
        for field in fields:
            outputs[field].extend(getattr(output, field).float().cpu().tolist())
        done = start + len(batch)
        if done % 100 < batch_size or done == len(records):
            print(f"Scored {done}/{len(records)}", flush=True)
    arrays = {field: np.asarray(values, dtype=np.float64) for field, values in outputs.items()}
    for field, array in arrays.items():
        if array.shape != (len(records), NUM_METRICS) or not np.isfinite(array).all():
            raise ValueError(f"Invalid Student output {field}: shape {array.shape}")
    return arrays


def _subset(arrays: Dict[str, np.ndarray], index: np.ndarray) -> Dict[str, np.ndarray]:
    return {field: array[index] for field, array in arrays.items()}


def selected_equal_three_way(records, arrays, mask) -> Dict[str, Any]:
    """Three-way metrics over every pair (ties included) for ``selected_equal``."""
    tie = arrays["tie_probs"]
    cond_a = 1.0 / (1.0 + np.exp(-(arrays["scores_A"] - arrays["scores_B"])))
    per_metric = np.stack([(1.0 - tie) * cond_a, (1.0 - tie) * (1.0 - cond_a), tie], axis=-1)
    count = mask.sum(axis=1, keepdims=True)
    probabilities = (mask[:, :, None] * per_metric).sum(axis=1) / count
    score_margins = (mask * (arrays["scores_A"] - arrays["scores_B"])).sum(axis=1) / count[:, 0]
    report = _three_way_metrics(records, probabilities.tolist(), score_margins.tolist())
    report["definition"] = (
        "P(A), P(B), P(Tie) = mean over selected metrics of (1-t_k)w_k, (1-t_k)(1-w_k), t_k; "
        "absolute-margin baseline uses mean_{k in S} d_k"
    )
    return report, probabilities


def _majority_cells(runs: Sequence[Dict[str, Any]]) -> List[str | None]:
    """Per metric: None when not relevant, else the majority choice (Tie without one)."""
    cells: List[str | None] = []
    for metric in METRIC_NAMES:
        applicable = [run["metric_groundtruth"][metric] for run in runs]
        relevant = [cell for cell in applicable if int(cell["targeted_metric"]) == 1]
        if len(relevant) * 3 < 2 * len(applicable):
            cells.append(None)
            continue
        counts = {c: sum(cell["choice"] == c for cell in relevant) for c in ("A", "B", "Tie")}
        winner = [c for c, n in counts.items() if n * 2 > len(relevant)]
        cells.append(winner[0] if winner else "Tie")
    return cells


def load_gemma_validation(path: Path, sample_ids: Sequence[str]):
    wanted = set(sample_ids)
    records, cells = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sample_id"] not in wanted:
                continue
            records.append(
                BenchmarkRecord(
                    sample_id=row["sample_id"],
                    source_image=row["source_image"],
                    edit_instruction=row["edit_instruction"],
                    image_a=row["edited_image_A"],
                    image_b=row["edited_image_B"],
                    label=0,
                )
            )
            cells.append(_majority_cells(row["runs"]))
    if len(records) != len(wanted):
        raise ValueError(f"GemmaGT validation pairs missing from {path}.")
    return records, cells


def gemma_tie_report(arrays, cells) -> Dict[str, Any]:
    per_metric: Dict[str, Any] = {}
    for k, metric in enumerate(METRIC_NAMES):
        choices = np.asarray([row[k] if row[k] is not None else "none" for row in cells])
        relevant = choices != "none"
        is_tie = (choices == "Tie")[relevant]
        t = arrays["tie_probs"][relevant, k]
        d = (arrays["scores_A"] - arrays["scores_B"])[relevant, k]
        labels = np.where(is_tie, 0, 1)
        per_metric[metric] = {
            "relevant_cells": int(relevant.sum()),
            "tie_cells": int(is_tie.sum()),
            "tie_from_tie_head": tie_ranking_metrics(labels, t),
            "tie_from_absolute_margin": tie_ranking_metrics(labels, -np.abs(d)),
            "tie_positive_brier": float(((1.0 - t[is_tie]) ** 2).mean()) if is_tie.any() else None,
            "tie_negative_brier": float((t[~is_tie] ** 2).mean()) if (~is_tie).any() else None,
            "mean_tie_probability_tie_cells": float(t[is_tie].mean()) if is_tie.any() else None,
            "mean_tie_probability_other_cells": float(t[~is_tie].mean()) if (~is_tie).any() else None,
        }
    pooled_t, pooled_d, pooled_labels = [], [], []
    for k in range(NUM_METRICS):
        for i, row in enumerate(cells):
            if row[k] is not None:
                pooled_t.append(arrays["tie_probs"][i, k])
                pooled_d.append(arrays["scores_A"][i, k] - arrays["scores_B"][i, k])
                pooled_labels.append(0 if row[k] == "Tie" else 1)
    pooled_labels = np.asarray(pooled_labels)
    return {
        "cell_rule": "relevant if >= 2 of 3 runs; strict majority choice, else Tie",
        "pooled_over_metrics": {
            "tie_from_tie_head": tie_ranking_metrics(pooled_labels, np.asarray(pooled_t)),
            "tie_from_absolute_margin": tie_ranking_metrics(
                pooled_labels, -np.abs(np.asarray(pooled_d))
            ),
        },
        "per_metric": per_metric,
    }


def bootstrap_intervals(labels, margins, image_scores, replicates, seed) -> Dict[str, Any]:
    """Percentile intervals over pairs resampled with replacement."""
    rng = np.random.default_rng(seed)
    winner_is_a = labels == 1
    positive = np.where(winner_is_a, image_scores[:, 0], image_scores[:, 1])
    negative = np.where(winner_is_a, image_scores[:, 1], image_scores[:, 0])
    correct = np.sign(margins) == labels
    accuracy, auc = [], []
    for _ in range(replicates):
        index = rng.integers(0, len(labels), len(labels))
        accuracy.append(correct[index].mean())
        pos, neg = positive[index], negative[index]
        greater = (pos[:, None] > neg[None, :]).sum()
        equal = (pos[:, None] == neg[None, :]).sum()
        auc.append((greater + 0.5 * equal) / (pos.size * neg.size))

    def interval(values):
        low, high = np.percentile(values, [2.5, 97.5])
        return [round(100.0 * float(low), 2), round(100.0 * float(high), 2)]

    return {
        "replicates": replicates,
        "sampling_unit": "pair",
        "accuracy_percent_95": interval(accuracy),
        "pooled_auc_percent_95": interval(auc),
    }


def evaluate_aggregation(records, arrays, aggregation, args) -> Dict[str, Any]:
    weights, beta = aggregation["weights"], aggregation["beta"]
    margins = beta * (arrays["signed_evidence"] @ weights)
    image_scores = np.stack([arrays["scores_A"] @ weights, arrays["scores_B"] @ weights], axis=1)
    tie_confidence = arrays["tie_probs"] @ weights
    report = summarize(records, margins.tolist(), tie_confidence.tolist())
    report["probability"] = _probability_metrics(records, margins.tolist())
    report["probability"]["probability_definition"] = (
        "sigmoid(beta * sum_k a_k * signed_evidence_k)"
    )
    report["pooled_auc"] = _pooled_auc(records, image_scores.tolist())
    report["pooled_auc"]["definition"] = (
        "ROC AUC of sum_k a_k * score_k(image) from the Score Head, "
        "winner images positive vs loser images negative, pooled"
    )
    labels = np.asarray([r.label for r in records])
    report["bootstrap"] = bootstrap_intervals(
        labels, margins, image_scores, args.bootstrap_replicates, args.seed
    )
    report["aggregation"] = {"weights": weights.tolist(), "beta": beta}
    return report, margins, image_scores


def selection_mask(relevance: np.ndarray) -> np.ndarray:
    mask = relevance >= RELEVANCE_THRESHOLD
    empty = ~mask.any(axis=1)
    mask[empty, relevance[empty].argmax(axis=1)] = True
    return mask


def evaluate_selected_aggregation(records, arrays, weights, args) -> Dict[str, Any]:
    """``weights`` is [N, 6] and zero outside each pair's selected metrics.

    The margin is the weighted mean of signed_evidence over the selected
    metrics (AUC-Margin.md section 2); a sum gives the same accuracy but a
    different margin AUC.
    """
    labels = np.asarray([r.label for r in records])
    normalizer = weights.sum(axis=1)
    margins = (weights * arrays["signed_evidence"]).sum(axis=1) / normalizer
    image_scores = np.stack(
        [(weights * arrays["scores_A"]).sum(axis=1) / normalizer,
         (weights * arrays["scores_B"]).sum(axis=1) / normalizer],
        axis=1,
    )
    correct = int((np.sign(margins) == labels).sum())
    report: Dict[str, Any] = {
        "accuracy": {
            "scope": "non_tie_samples",
            "correct": correct,
            "total": int(labels.size),
            "value": correct / labels.size,
            "percent": round(100.0 * correct / labels.size, 2),
        },
        "pooled_auc": _pooled_auc(records, image_scores.tolist()),
    }
    report["pooled_auc"]["definition"] = (
        "ROC AUC of sum_k w_k * score_k(image) / sum_k w_k over the selected metrics, "
        "winner images positive vs loser images negative, pooled"
    )
    report["bootstrap"] = bootstrap_intervals(
        labels, margins, image_scores, args.bootstrap_replicates, args.seed
    )
    return report, margins, image_scores


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    aggregator_path = (
        Path(args.preference_aggregator).expanduser().resolve()
        if args.preference_aggregator else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    file_name = "bench_evaluation.json" if args.datasets == "bench" else "evaluation.json"
    output_path = output_dir / (
        file_name if args.role == "student" else f"{args.role}_{file_name}"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    aggregations = read_aggregations(aggregator_path) if aggregator_path else {}
    datasets = load_datasets(args)
    print({name: len(records) for name, records in datasets.items()}, flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir else None
    if adapter_dir is not None:
        if "adapter_state_dict" not in checkpoint:
            raise ValueError("--adapter-dir needs a Stage1 warm-start checkpoint.")
        state = load_safetensors(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
        if set(state) != set(checkpoint["adapter_state_dict"]):
            raise ValueError(f"Adapter keys of {adapter_dir} differ from {checkpoint_path}")
        checkpoint["adapter_state_dict"] = state
        checkpoint["ms_swift_best_checkpoint"] = str(adapter_dir)
    prompt = checkpoint.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Student checkpoint is missing the prompt contract.")
    model, processor = load_refined_model(
        checkpoint, args.role, device, args.min_pixels, args.max_pixels,
        base_model_override=Path(args.base_model_path).expanduser().resolve(),
    )

    metrics: Dict[str, Any] = {}
    predictions: Dict[str, Any] = {}
    for dataset_name, all_records in datasets.items():
        print(f"Evaluating {dataset_name} ({len(all_records)} pairs)", flush=True)
        all_arrays = extract_outputs(model, processor, prompt, all_records, args.batch_size, device)
        metrics[dataset_name] = {}
        all_mask = selection_mask(all_arrays["relevance_probs"])
        if any(r.label == 0 for r in all_records):
            three_way, three_way_probs = selected_equal_three_way(all_records, all_arrays, all_mask)
            metrics[dataset_name]["selected_equal_three_way"] = three_way
            print(
                f"[selected_equal three-way] {dataset_name}: "
                f"three_class={three_way['three_class_accuracy_percent']}%, "
                f"tie_auc(t)={three_way['tie_from_tie_head'].get('roc_auc_percent')}%, "
                f"tie_auc(|d|)={three_way['tie_from_absolute_margin'].get('roc_auc_percent')}%",
                flush=True,
            )
        # Pairwise aggregations below use the non-tie pairs only.
        keep = np.asarray([r.label != 0 for r in all_records])
        records = [r for r, kept in zip(all_records, keep) if kept]
        arrays = _subset(all_arrays, keep)
        rows = [
            {"sample_id": r.sample_id, "label": r.label,
             # Kept per sample so wrongly judged pairs can be visualized later.
             "source_image": r.source_image, "edit_instruction": r.edit_instruction,
             "image_a": r.image_a, "image_b": r.image_b,
             **{field: arrays[field][i].tolist() for field in arrays},
             "per_metric_margin": (arrays["scores_A"][i] - arrays["scores_B"][i]).tolist()}
            for i, r in enumerate(records)
        ]
        for name in (AGGREGATIONS if aggregations else ()):
            report, margins, image_scores = evaluate_aggregation(
                records, arrays, aggregations[name], args
            )
            metrics[dataset_name][name] = report
            for row, margin, scores in zip(rows, margins, image_scores):
                row[f"{name}_margin"] = float(margin)
                row[f"{name}_image_scores"] = scores.tolist()
            print(
                f"[{name}] {dataset_name}: accuracy={report['accuracy']['percent']}% "
                f"({report['accuracy']['correct']}/{report['accuracy']['total']}), "
                f"auc={report['pooled_auc']['percent']}%, "
                f"brier={report['probability']['brier_score']:.4f}",
                flush=True,
            )
        mask = selection_mask(arrays["relevance_probs"])
        selected_weights = {"selected_equal": mask.astype(np.float64)}
        if aggregations:
            selected_weights["selected_fitted"] = mask * aggregations["fitted"]["weights"][None, :]
        for row, selected in zip(rows, mask):
            row["selected_metrics"] = [METRIC_NAMES[k] for k in np.flatnonzero(selected)]
        for name in selected_weights:
            report, margins, image_scores = evaluate_selected_aggregation(
                records, arrays, selected_weights[name], args
            )
            report["margin_auc"] = _margin_auc(records, margins.tolist())
            report["selection"] = {
                "rule": f"relevance_probs >= {RELEVANCE_THRESHOLD}, else the most relevant metric",
                "selected_metric_fraction": mask.mean(axis=0).round(4).tolist(),
                "mean_selected_metrics": round(float(mask.sum(axis=1).mean()), 4),
            }
            metrics[dataset_name][name] = report
            for row, margin, scores in zip(rows, margins, image_scores):
                row[f"{name}_margin"] = float(margin)
                row[f"{name}_image_scores"] = scores.tolist()
            print(
                f"[{name}] {dataset_name}: accuracy={report['accuracy']['percent']}% "
                f"({report['accuracy']['correct']}/{report['accuracy']['total']}), "
                f"margin_auc={report['margin_auc']['percent']}%",
                flush=True,
            )
        tie_rows = []
        for i, r in enumerate(all_records):
            if r.label != 0:
                continue
            tie_rows.append(
                {"sample_id": r.sample_id, "label": 0,
                 **{field: all_arrays[field][i].tolist() for field in all_arrays},
                 "selected_metrics": [METRIC_NAMES[k] for k in np.flatnonzero(all_mask[i])],
                 "selected_equal_three_way_probabilities": three_way_probs[i].tolist()}
            )
        predictions[dataset_name] = rows + tie_rows

    gemma_validation = None
    data = checkpoint.get("data") or {}
    validation_ids = data.get("validation_sample_ids")
    gemma_path = args.gemma_dataset or data.get("dataset_path")
    if not args.skip_gemma_validation and validation_ids and gemma_path:
        gemma_records, gemma_cells = load_gemma_validation(Path(gemma_path), validation_ids)
        print(f"Evaluating GemmaGT validation ({len(gemma_records)} pairs)", flush=True)
        gemma_arrays = extract_outputs(
            model, processor, prompt, gemma_records, args.batch_size, device
        )
        gemma_validation = {"dataset": str(gemma_path), "pairs": len(gemma_records),
                            **gemma_tie_report(gemma_arrays, gemma_cells)}
        pooled = gemma_validation["pooled_over_metrics"]
        print(
            f"[GemmaGT validation] tie_auc(t)={pooled['tie_from_tie_head'].get('roc_auc_percent')}%, "
            f"tie_auc(|d|)={pooled['tie_from_absolute_margin'].get('roc_auc_percent')}%",
            flush=True,
        )

    result = {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": args.role,
        "checkpoint": str(checkpoint_path),
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "ms_swift_best_checkpoint": checkpoint.get("ms_swift_best_checkpoint"),
        "selection": checkpoint.get("selection"),
        "preference_aggregator": str(aggregator_path) if aggregator_path else None,
        "tie_head_mode": checkpoint.get("tie_head_mode", "independent"),
        "gemma_validation_tie": gemma_validation,
        "metric_order": list(METRIC_NAMES),
        "dropout_enabled": False,
        "configuration": vars(args),
        "metrics": metrics,
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Evaluation output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
