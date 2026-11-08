#!/usr/bin/env python3
"""Recompute the overall preference from saved Student/Teacher evaluation files.

The overall is computed only over the metrics the Relevance Head selects for
each pair: S = {k : relevance_probs_k >= 0.5}, or the most relevant metric when
none passes, exactly as the model's inference wrapper. Both images of a pair
share S because the Relevance Head sees only the source input.

Aggregations (w_k = 0 outside S):

- ``all_metrics_fitted``: all six metrics, fitted a_k (the current evaluation),
- ``selected_equal``: selected metrics, w_k = 1 (primary result),
- ``selected_fitted``: selected metrics, fitted a_k.

Accuracy uses the sign of sum_k w_k * signed_evidence_k. Pooled AUC uses the
image score sum_k w_k * score_k / sum_k w_k. Each metric's score level is
arbitrary (the losses only use A - B), so when S differs between pairs the
pooled AUC also reflects which metrics a pair selected. ``centered`` AUC
therefore subtracts each metric's mean over all images of the dataset first.

Only reads the saved JSON files; no GPU is needed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

RELEVANCE_THRESHOLD = 0.5
AGGREGATIONS = ("all_metrics_fitted", "selected_equal", "selected_fitted")
DATASETS = ("EditReward-Data-Test", "EditReward-Bench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", action="append", required=True, metavar="NAME=PATH",
        help="Model name and evaluation file, e.g. Student-Stage2=.../evaluation.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def pooled_auc(labels: np.ndarray, image_scores: np.ndarray) -> float:
    """Winner image scores positive vs loser image scores negative, ties count 0.5."""
    winner_is_a = labels == 1
    positive = np.where(winner_is_a, image_scores[:, 0], image_scores[:, 1])
    negative = np.where(winner_is_a, image_scores[:, 1], image_scores[:, 0])
    greater = (positive[:, None] > negative[None, :]).sum()
    equal = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * equal) / (positive.size * negative.size))


def selection_mask(relevance: np.ndarray) -> np.ndarray:
    mask = relevance >= RELEVANCE_THRESHOLD
    empty = ~mask.any(axis=1)
    mask[empty, relevance[empty].argmax(axis=1)] = True
    return mask


def evaluate_dataset(rows, fitted_weights: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray([r["label"] for r in rows])
    if np.any(labels == 0):
        raise ValueError("Evaluation must contain non-tie pairs only.")
    evidence = np.asarray([r["signed_evidence"] for r in rows], dtype=np.float64)
    scores_a = np.asarray([r["scores_A"] for r in rows], dtype=np.float64)
    scores_b = np.asarray([r["scores_B"] for r in rows], dtype=np.float64)
    mask = selection_mask(np.asarray([r["relevance_probs"] for r in rows], dtype=np.float64))
    metric_mean = np.concatenate([scores_a, scores_b]).mean(axis=0)

    per_pair_weights = {
        "all_metrics_fitted": np.broadcast_to(fitted_weights, evidence.shape),
        "selected_equal": mask.astype(np.float64),
        "selected_fitted": mask * fitted_weights[None, :],
    }
    result: Dict[str, Any] = {
        "pairs": int(labels.size),
        "selected_metric_fraction": mask.mean(axis=0).round(4).tolist(),
        "mean_selected_metrics": round(float(mask.sum(axis=1).mean()), 4),
    }
    for name in AGGREGATIONS:
        w = per_pair_weights[name]
        margins = (w * evidence).sum(axis=1)
        normalizer = w.sum(axis=1)

        def image_scores(center: np.ndarray) -> np.ndarray:
            return np.stack(
                [(w * (scores_a - center)).sum(axis=1) / normalizer,
                 (w * (scores_b - center)).sum(axis=1) / normalizer],
                axis=1,
            )

        correct = int((np.sign(margins) == labels).sum())
        result[name] = {
            "accuracy_percent": round(100.0 * correct / labels.size, 2),
            "correct": correct,
            "pooled_auc_percent": round(100.0 * pooled_auc(labels, image_scores(0.0)), 2),
            "centered_pooled_auc_percent": round(
                100.0 * pooled_auc(labels, image_scores(metric_mean)), 2
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    models: Dict[str, Any] = {}
    for item in args.evaluation:
        name, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"--evaluation must be NAME=PATH, got {item}")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        models[name] = {"evaluation": str(Path(path).resolve()), "datasets": {}}
        for dataset in DATASETS:
            fitted = np.asarray(
                payload["metrics"][dataset]["fitted"]["aggregation"]["weights"], dtype=np.float64
            )
            models[name]["datasets"][dataset] = evaluate_dataset(
                payload["predictions"][dataset], fitted
            )
        metric_order = payload["metric_order"]

    for name in AGGREGATIONS:
        print(f"\n[{name}]  Test(1k)-Acc  Test(1k)-AUC  Bench(115)-Acc  Bench(115)-AUC"
              "   (centered AUC: Test / Bench)")
        for model, entry in models.items():
            test = entry["datasets"]["EditReward-Data-Test"][name]
            bench = entry["datasets"]["EditReward-Bench"][name]
            print(f"{model:16s} {test['accuracy_percent']:12.2f} {test['pooled_auc_percent']:13.2f} "
                  f"{bench['accuracy_percent']:15.2f} {bench['pooled_auc_percent']:15.2f}"
                  f"   ({test['centered_pooled_auc_percent']:.2f} / "
                  f"{bench['centered_pooled_auc_percent']:.2f})")

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_order": metric_order,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "definitions": {
            "selection": "relevance_probs >= 0.5, else the most relevant metric",
            "margin": "sum_k w_k * signed_evidence_k",
            "image_score": "sum_k w_k * score_k / sum_k w_k",
            "centered_image_score": "same, with score_k minus its mean over all images of the dataset",
        },
        "models": models,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
