"""Shared pairwise metrics for baseline reward-model evaluation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def parse_ranking(value: str) -> int:
    """Map benchmark ranking to 1 (A wins), -1 (B wins), or 0 (tie)."""
    ranking = str(value).strip().upper().replace(" ", "")
    if ranking == "A>B":
        return 1
    if ranking in {"B>A", "A<B"}:
        return -1
    if ranking in {"A=B", "TIE"}:
        return 0
    raise ValueError(f"Unsupported ranking: {value!r}")


def evaluate_pairwise(
    labels: Iterable[int], scores_a: Iterable[float], scores_b: Iterable[float]
) -> dict[str, Any]:
    """Compute non-tie accuracy and threshold-free tie retrieval AP/ROC-AUC."""
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    scores_a_arr = np.asarray(list(scores_a), dtype=np.float64)
    scores_b_arr = np.asarray(list(scores_b), dtype=np.float64)
    if not (labels_arr.shape == scores_a_arr.shape == scores_b_arr.shape):
        raise ValueError("labels, scores_a, and scores_b must have identical shapes")
    if labels_arr.ndim != 1 or labels_arr.size == 0:
        raise ValueError("Evaluation inputs must be non-empty one-dimensional arrays")
    if not np.isfinite(scores_a_arr).all() or not np.isfinite(scores_b_arr).all():
        raise ValueError("Model scores must all be finite")

    margins = scores_a_arr - scores_b_arr
    non_tie = labels_arr != 0
    correct = ((margins > 0) & (labels_arr == 1)) | (
        (margins < 0) & (labels_arr == -1)
    )
    non_tie_total = int(non_tie.sum())
    non_tie_correct = int((correct & non_tie).sum())

    tie_targets = labels_arr == 0
    tie_total = int(tie_targets.sum())
    tie_confidence = -np.abs(margins)
    summary: dict[str, Any] = {
        "total_samples": int(labels_arr.size),
        "non_tie_samples": non_tie_total,
        "tie_samples": tie_total,
        "non_tie_accuracy": {
            "correct": non_tie_correct,
            "total": non_tie_total,
            "value": non_tie_correct / non_tie_total if non_tie_total else None,
            "percent": round(100.0 * non_tie_correct / non_tie_total, 2) if non_tie_total else None,
        },
    }
    if tie_total == 0 or non_tie_total == 0:
        # Tie retrieval AP/ROC-AUC need both classes; e.g. EditReward-Data-Test has no ties.
        summary["tie_ranking"] = {
            "evaluable": False,
            "reason": "AP/ROC-AUC require both tie and non-tie samples",
        }
        return summary

    # Group equal confidence values, making AP independent of their input order.
    order = np.argsort(-tie_confidence, kind="mergesort")
    sorted_confidence = tie_confidence[order]
    sorted_targets = tie_targets[order]
    cumulative_tp = 0
    average_precision = 0.0
    start = 0
    while start < len(sorted_confidence):
        end = start + 1
        while (
            end < len(sorted_confidence)
            and sorted_confidence[end] == sorted_confidence[start]
        ):
            end += 1
        positives = int(sorted_targets[start:end].sum())
        cumulative_tp += positives
        if positives:
            average_precision += (positives / tie_total) * (cumulative_tp / end)
        start = end

    tie_scores = tie_confidence[tie_targets]
    non_tie_scores = tie_confidence[non_tie]
    comparisons = tie_scores[:, None] - non_tie_scores[None, :]
    roc_auc = float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
        / comparisons.size
    )

    tie_abs = np.abs(margins[tie_targets])
    non_tie_abs = np.abs(margins[non_tie])
    summary["tie_ranking"] = {
        "evaluable": True,
        "positive_class": "tie",
        "confidence": "-|score_A-score_B|",
        "average_precision": float(average_precision),
        "average_precision_percent": round(100.0 * average_precision, 2),
        "random_ap_baseline": tie_total / labels_arr.size,
        "roc_auc": roc_auc,
        "roc_auc_percent": round(100.0 * roc_auc, 2),
        "mean_abs_margin_tie": float(tie_abs.mean()),
        "mean_abs_margin_non_tie": float(non_tie_abs.mean()),
        "median_abs_margin_tie": float(np.median(tie_abs)),
        "median_abs_margin_non_tie": float(np.median(non_tie_abs)),
    }
    return summary
