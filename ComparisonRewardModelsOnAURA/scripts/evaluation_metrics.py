"""Pairwise metrics for comparing reward models on AURA-Test (one column = one bench metric).

Each pair has a human label A / B / Tie (A = edited_images[0] better), the human votes, the
observed rater agreement, the model scores of both images and, only for models that can predict
a tie (Our Model), the three-way probabilities P(A), P(B), P(Tie).

Definitions (EvaluationConfig/metric_mapping.json -> evaluation_metrics):
  margin                     score_A - score_B
  Accuracy                   non-tie pairs, sign(margin) agrees with the label; margin == 0 is wrong
  Margin AUC                 non-tie pairs, ROC AUC of label*margin against -label*margin (symmetric
                             within-pair AUC, same as RewardModelEvaluation/OurModel/recompute_margin_auc.py)
  Three-class Accuracy       all pairs, predicted A / B / Tie equals the label. Models with tie
                             probabilities (Our Model): argmax of (P(A), P(B), P(Tie)). Score-only models:
                             their native behaviour, Tie only when score_A == score_B exactly (no threshold
                             is tuned on the bench; user decision 2026-11-11)
  PLCC / SRCC                all pairs, Pearson / Spearman correlation of margin with the human vote
                             margin (votes A - votes B) / number of votes
  Tie AUC                    all pairs, ROC AUC of -|margin| with Tie as the positive class; for models
                             with tie probabilities also from P(Tie)
  Full-agreement Accuracy    Accuracy on non-tie pairs with observed_agreement == 1
Confidence intervals are 95% percentile bootstrap intervals over pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

LABELS = ("A", "B", "Tie")
LABEL_SIGN = {"A": 1, "B": -1, "Tie": 0}
PROBABILITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PairScore:
    """Model scores of one bench pair together with its human label."""

    sample_id: str
    metric: str
    human_label: str
    votes: Mapping[str, int]
    observed_agreement: Optional[float]
    score_a: float
    score_b: float
    probabilities: Optional[Sequence[float]] = None  # P(A), P(B), P(Tie)

    def __post_init__(self):
        if self.human_label not in LABELS:
            raise ValueError(f"{self.sample_id}: human_label must be one of {LABELS}, got {self.human_label!r}")
        for name in ("score_a", "score_b"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{self.sample_id}: {name} must be a finite number, got {value!r}")
        if set(self.votes) - set(LABELS) or any(int(self.votes.get(k, 0)) < 0 for k in LABELS):
            raise ValueError(f"{self.sample_id}: votes must have non-negative counts for {LABELS}, got {dict(self.votes)}")
        if sum(int(self.votes.get(k, 0)) for k in LABELS) == 0:
            raise ValueError(f"{self.sample_id}: votes are empty")
        if self.probabilities is not None:
            p = np.asarray(self.probabilities, dtype=np.float64)
            if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(p < 0) \
                    or abs(p.sum() - 1.0) > PROBABILITY_TOLERANCE:
                raise ValueError(f"{self.sample_id}: probabilities must be 3 finite non-negative values summing to 1")

    @property
    def margin(self) -> float:
        return float(self.score_a) - float(self.score_b)

    @property
    def vote_margin(self) -> float:
        total = sum(int(self.votes.get(k, 0)) for k in LABELS)
        return (int(self.votes.get("A", 0)) - int(self.votes.get("B", 0))) / total


# ---------------------------------------------------------------- basic statistics

def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    return (last_rank - (counts - 1) / 2.0)[inverse]


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(positive > negative) + 0.5 P(equal)."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("ROC AUC needs at least one positive and one negative value")
    ranks = average_ranks(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _signs_and_margins(labels: Sequence[str], margins: Sequence[float]):
    signs = np.asarray([LABEL_SIGN[label] for label in labels], dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    if signs.shape != margins.shape:
        raise ValueError(f"labels and margins differ in length: {signs.size} vs {margins.size}")
    if not np.all(np.isfinite(margins)):
        raise ValueError("margins must be finite")
    return signs, margins


def accuracy(labels: Sequence[str], margins: Sequence[float]) -> float:
    """Non-tie pairs only; a zero margin counts as wrong."""
    signs, margins = _signs_and_margins(labels, margins)
    keep = signs != 0
    if not keep.any():
        raise ValueError("Accuracy needs at least one non-tie pair")
    return float((np.sign(margins[keep]) == signs[keep]).mean())


def margin_auc(labels: Sequence[str], margins: Sequence[float]) -> float:
    """Symmetric within-pair margin AUC on non-tie pairs."""
    signs, margins = _signs_and_margins(labels, margins)
    keep = signs != 0
    if not keep.any():
        raise ValueError("Margin AUC needs at least one non-tie pair")
    winner_minus_loser = signs[keep] * margins[keep]
    return roc_auc(winner_minus_loser, -winner_minus_loser)


def tie_auc(labels: Sequence[str], tie_confidence: Sequence[float]) -> float:
    """ROC AUC of a tie confidence (higher = more likely Tie), Tie pairs as positives."""
    signs, confidence = _signs_and_margins(labels, tie_confidence)
    return roc_auc(confidence[signs == 0], confidence[signs != 0])


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN when either side is constant (correlation undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"inputs differ in length: {x.size} vs {y.size}")
    if x.size < 2:
        raise ValueError("correlation needs at least two pairs")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("correlation inputs must be finite")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation = Pearson correlation of average ranks."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"inputs differ in length: {x.size} vs {y.size}")
    if x.size < 2:
        raise ValueError("correlation needs at least two pairs")
    return pearson(average_ranks(x), average_ranks(y))


def predicted_labels_from_probabilities(probabilities: Sequence[Sequence[float]]) -> list:
    """argmax over (P(A), P(B), P(Tie)); an exact tie between columns takes the first column."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (pairs, 3)")
    return [LABELS[i] for i in probabilities.argmax(axis=1)]


def predicted_labels_from_margins(margins: Sequence[float]) -> list:
    """Native prediction of a score-only model: A if margin > 0, B if margin < 0, Tie only if exactly 0."""
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 1:
        raise ValueError("margins must be one-dimensional")
    if not np.all(np.isfinite(margins)):
        raise ValueError("margins must be finite")
    return ["A" if m > 0 else "B" if m < 0 else "Tie" for m in margins]


def three_class_report(labels: Sequence[str], predicted: Sequence[str]) -> Dict[str, object]:
    """Accuracy over all pairs plus recall and precision per class (A, B, Tie)."""
    labels = list(labels)
    predicted = list(predicted)
    if len(labels) != len(predicted):
        raise ValueError(f"labels and predictions differ in length: {len(labels)} vs {len(predicted)}")
    if not labels:
        raise ValueError("three-class accuracy needs at least one pair")
    if set(predicted) - set(LABELS):
        raise ValueError(f"predictions must be in {LABELS}")
    truth, guess = np.asarray(labels), np.asarray(predicted)
    per_class = {}
    for name in LABELS:
        true_mask, guess_mask = truth == name, guess == name
        hits = int((true_mask & guess_mask).sum())
        per_class[name] = {
            "true": int(true_mask.sum()),
            "predicted": int(guess_mask.sum()),
            "recall": hits / int(true_mask.sum()) if true_mask.any() else None,
            "precision": hits / int(guess_mask.sum()) if guess_mask.any() else None,
        }
    return {"accuracy": float((truth == guess).mean()), "pairs": len(labels), "per_class": per_class}


# ---------------------------------------------------------------- bootstrap

def bootstrap_interval(statistic: Callable[[np.ndarray], float], num_items: int, resamples: int = 2000,
                       seed: int = 0, confidence: float = 0.95) -> Dict[str, object]:
    """Percentile bootstrap interval of statistic(indices) over items drawn with replacement.

    Resamples where the statistic is undefined (ValueError or NaN, e.g. no tie pair drawn) are
    skipped and counted.
    """
    if num_items < 1:
        raise ValueError("bootstrap needs at least one item")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be >= 1 and confidence in (0, 1)")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indices = rng.integers(0, num_items, size=num_items)
        try:
            value = statistic(indices)
        except ValueError:
            continue
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"low": None, "high": None, "valid_resamples": 0, "resamples": resamples}
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return {"low": float(low), "high": float(high), "valid_resamples": len(values), "resamples": resamples}


def paired_bootstrap_difference(statistic_ours: Callable[[np.ndarray], float],
                                statistic_other: Callable[[np.ndarray], float], num_items: int,
                                resamples: int = 2000, seed: int = 0) -> Dict[str, object]:
    """Bootstrap of (ours - other) on the same resampled pairs; p_not_better = share of draws <= 0."""
    if num_items < 1:
        raise ValueError("bootstrap needs at least one item")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        indices = rng.integers(0, num_items, size=num_items)
        try:
            difference = statistic_ours(indices) - statistic_other(indices)
        except ValueError:
            continue
        if math.isfinite(difference):
            differences.append(difference)
    if not differences:
        return {"mean": None, "low": None, "high": None, "p_not_better": None, "valid_resamples": 0}
    differences = np.asarray(differences)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"mean": float(differences.mean()), "low": float(low), "high": float(high),
            "p_not_better": float((differences <= 0).mean()), "valid_resamples": int(differences.size)}


# ---------------------------------------------------------------- one table cell

def _safe(function, *args):
    try:
        value = function(*args)
    except ValueError:
        return None
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def summarize_column(pairs: Sequence[PairScore], metric: str, resamples: int = 2000,
                     seed: int = 0) -> Dict[str, object]:
    """All agreed metrics for one model on one bench metric (only pairs labelled for that metric)."""
    selected = [p for p in pairs if p.metric == metric]
    if not selected:
        raise ValueError(f"no pairs for metric {metric!r}")
    if len({p.sample_id for p in selected}) != len(selected):
        raise ValueError(f"duplicate sample_id for metric {metric!r}")
    with_probabilities = [p.probabilities is not None for p in selected]
    if any(with_probabilities) and not all(with_probabilities):
        raise ValueError(f"metric {metric!r}: tie probabilities given for some pairs but not all")
    has_tie_head = all(with_probabilities)

    labels = np.asarray([p.human_label for p in selected])
    margins = np.asarray([p.margin for p in selected])
    vote_margins = np.asarray([p.vote_margin for p in selected])
    full_agreement = np.asarray([p.observed_agreement == 1 for p in selected])
    probabilities = np.asarray([p.probabilities for p in selected], dtype=np.float64) if has_tie_head else None

    def stat_accuracy(idx):
        return accuracy(labels[idx], margins[idx])

    def stat_margin_auc(idx):
        return margin_auc(labels[idx], margins[idx])

    def stat_plcc(idx):
        return pearson(margins[idx], vote_margins[idx])

    def stat_srcc(idx):
        return spearman(margins[idx], vote_margins[idx])

    def stat_tie_auc(idx):
        return tie_auc(labels[idx], -np.abs(margins[idx]))

    statistics = {"accuracy": stat_accuracy, "margin_auc": stat_margin_auc, "plcc": stat_plcc,
                  "srcc": stat_srcc, "tie_auc_from_margin": stat_tie_auc}
    if has_tie_head:
        predicted = np.asarray(predicted_labels_from_probabilities(probabilities))
        three_class_rule = "argmax of P(A), P(B), P(Tie)"
        statistics["tie_auc_from_tie_probability"] = lambda idx: tie_auc(labels[idx], probabilities[idx, 2])
    else:
        predicted = np.asarray(predicted_labels_from_margins(margins))
        three_class_rule = "Tie only if score_A == score_B"
    statistics["three_class_accuracy"] = lambda idx: float((predicted[idx] == labels[idx]).mean())

    everything = np.arange(len(selected))
    report: Dict[str, object] = {
        "metric": metric,
        "pairs": len(selected),
        "non_tie_pairs": int((labels != "Tie").sum()),
        "tie_pairs": int((labels == "Tie").sum()),
        "has_tie_prediction": has_tie_head,
        "full_agreement_non_tie_pairs": int((full_agreement & (labels != "Tie")).sum()),
        "full_agreement_accuracy": _safe(accuracy, labels[full_agreement], margins[full_agreement]),
        "three_class_rule": three_class_rule,
        "three_class": three_class_report(labels, predicted),
    }
    for name, function in statistics.items():
        report[name] = _safe(function, everything)
        report[f"{name}_interval"] = bootstrap_interval(function, len(selected), resamples, seed)
    if not has_tie_head:
        report["tie_auc_from_tie_probability"] = None
    return report
