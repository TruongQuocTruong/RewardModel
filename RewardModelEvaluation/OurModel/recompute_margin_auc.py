#!/usr/bin/env python3
"""Recompute accuracy and the within-pair margin AUC from saved evaluation files.

Labels only say which edited image of a pair is better, so every metric here
uses only within-pair quantities:

- ``accuracy``: sign of margin = score(A) - score(B) against the label.
- ``margin_auc`` (primary AUC): with d_i = label_i * margin_i, i.e. score of the
  winning image minus score of the losing image, every pair is used twice,
  (d_i, positive) and (-d_i, negative), and the ROC AUC of these 2n values is
  reported. This equals P(d_i + d_j > 0) over two random pairs (ties count 0.5).
  It does not depend on the A/B order, and an offset shared by both images of a
  pair cancels out.

``pooled_auc`` (winner image scores against loser image scores of all pairs,
the AUC reported before 2026-10-21) is kept only for reference: almost all of
its comparisons cross different source images and edit instructions, where no
label exists.

Margins:

- Inspector: ``margin_A_minus_B`` (logit(yes) - logit(no) of A minus that of B).
- Student / Teacher, ``selected_equal`` (primary): metrics with relevance >= 0.5,
  or the most relevant metric when none passes, equal mean of ``signed_evidence``.
- Student / Teacher, ``all_metrics_fitted``: all six metrics, fitted a_k and beta
  from the evaluation file (the aggregation used before 2026-10-20).

Confidence intervals use 1000 bootstrap resamples of pairs. Paired comparisons
resample the same pairs for both models; accuracy also gets an exact McNemar test.
Only reads JSON files; no GPU is needed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

RESULTS = Path(__file__).resolve().parent / "evaluation_results"
INSPECTOR_DIR = RESULTS / "inspector"
STUDENT_DIR = RESULTS / "student"
TRAIN_LOOP_DIR = RESULTS / "train_loop"
DATASETS = ("EditReward-Data-Test", "EditReward-Bench")
RELEVANCE_THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 1000
SEED = 42

# name -> evaluation file; one block per results table.
GROUPS: Dict[str, Dict[str, Path]] = {
    "inspector": {
        "Inspector base model (zero-shot)": INSPECTOR_DIR / "overall_inspector/base_zero_shot.json",
        "Inspector old baseline r128 LLM+ViT+head 5e-6 4ep": INSPECTOR_DIR / "overall_inspector/checkpoint-200.json",
        "Inspector v1 r32 LLM+ViT 1e-5": INSPECTOR_DIR / "overall_inspector_r32/checkpoint-200.json",
        "Inspector v2 r16 LLM+ViT 1e-5": INSPECTOR_DIR / "overall_inspector_r16/checkpoint-200.json",
        "Inspector v3 r16 LLM 1e-5": INSPECTOR_DIR / "overall_inspector_r16_llm/checkpoint-200.json",
        "Inspector v4 r32 LLM 2e-5": INSPECTOR_DIR / "overall_inspector_r32_llm_lr2e-5/checkpoint-200.json",
        "Inspector v5 r32 LLM+ViT 5e-6": INSPECTOR_DIR / "overall_inspector_r32_lr5e-6/checkpoint-250.json",
        "Inspector v6 r16 LLM 2e-5": INSPECTOR_DIR / "overall_inspector_r16_llm_lr2e-5/checkpoint-200.json",
        "Inspector v7 r32 LLM 4e-5": INSPECTOR_DIR / "overall_inspector_r32_llm_lr4e-5/checkpoint-200.json",
        "Inspector v8 r64 LLM 2e-5": INSPECTOR_DIR / "overall_inspector_r64_llm_lr2e-5/checkpoint-200.json",
        "Inspector v9 r32 LLM 2e-5 1ep": INSPECTOR_DIR / "overall_inspector_r32_llm_lr2e-5_epoch1/checkpoint-200.json",
        "Inspector v10 r32 LLM 8e-5": INSPECTOR_DIR / "overall_inspector_r32_llm_lr8e-5/checkpoint-200.json",
        "Inspector v11 r32 LLM 5e-6": INSPECTOR_DIR / "overall_inspector_r32_llm_lr5e-6/checkpoint-250.json",
    },
    "student": {
        "Student r32 2e-5 2ep": STUDENT_DIR / "student_hardlabel_r32_lr2e-5/evaluation.json",
        "Student r32 4e-5 2ep": STUDENT_DIR / "student_hardlabel_r32_lr4e-5/evaluation.json",
        "Student r32 8e-5 2ep": STUDENT_DIR / "student_hardlabel_r32_lr8e-5/evaluation.json",
        "Student r64 2e-5 2ep": STUDENT_DIR / "student_hardlabel_r64_lr2e-5/evaluation.json",
        "Student r64 4e-5 2ep": STUDENT_DIR / "student_hardlabel_r64_lr4e-5/evaluation.json",
        "Student r64 8e-5 2ep": STUDENT_DIR / "student_hardlabel_r64_lr8e-5/evaluation.json",
    },
    "pipeline": {
        "Inspector v10": INSPECTOR_DIR / "overall_inspector_r32_llm_lr8e-5/checkpoint-200.json",
        "Student-Stage1": STUDENT_DIR / "student_hardlabel_r64_lr4e-5/evaluation.json",
        "Teacher-Stage2": TRAIN_LOOP_DIR / "train_loop_single_pass/teacher_evaluation.json",
        "Student-Stage2": TRAIN_LOOP_DIR / "train_loop_single_pass/evaluation.json",
    },
}

# Paired comparisons (left, right) per group.
COMPARISONS: Dict[str, List[Tuple[str, str]]] = {
    "inspector": [
        ("Inspector v10 r32 LLM 8e-5", "Inspector v7 r32 LLM 4e-5"),
        ("Inspector v10 r32 LLM 8e-5", "Inspector v4 r32 LLM 2e-5"),
        ("Inspector v7 r32 LLM 4e-5", "Inspector v8 r64 LLM 2e-5"),
        ("Inspector v10 r32 LLM 8e-5", "Inspector old baseline r128 LLM+ViT+head 5e-6 4ep"),
    ],
    "student": [
        ("Student r32 2e-5 2ep", "Student r64 2e-5 2ep"),
        ("Student r32 4e-5 2ep", "Student r64 4e-5 2ep"),
        ("Student r32 8e-5 2ep", "Student r64 8e-5 2ep"),
        ("Student r32 4e-5 2ep", "Student r32 2e-5 2ep"),
        ("Student r32 8e-5 2ep", "Student r32 4e-5 2ep"),
        ("Student r64 4e-5 2ep", "Student r64 2e-5 2ep"),
        ("Student r64 8e-5 2ep", "Student r64 4e-5 2ep"),
        ("Student r32 8e-5 2ep", "Student r64 4e-5 2ep"),
    ],
    "pipeline": [
        ("Student-Stage2", "Student-Stage1"),
        ("Student-Stage2", "Inspector v10"),
        ("Teacher-Stage2", "Student-Stage1"),
        ("Teacher-Stage2", "Student-Stage2"),
        ("Student-Stage1", "Inspector v10"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=sorted(GROUPS))
    parser.add_argument(
        "--evaluation", action="append", default=[], metavar="NAME=PATH",
        help="Extra evaluation file, added as its own group 'extra'.",
    )
    parser.add_argument("--output", required=True, help="JSON output; never overwritten.")
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share their average rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    return (last_rank - (counts - 1) / 2.0)[inverse]


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    ranks = average_ranks(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def margin_auc(labels: np.ndarray, margins: np.ndarray) -> float:
    winner_minus_loser = labels * margins
    return roc_auc(winner_minus_loser, -winner_minus_loser)


def pooled_auc(labels: np.ndarray, image_scores: np.ndarray) -> float:
    winner_is_a = labels == 1
    winners = np.where(winner_is_a, image_scores[:, 0], image_scores[:, 1])
    losers = np.where(winner_is_a, image_scores[:, 1], image_scores[:, 0])
    return roc_auc(winners, losers)


def accuracy(labels: np.ndarray, margins: np.ndarray) -> float:
    return float((np.sign(margins) == labels).mean())


def selection_mask(relevance: np.ndarray) -> np.ndarray:
    mask = relevance >= RELEVANCE_THRESHOLD
    empty = ~mask.any(axis=1)
    mask[empty, relevance[empty].argmax(axis=1)] = True
    return mask


def load_views(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per aggregation and dataset: sample ids, labels, margins, image scores."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    views: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset in DATASETS:
        rows = payload["predictions"][dataset]
        ids = [row["sample_id"] for row in rows]
        labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
        if np.any(labels == 0):
            raise ValueError(f"{path}: {dataset} contains ties.")
        if "margin_A_minus_B" in rows[0]:
            margins = np.asarray([row["margin_A_minus_B"] for row in rows], dtype=np.float64)
            scores = np.asarray([[row["score_A"], row["score_B"]] for row in rows], dtype=np.float64)
            views.setdefault("overall", {})[dataset] = {
                "ids": ids, "labels": labels, "margins": margins, "image_scores": scores,
            }
            continue
        evidence = np.asarray([row["signed_evidence"] for row in rows], dtype=np.float64)
        scores_a = np.asarray([row["scores_A"] for row in rows], dtype=np.float64)
        scores_b = np.asarray([row["scores_B"] for row in rows], dtype=np.float64)
        mask = selection_mask(np.asarray([row["relevance_probs"] for row in rows], dtype=np.float64))
        count = mask.sum(axis=1)
        views.setdefault("selected_equal", {})[dataset] = {
            "ids": ids, "labels": labels,
            "margins": (mask * evidence).sum(axis=1) / count,
            "image_scores": np.stack(
                [(mask * scores_a).sum(axis=1) / count, (mask * scores_b).sum(axis=1) / count], axis=1
            ),
        }
        aggregation = payload["metrics"][dataset]["fitted"]["aggregation"]
        weights = np.asarray(aggregation["weights"], dtype=np.float64)
        views.setdefault("all_metrics_fitted", {})[dataset] = {
            "ids": ids, "labels": labels,
            "margins": float(aggregation["beta"]) * (evidence @ weights),
            "image_scores": np.stack([scores_a @ weights, scores_b @ weights], axis=1),
        }
    return views


def interval(values: Sequence[float]) -> List[float]:
    return [round(100.0 * float(np.percentile(values, 2.5)), 2), round(100.0 * float(np.percentile(values, 97.5)), 2)]


def summarize(view: Dict[str, Any], resamples: np.ndarray) -> Dict[str, Any]:
    labels, margins = view["labels"], view["margins"]
    boot_acc = [accuracy(labels[i], margins[i]) for i in resamples]
    boot_auc = [margin_auc(labels[i], margins[i]) for i in resamples]
    return {
        "pairs": int(labels.size),
        "accuracy_percent": round(100.0 * accuracy(labels, margins), 2),
        "accuracy_percent_95": interval(boot_acc),
        "margin_auc_percent": round(100.0 * margin_auc(labels, margins), 2),
        "margin_auc_percent_95": interval(boot_auc),
        "pooled_auc_percent": round(100.0 * pooled_auc(labels, view["image_scores"]), 2),
    }


def mcnemar_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def compare(left: Dict[str, Any], right: Dict[str, Any], resamples: np.ndarray) -> Dict[str, Any]:
    if left["ids"] != right["ids"] or not np.array_equal(left["labels"], right["labels"]):
        raise ValueError("Paired comparison needs the same pairs in the same order.")
    labels = left["labels"]
    left_correct = np.sign(left["margins"]) == labels
    right_correct = np.sign(right["margins"]) == labels
    left_only = int((left_correct & ~right_correct).sum())
    right_only = int((~left_correct & right_correct).sum())
    deltas = [
        margin_auc(labels[i], left["margins"][i]) - margin_auc(labels[i], right["margins"][i])
        for i in resamples
    ]
    return {
        "accuracy_percent": [round(100.0 * left_correct.mean(), 2), round(100.0 * right_correct.mean(), 2)],
        "mcnemar_left_only_right_only": [left_only, right_only],
        "mcnemar_p": round(mcnemar_p(left_only, right_only), 4),
        "margin_auc_delta_percent": round(
            100.0 * (margin_auc(labels, left["margins"]) - margin_auc(labels, right["margins"])), 2
        ),
        "margin_auc_delta_percent_95": interval(deltas),
    }


def pick(views: Dict[str, Dict[str, Any]], aggregation: str) -> Dict[str, Dict[str, Any]]:
    """Inspector has a single overall margin; it serves every aggregation."""
    return views["overall"] if "overall" in views else views[aggregation]


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    groups = {name: dict(GROUPS[name]) for name in args.groups}
    for item in args.evaluation:
        name, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"--evaluation must be NAME=PATH, got {item}")
        groups.setdefault("extra", {})[name] = Path(path)

    rng = np.random.default_rng(SEED)
    resamples: Dict[str, np.ndarray] = {}
    cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}
    result: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "margin_auc": "ROC AUC of {d_i} vs {-d_i}, d_i = label_i * margin_i; equals P(d_i + d_j > 0)",
            "pooled_auc": "winner image scores vs loser image scores of all pairs (reference only)",
            "selected_equal_margin": "mean signed_evidence over metrics with relevance >= 0.5 (else argmax)",
            "all_metrics_fitted_margin": "beta * sum_k a_k * signed_evidence_k",
            "bootstrap": f"{BOOTSTRAP_REPLICATES} resamples of pairs, seed {SEED}",
        },
        "groups": {},
    }

    for group, models in groups.items():
        group_result: Dict[str, Any] = {"models": {}, "comparisons": []}
        for name, path in models.items():
            if path not in cache:
                cache[path] = load_views(path)
            views = cache[path]
            entry: Dict[str, Any] = {"evaluation": str(path)}
            for aggregation, per_dataset in views.items():
                entry[aggregation] = {}
                for dataset, view in per_dataset.items():
                    n = view["labels"].size
                    if dataset not in resamples:
                        resamples[dataset] = rng.integers(0, n, (BOOTSTRAP_REPLICATES, n))
                    entry[aggregation][dataset] = summarize(view, resamples[dataset])
            group_result["models"][name] = entry

        aggregations = ["selected_equal", "all_metrics_fitted"]
        if all("overall" in cache[path] for path in models.values()):
            aggregations = ["overall"]
        for left_name, right_name in COMPARISONS.get(group, []):
            for aggregation in aggregations:
                left = pick(cache[models[left_name]], aggregation)
                right = pick(cache[models[right_name]], aggregation)
                group_result["comparisons"].append({
                    "left": left_name, "right": right_name, "aggregation": aggregation,
                    **{dataset: compare(left[dataset], right[dataset], resamples[dataset]) for dataset in DATASETS},
                })
        result["groups"][group] = group_result

        for aggregation in aggregations:
            print(f"\n=== {group} [{aggregation}]   Acc / margin AUC [95%] / pooled AUC (reference)")
            for name in models:
                entry = group_result["models"][name]
                summary = entry["overall"] if "overall" in entry else entry[aggregation]
                cells = []
                for dataset in DATASETS:
                    s = summary[dataset]
                    cells.append(
                        f"{dataset.split('-')[-1]}: {s['accuracy_percent']:6.2f}  "
                        f"{s['margin_auc_percent']:6.2f} [{s['margin_auc_percent_95'][0]:.2f}-{s['margin_auc_percent_95'][1]:.2f}]  "
                        f"{s['pooled_auc_percent']:6.2f}"
                    )
                print(f"{name:52s} " + " | ".join(cells))
            for comparison in group_result["comparisons"]:
                if comparison["aggregation"] != aggregation:
                    continue
                cells = []
                for dataset in DATASETS:
                    c = comparison[dataset]
                    lo, hi = c["margin_auc_delta_percent_95"]
                    cells.append(
                        f"{dataset.split('-')[-1]}: {c['mcnemar_left_only_right_only'][0]}/{c['mcnemar_left_only_right_only'][1]} "
                        f"p={c['mcnemar_p']:.3f} dAUC {c['margin_auc_delta_percent']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                    )
                print(f"  {comparison['left']} vs {comparison['right']}: " + " | ".join(cells))

    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
