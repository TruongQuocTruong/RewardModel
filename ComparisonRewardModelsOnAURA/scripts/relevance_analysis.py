#!/usr/bin/env python
"""Relevance Head analysis of a Student score file on AURA-Test (CPU only, reads scores.jsonl written by
score_our_model.py; no GPU, no images).

Every AURA-Test pair is labelled for one metric k (its column). The Student scores it with head k; the
Relevance Head gives r_j = P(metric j is relevant to this edit) for all six metrics. Not used to select a head
(the bench gives the metric), only analysed here:

1. Per column: mean r_k of the column's own metric, share of pairs with r_k >= 0.5 (the training threshold),
   share of pairs where k is among the most relevant metrics (r_k equals the largest r_j; exact ties count
   for every tied metric, never broken by head order).
2. Per column, pairs split by r_k >= 0.5 / < 0.5: Accuracy and Margin AUC (non-tie pairs, margin of head k),
   Tie Head AUC from P(Tie) of head k (all pairs). "-" when a group has no pair of the needed kind.
3. Mean r_j of all six heads per column (rows: AURA column, columns: Student head).

Output: <output>.json and <output>.md. Usage:
    python scripts/relevance_analysis.py --scores results/Student-Stage1_Qwen3-VL-Reranker-8B/scores.jsonl \
        --output results/Student-Stage1_Qwen3-VL-Reranker-8B/relevance_analysis/relevance_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import bench_inputs as bi
import evaluation_metrics as em
from score_our_model import STUDENT_HEAD_OF_METRIC, STUDENT_HEADS

RELEVANCE_THRESHOLD = 0.5
COLUMNS = {"Aesthetic": "Aesthetic", "Realism": "Realism", "Fidelity": "Fidelity",
           "Background Consistency": "BC", "Foreground Consistency": "FC", "Structure Consistency": "SC"}


def _checked_records(records: Sequence[dict]) -> List[dict]:
    """Records with a known metric, the head of that metric, six finite relevance probabilities in [0, 1]."""
    if not records:
        raise ValueError("score file is empty")
    seen = set()
    for record in records:
        metric = record.get("metric")
        if metric not in STUDENT_HEAD_OF_METRIC:
            raise ValueError(f"{record.get('sample_id')}: unknown metric {metric!r}")
        if record.get("dimension") != STUDENT_HEAD_OF_METRIC[metric]:
            raise ValueError(f"{record['sample_id']}: dimension {record.get('dimension')!r} is not the head of {metric}")
        relevance = record.get("relevance_probs")
        if not isinstance(relevance, dict) or set(relevance) != set(STUDENT_HEADS):
            raise ValueError(f"{record['sample_id']}: relevance_probs must give all six Student heads")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in relevance.values()):
            raise ValueError(f"{record['sample_id']}: relevance_probs must be finite and in [0, 1]")
        if record["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id {record['sample_id']!r}")
        seen.add(record["sample_id"])
    return list(records)


def _safe(function, *args) -> Optional[float]:
    try:
        value = function(*args)
    except ValueError:
        return None
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def group_metrics(records: Sequence[dict]) -> Dict[str, object]:
    """Pair counts, Accuracy, Margin AUC (non-tie pairs) and Tie Head AUC (all pairs) of one group."""
    labels = [r["human_label"] for r in records]
    margins = [float(r["score_a"]) - float(r["score_b"]) for r in records]
    tie_probability = [float(r["probabilities"][2]) for r in records]
    return {
        "pairs": len(records),
        "non_tie_pairs": sum(label != "Tie" for label in labels),
        "tie_pairs": sum(label == "Tie" for label in labels),
        "accuracy": _safe(em.accuracy, labels, margins) if records else None,
        "margin_auc": _safe(em.margin_auc, labels, margins) if records else None,
        "tie_head_auc": _safe(em.tie_auc, labels, tie_probability) if records else None,
    }


def analyse(records: Sequence[dict], threshold: float = RELEVANCE_THRESHOLD) -> Dict[str, object]:
    records = _checked_records(records)
    columns = {}
    for metric, column in COLUMNS.items():
        rows = [r for r in records if r["metric"] == metric]
        if not rows:
            continue
        own = STUDENT_HEAD_OF_METRIC[metric]
        own_relevance = [float(r["relevance_probs"][own]) for r in rows]
        high = [r for r, v in zip(rows, own_relevance) if v >= threshold]
        low = [r for r, v in zip(rows, own_relevance) if v < threshold]
        most_relevant = [float(r["relevance_probs"][own]) == max(float(v) for v in r["relevance_probs"].values())
                         for r in rows]
        columns[column] = {
            "metric": metric, "head": own, "pairs": len(rows),
            "mean_own_relevance": sum(own_relevance) / len(rows),
            "share_own_relevance_at_least_threshold": len(high) / len(rows),
            "share_own_metric_most_relevant": sum(most_relevant) / len(rows),
            "all_pairs": group_metrics(rows),
            "own_relevance_at_least_threshold": group_metrics(high),
            "own_relevance_below_threshold": group_metrics(low),
            "mean_relevance_per_head": {h: sum(float(r["relevance_probs"][h]) for r in rows) / len(rows)
                                        for h in STUDENT_HEADS},
        }
    return {"threshold": threshold, "pairs": len(records), "columns": columns}


def _percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def render_markdown(report: dict) -> str:
    threshold, columns = report["threshold"], report["columns"]
    lines = [f"### Relevance of the column's own metric (threshold {threshold})", "",
             "| Column | Pairs | Mean r_own | % r_own >= threshold | % own metric most relevant (ties count) |", "|---|---|---|---|---|"]
    for name, c in columns.items():
        lines.append(f"| {name} | {c['pairs']} | {c['mean_own_relevance']:.3f} | "
                     f"{_percent(c['share_own_relevance_at_least_threshold'])} | "
                     f"{_percent(c['share_own_metric_most_relevant'])} |")
    lines += ["", f"### Head of the column's metric, pairs split by r_own (Acc \\| Margin AUC on non-tie pairs, "
                  f"Tie Head AUC on all pairs; %)", "",
              "| Column | Group | Pairs (non-tie / tie) | Acc | Margin AUC | Tie Head AUC |", "|---|---|---|---|---|---|"]
    for name, c in columns.items():
        for key, label in (("all_pairs", "all"), ("own_relevance_at_least_threshold", f">= {threshold}"),
                           ("own_relevance_below_threshold", f"< {threshold}")):
            g = c[key]
            lines.append(f"| {name} | {label} | {g['pairs']} ({g['non_tie_pairs']} / {g['tie_pairs']}) | "
                         f"{_percent(g['accuracy'])} | {_percent(g['margin_auc'])} | {_percent(g['tie_head_auc'])} |")
    lines += ["", "### Mean relevance of every Student head per column", "",
              "| Column | " + " | ".join(STUDENT_HEADS) + " |", "|---|" + "---|" * len(STUDENT_HEADS)]
    for name, c in columns.items():
        lines.append(f"| {name} | " + " | ".join(f"{c['mean_relevance_per_head'][h]:.3f}" for h in STUDENT_HEADS)
                     + " |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path without suffix; writes .json and .md")
    parser.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD)
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0, 1)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    report = analyse(bi.load_jsonl(args.scores), args.threshold)
    report["scores"] = str(args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
