#!/usr/bin/env python3
"""Compare the checkpoints of one Student Stage1 run on the evaluation sets.

Reads only the saved predictions of ``evaluate_student_checkpoints.py``
(``<eval-dir>/checkpoint-N/evaluation.json``, no images, no GPU) and the run's
``logging.jsonl`` for the step of each checkpoint. Each Student pair is reduced
to the ``selected_equal`` aggregation, the primary Student result:

- margin d = mean_{k in S} signed_evidence_k (AUC-Margin.md; evaluation.json
  files written before 2026-10-29 store the sum, which gives the same accuracy
  but not the same margin AUC),
- -|d| on sets with ties uses mean_{k in S} (score_A_k - score_B_k), as in
  evaluate_student_checkpoints.py and tie_head_ablation.md 12.1,
- P(A), P(B), P(Tie) = mean over S of (1-t_k)w_k, (1-t_k)(1-w_k), t_k,

with S = {k : relevance_probs_k >= 0.5} (the most relevant metric when none
passes). The metrics and figures are then computed exactly as for the
Inspector (``analyze_inspector_training.py``): EditReward-Data-Test, and
EditReward-Bench without and with its tied pairs, under the original and the
VLM-as-a-judge labels. Sets without ties: accuracy and symmetric margin AUC
({d} vs {-d}). Sets with ties (tie pairs positive, the rest negative): margin
AUC scored by -|d| and Tie Head AUC scored by P(Tie). One chart per figure,
one point per checkpoint.

Writes PNG figures and ``checkpoint_analysis.json`` to ``--output-dir``
(default ``<eval-dir>/checkpoint_analysis``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from analyze_inspector_training import (
    DEFAULT_VLM_PAIRS_CSV,
    checkpoint_metrics,
    load_vlm_labels,
    metric_lines,
    plot_checkpoint_figure,
)

RELEVANCE_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Student Stage1 training output directory.")
    parser.add_argument("--eval-dir", required=True, help="Directory holding checkpoint-N/evaluation.json.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vlm-pairs-csv", default=str(DEFAULT_VLM_PAIRS_CSV))
    return parser.parse_args()


def pairs_per_step(run_dir: Path) -> int:
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    return int(args["per_device_train_batch_size"]) * int(args["gradient_accumulation_steps"])


def selected_equal_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """One Student prediction in the row format of the Inspector evaluation."""
    relevance = np.asarray(row["relevance_probs"], dtype=np.float64)
    mask = relevance >= RELEVANCE_THRESHOLD
    if not mask.any():
        mask[relevance.argmax()] = True
    tie = np.asarray(row["tie_probs"], dtype=np.float64)
    cond_a = 1.0 / (1.0 + np.exp(-(np.asarray(row["scores_A"]) - np.asarray(row["scores_B"]))))
    probability_a = ((1.0 - tie) * cond_a)[mask].mean()
    probability_b = ((1.0 - tie) * (1.0 - cond_a))[mask].mean()
    raw = np.asarray(row["scores_A"], dtype=np.float64) - np.asarray(row["scores_B"], dtype=np.float64)
    return {
        "sample_id": row["sample_id"],
        "label": int(row["label"]),
        "margin_A_minus_B": float(np.asarray(row["signed_evidence"], dtype=np.float64)[mask].mean()),
        "selected_count": int(mask.sum()),
        "tie_margin": float(raw[mask].mean()),
        "probability_A": float(probability_a),
        "probability_B": float(probability_b),
        "probability_Tie": float(tie[mask].mean()),
    }


def load_student_checkpoints(eval_dir: Path, per_step: int) -> List[Dict[str, Any]]:
    checkpoints = []
    for path in eval_dir.glob("checkpoint-*/evaluation.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        step = int(re.sub(r"\D", "", path.parent.name))
        predictions = {dataset: [selected_equal_row(row) for row in rows]
                       for dataset, rows in result["predictions"].items()}
        for dataset, rows in predictions.items():
            stored = {r["sample_id"]: r.get("selected_equal_margin") for r in result["predictions"][dataset]}
            for row in rows:
                # Non-tie rows carry the selected_equal margin of evaluation.json: the mean
                # since the evaluation fix of 2026-10-29, the sum (mean x |S|) in older files.
                value = stored[row["sample_id"]]
                if value is not None and not (
                        np.isclose(value, row["margin_A_minus_B"])
                        or np.isclose(value, row["margin_A_minus_B"] * row["selected_count"])):
                    raise ValueError(f"selected_equal margin mismatch in {path}: {row['sample_id']}")
        checkpoints.append({"checkpoint": path.parent.name, "step": step, "pairs_seen": step * per_step,
                            "is_base": False, "predictions": predictions})
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-*/evaluation.json under {eval_dir}")
    return sorted(checkpoints, key=lambda item: item["step"])


def plot_figures(table, sets, output_dir: Path) -> List[str]:
    plain = [spec for spec in sets if not spec["with_ties"]]
    tied = [spec for spec in sets if spec["with_ties"]]
    labels = [f"{row['pairs_seen']} ({row['checkpoint']})" for row in table]
    figures = []

    def draw(name: str, lines, title: str, ylabel: str, **kwargs) -> None:
        plot_checkpoint_figure(labels, lines, title, ylabel, output_dir / name, **kwargs)
        figures.append(name)

    draw("checkpoint_accuracy.png", metric_lines(table, plain, "accuracy_percent"),
         "Student selected_equal A/B accuracy (%) on non-tie pairs", "Accuracy (%)",
         legend_loc="lower right")
    draw("checkpoint_margin_auc.png", metric_lines(table, plain, "margin_auc"),
         "Student selected_equal margin AUC (%) on non-tie pairs", "Margin AUC (%)",
         legend_loc="lower right")
    draw("checkpoint_log_loss.png", metric_lines(table, plain, "log_loss"),
         "Student selected_equal A/B log loss on non-tie pairs", "Log loss", legend_loc="upper left")
    # |d| does not depend on the labels, so only the original-label sets are drawn.
    draw("checkpoint_mean_abs_margin.png",
         metric_lines(table, [spec for spec in plain if spec["source"] == "original"], "mean_abs_margin"),
         "Student mean |d| on non-tie pairs (higher = more confident)",
         "|d| = |mean signed_evidence over selected metrics|", legend_loc="upper left")
    # Sets with ties: tie pairs positive, the rest negative; margin AUC from -|d|, Tie Head AUC from P(Tie).
    if tied:
        tie_label = lambda spec: (f"{spec['name']}: {spec['tie_pairs']} tie pairs"
                                  + (" (too few, reference only)" if spec["tie_pairs"] < 10 else ""))
        draw("checkpoint_margin_auc_with_ties.png", metric_lines(table, tied, "margin_auc", label=tie_label),
             "Student margin AUC (%) on sets with ties: tie pairs vs the rest, scored by -|d| (50 = random)",
             "AUC (%)", reference=50.0, legend_loc="lower right")
        draw("checkpoint_tie_head_auc.png", metric_lines(table, tied, "tie_head_auc", label=tie_label),
             "Student Tie Head AUC (%) on sets with ties: tie pairs vs the rest, scored by P(Tie) (50 = random)",
             "AUC (%)", reference=50.0, legend_loc="lower right")
    return figures


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    output_dir = Path(args.output_dir or eval_dir / "checkpoint_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = load_student_checkpoints(eval_dir, pairs_per_step(run_dir))
    vlm_labels = load_vlm_labels(Path(args.vlm_pairs_csv)) if args.vlm_pairs_csv else None
    evaluation = checkpoint_metrics(checkpoints, vlm_labels)
    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "aggregation": "selected_equal",
        "checkpoint_sets": evaluation["sets"],
        "checkpoints": evaluation["rows"],
        "figures": plot_figures(evaluation["rows"], evaluation["sets"], output_dir),
    }
    (output_dir / "checkpoint_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(report['figures'])} figures and checkpoint_analysis.json to {output_dir}")
    for row in evaluation["rows"]:
        cells = []
        for spec in evaluation["sets"]:
            m = row[spec["name"]]
            if spec["with_ties"]:
                cells.append(f"{spec['name']}: margin AUC (-|d|) {m['margin_auc']['percent']:.2f}"
                             f" / Tie Head AUC {m['tie_head_auc']['percent']:.2f}")
            else:
                cells.append(f"{spec['name']}: acc {m['accuracy_percent']:.2f} / margin AUC "
                             f"{m['margin_auc']['percent']:.2f} / |d| {m['mean_abs_margin']:.2f}"
                             f" / log loss {m['log_loss']:.3f}")
        print(f"{row['checkpoint']} ({row['pairs_seen']} pairs):\n  " + "\n  ".join(cells))


if __name__ == "__main__":
    main()
