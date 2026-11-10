#!/usr/bin/env python3
"""EditReward-Bench and VLM-Reannotate results of one Stage 2 iteration run, per iteration.

Reads only saved predictions (no images, no GPU):
  Student-Stage1 (start point): evaluation_results/student/<STAGE1>/checkpoint-126/bench_evaluation.json
  Student and Teacher of the run: evaluation_results/train_loop/<PREFIX><run>/checkpoint-N/
      bench_evaluation.json (Student) and teacher_bench_evaluation.json (Teacher)
  VLM labels: EvalDatasets/EditReward/EditReward-Bench_VLM_judge/pairs.csv (load_vlm_labels).
Each prediction is reduced with selected_equal_row (analyze_student_checkpoints.py) and the sets and
metrics are those of analyze_inspector_training.py (evaluation_sets, set_metrics), the definitions of
all earlier Student results. The checkpoint at position i (sorted by step) is the end of iteration i.

Paired bootstrap: BOOTSTRAP replicates of the pairs (seed 42); in each replicate both models are scored on
the same resampled pairs and the difference is taken. p = two-sided, 2 x min(P(diff <= 0), P(diff >= 0)).
Comparisons: Student / Teacher of each iteration minus Student-Stage1, Student minus Teacher of the same
checkpoint, Student of each iteration minus Student of iteration 1.

Outputs in evaluation_results/train_loop/<PREFIX><run>/bench_iteration_analysis/:
  summary.csv           point values per model, set and metric
  paired_bootstrap.csv  differences with 95% interval and p
  pair_changes.csv      pairs correct -> wrong and wrong -> correct against Student-Stage1 (sets without ties)
  bench_iterations_<set>.png  one figure per set, Student and Teacher per iteration, start point dashed
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_inspector_training import (  # noqa: E402
    DEFAULT_VLM_PAIRS_CSV, evaluation_sets, load_vlm_labels, set_metrics, weighted_auc, win_matrix,
)
from analyze_student_checkpoints import selected_equal_row  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "evaluation_results"
STAGE1 = (RESULTS / "student" / "student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32"
          "_initialized_from_inspector_checkpoint63" / "checkpoint-126" / "bench_evaluation.json")
PREFIX = "stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_"
DATASET = "EditReward-Bench"
BOOTSTRAP = 10000
SEED = 42
STUDENT_COLOR, TEACHER_COLOR, START_COLOR = "#2a78d6", "#eb6834", "#52514e"
GRID = "#e4e3df"


def load(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = [selected_equal_row(row) for row in result["predictions"][DATASET]]
    return {"role": result.get("model_role"), "checkpoint": str(result.get("checkpoint")), "rows": rows}


def per_pair(spec, rows):
    """Arrays of the set in spec['ids'] order: labels, margins, tie margins, P(A), P(B), P(Tie)."""
    by_id = {row["sample_id"]: row for row in rows}
    labels = np.asarray([spec["labels"][key] for key in spec["ids"]], dtype=np.int64)
    pick = lambda field: np.asarray([by_id[key][field] for key in spec["ids"]], dtype=np.float64)  # noqa: E731
    probs = np.stack([pick("probability_A"), pick("probability_B"), pick("probability_Tie")], axis=1)
    return labels, pick("margin_A_minus_B"), pick("tie_margin"), probs


def replicate_metrics(spec, rows, counts):
    """Metric values for every bootstrap replicate (rows of counts) and for the full set (counts of 1)."""
    labels, margins, tie_margins, probs = per_pair(spec, rows)
    is_tie, decided = labels == 0, labels != 0
    out = {}
    signed = labels[decided] * margins[decided]
    c = counts[:, decided]
    out["accuracy_percent"] = 100 * (c * (signed > 0)).sum(axis=1) / c.sum(axis=1)
    ones = np.ones(decided.sum())
    out["margin_auc_percent"] = 100 * weighted_auc(win_matrix(signed, -signed), ones, ones, c)
    if spec["with_ties"]:
        target = np.select([labels == 1, labels == -1], [0, 1], default=2)
        correct = probs.argmax(axis=1) == target
        out["three_class_accuracy_percent"] = 100 * (counts * correct).sum(axis=1) / counts.sum(axis=1)
        score = -np.abs(tie_margins)
        out["tie_margin_auc_percent"] = 100 * weighted_auc(
            win_matrix(score, score), is_tie.astype(float), decided.astype(float), counts)
        out["tie_head_auc_percent"] = 100 * weighted_auc(
            win_matrix(probs[:, 2], probs[:, 2]), is_tie.astype(float), decided.astype(float), counts)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name after the prefix, e.g. ema_iteration_0.99")
    args = parser.parse_args()
    eval_dir = RESULTS / "train_loop" / (PREFIX + args.run)
    out_dir = eval_dir / "bench_iteration_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted((p for p in eval_dir.glob("checkpoint-*") if p.is_dir()),
                         key=lambda p: int(re.sub(r"\D", "", p.name)))
    models = [("Student-Stage1", 0, 0, load(STAGE1))]
    for iteration, folder in enumerate(checkpoints, start=1):
        step = int(re.sub(r"\D", "", folder.name))
        for role, name in (("Student", "bench_evaluation.json"), ("Teacher", "teacher_bench_evaluation.json")):
            path = folder / name
            if not path.exists():
                sys.exit(f"missing {path}")
            model = load(path)
            if model["role"] != role.lower() or not model["checkpoint"].endswith(f"checkpoint-{step}.pt"):
                sys.exit(f"{path}: role {model['role']} / checkpoint {model['checkpoint']} do not match")
            models.append((role, iteration, step, model))
    if len(checkpoints) == 0:
        sys.exit(f"no checkpoint-* folders in {eval_dir}")

    vlm = load_vlm_labels(DEFAULT_VLM_PAIRS_CSV)
    sets = evaluation_sets({DATASET: [{"sample_id": r["sample_id"], "label": r["label"]}
                                      for r in models[0][3]["rows"]]}, vlm)
    rng = np.random.default_rng(SEED)
    summary, paired, changes = [], [], []
    for spec in sets:
        size = len(spec["ids"])
        counts = np.zeros((BOOTSTRAP, size))
        np.add.at(counts, (np.arange(BOOTSTRAP)[:, None], rng.integers(0, size, (BOOTSTRAP, size))), 1.0)
        full = np.ones((1, size))
        values, replicates = {}, {}
        for role, iteration, step, model in models:
            key = (role, iteration)
            # Point values from set_metrics (shared definitions); replicates for the paired bootstrap.
            reference = set_metrics(spec, {r["sample_id"]: r for r in model["rows"]}, full, False)
            point = {k: float(v[0]) for k, v in replicate_metrics(spec, model["rows"], full).items()}
            check = reference["non_tie_pairs"] if spec["with_ties"] else reference
            assert abs(point["accuracy_percent"] - check["accuracy_percent"]) < 1e-9
            assert abs(point["margin_auc_percent"] - check["margin_auc"]["percent"]) < 1e-9
            if spec["with_ties"]:
                assert abs(point["three_class_accuracy_percent"] - reference["three_class_accuracy_percent"]) < 1e-9
                assert abs(point["tie_margin_auc_percent"] - reference["margin_auc"]["percent"]) < 1e-9
                assert abs(point["tie_head_auc_percent"] - reference["tie_head_auc"]["percent"]) < 1e-9
            values[key] = point
            replicates[key] = replicate_metrics(spec, model["rows"], counts)
            for metric, value in point.items():
                summary.append({"set": spec["name"], "model": role, "iteration": iteration, "step": step,
                                "metric": metric, "value": round(value, 4),
                                "pairs": int(check["pairs"]) if metric in ("accuracy_percent", "margin_auc_percent")
                                else size})
        iterations = sorted({it for _, it, _, _ in models if it > 0})
        comparisons = []
        for it in iterations:
            comparisons += [(("Student", it), ("Student-Stage1", 0)), (("Teacher", it), ("Student-Stage1", 0)),
                            (("Student", it), ("Teacher", it))]
            if it > 1:
                comparisons.append((("Student", it), ("Student", 1)))
        for first, second in comparisons:
            for metric in values[first]:
                diff = replicates[first][metric] - replicates[second][metric]
                diff = diff[np.isfinite(diff)]
                low, high = np.percentile(diff, [2.5, 97.5])
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                paired.append({"set": spec["name"], "metric": metric,
                               "model": f"{first[0]} iteration {first[1]}" if first[1] else first[0],
                               "minus": f"{second[0]} iteration {second[1]}" if second[1] else second[0],
                               "difference": round(values[first][metric] - values[second][metric], 4),
                               "ci95_low": round(float(low), 4), "ci95_high": round(float(high), 4),
                               "p_two_sided": round(float(p), 4), "replicates": int(diff.size)})
        if not spec["with_ties"]:
            labels, start_margins, _, _ = per_pair(spec, models[0][3]["rows"])
            decided = labels != 0
            start_ok = (labels * start_margins > 0)[decided]
            for role, iteration, step, model in models[1:]:
                _, margins, _, _ = per_pair(spec, model["rows"])
                ok = (labels * margins > 0)[decided]
                changes.append({"set": spec["name"], "model": role, "iteration": iteration, "step": step,
                                "pairs": int(decided.sum()), "correct": int(ok.sum()),
                                "correct_to_wrong": int((start_ok & ~ok).sum()),
                                "wrong_to_correct": int((~start_ok & ok).sum())})
        plot_set(spec, values, iterations, out_dir)

    for name, rows in (("summary.csv", summary), ("paired_bootstrap.csv", paired), ("pair_changes.csv", changes)):
        with open(out_dir / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", out_dir / name)


def plot_set(spec, values, iterations, out_dir):
    metrics = list(values[("Student-Stage1", 0)])
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for role, color in (("Student", STUDENT_COLOR), ("Teacher", TEACHER_COLOR)):
            y = [values[(role, it)][metric] for it in iterations]
            ax.plot(iterations, y, color=color, linewidth=2, marker="o", markersize=7, label=role)
        start = values[("Student-Stage1", 0)][metric]
        ax.axhline(start, color=START_COLOR, linestyle="--", linewidth=1.2, label=f"Student-Stage1 ({start:.2f})")
        ax.set_xticks(iterations)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_title(metric.replace("_percent", " (%)").replace("_", " "), fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{spec['name']}: Student and Teacher per iteration (differences and p in paired_bootstrap.csv)",
                 fontsize=11)
    fig.tight_layout()
    tag = re.sub(r"[^A-Za-z0-9]+", "_", spec["name"]).strip("_")
    path = out_dir / f"bench_iterations_{tag}.png"
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
