"""Analysis of the single-operation augmentation experiments of Stage 2 (2026-11-05).

One baseline run (no augmentation) and up to 11 runs that each apply ONE augmentation
operation to the Student input (stage2_tuning.md section 8.3). Everything is read from
the run folders themselves (training_log_<job>.jsonl, per_sample_log_<job>.jsonl) and
written into <run folder>/analysis/ (user, 2026-11-05: analysis lives next to the run):

  step_losses.csv          one row per optimizer step: HumanGT CE, Gemma preference, Gemma
                           relevance (mask), Unlabeled loss and its parts, total loss, lr, clip
  samples_overall.csv      one row per Unlabeled sample and step: overall labels of Teacher /
                           Student / Inspector, same or not, confidences, gap, KL, CE, and the
                           augmentation (applied, reason, strength)
  samples_metrics.csv      one row per Unlabeled sample, step and metric (6 metrics): labels,
                           same or not, confidences, gap, KL, CE, relevance, selected
  summary.json             all numbers below (overall and per metric, by strength bin)
  loss_vs_baseline.png     6 losses per step, operation vs baseline (raw + moving average)
  metrics_vs_baseline.png  overall + 6 metrics: Student = Teacher rate, Teacher / Student
                           confidence on the Teacher label, gap, KL; operation vs baseline
  label_transitions.png    Teacher -> Student label matrices (A / B / Tie), overall and per
                           metric: operation, baseline, difference

The baseline folder gets the same CSVs / summary / own plots, plus ranking.csv and
ranking.png comparing every operation with the baseline (used to pick the top 4-5).

Rates of an operation run are computed on the samples where the operation was APPLIED
(augmentation_applied = true); the baseline uses all its samples. Light: reads logs
only, no model, no image.

Usage:
  python analyze_single_operation_augmentation.py \
      [--output-root .../output_training] [--prefix stage2_tuning_unlabeled1008] \
      [--suffix _test_2_steps] [--moving-average 8] [--group combined]

--group combined (2026-11-06): the combined-operation runs and the Inspector one-hot runs
(stage2_tuning.md 8.5). Writes analysis/ only in those runs (plus learnable_loss_vs_baseline.png
and training_dynamics_vs_baseline.png); the baseline and single-operation runs are read only.
Cross-run tables: <baseline>/analysis/combined_operations/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(
    "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training")
DEFAULT_PREFIX = "stage2_tuning_unlabeled1008"
OPERATIONS = ("composite", "blur", "noise", "mask_content", "black_rectangles", "typo", "flip",
              "drop_action", "distractor_instruction", "rotation", "injection")
METRICS = ("aesthetic", "fidelity", "realism", "background_consistency",
           "foreground_consistency", "structure_consistency")
CLASSES = ("A", "B", "Tie")
TARGET_AGREEMENT = (0.75, 0.80)  # user, 2026-11-06 (was 0.70-0.75)
# Strength value used to split an operation's samples into bins (None = no bins).
STRENGTH_KEY = {"mask_content": "fraction_changed", "typo": "fraction_changed", "blur": "radius",
                "noise": "sigma", "rotation": "degrees", "black_rectangles": "area",
                "composite": "main_image_position"}
TEXT_FRACTION_BINS = ((0.0, 0.3), (0.3, 1.01))  # drawn 20-40% -> realised fraction bins

# Reference palette (dataviz skill): slot 1 = operation, slot 2 = baseline.
COLOR_OPERATION = "#2a78d6"
COLOR_BASELINE = "#eb6834"
COLOR_COUNTERPART = "#7a5cc7"  # soft-Inspector counterpart of a one-hot run
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", [COLOR_OPERATION, "#f0efec", COLOR_BASELINE])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])

# Unified loss names (2026-11-07): L_human, L_preference (GemmaGT, = L_tie + L_winner),
# L_relevance (GemmaGT), L_unlabeled = CE(Inspector, Student overall) + lambda_C * KL(Teacher||Student),
# L_total = L_human + L_preference + L_relevance + lambda_U * L_unlabeled.
LOSS_PANELS = (
    ("human", "L_human = CE(HumanGT label, Student overall)"),
    ("gemma", "L_preference = L_tie + L_winner (GemmaGT)"),
    ("mask", "L_relevance = BCE(GemmaGT relevance, Student relevance)"),
    ("unlabeled_inspector_cross_entropy", "CE(Inspector, Student overall)  [part of L_unlabeled]"),
    ("unlabeled_teacher_kl", "KL(Teacher‖Student), selected metrics  [part of L_unlabeled]"),
    ("total", "L_total = L_human + L_preference + L_relevance + λ_U·L_unlabeled"),
)
# Learnable part (floor removed): the soft Inspector target has floor = its entropy; one-hot floor 0.
LEARNABLE_PANELS = (
    ("total_learnable", "L_total, learnable part (floor removed)"),
    ("unlabeled_learnable", "L_unlabeled, learnable = KL(Inspector‖Student) + λ_C·KL(Teacher‖Student)"),
    ("unlabeled_inspector_kl", "KL(Inspector‖Student) = CE(Inspector, Student) − floor"),
    ("unlabeled_floor_inspector_entropy", "Floor = entropy of the Inspector target (no gradient; 0 for one-hot)"),
    ("tie", "L_tie  [part of L_preference]"),
    ("winner", "L_winner  [part of L_preference]"),
)
NOT_SMOOTHED = {"learning_rate"}
DYNAMICS_PANELS = (
    ("gradient_norm_before_clipping", "Gradient norm before clipping (whole model)"),
    ("clip_coefficient", "Clip coefficient = max_norm / gradient norm"),
    ("learning_rate", "Learning rate"),
    ("gradient_unlabeled_over_labeled", "Gradient norm ratio: L_unlabeled / labeled losses"),
    ("gradient_cosine_labeled_unlabeled", "Cosine(labeled-loss gradient, L_unlabeled gradient)"),
    ("gradient_norm_heads", "Gradient norm of the heads"),
)

# Combined-operation round (stage2_tuning.md 8.5, 2026-11-06): run suffix -> reference runs.
# Every combined run is compared with the baseline; an Inspector one-hot run is also compared
# with its soft-Inspector counterpart. Single-operation runs are read only (reference columns).
COMBINED_RUNS = (
    "augmentation_composite_blur_noise_mask_content",
    "augmentation_composite_blur_noise", "augmentation_composite_blur_mask_content",
    "augmentation_composite_noise_mask_content", "augmentation_blur_noise_mask_content",
    "augmentation_composite_blur", "augmentation_composite_noise", "augmentation_composite_mask_content",
    "augmentation_blur_noise", "augmentation_blur_mask_content", "augmentation_noise_mask_content",
    "augmentation_composite_blur_noise_mask_content_inspector_one_hot", "baseline_inspector_one_hot",
    # Labeled pairs augmented too (--augment-labeled, 2026-11-06/07; jobs 10842, 10843, 10926, 10927).
    "augmentation_composite_blur_noise_mask_content_labeled_augmented",
    "augmentation_composite_blur_noise_mask_content_inspector_one_hot_labeled_augmented",
    "augmentation_composite_blur_mask_content_labeled_augmented",
    "augmentation_composite_noise_mask_content_labeled_augmented",
)
SOFT_COUNTERPART = {
    "augmentation_composite_blur_noise_mask_content_inspector_one_hot": "augmentation_composite_blur_noise_mask_content",
    "baseline_inspector_one_hot": "baseline",
    "augmentation_composite_blur_noise_mask_content_inspector_one_hot_labeled_augmented":
        "augmentation_composite_blur_noise_mask_content_labeled_augmented",
}
SINGLE_REFERENCE_RUNS = ("augmentation_composite", "augmentation_blur", "augmentation_noise", "augmentation_mask_content")
STEP_PHASES = ((1, 15), (16, 63), (64, 126))  # composite KL Teacher drops fast in the first ~15 steps


# ----------------------------------------------------------------------------- reading

def latest_job_file(run: Path, stem: str) -> Optional[Path]:
    files = sorted(run.glob(f"{stem}_*.jsonl"), key=lambda p: int(re.findall(r"_(\d+)\.jsonl$", p.name)[0])
                   if re.findall(r"_(\d+)\.jsonl$", p.name) else -1)
    return files[-1] if files else None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut by a stopped job
    return rows


def step_losses(run: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    path = latest_job_file(run, "training_log")
    if path is None:
        return [], None
    out = []
    for row in read_jsonl(path):
        if row.get("kind") != "step" or "diagnostics" not in row:
            continue
        losses = row["diagnostics"]["losses"]
        lam = float(row.get("lambda_unlabeled", row["diagnostics"].get("lambda_unlabeled", 1.0)))
        record = {"step": int(row["step"]), "round": int(row.get("round", 1)), "lambda_unlabeled": lam}
        for key in ("human", "gemma", "mask", "tie", "winner", "unlabeled", "unlabeled_inspector_cross_entropy",
                    "unlabeled_inspector_target_entropy", "unlabeled_inspector_kl", "unlabeled_teacher_kl"):
            record[key] = losses.get(key)
        # Single loop: replay scales are 1 (round 1); same total as plot_job_losses.py.
        record["total"] = sum(float(losses.get(k) or 0.0) for k in ("human", "gemma", "mask")) \
            + lam * float(losses.get("unlabeled") or 0.0)
        # Learnable part (user, 2026-11-06): the Unlabeled loss minus the entropy of the soft
        # Inspector label (a constant, no gradient) = KL(Inspector||Student) + lambda_C * KL Teacher.
        # HumanGT, Gemma preference and Gemma relevance use one-hot / 0-1 labels (majority vote),
        # so their floor is 0 and the whole value is learnable.
        entropy = losses.get("unlabeled_inspector_target_entropy")
        record["unlabeled_floor_inspector_entropy"] = entropy
        record["unlabeled_learnable"] = (None if losses.get("unlabeled") is None or entropy is None
                                         else float(losses["unlabeled"]) - float(entropy))
        record["total_learnable"] = (None if record["unlabeled_learnable"] is None else
                                     sum(float(losses.get(k) or 0.0) for k in ("human", "gemma", "mask"))
                                     + lam * record["unlabeled_learnable"])
        rates = row["diagnostics"].get("learning_rates") or [None]
        record["learning_rate"] = rates[0]
        record["clip_coefficient"] = row["diagnostics"].get("clip_coefficient")
        # Gradient norms before clipping (2026-11-06): whole model, labeled vs Unlabeled part,
        # their ratio and cosine; same for the heads only.
        norms = row["diagnostics"].get("gradient_norm_before_clipping") or {}
        whole, heads = norms.get("all") or {}, norms.get("heads") or {}
        record["gradient_norm_before_clipping"] = whole.get("total", row.get("gradient_norm_before_clipping"))
        record["gradient_norm_labeled"] = whole.get("labeled")
        record["gradient_norm_unlabeled"] = whole.get("unlabeled")
        record["gradient_unlabeled_over_labeled"] = whole.get("unlabeled_over_labeled")
        record["gradient_cosine_labeled_unlabeled"] = whole.get("cosine")
        record["gradient_norm_heads"] = heads.get("total")
        record["gradient_norm_heads_labeled"] = heads.get("labeled")
        record["gradient_norm_heads_unlabeled"] = heads.get("unlabeled")
        record["gradient_cosine_heads"] = heads.get("cosine")
        out.append(record)
    return out, re.findall(r"_(\d+)\.jsonl$", path.name)[0]


def unlabeled_rows(run: Path) -> List[Dict[str, Any]]:
    path = latest_job_file(run, "per_sample_log")
    if path is None:
        return []
    return [row for row in read_jsonl(path) if row.get("kind") == "unlabeled" and "metrics" in row]


# ----------------------------------------------------------------------------- numbers

def mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return round(sum(values) / len(values), 6) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return round(statistics.median(values), 6) if values else None


def block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall and per-metric numbers of a set of per-sample rows."""
    def part(get) -> Dict[str, Any]:
        items = [get(r) for r in rows]
        gaps = [i["confidence_gap"] for i in items]
        transitions = Counter((i["teacher_label"], i["student_label"]) for i in items)
        return {
            "samples": len(items),
            "student_same_as_teacher": mean([float(i["student_same_as_teacher"]) for i in items]),
            "teacher_confidence_mean": mean([i["teacher_confidence"] for i in items]),
            "student_confidence_on_teacher_label_mean": mean([i["student_confidence_on_teacher_label"] for i in items]),
            "confidence_gap_mean": mean(gaps),
            "confidence_gap_median": median(gaps),
            "confidence_gap_above_0.1": mean([float(g > 0.1) for g in gaps]),
            "confidence_gap_above_0.2": mean([float(g > 0.2) for g in gaps]),
            "kl_teacher_student_mean": mean([i["kl"] for i in items]),
            "kl_teacher_student_median": median([i["kl"] for i in items]),
            "transitions": {f"{t}->{s}": transitions.get((t, s), 0) for t in CLASSES for s in CLASSES},
        }

    if not rows:
        return {"samples": 0}
    overall = part(lambda r: {"teacher_label": r["teacher_label"], "student_label": r["student_label"],
                              "student_same_as_teacher": r["student_same_as_teacher"],
                              "teacher_confidence": r["teacher_confidence"],
                              "student_confidence_on_teacher_label": r["student_confidence_on_teacher_label"],
                              "confidence_gap": r["confidence_gap"], "kl": r["overall_kl_teacher_student"]})
    overall["teacher_kl_loss_mean"] = mean([r["teacher_kl"] for r in rows])
    overall["inspector_cross_entropy_mean"] = mean([r["inspector_cross_entropy"] for r in rows])
    overall["student_same_as_inspector"] = mean([float(r["student_same_as_inspector"]) for r in rows])
    metrics = {}
    for name in METRICS:
        metrics[name] = part(lambda r, n=name: {**r["metrics"][n], "kl": r["metrics"][n]["kl_teacher_student"]})
        metrics[name]["teacher_selected_fraction"] = mean([float(r["metrics"][name]["teacher_selected"]) for r in rows])
    return {"samples": len(rows), "overall": overall, "metrics": metrics}


def strength_bins(operation: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    key = STRENGTH_KEY.get(operation)
    if key is None or not rows:
        return {}
    values = [(r.get("augmentation_strength") or {}).get(key) for r in rows]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    if all(isinstance(v, str) for v in values if v is not None):
        for r, v in zip(rows, values):
            groups.setdefault(f"{key}={v}", []).append(r)
    else:
        numbers = [abs(float(v)) for v in values if v is not None]
        if not numbers:
            return {}
        if key == "fraction_changed":
            edges = TEXT_FRACTION_BINS
        else:  # tertiles of the drawn values
            ordered = sorted(numbers)
            cut = [ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]]
            edges = ((ordered[0], cut[0]), (cut[0], cut[1]), (cut[1], ordered[-1] + 1e-9))
        for r, v in zip(rows, values):
            if v is None:
                continue
            v = abs(float(v))
            for low, high in edges:
                if low <= v < high:
                    groups.setdefault(f"{key} in [{low:.3g}, {min(high, 1.0) if key == 'fraction_changed' else high:.3g})",
                                      []).append(r)
                    break
    return {name: {"samples": len(g), "overall": block(g).get("overall", {})} for name, g in sorted(groups.items())}


# ----------------------------------------------------------------------------- writing

def write_csvs(analysis: Path, losses: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    if losses:
        with (analysis / "step_losses.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(losses[0]))
            writer.writeheader()
            writer.writerows(losses)
    overall_fields = ["step", "sample_id", "teacher_label", "student_label", "inspector_label",
                      "student_same_as_teacher", "student_same_as_inspector", "teacher_same_as_inspector",
                      "teacher_A", "teacher_B", "teacher_Tie", "student_A", "student_B", "student_Tie",
                      "inspector_A", "inspector_B", "inspector_Tie", "teacher_confidence",
                      "student_confidence_on_teacher_label", "confidence_gap", "overall_kl_teacher_student",
                      "teacher_kl", "inspector_cross_entropy", "inspector_kl", "student_selected_metrics",
                      "teacher_selected_metrics", "augmentation_operation", "augmentation_applied",
                      "augmentation_not_applied_reason", "augmentation_strength", "original_instruction",
                      "student_instruction"]
    with (analysis / "samples_overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = dict(r)
            for who in ("teacher", "student", "inspector"):
                for c, v in zip(CLASSES, r[who]):
                    flat[f"{who}_{c}"] = v
            flat["augmentation_strength"] = json.dumps(r.get("augmentation_strength"))
            writer.writerow(flat)
    metric_fields = ["step", "sample_id", "metric", "teacher_label", "student_label", "student_same_as_teacher",
                     "teacher_A", "teacher_B", "teacher_Tie", "student_A", "student_B", "student_Tie",
                     "teacher_confidence", "student_confidence_on_teacher_label", "confidence_gap",
                     "kl_teacher_student", "cross_entropy_teacher_student", "teacher_relevance",
                     "student_relevance", "teacher_selected", "student_selected", "augmentation_applied"]
    with (analysis / "samples_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            for name in METRICS:
                m = dict(r["metrics"][name])
                for who in ("teacher", "student"):
                    for c, v in zip(CLASSES, m[who]):
                        m[f"{who}_{c}"] = v
                m.update(step=r["step"], sample_id=r["sample_id"], metric=name,
                         augmentation_applied=r.get("augmentation_applied"))
                writer.writerow(m)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    out = []
    for i in range(len(values)):
        part = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        out.append(sum(part) / len(part) if part else float("nan"))
    return out


def style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_losses(path: Path, title: str, series: List[Tuple[str, str, List[Dict[str, Any]]]], window: int,
                panels: Sequence[Tuple[str, str]] = LOSS_PANELS) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    for ax, (key, label) in zip(axes.flat, panels):
        for name, color, losses in series:
            x = [r["step"] for r in losses]
            y = [r.get(key) if r.get(key) is not None else float("nan") for r in losses]
            if key in NOT_SMOOTHED:  # a schedule, not noisy: exact value per step
                ax.plot(x, y, color=color, linewidth=2, label=name if len(series) > 1 else "value per step")
                continue
            single = len(series) == 1
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.3, label="value per step" if single else None)
            ax.plot(x, moving_average(y, window), color=color, linewidth=2,
                    label=f"moving average, {window} steps" if single else f"{name} (moving average, {window} steps)")
        ax.set_title(label, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("optimizer step", fontsize=8, color=INK_MUTED)
        style(ax)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metrics(path: Path, title: str, series: List[Tuple[str, str, Dict[str, Any]]]) -> None:
    groups = ["overall"] + list(METRICS)
    labels = ["overall"] + [m.replace("_consistency", "\nconsistency") for m in METRICS]
    panels = (
        ("student_same_as_teacher", "Student = Teacher (one-hot)", (0, 1)),
        ("teacher_confidence_mean", "Teacher confidence in its own label", (0, 1)),
        ("student_confidence_on_teacher_label_mean", "Student confidence in the Teacher label", (0, 1)),
        ("confidence_gap_mean", "Confidence gap = Teacher − Student (on the Teacher label)", None),
        ("kl_teacher_student_mean", "KL(Teacher‖Student)", None),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.1 * len(panels)))
    width = 0.8 / max(1, len(series))
    for ax, (key, label, limits) in zip(axes, panels):
        for k, (name, color, summary) in enumerate(series):
            values = [(summary["overall"] if g == "overall" else summary["metrics"][g]).get(key) for g in groups]
            xs = [i + (k - (len(series) - 1) / 2) * width for i in range(len(groups))]
            bars = ax.bar(xs, [v if v is not None else 0 for v in values], width=width * 0.92, color=color, label=name)
            for bar, v in zip(bars, values):
                if v is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center",
                            va="bottom", fontsize=7, color=INK_MUTED)
        if key == "student_same_as_teacher":
            ax.axhspan(*TARGET_AGREEMENT, color=GRID, alpha=0.6, zorder=0)
            ax.text(len(groups) - 0.5, TARGET_AGREEMENT[1], "target 75–80%", ha="right", va="bottom",
                    fontsize=7, color=INK_MUTED)
        if limits:
            ax.set_ylim(*limits)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(label, fontsize=10, loc="left", color=INK)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def transition_matrix(summary_part: Dict[str, Any]) -> List[List[float]]:
    """Row-normalised Teacher -> Student fractions (rows: Teacher label)."""
    counts = summary_part["transitions"]
    matrix = []
    for t in CLASSES:
        row = [counts.get(f"{t}->{s}", 0) for s in CLASSES]
        total = sum(row)
        matrix.append([v / total if total else float("nan") for v in row])
    return matrix


def plot_transitions(path: Path, title: str, operation: Optional[Tuple[str, Dict[str, Any]]],
                     baseline: Tuple[str, Dict[str, Any]]) -> None:
    groups = ["overall"] + list(METRICS)
    rows = [baseline] if operation is None else [operation, baseline, None]
    fig, axes = plt.subplots(len(rows), len(groups), figsize=(2.6 * len(groups), 2.7 * len(rows)), squeeze=False)
    for r, entry in enumerate(rows):
        for c, group in enumerate(groups):
            ax = axes[r][c]
            if entry is None:
                part_o = operation[1]["overall"] if group == "overall" else operation[1]["metrics"][group]
                part_b = baseline[1]["overall"] if group == "overall" else baseline[1]["metrics"][group]
                mo, mb = transition_matrix(part_o), transition_matrix(part_b)
                matrix = [[mo[i][j] - mb[i][j] for j in range(3)] for i in range(3)]
                image = ax.imshow(matrix, cmap=DIVERGING.reversed(), vmin=-0.3, vmax=0.3)
                name = "difference (run − baseline)"
            else:
                part = entry[1]["overall"] if group == "overall" else entry[1]["metrics"][group]
                matrix = transition_matrix(part)
                image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)
                name = entry[0]
            for i in range(3):
                for j in range(3):
                    v = matrix[i][j]
                    if v == v:
                        ax.text(j, i, f"{v:+.2f}" if entry is None else f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color=INK if entry is None or v < 0.6 else "#ffffff")
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"S={c}" for c in CLASSES], fontsize=7)
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"T={c}" for c in CLASSES], fontsize=7)
            if r == 0:
                ax.set_title(group.replace("_consistency", "\nconsistency"), fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=9, color=INK)
    fig.suptitle(title + " — each row: fraction from Teacher label (T) to Student label (S)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_ranking(path: Path, ranking: List[Dict[str, Any]], baseline_overall: Dict[str, Any]) -> None:
    ranking = sorted(ranking, key=lambda r: r["student_same_as_teacher"] if r["student_same_as_teacher"] is not None else 2)
    names = [r["operation"] for r in ranking]
    panels = (("student_same_as_teacher", "Student = Teacher (one-hot, trên mẫu được áp dụng)"),
              ("confidence_gap_mean", "Khoảng chênh độ tự tin Teacher − Student"),
              ("kl_teacher_student_mean", "KL(Teacher‖Student) overall"))
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 0.45 * len(names) + 2), sharey=True)
    for ax, (key, label) in zip(axes, panels):
        values = [r[key] if r[key] is not None else 0 for r in ranking]
        bars = ax.barh(range(len(names)), values, color=COLOR_OPERATION, height=0.7)
        reference = baseline_overall.get(key)
        if reference is not None:
            ax.axvline(reference, color=COLOR_BASELINE, linewidth=2, label=f"baseline = {reference:.3f}")
        if key == "student_same_as_teacher":
            ax.axvspan(*TARGET_AGREEMENT, color=GRID, alpha=0.6, zorder=0, label="mục tiêu 75–80%")
        for bar, v in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {v:.3f}", va="center", fontsize=8,
                    color=INK_MUTED)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(label, fontsize=10, loc="left", color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        style(ax)
    fig.suptitle("Xếp hạng các phép augmentation so với baseline", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- comparison
# Cross-operation tables (user, 2026-11-06): rows = overall + 6 metrics, columns = baseline and
# the operations; written to <baseline>/analysis/comparison/. Every operation uses only its
# APPLIED samples; "loss (selected metrics)" = per sample, mean over the metrics the Student
# relevance selects (the training KL term), then mean over samples.
LOSS_ROW = "loss (selected metrics)"
DIVERGENCES = {
    "kl_teacher_student": "KL(Teacher||Student) = sum T log(T/S); the loss used in training",
    "kl_student_teacher": "reverse KL(Student||Teacher) = sum S log(S/T)",
    "jensen_shannon": "Jensen-Shannon divergence of T and S (natural log)",
    "cross_entropy_soft": "-sum T log S = KL + entropy(T); the Teacher entropy is the same in every run",
    "bce_a_vs_b": "BCE between the A-vs-B shares T_A/(T_A+T_B) and S_A/(S_A+S_B) (only the A-vs-B part; Tie in bce_tie)",
    "kl_a_vs_b": "binary KL of the A-vs-B shares = bce_a_vs_b - binary entropy of the Teacher share",
    "bce_tie": "BCE between P_Teacher(Tie) and P_Student(Tie) (the Tie part; with bce_a_vs_b covers the 3 classes)",
    "kl_tie": "binary KL of P(Tie) = bce_tie - binary entropy of P_Teacher(Tie)",
    "cross_entropy_hard": "-log S(argmax label of the Teacher)",
}
# Illustrative thresholds (not agreed thresholds). A-vs-B version, Teacher label A or B: Teacher
# A-vs-B share of its label >= 0.75 and Student share <= 0.60. Three-class version (every label,
# Tie included): Teacher probability of its label >= 0.75 and Student probability of that label <= 0.50.
TEACHER_CERTAIN_SHARE = 0.75
STUDENT_NEAR_RANDOM_SHARE = 0.60
TEACHER_CERTAIN_THREE_CLASS = 0.75
STUDENT_UNCERTAIN_THREE_CLASS = 0.50
BOOTSTRAP_RESAMPLES = 1000
EPS = 1e-8


def divergences(teacher: Sequence[float], student: Sequence[float], teacher_label: str) -> Dict[str, float]:
    t = [max(float(v), EPS) for v in teacher]
    s = [max(float(v), EPS) for v in student]
    m = [(a + b) / 2 for a, b in zip(t, s)]
    ta = t[0] / (t[0] + t[1])
    sa = min(max(s[0] / (s[0] + s[1]), EPS), 1 - EPS)
    bce = -(ta * math.log(sa) + (1 - ta) * math.log(1 - sa))
    binary_entropy = -(ta * math.log(ta) + (1 - ta) * math.log(1 - ta)) if 0 < ta < 1 else 0.0
    tt, st = min(max(t[2], EPS), 1 - EPS), min(max(s[2], EPS), 1 - EPS)
    bce_tie = -(tt * math.log(st) + (1 - tt) * math.log(1 - st))
    tie_entropy = -(tt * math.log(tt) + (1 - tt) * math.log(1 - tt))
    return {
        "kl_teacher_student": sum(a * math.log(a / b) for a, b in zip(t, s)),
        "kl_student_teacher": sum(b * math.log(b / a) for a, b in zip(t, s)),
        "jensen_shannon": 0.5 * sum(a * math.log(a / c) for a, c in zip(t, m)) + 0.5 * sum(b * math.log(b / c) for b, c in zip(s, m)),
        "cross_entropy_soft": -sum(a * math.log(b) for a, b in zip(t, s)),
        "bce_a_vs_b": bce,
        "kl_a_vs_b": bce - binary_entropy,
        "bce_tie": bce_tie,
        "kl_tie": bce_tie - tie_entropy,
        "cross_entropy_hard": -math.log(s[CLASSES.index(teacher_label)]),
    }


def ab_share(probabilities: Sequence[float], label: str) -> Optional[float]:
    if label == "Tie":
        return None
    a, b = float(probabilities[0]), float(probabilities[1])
    return float(probabilities[CLASSES.index(label)]) / (a + b) if a + b > 0 else None


def sample_values(row: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per group (overall + 6 metrics): agreement, confidences, gap, divergences, near-random flags."""
    groups = {"overall": {"teacher": row["teacher"], "student": row["student"], "teacher_label": row["teacher_label"],
                          "student_label": row["student_label"]}}
    for name in METRICS:
        groups[name] = row["metrics"][name]
    out = {}
    for name, g in groups.items():
        label = g["teacher_label"]
        index = CLASSES.index(label)
        values = {
            "student_same_as_teacher": float(g["student_label"] == label),
            "teacher_confidence": float(g["teacher"][index]),
            "student_confidence_on_teacher_label": float(g["student"][index]),
            "confidence_gap": float(g["teacher"][index]) - float(g["student"][index]),
            "confidence_gap_at_least_0.2": float(float(g["teacher"][index]) - float(g["student"][index]) >= 0.2),
            **divergences(g["teacher"], g["student"], label),
        }
        tc, sc = float(g["teacher"][index]), float(g["student"][index])
        uncertain = float(tc >= TEACHER_CERTAIN_THREE_CLASS and sc <= STUDENT_UNCERTAIN_THREE_CLASS)
        values["teacher_certain_student_uncertain_three_class"] = uncertain
        values["teacher_certain_student_uncertain_three_class_same_label"] = uncertain * values["student_same_as_teacher"]
        values["teacher_label_tie"] = float(label == "Tie")
        values["student_label_tie"] = float(g["student_label"] == "Tie")
        values["teacher_a_or_b_student_tie"] = float(label != "Tie" and g["student_label"] == "Tie")
        values["teacher_tie_student_a_or_b"] = float(label == "Tie" and g["student_label"] != "Tie")
        values["student_tie_probability"] = float(g["student"][2])
        values["teacher_tie_probability"] = float(g["teacher"][2])
        ts, ss = ab_share(g["teacher"], label), ab_share(g["student"], label)
        if ts is not None:
            near = float(ts >= TEACHER_CERTAIN_SHARE and ss <= STUDENT_NEAR_RANDOM_SHARE)
            values["teacher_certain_student_near_random"] = near
            values["teacher_certain_student_near_random_same_label"] = near * values["student_same_as_teacher"]
        else:  # Teacher label Tie: no A-vs-B share; counted as 0 so every row uses all samples
            values["teacher_certain_student_near_random"] = 0.0
            values["teacher_certain_student_near_random_same_label"] = 0.0
        out[name] = values
    selected = [n for n in METRICS if row["metrics"][n]["student_selected"]] or list(METRICS)
    out[LOSS_ROW] = {key: sum(out[n][key] for n in selected) / len(selected) for key in DIVERGENCES}
    return out


QUANTITIES = (
    ("student_same_as_teacher", "Student = Teacher, one-hot label"),
    ("teacher_confidence", "Teacher confidence in its own label"),
    ("student_confidence_on_teacher_label", "Student (augmented) confidence in the Teacher label"),
    ("confidence_gap", "confidence gap = Teacher - Student, on the Teacher label"),
    ("confidence_gap_at_least_0.2", "fraction of samples with confidence gap >= 0.2"),
    ("teacher_certain_student_uncertain_three_class",
     f"fraction (every label, Tie included): Teacher P(its label) >= {TEACHER_CERTAIN_THREE_CLASS}, "
     f"Student P(that label) <= {STUDENT_UNCERTAIN_THREE_CLASS}"),
    ("teacher_certain_student_uncertain_three_class_same_label", "same as above and the Student keeps the Teacher label"),
    ("teacher_certain_student_near_random",
     f"fraction of ALL samples with Teacher label A/B and Teacher A-vs-B share >= {TEACHER_CERTAIN_SHARE}, "
     f"Student share <= {STUDENT_NEAR_RANDOM_SHARE} (samples with Teacher label Tie count as 0)"),
    ("teacher_certain_student_near_random_same_label", "same as above and the Student keeps the Teacher label"),
    ("teacher_label_tie", "fraction of samples whose Teacher label is Tie"),
    ("student_label_tie", "fraction of samples whose Student label is Tie"),
    ("teacher_a_or_b_student_tie", "fraction: Teacher label A or B, Student label Tie"),
    ("teacher_tie_student_a_or_b", "fraction: Teacher label Tie, Student label A or B"),
    ("teacher_tie_probability", "mean P_Teacher(Tie)"),
    ("student_tie_probability", "mean P_Student(Tie)"),
) + tuple((key, text) for key, text in DIVERGENCES.items())


def write_comparison(data: Dict[str, Dict[str, Any]], out_dir: Path, columns: Optional[Sequence[str]] = None,
                     heading: str = "Single-operation augmentation: operations compared with the baseline") -> None:
    out_dir.mkdir(exist_ok=True)
    operations = ["baseline"] + [op for op in (columns if columns is not None else OPERATIONS) if op in data]
    per_sample = {op: {r["sample_id"]: sample_values(r) for r in data[op]["applied_rows"]} for op in operations}
    rows_of = ["overall"] + list(METRICS)

    def mean_of(op: str, group: str, key: str) -> Optional[float]:
        values = [v[group][key] for v in per_sample[op].values() if group in v and key in v[group]]
        return sum(values) / len(values) if values else None

    tables: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for key, _ in QUANTITIES:
        groups = ([LOSS_ROW] if key in DIVERGENCES else []) + rows_of
        tables[key] = {g: {op: mean_of(op, g, key) for op in operations} for g in groups}
    md = [f"# {heading}", "",
          "Rows: overall and the 6 metrics; columns: baseline and the operations (applied samples only). "
          f"Applied samples: " + ", ".join(f"{op} {len(per_sample[op])}" for op in operations) + ".", ""]
    for key, text in QUANTITIES:
        table = tables[key]
        with (out_dir / f"{key}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row"] + operations + [f"{op}_minus_baseline" for op in operations[1:]]
                            + [f"{op}_over_baseline" for op in operations[1:]])
            for group, values in table.items():
                b = values["baseline"]
                writer.writerow([group] + [None if values[op] is None else round(values[op], 6) for op in operations]
                                + [None if values[op] is None or b is None else round(values[op] - b, 6) for op in operations[1:]]
                                + [None if values[op] is None or not b else round(values[op] / b, 4) for op in operations[1:]])
        ratio = key in DIVERGENCES
        md += [f"## {key}", "", text + (" — value (times baseline)" if ratio else " — value (difference from baseline)"), "",
               "| row | " + " | ".join(operations) + " |", "|---" * (len(operations) + 1) + "|"]
        for group, values in table.items():
            b = values["baseline"]
            cells = []
            for op in operations:
                v = values[op]
                if v is None:
                    cells.append("—")
                elif op == "baseline" or b is None:
                    cells.append(f"{v:.3f}")
                elif ratio:
                    cells.append(f"{v:.3f} (×{v / b:.1f})" if b else f"{v:.3f}")
                else:
                    cells.append(f"{v:.3f} ({v - b:+.3f})")
            md.append(f"| {group} | " + " | ".join(cells) + " |")
        md.append("")
    # Paired bootstrap of the difference from the baseline on the same samples (overall + metrics).
    rng = random.Random(0)
    boot_rows = []
    for op in operations[1:]:
        ids = [i for i in per_sample[op] if i in per_sample["baseline"]]
        for group in [LOSS_ROW] + rows_of:
            for key in ("student_same_as_teacher", "confidence_gap", "kl_teacher_student"):
                if key not in per_sample[op][ids[0]].get(group, {}):
                    continue
                diffs = [per_sample[op][i][group][key] - per_sample["baseline"][i][group][key] for i in ids]
                means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP_RESAMPLES))
                boot_rows.append({"operation": op, "row": group, "quantity": key, "samples": len(ids),
                                  "difference": round(sum(diffs) / len(diffs), 6),
                                  "ci95_low": round(means[int(0.025 * BOOTSTRAP_RESAMPLES)], 6),
                                  "ci95_high": round(means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1], 6)})
    with (out_dir / "difference_from_baseline_bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boot_rows[0]))
        writer.writeheader()
        writer.writerows(boot_rows)
    md += ["## Difference from the baseline on the same samples (overall), 95% paired bootstrap", "",
           "| operation | samples | Δ Student = Teacher | Δ confidence gap | Δ KL(Teacher‖Student) |", "|---|---|---|---|---|"]
    for op in operations[1:]:
        cell = {r["quantity"]: r for r in boot_rows if r["operation"] == op and r["row"] == "overall"}
        fmt = lambda r: f"{r['difference']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"  # noqa: E731
        md.append(f"| {op} | {cell['student_same_as_teacher']['samples']} | {fmt(cell['student_same_as_teacher'])} | "
                  f"{fmt(cell['confidence_gap'])} | {fmt(cell['kl_teacher_student'])} |")
    (out_dir / "comparison_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # Heatmaps: one per divergence, cell = times the baseline.
    for key in DIVERGENCES:
        table = tables[key]
        groups = list(table)
        matrix = [[(table[g][op] / table[g]["baseline"]) if table[g][op] is not None and table[g]["baseline"] else float("nan")
                   for op in operations[1:]] for g in groups]
        fig, ax = plt.subplots(figsize=(1.05 * len(operations) + 3, 0.55 * len(groups) + 1.8))
        image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=1.0, vmax=max(2.0, max(max(r) for r in matrix)), aspect="auto")
        for i, g in enumerate(groups):
            for j, op in enumerate(operations[1:]):
                v = matrix[i][j]
                ax.text(j, i, f"×{v:.1f}\n{table[g][op]:.3f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 3.5 else INK)
        ax.set_xticks(range(len(operations) - 1))
        ax.set_xticklabels(operations[1:], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([f"{g} (baseline {table[g]['baseline']:.3f})" for g in groups], fontsize=8)
        ax.set_title(f"{key}: times the baseline (value below)\n{DIVERGENCES[key]}", fontsize=9, loc="left", color=INK)
        fig.colorbar(image, ax=ax, fraction=0.025)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{key}.png", dpi=130)
        plt.close(fig)



# ----------------------------------------------------------------------------- combined round

def phase_label(low: int, high: int) -> str:
    return f"step {low}-{high}"


def by_phase(rows: Sequence[Dict[str, Any]], losses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Student = Teacher and the main loss / gradient numbers per step phase."""
    out = {}
    for low, high in STEP_PHASES:
        part = [r for r in rows if low <= int(r["step"]) <= high]
        steps = [r for r in losses if low <= int(r["step"]) <= high]
        out[phase_label(low, high)] = {
            "samples": len(part),
            "student_same_as_teacher": mean([float(r["student_same_as_teacher"]) for r in part]),
            "confidence_gap_mean": mean([r["confidence_gap"] for r in part]),
            "student_same_as_inspector": mean([float(r["student_same_as_inspector"]) for r in part]),
            **{key: mean([r.get(key) for r in steps]) for key in (
                "unlabeled_teacher_kl", "unlabeled_inspector_kl", "unlabeled_learnable", "total_learnable",
                "gradient_norm_before_clipping", "gradient_unlabeled_over_labeled",
                "gradient_cosine_labeled_unlabeled")},
        }
    return out


def paired_bootstrap(first: Dict[str, Dict[str, Any]], second: Dict[str, Dict[str, Any]], key: str,
                     rng: random.Random) -> Optional[Dict[str, Any]]:
    """Mean of first - second on the same sample ids, overall row, 95% bootstrap interval."""
    ids = [i for i in first if i in second and key in first[i]["overall"] and key in second[i]["overall"]]
    if not ids:
        return None
    diffs = [first[i]["overall"][key] - second[i]["overall"][key] for i in ids]
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP_RESAMPLES))
    return {"samples": len(ids), "difference": round(sum(diffs) / len(diffs), 6),
            "ci95_low": round(means[int(0.025 * BOOTSTRAP_RESAMPLES)], 6),
            "ci95_high": round(means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1], 6)}


def write_summary_tables(out_dir: Path, single_dir: Path, runs: Sequence[str],
                         phases: Dict[str, Dict[str, Any]]) -> None:
    """summary_tables.md (2026-11-07): the tables of the single-operation report, for this round.

    Read from the CSVs that write_comparison just wrote in out_dir; the single-operation
    reference columns come from <baseline>/analysis/comparison/ (their run folders may be gone).
    Rows = overall + 6 metrics (+ the training loss row for KL); columns = baseline, the runs,
    then the single operations as reference.
    """
    def load(directory: Path, key: str) -> Dict[str, Dict[str, str]]:
        path = directory / f"{key}.csv"
        if not path.is_file():
            return {}
        return {row["row"]: row for row in csv.DictReader(path.open(encoding="utf-8"))}

    singles = [label.replace("augmentation_", "", 1) for label in SINGLE_REFERENCE_RUNS]
    columns = ["baseline"] + list(runs) + [f"{name} (phép đơn)" for name in singles]

    def value(key: str, group: str, column: str) -> Optional[float]:
        if column.endswith(" (phép đơn)"):
            table, name = load(single_dir, key), column[: -len(" (phép đơn)")]
        else:
            table, name = load(out_dir, key), column
        cell = table.get(group, {}).get(name)
        return None if cell in (None, "") else float(cell)

    def fmt(v: Optional[float], digits: int = 3) -> str:
        return "—" if v is None else f"{v:.{digits}f}"

    groups = ["overall"] + list(METRICS)
    header = "| | " + " | ".join(columns) + " |"
    rule = "|---" * (len(columns) + 1) + "|"
    md = ["# Combined operations and Inspector one-hot: summary tables", "",
          "Written by analyze_single_operation_augmentation.py --group combined. Columns: baseline, the runs of "
          "this round, then single operations as reference (read from the single-operation comparison CSVs; "
          "their step phases are not available). Every value is a mean over the applied Unlabeled samples "
          "of the run (all 126 steps).", ""]

    md += ["## 1. Student = Teacher (one-hot), by step phase, and same label with a near-random Student", "",
           "Near-random Student: Teacher label A or B with Teacher A-vs-B share >= "
           f"{TEACHER_CERTAIN_SHARE}, Student keeps the label with share <= {STUDENT_NEAR_RANDOM_SHARE} "
           "(fraction of all samples).", "",
           "| run | Student = Teacher | difference from baseline | "
           + " | ".join(phase_label(*p) for p in STEP_PHASES) + " | same label, Student near random |",
           "|---" * (len(STEP_PHASES) + 4) + "|"]
    base_agreement = value("student_same_as_teacher", "overall", "baseline")
    for column in columns:
        agreement = value("student_same_as_teacher", "overall", column)
        phase = phases.get(column, {})
        cells = [fmt(phase.get(phase_label(*p), {}).get("student_same_as_teacher")) for p in STEP_PHASES]
        near = value("teacher_certain_student_near_random_same_label", "overall", column)
        diff = None if agreement is None or base_agreement is None else agreement - base_agreement
        md.append(f"| {column} | {fmt(agreement)} | {'—' if column == 'baseline' or diff is None else f'{diff:+.3f}'} | "
                  + " | ".join(cells) + f" | {'—' if near is None else f'{100 * near:.1f}%'} |")
    md.append("")

    md += ["## 2. Student = Teacher (one-hot) per metric", "", header, rule]
    for group in groups:
        md.append(f"| {group} | " + " | ".join(fmt(value("student_same_as_teacher", group, c)) for c in columns) + " |")
    md.append("")

    md += ["## 3. Student confidence in the Teacher label / confidence gap (gap = Teacher − Student)", "",
           "Last column: Teacher confidence in its own label (baseline run; the Teacher always sees the "
           "original sample).", "", header[:-1] + "| Teacher |", rule + "---|"]
    for group in groups:
        cells = []
        for column in columns:
            student = value("student_confidence_on_teacher_label", group, column)
            gap = value("confidence_gap", group, column)
            cells.append("—" if student is None or gap is None else f"{student:.3f} / {gap:.3f}")
        md.append(f"| {group} | " + " | ".join(cells) + f" | {fmt(value('teacher_confidence', group, 'baseline'))} |")
    md.append("")

    md += ["## 4. KL(Teacher‖Student) per metric, value (times the baseline)", "",
           f"Three classes, Tie included. Row \"{LOSS_ROW}\" = the training loss: per sample, mean over the "
           "metrics the Student relevance selects, then mean over samples. Row overall = KL of the aggregated "
           "overall distributions (reference only).", "", header, rule]
    for group in [LOSS_ROW] + groups:
        base = value("kl_teacher_student", group, "baseline")
        cells = []
        for column in columns:
            v = value("kl_teacher_student", group, column)
            if v is None:
                cells.append("—")
            elif column == "baseline" or not base:
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.3f} (×{v / base:.1f})")
        md.append(f"| {group} | " + " | ".join(cells) + " |")
    md.append("")
    (out_dir / "summary_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def plot_run(analysis: Path, name: str, entry: Dict[str, Any], window: int) -> None:
    """The 5 per-run figures (English, this run only; user 2026-11-07)."""
    own = [(name, COLOR_OPERATION, entry["losses"])]
    head = f"{name} (job {entry['job']})"
    plot_losses(analysis / "losses.png", head + ": losses per optimizer step", own, window)
    plot_losses(analysis / "learnable_loss.png", head + ": learnable part of the losses (floor removed)", own,
                window, LEARNABLE_PANELS)
    plot_losses(analysis / "training_dynamics.png", head + ": gradients and learning rate", own, window,
                DYNAMICS_PANELS)
    for old in ("loss_vs_baseline.png", "learnable_loss_vs_baseline.png", "training_dynamics_vs_baseline.png",
                "metrics_vs_baseline.png", "loss_per_step.png"):
        (analysis / old).unlink(missing_ok=True)  # replaced by the per-run plots
    samples = "augmented" if name != "baseline" else "(no augmentation)"
    plot_metrics(analysis / "metrics.png",
                 head + f": Student vs Teacher on {len(entry['applied_rows'])} {samples} Unlabeled samples",
                 [(name, COLOR_OPERATION, entry["applied"])])
    plot_transitions(analysis / "label_transitions.png", head, None, (name, entry["applied"]))


def main_combined(args: argparse.Namespace) -> None:
    """Combined operations + Inspector one-hot (stage2_tuning.md 8.5).

    Writes analysis/ in the runs of COMBINED_RUNS and the 5 per-run figures (+ csv) of the baseline
    (2026-11-07; the baseline summary.json is kept). Single-operation runs are READ ONLY.
    --plots-only stops after the per-run analysis/ (combined_operations/ not rewritten).
    Cross-run tables go to <baseline>/analysis/combined_operations/ (new folder).
    """
    root = Path(args.output_root)
    label_of = lambda name: name.replace("augmentation_", "", 1)  # noqa: E731
    runs = {"baseline": root / f"{args.prefix}_baseline{args.suffix}"}
    runs.update({label_of(n): root / f"{args.prefix}_{n}{args.suffix}" for n in SINGLE_REFERENCE_RUNS})
    runs.update({label_of(n): root / f"{args.prefix}_{n}{args.suffix}" for n in COMBINED_RUNS})
    combined = [label_of(n) for n in COMBINED_RUNS]
    if args.only:  # restrict the analysed runs (baseline is always read)
        unknown = [n for n in args.only if n not in combined]
        if unknown:
            raise SystemExit(f"--only: not in COMBINED_RUNS: {unknown}")
        combined = [n for n in combined if n in args.only]
        runs = {k: v for k, v in runs.items() if k == "baseline" or k in combined}
    counterpart = {label_of(k): label_of(v) for k, v in SOFT_COUNTERPART.items()}

    data: Dict[str, Dict[str, Any]] = {}
    for name, run in runs.items():
        if not run.is_dir():
            print(f"skip {name}: no folder {run}")
            continue
        losses, job = step_losses(run)
        rows = unlabeled_rows(run)
        if not losses or not rows:
            print(f"skip {name}: no step log or no per-sample rows in {run}")
            continue
        if name in combined and len(losses) < 126 and not args.suffix:
            print(f"skip {name}: still running ({len(losses)} steps)")
            continue
        applied = rows if name.startswith("baseline") else [r for r in rows if r.get("augmentation_applied")]
        data[name] = {"run": run, "losses": losses, "rows": rows, "applied_rows": applied, "job": job,
                      "applied": block(applied)}
        print(f"{name}: job {job}, {len(losses)} steps, {len(rows)} Unlabeled rows, applied {len(applied)}"
              + ("" if name in combined else " (read only)"))
    if "baseline" not in data:
        raise SystemExit("The baseline run has no usable logs; nothing is compared.")
    base = data["baseline"]
    window = args.moving_average
    ranking = []
    for name in combined:
        if name not in data:
            continue
        entry = data[name]
        analysis = entry["run"] / "analysis"
        analysis.mkdir(exist_ok=True)
        write_csvs(analysis, entry["losses"], entry["rows"])
        series = [(name, COLOR_OPERATION, entry["losses"]), ("baseline", COLOR_BASELINE, base["losses"])]
        metric_series = [(name, COLOR_OPERATION, entry["applied"]), ("baseline", COLOR_BASELINE, base["applied"])]
        soft = counterpart.get(name)
        if soft and soft != "baseline" and soft in data:
            series.append((f"{soft} (Inspector mềm)", COLOR_COUNTERPART, data[soft]["losses"]))
            metric_series.append((f"{soft} (Inspector mềm)", COLOR_COUNTERPART, data[soft]["applied"]))
        title = f"{name} (job {entry['job']}) so với baseline (job {base['job']})"
        if len(series) == 3:
            title += f" và {soft} (job {data[soft]['job']})"
        plot_run(analysis, name, entry, window)
        o, b = entry["applied"]["overall"], base["applied"]["overall"]
        delta = lambda x, y, key: None if x.get(key) is None or y.get(key) is None else round(x[key] - y[key], 6)  # noqa: E731
        keys = ("student_same_as_teacher", "teacher_confidence_mean", "student_confidence_on_teacher_label_mean",
                "confidence_gap_mean", "kl_teacher_student_mean", "teacher_kl_loss_mean",
                "inspector_cross_entropy_mean", "student_same_as_inspector")
        summary = {
            "run": str(entry["run"]), "job": entry["job"], "operation": name, "steps": len(entry["losses"]),
            "unlabeled_rows": len(entry["rows"]), "applied_rows": len(entry["applied_rows"]),
            "applied_operations": dict(Counter("+".join(r.get("augmentation_applied_operations") or [])
                                               for r in entry["rows"])),
            "inspector_target": (entry["rows"][0].get("inspector_target") or "soft"),
            "applied": entry["applied"], "all_rows": block(entry["rows"]),
            "by_step_phase": by_phase(entry["applied_rows"], entry["losses"]),
            "losses_mean": {key: mean([r.get(key) for r in entry["losses"]])
                            for key, _ in LOSS_PANELS + LEARNABLE_PANELS + DYNAMICS_PANELS},
            "difference_from_baseline": {key: delta(o, b, key) for key in keys},
            "difference_from_baseline_per_metric": {
                metric: {key: delta(entry["applied"]["metrics"][metric], base["applied"]["metrics"][metric], key)
                         for key in ("student_same_as_teacher", "confidence_gap_mean", "kl_teacher_student_mean")}
                for metric in METRICS},
        }
        if soft and soft in data:
            summary["soft_inspector_counterpart"] = soft
            summary["difference_from_soft_counterpart"] = {
                key: delta(o, data[soft]["applied"]["overall"], key) for key in keys}
        (analysis / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        agreement = o.get("student_same_as_teacher")
        ranking.append({
            "operation": name, "job": entry["job"], "inspector_target": summary["inspector_target"],
            "applied_rows": len(entry["applied_rows"]), "student_same_as_teacher": agreement,
            "student_same_as_teacher_minus_baseline": delta(o, b, "student_same_as_teacher"),
            "confidence_gap_mean": o.get("confidence_gap_mean"),
            "confidence_gap_minus_baseline": delta(o, b, "confidence_gap_mean"),
            "kl_teacher_student_mean": o.get("kl_teacher_student_mean"),
            "kl_minus_baseline": delta(o, b, "kl_teacher_student_mean"),
            "teacher_kl_loss_mean": o.get("teacher_kl_loss_mean"),
            "student_same_as_inspector": o.get("student_same_as_inspector"),
            "in_target_75_80": agreement is not None and TARGET_AGREEMENT[0] <= agreement <= TARGET_AGREEMENT[1],
            **{f"student_same_as_teacher_{phase_label(*p).replace(' ', '_')}":
               summary["by_step_phase"][phase_label(*p)]["student_same_as_teacher"] for p in STEP_PHASES},
            **{f"{metric}_same_as_teacher": entry["applied"]["metrics"][metric]["student_same_as_teacher"]
               for metric in METRICS},
        })

    # Baseline: the same 5 per-run figures (2026-11-07). Its summary.json (single-operation round) is kept.
    base_analysis = base["run"] / "analysis"
    base_analysis.mkdir(exist_ok=True)
    write_csvs(base_analysis, base["losses"], base["rows"])
    plot_run(base_analysis, "baseline", base, window)
    print(f"baseline: plots written in {base_analysis}")
    if args.plots_only:
        print(f"--plots-only: analysis/ written in {len(ranking)} runs; combined_operations/ not touched")
        return
    out_dir = base["run"] / "analysis" / "combined_operations"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Reference rows of the read-only runs (baseline, single operations) for the ranking plot.
    for name in ["baseline"] + [label_of(n) for n in SINGLE_REFERENCE_RUNS]:
        if name in data and name != "baseline":
            o = data[name]["applied"]["overall"]
            ranking.append({"operation": f"{name} (phép đơn)", "job": data[name]["job"],
                            "student_same_as_teacher": o.get("student_same_as_teacher"),
                            "confidence_gap_mean": o.get("confidence_gap_mean"),
                            "kl_teacher_student_mean": o.get("kl_teacher_student_mean")})
    if ranking:
        ranking.sort(key=lambda r: (r["student_same_as_teacher"] is None, r["student_same_as_teacher"]))
        fields = list(dict.fromkeys(k for r in ranking for k in r))
        with (out_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ranking)
        plot_ranking(out_dir / "ranking.png", ranking, base["applied"]["overall"])
    columns = [n for n in combined if n in data] + [label_of(n) for n in SINGLE_REFERENCE_RUNS if label_of(n) in data]
    write_comparison(data, out_dir, columns,
                     "Combined augmentation operations and Inspector one-hot compared with the baseline "
                     "(single operations = reference columns)")
    # Extra sections: step phases, one-hot vs soft counterpart, loss / gradient means.
    md = ["## Student = Teacher (one-hot) by step phase", "",
          "| run | " + " | ".join(phase_label(*p) for p in STEP_PHASES) + " | all steps |",
          "|---" * (len(STEP_PHASES) + 2) + "|"]
    for name in ["baseline"] + columns:
        phases = by_phase(data[name]["applied_rows"], data[name]["losses"])
        cells = [phases[phase_label(*p)]["student_same_as_teacher"] for p in STEP_PHASES]
        overall = data[name]["applied"]["overall"]["student_same_as_teacher"]
        md.append(f"| {name} | " + " | ".join("—" if c is None else f"{c:.3f}" for c in cells) + f" | {overall:.3f} |")
    md += ["", "## Inspector one-hot minus its soft-Inspector counterpart (same samples, overall, 95% paired bootstrap)",
           "", "| one-hot run | counterpart | samples | Δ Student = Teacher | Δ Student = Inspector | Δ confidence gap "
           "| Δ KL(Teacher‖Student) |", "|---|---|---|---|---|---|---|"]
    rng = random.Random(0)
    for name, soft in counterpart.items():
        if name not in data or soft not in data:
            continue
        first = {r["sample_id"]: {"overall": {**sample_values(r)["overall"],
                                              "student_same_as_inspector": float(r["student_same_as_inspector"])}}
                 for r in data[name]["applied_rows"]}
        second = {r["sample_id"]: {"overall": {**sample_values(r)["overall"],
                                               "student_same_as_inspector": float(r["student_same_as_inspector"])}}
                  for r in data[soft]["applied_rows"]}
        cells = []
        for key in ("student_same_as_teacher", "student_same_as_inspector", "confidence_gap", "kl_teacher_student"):
            res = paired_bootstrap(first, second, key, rng)
            cells.append("—" if res is None else
                         f"{res['difference']:+.3f} [{res['ci95_low']:+.3f}, {res['ci95_high']:+.3f}]")
        samples = len([i for i in first if i in second])
        md.append(f"| {name} | {soft} | {samples} | " + " | ".join(cells) + " |")
    md += ["", "## Mean over steps: losses (learnable part) and gradients", "",
           "| run | total learnable | Unlabeled learnable | KL(Inspector‖S) | KL Teacher | floor | gradient norm "
           "| Unlabeled / labeled | cosine |", "|---" * 9 + "|"]
    dyn_rows = []
    for name in ["baseline"] + columns:
        losses = data[name]["losses"]
        values = {key: mean([r.get(key) for r in losses]) for key in (
            "total_learnable", "unlabeled_learnable", "unlabeled_inspector_kl", "unlabeled_teacher_kl",
            "unlabeled_floor_inspector_entropy", "gradient_norm_before_clipping", "gradient_unlabeled_over_labeled",
            "gradient_cosine_labeled_unlabeled", "human", "gemma", "mask", "total")}
        dyn_rows.append({"run": name, "job": data[name]["job"], **values})
        md.append(f"| {name} | " + " | ".join("—" if values[k] is None else f"{values[k]:.3f}" for k in (
            "total_learnable", "unlabeled_learnable", "unlabeled_inspector_kl", "unlabeled_teacher_kl",
            "unlabeled_floor_inspector_entropy", "gradient_norm_before_clipping", "gradient_unlabeled_over_labeled",
            "gradient_cosine_labeled_unlabeled")) + " |")
    with (out_dir / "loss_and_gradient_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dyn_rows[0]))
        writer.writeheader()
        writer.writerows(dyn_rows)
    tables = out_dir / "comparison_tables.md"
    tables.write_text(tables.read_text(encoding="utf-8") + "\n" + "\n".join(md) + "\n", encoding="utf-8")
    write_summary_tables(out_dir, base["run"] / "analysis" / "comparison", [n for n in combined if n in data],
                         {name: by_phase(data[name]["applied_rows"], data[name]["losses"])
                          for name in ["baseline"] + [n for n in combined if n in data]})
    print(f"Written analysis/ in {len([n for n in combined if n in data])} combined runs; tables in {out_dir}")

def plot_stage1(run: Path) -> None:
    """Stage 1 Student (MS-Swift) losses per optimizer step, same style as the Stage 2 per-run plots.

    MS-Swift logs only the total loss, grad_norm and learning rate, every logging_steps steps
    (each loss value = mean over that window). Stage 1 loss = L_tie + L_winner + L_relevance
    (GemmaGT majority vote) = L_preference + L_relevance in the Stage 2 names. Writes
    <run>/analysis/losses.png and step_losses.csv.
    """
    path = run / "logging.jsonl"
    rows = [r for r in read_jsonl(path) if "loss" in r and "global_step/max_steps" in r]
    if not rows:
        raise SystemExit(f"No step rows in {path}")
    steps = [{"step": int(str(r["global_step/max_steps"]).split("/")[0]), "loss": float(r["loss"]),
              "grad_norm": float(r["grad_norm"]), "learning_rate": float(r["learning_rate"])} for r in rows]
    window = json.loads((run / "args.json").read_text(encoding="utf-8")).get("logging_steps")
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "step_losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)
    panels = (("loss", "L_stage1 = L_preference + L_relevance (GemmaGT)"),
              ("grad_norm", "Gradient norm before clipping (whole model, max_norm 1.0)"),
              ("learning_rate", "Learning rate"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    x = [r["step"] for r in steps]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(x, [r[key] for r in steps], color=COLOR_OPERATION, linewidth=2, marker="o", markersize=3,
                label=f"logged every {window} steps (loss = mean of the window)")
        ax.set_title(label, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("optimizer step", fontsize=8, color=INK_MUTED)
        style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{run.name}: Stage 1 Student losses per optimizer step", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(analysis / "losses.png", dpi=130)
    plt.close(fig)
    print(f"Stage 1: {len(steps)} logged steps; written {analysis / 'losses.png'}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--suffix", default="", help="e.g. _test_2_steps for the code-check runs")
    parser.add_argument("--moving-average", type=int, default=8)
    parser.add_argument("--group", choices=("single", "combined"), default="single",
                        help="single = baseline + 11 single operations (writes the baseline analysis); "
                             "combined = combined operations + Inspector one-hot (baseline and single runs read only)")
    parser.add_argument("--stage1-run", default="",
                        help="Stage 1 Student run folder (MS-Swift logging.jsonl): only plot its losses")
    parser.add_argument("--plots-only", action="store_true",
                        help="--group combined: write only the per-run analysis/ (plots, csv, summary.json); "
                             "do not rewrite <baseline>/analysis/combined_operations/")
    parser.add_argument("--only", nargs="+", default=[],
                        help="--group combined: analyse only these runs (names without 'augmentation_')")
    args = parser.parse_args()
    if args.stage1_run:
        plot_stage1(Path(args.stage1_run))
        return
    if args.group == "combined":
        main_combined(args)
        return

    root = Path(args.output_root)
    baseline_run = root / f"{args.prefix}_baseline{args.suffix}"
    runs = {"baseline": baseline_run}
    runs.update({op: root / f"{args.prefix}_augmentation_{op}{args.suffix}" for op in OPERATIONS})

    data: Dict[str, Dict[str, Any]] = {}
    for name, run in runs.items():
        if not run.is_dir():
            print(f"skip {name}: no folder {run}")
            continue
        losses, job = step_losses(run)
        rows = unlabeled_rows(run)
        if not losses or not rows:
            print(f"skip {name}: no step log or no per-sample rows with metric detail in {run}")
            continue
        applied = rows if name == "baseline" else [r for r in rows if r.get("augmentation_applied")]
        analysis = run / "analysis"
        analysis.mkdir(exist_ok=True)
        write_csvs(analysis, losses, rows)
        summary = {
            "run": str(run), "job": job, "operation": name, "steps": len(losses),
            "unlabeled_rows": len(rows), "applied_rows": len(applied),
            "applied_fraction": round(len(applied) / len(rows), 6),
            "not_applied_reasons": dict(Counter(r.get("augmentation_not_applied_reason") for r in rows
                                                if name != "baseline" and not r.get("augmentation_applied"))),
            "applied": block(applied),
            "all_rows": block(rows),
            "by_strength": strength_bins(name, applied),
            "losses_mean": {key: mean([r[key] for r in losses]) for key, _ in LOSS_PANELS},
        }
        data[name] = {"run": run, "analysis": analysis, "losses": losses, "summary": summary, "applied_rows": applied}
        print(f"{name}: job {job}, {len(losses)} steps, {len(rows)} Unlabeled rows, applied {len(applied)}")

    if "baseline" not in data:
        raise SystemExit("The baseline run has no usable logs; nothing is compared.")
    base = data["baseline"]
    base_label = "baseline"
    base_applied = base["summary"]["applied"]

    ranking = []
    for name, entry in data.items():
        summary = entry["summary"]
        if name != "baseline":
            label = name
            title = f"{label} (job {summary['job']}) so với baseline (job {base['summary']['job']})"
            plot_losses(entry["analysis"] / "loss_vs_baseline.png", title,
                        [(label, COLOR_OPERATION, entry["losses"]), (base_label, COLOR_BASELINE, base["losses"])],
                        args.moving_average)
            plot_metrics(entry["analysis"] / "metrics_vs_baseline.png",
                         title + f" — {label}: {summary['applied_rows']} lượt mẫu được áp dụng",
                         [(label, COLOR_OPERATION, summary["applied"]), (base_label, COLOR_BASELINE, base_applied)])
            plot_transitions(entry["analysis"] / "label_transitions.png", title, (label, summary["applied"]),
                             (base_label, base_applied))
            o, b = summary["applied"]["overall"], base_applied["overall"]
            delta = lambda key: None if o.get(key) is None or b.get(key) is None else round(o[key] - b[key], 6)  # noqa: E731
            summary["difference_from_baseline"] = {
                key: delta(key) for key in ("student_same_as_teacher", "teacher_confidence_mean",
                                            "student_confidence_on_teacher_label_mean", "confidence_gap_mean",
                                            "kl_teacher_student_mean", "teacher_kl_loss_mean",
                                            "inspector_cross_entropy_mean")}
            summary["difference_from_baseline_per_metric"] = {
                metric: {key: None if summary["applied"]["metrics"][metric].get(key) is None
                         else round(summary["applied"]["metrics"][metric][key] - base_applied["metrics"][metric][key], 6)
                         for key in ("student_same_as_teacher", "confidence_gap_mean", "kl_teacher_student_mean")}
                for metric in METRICS}
            agreement = o.get("student_same_as_teacher")
            ranking.append({
                "operation": name, "job": summary["job"], "applied_fraction": summary["applied_fraction"],
                "applied_rows": summary["applied_rows"],
                "student_same_as_teacher": agreement,
                "student_same_as_teacher_minus_baseline": delta("student_same_as_teacher"),
                "confidence_gap_mean": o.get("confidence_gap_mean"),
                "confidence_gap_minus_baseline": delta("confidence_gap_mean"),
                "kl_teacher_student_mean": o.get("kl_teacher_student_mean"),
                "kl_minus_baseline": delta("kl_teacher_student_mean"),
                "teacher_kl_loss_mean": o.get("teacher_kl_loss_mean"),
                "in_target_75_80": agreement is not None and TARGET_AGREEMENT[0] <= agreement <= TARGET_AGREEMENT[1],
                **{f"{metric}_same_as_teacher": summary["applied"]["metrics"][metric]["student_same_as_teacher"]
                   for metric in METRICS},
                **{f"{metric}_confidence_gap": summary["applied"]["metrics"][metric]["confidence_gap_mean"]
                   for metric in METRICS},
            })
        else:
            title = f"baseline (job {summary['job']}), không augmentation"
            plot_losses(entry["analysis"] / "loss_per_step.png", title, [(base_label, COLOR_BASELINE, base["losses"])],
                        args.moving_average)
            plot_metrics(entry["analysis"] / "metrics.png", title, [(base_label, COLOR_BASELINE, base_applied)])
            plot_transitions(entry["analysis"] / "label_transitions.png", title, None, (base_label, base_applied))
        (entry["analysis"] / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")

    if ranking:
        ranking.sort(key=lambda r: (r["student_same_as_teacher"] is None, r["student_same_as_teacher"]))
        with (base["analysis"] / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(ranking)
        plot_ranking(base["analysis"] / "ranking.png", ranking, base_applied["overall"])
    comparison = base["analysis"] / "comparison"
    write_comparison(data, comparison)
    print(f"Written analysis/ in {len(data)} run folders; ranking in {base['analysis']}; "
          f"operation comparison tables in {comparison}")


if __name__ == "__main__":
    main()
