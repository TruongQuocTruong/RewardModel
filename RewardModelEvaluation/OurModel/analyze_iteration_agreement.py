"""Training-process analysis of the Stage 2 iteration runs (2026-11-07).

Runs: stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_
{ema_step_0.99, ema_step_0.99_teacher_kl_relevance_0.5, ema_iteration_0.99,
ema_iteration_0.99_teacher_kl_relevance_0.5}. Purpose: judge the training process from losses,
gradients and agreement BEFORE training ends and WITHOUT EditReward-Bench. Reads logs only
(training_log_<job>.jsonl, per_sample_log_<job>.jsonl, round_0N_summary.json,
iteration_selection_round_0N.jsonl); works on running jobs (the iteration in progress is marked
"complete = False"). No model, no image. Output: <run folder>/analysis/iteration_agreement/

  steps.csv                one row per optimizer step: losses (with the floor removed), gradient
                           norms (whole model, labeled / Unlabeled parts, cosine, Relevance Head),
                           learning rate, clip coefficient, agreement rates
  iterations.csv           per iteration: mean / first 15 / last 15 steps / slope per 100 steps
  stop_condition.csv       early-stop check per iteration, raw loss and learnable part
  same_samples.csv         agreement on the SAME samples across iterations (removes the effect
                           of keeping only Teacher = Inspector samples)
  dropped_samples.csv      iteration-1 rates of samples dropped later vs kept
  selection_per_metric.csv per iteration and metric: Student / Teacher selected (relevance),
                           extra / missed, relevance difference, mismatch near the 0.5 threshold
  selection_mismatch_count.csv  per iteration: number of metrics where S Student != S Teacher
  by_operation.csv         per iteration and applied augmentation operations
  consecutive_iterations.csv    same sample in iteration k and k+1: S agreement transitions, how
                           much Teacher and Student relevance change, same / different operations
  labeled.csv              labeled pairs per iteration (L_human, L_preference, L_relevance per pair)
  summary.json, losses.png, dynamics.png, agreement.png, selection_per_metric.png,
  same_samples.png         (one set of plots per job, English labels)

Cross-run table (no cross-run plot): <ema_step_0.99 run>/analysis/iteration_agreement/comparison/
iterations_all_runs.csv.

Usage: python analyze_iteration_agreement.py [--output-root ...] [--only ema_step_0.99 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(
    "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training")
PREFIX = "stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_"
RUNS = ("ema_step_0.99", "ema_step_0.99_teacher_kl_relevance_0.5",
        "ema_iteration_0.99", "ema_iteration_0.99_teacher_kl_relevance_0.5")
METRICS = ("aesthetic", "fidelity", "realism", "background_consistency",
           "foreground_consistency", "structure_consistency")
TARGET_AGREEMENT = (0.75, 0.80)
NEAR_THRESHOLD = 0.1  # |Teacher relevance - 0.5| < 0.1 counts as "near the selection threshold"
EDGE_STEPS = 15
MOVING_AVERAGE = 8

COLORS = ("#2a78d6", "#eb6834", "#2f9e6e", "#7a5cc7", "#c2452d", "#8a6d1f")
INK_MUTED, GRID = "#52514e", "#d9d8d4"

# Unified loss names (same as analyze_single_operation_augmentation.py, 2026-11-07).
LOSS_PANELS = (
    ("human", "L_human = CE(HumanGT label, Student overall)"),
    ("gemma", "L_preference = L_tie + L_winner (GemmaGT)"),
    ("mask", "L_relevance = BCE(GemmaGT relevance, Student relevance)"),
    ("unlabeled_inspector_kl", "KL(Inspector‖Student) = CE(Inspector, Student) − floor"),
    ("unlabeled_teacher_kl_preference", "KL_pref(Teacher‖Student), metrics in S (Student)"),
    ("unlabeled_teacher_kl_relevance", "KL_rel(Teacher‖Student), mean of 6 metrics"),
    ("unlabeled_learnable", "L_unlabeled, learnable = KL(Inspector‖Student) + λ_C·KL_teacher"),
    ("unlabeled_floor_inspector_entropy", "Floor = entropy of the Inspector target (no gradient)"),
    ("unlabeled", "L_unlabeled (raw, floor included)"),
)
DYNAMICS_PANELS = (
    ("gradient_all_total", "Gradient norm before clipping (whole model)"),
    ("gradient_all_labeled", "Gradient norm, labeled losses only"),
    ("gradient_all_unlabeled", "Gradient norm, L_unlabeled only"),
    ("gradient_all_cosine", "Cosine(labeled-loss gradient, L_unlabeled gradient)"),
    ("gradient_relevance_head_labeled", "Relevance Head gradient norm, labeled losses"),
    ("gradient_relevance_head_unlabeled", "Relevance Head gradient norm, L_unlabeled"),
    ("learning_rate", "Learning rate"),
    ("clip_coefficient", "Clip coefficient = max_norm / gradient norm"),
    ("ema_decay", "EMA decay used at this step (1.0 = Teacher not updated)"),
)
AGREEMENT_PANELS = (
    ("student_same_as_teacher", "Student = Teacher (overall label)"),
    ("student_same_as_inspector", "Student = Inspector (overall label)"),
    ("student_selected_same_as_teacher_selected", "S Student = S Teacher (all 6 metrics)"),
)
NOT_SMOOTHED = {"learning_rate", "ema_decay"}
TREND_KEYS = ("human", "gemma", "mask", "unlabeled_inspector_kl", "unlabeled_teacher_kl_preference",
              "unlabeled_teacher_kl_relevance", "unlabeled_learnable", "gradient_all_total",
              "gradient_all_unlabeled_over_labeled", "gradient_relevance_head_unlabeled",
              "student_same_as_teacher", "student_same_as_inspector",
              "student_selected_same_as_teacher_selected")


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def slope_per_100(xs: Sequence[float], ys: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    var = sum((p[0] - mx) ** 2 for p in pairs)
    return None if var == 0 else 100.0 * sum((p[0] - mx) * (p[1] - my) for p in pairs) / var


def rnd(value: Any, digits: int = 6) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns: List[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: rnd(value) for key, value in row.items()})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # last line may be half-written by a running job
                    pass
    return rows


def single_file(run: Path, pattern: str) -> Path:
    files = sorted(run.glob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"{run.name}: expected one {pattern}, found {[f.name for f in files]}")
    return files[0]


# ----------------------------------------------------------------------------- steps
def step_rows(training_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in training_log:
        if record.get("kind") != "step":
            continue
        diagnostics = record.get("diagnostics", {})
        losses = diagnostics.get("losses", {})
        gradient = diagnostics.get("gradient_norm_before_clipping", {})
        agreement = record.get("agreement", {})
        row: Dict[str, Any] = {"step": int(record["step"]), "iteration": int(record["round"])}
        for key in ("human", "gemma", "mask", "tie", "winner", "unlabeled", "unlabeled_inspector_cross_entropy",
                    "unlabeled_inspector_kl", "unlabeled_teacher_kl", "unlabeled_teacher_kl_preference",
                    "unlabeled_teacher_kl_relevance", "teacher_kl_relevance_weight"):
            row[key] = losses.get(key)
        row["unlabeled_floor_inspector_entropy"] = losses.get("unlabeled_inspector_target_entropy")
        if row["unlabeled"] is not None and row["unlabeled_floor_inspector_entropy"] is not None:
            row["unlabeled_learnable"] = row["unlabeled"] - row["unlabeled_floor_inspector_entropy"]
        for group in ("all", "vit", "llm", "heads", "relevance_head"):
            for part in ("labeled", "unlabeled", "total", "cosine"):
                row[f"gradient_{group}_{part}"] = (gradient.get(group) or {}).get(part)
            labeled, unlabeled = row[f"gradient_{group}_labeled"], row[f"gradient_{group}_unlabeled"]
            row[f"gradient_{group}_unlabeled_over_labeled"] = (
                unlabeled / labeled if labeled and unlabeled is not None else None)
        row["update_ratio_all"] = (diagnostics.get("update_ratio") or {}).get("all")
        rates = record.get("learning_rates") or diagnostics.get("learning_rates") or [None]
        row["learning_rate"] = rates[0]
        row["clip_coefficient"] = diagnostics.get("clip_coefficient")
        row["ema_decay"] = record.get("ema_decay")
        for key in ("student_same_as_teacher", "student_same_as_inspector",
                    "student_selected_same_as_teacher_selected", "teacher_kl_relevance_mean"):
            row[key] = agreement.get(key)
        rows.append(row)
    rows.sort(key=lambda r: r["step"])
    return rows


def iteration_rows(steps: List[Dict[str, Any]], complete: Dict[int, bool]) -> List[Dict[str, Any]]:
    out = []
    keys = [k for k in steps[0] if k not in ("step", "iteration")] if steps else []
    for iteration in sorted({r["iteration"] for r in steps}):
        rows = [r for r in steps if r["iteration"] == iteration]
        row: Dict[str, Any] = {"iteration": iteration, "complete": complete.get(iteration, False),
                               "steps": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"]}
        for key in keys:
            row[f"{key}_mean"] = mean(r[key] for r in rows)
        for key in TREND_KEYS:
            row[f"{key}_first{EDGE_STEPS}"] = mean(r[key] for r in rows[:EDGE_STEPS])
            row[f"{key}_last{EDGE_STEPS}"] = mean(r[key] for r in rows[-EDGE_STEPS:])
            row[f"{key}_slope_per_100_steps"] = slope_per_100([r["step"] for r in rows], [r[key] for r in rows])
        out.append(row)
    return out


def stop_condition_rows(run: Path, iterations: List[Dict[str, Any]], tolerance: Optional[float]) -> List[Dict[str, Any]]:
    """Early stop in train_loop.py: after iteration k >= 2, stop if BOTH relative decreases
    (previous - current) / previous of the iteration-mean L_unlabeled (raw) and L_human are
    < tolerance (an increase is a negative decrease, so it also counts)."""
    by_iteration = {r["iteration"]: r for r in iterations}
    out = []
    for iteration in sorted(by_iteration):
        summary_path = run / f"round_{iteration:02d}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        selection = summary.get("next_iteration_selection") or {}
        row: Dict[str, Any] = {"iteration": iteration, "complete": by_iteration[iteration]["complete"],
                               "logged_relative_decrease": json.dumps(selection.get("relative_decrease")),
                               "logged_loss_plateau": selection.get("loss_plateau"),
                               "logged_stop_reason": selection.get("stop_reason")}
        current = by_iteration[iteration]
        previous = by_iteration.get(iteration - 1)
        for key in ("unlabeled", "human", "unlabeled_learnable", "unlabeled_floor_inspector_entropy"):
            row[f"{key}_mean"] = current[f"{key}_mean"]
            if previous and previous[f"{key}_mean"]:
                row[f"{key}_relative_decrease"] = (previous[f"{key}_mean"] - current[f"{key}_mean"]) / previous[f"{key}_mean"]
        if tolerance is not None and previous:
            row["tolerance"] = tolerance
            row["would_stop_raw (unlabeled and human)"] = (
                row.get("unlabeled_relative_decrease", 1) < tolerance and row.get("human_relative_decrease", 1) < tolerance)
            row["would_stop_learnable (unlabeled_learnable and human)"] = (
                row.get("unlabeled_learnable_relative_decrease", 1) < tolerance
                and row.get("human_relative_decrease", 1) < tolerance)
        out.append(row)
    return out


# ----------------------------------------------------------------------------- per sample
def rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": len(rows),
        "student_same_as_teacher": mean(r["student_same_as_teacher"] for r in rows),
        "student_same_as_inspector": mean(r["student_same_as_inspector"] for r in rows),
        "teacher_same_as_inspector": mean(r["teacher_same_as_inspector"] for r in rows),
        "student_selected_same_as_teacher_selected": mean(r.get("student_selected_same_as_teacher_selected") for r in rows),
        "teacher_kl_mean": mean(r["teacher_kl"] for r in rows),
        "teacher_kl_relevance_mean": mean(r.get("teacher_kl_relevance") for r in rows),
        "inspector_kl_mean": mean(r["inspector_kl"] for r in rows),
        "student_selected_metrics_mean": mean(r["student_selected_metrics"] for r in rows),
        "teacher_selected_metrics_mean": mean(r["teacher_selected_metrics"] for r in rows),
    }


def operations_of(row: Dict[str, Any]) -> str:
    if not row.get("augmentation_applied"):
        return "none"
    return "+".join(sorted(row.get("augmentation_applied_operations") or [])) or "none"


def same_sample_rows(by_iteration: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    iterations = sorted(by_iteration)
    for reference in iterations:
        reference_ids = set(by_iteration[reference])
        for iteration in iterations:
            if iteration > reference:
                continue
            ids = reference_ids & set(by_iteration[iteration])
            row = {"sample_set": f"samples of iteration {reference}", "reference_iteration": reference,
                   "iteration": iteration}
            row.update(rates([by_iteration[iteration][i] for i in sorted(ids)]))
            out.append(row)
    return out


def dropped_sample_rows(run: Path, by_iteration: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for iteration in sorted(by_iteration):
        selection_path = run / f"iteration_selection_round_{iteration:02d}.jsonl"
        if not selection_path.exists():
            continue
        kept = {r["sample_id"] for r in read_jsonl(selection_path) if r.get("kept_for_next_iteration")}
        rows = by_iteration[iteration]
        for name, ids in (("kept", [i for i in rows if i in kept]), ("dropped", [i for i in rows if i not in kept])):
            row = {"iteration": iteration, "group": f"{name} for iteration {iteration + 1}"}
            row.update(rates([rows[i] for i in ids]))
            out.append(row)
    return out


def selection_metric_rows(by_iteration: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for iteration in sorted(by_iteration):
        rows = list(by_iteration[iteration].values())
        for metric in METRICS:
            values = [r["metrics"][metric] for r in rows if metric in r.get("metrics", {})]
            if not values:
                continue
            mismatch = [v for v in values if v["student_selected"] != v["teacher_selected"]]
            near = [v for v in values if abs(v["teacher_relevance"] - 0.5) < NEAR_THRESHOLD]
            far = [v for v in values if abs(v["teacher_relevance"] - 0.5) >= NEAR_THRESHOLD]
            out.append({
                "iteration": iteration, "metric": metric, "samples": len(values),
                "teacher_selected_rate": mean(v["teacher_selected"] for v in values),
                "student_selected_rate": mean(v["student_selected"] for v in values),
                "mismatch_rate": len(mismatch) / len(values),
                "extra_rate (Student selects, Teacher not)": mean(
                    v["student_selected"] and not v["teacher_selected"] for v in values),
                "missed_rate (Teacher selects, Student not)": mean(
                    v["teacher_selected"] and not v["student_selected"] for v in values),
                "relevance_difference_mean (Student - Teacher)": mean(
                    v["student_relevance"] - v["teacher_relevance"] for v in values),
                "relevance_absolute_difference_mean": mean(
                    abs(v["student_relevance"] - v["teacher_relevance"]) for v in values),
                "teacher_relevance_mean": mean(v["teacher_relevance"] for v in values),
                "student_relevance_mean": mean(v["student_relevance"] for v in values),
                "teacher_near_threshold_share": len(near) / len(values),
                "mismatch_rate_teacher_near_threshold": mean(
                    v["student_selected"] != v["teacher_selected"] for v in near),
                "mismatch_rate_teacher_far_from_threshold": mean(
                    v["student_selected"] != v["teacher_selected"] for v in far),
                "share_of_mismatches_teacher_near_threshold": (
                    mean(abs(v["teacher_relevance"] - 0.5) < NEAR_THRESHOLD for v in mismatch)),
                "metric_kl_preference_mean": mean(v["kl_teacher_student"] for v in values),
                "metric_same_label_rate": mean(v["student_same_as_teacher"] for v in values),
            })
    return out


def mismatch_count_rows(by_iteration: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for iteration in sorted(by_iteration):
        rows = list(by_iteration[iteration].values())
        counts = Counter(sum(r["metrics"][m]["student_selected"] != r["metrics"][m]["teacher_selected"]
                             for m in METRICS if m in r.get("metrics", {})) for r in rows)
        out.append({"iteration": iteration, "samples": len(rows),
                    **{f"metrics_differing_{k}": counts.get(k, 0) / len(rows) for k in range(0, 4)},
                    "metrics_differing_4_or_more": sum(v for k, v in counts.items() if k >= 4) / len(rows)})
    return out


def operation_rows(by_iteration: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for iteration in sorted(by_iteration):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in by_iteration[iteration].values():
            groups[operations_of(row)].append(row)
            if "composite" in operations_of(row):
                groups[f"composite position {row.get('augmentation_position')}"].append(row)
        for name in sorted(groups):
            row = {"iteration": iteration, "operations": name}
            row.update(rates(groups[name]))
            out.append(row)
    return out


def consecutive_rows(by_iteration: Dict[int, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    iterations = sorted(by_iteration)
    for first, second in zip(iterations, iterations[1:]):
        ids = sorted(set(by_iteration[first]) & set(by_iteration[second]))
        for group in ("all", "same operations", "different operations"):
            pairs = [(by_iteration[first][i], by_iteration[second][i]) for i in ids]
            if group == "same operations":
                pairs = [p for p in pairs if operations_of(p[0]) == operations_of(p[1])]
            elif group == "different operations":
                pairs = [p for p in pairs if operations_of(p[0]) != operations_of(p[1])]
            if not pairs:
                continue
            transitions = Counter((bool(a["student_selected_same_as_teacher_selected"]),
                                   bool(b["student_selected_same_as_teacher_selected"])) for a, b in pairs)

            def change(side: str) -> Optional[float]:
                return mean(mean(abs(b["metrics"][m][f"{side}_relevance"] - a["metrics"][m][f"{side}_relevance"])
                                 for m in METRICS) for a, b in pairs)

            def set_changed(side: str) -> Optional[float]:
                return mean(any(a["metrics"][m][f"{side}_selected"] != b["metrics"][m][f"{side}_selected"]
                                for m in METRICS) for a, b in pairs)

            out.append({
                "from_iteration": first, "to_iteration": second, "group": group, "samples": len(pairs),
                "s_same_then_same": transitions[(True, True)] / len(pairs),
                "s_same_then_different": transitions[(True, False)] / len(pairs),
                "s_different_then_same": transitions[(False, True)] / len(pairs),
                "s_different_then_different": transitions[(False, False)] / len(pairs),
                "teacher_relevance_absolute_change_mean": change("teacher"),
                "student_relevance_absolute_change_mean": change("student"),
                "teacher_selected_set_changed_rate": set_changed("teacher"),
                "student_selected_set_changed_rate": set_changed("student"),
            })
    return out


def labeled_rows(per_sample: List[Dict[str, Any]], iteration_of_step: Dict[int, int]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        if row.get("kind") != "labeled":
            continue
        iteration = iteration_of_step.get(int(row["step"]))
        if iteration is None:
            continue
        groups[(iteration, "all")].append(row)
        groups[(iteration, operations_of(row))].append(row)
    out = []
    for (iteration, name) in sorted(groups):
        rows = groups[(iteration, name)]
        out.append({"iteration": iteration, "operations": name, "pairs": len(rows),
                    "human_cross_entropy_mean": mean(r["human_cross_entropy"] for r in rows),
                    "gemma_preference_loss_mean": mean(r.get("gemma_preference_loss") for r in rows),
                    "gemma_relevance_loss_mean": mean(r.get("gemma_relevance_loss") for r in rows),
                    "student_same_as_human": mean(r["student_same_as_human"] for r in rows),
                    "student_selected_metrics_mean": mean(r["student_selected_metrics"] for r in rows)})
    return out


# ----------------------------------------------------------------------------- plots
def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    out = []
    for index in range(len(values)):
        out.append(mean(values[max(0, index - window + 1): index + 1]))
    return out


def plot_steps(path: Path, title: str, steps: List[Dict[str, Any]], panels: Sequence[tuple],
               boundaries: List[int], target: bool = False) -> None:
    panels = [p for p in panels if any(r.get(p[0]) is not None for r in steps)]
    if not panels:
        return
    columns = 3
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.3 * rows), squeeze=False)
    x = [r["step"] for r in steps]
    for ax, (key, label) in zip(axes.flat, panels):
        y = [r.get(key) for r in steps]
        if key in NOT_SMOOTHED:
            ax.plot(x, y, color=COLORS[0], linewidth=1.2)
        else:
            ax.plot(x, y, color=COLORS[0], linewidth=0.8, alpha=0.3, label="value per step")
            ax.plot(x, moving_average(y, MOVING_AVERAGE), color=COLORS[0], linewidth=1.6,
                    label=f"moving average ({MOVING_AVERAGE} steps)")
        for boundary in boundaries:
            ax.axvline(boundary + 0.5, color=INK_MUTED, linestyle="--", linewidth=0.8)
        if target and key == "student_same_as_teacher":
            ax.axhspan(*TARGET_AGREEMENT, color=GRID, alpha=0.6, zorder=0, label="target 75–80%")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("optimizer step (dashed line = end of an iteration)", fontsize=8)
        ax.grid(color=GRID, linewidth=0.5)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_selection(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    iterations = sorted({r["iteration"] for r in rows})
    if not iterations:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    width = 0.8 / len(iterations)
    for ax, (key, label) in zip(axes, (
            ("extra_rate (Student selects, Teacher not)", "Extra: Student selects, Teacher does not"),
            ("missed_rate (Teacher selects, Student not)", "Missed: Teacher selects, Student does not"),
            ("relevance_difference_mean (Student - Teacher)", "Mean relevance difference (Student − Teacher)"))):
        for index, iteration in enumerate(iterations):
            values = [next((r[key] for r in rows if r["iteration"] == iteration and r["metric"] == m), 0) or 0
                      for m in METRICS]
            ax.bar([i + index * width for i in range(len(METRICS))], values, width,
                   color=COLORS[index % len(COLORS)], label=f"iteration {iteration}")
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(METRICS))])
        ax.set_xticklabels([m.replace("_consistency", "_cons.") for m in METRICS], rotation=30, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.axhline(0, color=INK_MUTED, linewidth=0.6)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_same_samples(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    references = sorted({r["reference_iteration"] for r in rows})
    if not references:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (key, label) in zip(axes, AGREEMENT_PANELS):
        for index, reference in enumerate(references):
            subset = sorted((r for r in rows if r["reference_iteration"] == reference), key=lambda r: r["iteration"])
            ax.plot([r["iteration"] for r in subset], [r[key] for r in subset], marker="o",
                    color=COLORS[index % len(COLORS)],
                    label=f"samples of iteration {reference} (n={subset[-1]['samples']})")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_xticks(references)
        ax.grid(color=GRID, linewidth=0.5)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- run
def analyze_run(run: Path) -> List[Dict[str, Any]]:
    if not list(run.glob("training_log_*.jsonl")):
        print(f"{run.name}: no training log yet (job not started), skipped")
        return []
    training_log_path = single_file(run, "training_log_*.jsonl")
    job = training_log_path.stem.rsplit("_", 1)[-1]
    # train_loop arguments are not in args.json (Hugging Face arguments); the tolerance is in the
    # round summaries, the EMA / KL_rel setting is in the folder name.
    tolerance = None
    for path in sorted(run.glob("round_*_summary.json")):
        tolerance = (json.loads(path.read_text()).get("next_iteration_selection") or {}).get("loss_plateau_tolerance")
    steps = step_rows(read_jsonl(training_log_path))
    if not steps:
        print(f"{run.name}: no step yet")
        return []
    complete = {i: (run / f"round_{i:02d}_summary.json").exists() for i in {r["iteration"] for r in steps}}
    iteration_of_step = {r["step"]: r["iteration"] for r in steps}
    per_sample = read_jsonl(single_file(run, "per_sample_log_*.jsonl"))
    by_iteration: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in per_sample:
        if row.get("kind") == "unlabeled":
            iteration = iteration_of_step.get(int(row["step"]))
            if iteration is not None:
                by_iteration[iteration][row["sample_id"]] = row  # one row per sample and iteration
    by_iteration = dict(by_iteration)

    out_dir = run / "analysis" / "iteration_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations = iteration_rows(steps, complete)
    tables = {
        "steps": steps,
        "iterations": iterations,
        "stop_condition": stop_condition_rows(run, iterations, tolerance),
        "same_samples": same_sample_rows(by_iteration),
        "dropped_samples": dropped_sample_rows(run, by_iteration),
        "selection_per_metric": selection_metric_rows(by_iteration),
        "selection_mismatch_count": mismatch_count_rows(by_iteration),
        "by_operation": operation_rows(by_iteration),
        "consecutive_iterations": consecutive_rows(by_iteration),
        "labeled": labeled_rows(per_sample, iteration_of_step),
    }
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)
    summary = {"run": run.name, "job": job, "last_step": steps[-1]["step"],
               "iteration_complete": {str(k): v for k, v in sorted(complete.items())},
               "loss_plateau_tolerance": tolerance,
               **{name: rows for name, rows in tables.items() if name != "steps"}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    title = f"{run.name[len(PREFIX):]} (job {job}), step {steps[-1]['step']}"
    boundaries = [r["last_step"] for r in iterations if r["complete"]]
    plot_steps(out_dir / "losses.png", f"Losses — {title}", steps, LOSS_PANELS, boundaries)
    plot_steps(out_dir / "dynamics.png", f"Training dynamics — {title}", steps, DYNAMICS_PANELS, boundaries)
    plot_steps(out_dir / "agreement.png", f"Agreement on Unlabeled — {title}", steps,
               AGREEMENT_PANELS + (("teacher_kl_relevance_mean", "KL_rel(Teacher‖Student), logged in every run"),),
               boundaries, target=True)
    plot_selection(out_dir / "selection_per_metric.png", f"Relevance selection S per metric — {title}",
                   tables["selection_per_metric"])
    plot_same_samples(out_dir / "same_samples.png", f"Agreement on the same samples — {title}",
                      tables["same_samples"])
    print(f"{run.name}: job {job}, step {steps[-1]['step']}, iterations {sorted(complete)} -> {out_dir}")
    return [{"run": run.name[len(PREFIX):], "job": job, **row} for row in iterations]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--only", nargs="+", default=[], choices=RUNS)
    args = parser.parse_args()
    root = Path(args.output_root)
    all_rows: List[Dict[str, Any]] = []
    for name in args.only or RUNS:
        run = root / f"{PREFIX}{name}"
        if not run.is_dir():
            print(f"{run.name}: folder missing, skipped")
            continue
        all_rows.extend(analyze_run(run))
    comparison = root / f"{PREFIX}{RUNS[0]}" / "analysis" / "iteration_agreement" / "comparison"
    if all_rows and comparison.parent.is_dir():
        comparison.mkdir(exist_ok=True)
        write_csv(comparison / "iterations_all_runs.csv", all_rows)
        print(f"comparison -> {comparison}")


if __name__ == "__main__":
    main()
