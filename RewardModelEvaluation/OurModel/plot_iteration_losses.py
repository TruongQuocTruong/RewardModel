#!/usr/bin/env python3
"""Plot each Stage 2 loss component of the iteration runs: per step inside each iteration and across iterations.

Input (read only): Qwen3-VL-Embedding/output_training/<version>/training_log_<job>.jsonl,
records with kind == "step", field diagnostics.losses (values of that step only, not running means).

Outputs: evaluation_results/train_loop/stage2_tuning_comparison/iteration_losses/
  <loss>.png           row 1: one subplot per run, x = step inside the iteration, one line per iteration
                       (faint = per step, bold = mean of the last 10 steps);
                       row 2: mean of the loss per iteration for every run, error bar = 95% confidence
                       interval of the mean (1.96 * standard deviation / sqrt(steps)); hollow marker =
                       iteration not finished.
  iteration_means.csv  the numbers of row 2.
Runs where a loss is 0 at every step (not computed, weight 0) are shown as "not computed in this run".

With --per-run: one figure set per finished run (PER_RUN), saved in that run's training output:
  Qwen3-VL-Embedding/output_training/<version>/analysis/losses/
  <loss>.png           row 1: one subplot per iteration, x = step inside the iteration (faint = per step,
                       bold = mean of the last 10 steps), shared y axis;
                       row 2: mean of the loss per iteration with 95% confidence interval.
  iteration_means.csv  the numbers of row 2.
A loss that is 0 at every step of the run (not computed) gets no figure.
Grouped losses (GROUPED, same layout, title = formula; checked on every step of the 3 runs:
gemma = tie + winner, unlabeled = CE(Inspector, Student) + KL teacher term, KL teacher term =
(1 - w) * KL_score + w * KL_rel, CE(Inspector, Student) = KL(Inspector||Student) + Inspector label entropy):
  L_human.png, L_pref.png, L_rel.png, CE_inspector_student.png, KL_score_teacher_student.png,
  KL_rel_teacher_student.png (every run: from the monitored step value agreement.teacher_kl_relevance_mean,
  logged in every run; equal to the loss value unlabeled_teacher_kl_relevance within 4e-7 in the w = 0.5 run). A grouped loss with components gets a second panel in
  row 2: mean per iteration of each component.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/datastore/truongtq/TrainingRewardModel")
OUTPUT_TRAINING = ROOT / "TrainingRewardModel/Qwen3-VL-Embedding/output_training"
OUT_DIR = ROOT / "RewardModelEvaluation/OurModel/evaluation_results/train_loop/stage2_tuning_comparison/iteration_losses"
PREFIX = "stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_"
RUNS = [  # (short name, job)
    ("ema_step_0.99", 11199),
    ("ema_step_0.99_teacher_kl_relevance_0.5", 11200),
    ("ema_iteration_0.99", 11216),
    ("ema_iteration_0.99_teacher_kl_relevance_0.5", 11500),
]
PLANNED_ITERATIONS = 5
PER_RUN = ["ema_step_0.99", "ema_step_0.99_teacher_kl_relevance_0.5", "ema_iteration_0.99"]
LOSSES = [
    "human", "gemma", "mask", "tie", "winner",
    "unlabeled", "unlabeled_inspector_cross_entropy", "unlabeled_inspector_target_entropy",
    "unlabeled_inspector_kl", "unlabeled_teacher_kl", "unlabeled_teacher_kl_preference",
    "unlabeled_teacher_kl_relevance",
]
# (file name, log field, title, components [(log field, label)])
GROUPED = [
    ("L_human", "human", "L_human = CE(HumanGT label, Student overall)", []),
    ("L_pref", "gemma", "L_pref = L_tie + L_winner  (GemmaGT, part of L_gemma)",
     [("tie", "L_tie"), ("winner", "L_winner")]),
    ("L_rel", "mask", "L_rel = BCE(GemmaGT relevance, Student relevance)  (part of L_gemma)", []),
    ("CE_inspector_student", "unlabeled_inspector_cross_entropy",
     "CE(Inspector\u2016Student) = KL(Inspector\u2016Student) + entropy of the Inspector label  (part of L_unlabeled)",
     [("unlabeled_inspector_kl", "KL(Inspector\u2016Student)"),
      ("unlabeled_inspector_target_entropy", "entropy of the Inspector label (floor, no gradient)")]),
    ("KL_score_teacher_student", "unlabeled_teacher_kl_preference",
     "KL_score(Teacher\u2016Student)  (L_unlabeled term = (1-w)\u00b7KL_score + w\u00b7KL_rel)", []),
    ("KL_rel_teacher_student", "monitor_teacher_kl_relevance",
     "KL_rel(Teacher\u2016Student), mean of 6 metrics  (logged in every run; in L_unlabeled only when w > 0; w = {w})", []),
]
ROLLING = 10
COMPONENT_COLORS = ["#eb6834", "#1baf7a"]  # categorical slots 2-3
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (fixed order)
ITERATION_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue ramp 250..650
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def load(name, job):
    """Return {iteration: [(step_in_iteration, losses dict)]}, plus the last logged iteration-selection rounds."""
    path = OUTPUT_TRAINING / (PREFIX + name) / f"training_log_{job}.jsonl"
    per_iteration, finished = {}, set()
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # last line may be half written for a running job
                continue
            if record["kind"] == "step":
                losses = dict(record["diagnostics"]["losses"])
                losses["monitor_teacher_kl_relevance"] = (record.get("agreement") or {}).get("teacher_kl_relevance_mean")
                per_iteration.setdefault(record["round"], []).append(losses)
            elif record["kind"] == "iteration_selection":
                finished.add(record["round"])
    return per_iteration, finished


def trailing_mean(values, window=ROLLING):
    out = []
    for index in range(len(values)):
        chunk = [v for v in values[max(0, index - window + 1): index + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return out


def mean_ci(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    n = len(values)
    if n == 0:
        return math.nan, math.nan, 0
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, 1.96 * sd / math.sqrt(n), n


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_per_run(name, job, per_iteration, finished):
    out_dir = OUTPUT_TRAINING / (PREFIX + name) / "analysis" / "losses"
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations = sorted(per_iteration)
    rows = []
    figures = [(loss, loss, f"loss '{loss}' (value of each step)", []) for loss in LOSSES] + GROUPED
    for filename, loss, title, components in figures:
        if all(not step.get(loss) for steps in per_iteration.values() for step in steps):
            print(f"skip {name} {filename}: not computed in this run")
            continue
        fig = plt.figure(figsize=(20, 9))
        grid = fig.add_gridspec(2, 30, height_ratios=[1, 1], hspace=0.35, wspace=1.2)
        width = 30 // len(iterations)
        top_axes = [fig.add_subplot(grid[0, i * width:(i + 1) * width - 1]) for i in range(len(iterations))]
        if components:
            across = fig.add_subplot(grid[1, :14])
            parts = fig.add_subplot(grid[1, 16:])
        else:
            across = fig.add_subplot(grid[1, 6:24])
        smooth_values, means = [], []
        for column, iteration in enumerate(iterations):
            values = [step.get(loss) for step in per_iteration[iteration]]
            x = list(range(1, len(values) + 1))
            color = ITERATION_COLORS[(iteration - 1) % len(ITERATION_COLORS)]
            smooth = trailing_mean(values)
            smooth_values += [v for v in smooth if not math.isnan(v)]
            ax = top_axes[column]
            ax.plot(x, [math.nan if v is None else v for v in values], color=color, linewidth=0.8, alpha=0.3)
            ax.plot(x, smooth, color=color, linewidth=2)
            done = iteration in finished
            ax.set_title(f"iteration {iteration} ({len(values)} steps)" + ("" if done else ", running"), fontsize=9)
            ax.set_xlabel("step inside the iteration", fontsize=8)
            if column == 0:
                ax.set_ylabel(f"{filename} (bold = mean of last {ROLLING} steps)", fontsize=8)
            style(ax)
            mean, half, n = mean_ci(values)
            means.append((iteration, mean, half, done))
            rows.append({"loss": filename, "run": name, "job": job, "iteration": iteration, "steps": n,
                         "finished": done, "mean": round(mean, 6), "ci95_half_width": round(half, 6)})
        if smooth_values:
            low, high = min(smooth_values), max(smooth_values)
            pad = (high - low) * 0.15 or 0.01
            for ax in top_axes:
                ax.set_ylim(low - pad, high + pad)
        across.errorbar([m[0] for m in means], [m[1] for m in means], yerr=[m[2] for m in means],
                        color=RUN_COLORS[0], linewidth=2, capsize=3, zorder=2)
        for iteration, mean, half, done in means:
            across.plot(iteration, mean, marker="o", markersize=8, color=RUN_COLORS[0],
                        markerfacecolor=RUN_COLORS[0] if done else "white", markeredgewidth=2, zorder=3)
            across.annotate(f"{mean:.4f}", (iteration, mean), xytext=(8, 0), textcoords="offset points",
                            fontsize=8, color=TEXT_SECONDARY, va="center")
        across.set_xticks(range(1, PLANNED_ITERATIONS + 1))
        across.set_xlim(0.7, PLANNED_ITERATIONS + 0.5)
        across.set_title("Mean per iteration (95% CI)", fontsize=9)
        across.set_ylabel(filename, fontsize=8)
        across.set_xlabel("iteration", fontsize=8)
        style(across)
        if components:
            for color, (field, label) in zip(COMPONENT_COLORS, components):
                stats = [(it, *mean_ci([step.get(field) for step in per_iteration[it]])) for it in iterations]
                parts.errorbar([t[0] for t in stats], [t[1] for t in stats], yerr=[t[2] for t in stats],
                               color=color, linewidth=1.5, capsize=3, marker="o", markersize=6, label=label)
                for it, mean, half, n in stats:
                    rows.append({"loss": f"{filename}:{field}", "run": name, "job": job, "iteration": it,
                                 "steps": n, "finished": it in finished, "mean": round(mean, 6),
                                 "ci95_half_width": round(half, 6)})
            parts.set_xticks(range(1, PLANNED_ITERATIONS + 1))
            parts.set_xlim(0.7, PLANNED_ITERATIONS + 0.3)
            parts.set_title("Components, mean per iteration (95% CI)", fontsize=9)
            parts.set_xlabel("iteration", fontsize=8)
            parts.legend(fontsize=7, frameon=False)
            style(parts)
        weight = next(iter(per_iteration.values()))[0].get("teacher_kl_relevance_weight")
        fig.suptitle(f"{name} (job {job}): {title.replace('{w}', str(weight))}", fontsize=12)
        path = out_dir / f"{filename}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path)
    with open(out_dir / "iteration_means.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out_dir / "iteration_means.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-run", action="store_true", help="one figure set per run in its training output")
    args = parser.parse_args()
    if args.per_run:
        for name, job in RUNS:
            if name in PER_RUN:
                plot_per_run(name, job, *load(name, job))
    else:
        compare_runs()


def compare_runs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {name: load(name, job) for name, job in RUNS}
    rows = []
    for loss in LOSSES:
        fig = plt.figure(figsize=(18, 9))
        grid = fig.add_gridspec(2, len(RUNS), height_ratios=[1, 1], hspace=0.35, wspace=0.18)
        top_axes = [fig.add_subplot(grid[0, i]) for i in range(len(RUNS))]
        bottom = fig.add_subplot(grid[1, :])
        lows, highs = [], []
        for column, (name, job) in enumerate(RUNS):
            per_iteration, finished = data[name]
            ax = top_axes[column]
            if all(not step.get(loss) for steps in per_iteration.values() for step in steps):
                # Every step is 0 or missing: the loss is not computed in this run (weight 0), do not plot it as 0.
                ax.text(0.5, 0.5, "not computed in this run", transform=ax.transAxes, ha="center",
                        va="center", fontsize=9, color=TEXT_SECONDARY)
                ax.set_xticks([])
                ax.set_title(f"{name}\n(job {job})", fontsize=9)
                style(ax)
                continue
            for iteration in sorted(per_iteration):
                values = [step.get(loss) for step in per_iteration[iteration]]
                x = list(range(1, len(values) + 1))
                color = ITERATION_COLORS[(iteration - 1) % len(ITERATION_COLORS)]
                raw = [math.nan if v is None else v for v in values]
                ax.plot(x, raw, color=color, linewidth=0.8, alpha=0.18)
                smooth = trailing_mean(values)
                label = f"iteration {iteration}" + ("" if iteration in finished else " (running)")
                ax.plot(x, smooth, color=color, linewidth=2, label=label)
                lows += [v for v in smooth if not math.isnan(v)]
                mean, half, n = mean_ci(values)
                rows.append({"loss": loss, "run": name, "job": job, "iteration": iteration,
                             "steps": n, "finished": iteration in finished,
                             "mean": round(mean, 6), "ci95_half_width": round(half, 6)})
            ax.set_title(f"{name}\n(job {job})", fontsize=9)
            ax.set_xlabel("step inside the iteration", fontsize=8)
            if column == 0:
                ax.set_ylabel(f"{loss} (bold = mean of last {ROLLING} steps)", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            style(ax)
        # Same y range on row 1 from the smoothed lines, so the four runs compare directly.
        if lows:
            low, high = min(lows), max(lows)
            pad = (high - low) * 0.15 or 0.01
            for ax in top_axes:
                ax.set_ylim(low - pad, high + pad)
        for column, (name, job) in enumerate(RUNS):
            run_rows = [r for r in rows if r["loss"] == loss and r["run"] == name]
            if not run_rows:
                continue
            x = [r["iteration"] for r in run_rows]
            y = [r["mean"] for r in run_rows]
            err = [r["ci95_half_width"] for r in run_rows]
            color = RUN_COLORS[column]
            offset = (column - (len(RUNS) - 1) / 2) * 0.06
            xs = [v + offset for v in x]
            bottom.errorbar(xs, y, yerr=err, color=color, linewidth=2, capsize=3, label=name, zorder=2)
            for xi, yi, r in zip(xs, y, run_rows):
                bottom.plot(xi, yi, marker="o", markersize=8, color=color,
                            markerfacecolor=color if r["finished"] else "white",
                            markeredgecolor=color, markeredgewidth=2, zorder=3)
        bottom.set_xticks(range(1, PLANNED_ITERATIONS + 1))
        bottom.set_xlim(0.7, PLANNED_ITERATIONS + 0.9)
        bottom.set_xlabel("iteration", fontsize=9)
        bottom.set_ylabel(f"mean {loss} per iteration (95% CI)", fontsize=8)
        bottom.set_title("Across iterations (hollow marker = iteration not finished)", fontsize=9)
        bottom.legend(fontsize=8, frameon=False, loc="best")
        style(bottom)
        fig.suptitle(f"Stage 2 iteration runs: loss '{loss}' (value of each step)", fontsize=12)
        path = OUT_DIR / f"{loss}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path)
    with open(OUT_DIR / "iteration_means.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT_DIR / "iteration_means.csv")


if __name__ == "__main__":
    main()
