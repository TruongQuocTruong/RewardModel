#!/usr/bin/env python3
"""Plot Stage 2 tuning runs from their training logs (no dataset or checkpoint access).

Inputs per job:
  TrainingRewardModel/logs/train_loop_<job>.out
  Qwen3-VL-Embedding/output_training/<version>/round_*_summary.json
Job -> version mapping comes from TrainingRewardModel/experiment_submissions.log.

The train_loop log prints, every step, the running mean of each loss since the start of the
round. This script converts them back to per-step values:
  value_k = (mean_k * count_k - mean_{k-1} * count_{k-1}) / (count_k - count_{k-1})
with count = Unlabeled occurrences (sum of status counts) for the Unlabeled loss and
labeled pairs per step x step for the labeled losses.

Outputs:
  evaluation_results/train_loop/<version>/stage2_tuning_{losses,gradient_norm,schedule}.png
  evaluation_results/train_loop/<version>/stage2_tuning_unlabeled_loss_by_pass.png (runs with passes > 1)
  evaluation_results/train_loop/stage2_tuning_comparison/*.png and summary.csv
Runs with REFINEMENT_STEP_DIAGNOSTICS (one 'step=N diagnostics {json}' line per step, values of
that step only, losses before lambda):
  evaluation_results/train_loop/<version>/stage2_tuning_step_diagnostics.png
Runs with evaluations on the Stage 2 training data (evaluate_stage2_tuning_checkpoints.py),
plotted against the step together with step 0 (the Student-Stage1 evaluation):
  evaluation_results/train_loop/<version>/stage2_tuning_training_data_evaluation.png
  evaluation_results/train_loop/stage2_tuning_comparison/<--comparison-name>_training_data_evaluation.png
"""

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/datastore/truongtq/TrainingRewardModel")
PROJECT = ROOT / "TrainingRewardModel"
LOG_DIR = PROJECT / "logs"
OUTPUT_TRAINING = PROJECT / "Qwen3-VL-Embedding" / "output_training"
SUBMISSION_LOG = PROJECT / "experiment_submissions.log"
RESULTS = ROOT / "RewardModelEvaluation" / "OurModel" / "evaluation_results" / "train_loop"
COMPARISON_DIR = RESULTS / "stage2_tuning_comparison"

# Categorical palette (fixed order, light mode); colour follows the version, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ROLLING = 10

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "font.size": 10,
})

STEP_LINE = re.compile(
    r"round=(\d+) step=(\d+) human=([\d.]+) gemma=([\d.]+) mask=([\d.]+) set=([\d.]+) "
    r"lambda_set=([\d.]+) set_scale=([\d.]+) ema_decay=([\d.]+) statuses=(\{.*\})"
)
GRAD_LINE = re.compile(r"step=(\d+) grad_norm_before_clipping \(last (\d+) steps\): mean=([\d.]+)")
DIAGNOSTICS_LINE = re.compile(r"step=(\d+) diagnostics (\{.*\})$")
TIME_LINE = re.compile(
    r"step=(\d+) seconds_per_step teacher=([\d.]+) prepare=([\d.]+) "
    r"student_including_prepare=([\d.]+) .*peak_memory_gib=([\d.]+)"
)


def literal(text):
    return ast.literal_eval(re.sub(r"\bnan\b", "None", text))


def job_versions():
    """Map job id -> version from experiment_submissions.log."""
    mapping, current = {}, None
    for line in SUBMISSION_LOG.read_text().splitlines():
        header = re.match(r"===== .*\| job (\d+) \| user (\w+)", line)
        if header:
            current = header.group(1)
            continue
        output = re.search(r"REFINEMENT_OUTPUT_DIR=.*/([^/\s]+)$", line)
        if output and current and current not in mapping:
            mapping[current] = output.group(1)
    return mapping


def parse_header(lines):
    header = {"labeled_pairs_per_step": 0, "passes": 1, "total_steps": None}
    for line in lines:
        match = re.search(r"Loss coefficients:\s+human=([\d.]+), G=([\d.]+), mask=([\d.]+), set=([\d.]+)", line)
        if match:
            header.update(zip(("lambda_human", "lambda_gemma", "lambda_mask", "lambda_set"), map(float, match.groups())))
        match = re.search(r"labeled pairs per step: (\d+)", line)
        if match:
            header["labeled_pairs_per_step"] = int(match.group(1))
        match = re.search(r"Learning rate:\s+([\d.e-]+); heads: ([^;]+);.*schedule: (\S+), warmup ratio ([\d.]+)", line)
        if match:
            header["learning_rate"] = float(match.group(1))
            header["head_learning_rate"] = match.group(2).strip()
            header["scheduler"] = match.group(3).rstrip(",")
            header["warmup_ratio"] = float(match.group(4))
        match = re.search(r"EMA decay: .* over (\d+) steps", line)
        if match:
            header["total_steps"] = int(match.group(1))
        match = re.search(r"Unlabeled subset / passes: (\d+) / (\d+)", line)
        if match:
            header["subset"], header["passes"] = int(match.group(1)), int(match.group(2))
        match = re.search(r"Teacher MC passes: (\d+)", line)
        if match:
            header["mc_passes"] = int(match.group(1))
        if line.startswith("Set ramp / EMA:"):
            header["set_ramp"] = line.split(":", 1)[1].strip()
    return header


def parse_job(job):
    lines = (LOG_DIR / f"train_loop_{job}.out").read_text().splitlines()
    run = {"job": job, "header": parse_header(lines), "steps": [], "grad": {}, "time": {}, "diagnostics": {},
           "initial": None, "student_end": None, "teacher_end": None, "probe_initial": None,
           "finished": any(line.startswith("Refinement checkpoint:") for line in lines)}
    for line in lines:
        match = STEP_LINE.match(line)
        if match:
            g = match.groups()
            statuses = literal(g[9])
            run["steps"].append({
                "round": int(g[0]), "step": int(g[1]), "human": float(g[2]), "gemma": float(g[3]),
                "mask": float(g[4]), "set": float(g[5]), "lambda_set": float(g[6]),
                "ema_decay": float(g[8]), "statuses": statuses,
            })
            continue
        match = DIAGNOSTICS_LINE.match(line)
        if match:
            run["diagnostics"][int(match.group(1))] = json.loads(match.group(2))
            continue
        match = GRAD_LINE.match(line)
        if match and match.group(2) == "1":
            run["grad"][int(match.group(1))] = float(match.group(3))
            continue
        match = TIME_LINE.match(line)
        if match:
            run["time"][int(match.group(1))] = tuple(map(float, match.groups()[1:]))
            continue
        if line.startswith("Initial Student Human validation:"):
            run["initial"] = literal(line.split(": ", 1)[1])
        elif re.match(r"Round \d+ Student validation:", line):
            run["student_end"] = literal(line.split(": ", 1)[1])
        elif line.startswith("Completed round"):
            text = line.split("validation=", 1)[1].rsplit(" remaining", 1)[0]
            run["teacher_end"] = literal(text)
        elif line.startswith("Initial fixed-probe diagnostics:"):
            match = re.search(r"width=([\d.]+) inside=([\d.]+) projection_kl=([\d.]+)", line)
            run["probe_initial"] = dict(zip(("width", "inside", "projection_kl"), map(float, match.groups())))
    run["per_step"] = per_step_values(run)
    summaries = sorted((OUTPUT_TRAINING / run.get("version", "")).glob("round_*_summary.json")) if run.get("version") else []
    run["summaries"] = summaries
    return run


def per_step_values(run):
    """Undo the running means (per round) into per-step values."""
    pairs = run["header"]["labeled_pairs_per_step"]
    values, previous = [], None
    for entry in run["steps"]:
        if previous is not None and previous["round"] != entry["round"]:
            previous = None
        unlabeled_count = sum(entry["statuses"].values())
        steps_in_round = entry["step"] - (previous_round_last_step(run, entry))
        labeled_count = pairs * steps_in_round
        row = {"step": entry["step"], "lambda_set": entry["lambda_set"], "ema_decay": entry["ema_decay"]}
        for key, count in (("human", labeled_count), ("gemma", labeled_count), ("mask", labeled_count), ("set", unlabeled_count)):
            if previous is None:
                previous_total, previous_count = 0.0, 0
            else:
                previous_unlabeled = sum(previous["statuses"].values())
                previous_steps = previous["step"] - previous_round_last_step(run, previous)
                previous_count = previous_unlabeled if key == "set" else pairs * previous_steps
                previous_total = previous[key] * previous_count
            delta = count - previous_count
            row[key] = (entry[key] * count - previous_total) / delta if delta > 0 else math.nan
        new_status = dict(entry["statuses"])
        if previous is not None:
            for name, value in previous["statuses"].items():
                new_status[name] = new_status.get(name, 0) - value
        total_new = sum(new_status.values())
        row["conflict_fraction"] = new_status.get("constraint_conflict", 0) / total_new if total_new else math.nan
        values.append(row)
        previous = entry
    return values


def previous_round_last_step(run, entry):
    last = 0
    for other in run["steps"]:
        if other["round"] < entry["round"]:
            last = max(last, other["step"])
    return last


def rolling(values, window=ROLLING):
    out = []
    for index in range(len(values)):
        chunk = [v for v in values[max(0, index - window + 1): index + 1] if not math.isnan(v)]
        out.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return out


def learning_rate_curve(header, steps):
    """Reconstructed from the logged schedule settings (the log has no per-step learning rate)."""
    total = header.get("total_steps") or (steps[-1] if steps else 1)
    warmup = math.ceil(header.get("warmup_ratio", 0.0) * total)
    base = header.get("learning_rate", math.nan)
    curve = []
    for step in steps:
        index = step - 1  # learning rate used by optimizer step `step`
        if index < warmup:
            factor = index / max(1, warmup)
        elif header.get("scheduler") == "cosine":
            progress = (index - warmup) / max(1, total - warmup)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        else:
            factor = 1.0
        curve.append(base * factor)
    return curve


def has_labeled(run):
    return run["header"]["labeled_pairs_per_step"] > 0 and any(
        run["header"].get(key, 0) > 0 for key in ("lambda_human", "lambda_gemma", "lambda_mask"))


def finish(fig, path, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def plot_series(ax, steps, values, color, label_raw="per step", label_mean=f"mean of last {ROLLING} steps"):
    ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.35, label=label_raw)
    ax.plot(steps, rolling(values), color=color, linewidth=2.0, label=label_mean)


def plot_losses(run, out_dir):
    rows, header = run["per_step"], run["header"]
    steps = [r["step"] for r in rows]
    panels = []
    if has_labeled(run):
        panels += [("human", "HumanGT: CE(one-hot, overall)"),
                   ("gemma", "GemmaGT preference (factorized)"),
                   ("mask", "GemmaGT relevance BCE")]
    panels.append(("set", "Unlabeled: CE(Inspector soft label, Student) + λ_C·KL(Teacher‖Student)"))
    panels.append(("total", "Weighted sum λ_H·L_H + λ_G·L_G + λ_M·L_M + λ_U(t)·L_U "
                            "(logged means, per-pair normalisation not applied)"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        if key == "total":
            values = [header.get("lambda_human", 0) * r["human"] + header.get("lambda_gemma", 0) * r["gemma"]
                      + header.get("lambda_mask", 0) * r["mask"] + r["lambda_set"] * r["set"] for r in rows]
        else:
            values = [r[key] for r in rows]
        plot_series(ax, steps, values, SERIES[0])
        ax.set_title(title)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("optimizer step")
    finish(fig, out_dir / "stage2_tuning_losses.png", f"{run['version']} — Stage 2 losses (job {run['job']})")


def plot_gradient(run, out_dir):
    grad = run["grad"]
    steps = sorted(grad)
    values = [grad[s] for s in steps]
    clipped = sum(v > 1.0 for v in values)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    plot_series(ax, steps, values, SERIES[0])
    ax.axhline(1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
    ax.text(steps[0], 1.08, "max_grad_norm = 1.0", color=TEXT_SECONDARY, fontsize=9, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("gradient norm before clipping")
    median = sorted(values)[len(values) // 2] if values else math.nan
    ax.set_title(f"Clipped steps: {clipped}/{len(values)}; median {median:.1f}, max {max(values):.1f}")
    ax.legend(loc="upper right")
    finish(fig, out_dir / "stage2_tuning_gradient_norm.png", f"{run['version']} — gradient norm (job {run['job']})")


def plot_schedule(run, out_dir):
    rows = run["per_step"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.4), sharex=True)
    axes[0].plot(steps, learning_rate_curve(run["header"], steps), color=SERIES[0])
    axes[0].set_title(f"Backbone learning rate (reconstructed from schedule: {run['header'].get('scheduler')}, "
                      f"warmup {run['header'].get('warmup_ratio')}; heads: {run['header'].get('head_learning_rate')})")
    axes[1].plot(steps, [r["lambda_set"] for r in rows], color=SERIES[0])
    axes[1].set_title("λ_U(t) (logged)")
    axes[2].plot(steps, [r["ema_decay"] for r in rows], color=SERIES[0])
    axes[2].set_title("Teacher EMA decay (logged)")
    plot_series(axes[3], steps, [r["conflict_fraction"] for r in rows], SERIES[0])
    axes[3].set_title("Share of Unlabeled in the step with status constraint_conflict (Teacher against Inspector)")
    axes[3].legend(loc="upper right")
    axes[-1].set_xlabel("optimizer step")
    finish(fig, out_dir / "stage2_tuning_schedule.png", f"{run['version']} — schedules and Unlabeled status (job {run['job']})")


def pass_means(run):
    passes = run["header"].get("passes", 1)
    rows = run["per_step"]
    if passes <= 1 or not rows:
        return []
    size = len(rows) // passes
    out = []
    for index in range(passes):
        chunk = [r["set"] for r in rows[index * size:(index + 1) * size]]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_passes(run, out_dir):
    means = pass_means(run)
    if not means:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    positions = list(range(1, len(means) + 1))
    ax.bar(positions, means, width=0.55, color=SERIES[0])
    for x, y in zip(positions, means):
        ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY)
    low = min(means)
    ax.set_ylim(low - 0.05, max(means) + 0.02)
    ax.set_xticks(positions, [f"pass {p}" for p in positions])
    ax.set_ylabel("mean per-step Unlabeled loss")
    ax.set_title(f"{run['header'].get('subset')} Unlabeled samples, {len(means)} passes (y axis does not start at 0)")
    finish(fig, out_dir / "stage2_tuning_unlabeled_loss_by_pass.png", f"{run['version']} — Unlabeled loss per pass (job {run['job']})")


def short(version):
    return version.replace("stage2_tuning_", "")


def families(runs):
    """Runs with the same number of steps share comparison charts (at most 8 per chart)."""
    groups = {}
    for run in runs:
        groups.setdefault(len(run["per_step"]), []).append(run)
    out = []
    for steps, members in sorted(groups.items()):
        for start in range(0, len(members), len(SERIES)):
            out.append((steps, members[start:start + len(SERIES)]))
    return out


def plot_comparison_lines(runs, key_list, filename, title, log_scale=False):
    for steps, members in families(runs):
        keys = [k for k in key_list if k != "human" or any(has_labeled(r) for r in members)]
        if key_list != ["grad"]:
            keys = [k for k in keys if k == "set" or any(has_labeled(r) for r in members)]
        if not keys:
            continue
        fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3.2 * len(keys)), sharex=True, squeeze=False)
        for ax, key in zip(axes[:, 0], keys):
            for index, run in enumerate(members):
                if key == "grad":
                    xs = sorted(run["grad"])
                    ys = [run["grad"][s] for s in xs]
                else:
                    if key != "set" and not has_labeled(run):
                        continue
                    xs = [r["step"] for r in run["per_step"]]
                    ys = [r[key] for r in run["per_step"]]
                ax.plot(xs, rolling(ys), color=SERIES[index], label=short(run["version"]))
            ax.set_title({"human": "HumanGT loss", "gemma": "GemmaGT preference loss", "mask": "GemmaGT relevance BCE",
                          "set": "Unlabeled loss", "grad": "gradient norm before clipping (max_grad_norm = 1.0)"}[key]
                         + f" — mean of last {ROLLING} steps")
            if log_scale:
                ax.set_yscale("log")
        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        axes[-1, 0].set_xlabel("optimizer step")
        finish(fig, COMPARISON_DIR / filename.format(steps=steps), title + f" ({steps}-step runs)")


GEMMA_METRICS = [("gemma_preference_loss", "Preference loss"), ("gemma_mask_bce", "Relevance BCE"),
                 ("gemma_tie_brier", "Tie Brier"), ("gemma_winner_brier", "Winner Brier")]


def plot_gemma_monitor(runs):
    fig, axes = plt.subplots(len(GEMMA_METRICS), 1, figsize=(10, 3.0 * len(GEMMA_METRICS)), sharex=True)
    labels = [short(r["version"]) for r in runs]
    positions = list(range(len(runs)))
    width = 0.36
    for ax, (key, title) in zip(axes, GEMMA_METRICS):
        initial = runs[0]["initial"][key]
        student = [r["student_end"][key] for r in runs]
        teacher = [r["teacher_end"][key] for r in runs]
        ax.bar([p - width / 2 - 0.01 for p in positions], student, width=width, color=SERIES[0], label="Student, end")
        ax.bar([p + width / 2 + 0.01 for p in positions], teacher, width=width, color=SERIES[1], label="Teacher (EMA), end")
        ax.axhline(initial, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        for x, y in zip(positions, student):
            ax.text(x - width / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
        low = min(student + teacher + [initial])
        high = max(student + teacher + [initial])
        ax.set_ylim(low - 0.15 * (high - low) - 1e-3, high + 0.15 * (high - low) + 1e-3)
        ax.set_title(f"{title} (lower is better; dashed: start {initial:.3f}; y axis does not start at 0)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xticks(positions, labels, rotation=20, ha="right")
    finish(fig, COMPARISON_DIR / "gemma_monitor_by_version.png",
           "32 GemmaGT monitoring pairs (inside the 1008 training pairs), eval mode, start vs end")


def probe_end(run):
    if not run["summaries"]:
        return {}
    data = json.loads(run["summaries"][-1].read_text())
    data = data[-1] if isinstance(data, list) else data
    return data.get("fixed_probe", {})


def plot_probe(runs):
    labels = [short(r["version"]) for r in runs]
    positions = list(range(len(runs)))
    kl = [probe_end(r).get("mean_projection_kl", math.nan) for r in runs]
    inside = [probe_end(r).get("inside_rate", math.nan) for r in runs]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    axes[0].bar(positions, kl, width=0.55, color=SERIES[0])
    axes[0].set_title(f"Projection KL of Student onto the Teacher box, end (start {runs[0]['probe_initial']['projection_kl']:.6f})")
    axes[1].bar(positions, inside, width=0.55, color=SERIES[0])
    axes[1].axhline(runs[0]["probe_initial"]["inside"], color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
    axes[1].set_title(f"Share of probe samples with Student inside the Teacher box, end "
                      f"(dashed: start {runs[0]['probe_initial']['inside']:.3f})")
    axes[1].set_ylim(0, 1.05)
    for ax, values in zip(axes, (kl, inside)):
        for x, y in zip(positions, values):
            ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
    axes[-1].set_xticks(positions, labels, rotation=20, ha="right")
    finish(fig, COMPARISON_DIR / "probe_by_version.png", "Fixed probe: 32 Unlabeled samples")


def write_summary(runs):
    path = COMPARISON_DIR / "summary.csv"
    fields = ["job", "version", "steps", "labeled_pairs_per_step", "lambda_human", "lambda_gemma", "lambda_mask",
              "lambda_set", "learning_rate", "head_learning_rate", "scheduler", "mc_passes",
              "human_first10", "human_last10", "gemma_first10", "gemma_last10", "mask_first10", "mask_last10",
              "unlabeled_first10", "unlabeled_last10", "unlabeled_by_pass",
              "grad_norm_median", "grad_norm_max", "clipped_steps",
              "monitor_preference_start", "monitor_preference_student_end", "monitor_preference_teacher_end",
              "monitor_mask_bce_student_end", "monitor_tie_brier_student_end", "monitor_winner_brier_student_end",
              "probe_projection_kl_end", "probe_inside_end", "mean_seconds_per_step", "peak_memory_gib"]
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            rows, header = run["per_step"], run["header"]

            def window(key, last):
                chunk = rows[-10:] if last else rows[:10]
                if key != "set" and not has_labeled(run):
                    return ""
                return round(sum(r[key] for r in chunk) / len(chunk), 4)

            grads = sorted(run["grad"].values())
            times = list(run["time"].values())
            probe = probe_end(run)
            writer.writerow({
                "job": run["job"], "version": run["version"], "steps": len(rows),
                "labeled_pairs_per_step": header["labeled_pairs_per_step"],
                "lambda_human": header.get("lambda_human"), "lambda_gemma": header.get("lambda_gemma"),
                "lambda_mask": header.get("lambda_mask"), "lambda_set": header.get("lambda_set"),
                "learning_rate": header.get("learning_rate"), "head_learning_rate": header.get("head_learning_rate"),
                "scheduler": header.get("scheduler"), "mc_passes": header.get("mc_passes"),
                "human_first10": window("human", False), "human_last10": window("human", True),
                "gemma_first10": window("gemma", False), "gemma_last10": window("gemma", True),
                "mask_first10": window("mask", False), "mask_last10": window("mask", True),
                "unlabeled_first10": window("set", False), "unlabeled_last10": window("set", True),
                "unlabeled_by_pass": " ".join(f"{v:.4f}" for v in pass_means(run)),
                "grad_norm_median": round(grads[len(grads) // 2], 2), "grad_norm_max": round(grads[-1], 2),
                "clipped_steps": f"{sum(g > 1.0 for g in grads)}/{len(grads)}",
                "monitor_preference_start": round(run["initial"]["gemma_preference_loss"], 4),
                "monitor_preference_student_end": round(run["student_end"]["gemma_preference_loss"], 4),
                "monitor_preference_teacher_end": round(run["teacher_end"]["gemma_preference_loss"], 4),
                "monitor_mask_bce_student_end": round(run["student_end"]["gemma_mask_bce"], 4),
                "monitor_tie_brier_student_end": round(run["student_end"]["gemma_tie_brier"], 4),
                "monitor_winner_brier_student_end": round(run["student_end"]["gemma_winner_brier"], 4),
                "probe_projection_kl_end": round(probe.get("mean_projection_kl", math.nan), 5),
                "probe_inside_end": probe.get("inside_rate"),
                "mean_seconds_per_step": round(sum(t[0] + t[2] for t in times) / len(times), 1) if times else "",
                "peak_memory_gib": max(t[3] for t in times) if times else "",
            })
    print(f"saved {path}")


def plot_step_diagnostics(run, out_dir):
    """Per-step diagnostics: losses before lambda, gradient norm by source, ratio, cosine, update size."""
    diagnostics = run["diagnostics"]
    if not diagnostics:
        return
    steps = sorted(diagnostics)
    def series(getter):
        values = []
        for step in steps:
            try:
                value = getter(diagnostics[step])
            except (KeyError, TypeError):
                value = None
            values.append(float("nan") if value is None else float(value))
        return values
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    ax = axes[0][0]
    for color, key, label in ((SERIES[0], "human", "HumanGT CE"), (SERIES[1], "gemma", "Gemma preference"),
                              (SERIES[2], "mask", "Gemma relevance BCE")):
        values = series(lambda d, k=key: d["losses"][k])
        ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.3)
        ax.plot(steps, rolling(values), color=color, label=label)
    ax.set_title("Labeled losses of the step, before lambda (thin: per step; thick: rolling mean)")
    ax.legend(loc="upper right")
    ax = axes[0][1]
    for color, key, label in ((SERIES[3], "unlabeled_inspector_kl", "KL(Inspector || Student)"),
                              (SERIES[4], "unlabeled_teacher_kl", "KL(Teacher || Student)")):
        values = series(lambda d, k=key: d["losses"][k])
        ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.3)
        ax.plot(steps, rolling(values), color=color, label=label)
    ax.set_title("Unlabeled losses of the step, before lambda")
    ax.legend(loc="upper right")
    ax = axes[1][0]
    for color, key, label in ((SERIES[0], "labeled", "labeled"), (SERIES[3], "unlabeled", "Unlabeled"),
                              (TEXT_SECONDARY, "total", "total")):
        ax.plot(steps, series(lambda d, k=key: d["gradient_norm_before_clipping"]["all"][k]), color=color, label=label)
    ax.axhline(1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("Gradient norm before clipping by source (dashed: clip 1.0)")
    ax.legend(loc="upper right")
    ax = axes[1][1]
    values = series(lambda d: d["gradient_norm_before_clipping"]["all"]["unlabeled_over_labeled"])
    ax.plot(steps, values, color=SERIES[3], linewidth=1.0, alpha=0.3)
    ax.plot(steps, rolling(values), color=SERIES[3])
    ax.set_yscale("log")
    ax.set_title("Effective weight: Unlabeled / labeled gradient norm")
    ax = axes[2][0]
    groups = [g for g in ("all", "vit", "aligner", "llm", "heads") if g in diagnostics[steps[0]]["gradient_norm_before_clipping"]]
    for color, group in zip(SERIES, groups):
        values = series(lambda d, g=group: d["gradient_norm_before_clipping"][g]["cosine"])
        ax.plot(steps, rolling(values), color=color, label=group)
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=1.0)
    ax.set_title(f"Cosine of labeled and Unlabeled gradients (rolling mean of {ROLLING} steps)")
    ax.legend(loc="upper right", ncol=len(groups))
    ax = axes[2][1]
    for color, group in zip(SERIES, groups):
        ax.plot(steps, series(lambda d, g=group: d["update_ratio"].get(g)), color=color, label=group)
    ax.set_yscale("log")
    ax.set_title("Update size per step |dW| / |W| by group")
    ax.legend(loc="upper right", ncol=len(groups))
    ax = axes[3][0]
    ax.plot(steps, series(lambda d: d["lambda_unlabeled"]), color=SERIES[5])
    ax.set_title("lambda_U(t)")
    ax.set_xlabel("Step")
    ax = axes[3][1]
    for position, color in zip(range(len(diagnostics[steps[0]].get("learning_rates", []))), SERIES):
        ax.plot(steps, series(lambda d, k=position: d["learning_rates"][k]), color=color,
                label=f"optimizer group {position}")
    ax.set_title("Learning rate by optimizer group")
    ax.legend(loc="upper right")
    ax.set_xlabel("Step")
    finish(fig, out_dir / "stage2_tuning_step_diagnostics.png", f"{run['version']}: step diagnostics")


TRAINING_DATA_PANELS = (
    ("untrained_unlabeled", "inspector_kl", "KL(Inspector || Student), untrained Unlabeled"),
    ("trained", "inspector_kl", "KL(Inspector || Student), trained Unlabeled"),
    ("untrained_unlabeled", "student_mean_absolute_margin", "Student mean |d|, untrained Unlabeled"),
    ("labeled", "human_three_way_cross_entropy", "HumanGT three-class CE, labeled pairs"),
    ("labeled", "gemma_preference_loss_majority_vote", "Gemma preference loss, labeled pairs"),
    ("labeled", "gemma_relevance_loss_majority_vote", "Gemma relevance BCE, labeled pairs"),
)


def submitted_student_checkpoint(version):
    current, student = None, None
    for line in SUBMISSION_LOG.read_text().splitlines():
        match = re.match(r"version:\s+(\S+)", line)
        if match:
            current = match.group(1)
        match = re.search(r"TRAIN_LOOP_STUDENT_CHECKPOINT=(\S+)", line)
        if match and current == version:
            student = match.group(1)
    return student


def training_data_points(version):
    """{(set kind, metric): {step: value}} from stage2_tuning_diagnostics; 1008-pair files override 'labeled'."""
    points = {}
    files = []
    student = submitted_student_checkpoint(version)
    if student:
        stage1 = RESULTS.parent / "student" / Path(student).parent.name / "stage2_tuning_diagnostics"
        files += [(0, stage1 / "student_warmstart.json"), (0, stage1 / "student_warmstart_labeled_all.json")]
    for path in sorted((RESULTS / version / "stage2_tuning_diagnostics").glob("checkpoint-*.json")):
        match = re.match(r"checkpoint-(\d+)", path.stem)
        files.append((int(match.group(1)), path))
    # The trained set scored at the checkpoints; step-0 files (Student-Stage1) hold every
    # trained set, so the same key must be read there (not simply the first one).
    run_trained_keys = set()
    for step, path in files:
        if step > 0 and path.exists() and path.stat().st_size > 0:
            run_trained_keys.update(
                key for key in json.loads(path.read_text())["metrics"] if key.startswith("trained_unlabeled_")
            )
    if len(run_trained_keys) > 1:
        raise ValueError(f"{version}: several trained sets at the checkpoints: {sorted(run_trained_keys)}")
    for step, path in files:
        if not path.exists() or path.stat().st_size == 0:
            continue
        metrics = json.loads(path.read_text())["metrics"]
        labeled_all = path.stem.endswith("_labeled_all")
        for kind, name, _ in TRAINING_DATA_PANELS:
            if kind == "trained":
                keys = sorted(run_trained_keys) or [key for key in metrics if key.startswith("trained_unlabeled_")]
                value = metrics[keys[0]].get(name) if keys and keys[0] in metrics else None
            elif kind == "labeled":
                # A 1008-pair file also holds the 256-pair 'labeled' report (same rows).
                for kind_key in (("labeled_all", "labeled") if labeled_all else ("labeled",)):
                    value = metrics.get(kind_key, {}).get(name)
                    if value is not None:
                        points.setdefault((kind_key, name), {})[step] = value
                continue
            else:
                value = metrics.get(kind, {}).get(name)
            if value is not None:
                points.setdefault((kind, name), {})[step] = value
    return points


def plot_training_data(versions, path, title):
    data = {version: training_data_points(version) for version in versions}
    # Nothing to draw before any Stage 2 checkpoint has been evaluated (step 0 alone).
    if not any(step > 0 for points in data.values() for values in points.values() for step in values):
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for ax, (kind, name, label) in zip(axes.flat, TRAINING_DATA_PANELS):
        # Sets not scored at any checkpoint (e.g. untrained Unlabeled skipped) are left empty.
        if not any(step > 0 for version in versions for step in data[version].get((kind, name), {})):
            ax.set_title(label + " (not evaluated)")
            ax.axis("off")
            continue
        for color, version in zip(SERIES, versions):
            for key, style, suffix in (((kind, name), "-", ""), (("labeled_all", name), ":", " (1008 pairs)")):
                if key[0] == "labeled_all" and kind != "labeled":
                    continue
                values = data[version].get(key)
                if not values:
                    continue
                steps = sorted(values)
                ax.plot(steps, [values[s] for s in steps], color=color, linestyle=style, marker="o", markersize=5,
                        label=short(version) + suffix)
        ax.set_title(label)
        ax.set_xlabel("Step (0 = Student-Stage1)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.06 + 0.02 * (len(labels) // 4), 1, 0.97))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", nargs="*", help="job ids (default: every finished stage2_tuning_ job in the submission log)")
    parser.add_argument("--versions", nargs="*",
                        help="version names (latest job of each); plots only these runs")
    parser.add_argument("--comparison-name", default=None,
                        help="with --versions: also write stage2_tuning_comparison/<name>_*.png for these versions")
    args = parser.parse_args()
    mapping = job_versions()
    if args.versions:
        latest = {}
        for job, version in mapping.items():
            if version in args.versions and int(job) > int(latest.get(version, 0)):
                latest[version] = job
        missing = sorted(set(args.versions) - set(latest))
        if missing:
            print(f"versions without a submitted job: {missing}")
        jobs = [latest[v] for v in args.versions if v in latest]
    else:
        jobs = args.jobs or [j for j, v in mapping.items() if v.startswith("stage2_tuning_") and "timing_test" not in v]
    runs = []
    for job in jobs:
        version = mapping.get(job)
        if version is None or not (LOG_DIR / f"train_loop_{job}.out").exists():
            print(f"skip job {job}: no version or no log")
            continue
        run = parse_job(job)
        run["version"] = version
        run["summaries"] = sorted((OUTPUT_TRAINING / version).glob("round_*_summary.json"))
        if not run["finished"]:
            print(f"skip job {job} ({version}): not finished")
            continue
        runs.append(run)
    for run in runs:
        out_dir = RESULTS / run["version"]
        plot_losses(run, out_dir)
        plot_gradient(run, out_dir)
        plot_schedule(run, out_dir)
        plot_passes(run, out_dir)
        plot_step_diagnostics(run, out_dir)
        plot_training_data([run["version"]], out_dir / "stage2_tuning_training_data_evaluation.png",
                           f"{run['version']}: evaluation on the Stage 2 training data")
    if args.versions and args.comparison_name:
        names = [run["version"] for run in runs]
        plot_training_data(names, COMPARISON_DIR / f"{args.comparison_name}_training_data_evaluation.png",
                           f"{args.comparison_name}: evaluation on the Stage 2 training data")
    if runs and not args.versions:
        plot_comparison_lines(runs, ["set", "human", "gemma", "mask"], "loss_by_version_{steps}_steps.png", "Training losses by version")
        plot_comparison_lines(runs, ["grad"], "gradient_norm_by_version_{steps}_steps.png", "Gradient norm by version", log_scale=True)
        plot_gemma_monitor(runs)
        # Versions with --diagnostic-probe-size 0 have no fixed probe.
        probe_runs = [r for r in runs if r["probe_initial"] is not None]
        if probe_runs:
            plot_probe(probe_runs)
        write_summary(runs)


if __name__ == "__main__":
    main()
