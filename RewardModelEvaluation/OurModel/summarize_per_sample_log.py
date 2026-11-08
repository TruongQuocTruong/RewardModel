"""Summarize the per-sample log of a Stage 2 tuning run (no GPU, no dataset read).

The log (train_loop.py --per-sample-log) has one JSON row per pair per optimizer step:
Unlabeled rows carry the three-way labels of Teacher, Student and Inspector and whether
they agree; labeled rows carry the human label and the Student label. Teacher = the
dropout pass used as the loss target; Student = the training forward (train mode, before
that step's update).

Usage:
  python summarize_per_sample_log.py --version <version> [--job <job id>]

Without --job, the last job of the version in experiment_submissions.log is used.
Outputs (evaluation_results/train_loop/<version>/per_sample/):
  per_sample_summary.json        rates per step, overall, transitions, by Inspector confidence
  student_teacher_disagreements.jsonl  every Unlabeled row where Student != Teacher
  per_sample_agreement.png       agreement rates per step + transitions + by confidence
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/datastore/truongtq/TrainingRewardModel")
PROJECT = ROOT / "TrainingRewardModel"
LOG_DIR = PROJECT / "logs"  # per-sample logs of jobs before 2026-11-05
OUTPUT_TRAINING = PROJECT / "Qwen3-VL-Embedding" / "output_training"
SUBMISSION_LOG = PROJECT / "experiment_submissions.log"
RESULTS = ROOT / "RewardModelEvaluation" / "OurModel" / "evaluation_results" / "train_loop"

CLASSES = ("A", "B", "Tie")
# Inspector confidence bins (largest probability), as in stage2_tuning.md section 7.6.
CONFIDENCE_BINS = ((0.0, 0.45), (0.45, 0.55), (0.55, 1.01))
UNLABELED_FLAGS = ("student_same_as_teacher", "student_same_as_inspector", "teacher_same_as_inspector")
FLAG_NAMES = {
    "student_same_as_teacher": "Student = Teacher",
    "student_same_as_inspector": "Student = Inspector",
    "teacher_same_as_inspector": "Teacher = Inspector",
    "student_same_as_human": "Student = nhãn người (cặp có nhãn)",
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ROLLING = 10

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT_PRIMARY, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2.0,
})


def last_job_of(version: str) -> str:
    job = None
    pattern = re.compile(r"^===== .* \| job (\d+) \|")
    current = None
    for line in SUBMISSION_LOG.read_text().splitlines():
        match = pattern.match(line)
        if match:
            current = match.group(1)
        elif line.startswith("version:") and line.split(":", 1)[1].strip() == version and current:
            job = current
    if job is None:
        raise SystemExit(f"No submitted job of {version} in {SUBMISSION_LOG}")
    return job


def rate(values) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else None


def rolling(values, window):
    out = []
    for i in range(len(values)):
        part = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        out.append(sum(part) / len(part) if part else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--job", default=None)
    parser.add_argument("--log", default=None, help="log path override (code checks only)")
    parser.add_argument("--output-dir", default=None, help="output override (code checks only)")
    args = parser.parse_args()
    job = args.job or ("manual" if args.log else last_job_of(args.version))
    if args.log:
        log_path = Path(args.log)
    else:
        # From 2026-11-05 the log is in the output folder; earlier jobs kept it in logs/.
        log_path = OUTPUT_TRAINING / args.version / f"per_sample_log_{job}.jsonl"
        if not log_path.exists():
            log_path = LOG_DIR / f"train_loop_{job}_per_sample.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    unlabeled = [r for r in rows if r["kind"] == "unlabeled"]
    labeled = [r for r in rows if r["kind"] == "labeled"]
    missing = [r for r in unlabeled if any(key not in r for key in UNLABELED_FLAGS + ("teacher_label",))]
    if missing:
        raise SystemExit(f"{len(missing)} Unlabeled rows lack the label fields (log written by older code).")

    steps = sorted({r["step"] for r in rows})
    by_step_u = defaultdict(list)
    by_step_l = defaultdict(list)
    for r in unlabeled:
        by_step_u[r["step"]].append(r)
    for r in labeled:
        by_step_l[r["step"]].append(r)
    per_step = []
    for step in steps:
        entry = {"step": step, "unlabeled": len(by_step_u[step]), "labeled": len(by_step_l[step])}
        for flag in UNLABELED_FLAGS:
            entry[flag] = rate(r[flag] for r in by_step_u[step])
        entry["student_same_as_human"] = rate(r["student_same_as_human"] for r in by_step_l[step])
        entry["teacher_kl_mean"] = rate(r["teacher_kl"] for r in by_step_u[step])
        per_step.append(entry)

    ids = Counter(r["sample_id"] for r in unlabeled)
    transitions = Counter(f"Teacher {r['teacher_label']} → Student {r['student_label']}"
                          for r in unlabeled if not r["student_same_as_teacher"])
    by_confidence = []
    for low, high in CONFIDENCE_BINS:
        part = [r for r in unlabeled if low <= max(r["inspector"]) < high]
        by_confidence.append({"inspector_max_probability": [low, min(high, 1.0)], "count": len(part),
                              **{flag: rate(r[flag] for r in part) for flag in UNLABELED_FLAGS}})
    # Online augmentation of the Student input: rates by scenario and by A<->B swap.
    def group_report(part):
        return {"count": len(part),
                **{flag: rate(r[flag] for r in part) for flag in UNLABELED_FLAGS},
                "teacher_kl_mean": rate(r["teacher_kl"] for r in part),
                "inspector_kl_mean": rate(r["inspector_kl"] for r in part)}
    augmented = [r for r in unlabeled if r.get("augmentation_scenario")]
    by_augmentation = None
    if augmented:
        scenarios = sorted({r["augmentation_scenario"] for r in augmented})
        by_augmentation = {
            "by_scenario": {name: group_report([r for r in augmented if r["augmentation_scenario"] == name])
                            for name in scenarios},
            "by_swap": {str(flag): group_report([r for r in augmented if r["augmentation_swap"] == flag])
                        for flag in (False, True)},
            "fallback_to_original_instruction": sum(1 for r in augmented if r.get("augmentation_fallback")),
        }
        combined = [r for r in augmented if r.get("augmentation_text_operation")]
        if combined:
            # Combined augmentation: a row counts for every operation it used.
            text_names = sorted({n for r in combined for n in r["augmentation_text_operation"].split("+")})
            by_augmentation["by_text_operation"] = {
                name: group_report([r for r in combined if name in r["augmentation_text_operation"].split("+")])
                for name in text_names}
            by_augmentation["by_layout"] = {
                name: group_report([r for r in combined if r["augmentation_layout"] == name])
                for name in sorted({r["augmentation_layout"] for r in combined})}
            degradation_names = sorted({d["name"] for r in combined for d in r["augmentation_degradation"]})
            by_augmentation["by_degradation"] = {
                **{f"{n} degradations": group_report([r for r in combined if len(r["augmentation_degradation"]) == n])
                   for n in (0, 1, 2)},
                **{name: group_report([r for r in combined
                                       if any(d["name"] == name for d in r["augmentation_degradation"])])
                   for name in degradation_names}}
    summary = {
        "version": args.version, "job": job, "log": str(log_path),
        "definitions": {
            "teacher": "Teacher dropout pass used as the loss target (aggregated over the metrics the Teacher selects)",
            "student": "Student training forward (train mode, dropout on, before the step's update)",
            "label": "argmax of the three-way probabilities (A / B / Tie)",
        },
        "checks": {
            "steps": len(steps), "unlabeled_rows": len(unlabeled), "labeled_rows": len(labeled),
            "unlabeled_rows_per_step": sorted(Counter(e["unlabeled"] for e in per_step).items()),
            "labeled_rows_per_step": sorted(Counter(e["labeled"] for e in per_step).items()),
            "unlabeled_seen_more_than_once": sum(1 for c in ids.values() if c > 1),
        },
        "overall": {
            **{flag: rate(r[flag] for r in unlabeled) for flag in UNLABELED_FLAGS},
            "student_same_as_human": rate(r["student_same_as_human"] for r in labeled),
            "label_counts": {who: dict(Counter(r[f"{who}_label"] for r in unlabeled))
                             for who in ("teacher", "student", "inspector")},
        },
        "student_teacher_transitions": dict(transitions.most_common()),
        "by_inspector_confidence": by_confidence,
        "by_augmentation": by_augmentation,
        "per_step": per_step,
    }
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS / args.version / "per_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_sample_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    with (out_dir / "student_teacher_disagreements.jsonl").open("w") as handle:
        for r in unlabeled:
            if not r["student_same_as_teacher"]:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    panels = 4 if by_augmentation else 3
    wide = 6 if by_augmentation and "by_text_operation" in by_augmentation else 0
    fig, axes = plt.subplots(1, panels, figsize=(17 + 5 * (panels - 3) + wide, 4.8),
                             gridspec_kw={"width_ratios": [2.2, 1.3, 1.3, 1.6][:panels]})
    ax = axes[0]
    for color, flag in zip(SERIES, UNLABELED_FLAGS + ("student_same_as_human",)):
        values = [e[flag] for e in per_step]
        if all(v is None for v in values):
            continue
        ax.plot(steps, [None if v is None else 100 * v for v in values], color=color, alpha=0.25, linewidth=1)
        smooth = rolling(values, ROLLING)
        ax.plot(steps, [None if v is None else 100 * v for v in smooth], color=color,
                label=f"{FLAG_NAMES[flag]} ({100 * summary['overall'][flag]:.1f}%)")
    ax.set_ylim(0, 102)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("% mẫu cùng nhãn (argmax)")
    ax.set_title(f"Tỉ lệ cùng nhãn theo step (đậm: trung bình trượt {ROLLING} step)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2)

    ax = axes[1]
    items = transitions.most_common(8)
    if items:
        names = [k for k, _ in items][::-1]
        counts = [v for _, v in items][::-1]
        ax.barh(names, counts, color=SERIES[0], height=0.6)
        for y, c in enumerate(counts):
            ax.text(c, y, f" {c}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Số Unlabeled")
    ax.set_title(f"Student khác Teacher: kiểu khác ({sum(transitions.values())} / {len(unlabeled)})")

    ax = axes[2]
    bin_names = ["< 0.45", "0.45–0.55", "≥ 0.55"]
    labels = [f"{name}\n(n={b['count']})" for name, b in zip(bin_names, by_confidence)]
    width = 0.26
    for k, (color, flag) in enumerate(zip(SERIES, UNLABELED_FLAGS)):
        xs = [i + (k - 1) * width for i in range(len(by_confidence))]
        ys = [0 if b[flag] is None else 100 * b[flag] for b in by_confidence]
        ax.bar(xs, ys, width=width - 0.02, color=color, label=FLAG_NAMES[flag])
    ax.set_xticks(range(len(by_confidence)), labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% mẫu cùng nhãn")
    ax.set_xlabel("P lớn nhất của Inspector")
    ax.set_title("Theo mức tự tin của Inspector")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)

    if by_augmentation:
        ax = axes[3]
        if "by_text_operation" in by_augmentation:
            groups = (list(by_augmentation["by_text_operation"].items())
                      + list(by_augmentation["by_layout"].items())
                      + list(by_augmentation["by_degradation"].items()))
        else:
            groups = list(by_augmentation["by_scenario"].items())
        groups += [("giữ A,B" if k == "False" else "đổi A↔B", v) for k, v in by_augmentation["by_swap"].items()]
        groups = [(name, g) for name, g in groups if g["count"]]
        labels = [f"{name}\n(n={g['count']})" for name, g in groups]
        width = 0.38
        for k, (color, flag) in enumerate(zip(SERIES, ("student_same_as_teacher", "student_same_as_inspector"))):
            xs = [i + (k - 0.5) * width for i in range(len(groups))]
            ax.bar(xs, [0 if g[flag] is None else 100 * g[flag] for _, g in groups], width=width - 0.02,
                   color=color, label=FLAG_NAMES[flag])
        ax.set_xticks(range(len(groups)), labels, fontsize=7, rotation=30 if wide else 0)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% mẫu cùng nhãn")
        ax.set_title("Theo kịch bản augmentation của Student")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)

    fig.suptitle(f"{args.version} (job {job}): nhãn Teacher, Student, Inspector từng mẫu trong lúc train",
                 x=0.01, ha="left", fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_dir / "per_sample_agreement.png", dpi=150, bbox_inches="tight")
    print(json.dumps({k: summary[k] for k in ("checks", "overall", "student_teacher_transitions", "by_augmentation")},
                     indent=1, ensure_ascii=False))
    print("written", out_dir)


if __name__ == "__main__":
    main()
