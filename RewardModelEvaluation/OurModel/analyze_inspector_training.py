#!/usr/bin/env python3
"""Plot and summarize one Overall Inspector training run.

Reads only log files (no images, no GPU):

- ``<run-dir>/logging.jsonl``: MS-Swift training log (loss, learning rate,
  gradient norm, and eval loss when a validation split exists),
- ``<run-dir>/progressive_loss.jsonl``: per-pair loss and P(A), P(B), P(Tie)
  computed before the update on that pair, i.e. on pairs not seen yet during
  the first pass,
- ``<run-dir>/train_order.jsonl``: fixed training order (pass of each position),
- ``<eval-dir>/<checkpoint>.json`` (optional): saved predictions of
  ``evaluate_inspector_checkpoints.py``. Metrics are recomputed from them for each
  evaluation set: every dataset without and with its tied pairs, and for
  EditReward-Bench also under the VLM-as-a-judge labels (``--vlm-pairs-csv``).
  EditReward-Data-Test is skipped even when present in older evaluation files:
  it is drawn from EditReward-Data, the training data of the Inspector.
  One chart per figure, one point per checkpoint. Sets without ties:
  checkpoint_accuracy, checkpoint_margin_auc, checkpoint_log_loss,
  checkpoint_mean_abs_margin (symmetric margin AUC {d} vs {-d}). Sets with ties
  (tie pairs positive, the rest negative): checkpoint_margin_auc_with_ties
  (scored by -|d|), checkpoint_tie_head_auc (scored by P(Tie)). Plus checkpoint_validation_loss when the run has a
  validation split.

Writes PNG figures and ``training_analysis.json`` to ``--output-dir`` (default
``<eval-dir>/training_analysis``, or ``<run-dir>/training_analysis`` without an
evaluation directory). Works for Inspector Baseline (one pass, no validation)
and Inspector pipeline (several passes, validation split).

Symbols: d = score_A - score_B; t = P(Tie); pairs seen = step x pairs per step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical order (blue, orange, aqua, yellow), text in ink tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
Z95 = 1.96
# Drawn from the Inspector training data (EditReward-Data), so never reported.
EXCLUDED_DATASETS = ("EditReward-Data-Test",)
DEFAULT_VLM_PAIRS_CSV = (
    Path(__file__).resolve().parents[1] / "EvalDatasets" / "EditReward" / "EditReward-Bench_VLM_judge" / "pairs.csv"
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "lines.linewidth": 2.0,
        "font.size": 10,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--window-pairs", type=int, default=2000, help="Pairs per window of the unseen-pair curves."
    )
    parser.add_argument(
        "--saturation-start", type=int, default=1000, help="First block size of the N -> 2N test."
    )
    parser.add_argument(
        "--vlm-pairs-csv",
        default=str(DEFAULT_VLM_PAIRS_CSV),
        help="VLM-as-a-judge majority labels of EditReward-Bench (pairs.csv); empty skips the VLM sets.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if positive.size == 0 or negative.size == 0:
        return None
    ranks = average_ranks(np.concatenate([positive, negative]))
    n_pos, n_neg = positive.size, negative.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mean_ci(values: np.ndarray) -> Dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "half_width_95": None, "n": 0}
    half = Z95 * values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else None
    return {"mean": float(values.mean()), "half_width_95": half, "n": int(values.size)}


def pairs_per_step(run_dir: Path) -> int:
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    return int(args["per_device_train_batch_size"]) * int(args["gradient_accumulation_steps"])


def add_pairs_axis(ax, per_step: int) -> None:
    """Second x-axis with the same scale: pairs seen = step x pairs per step."""
    top = ax.secondary_xaxis("top", functions=(lambda s: s * per_step, lambda p: p / per_step))
    top.set_xlabel("Training pairs seen", color=INK_SECONDARY)
    top.tick_params(colors=INK_SECONDARY)


# ---------------------------------------------------------------------------
# Training log


def training_log(run_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    train: Dict[str, List[float]] = {"step": [], "loss": [], "learning_rate": [], "grad_norm": []}
    evaluation: Dict[str, List[float]] = {"step": [], "eval_loss": []}
    for row in read_jsonl(run_dir / "logging.jsonl"):
        progress = row.get("global_step/max_steps")
        if not isinstance(progress, str):
            continue
        step = int(progress.split("/")[0])
        if "eval_loss" in row:
            evaluation["step"].append(step)
            evaluation["eval_loss"].append(row["eval_loss"])
        elif "loss" in row and "grad_norm" in row:
            train["step"].append(step)
            train["loss"].append(row["loss"])
            train["learning_rate"].append(row["learning_rate"])
            train["grad_norm"].append(row["grad_norm"])
    return {
        "train": {key: np.asarray(value, dtype=np.float64) for key, value in train.items()},
        "eval": {key: np.asarray(value, dtype=np.float64) for key, value in evaluation.items()},
    }


def plot_training_dynamics(log, per_step: int, output: Path) -> None:
    train = log["train"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    axes[0].plot(train["step"], train["loss"], color=SERIES[0], linewidth=1.0, alpha=0.35)
    if train["loss"].size >= 20:
        kernel = np.ones(20) / 20
        smooth = np.convolve(train["loss"], kernel, mode="valid")
        axes[0].plot(train["step"][19:], smooth, color=SERIES[0], label="moving average over 20 logs")
        axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Loss train")
    axes[0].set_title("Training loss by step (one point per log)", loc="left")
    add_pairs_axis(axes[0], per_step)
    axes[1].plot(train["step"], train["learning_rate"], color=SERIES[0])
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Learning rate schedule", loc="left")
    axes[2].plot(train["step"], train["grad_norm"], color=SERIES[0], linewidth=1.0)
    axes[2].axhline(1.0, color=INK_SECONDARY, linestyle="--", linewidth=1.0)
    axes[2].annotate("clipping threshold 1.0", (train["step"].max(), 1.0), color=INK_SECONDARY,
                     ha="right", va="bottom", fontsize=9)
    axes[2].set_ylabel("Gradient norm")
    axes[2].set_title("Gradient norm before clipping", loc="left")
    axes[2].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Progressive (unseen-pair) loss


def load_progressive(run_dir: Path) -> Dict[str, np.ndarray]:
    order = read_jsonl(run_dir / "train_order.jsonl")
    pass_of_position = np.asarray([row["pass"] for row in order], dtype=np.int64)
    rows = read_jsonl(run_dir / "progressive_loss.jsonl")
    position = np.asarray([row["position"] for row in rows], dtype=np.int64)
    probs = np.asarray([[row["prob_A"], row["prob_B"], row["prob_Tie"]] for row in rows])
    vote = np.asarray([row["vote"] for row in rows])
    cond_a = probs[:, 0] / np.clip(probs[:, 0] + probs[:, 1], 1e-12, None)
    cond_a = np.clip(cond_a, 1e-7, 1 - 1e-7)
    return {
        "position": position,
        "pass": pass_of_position[position],
        "step": np.asarray([row["step"] for row in rows], dtype=np.int64),
        "loss": np.asarray([row["loss"] for row in rows], dtype=np.float64),
        "probs": probs,
        "vote": vote,
        "abs_margin": np.abs(np.log(cond_a / (1 - cond_a))),
    }


def window_stats(data: Dict[str, np.ndarray], index: np.ndarray) -> Dict[str, Any]:
    vote, probs = data["vote"][index], data["probs"][index]
    loss, abs_margin = data["loss"][index], data["abs_margin"][index]
    is_tie = vote == "Tie"
    classes = np.asarray(["A", "B", "Tie"])[probs.argmax(axis=1)]
    pairwise = ~is_tie
    pairwise_correct = (probs[pairwise, 0] > probs[pairwise, 1]) == (vote[pairwise] == "A")
    return {
        "pairs": int(index.size),
        "loss": mean_ci(loss),
        "loss_pairwise": mean_ci(loss[pairwise]),
        "loss_tie": mean_ci(loss[is_tie]),
        "three_class_accuracy": float((classes == vote).mean()),
        "pairwise_accuracy": float(pairwise_correct.mean()) if pairwise.any() else None,
        "mean_tie_probability_tie": float(probs[is_tie, 2].mean()) if is_tie.any() else None,
        "mean_tie_probability_pairwise": float(probs[pairwise, 2].mean()) if pairwise.any() else None,
        "tie_auc_from_tie_probability": roc_auc(probs[is_tie, 2], probs[pairwise, 2]),
        "tie_auc_from_absolute_margin": roc_auc(-abs_margin[is_tie], -abs_margin[pairwise]),
        "mean_abs_margin_tie": float(abs_margin[is_tie].mean()) if is_tie.any() else None,
        "mean_abs_margin_pairwise": float(abs_margin[pairwise].mean()) if pairwise.any() else None,
    }


def unseen_windows(data, window: int) -> List[Dict[str, Any]]:
    first = np.flatnonzero(data["pass"] == 0)
    first = first[np.argsort(data["position"][first])]
    windows = []
    for start in range(0, first.size, window):
        index = first[start : start + window]
        if index.size < window // 2:
            break
        stats = window_stats(data, index)
        stats["pairs_seen_start"] = int(data["position"][index[0]])
        stats["pairs_seen_end"] = int(data["position"][index[-1]]) + 1
        windows.append(stats)
    return windows


def saturation_blocks(data, start: int) -> List[Dict[str, Any]]:
    """Blocks [N, 2N) of the first pass; drop to the next block with a 95% interval."""
    first = data["pass"] == 0
    total = int(first.sum())
    blocks = []
    low = 0
    high = start
    while low < total:
        mask = first & (data["position"] >= low) & (data["position"] < min(high, total))
        stats = mean_ci(data["loss"][mask])
        blocks.append({"pairs_from": low, "pairs_to": min(high, total), **stats})
        low, high = high, high * 2
    for previous, current in zip(blocks, blocks[1:]):
        if previous["half_width_95"] is None or current["half_width_95"] is None:
            continue
        drop = previous["mean"] - current["mean"]
        se = math.sqrt((previous["half_width_95"] / Z95) ** 2 + (current["half_width_95"] / Z95) ** 2)
        current["drop_from_previous"] = drop
        current["drop_interval_95"] = [drop - Z95 * se, drop + Z95 * se]
        current["significant_drop"] = bool(drop - Z95 * se > 0)
    return blocks


def plot_unseen(windows, output: Path) -> None:
    centers = np.asarray([(w["pairs_seen_start"] + w["pairs_seen_end"]) / 2 for w in windows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for key, label, color in (
        ("loss", "all pairs", SERIES[0]),
        ("loss_pairwise", "A/B pairs", SERIES[2]),
        ("loss_tie", "tie pairs", SERIES[1]),
    ):
        mean = np.asarray([w[key]["mean"] for w in windows], dtype=np.float64)
        half = np.asarray([w[key]["half_width_95"] or 0.0 for w in windows], dtype=np.float64)
        axes[0].plot(centers, mean, color=color, label=label)
        axes[0].fill_between(centers, mean - half, mean + half, color=color, alpha=0.15, linewidth=0)
    axes[0].set_ylabel("Loss (-log P(true label))")
    axes[0].set_title("Loss on not-yet-seen pairs (first pass), 95% interval", loc="left")
    axes[0].legend(loc="upper right")
    axes[1].plot(centers, [100 * w["three_class_accuracy"] for w in windows], color=SERIES[0],
                 label="3-class accuracy")
    axes[1].plot(centers, [100 * (w["pairwise_accuracy"] or math.nan) for w in windows], color=SERIES[2],
                 label="A/B accuracy (ties excluded)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Training pairs seen")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_tie(windows, output: Path) -> None:
    centers = np.asarray([(w["pairs_seen_start"] + w["pairs_seen_end"]) / 2 for w in windows])
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(centers, [w["mean_tie_probability_tie"] for w in windows], color=SERIES[1], label="tie pairs")
    axes[0].plot(centers, [w["mean_tie_probability_pairwise"] for w in windows], color=SERIES[2],
                 label="A/B pairs")
    axes[0].set_ylabel("mean t")
    axes[0].set_title("Mean P(Tie) by label", loc="left")
    axes[0].legend(loc="upper right")
    axes[1].plot(centers, [100 * (w["tie_auc_from_tie_probability"] or math.nan) for w in windows],
                 color=SERIES[0], label="from t")
    axes[1].plot(centers, [100 * (w["tie_auc_from_absolute_margin"] or math.nan) for w in windows],
                 color=SERIES[3], label="from small |d| only")
    axes[1].axhline(50, color=INK_SECONDARY, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Tie AUC (%)")
    axes[1].set_title("Separating tie pairs from A/B pairs (50 = random)", loc="left")
    axes[1].legend(loc="upper left")
    axes[2].plot(centers, [w["mean_abs_margin_tie"] for w in windows], color=SERIES[1], label="tie pairs")
    axes[2].plot(centers, [w["mean_abs_margin_pairwise"] for w in windows], color=SERIES[2], label="A/B pairs")
    axes[2].set_ylabel("mean |d|")
    axes[2].set_title("Score gap |d| = |score A - score B|", loc="left")
    axes[2].set_xlabel("Training pairs seen")
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint evaluation (recomputed from the saved predictions of each checkpoint)

# pairs.csv of the VLM-as-a-judge relabeling: 0 = candidate_1 (A), 1 = candidate_2 (B), 2 = tie.
VLM_LABEL = {"0": 1, "1": -1, "2": 0}
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42


def load_vlm_labels(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["pair_id"].split("/")[0]: VLM_LABEL[row["majority_overall"]]
            for row in csv.DictReader(handle)
            if row["subset"] == "2pair"
        }


def load_checkpoint_predictions(eval_dir: Path, per_step: int) -> List[Dict[str, Any]]:
    checkpoints = []
    for path in eval_dir.glob("*.json"):
        if path.name == "summary.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "predictions" not in result:
            continue
        step = int(result.get("global_step") or 0)
        checkpoints.append({
            "checkpoint": result["checkpoint"],
            "step": step,
            "pairs_seen": step * per_step,
            "is_base": result["checkpoint"] == "base_zero_shot",
            "predictions": {name: rows for name, rows in result["predictions"].items()
                            if name not in EXCLUDED_DATASETS},
        })
    return sorted(checkpoints, key=lambda item: (not item["is_base"], item["pairs_seen"]))


def evaluation_sets(predictions: Dict[str, List[Dict[str, Any]]],
                    vlm_labels: Dict[str, int] | None) -> List[Dict[str, Any]]:
    """Evaluation sets = (dataset, label source, with or without the originally tied pairs).

    The set without ties keeps the pairs that are not tied under the original labels, so the
    original and VLM versions of it contain the same pairs (115 on EditReward-Bench).
    """
    sets = []
    for dataset, rows in predictions.items():
        original = {row["sample_id"]: int(row["label"]) for row in rows}
        sources = [("original", original)]
        if vlm_labels is not None and dataset == "EditReward-Bench":
            missing = set(original) - set(vlm_labels)
            if missing:
                raise ValueError(f"{len(missing)} {dataset} pairs have no VLM label")
            sources.append(("vlm", {key: vlm_labels[key] for key in original}))
        not_tied = [key for key, label in original.items() if label != 0]
        for source, labels in sources:
            tag = "" if source == "original" else "VLM-Reannotate, "
            sets.append({"name": f"{dataset} ({tag}{len(not_tied)})", "dataset": dataset, "source": source,
                         "with_ties": False, "ids": not_tied, "labels": labels})
            if len(not_tied) < len(original):
                sets.append({"name": f"{dataset} ({tag}{len(original)}, w/ Tie)", "dataset": dataset,
                             "source": source, "with_ties": True, "ids": list(original), "labels": labels})
    return sets


def bootstrap_counts(size: int) -> np.ndarray:
    """How often each pair appears in each bootstrap replicate (same draws as recompute_margin_auc.py)."""
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, size, (BOOTSTRAP_REPLICATES, size))
    counts = np.zeros((BOOTSTRAP_REPLICATES, size), dtype=np.float64)
    np.add.at(counts, (np.arange(BOOTSTRAP_REPLICATES)[:, None], draws), 1.0)
    return counts


def weighted_auc(win: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                 counts: np.ndarray) -> np.ndarray:
    """AUC for each row of pair weights; win[i, j] = 1 if pair i outranks pair j, 0.5 on ties."""
    left, right = counts * positive, counts * negative
    numerator = np.einsum("bi,ij,bj->b", left, win, right)
    denominator = left.sum(axis=1) * right.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def auc_with_interval(win: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                      counts: np.ndarray) -> Dict[str, Any]:
    point = weighted_auc(win, positive, negative, np.ones((1, positive.size)))[0]
    if not np.isfinite(point):
        return {"percent": None, "interval_95": None}
    replicates = weighted_auc(win, positive, negative, counts)
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return {"percent": 100 * float(point), "interval_95": [100 * float(low), 100 * float(high)]}


def win_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None] - right[None, :]
    return (diff > 0) + 0.5 * (diff == 0)


def set_metrics(spec: Dict[str, Any], rows: Dict[str, Dict[str, Any]], counts: np.ndarray,
                is_base: bool) -> Dict[str, Any]:
    labels = np.asarray([spec["labels"][key] for key in spec["ids"]], dtype=np.int64)
    margins = np.asarray([rows[key]["margin_A_minus_B"] for key in spec["ids"]], dtype=np.float64)
    # -|d| for the with-tie margin AUC; Student rows give it separately (mean raw score difference).
    tie_margins = np.asarray([rows[key].get("tie_margin", rows[key]["margin_A_minus_B"])
                              for key in spec["ids"]], dtype=np.float64)
    is_tie = labels == 0
    decided = ~is_tie
    signed = labels[decided] * margins[decided]
    everywhere = np.ones(decided.sum())
    # A/B metrics on the pairs that are not tied under this set's labels; on a set with ties the
    # bootstrap still resamples all of its pairs.
    pairwise: Dict[str, Any] = {
        "pairs": int(decided.sum()),
        "accuracy_percent": 100 * float((signed > 0).mean()),
        # Symmetric within-pair margin AUC: {d_i} against {-d_i}, d = label x margin.
        "margin_auc": auc_with_interval(win_matrix(signed, -signed), everywhere, everywhere,
                                        counts[:, decided]),
        "log_loss": float(np.logaddexp(0.0, -signed).mean()),
        "mean_abs_margin": float(np.abs(signed).mean()),
    }
    if not spec["with_ties"]:
        return {"pairs": int(labels.size), "tie_pairs": 0, **{k: v for k, v in pairwise.items() if k != "pairs"}}
    # Sets with ties: tie pairs are positive, all other pairs negative (AUC-Margin.md,
    # tie_head_ablation.md 12.3). Margin AUC scores each pair by -|d|; Tie Head AUC by P(Tie).
    metrics: Dict[str, Any] = {
        "pairs": int(labels.size),
        "tie_pairs": int(is_tie.sum()),
        "margin_auc": auc_with_interval(
            win_matrix(-np.abs(tie_margins), -np.abs(tie_margins)), is_tie.astype(float),
            decided.astype(float), counts),
        "non_tie_pairs": pairwise,
    }
    if not is_base:
        # The Tie Head of the base model is untrained, so Tie Head metrics start at the first checkpoint.
        probs = np.asarray([[rows[key]["probability_A"], rows[key]["probability_B"],
                             rows[key]["probability_Tie"]] for key in spec["ids"]], dtype=np.float64)
        target = np.select([labels == 1, labels == -1], [0, 1], default=2)
        tie = probs[:, 2]
        metrics.update({
            "tie_head_auc": auc_with_interval(
                win_matrix(tie, tie), is_tie.astype(float), decided.astype(float), counts),
            "three_class_accuracy_percent": 100 * float((probs.argmax(axis=1) == target).mean()),
            "three_class_log_loss": float(-np.log(np.clip(probs[np.arange(labels.size), target],
                                                          1e-12, None)).mean()),
            "mean_tie_probability_tie": float(tie[is_tie].mean()) if is_tie.any() else None,
            "mean_tie_probability_not_tie": float(tie[decided].mean()),
        })
    return metrics


def checkpoint_metrics(checkpoints, vlm_labels) -> Dict[str, Any]:
    sets = evaluation_sets(checkpoints[-1]["predictions"], vlm_labels)
    counts = {spec["name"]: bootstrap_counts(len(spec["ids"])) for spec in sets}
    table = []
    for item in checkpoints:
        row = {key: item[key] for key in ("checkpoint", "step", "pairs_seen", "is_base")}
        for spec in sets:
            rows = {pred["sample_id"]: pred for pred in item["predictions"][spec["dataset"]]}
            row[spec["name"]] = set_metrics(spec, rows, counts[spec["name"]], item["is_base"])
        table.append(row)
    return {"sets": [{key: spec[key] for key in ("name", "dataset", "source", "with_ties")} | {
        "pairs": len(spec["ids"]),
        "tie_pairs": sum(spec["labels"][key] == 0 for key in spec["ids"]),
    } for spec in sets], "rows": table}


def checkpoint_label(pairs_seen: int, is_base: bool = False) -> str:
    return "0 (base)" if is_base else f"{pairs_seen:,}"


def plot_checkpoint_figure(labels: Sequence[str], lines: Sequence[Dict[str, Any]], title: str, ylabel: str,
                           output: Path, reference: float | None = None, legend_loc: str = "best") -> None:
    """One chart per figure; every point is one checkpoint, evenly spaced and labeled by pairs seen."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels) + 2.5), 4.8))
    for line in lines:
        ax.plot(x, line["y"], color=line["color"], marker="o", markersize=5, label=line["label"])
    if reference is not None:
        ax.axhline(reference, color=INK_SECONDARY, linestyle="--", linewidth=1.0)
        # Free band below the reference line for the legend.
        ax.set_ylim(bottom=reference - 12)
    ax.legend(loc=legend_loc, fontsize=8)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Checkpoint (training pairs seen)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def metric_lines(table, specs, key: str, label=None) -> List[Dict[str, Any]]:
    """Point estimates only; bootstrap intervals stay in training_analysis.json."""
    lines = []
    for color, spec in zip(SERIES, specs):
        values = [row[spec["name"]].get(key) for row in table]
        values = [value.get("percent") if isinstance(value, dict) else value for value in values]
        lines.append({"y": [math.nan if value is None else value for value in values], "color": color,
                      "label": label(spec) if label else spec["name"]})
    return lines


def plot_checkpoint_figures(table, sets, log, per_step: int, output_dir: Path) -> List[str]:
    plain = [spec for spec in sets if not spec["with_ties"]]
    tied = [spec for spec in sets if spec["with_ties"]]
    labels = [checkpoint_label(row["pairs_seen"], row["is_base"]) for row in table]
    figures = []

    def draw(name: str, lines, title: str, ylabel: str, **kwargs) -> None:
        plot_checkpoint_figure(labels, lines, title, ylabel, output_dir / name, **kwargs)
        figures.append(name)

    # Sets without ties: accuracy, margin AUC, log loss and |d|.
    draw("checkpoint_accuracy.png", metric_lines(table, plain, "accuracy_percent"),
         "A/B accuracy (%) on non-tie pairs", "Accuracy (%)", legend_loc="lower right")
    draw("checkpoint_margin_auc.png", metric_lines(table, plain, "margin_auc"),
         "Margin AUC (%) on non-tie pairs", "Margin AUC (%)", legend_loc="lower right")
    draw("checkpoint_log_loss.png", metric_lines(table, plain, "log_loss"),
         "A/B log loss on non-tie pairs", "Log loss", legend_loc="upper left")
    # |d| does not depend on the labels, so only the original-label set is drawn.
    draw("checkpoint_mean_abs_margin.png",
         metric_lines(table, [spec for spec in plain if spec["source"] == "original"], "mean_abs_margin"),
         "Mean |d| on non-tie pairs (higher = more confident)", "|d| = |score A - score B|",
         legend_loc="upper left")

    # Sets with ties: tie pairs positive, the rest negative; margin AUC from -|d|, Tie Head AUC from P(Tie).
    if tied:
        tie_label = lambda spec: (f"{spec['name']}: {spec['tie_pairs']} tie pairs"
                                  + (" (too few, reference only)" if spec["tie_pairs"] < 10 else ""))
        draw("checkpoint_margin_auc_with_ties.png", metric_lines(table, tied, "margin_auc", label=tie_label),
             "Margin AUC (%) on sets with ties: tie pairs vs the rest, scored by -|d| (50 = random)",
             "AUC (%)", reference=50.0, legend_loc="lower right")
        draw("checkpoint_tie_head_auc.png", metric_lines(table, tied, "tie_head_auc", label=tie_label),
             "Tie Head AUC (%) on sets with ties: tie pairs vs the rest, scored by P(Tie) (50 = random)",
             "AUC (%)", reference=50.0, legend_loc="lower right")

    # Validation loss of the run, at each evaluation step.
    evaluation = log["eval"]
    if evaluation["step"].size:
        eval_labels = [checkpoint_label(int(step) * per_step) for step in evaluation["step"]]
        plot_checkpoint_figure(eval_labels, [{"y": evaluation["eval_loss"], "color": INK,
                                              "label": "3-class validation loss"}],
                               "3-class validation loss on the run's validation split", "Loss",
                               output_dir / "checkpoint_validation_loss.png", legend_loc="upper right")
        figures.append("checkpoint_validation_loss.png")
    return figures


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve() if args.eval_dir else None
    output_dir = Path(
        args.output_dir or (eval_dir or run_dir) / "training_analysis"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_step = pairs_per_step(run_dir)

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir) if eval_dir else None,
        "pairs_per_step": per_step,
        "notes": [
            "Unseen-pair metrics use only the first pass of train_order.jsonl.",
            "Loss drops over the run also reflect the cosine learning-rate schedule.",
        ],
        "figures": [],
    }

    log = training_log(run_dir)
    plot_training_dynamics(log, per_step, output_dir / "training_dynamics.png")
    report["figures"].append("training_dynamics.png")
    if log["eval"]["step"].size:
        best = int(np.argmin(log["eval"]["eval_loss"]))
        report["best_eval_loss"] = {
            "step": int(log["eval"]["step"][best]),
            "eval_loss": float(log["eval"]["eval_loss"][best]),
        }

    if (run_dir / "progressive_loss.jsonl").is_file():
        data = load_progressive(run_dir)
        windows = unseen_windows(data, args.window_pairs)
        report["unseen_windows"] = windows
        report["saturation_blocks"] = saturation_blocks(data, args.saturation_start)
        report["later_passes"] = [
            {"pass": int(p), **mean_ci(data["loss"][data["pass"] == p])}
            for p in np.unique(data["pass"]) if p > 0
        ]
        if windows:
            plot_unseen(windows, output_dir / "unseen_pair_loss.png")
            plot_tie(windows, output_dir / "tie_diagnostics.png")
            report["figures"] += ["unseen_pair_loss.png", "tie_diagnostics.png"]

    checkpoints = load_checkpoint_predictions(eval_dir, per_step) if eval_dir is not None else []
    if checkpoints:
        vlm_labels = load_vlm_labels(Path(args.vlm_pairs_csv)) if args.vlm_pairs_csv else None
        evaluation = checkpoint_metrics(checkpoints, vlm_labels)
        report["checkpoint_sets"] = evaluation["sets"]
        report["checkpoints"] = evaluation["rows"]
        report["figures"] += plot_checkpoint_figures(evaluation["rows"], evaluation["sets"], log, per_step,
                                                     output_dir)

    (output_dir / "training_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(report['figures'])} figures and training_analysis.json to {output_dir}")
    for block in report.get("saturation_blocks", []):
        drop = block.get("drop_from_previous")
        print(
            f"pairs [{block['pairs_from']:>7}, {block['pairs_to']:>7}): loss {block['mean']:.4f}"
            + (f", drop {drop:+.4f} {['', 'significant'][block['significant_drop']]}" if drop is not None else "")
        )


if __name__ == "__main__":
    main()
