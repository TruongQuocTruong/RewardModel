#!/usr/bin/env python3
"""Plots for diagnose_gradient_components.py results (one or more Student weights).

Usage:
  python plot_gradient_components.py --output-dir DIR \
      --inputs "Student-Stage1=path/a.json" "checkpoint 126=path/b.json"

Figures (in DIR):
  gradient_component_norms.png        per-step gradient norm of every loss component, per group
  gradient_component_cosines.png      cosine between components, mean over the replayed steps
  gradient_signal_vs_noise.png        ||mean gradient|| / mean ||gradient|| and consecutive-step cosine
  gradient_vit_norm_vs_image_patches.png   ViT gradient norm of each microbatch vs its image patches
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMPONENTS = ("human", "gemma_tie", "gemma_winner", "mask", "unlabeled_inspector", "unlabeled_teacher")
LABEL = {
    "human": "L_H (HumanGT CE)",
    "gemma_tie": "L_G tie part",
    "gemma_winner": "L_G winner part",
    "mask": "L_M (relevance)",
    "unlabeled_inspector": "L_U Inspector CE",
    "unlabeled_teacher": "L_U Teacher KL",
    "total": "sum (total gradient)",
}
SHORT = {"human": "L_H", "gemma_tie": "L_G tie", "gemma_winner": "L_G winner", "mask": "L_M",
         "unlabeled_inspector": "L_U Insp.", "unlabeled_teacher": "L_U Teacher"}
COLOR = {"human": "#2a78d6", "gemma_tie": "#eb6834", "gemma_winner": "#1baf7a", "mask": "#eda100",
         "unlabeled_inspector": "#e87ba4", "unlabeled_teacher": "#008300", "total": "#0b0b0b"}
GROUPS = ("all", "vit", "llm", "heads")
GROUP_LABEL = {"all": "all parameters", "vit": "ViT", "llm": "LLM", "heads": "heads", "aligner": "aligner"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED)


def load(inputs):
    runs = []
    for item in inputs:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--inputs expects label=path, got {item!r}")
        data = json.loads(Path(path).read_text())
        if data.get("failed_checks"):
            raise ValueError(f"{path} has failed checks: {data['failed_checks']}")
        runs.append((label, data))
    return runs


def plot_norms(runs, out: Path) -> None:
    fig, axes = plt.subplots(len(GROUPS), len(runs), figsize=(6.2 * len(runs) + 1, 3.0 * len(GROUPS)),
                             squeeze=False, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for column, (label, data) in enumerate(runs):
        steps = [s["step"] for s in data["steps"]]
        for row, group in enumerate(GROUPS):
            axis = axes[row, column]
            style(axis)
            for c in COMPONENTS:
                axis.plot(steps, [s["groups"][group]["component_norm"][c] for s in data["steps"]],
                          color=COLOR[c], lw=1.8, marker="o", ms=4, label=LABEL[c])
            axis.plot(steps, [s["groups"][group]["total_norm"] for s in data["steps"]],
                      color=COLOR["total"], lw=1.4, ls="--", marker="s", ms=4, label=LABEL["total"])
            axis.set_yscale("log")
            axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ratio = np.mean([s["groups"][group]["total_squared_over_sum_of_squares"] for s in data["steps"]])
            axis.set_title(f"{label} — {GROUP_LABEL[group]}\n"
                           f"mean ||sum||² / Σ||component||² = {ratio:.2f} (1: orthogonal, <1: cancel, >1: aligned)",
                           loc="left", fontsize=9.5, color=INK)
            if row == len(GROUPS) - 1:
                axis.set_xlabel("replayed step (training data of that step)", color=MUTED)
            if column == 0:
                axis.set_ylabel("gradient norm (weighted as in training)", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out / "gradient_component_norms.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_cosines(runs, out: Path) -> None:
    groups = ("all", "vit", "llm", "heads")
    fig, axes = plt.subplots(len(runs), len(groups), figsize=(5.4 * len(groups), 4.8 * len(runs)), squeeze=False,
                             gridspec_kw={"wspace": 0.45, "hspace": 0.45})
    fig.patch.set_facecolor(SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    for row, (label, data) in enumerate(runs):
        for column, group in enumerate(groups):
            axis = axes[row, column]
            n = len(COMPONENTS)
            matrix = np.full((n, n), np.nan)
            spread = np.full((n, n), np.nan)
            for i, a in enumerate(COMPONENTS):
                matrix[i, i] = 1.0
                for j, b in enumerate(COMPONENTS[i + 1:], start=i + 1):
                    values = [s["groups"][group]["cosine"][f"{a}|{b}"] for s in data["steps"]]
                    values = [v for v in values if v is not None]
                    if values:
                        matrix[i, j] = matrix[j, i] = float(np.mean(values))
                        spread[i, j] = spread[j, i] = float(np.std(values))
            image = axis.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
            for i in range(n):
                for j in range(n):
                    if i != j and not math.isnan(matrix[i, j]):
                        axis.text(j, i, f"{matrix[i, j]:+.2f}\n±{spread[i, j]:.2f}", ha="center", va="center",
                                  fontsize=7, color=INK)
            axis.set_xticks(range(n), [SHORT[c] for c in COMPONENTS], rotation=45, ha="right", fontsize=8)
            axis.set_yticks(range(n), [SHORT[c] for c in COMPONENTS], fontsize=8)
            axis.set_title(f"{label} — {GROUP_LABEL[group]}", loc="left", fontsize=10, color=INK)
    fig.colorbar(image, ax=axes, shrink=0.6, label="cosine (mean over steps; ± = std over steps)")
    fig.suptitle("Cosine between loss-component gradients (negative = the two losses pull in opposite directions; "
                 "bf16 precision about ±0.03)", fontsize=11, color=INK)
    fig.savefig(out / "gradient_component_cosines.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_signal_vs_noise(runs, out: Path) -> None:
    keys = COMPONENTS + ("total",)
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(4.6 * len(GROUPS), 8.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    width = 0.8 / len(runs)
    hatches = ["", "//", "..", "xx"]
    for column, group in enumerate(GROUPS):
        axis = axes[0, column]
        style(axis)
        for r, (label, data) in enumerate(runs):
            values = [data["across_steps"][k][group]["ratio"] or 0.0 for k in keys]
            axis.bar(np.arange(len(keys)) + (r - (len(runs) - 1) / 2) * width, values, width=width * 0.92,
                     color=[COLOR[k] for k in keys], hatch=hatches[r % len(hatches)], edgecolor=SURFACE,
                     label=label)
        noise = runs[0][1]["pure_noise_ratio_reference"]
        axis.axhline(noise, color=MUTED, ls=":", lw=1.2)
        axis.text(len(keys) - 0.5, noise, f"pure noise ≈ {noise:.2f}", ha="right", va="bottom", fontsize=8, color=MUTED)
        axis.axhline(1.0, color=MUTED, lw=0.8)
        axis.set_ylim(0, 1.05)
        axis.set_xticks(range(len(keys)), [SHORT.get(k, "sum") for k in keys], rotation=45, ha="right", fontsize=8)
        axis.set_title(f"{GROUP_LABEL[group]}: ||mean over steps|| / mean ||per step||\n"
                       "(1 = same direction every step)", loc="left", fontsize=9.5, color=INK)
        axis = axes[1, column]
        style(axis)
        for r, (label, data) in enumerate(runs):
            points = [(s["step"], s["groups"][group].get("cosine_with_previous_step_total"))
                      for s in data["steps"]]
            points = [(x, y) for x, y in points if y is not None]
            if points:
                axis.plot(*zip(*points), marker="o", ms=4, lw=1.6, ls=["-", "--", ":", "-."][r % 4],
                          color=INK if r == 0 else MUTED, label=label)
        axis.axhline(0, color=MUTED, lw=0.8)
        axis.set_ylim(-1, 1)
        axis.set_xlabel("replayed step", color=MUTED)
        axis.set_title(f"{GROUP_LABEL[group]}: cosine of the total gradient\nwith the previous step",
                       loc="left", fontsize=9.5, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, title="hatch = Student weights", title_fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gradient_signal_vs_noise.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_vit_vs_patches(runs, out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(6.4 * len(runs), 4.8), squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for column, (label, data) in enumerate(runs):
        axis = axes[0, column]
        style(axis)
        notes = []
        for c in COMPONENTS:
            xs, ys = [], []
            for s in data["steps"]:
                for mb in s["microbatches"]:
                    if c in mb["norms"]:
                        xs.append(mb["image_patches_total"])
                        ys.append(mb["norms"][c]["vit"])
            if xs:
                axis.scatter(xs, ys, s=22, color=COLOR[c], edgecolor=SURFACE, linewidth=0.6, label=LABEL[c])
                if len(xs) > 2 and np.std(xs) > 0 and min(ys) > 0:
                    notes.append(f"{SHORT[c]} r={np.corrcoef(np.log(xs), np.log(ys))[0, 1]:+.2f}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("image patches in the microbatch (3 images: A, B, source)", color=MUTED)
        half = (len(notes) + 1) // 2
        axis.set_title(f"{label}: ViT gradient norm per microbatch\ncorrelation of logs: "
                       + ", ".join(notes[:half]) + "\n" + ", ".join(notes[half:]),
                       loc="left", fontsize=9, color=INK)
    axes[0, 0].set_ylabel("ViT gradient norm of the component", color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gradient_vit_norm_vs_image_patches.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", nargs="+", required=True, help="label=path/to/gradient_components.json")
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args()
    runs = load(options.inputs)
    out = Path(options.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_norms(runs, out)
    plot_cosines(runs, out)
    plot_signal_vs_noise(runs, out)
    plot_vit_vs_patches(runs, out)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
