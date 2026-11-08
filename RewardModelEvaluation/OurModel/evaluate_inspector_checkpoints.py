#!/usr/bin/env python3
"""Evaluate every saved three-way Overall Inspector Model checkpoint of one run.

The base model and LoRA structure are loaded once. Each ``checkpoint-*``
adapter (including the three-way Inspector head) is then loaded in turn and
scored on EditReward-Bench, all pairs including ties. EditReward-Data-Test is
not used: it is drawn from EditReward-Data, the training data of the Inspector.

Pairwise metrics (accuracy, margin AUC, Brier, log loss) use the non-tie pairs.
Three-way metrics (three-class accuracy, three-way log loss, Tie AUC from P(Tie)
and from -|d|, with d = score_A - score_B) use every pair.

Images are resized as in MS-Swift training: qwen_vl_utils with the model patch
size, then the processor defaults from ``preprocessor_config.json`` unless
``--min-pixels``/``--max-pixels`` are given.

``--include-base`` also scores the untrained Qwen3-VL-Reranker with the head
initialized from ``W_yes - W_no`` (zero-shot reference).

Every checkpoint writes its own JSON (skipped when it already exists, so an
interrupted job can resume), and ``summary.json`` collects all of them.
No checkpoint is selected here: EditReward-Bench is never used for selection.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from peft import LoraConfig, PeftConfig, get_peft_model, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_EVAL_DIR = Path(__file__).resolve().parent  # RewardModelEvaluation/OurModel
_PROJECT_ROOT = _EVAL_DIR.parents[1] / "TrainingRewardModel" / "Qwen3-VL-Embedding"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_reward_model import (  # noqa: E402
    BenchmarkRecord,
    load_benchmark,
    summarize,
    tie_ranking_metrics,
)
from infer_overall_inspector import infer_batch  # noqa: E402
from src.models.custom_qwen3_vl_reranker import (  # noqa: E402
    install_overall_inspector_head,
    tie_head_mode_from_state_keys,
)
from src.prompt_config import (  # noqa: E402
    OVERALL_INSPECTOR_EVALUATION_INSTRUCTION,
    OVERALL_INSPECTOR_SYSTEM_MESSAGE,
)
from train_overall_inspector import CHECKPOINT_FILENAME  # noqa: E402

BASE_NAME = "base_zero_shot"
BENCH_DATASET = "EditReward-Bench"
TOKEN_SAMPLE_RECORDS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Checkpoint directory names; default is every checkpoint-* in --run-dir.",
    )
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--editreward-bench", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_datasets(args: argparse.Namespace) -> Dict[str, List[BenchmarkRecord]]:
    # Ties are kept: pairwise metrics skip them, three-way metrics use them.
    return {BENCH_DATASET: load_benchmark(Path(args.editreward_bench))}


def discover_checkpoints(run_dir: Path, names: Sequence[str] | None) -> List[Path]:
    if names:
        paths = [run_dir / name for name in names]
    else:
        paths = sorted(
            (p for p in run_dir.glob("checkpoint-*") if p.is_dir() and not p.is_symlink()),
            key=lambda p: int(re.sub(r"\D", "", p.name) or 0),
        )
    for path in paths:
        weights = path / "adapter_model.safetensors"
        if not weights.is_file() or weights.stat().st_size == 0:
            raise FileNotFoundError(f"Adapter weights are missing: {weights}")
    return paths


def resolve_prompt(run_dir: Path) -> Dict[str, Any]:
    """Use the prompt contract stored by the run; fall back to prompt_config."""
    consolidated = run_dir / CHECKPOINT_FILENAME
    if consolidated.is_file():
        checkpoint = torch.load(consolidated, map_location="cpu", mmap=True, weights_only=False)
        prompt = checkpoint.get("prompt", {})
        if isinstance(prompt.get("system_message"), str):
            return {**prompt, "source": str(consolidated)}
    return {
        "system_message": OVERALL_INSPECTOR_SYSTEM_MESSAGE,
        "evaluation_instruction": OVERALL_INSPECTOR_EVALUATION_INSTRUCTION,
        "source": "src.prompt_config (no consolidated checkpoint in run dir)",
    }


def _build_peft_config(config_dict: Dict[str, Any]) -> PeftConfig:
    supported = set(inspect.signature(LoraConfig).parameters)
    filtered = {key: value for key, value in config_dict.items() if key in supported}
    filtered["inference_mode"] = True
    return PeftConfig.from_peft_type(**filtered)


def load_model(
    args: argparse.Namespace, adapter_config: Dict[str, Any], tie_head_mode: str, device
):
    pixel_kwargs = {
        key: value
        for key, value in (("min_pixels", args.min_pixels), ("max_pixels", args.max_pixels))
        if value is not None
    }
    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        padding_side="left",
        **pixel_kwargs,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    install_overall_inspector_head(base_model, processor.tokenizer, tie_head_mode=tie_head_mode)
    model = get_peft_model(base_model, _build_peft_config(adapter_config))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, processor


def load_adapter(model, checkpoint_dir: Path) -> None:
    state = load_safetensors(str(checkpoint_dir / "adapter_model.safetensors"), device="cpu")
    # Without trained head weights, the head keeps its W_yes - W_no init,
    # which is MS-Swift's generative_reranker score.
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys[:10]}")
    # Base weights and the frozen original head are expected to be missing.
    missing = [
        key for key in result.missing_keys if "lora_" in key or "modules_to_save" in key
    ]
    if missing:
        raise RuntimeError(f"Missing adapter keys: {missing[:10]}")


def image_token_stats(processor, prompt, records: Sequence[BenchmarkRecord]) -> Dict[str, Any]:
    """Visual tokens per image actually fed to the model (first records only)."""
    messages = []
    for record in records[:TOKEN_SAMPLE_RECORDS]:
        for path in (record.source_image, record.image_a):
            messages.append([{"role": "user", "content": [{"type": "image", "image": f"file://{path}"}]}])
    images, _ = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size
    )
    grid = processor.image_processor(images=images, return_tensors="pt")["image_grid_thw"]
    merge = int(getattr(processor.image_processor, "merge_size", 2))
    tokens = (grid.prod(dim=-1) // (merge * merge)).float()
    return {
        "sampled_images": int(tokens.numel()),
        "mean_tokens_per_image": float(tokens.mean()),
        "min_tokens_per_image": int(tokens.min()),
        "max_tokens_per_image": int(tokens.max()),
    }


def _probability_metrics(records: Sequence[BenchmarkRecord], margins: Sequence[float]) -> Dict[str, Any]:
    labels = np.asarray([record.label for record in records])
    margin = np.asarray(margins, dtype=np.float64)
    keep = labels != 0
    target = (labels[keep] == 1).astype(np.float64)
    prob_a = 1.0 / (1.0 + np.exp(-margin[keep]))
    eps = 1e-7
    clipped = np.clip(prob_a, eps, 1.0 - eps)
    log_loss = -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))
    return {
        "probability_definition": "w = sigmoid(score_A - score_B), P(A wins | not tie)",
        "log_loss": float(log_loss.mean()),
        "brier_score": float(((prob_a - target) ** 2).mean()),
        "mean_abs_margin": float(np.abs(margin[keep]).mean()),
    }


def _margin_auc(records: Sequence[BenchmarkRecord], margins: Sequence[float]) -> Dict[str, Any]:
    """Within-pair margin AUC: (label*margin, positive) vs (-label*margin, negative)."""
    labels = np.asarray([record.label for record in records], dtype=np.float64)
    margin = np.asarray(margins, dtype=np.float64)
    keep = labels != 0
    winner_minus_loser = labels[keep] * margin[keep]
    comparisons = winner_minus_loser[:, None] + winner_minus_loser[None, :]
    auc = float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)
    return {"pairs": int(keep.sum()), "value": auc, "percent": round(100.0 * auc, 2)}


def _three_way_metrics(
    records: Sequence[BenchmarkRecord],
    probabilities: Sequence[Sequence[float]],
    margins: Sequence[float],
) -> Dict[str, Any]:
    """Metrics over every pair, ties included; columns are P(A), P(B), P(Tie)."""
    labels = np.asarray([record.label for record in records])
    probs = np.asarray(probabilities, dtype=np.float64)
    true_column = np.select([labels == 1, labels == -1], [0, 1], default=2)
    eps = 1e-7
    true_probability = np.clip(probs[np.arange(len(labels)), true_column], eps, 1.0)
    one_hot = np.eye(3)[true_column]
    report: Dict[str, Any] = {
        "pairs": int(labels.size),
        "class_counts": {
            name: int((true_column == column).sum())
            for column, name in enumerate(("A", "B", "Tie"))
        },
        "three_class_accuracy_percent": round(
            100.0 * float((probs.argmax(axis=1) == true_column).mean()), 2
        ),
        "three_way_log_loss": float(-np.log(true_probability).mean()),
        "three_way_brier": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        "mean_tie_probability": {
            name: float(probs[true_column == column, 2].mean())
            for column, name in enumerate(("A", "B", "Tie"))
            if (true_column == column).any()
        },
    }
    report["tie_from_tie_head"] = {
        "confidence_definition": "P(Tie)",
        **tie_ranking_metrics(labels, probs[:, 2]),
    }
    report["tie_from_absolute_margin"] = {
        "confidence_definition": "-|score_A - score_B|",
        **tie_ranking_metrics(labels, -np.abs(np.asarray(margins, dtype=np.float64))),
    }
    return report


def _pooled_auc(records: Sequence[BenchmarkRecord], scores: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """ROC AUC of winner scores (positive) against loser scores (negative), pooled over pairs."""
    labels = np.asarray([record.label for record in records])
    pair_scores = np.asarray(scores, dtype=np.float64)
    keep = labels != 0
    winner_is_a = labels[keep] == 1
    kept = pair_scores[keep]
    positive = np.where(winner_is_a, kept[:, 0], kept[:, 1])
    negative = np.where(winner_is_a, kept[:, 1], kept[:, 0])
    # Mann-Whitney: fraction of (winner, loser) combinations ranked correctly, ties count 0.5.
    greater = (positive[:, None] > negative[None, :]).sum()
    equal = (positive[:, None] == negative[None, :]).sum()
    auc = (greater + 0.5 * equal) / (positive.size * negative.size)
    return {
        "definition": "ROC AUC, winner image scores positive vs loser image scores negative, pooled",
        "num_positive": int(positive.size),
        "num_negative": int(negative.size),
        "value": float(auc),
        "percent": round(100.0 * float(auc), 2),
    }


@torch.inference_mode()
def score_dataset(model, processor, prompt, records, batch_size, device):
    contract = {"prompt": prompt}
    margins: List[float] = []
    scores: List[List[float]] = []
    probabilities: List[List[float]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        compatible = [
            {
                "sample_id": record.sample_id,
                "source_image": record.source_image,
                "edit_instruction": record.edit_instruction,
                "edited_image_A": record.image_a,
                "edited_image_B": record.image_b,
            }
            for record in batch
        ]
        output = infer_batch(model, processor, compatible, contract, device, temperature=1.0)
        margins.extend(float(item["margin_A_minus_B"]) for item in output)
        scores.extend([float(item["score_A"]), float(item["score_B"])] for item in output)
        probabilities.extend(
            [item["probability_A"], item["probability_B"], item["probability_Tie"]]
            for item in output
        )
    return margins, scores, probabilities


def evaluate(name, model, processor, prompt, datasets, args, device, extra) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    predictions: Dict[str, Any] = {}
    for dataset_name, records in datasets.items():
        margins, scores, probabilities = score_dataset(
            model, processor, prompt, records, args.batch_size, device
        )
        if not all(math.isfinite(m) for m in margins):
            raise ValueError(f"{name} produced a non-finite margin on {dataset_name}.")
        signed_evidence = [p[0] - p[1] for p in probabilities]
        report = summarize(records, margins, None)
        report["probability"] = _probability_metrics(records, margins)
        report["margin_auc"] = _margin_auc(records, margins)
        report["signed_evidence_margin_auc"] = _margin_auc(records, signed_evidence)
        report["pooled_auc"] = _pooled_auc(records, scores)
        report["three_way"] = _three_way_metrics(records, probabilities, margins)
        metrics[dataset_name] = report
        predictions[dataset_name] = [
            {
                "sample_id": r.sample_id,
                "label": r.label,
                "margin_A_minus_B": m,
                "score_A": s[0],
                "score_B": s[1],
                "probability_A": p[0],
                "probability_B": p[1],
                "probability_Tie": p[2],
            }
            for r, m, s, p in zip(records, margins, scores, probabilities)
        ]
        three_way = report["three_way"]
        print(
            f"[{name}] {dataset_name}: accuracy={report['accuracy']['percent']}% "
            f"({report['accuracy']['correct']}/{report['accuracy']['total']}), "
            f"margin_auc={report['margin_auc']['percent']}%, "
            f"brier={report['probability']['brier_score']:.4f}, "
            f"three_class={three_way['three_class_accuracy_percent']}%, "
            f"tie_auc={three_way['tie_from_tie_head'].get('roc_auc_percent')}%",
            flush=True,
        )
    return {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_role": "overall_inspector",
        "checkpoint": name,
        **extra,
        "metrics": metrics,
        "predictions": predictions,
    }


def write_summary(output_dir: Path, names: Sequence[str], datasets: Sequence[str]) -> None:
    rows = []
    for name in names:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        row: Dict[str, Any] = {"checkpoint": name, "global_step": result.get("global_step")}
        for dataset in datasets:
            report = result["metrics"][dataset]
            three_way = report["three_way"]
            row[dataset] = {
                "accuracy_percent": report["accuracy"]["percent"],
                "margin_auc_percent": report["margin_auc"]["percent"],
                "signed_evidence_margin_auc_percent": report["signed_evidence_margin_auc"]["percent"],
                "pooled_auc_percent": report["pooled_auc"]["percent"],
                "brier_score": report["probability"]["brier_score"],
                "log_loss": report["probability"]["log_loss"],
                "three_class_accuracy_percent": three_way["three_class_accuracy_percent"],
                "three_way_log_loss": three_way["three_way_log_loss"],
                "tie_auc_from_tie_head_percent": three_way["tie_from_tie_head"].get("roc_auc_percent"),
                "tie_auc_from_absolute_margin_percent": three_way["tie_from_absolute_margin"].get(
                    "roc_auc_percent"
                ),
            }
        rows.append(row)
    summary = {"rows": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    header = ["checkpoint"] + list(datasets)
    print("\n" + " | ".join(header))
    for row in rows:
        cells = [row["checkpoint"]] + [
            f"{row[d]['accuracy_percent']:.2f} (AUC={row[d]['margin_auc_percent']:.2f}, "
            f"3-class={row[d]['three_class_accuracy_percent']:.2f}, "
            f"tieAUC={row[d]['tie_auc_from_tie_head_percent']})"
            for d in datasets
        ]
        print(" | ".join(cells))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoints = discover_checkpoints(run_dir, args.checkpoints)
    if not checkpoints and not args.include_base:
        raise RuntimeError(f"No checkpoint-* directories in {run_dir}")
    adapter_configs = [
        json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        for path in checkpoints
    ]
    if not adapter_configs:
        raise RuntimeError("--include-base needs at least one checkpoint for the LoRA structure.")

    def _normalized(value):
        return sorted(value) if isinstance(value, list) else value

    reference = adapter_configs[0]
    for path, config in zip(checkpoints, adapter_configs):
        for key in ("r", "lora_alpha", "target_modules", "modules_to_save"):
            if _normalized(config[key]) != _normalized(reference[key]):
                raise ValueError(f"{path.name} has a different LoRA {key}.")

    datasets = load_datasets(args)
    prompt = resolve_prompt(run_dir)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Prompt source: {prompt['source']}", flush=True)
    print(f"Checkpoints: {[p.name for p in checkpoints]}", flush=True)
    print({name: len(records) for name, records in datasets.items()}, flush=True)

    # Head structure follows the saved weights; every checkpoint of a run shares it.
    tie_modes = set()
    for path in checkpoints:
        with safe_open(str(path / "adapter_model.safetensors"), framework="pt") as handle:
            tie_modes.add(tie_head_mode_from_state_keys(handle.keys()))
    if len(tie_modes) != 1:
        raise ValueError(f"Checkpoints use different Tie Head structures: {sorted(tie_modes)}")
    tie_head_mode = tie_modes.pop()
    print(f"Tie Head structure: {tie_head_mode}", flush=True)
    model, processor = load_model(args, adapter_configs[0], tie_head_mode, device)
    token_stats = {
        name: image_token_stats(processor, prompt, records) for name, records in datasets.items()
    }
    print(f"Visual tokens per image: {token_stats}", flush=True)
    shared = {
        "run_dir": str(run_dir),
        "prompt": prompt,
        "image_token_stats": token_stats,
        "configuration": vars(args),
        "tie_head_structure": tie_head_mode,
    }

    names: List[str] = []
    if args.include_base:
        names.append(BASE_NAME)
        output_path = output_dir / f"{BASE_NAME}.json"
        if output_path.exists():
            print(f"Skipping existing result: {output_path}", flush=True)
        else:
            with model.disable_adapter():
                result = evaluate(BASE_NAME, model, processor, prompt, datasets, args, device,
                                  {**shared, "global_step": 0})
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for checkpoint_dir in checkpoints:
        names.append(checkpoint_dir.name)
        output_path = output_dir / f"{checkpoint_dir.name}.json"
        if output_path.exists():
            print(f"Skipping existing result: {output_path}", flush=True)
            continue
        load_adapter(model, checkpoint_dir)
        state_path = checkpoint_dir / "trainer_state.json"
        global_step = (
            json.loads(state_path.read_text(encoding="utf-8")).get("global_step")
            if state_path.is_file() else None
        )
        result = evaluate(checkpoint_dir.name, model, processor, prompt, datasets, args, device,
                          {**shared, "checkpoint_dir": str(checkpoint_dir), "global_step": global_step})
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    write_summary(output_dir, names, list(datasets))


if __name__ == "__main__":
    main()
