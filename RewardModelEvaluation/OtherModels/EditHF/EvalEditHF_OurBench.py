#!/usr/bin/env python3
"""Evaluation of three-head EditHF ensemble baseline on OurBench.json.

Groups:
  - Visual Quality: Aesthetics, Realism
  - Editing Alignment: Fidelity
  - Attribute Preservation: Foreground Consistency, Background Consistency, Structure Consistency

Evaluates each sample ONLY on its corresponding metric heads.
If a sample has both Aesthetics and Realism, Visual Quality is evaluated exactly once.
Results are saved to /datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/evaluation_results/ourbench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from peft import PeftModel
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluation_metrics import evaluate_pairwise, parse_ranking

DEFAULT_BENCHMARK_PATH = Path("/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/EvalDatasets/OurBench.json")
DEFAULT_MODEL_PATH = Path("/datastore/raccoon/.cache/huggingface/models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79")
DEFAULT_CHECKPOINT_ROOT = Path("/datastore/raccoon/truongtq/ImageEditingBenchmark/TRAINING_REWARD_MODEL/ModelEvaluation/EditHF/pretrained")
DEFAULT_OUTPUT_DIR = Path("/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/evaluation_results/ourbench")

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

HEADS = {
    "visual": "lora_checkpoints_visual",
    "alignment": "lora_checkpoints_editing",
    "preservation": "lora_checkpoints_preservation",
}

# Groupings mapping EditHF heads to OurBench dimension keys
HEAD_METRIC_MAP = {
    "visual": ["aesthetics", "realism", "aesthetic"],
    "alignment": ["fidelity"],
    "preservation": [
        "foreground_consistency",
        "background_consistency",
        "structure_consistency",
        "fc",
        "bc",
        "sc",
    ],
}

HEAD_DISPLAY_NAMES = {
    "visual": "Visual Quality",
    "alignment": "Editing Alignment",
    "preservation": "Attribute Preservation",
}


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 4096) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


def build_query(instruction: str, mode: str) -> str:
    aspects = {
        "visual": "visual quality",
        "alignment": "editing instruction alignment",
        "preservation": "attribute preservation",
    }
    return (
        f"With the image editing prompt [{instruction}], "
        "the source image <image> is edited to generate the target image <image>. "
        f"Please rate the target image from {aspects[mode]} aspect on a scale of 0-100. "
        "Just reply a sentence in the following format: "
        "[The quality score is XX.XX (a number with two decimal)]"
        " The quality score is 00.00"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def get_active_heads_for_sample(sample: Dict[str, Any]) -> List[str]:
    """Identify which EditHF heads apply to this sample based on present metric vote keys."""
    active = []
    for mode, keys in HEAD_METRIC_MAP.items():
        if any(f"{k}_1" in sample for k in keys):
            active.append(mode)
    return active


def get_head_ground_truth_label(sample: Dict[str, Any], mode: str) -> int:
    """Compute ground-truth pairwise label (1 for A, -1 for B, 0 for tie) for a specific head."""
    keys = HEAD_METRIC_MAP[mode]
    votes_a = sum(sample.get(f"{k}_1", 0) for k in keys)
    votes_b = sum(sample.get(f"{k}_2", 0) for k in keys)
    if votes_a > votes_b:
        return 1
    elif votes_b > votes_a:
        return -1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EditHF on OurBench.json with metric grouping and selective head execution"
    )
    parser.add_argument("--benchmark_path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EditHF Evaluation on OurBench.json")
    print("=" * 80)
    print(f"Benchmark Path  : {args.benchmark_path}")
    print(f"Base Model Path : {args.model_path}")
    print(f"Checkpoint Root : {args.checkpoint_root}")
    print(f"Output Directory: {output_dir}")
    print(f"Device          : {args.device}")
    print("=" * 80)

    if not args.benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark_path}")

    samples = load_json(args.benchmark_path)
    total_samples = len(samples)
    print(f"Loaded {total_samples} samples from {args.benchmark_path.name}\n")

    # 1. Analyze and print statistics of EditHF metric groups
    sample_to_active_heads: Dict[str, List[str]] = {}
    head_sample_ids: Dict[str, List[str]] = {m: [] for m in HEADS}
    comb_counter = Counter()

    for sample in samples:
        sid = sample["sample_id"]
        active = get_active_heads_for_sample(sample)
        sample_to_active_heads[sid] = active
        for mode in active:
            head_sample_ids[mode].append(sid)
        comb_counter[tuple(sorted(active))] += 1

    print("=" * 80)
    print("STATISTICS: EditHF METRIC GROUPS DISTRIBUTION ON OURBENCH")
    print("=" * 80)
    print(f"{'EditHF Metric Group':<26} | {'Mapped OurBench Metrics':<40} | {'Sample Count':<12}")
    print("-" * 80)
    print(f"{HEAD_DISPLAY_NAMES['visual']:<26} | {'Aesthetics, Realism':<40} | {len(head_sample_ids['visual']):<12}")
    print(f"{HEAD_DISPLAY_NAMES['alignment']:<26} | {'Fidelity':<40} | {len(head_sample_ids['alignment']):<12}")
    print(f"{HEAD_DISPLAY_NAMES['preservation']:<26} | {'Foreground/Background/Structure Cons.':<40} | {len(head_sample_ids['preservation']):<12}")
    print("-" * 80)
    print("Combinations of active metric groups per sample:")
    for comb, cnt in comb_counter.most_common():
        names = ", ".join(HEAD_DISPLAY_NAMES[m] for m in comb)
        print(f"  - [{names}]: {cnt} samples ({100.0 * cnt / total_samples:.2f}%)")
    print("=" * 80 + "\n")

    # 2. Check existing predictions for resume / caching
    predictions_path = output_dir / "head_predictions.json"
    all_head_scores: Dict[str, Dict[str, Dict[str, float]]] = (
        load_json(predictions_path) if predictions_path.exists() else {}
    )

    sample_by_id = {s["sample_id"]: s for s in samples}

    missing_by_head: Dict[str, List[str]] = {}
    for mode in ["visual", "alignment", "preservation"]:
        existing_sids = set(all_head_scores.get(mode, {}).keys())
        missing_by_head[mode] = [sid for sid in head_sample_ids[mode] if sid not in existing_sids]

    total_missing = sum(len(sids) for sids in missing_by_head.values())

    if total_missing > 0:
        # Check GPU environment
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this system.")

        device = torch.device(args.device)
        dtype = (
            torch.bfloat16
            if (args.device == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
            if args.device == "cuda"
            else torch.float32
        )

        transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def image_tensor(path: str) -> torch.Tensor:
            with Image.open(path) as image:
                return transform(image.convert("RGB"))

        # Load Model and Tokenizer
        print(f"Loading InternVL base model from {args.model_path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, use_fast=False, local_files_only=True
        )
        tokenizer.padding_side = "left"
        config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True
        )
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        image_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.img_context_token_id = image_context_id

        # Load multi-head LoRA adapters
        first_mode = next(iter(HEADS))
        first_dir = args.checkpoint_root / HEADS[first_mode]
        model.vision_model = PeftModel.from_pretrained(
            model.vision_model,
            first_dir / "vision_lora",
            adapter_name=first_mode,
            torch_dtype=dtype,
            is_trainable=False,
        )
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            first_dir / "llm_lora",
            adapter_name=first_mode,
            torch_dtype=dtype,
            is_trainable=False,
        )
        for mode, directory in list(HEADS.items())[1:]:
            head_dir = args.checkpoint_root / directory
            model.vision_model.load_adapter(head_dir / "vision_lora", adapter_name=mode)
            model.language_model.load_adapter(head_dir / "llm_lora", adapter_name=mode)

        model.to(device).eval()
        llm_config = getattr(config, "llm_config", None)
        hidden_size = int(getattr(llm_config, "hidden_size", getattr(config, "hidden_size", 4096)))

        def query_with_tokens(instruction: str, mode: str) -> str:
            query = build_query(instruction, mode)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
            return query.replace("<image>", image_tokens, 1).replace("<image>", image_tokens, 1)

        # 3. Head-level selective evaluation
        for mode in ["visual", "alignment", "preservation"]:
            directory = HEADS[mode]
            target_sids = set(head_sample_ids[mode])
            needed_sids = missing_by_head[mode]
            existing_count = len(target_sids) - len(needed_sids)
            print(f"\nProcessing head: {HEAD_DISPLAY_NAMES[mode]} ({mode}) - {len(target_sids)} target samples", flush=True)

            if mode not in all_head_scores:
                all_head_scores[mode] = {}

            if not needed_sids:
                print(f"All {len(target_sids)} predictions for {mode} already cached.", flush=True)
                continue

            print(f"Evaluating {len(needed_sids)} samples for {mode} (already cached: {existing_count})...", flush=True)

            model.vision_model.set_adapter(mode)
            model.language_model.set_adapter(mode)
            mlp = MLPRegressor(hidden_size).to(device=device, dtype=torch.float32)
            mlp_state = torch.load(
                args.checkpoint_root / directory / "mlp_head.pth",
                map_location="cpu",
                weights_only=True,
            )
            mlp.load_state_dict(mlp_state, strict=True)
            mlp.eval()

            with torch.inference_mode():
                for sid in tqdm(needed_sids, desc=f"EditHF/{mode}"):
                    sample = sample_by_id[sid]
                    source = image_tensor(sample["source_image"])
                    candidate_a = image_tensor(sample["candidate_1"])
                    candidate_b = image_tensor(sample["candidate_2"])
                    pixel_values = torch.stack([source, candidate_a, source, candidate_b], dim=0).to(
                        device=device, dtype=dtype
                    )

                    query = query_with_tokens(sample["edit_instruction"], mode)
                    inputs = tokenizer([query, query], padding=True, return_tensors="pt")
                    inputs = {key: val.to(device) for key, val in inputs.items()}
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image_flags=torch.ones((4, 1), dtype=torch.long, device=device),
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    features = outputs.hidden_states[-1][:, -6, :].float()
                    scores = mlp(features).abs().squeeze(-1) * 100.0
                    all_head_scores[mode][sid] = {
                        "score_a": float(scores[0]),
                        "score_b": float(scores[1]),
                    }

                    # Periodic save
                    if len(all_head_scores[mode]) % 20 == 0:
                        save_json(predictions_path, all_head_scores)

            save_json(predictions_path, all_head_scores)
            del mlp
            if args.device == "cuda":
                torch.cuda.empty_cache()
    else:
        print("All predictions for all heads are already cached in head_predictions.json.")
        print("Skipping model loading and proceeding directly to metric calculation.\n", flush=True)

    # 4. Compute Evaluation Metrics
    # A. Head-level Metrics
    head_summaries = {}
    for mode in ["visual", "alignment", "preservation"]:
        target_sids = head_sample_ids[mode]
        head_labels = [get_head_ground_truth_label(sample_by_id[sid], mode) for sid in target_sids]
        head_scores_a = [all_head_scores[mode][sid]["score_a"] for sid in target_sids]
        head_scores_b = [all_head_scores[mode][sid]["score_b"] for sid in target_sids]

        summary_h = evaluate_pairwise(head_labels, head_scores_a, head_scores_b)
        summary_h["metric_group"] = HEAD_DISPLAY_NAMES[mode]
        summary_h["mode"] = mode
        summary_h["total_evaluated_samples"] = len(target_sids)
        head_summaries[mode] = summary_h

    # B. Overall Metrics (averaged over active heads for each sample)
    overall_labels = [parse_ranking(sample["metadata_ranking"]) for sample in samples]
    overall_scores_a = []
    overall_scores_b = []

    detailed_records = []
    for sample in samples:
        sid = sample["sample_id"]
        active = sample_to_active_heads[sid]
        sa = sum(all_head_scores[m][sid]["score_a"] for m in active) / len(active)
        sb = sum(all_head_scores[m][sid]["score_b"] for m in active) / len(active)
        overall_scores_a.append(sa)
        overall_scores_b.append(sb)

        margin = sa - sb
        rec = {
            "sample_id": sid,
            "ground_truth": sample["metadata_ranking"],
            "active_heads": active,
            "score_a": sa,
            "score_b": sb,
            "margin": margin,
            "tie_confidence": -abs(margin),
            "head_scores": {m: all_head_scores[m][sid] for m in active},
        }
        detailed_records.append(rec)

    overall_summary = evaluate_pairwise(overall_labels, overall_scores_a, overall_scores_b)
    overall_summary.update(
        {
            "model": "EditHF",
            "benchmark": "OurBench",
            "benchmark_path": str(args.benchmark_path),
            "score_definition": "Average score over active EditHF heads corresponding to sample metrics",
            "grouping": {
                "Visual Quality": "Aesthetics, Realism (evaluated once per sample)",
                "Editing Alignment": "Fidelity",
                "Attribute Preservation": "Foreground Consistency, Background Consistency, Structure Consistency",
            },
            "sample_statistics": {
                "total_benchmark_samples": total_samples,
                "visual_quality_samples": len(head_sample_ids["visual"]),
                "editing_alignment_samples": len(head_sample_ids["alignment"]),
                "attribute_preservation_samples": len(head_sample_ids["preservation"]),
            },
            "per_head_metrics": head_summaries,
        }
    )

    # Macro-average of the 3 metrics
    macro_acc = sum(head_summaries[m]["non_tie_accuracy"]["percent"] for m in HEADS) / len(HEADS)
    macro_tie_ap = sum(head_summaries[m]["tie_ranking"]["average_precision_percent"] for m in HEADS) / len(HEADS)
    macro_tie_auc = sum(head_summaries[m]["tie_ranking"]["roc_auc_percent"] for m in HEADS) / len(HEADS)

    # Micro-average across all head evaluations
    total_head_correct = sum(head_summaries[m]["non_tie_accuracy"]["correct"] for m in HEADS)
    total_head_non_tie = sum(head_summaries[m]["non_tie_accuracy"]["total"] for m in HEADS)
    micro_acc = (total_head_correct / total_head_non_tie * 100.0) if total_head_non_tie > 0 else 0.0

    overall_summary["macro_average"] = {
        "non_tie_accuracy_percent": round(macro_acc, 2),
        "tie_ap_percent": round(macro_tie_ap, 2),
        "tie_roc_auc_percent": round(macro_tie_auc, 2),
    }
    overall_summary["micro_average"] = {
        "non_tie_accuracy_percent": round(micro_acc, 2),
        "correct": total_head_correct,
        "total": total_head_non_tie,
    }

    # 5. Save all results
    save_json(output_dir / "summary.json", overall_summary)
    save_json(output_dir / "detailed_samples.json", detailed_records)

    # 6. Display Final Summary
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS - EDITHF ON OURBENCH")
    print("=" * 80)
    print(f"{'Metric':<26} | {'Samples':<8} | {'Non-tie Acc (%)':<18} | {'Tie AP (%)':<14} | {'Tie ROC-AUC (%)':<14}")
    print("-" * 80)
    for mode in ["visual", "alignment", "preservation"]:
        hs = head_summaries[mode]
        acc = hs["non_tie_accuracy"]["percent"]
        tie_ap = hs["tie_ranking"]["average_precision_percent"]
        tie_auc = hs["tie_ranking"]["roc_auc_percent"]
        n_samp = hs["total_evaluated_samples"]
        print(f"{HEAD_DISPLAY_NAMES[mode]:<26} | {n_samp:<8} | {acc:<18.2f} | {tie_ap:<14.2f} | {tie_auc:<14.2f}")
    print("-" * 80)
    print(f"{'OVERALL (Macro Average)':<26} | {'-':<8} | {macro_acc:<18.2f} | {macro_tie_ap:<14.2f} | {macro_tie_auc:<14.2f}")
    print(f"{'OVERALL (Micro Average)':<26} | {total_head_non_tie:<8} | {micro_acc:<18.2f} | {'-':<14} | {'-':<14}")
    o_acc = overall_summary["non_tie_accuracy"]["percent"]
    o_ap = overall_summary["tie_ranking"]["average_precision_percent"]
    o_auc = overall_summary["tie_ranking"]["roc_auc_percent"]
    print(f"{'OVERALL (Sample Ensemble)':<26} | {total_samples:<8} | {o_acc:<18.2f} | {o_ap:<14.2f} | {o_auc:<14.2f}")
    print("=" * 80)
    print(f"Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
