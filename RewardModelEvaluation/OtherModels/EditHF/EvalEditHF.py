#!/usr/bin/env python3
"""Unified evaluation of three-head EditHF ensemble baseline.

Evaluates on EditReward-Bench and EditHF-Test benchmarks.
Uses pretrained weights stored in /datastore/raccoon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from peft import PeftModel
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from evaluation_metrics import evaluate_pairwise, parse_ranking

DEFAULT_BENCHMARK_DIR = Path("/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/EvalDatasets")
DEFAULT_MODEL_PATH = Path("/datastore/raccoon/.cache/huggingface/models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79")
DEFAULT_CHECKPOINT_ROOT = Path("/datastore/raccoon/truongtq/ImageEditingBenchmark/TRAINING_REWARD_MODEL/ModelEvaluation/EditHF/pretrained")
DEFAULT_OUTPUT_DIR = Path("/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/EditHF/evaluation_results")

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
HEADS = {
    "alignment": "lora_checkpoints_editing",
    "preservation": "lora_checkpoints_preservation",
    "visual": "lora_checkpoints_visual",
}

BENCHMARK_FILES = {
    "editreward_bench": "EditReward/EditReward-Bench_pairwise.json",
    "edithf_test": "EditHF/EditHF-Test_pairwise_200.json",
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


def evaluate_single_benchmark(
    model: Any,
    tokenizer: Any,
    transform: Any,
    num_image_token: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_root: Path,
    benchmark_name: str,
    benchmark_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    print(f"\n{'='*70}\nEvaluating EditHF on: {benchmark_name} ({benchmark_path})\n{'='*70}", flush=True)
    samples = load_json(benchmark_path)
    benchmark_out = output_dir / benchmark_name
    benchmark_out.mkdir(parents=True, exist_ok=True)
    predictions_path = benchmark_out / "head_predictions.json"
    all_head_scores = load_json(predictions_path) if predictions_path.exists() else {}

    def image_tensor(path: str) -> torch.Tensor:
        with Image.open(path) as image:
            return transform(image.convert("RGB"))

    def query_with_tokens(instruction: str, mode: str) -> str:
        query = build_query(instruction, mode)
        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
        )
        return query.replace("<image>", image_tokens, 1).replace(
            "<image>", image_tokens, 1
        )

    for mode, directory in HEADS.items():
        if mode in all_head_scores and len(all_head_scores[mode]) == len(samples):
            print(f"Reusing complete {mode} predictions for {benchmark_name}", flush=True)
            continue

        model.vision_model.set_adapter(mode)
        model.language_model.set_adapter(mode)
        mlp = MLPRegressor(hidden_size).to(device=device, dtype=torch.float32)
        mlp_state = torch.load(
            checkpoint_root / directory / "mlp_head.pth",
            map_location="cpu",
            weights_only=True,
        )
        mlp.load_state_dict(mlp_state, strict=True)
        mlp.eval()

        head_scores = []
        with torch.inference_mode():
            for sample in tqdm(samples, desc=f"EditHF/{benchmark_name}/{mode}"):
                source = image_tensor(sample["source_image"])
                candidate_a = image_tensor(sample["candidate_1"])
                candidate_b = image_tensor(sample["candidate_2"])
                pixel_values = torch.stack(
                    [source, candidate_a, source, candidate_b], dim=0
                ).to(device=device, dtype=dtype)

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
                head_scores.append(
                    {"score_a": float(scores[0]), "score_b": float(scores[1])}
                )

        all_head_scores[mode] = head_scores
        save_json(predictions_path, all_head_scores)
        del mlp
        torch.cuda.empty_cache()

    labels = [parse_ranking(sample["metadata_ranking"]) for sample in samples]
    scores_a = [
        sum(all_head_scores[mode][i]["score_a"] for mode in HEADS) / len(HEADS)
        for i in range(len(samples))
    ]
    scores_b = [
        sum(all_head_scores[mode][i]["score_b"] for mode in HEADS) / len(HEADS)
        for i in range(len(samples))
    ]
    summary = evaluate_pairwise(labels, scores_a, scores_b)
    summary.update(
        {
            "model": "EditHF",
            "benchmark": benchmark_name,
            "benchmark_path": str(benchmark_path),
            "score_definition": (
                "Equal-weight mean of EditHF alignment, preservation, and visual scores"
            ),
            "checkpoint_root": str(checkpoint_root),
        }
    )

    details = []
    for i, sample in enumerate(samples):
        margin = scores_a[i] - scores_b[i]
        details.append(
            {
                "sample_id": sample.get("sample_id", str(i)),
                "ground_truth": sample["metadata_ranking"],
                "score_a": scores_a[i],
                "score_b": scores_b[i],
                "margin": margin,
                "tie_confidence": -abs(margin),
                "head_scores": {
                    mode: all_head_scores[mode][i] for mode in HEADS
                },
            }
        )
    save_json(benchmark_out / "summary.json", summary)
    save_json(benchmark_out / "detailed_samples.json", details)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EditHF on EditReward-Bench and EditHF-Test")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "editreward_bench", "edithf_test"],
        help="Benchmark to evaluate ('all', 'editreward_bench', or 'edithf_test')",
    )
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint_root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("EditHF evaluation requires a CUDA GPU")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    transform = T.Compose(
        [
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f"Loading InternVL base model from {args.model_path}", flush=True)
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

    # Determine benchmarks
    targets = (
        list(BENCHMARK_FILES.keys())
        if args.benchmark == "all"
        else [args.benchmark]
    )

    all_summaries = {}
    for name in targets:
        bench_file = args.benchmark_dir / BENCHMARK_FILES[name]
        if not bench_file.is_file():
            print(f"ERROR: Benchmark file not found: {bench_file}", file=sys.stderr)
            continue
        summary = evaluate_single_benchmark(
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            num_image_token=num_image_token,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            checkpoint_root=args.checkpoint_root,
            benchmark_name=name,
            benchmark_path=bench_file,
            output_dir=args.output_dir,
        )
        all_summaries[name] = summary

    # Save overall summary
    save_json(args.output_dir / "overall_summary.json", all_summaries)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - EDITHF BASELINE")
    print("=" * 70)
    for name, s in all_summaries.items():
        acc = s["non_tie_accuracy"]["percent"]
        tie_ap = s["tie_ranking"]["average_precision_percent"]
        tie_auc = s["tie_ranking"]["roc_auc_percent"]
        print(f"[{name.upper()}] Non-tie Acc: {acc}% | Tie AP: {tie_ap}% | Tie ROC-AUC: {tie_auc}%")
    print(f"Results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
