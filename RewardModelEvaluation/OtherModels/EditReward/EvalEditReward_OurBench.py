#!/usr/bin/env python3
"""Evaluate official EditReward-Qwen2.5-VL-7B on OurBench.json.

Uses the two native EditReward reward dimensions from the official multi-head model:
  - dim1 / Instruction Following: Fidelity, BC, FC, SC
  - dim2 / Visual Quality: Aesthetics, Realism

Each candidate is inferred once and both native EditReward head scores are kept.
OurBench annotations are grouped to the corresponding EditReward dimension.

Outputs:
  predictions.json
  summary.json
  detailed_samples.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluation_metrics import evaluate_pairwise, parse_ranking

DEFAULT_BENCHMARK_PATH = Path(
    "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/EvalDatasets/OurBench.json"
)
DEFAULT_SOURCE_ROOT = Path(
    "/datastore/raccoon/truongtq/ImageEditingBenchmark/TRAINING_REWARD_MODEL/ModelEvaluation/EditReward"
)
DEFAULT_MODEL_DIR = DEFAULT_SOURCE_ROOT / "pretrained"
DEFAULT_FALLBACK_DIR = Path("/tmp/EditReward-Qwen2.5-VL-7B")
DEFAULT_OUTPUT_DIR = Path(
    "/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OtherModels/"
    "evaluation_results/ourbench/editreward"
)

# Native EditReward dimensions. The official inferencer constructs dim1 with
# INSTRUCTION_EDIT_FOLLOWING and dim2 with INSTRUCTION_EDIT_QUALITY when
# rm_head_type="ranknet_multi_head".
GROUPS: Dict[str, Dict[str, Any]] = {
    "instruction_following": {
        "display": "Instruction Following",
        "reward_column": 0,  # EditReward dim1
        "metrics": {
            "fidelity": ["fidelity"],
            "background_consistency": ["background_consistency", "bc"],
            "foreground_consistency": ["foreground_consistency", "fc"],
            "structure_consistency": ["structure_consistency", "sc"],
        },
    },
    "visual_quality": {
        "display": "Visual Quality",
        "reward_column": 1,  # EditReward dim2
        "metrics": {
            "aesthetic": ["aesthetic", "aesthetics"],
            "realism": ["realism"],
        },
    },
}

METRIC_DISPLAY = {
    "aesthetic": "Aesthetic",
    "realism": "Realism",
    "fidelity": "Fidelity",
    "background_consistency": "Background Consistency",
    "foreground_consistency": "Foreground Consistency",
    "structure_consistency": "Structure Consistency",
}



def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _as_number(value: Any) -> Optional[float]:
    """Return a numeric vote value when possible; otherwise None."""
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def metric_vote_pair(sample: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """Find A/B vote fields for one canonical OurBench metric."""
    aliases = None
    for group_cfg in GROUPS.values():
        if metric in group_cfg["metrics"]:
            aliases = group_cfg["metrics"][metric]
            break
    if aliases is None:
        raise KeyError(f"Unknown metric: {metric}")

    for alias in aliases:
        key_a, key_b = f"{alias}_1", f"{alias}_2"
        if key_a in sample or key_b in sample:
            va = _as_number(sample.get(key_a, 0))
            vb = _as_number(sample.get(key_b, 0))
            if va is not None and vb is not None:
                return va, vb
    return None


def get_active_metrics(sample: Dict[str, Any], group: Optional[str] = None) -> List[str]:
    groups = [group] if group is not None else list(GROUPS)
    active: List[str] = []
    for group_name in groups:
        for metric in GROUPS[group_name]["metrics"]:
            if metric_vote_pair(sample, metric) is not None:
                active.append(metric)
    return active


def get_active_groups(sample: Dict[str, Any]) -> List[str]:
    return [g for g in GROUPS if get_active_metrics(sample, g)]


def get_metric_ground_truth_label(sample: Dict[str, Any], metric: str) -> int:
    votes = metric_vote_pair(sample, metric)
    if votes is None:
        raise KeyError(f"Sample {sample.get('sample_id')} has no annotation for metric {metric}")
    votes_a, votes_b = votes
    return 1 if votes_a > votes_b else -1 if votes_b > votes_a else 0


def get_group_ground_truth_label(sample: Dict[str, Any], group: str) -> int:
    """Aggregate all available OurBench votes belonging to one EditReward dimension."""
    votes_a = 0.0
    votes_b = 0.0
    found = False
    for metric in GROUPS[group]["metrics"]:
        pair = metric_vote_pair(sample, metric)
        if pair is not None:
            found = True
            votes_a += pair[0]
            votes_b += pair[1]
    if not found:
        raise KeyError(f"Sample {sample.get('sample_id')} has no annotation for group {group}")
    return 1 if votes_a > votes_b else -1 if votes_b > votes_a else 0



def download_editreward_weights(destination: Path) -> Path:
    """Download EditReward weights + processor and align vocabulary dimensions."""
    print(f"Preparing EditReward weights at {destination}...", flush=True)
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(
            f"WARNING: Permission denied writing to {destination}. Switching to fallback /tmp...",
            flush=True,
        )
        destination = DEFAULT_FALLBACK_DIR
        destination.mkdir(parents=True, exist_ok=True)

    checkpoint = destination / "model.safetensors"
    if not checkpoint.is_file():
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        print(f"Downloading TIGER-Lab/EditReward-Qwen2.5-VL-7B to {destination}...", flush=True)
        snapshot_download(repo_id="TIGER-Lab/EditReward-Qwen2.5-VL-7B", local_dir=destination)

        processor_dir = destination.parent / "qwen_processor"
        processor_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            local_dir=processor_dir,
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
        )
        for source in processor_dir.iterdir():
            target = destination / source.name
            if target.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint was not downloaded successfully: {checkpoint}")

        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            vocab_keys = [
                key
                for key in handle.keys()
                if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")
            ]
            if not vocab_keys:
                raise KeyError("Could not find an embedding or lm_head tensor in checkpoint")
            vocab_sizes = {handle.get_slice(key).get_shape()[0] for key in vocab_keys}
            if len(vocab_sizes) != 1:
                raise ValueError(f"Inconsistent vocabulary sizes in checkpoint: {vocab_sizes}")
            checkpoint_vocab_size = vocab_sizes.pop()

        config_path = destination / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["vocab_size"] = checkpoint_vocab_size
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Aligned runtime vocab_size to {checkpoint_vocab_size}", flush=True)

    return destination


def write_runtime_config(template: Path, output: Path, model_dir: Path) -> None:
    lines = template.read_text(encoding="utf-8").splitlines()
    replacements = {
        "model_name_or_path:": f'model_name_or_path: "{model_dir}"',
        "output_dir:": f'output_dir: "{output.parent / "runtime_output"}"',
    }
    result: List[str] = []
    for line in lines:
        stripped = line.strip()
        replacement = next(
            (value for prefix, value in replacements.items() if stripped.startswith(prefix)),
            None,
        )
        result.append(replacement if replacement is not None else line)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


def run_inference(
    inferencer: Any,
    samples: List[Dict[str, Any]],
    predictions_path: Path,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Run EditReward once per sample, with resumable prediction caching."""
    cached = load_json(predictions_path) if predictions_path.exists() else []
    cache_by_id = {str(row["sample_id"]): row for row in cached}
    sample_ids = [str(s.get("sample_id", i)) for i, s in enumerate(samples)]
    missing_indices = [i for i, sid in enumerate(sample_ids) if sid not in cache_by_id]

    print(
        f"Predictions cached: {len(cache_by_id)} | missing: {len(missing_indices)} / {len(samples)}",
        flush=True,
    )

    for pos in tqdm(range(0, len(missing_indices), batch_size), desc="EditReward/OurBench"):
        indices = missing_indices[pos : pos + batch_size]
        batch = [samples[i] for i in indices]
        prompts: List[str] = []
        sources: List[str] = []
        candidates: List[str] = []

        for sample in batch:
            prompts.extend([sample["edit_instruction"], sample["edit_instruction"]])
            sources.extend([sample["source_image"], sample["source_image"]])
            candidates.extend([sample["candidate_1"], sample["candidate_2"]])

        with torch.inference_mode():
            rewards = inferencer.reward(
                prompts=prompts,
                image_src=sources,
                image_paths=candidates,
            )

        rewards = rewards.detach().float().cpu()
        if rewards.ndim != 2 or rewards.shape[1] < 2:
            raise RuntimeError(
                f"Expected EditReward ranknet_multi_head logits with >=2 columns "
                f"(dim1 Instruction Following, dim2 Visual Quality), got {tuple(rewards.shape)}"
            )

        # Official EditReward mapping:
        #   rewards[:, 0] = dim1 = Instruction Following
        #   rewards[:, 1] = dim2 = Visual Quality
        for offset, sample_index in enumerate(indices):
            sid = sample_ids[sample_index]
            a_idx = 2 * offset
            b_idx = 2 * offset + 1
            cache_by_id[sid] = {
                "sample_id": sid,
                "instruction_following_score_a": float(rewards[a_idx, 0]),
                "instruction_following_score_b": float(rewards[b_idx, 0]),
                "visual_quality_score_a": float(rewards[a_idx, 1]),
                "visual_quality_score_b": float(rewards[b_idx, 1]),
            }

        # Save in benchmark order so resume is deterministic and human-readable.
        ordered = [cache_by_id[sid] for sid in sample_ids if sid in cache_by_id]
        save_json(predictions_path, ordered)

    return [cache_by_id[sid] for sid in sample_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official EditReward on OurBench.json")
    parser.add_argument("--benchmark_path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    if not args.benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {args.benchmark_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("EditReward evaluation requires a CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_json(args.benchmark_path)
    if not isinstance(samples, list):
        raise TypeError("OurBench JSON must contain a top-level list of samples")
    total_samples = len(samples)

    print("=" * 88)
    print("EDITREWARD EVALUATION ON OURBENCH")
    print("=" * 88)
    print(f"Benchmark : {args.benchmark_path}")
    print(f"Samples   : {total_samples}")
    print(f"Output    : {args.output_dir}")

    # Analyze native EditReward group coverage before model loading.
    sample_active_metrics: Dict[str, List[str]] = {}
    sample_active_groups: Dict[str, List[str]] = {}
    group_sample_ids: Dict[str, List[str]] = {g: [] for g in GROUPS}
    metric_sample_ids: Dict[str, List[str]] = {
        m: [] for g in GROUPS.values() for m in g["metrics"]
    }
    sample_by_id: Dict[str, Dict[str, Any]] = {}

    for i, sample in enumerate(samples):
        sid = str(sample.get("sample_id", i))
        sample_by_id[sid] = sample
        active_metrics = get_active_metrics(sample)
        active_groups = get_active_groups(sample)
        sample_active_metrics[sid] = active_metrics
        sample_active_groups[sid] = active_groups
        for group in active_groups:
            group_sample_ids[group].append(sid)
        for metric in active_metrics:
            metric_sample_ids[metric].append(sid)

    print("-" * 88)
    print(f"{'EditReward dimension':<30} | {'Mapped OurBench metrics':<38} | {'Samples':>8}")
    print("-" * 88)
    for group, cfg in GROUPS.items():
        mapped = ", ".join(METRIC_DISPLAY[m] for m in cfg["metrics"])
        print(f"{cfg['display']:<30} | {mapped:<38} | {len(group_sample_ids[group]):>8}")
    print("=" * 88)

    # Prepare official EditReward model exactly as in EvalEditReward.py.
    model_dir = args.model_dir
    if not (model_dir / "model.safetensors").is_file():
        model_dir = download_editreward_weights(model_dir)

    sys.path.insert(0, str(args.source_root))
    from transformers import AutoProcessor

    original_from_pretrained = AutoProcessor.from_pretrained

    def slow_processor_from_pretrained(cls, pretrained_model_name_or_path, *p_args, **p_kwargs):
        p_kwargs.setdefault("use_fast", False)
        return original_from_pretrained(pretrained_model_name_or_path, *p_args, **p_kwargs)

    AutoProcessor.from_pretrained = classmethod(slow_processor_from_pretrained)
    from EditReward import EditRewardInferencer

    config_template = args.source_root / "EditReward" / "config" / "EditReward-Qwen2.5-7B-VL.yaml"
    if not config_template.is_file():
        raise FileNotFoundError(f"EditReward config template not found: {config_template}")
    runtime_config = args.output_dir / "runtime_config.yaml"
    write_runtime_config(config_template, runtime_config, model_dir)

    print(f"Loading official EditReward model from {model_dir}", flush=True)
    inferencer = EditRewardInferencer(
        config_path=str(runtime_config),
        checkpoint_path=str(model_dir),
        device="cuda",
        reward_dim="dim1",  # multi-head path computes both dim1 and dim2
        rm_head_type="ranknet_multi_head",
    )

    predictions = run_inference(
        inferencer=inferencer,
        samples=samples,
        predictions_path=args.output_dir / "predictions.json",
        batch_size=args.batch_size,
    )
    pred_by_id = {str(p["sample_id"]): p for p in predictions}

    # 1) Evaluate the two native EditReward dimensions.
    group_summaries: Dict[str, Any] = {}
    for group, cfg in GROUPS.items():
        sids = group_sample_ids[group]
        labels = [get_group_ground_truth_label(sample_by_id[sid], group) for sid in sids]
        scores_a = [pred_by_id[sid][f"{group}_score_a"] for sid in sids]
        scores_b = [pred_by_id[sid][f"{group}_score_b"] for sid in sids]
        summary = evaluate_pairwise(labels, scores_a, scores_b)
        summary.update({
            "dimension": cfg["display"],
            "dimension_key": group,
            "editreward_head": "dim1" if group == "instruction_following" else "dim2",
            "mapped_ourbench_metrics": [METRIC_DISPLAY[m] for m in cfg["metrics"]],
            "total_evaluated_samples": len(sids),
        })
        group_summaries[group] = summary

    # 2) Optional per-OurBench-dimension breakdown using the corresponding native head.
    metric_summaries: Dict[str, Any] = {}
    for group, cfg in GROUPS.items():
        for metric in cfg["metrics"]:
            sids = metric_sample_ids[metric]
            if not sids:
                continue
            labels = [get_metric_ground_truth_label(sample_by_id[sid], metric) for sid in sids]
            scores_a = [pred_by_id[sid][f"{group}_score_a"] for sid in sids]
            scores_b = [pred_by_id[sid][f"{group}_score_b"] for sid in sids]
            sm = evaluate_pairwise(labels, scores_a, scores_b)
            sm.update({
                "metric": METRIC_DISPLAY[metric],
                "metric_key": metric,
                "editreward_dimension": cfg["display"],
                "total_evaluated_samples": len(sids),
            })
            metric_summaries[metric] = sm

    # 3) Sample-level ensemble: average only the native EditReward dimensions active
    # for that sample. This is NOT the model's overall head.
    ensemble_labels: List[int] = []
    ensemble_scores_a: List[float] = []
    ensemble_scores_b: List[float] = []
    ensemble_sids: List[str] = []
    for i, sample in enumerate(samples):
        sid = str(sample.get("sample_id", i))
        active_groups = sample_active_groups[sid]
        if not active_groups:
            continue
        ensemble_sids.append(sid)
        ensemble_labels.append(parse_ranking(sample["metadata_ranking"]))
        ensemble_scores_a.append(
            sum(pred_by_id[sid][f"{g}_score_a"] for g in active_groups) / len(active_groups)
        )
        ensemble_scores_b.append(
            sum(pred_by_id[sid][f"{g}_score_b"] for g in active_groups) / len(active_groups)
        )

    overall_summary = evaluate_pairwise(ensemble_labels, ensemble_scores_a, ensemble_scores_b)

    macro_acc = sum(group_summaries[g]["non_tie_accuracy"]["percent"] for g in GROUPS) / len(GROUPS)
    macro_ap = sum(group_summaries[g]["tie_ranking"]["average_precision_percent"] for g in GROUPS) / len(GROUPS)
    macro_auc = sum(group_summaries[g]["tie_ranking"]["roc_auc_percent"] for g in GROUPS) / len(GROUPS)
    micro_correct = sum(group_summaries[g]["non_tie_accuracy"]["correct"] for g in GROUPS)
    micro_total = sum(group_summaries[g]["non_tie_accuracy"]["total"] for g in GROUPS)
    micro_acc = 100.0 * micro_correct / micro_total if micro_total else 0.0

    overall_summary.update({
        "model": "EditReward-Qwen2.5-VL-7B",
        "benchmark": "OurBench",
        "benchmark_path": str(args.benchmark_path),
        "native_dimension_mapping": {
            "Instruction Following (dim1)": ["Fidelity", "Background Consistency", "Foreground Consistency", "Structure Consistency"],
            "Visual Quality (dim2)": ["Aesthetic", "Realism"],
        },
        "score_definition": (
            "Uses the two native outputs of EditReward ranknet_multi_head directly: "
            "column 0 = Instruction Following (dim1), column 1 = Visual Quality (dim2)."
        ),
        "per_editreward_dimension": group_summaries,
        "per_ourbench_metric": metric_summaries,
        "macro_average_over_editreward_dimensions": {
            "non_tie_accuracy_percent": round(macro_acc, 2),
            "tie_ap_percent": round(macro_ap, 2),
            "tie_roc_auc_percent": round(macro_auc, 2),
        },
        "micro_average_over_editreward_dimensions": {
            "non_tie_accuracy_percent": round(micro_acc, 2),
            "correct": micro_correct,
            "total": micro_total,
        },
        "sample_ensemble": {
            "note": "Average of only active native EditReward dimensions; no overall reward head is used.",
            "samples": len(ensemble_sids),
        },
    })

    # Detailed records preserve both native head scores and grouped GT labels.
    detailed_records: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        sid = str(sample.get("sample_id", i))
        pred = pred_by_id[sid]
        active_groups = sample_active_groups[sid]
        rec: Dict[str, Any] = {
            **pred,
            "ground_truth": sample.get("metadata_ranking"),
            "active_groups": active_groups,
            "active_metrics": sample_active_metrics[sid],
            "group_labels": {g: get_group_ground_truth_label(sample, g) for g in active_groups},
            "metric_labels": {m: get_metric_ground_truth_label(sample, m) for m in sample_active_metrics[sid]},
        }
        if active_groups:
            sa = sum(pred[f"{g}_score_a"] for g in active_groups) / len(active_groups)
            sb = sum(pred[f"{g}_score_b"] for g in active_groups) / len(active_groups)
            rec.update({
                "ensemble_score_a": sa,
                "ensemble_score_b": sb,
                "ensemble_margin": sa - sb,
                "ensemble_tie_confidence": -abs(sa - sb),
            })
        detailed_records.append(rec)

    save_json(args.output_dir / "summary.json", overall_summary)
    save_json(args.output_dir / "detailed_samples.json", detailed_records)

    print("\n" + "=" * 104)
    print("FINAL EVALUATION RESULTS - EDITREWARD NATIVE DIMS ON OURBENCH")
    print("=" * 104)
    print(
        f"{'EditReward dimension':<30} | {'Samples':>8} | {'Non-tie Acc (%)':>16} | "
        f"{'Tie AP (%)':>12} | {'Tie ROC-AUC (%)':>16}"
    )
    print("-" * 104)
    for group, cfg in GROUPS.items():
        gs = group_summaries[group]
        print(
            f"{cfg['display']:<30} | {gs['total_evaluated_samples']:>8} | "
            f"{gs['non_tie_accuracy']['percent']:>16.2f} | "
            f"{gs['tie_ranking']['average_precision_percent']:>12.2f} | "
            f"{gs['tie_ranking']['roc_auc_percent']:>16.2f}"
        )
    print("-" * 104)
    print(f"{'NATIVE-DIM MACRO':<30} | {'-':>8} | {macro_acc:>16.2f} | {macro_ap:>12.2f} | {macro_auc:>16.2f}")
    print(f"{'NATIVE-DIM MICRO':<30} | {micro_total:>8} | {micro_acc:>16.2f} | {'-':>12} | {'-':>16}")
    print("-" * 104)
    print(
        f"{'SAMPLE ENSEMBLE':<30} | {len(ensemble_sids):>8} | "
        f"{overall_summary['non_tie_accuracy']['percent']:>16.2f} | "
        f"{overall_summary['tie_ranking']['average_precision_percent']:>12.2f} | "
        f"{overall_summary['tie_ranking']['roc_auc_percent']:>16.2f}"
    )
    print("=" * 104)
    print(f"Results saved to: {args.output_dir}\n")



if __name__ == "__main__":
    main()
