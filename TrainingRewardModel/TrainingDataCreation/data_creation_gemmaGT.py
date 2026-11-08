#!/usr/bin/env python3
"""Create the three-run GemmaGT JSONL consumed by Student warm-start.

The Gemma generation job writes one JSONL row per inference run. This script
groups runs 1, 2, and 3 by ``sample_id`` and converts every judgement back to
the original candidate order. It deliberately keeps all three runs instead
of applying majority voting because the warm-start loss learns from the
empirical A/B/Tie distribution and the mean relevance target.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping


GEMMAGT_RAW = Path(
    "/datastore/truongtq/TrainingRewardModel/GemmaGenGroundtruth/results/editreward_overall_humanGT_GemmaGT_3k.jsonl"
)
GEMMAGT_TRAINING_DATA = Path(
    "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation/editreward_overall_finegrained_gemmaGT_3k.jsonl"
)

METRIC_ORDER = (
    "aesthetic",
    "fidelity",
    "realism",
    "background_consistency",
    "foreground_consistency",
    "structure_consistency",
)
METRIC_FIELDS = ("targeted_metric", "reason", "choice")
CHOICES = ("A", "B", "Tie")
REQUIRED_RUNS = (1, 2, 3)
REQUIRED_TEXT_FIELDS = (
    "sample_id",
    "source_image",
    "edit_instruction",
    "edited_image_A",
    "edited_image_B",
)
IMAGE_FIELDS = ("source_image", "edited_image_A", "edited_image_B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group three Gemma runs per sample and create canonical GemmaGT "
            "JSONL for six-metric Student warm-start."
        )
    )
    parser.add_argument("--input-file", type=Path, default=GEMMAGT_RAW)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Read every worker-*/part-*.jsonl in this folder instead of --input-file.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Prefix for relative source/A/B image paths in the input.",
    )
    parser.add_argument("--output-file", type=Path, default=GEMMAGT_TRAINING_DATA)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of complete samples to write; default writes every sample.",
    )
    parser.add_argument(
        "--selection",
        choices=("first", "random"),
        default="first",
        help="Select samples in input order or deterministically at random.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Do not check whether source/A/B image files exist.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize the input without writing the output file.",
    )
    return parser.parse_args()


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def validate_metric_groundtruth(raw_output: Any) -> None:
    """Validate the strict six-metric object emitted by Gemma."""
    if not isinstance(raw_output, dict) or tuple(raw_output) != METRIC_ORDER:
        raise ValueError(
            "raw_output must contain exactly these metrics in order: "
            f"{list(METRIC_ORDER)}"
        )

    for metric_name in METRIC_ORDER:
        annotation = raw_output[metric_name]
        if not isinstance(annotation, dict) or tuple(annotation) != METRIC_FIELDS:
            raise ValueError(
                f"{metric_name} must contain fields in order: {list(METRIC_FIELDS)}"
            )

        targeted = annotation["targeted_metric"]
        reason = annotation["reason"]
        choice = annotation["choice"]
        if type(targeted) is not int or targeted not in (0, 1):
            raise ValueError(f"{metric_name}.targeted_metric must be integer 0 or 1")
        if metric_name == "fidelity" and targeted != 1:
            raise ValueError("fidelity.targeted_metric must be 1")
        if choice not in CHOICES:
            raise ValueError(f"{metric_name}.choice must be A, B, or Tie")
        if targeted == 0 and choice != "Tie":
            raise ValueError(f"{metric_name}.choice must be Tie when inactive")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{metric_name}.reason must be a non-empty string")
        if "\n" in reason or "\r" in reason:
            raise ValueError(f"{metric_name}.reason must be one line")


def validate_run(record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("each JSONL row must be an object")
    for key in REQUIRED_TEXT_FIELDS:
        _required_text(record, key)

    run_index = record.get("run_index")
    if type(run_index) is not int or run_index not in REQUIRED_RUNS:
        raise ValueError("run_index must be integer 1, 2, or 3")
    if type(record.get("is_swapped")) is not bool:
        raise ValueError("is_swapped must be boolean")
    if type(record.get("seed")) is not int:
        raise ValueError("seed must be integer")
    validate_metric_groundtruth(record.get("raw_output"))


def canonical_choice(choice: str, is_swapped: bool) -> str:
    """Map a displayed A/B judgement back to the original candidate order."""
    if not is_swapped or choice == "Tie":
        return choice
    return "B" if choice == "A" else "A"


def canonical_reason(reason: str, is_swapped: bool) -> str:
    """Map Image A/Image B mentions back to the original candidate order."""
    reason = reason.strip()
    if not is_swapped:
        return reason
    return (
        reason.replace("Image A", "__ORIGINAL_IMAGE_B__")
        .replace("Image B", "Image A")
        .replace("__ORIGINAL_IMAGE_B__", "Image B")
    )


def canonical_metric_groundtruth(
    raw_output: Mapping[str, Mapping[str, Any]],
    is_swapped: bool,
) -> dict[str, dict[str, Any]]:
    canonical: dict[str, dict[str, Any]] = {}
    for metric_name in METRIC_ORDER:
        annotation = raw_output[metric_name]
        canonical[metric_name] = {
            "targeted_metric": annotation["targeted_metric"],
            "reason": canonical_reason(annotation["reason"], is_swapped),
            "choice": canonical_choice(annotation["choice"], is_swapped),
        }
    return canonical


def resolve_input_files(input_file: Path, input_dir: Path | None) -> list[Path]:
    """Return the run JSONL files: one file, or every worker part file."""
    if input_dir is None:
        if not input_file.is_file():
            raise FileNotFoundError(f"Input JSONL does not exist: {input_file}")
        return [input_file]
    files = sorted(input_dir.glob("worker-*/part-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No worker-*/part-*.jsonl files in: {input_dir}")
    return files


def iter_lines(input_files: Iterable[Path]) -> Iterable[tuple[str, str]]:
    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                yield f"{input_file}:{line_number}", line


def load_complete_samples(
    input_files: list[Path],
    *,
    check_images: bool = True,
    image_root: str | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Load, validate, group, and canonicalize complete three-run samples."""
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    invalid_sample_ids: set[str] = set()
    stats: Counter[str] = Counter()

    for line_number, line in iter_lines(input_files):
        if not line.strip():
            continue
        stats["nonempty_lines"] += 1
        sample_id = ""
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                sample_id = str(record.get("sample_id", "")).strip()
            validate_run(record)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            stats["invalid_run_lines"] += 1
            if sample_id:
                invalid_sample_ids.add(sample_id)
            print(f"Skipping invalid line {line_number}: {error}")
            continue

        metadata = {key: _required_text(record, key) for key in REQUIRED_TEXT_FIELDS}
        if image_root:
            for key in IMAGE_FIELDS:
                if not metadata[key].startswith("/"):
                    metadata[key] = str(Path(image_root) / metadata[key])
        edit_type = record.get("edit_type")
        if edit_type is not None:
            if not isinstance(edit_type, str) or not edit_type.strip():
                stats["invalid_run_lines"] += 1
                invalid_sample_ids.add(sample_id)
                print(f"Skipping invalid line {line_number}: invalid edit_type")
                continue
            edit_type = edit_type.strip()

        if sample_id not in grouped:
            grouped[sample_id] = {
                "metadata": metadata,
                "edit_type": edit_type,
                "runs": {},
            }
        sample = grouped[sample_id]
        if sample["metadata"] != metadata or sample["edit_type"] != edit_type:
            stats["metadata_conflicts"] += 1
            invalid_sample_ids.add(sample_id)
            continue

        run_index = record["run_index"]
        if run_index in sample["runs"]:
            if sample["runs"][run_index] == record:
                stats["identical_duplicate_runs"] += 1
            else:
                stats["conflicting_duplicate_runs"] += 1
                invalid_sample_ids.add(sample_id)
            continue
        sample["runs"][run_index] = record

    complete_samples: list[dict[str, Any]] = []
    for sample_id, sample in grouped.items():
        if sample_id in invalid_sample_ids:
            stats["rejected_invalid_samples"] += 1
            continue
        if set(sample["runs"]) != set(REQUIRED_RUNS):
            stats["incomplete_samples"] += 1
            continue

        runs = [sample["runs"][run_index] for run_index in REQUIRED_RUNS]
        if runs[0]["is_swapped"] or not runs[1]["is_swapped"]:
            stats["invalid_order_samples"] += 1
            continue
        if len({run["seed"] for run in runs}) != len(REQUIRED_RUNS):
            stats["duplicate_seed_samples"] += 1
            continue

        if check_images:
            missing = [
                key
                for key in IMAGE_FIELDS
                if not Path(sample["metadata"][key]).is_file()
            ]
            if missing:
                stats["missing_image_samples"] += 1
                continue

        output_sample = dict(sample["metadata"])
        if sample["edit_type"] is not None:
            output_sample["edit_type"] = sample["edit_type"]
        output_sample["runs"] = [
            {
                "run_index": run["run_index"],
                "seed": run["seed"],
                "metric_groundtruth": canonical_metric_groundtruth(
                    run["raw_output"], run["is_swapped"]
                ),
            }
            for run in runs
        ]
        complete_samples.append(output_sample)

    stats["unique_sample_ids"] = len(grouped)
    stats["complete_samples"] = len(complete_samples)
    return complete_samples, stats


def select_samples(
    samples: list[dict[str, Any]],
    num_samples: int | None,
    selection: str,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples is None:
        return samples
    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")
    if num_samples > len(samples):
        raise ValueError(
            f"Only {len(samples)} complete samples are available; "
            f"requested {num_samples}."
        )
    if selection == "random":
        return random.Random(seed).sample(samples, num_samples)
    return samples[:num_samples]


def write_jsonl(samples: Iterable[Mapping[str, Any]], output_file: Path) -> int:
    """Write directly to the confirmed destination; no intermediate file."""
    if not output_file.parent.is_dir():
        raise FileNotFoundError(
            "Output directory does not exist (not creating it automatically): "
            f"{output_file.parent}"
        )
    count = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def print_summary(
    stats: Counter[str],
    selected_samples: list[dict[str, Any]],
    output_file: Path,
    *,
    validate_only: bool,
) -> None:
    print("GemmaGT preparation summary")
    print(f"  Non-empty input lines: {stats['nonempty_lines']}")
    print(f"  Unique sample IDs:     {stats['unique_sample_ids']}")
    print(f"  Complete valid samples:{stats['complete_samples']:>7}")
    print(f"  Selected samples:      {len(selected_samples):>7}")
    for key in (
        "invalid_run_lines",
        "metadata_conflicts",
        "conflicting_duplicate_runs",
        "identical_duplicate_runs",
        "rejected_invalid_samples",
        "incomplete_samples",
        "invalid_order_samples",
        "duplicate_seed_samples",
        "missing_image_samples",
    ):
        if stats[key]:
            print(f"  {key}: {stats[key]}")
    if validate_only:
        print("  Output: not written (--validate-only)")
    else:
        print(f"  Output: {output_file}")


def main() -> None:
    args = parse_args()
    samples, stats = load_complete_samples(
        resolve_input_files(args.input_file, args.input_dir),
        check_images=not args.skip_image_check,
        image_root=args.image_root,
    )
    selected_samples = select_samples(
        samples,
        args.num_samples,
        args.selection,
        args.seed,
    )
    if not args.validate_only:
        write_jsonl(selected_samples, args.output_file)
    print_summary(
        stats,
        selected_samples,
        args.output_file,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
