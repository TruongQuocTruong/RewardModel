"""Prepare the Student augmentation scenarios of the selected Unlabeled samples (CPU, text only).

Reads the selected sample IDs (score_teacher_inspector_agreement.py) and their rows of
the Unlabeled dataset, builds the instruction variants of every scenario with the
rules of student_augmentation.py, checks them, and lists the distractor images of the
spatial scenario (source images of Unlabeled samples outside the selection, one per
source, none sharing a source with a selected sample). No image is opened.

Outputs (--output-dir):
  augmentation_scenarios_<N>.json   per sample: eligible scenarios and instruction
                                    variants; the rules; the distractor list
  augmentation_scenarios_preview.md original instruction and its variants for a
                                    sample of instructions, for reading
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import student_augmentation as aug

VARIANTS_PER_SCENARIO = 3
PREVIEW_COUNT = 50
FORMAT_VERSION = 1

check_masking = aug.check_masking
check_contains = aug.check_contains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selected", required=True, help="selected_unlabeled_<N>.json")
    parser.add_argument("--unlabeled-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("single", "combined"), default="single",
                        help="single = one scenario per sample (2026-11-04); combined = combined strong "
                             "augmentation (2026-11-05), written to augmentation_combined_scenarios_<N>.json")
    args = parser.parse_args()

    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    selected_ids = list(selected["sample_ids"])
    selected_set = set(selected_ids)
    rows = {}
    other_sources = {}
    for line in Path(args.unlabeled_dataset).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["unique_key"] in selected_set:
            rows[row["unique_key"]] = row
        else:
            other_sources.setdefault(row["key"], row)
    if len(rows) != len(selected_ids):
        raise ValueError(f"{len(selected_ids) - len(rows)} selected IDs are missing from the dataset.")
    selected_sources = {row["key"] for row in rows.values()}
    distractors = [
        {"sample_id": row["unique_key"], "source_key": key, "source_img": row["source_img"],
         "instruction": row["instruction"]}
        for key, row in sorted(other_sources.items()) if key not in selected_sources
    ]

    if args.mode == "combined":
        build_combined(args, selected_ids, rows, distractors)
        return

    samples, problems = {}, []
    eligible_counts, flip_reasons = Counter(), Counter()
    for sample_id in selected_ids:
        instruction = rows[sample_id]["instruction"]
        masking = aug.masking_variants(instruction, VARIANTS_PER_SCENARIO, args.seed)
        injection = aug.injection_variants(instruction, VARIANTS_PER_SCENARIO, args.seed)
        spatial = aug.spatial_variants(instruction)
        reason = aug.flip_exclusion(instruction)
        flip = None if reason else aug.flip_instruction(instruction)
        eligible = (["masking"] if masking else []) + ["injection", "spatial"] + ([] if reason else ["flip"])
        for variant in masking:
            problems += [(sample_id, "masking", variant, p) for p in check_masking(instruction, variant)]
        for variant in injection:
            problems += [(sample_id, "injection", variant, p) for p in check_contains(instruction, variant)]
        for position, variants in spatial.items():
            for variant in variants:
                found = check_contains(instruction, variant)
                if position not in variant:
                    found.append("position missing")
                problems += [(sample_id, "spatial", variant, p) for p in found]
        if flip is not None and flip == instruction and aug.LEFT_RIGHT_PATTERN.search(instruction):
            problems.append((sample_id, "flip", flip, "left/right not swapped"))
        eligible_counts.update(eligible)
        flip_reasons[reason or "flip_allowed"] += 1
        samples[sample_id] = {
            "instruction": instruction,
            "source_key": rows[sample_id]["key"],
            "eligible": eligible,
            "masking": masking,
            "injection": injection,
            "spatial": spatial,
            "flip": flip,
            "flip_excluded_reason": reason,
        }
    if problems:
        for problem in problems[:40]:
            print("PROBLEM", problem)
        raise SystemExit(f"{len(problems)} variant problems; nothing written.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = {
        "scenarios": list(aug.SCENARIOS),
        "swap_probability": aug.SWAP_PROBABILITY,
        "choice": "per sample and iteration: one eligible scenario uniformly, plus A<->B swap with swap_probability",
        "masking_articles": list(aug.MASKABLE_ARTICLES),
        "masking_openers": list(aug.MASKABLE_OPENERS),
        "masking_image_phrases": list(aug.MASKABLE_IMAGE_PHRASES),
        "injection_prefixes": list(aug.INJECTION_PREFIXES),
        "injection_suffixes": list(aug.INJECTION_SUFFIXES),
        "spatial_templates": list(aug.SPATIAL_TEMPLATES),
        "spatial_positions": list(aug.POSITIONS),
        "spatial_size": "composite scaled to the source pixel count, aspect ratio kept",
        "flip_excluded": "text / number / quote / non-ASCII; rotation or mirror words",
    }
    result = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "selected": str(Path(args.selected).resolve()),
        "unlabeled_dataset": str(Path(args.unlabeled_dataset).resolve()),
        "rules": rules,
        "summary": {
            "samples": len(samples),
            "unique_instructions": len({s["instruction"] for s in samples.values()}),
            "eligible_counts": dict(eligible_counts),
            "flip": dict(flip_reasons),
            "distractors": len(distractors),
        },
        "distractors": distractors,
        "samples": samples,
    }
    target = output_dir / f"augmentation_scenarios_{len(samples)}.json"
    target.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    unique = {}
    for sample_id in selected_ids:
        unique.setdefault(samples[sample_id]["instruction"], sample_id)
    preview_ids = sorted(random.Random(args.seed).sample(sorted(unique.values()), min(PREVIEW_COUNT, len(unique))))
    lines = [f"# Augmentation scenarios: preview of {len(preview_ids)} instructions", "",
             f"Summary: {json.dumps(result['summary'], ensure_ascii=False)}", ""]
    for number, sample_id in enumerate(preview_ids, 1):
        s = samples[sample_id]
        lines += [f"## {number}. {s['instruction']}", "", f"- eligible: {', '.join(s['eligible'])}"]
        lines += [f"- masking: {v}" for v in s["masking"]] or ["- masking: (none)"]
        lines += [f"- injection: {v}" for v in s["injection"]]
        lines += [f"- spatial ({p}): {s['spatial'][p][i]}" for i, p in enumerate(("left", "top", "right"))]
        lines += [f"- flip: {s['flip']}" if s["flip"] else f"- flip: excluded ({s['flip_excluded_reason']})", ""]
    (output_dir / "augmentation_scenarios_preview.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False))
    print("written", target)


def build_combined(args, selected_ids, rows, distractors) -> None:
    """Combined strong augmentation: eligibility per sample, an online dry run, a preview."""
    samples = {}
    text_counts, layout_counts, degradation_counts, excluded_counts = Counter(), Counter(), Counter(), Counter()
    mild = 0
    for sample_id in selected_ids:
        instruction = rows[sample_id]["instruction"]
        reason = aug.flip_exclusion(instruction)
        degradations, is_mild, excluded = aug.degradation_eligibility(instruction)
        sample = {
            "instruction": instruction,
            "source_key": rows[sample_id]["key"],
            "text_eligible": aug.text_eligibility(instruction),
            "layout_eligible": ["none", "spatial"] + ([] if reason else ["flip"]),
            "flip_excluded_reason": reason,
            "degradation_eligible": degradations,
            "degradation_mild": is_mild,
            "degradation_excluded": excluded,
        }
        samples[sample_id] = sample
        text_counts.update(sample["text_eligible"])
        layout_counts.update(sample["layout_eligible"])
        degradation_counts.update(degradations)
        excluded_counts.update(f"{name}:{why}" for name, why in excluded.items())
        mild += int(is_mild)

    # Dry run of the online choice: 5 iterations x every sample (text only, no image).
    drawn, fallbacks = Counter(), Counter()
    for iteration in range(5):
        for sample_id in selected_ids:
            choice = aug.choose_combined_augmentation(samples[sample_id], sample_id, iteration, args.seed, distractors)
            drawn.update([f"text:{choice['text_operation']}", f"layout:{choice['layout']}",
                          f"degradations:{len(choice['degradation'])}", f"swap:{choice['swap']}"])
            drawn.update(f"degradation:{d['name']}" for d in choice["degradation"])
            if choice.get("fallback"):
                fallbacks.update(f"{choice['text_operation']}: {p}" for p in choice["fallback"])
    summary = {
        "samples": len(samples),
        "unique_instructions": len({s["instruction"] for s in samples.values()}),
        "text_eligible_counts": dict(text_counts),
        "layout_eligible_counts": dict(layout_counts),
        "degradation_eligible_counts": dict(degradation_counts),
        "degradation_excluded_counts": dict(excluded_counts),
        "degradation_mild_samples": mild,
        "distractors": len(distractors),
        "dry_run_5_iterations": dict(sorted(drawn.items())),
        "dry_run_fallbacks": dict(fallbacks),
    }
    rules = {
        "mode": "combined",
        "text_operations": list(aug.TEXT_OPERATIONS),
        "text_fraction": list(aug.TEXT_FRACTION),
        "mask_token": aug.MASK_TOKEN,
        "distractor_templates": list(aug.DISTRACTOR_TEMPLATES),
        "spatial_distractor_template": aug.SPATIAL_DISTRACTOR_TEMPLATE,
        "layouts": list(aug.LAYOUTS),
        "spatial_templates": list(aug.SPATIAL_TEMPLATES),
        "degradations": list(aug.DEGRADATIONS),
        "degradation_probability": aug.DEGRADATION_PROBABILITY,
        "degradation_levels": aug.DEGRADATION_LEVELS,
        "swap_probability": aug.SWAP_PROBABILITY,
        "choice": "per sample and iteration: one eligible text operation, one eligible layout, "
                  "1-2 degradations with degradation_probability, A<->B swap with swap_probability",
    }
    result = {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "selected": str(Path(args.selected).resolve()),
        "unlabeled_dataset": str(Path(args.unlabeled_dataset).resolve()),
        "rules": rules,
        "summary": summary,
        "distractors": distractors,
        "samples": samples,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"augmentation_combined_scenarios_{len(samples)}.json"
    target.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    unique = {}
    for sample_id in selected_ids:
        unique.setdefault(samples[sample_id]["instruction"], sample_id)
    preview_ids = sorted(random.Random(args.seed).sample(sorted(unique.values()), min(PREVIEW_COUNT, len(unique))))
    lines = [f"# Combined strong augmentation: preview of {len(preview_ids)} instructions", "",
             f"Summary: {json.dumps(summary, ensure_ascii=False)}", ""]
    for number, sample_id in enumerate(preview_ids, 1):
        s = samples[sample_id]
        rng = random.Random(f"{args.seed}-{sample_id}")
        lines += [f"## {number}. {s['instruction']}", "",
                  f"- text eligible: {', '.join(s['text_eligible'])}; layout: {', '.join(s['layout_eligible'])}; "
                  f"degradation: {', '.join(s['degradation_eligible']) or '(none)'}"
                  f"{' (mild)' if s['degradation_mild'] else ''}"]
        for name in ("drop_action", "mask_content", "typo"):
            if name in s["text_eligible"]:
                variant, problems = aug.apply_text_operation(name, s["instruction"], rng)
                lines.append(f"- {name}: {variant}" + (f"  [fallback: {problems}]" if problems else ""))
        for iteration in range(3):
            choice = aug.choose_combined_augmentation(s, sample_id, iteration, args.seed, distractors)
            degradation = ", ".join(
                f"{d['name']}(" + ", ".join(f"{k}={v}" for k, v in d.items() if k != "name") + ")"
                for d in choice["degradation"]) or "none"
            lines.append(f"- iteration {iteration + 1} [{choice['text_operation']} | {choice['layout']}"
                         f"{' ' + choice['position'] if choice.get('position') else ''} | {degradation}"
                         f" | swap {choice['swap']}]: {choice['instruction']}"
                         + (f"  [fallback: {choice['fallback']}]" if choice.get("fallback") else ""))
        lines.append("")
    (output_dir / "augmentation_combined_preview.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("written", target)


if __name__ == "__main__":
    main()
