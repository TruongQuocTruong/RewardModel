"""How often the Teacher label changes when the images are degraded (per metric).

Requested 2026-11-05 to choose the degradation levels of the combined Student
augmentation from evidence: blur / downscale / noise / JPEG may hide the evidence of
realism and aesthetic (AI artifacts, texture) more than that of the consistency metrics.

For --sample-count samples of the Teacher-Inspector agreement subset (seeded random
choice), the Teacher (eval mode, dropout off) scores the original pair and the pair
after each degradation at each level, applied with the same parameters to the source,
A and B images (student_augmentation.degrade_image; no layout change, no text change).
A degradation excluded for the instruction (degradation_eligibility) is skipped.

Outputs (--output-dir):
  degradation_scored.jsonl   one row per (sample, variant): overall and per-metric
                             three-way probabilities and labels, relevance (appended and
                             flushed, so a stopped job resumes)
  degradation_summary.json   per (degradation, level): label change rate vs the original,
                             overall and per metric (all samples / text-edit samples /
                             other samples), mean KL(original || degraded)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent  # Qwen3-VL-Embedding
_EVAL_DIR = _PROJECT_ROOT.parents[1] / "RewardModelEvaluation" / "OurModel"
for path in (_PROJECT_ROOT, _SCRIPTS_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_reward_model import load_refined_model  # noqa: E402
from train_loop import (  # noqa: E402
    DEFAULT_EDITREWARD_ROOT,
    PROBABILITY_EPS,
    THREE_WAY_CLASSES,
    forward_pairs,
    index_unlabeled_pool,
    materialize_unlabeled_round,
    overall_three_way_probs,
)
from src.models.custom_qwen3_vl_reranker import METRIC_NAMES  # noqa: E402
import student_augmentation as aug  # noqa: E402

# Level 1 ~ mild range, levels 2-3 ~ the chosen normal range, level 4 ~ the strong
# levels proposed first (2026-11-05).
LEVELS = {
    "jpeg": [{"quality": 40}, {"quality": 25}, {"quality": 15}, {"quality": 8}],
    "blur": [{"radius": 1.25}, {"radius": 2.0}, {"radius": 3.0}, {"radius": 5.0}],
    "downscale": [{"scale": 0.6}, {"scale": 0.45}, {"scale": 0.3}, {"scale": 0.2}],
    "noise": [{"sigma": 9.0}, {"sigma": 15.0}, {"sigma": 25.0}, {"sigma": 40.0}],
}
SCORED_FILE = "degradation_scored.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True, help="student_warmstart.pt (prompt contract)")
    parser.add_argument("--scenarios", required=True, help="augmentation_combined_scenarios_<N>.json")
    parser.add_argument("--unlabeled-dataset", required=True)
    parser.add_argument("--editreward-root", default=str(DEFAULT_EDITREWARD_ROOT))
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def kl(p: list, q: list) -> float:
    return float(sum(a * (torch.log(torch.tensor(max(a, PROBABILITY_EPS))) -
                          torch.log(torch.tensor(max(b, PROBABILITY_EPS)))) for a, b in zip(p, q)))


def summarize(rows: list) -> dict:
    by_sample = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["variant"]] = row
    groups = {"all": lambda r: True, "text_edit": lambda r: r["mild"], "other": lambda r: not r["mild"]}
    summary = {}
    for name, levels in LEVELS.items():
        for level in range(1, len(levels) + 1):
            variant = f"{name}_level{level}"
            entry = {"parameters": levels[level - 1]}
            for group, keep in groups.items():
                pairs = [(v["original"], v[variant]) for v in by_sample.values()
                         if "original" in v and variant in v and keep(v["original"])]
                if not pairs:
                    continue
                result = {"samples": len(pairs),
                          "overall_label_change": round(sum(o["overall_label"] != d["overall_label"] for o, d in pairs) / len(pairs), 4),
                          "overall_kl": round(sum(kl(o["overall"], d["overall"]) for o, d in pairs) / len(pairs), 5)}
                for index, metric in enumerate(METRIC_NAMES):
                    result[f"{metric}_label_change"] = round(
                        sum(o["metric_labels"][index] != d["metric_labels"][index] for o, d in pairs) / len(pairs), 4)
                    result[f"{metric}_kl"] = round(
                        sum(kl(o["metric_probs"][index], d["metric_probs"][index]) for o, d in pairs) / len(pairs), 5)
                result["selected_metrics_changed"] = round(
                    sum(o["selected"] != d["selected"] for o, d in pairs) / len(pairs), 4)
                entry[group] = result
            summary[variant] = entry
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "degraded_tmp"
    work_dir.mkdir(exist_ok=True)
    scored_path = output_dir / SCORED_FILE

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))["samples"]
    root = Path(args.editreward_root).expanduser().resolve()
    pool = index_unlabeled_pool(Path(args.unlabeled_dataset), root)
    position = {sample_id: index for index, sample_id in enumerate(pool.sample_ids)}
    chosen = sorted(random.Random(args.seed).sample(sorted(scenarios), args.sample_count))
    variants = [("original", None, None)] + [
        (f"{name}_level{level + 1}", name, params) for name, levels in LEVELS.items()
        for level, params in enumerate(levels)
    ]
    done = set()
    rows = []
    if scored_path.exists():
        rows = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        done = {(row["sample_id"], row["variant"]) for row in rows}
        print(f"Resuming after {len(rows)} rows", flush=True)

    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    prompt = checkpoint.get("prompt")
    if not isinstance(prompt, dict):
        student = torch.load(args.student_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        prompt = student.get("prompt")
        del student
    if not isinstance(prompt, dict):
        raise ValueError("No prompt contract in the Teacher or Student-Stage1 checkpoint.")
    model, processor = load_refined_model(
        checkpoint, "teacher", device, None, None,
        base_model_override=Path(args.base_model_path).expanduser().resolve(),
    )
    del checkpoint

    from PIL import Image

    with scored_path.open("a", encoding="utf-8") as handle:
        for count, sample_id in enumerate(chosen, 1):
            scenario = scenarios[sample_id]
            record = materialize_unlabeled_round(pool, [position[sample_id]])[position[sample_id]]
            for variant, name, params in variants:
                if (sample_id, variant) in done:
                    continue
                if name is not None and name not in scenario["degradation_eligible"]:
                    continue
                created = []
                student_record = record
                if name is not None:
                    paths = []
                    for tag, original in (("source", record.source_image), ("a", record.image_a), ("b", record.image_b)):
                        with Image.open(original) as opened:
                            image = aug.degrade_image(opened, [{"name": name, **params}], f"{sample_id}_{variant}")
                        path = work_dir / f"{count}_{variant}_{tag}.png"
                        image.save(path)
                        paths.append(str(path))
                    created = paths
                    student_record = replace(record, source_image=paths[0], image_a=paths[1], image_b=paths[2])
                with torch.no_grad():
                    output = forward_pairs(model, processor, [student_record], prompt, device)
                for path in created:
                    Path(path).unlink(missing_ok=True)
                overall = overall_three_way_probs(output, None, "selected_equal")[0].float().cpu()
                metric_probs = output.three_way_probs.float()[0].cpu()
                relevance = output.relevance_probs.float()[0].cpu()
                row = {
                    "sample_id": sample_id,
                    "variant": variant,
                    "mild": bool(scenario["degradation_mild"]),
                    "overall": [round(float(v), 6) for v in overall],
                    "overall_label": THREE_WAY_CLASSES[int(overall.argmax())],
                    "metric_probs": [[round(float(v), 6) for v in m] for m in metric_probs],
                    "metric_labels": [THREE_WAY_CLASSES[int(m.argmax())] for m in metric_probs],
                    "relevance": [round(float(v), 6) for v in relevance],
                    "selected": [int(v >= 0.5) for v in relevance],
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)
            if count % 10 == 0:
                print(f"Scored {count}/{len(chosen)} samples", flush=True)

    summary = {"sample_count": len(chosen), "metrics": list(METRIC_NAMES), "levels": LEVELS,
               "results": summarize(rows)}
    (output_dir / "degradation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for variant, entry in summary["results"].items():
        a = entry.get("all", {})
        print(variant, entry["parameters"], "overall", a.get("overall_label_change"),
              {m: a.get(f"{m}_label_change") for m in METRIC_NAMES}, flush=True)
    print(f"Written {output_dir}", flush=True)


if __name__ == "__main__":
    main()
