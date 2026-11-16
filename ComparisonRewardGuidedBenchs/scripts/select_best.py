"""Best-of-N choice per sample from selector scores; writes results/selections/<Selector>/<Bench>.jsonl.

Choice = argmax of the per-image score over the 9 seeds; ties -> smallest seed (fixed rule).
FLUX.2-Klein-9B (no selection) = seed 0 for every sample.
FIRE-Reward-Stage2-Teacher per-image score rule (--fire-score-rule):
  aesthetic_fidelity  equal mean of the aesthetic and fidelity metric scores (Table 4 rule for benches
                      outside the training distribution)
  selected_equal      mean of the metrics the Relevance Head selects (the stored "score")
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_protocols as bp  # noqa: E402

PROJECT_DIR = HERE.parent
METRIC_NAMES = ("aesthetic", "fidelity", "realism", "background_consistency", "foreground_consistency",
                "structure_consistency")  # = ComparisonRewardModelsOnOtherBenchs/scripts/compute_bench_metrics.py
FIRE = "FIRE-Reward-Stage2-Teacher"
FIRE_RULES = ("aesthetic_fidelity", "selected_equal")
BASE_ROW = "FLUX.2-Klein-9B"


def image_score(row: Mapping[str, Any], selector: str, fire_rule: str) -> float:
    if selector == FIRE and fire_rule == "aesthetic_fidelity":
        m = row["metric_scores"]
        value = (m[METRIC_NAMES.index("aesthetic")] + m[METRIC_NAMES.index("fidelity")]) / 2
    elif selector == FIRE and fire_rule != "selected_equal":
        raise ValueError(f"Unknown FIRE score rule: {fire_rule!r}")
    else:
        value = row["score"]
    value = float(value)
    if math.isnan(value):
        raise ValueError(f"NaN score for {row.get('sample_id')} seed {row.get('seed')}")
    return value


def choose(rows: Iterable[Mapping[str, Any]], selector: str, fire_rule: str = "aesthetic_fidelity",
           expected_seeds=bp.SEEDS) -> List[Dict[str, Any]]:
    by_sample: Dict[str, Dict[int, float]] = {}
    for row in rows:
        seeds = by_sample.setdefault(row["sample_id"], {})
        if int(row["seed"]) in seeds:
            raise ValueError(f"Duplicate score {row['sample_id']} seed {row['seed']}")
        seeds[int(row["seed"])] = image_score(row, selector, fire_rule)
    out = []
    for sample_id, scores in by_sample.items():
        if sorted(scores) != sorted(expected_seeds):
            raise ValueError(f"{sample_id}: seeds {sorted(scores)} != {sorted(expected_seeds)}")
        best = max(scores.values())
        seed = min(s for s, v in scores.items() if v == best)
        out.append({"sample_id": sample_id, "seed": seed, "score": best, "selector": selector})
    return out


def base_selection(samples: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{"sample_id": s["sample_id"], "seed": 0, "score": None, "selector": BASE_ROW} for s in samples]


def selection_path(root: Path, selector: str, bench: str) -> Path:
    return Path(root) / selector / f"{bench}.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selector", required=True,
                   choices=(BASE_ROW, "Qwen2.5-VL-7B", "ImageReward", "CLIP", FIRE, "EditScore", "EditReward"))
    p.add_argument("--run-name", default=None, help="Scores / selections folder name (default: the selector name).")
    p.add_argument("--benches", nargs="+", default=list(bp.BENCHES), choices=bp.BENCHES)
    p.add_argument("--scores-root", default=str(PROJECT_DIR / "SelectorScores"))
    p.add_argument("--output-root", default=str(PROJECT_DIR / "results" / "selections"))
    p.add_argument("--fire-score-rule", default="aesthetic_fidelity", choices=FIRE_RULES)
    args = p.parse_args(argv)
    run_name = args.run_name or args.selector
    if args.selector == BASE_ROW and run_name != BASE_ROW:
        raise ValueError("FLUX.2-Klein-9B has no run name")
    for bench in args.benches:
        samples = bp.load_samples(bench)
        if args.selector == BASE_ROW:
            rows = base_selection(samples)
        else:
            rows = choose(read_jsonl(Path(args.scores_root) / run_name / bench / "scores.jsonl"),
                          args.selector, args.fire_score_rule)
            for r in rows:
                r["selector"] = run_name
            if len(rows) != len(samples):
                raise ValueError(f"{bench}: {len(rows)} selections for {len(samples)} samples")
        write_jsonl(selection_path(Path(args.output_root), run_name, bench), rows)
        print(f"{run_name} {bench}: {len(rows)} selections", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
