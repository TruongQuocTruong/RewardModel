"""Score all 9 rollouts of every sample of the given benches with one selector (resumable per sample).

Output: SelectorScores/<run name>/<Bench>/scores.jsonl, one line per (sample_id, seed):
  {"sample_id", "bench", "category", "seed", "selector", "score", ...selector extras}
then select_best.py writes results/selections/<run name>/<Bench>.jsonl.
<run name> = --run-name, default the selector name (a new checkpoint of the same selector needs a new run name,
otherwise the resume rule skips every sample already scored by the old checkpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_protocols as bp  # noqa: E402
from judge_bench import check_files, read_done  # noqa: E402
from best_of_n_selectors import DEFAULT_PATHS, SELECTORS, load_selector  # noqa: E402

PROJECT_DIR = HERE.parent


def score_bench(selector, bench: str, samples, out_path: Path, run_name: str | None = None) -> int:
    done = read_done(out_path)
    written = 0
    with out_path.open("a", encoding="utf-8") as stream:
        for n, sample in enumerate(samples, 1):
            items = bp.expand_seeds([sample])
            if all((sample["sample_id"], it["seed"]) in done for it in items):
                continue
            start = time.perf_counter()
            rows = selector.score_sample(sample, items)
            if sorted(r["seed"] for r in rows) != sorted(it["seed"] for it in items):
                raise ValueError(f"Selector returned wrong seeds for {sample['sample_id']}")
            for row in rows:
                row.update({"sample_id": sample["sample_id"], "bench": bench, "category": sample["category"],
                            "selector": run_name or selector.name})
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            written += len(rows)
            print(f"[{bench} {n}/{len(samples)}] {sample['sample_id']} {time.perf_counter() - start:.1f}s", flush=True)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector", required=True, choices=SELECTORS)
    p.add_argument("--model-path", default=None)
    p.add_argument("--benches", nargs="+", default=list(bp.BENCHES), choices=bp.BENCHES)
    p.add_argument("--output-root", default=str(PROJECT_DIR / "SelectorScores"))
    p.add_argument("--max-samples", type=int, default=None, help="Smoke test only.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-name", default=None, help="Output folder name (default: the selector name).")
    p.add_argument("--max-source-pixels", type=int, default=None,
                   help="EditScore only: scale the source image down to at most this many pixels before scoring.")
    args = p.parse_args(argv)
    run_name = args.run_name or args.selector
    if args.max_source_pixels is not None and args.selector != "EditScore":
        raise ValueError("--max-source-pixels is only used by EditScore")

    plan = {}
    for bench in args.benches:
        samples = bp.load_samples(bench)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        check_files(bp.expand_seeds(samples))
        plan[bench] = samples
    selector = load_selector(args.selector, args.model_path, args.device,
                             work_dir=str(Path(args.output_root) / run_name), max_source_pixels=args.max_source_pixels)
    for bench, samples in plan.items():
        out_dir = Path(args.output_root) / run_name / bench
        out_dir.mkdir(parents=True, exist_ok=True)
        config = {"started": datetime.now(timezone.utc).isoformat(), "selector": args.selector, "run_name": run_name,
                  "max_source_pixels": args.max_source_pixels, "model_path": args.model_path or DEFAULT_PATHS[args.selector], "bench": bench,
                  "samples": len(samples), "seeds": list(bp.SEEDS), "max_samples": args.max_samples,
                  "code_md5": {f.name: hashlib.md5(f.read_bytes()).hexdigest()
                               for f in (HERE / "best_of_n_selectors.py", Path(__file__), HERE / "bench_protocols.py")},
                  "description": getattr(selector, "description", None)}
        (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"{bench}: wrote {score_bench(selector, bench, samples, out_dir / 'scores.jsonl', run_name)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
