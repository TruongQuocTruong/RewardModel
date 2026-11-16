"""Judge the rollout images of one bench that the table needs; resumable JSONL.

Images judged (decision 2026-11-15): seed 0 of every sample (row FLUX.2-Klein-9B) plus the seed chosen by
each selector (results/selections/<Selector>/<Bench>.jsonl, written by select_best.py), so at most
5 images per sample. Selections that do not exist yet are waited for (polling), so a job that starts
early judges seed 0 first and keeps its GPU slot. --all-seeds judges all 9 seeds instead.

Output: <output-dir>/scores.jsonl (one line per (sample_id, seed)) + run_config.json.
Each line: sample_id, bench, category, seed, judge, scores {column: value|None}, raw_outputs {call: text},
parse {call: "strict" | "retry_longer" | "guessed" | "failed"}.

Parse policy (greedy decoding is deterministic, so regenerating with the same budget repeats the answer):
  1. generate with the protocol budget and parse;
  2. if parsing fails, regenerate once with twice the budget (a truncated answer is the usual cause);
  3. GEdit only: if still unparsable, viescore's give-up rule (random 0-10 guess, seeded per item)
     like VIEScore's second loop; other benches keep None, as the original code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_protocols as bp  # noqa: E402

PROJECT_DIR = HERE.parent
TABLE_SELECTORS = ("Qwen2.5-VL-7B", "ImageReward", "CLIP", "FIRE-Reward-Stage2-Teacher")
DEFAULT_JUDGE = {"GEdit-Bench-EN": "gemma-4-31B-it", "KRIS-Bench": "gemma-4-31B-it",
                 "ImgEdit": "Qwen2.5-VL-72B-Instruct-AWQ"}


def _parse_call(bench: str, name: str, text: str):
    """Return a parsed value, or None if the answer cannot be parsed for this call."""
    if bench == "GEdit-Bench-EN":
        return bp.gedit_min_score(bp.viescore_output_to_dict(text))
    if bench == "ImgEdit":
        scores = bp.imgedit_scores(text)
        return scores if len(scores) == 3 else None
    if bench == "KRIS-Bench":
        if name == "dual":
            return bp.kris_dual_scores(text)["instruction_score"]
        key = {"consistency": "consistency_score", "quality": "quality_score", "instruction": "instruction_score"}[name]
        return bp.kris_score(text, key)
    raise ValueError(f"Unknown bench: {bench!r}")


def judge_item(item: Mapping[str, Any], prompts: Mapping[str, Any], generate: Callable[[Sequence, int], str],
               kris_anomaly_prompt: str = "original_code") -> Dict[str, Any]:
    bench = item["bench"]
    raw, parse = {}, {}
    parsed: Dict[str, Any] = {}
    for call in bp.judge_calls(item, prompts, kris_anomaly_prompt):
        name = call["name"]
        text = generate(call["content"], call["max_new_tokens"])
        value, mode = _parse_call(bench, name, text), "strict"
        if value is None:
            text = generate(call["content"], call["max_new_tokens"] * 2)
            value, mode = _parse_call(bench, name, text), "retry_longer"
        if value is None and bench == "GEdit-Bench-EN":
            rng = random.Random(f"{item['sample_id']}|{item['seed']}|{name}")
            value = bp.gedit_min_score(bp.viescore_output_to_dict(text, give_up_parsing=True, rng=rng))
            mode = "guessed"
        if value is None:
            mode = "failed"
        raw[name], parse[name], parsed[name] = text, mode, value

    if bench == "GEdit-Bench-EN":
        scores = bp.gedit_columns(parsed["SC"], parsed["PQ"])
    elif bench == "ImgEdit":
        scores = bp.imgedit_columns(bp.imgedit_scores(raw["ImgEdit"]), item["category"])
    else:
        scores = bp.kris_columns(raw)
    return {"sample_id": item["sample_id"], "bench": bench, "category": item["category"], "seed": item["seed"],
            "edited_image": item["edited_image"], "scores": scores, "raw_outputs": raw, "parse": parse}


def read_done(path: Path) -> set:
    """Keys already written; drop an interrupted last line, reject corruption elsewhere."""
    done = set()
    if not path.exists():
        return done
    lines = path.read_bytes().splitlines(keepends=True)
    good = []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
            key = (row["sample_id"], int(row["seed"]))
        except (ValueError, KeyError, TypeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Corrupt line {i + 1} in {path}")
        if key in done:
            raise ValueError(f"Duplicate result {key} in {path}")
        done.add(key)
        good.append(line if line.endswith(b"\n") else line + b"\n")
    if len(good) != len(lines) or (lines and not lines[-1].endswith(b"\n")):
        path.write_bytes(b"".join(good))
    return done


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def select_items(bench: str, rollout_root: Path, source_root: Path, max_samples: int | None):
    samples = bp.load_samples(bench, rollout_root, source_root)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        samples = samples[:max_samples]
    return bp.expand_seeds(samples)


def wanted_items(items: Sequence[Mapping[str, Any]], selections_root: Path, bench: str,
                 selectors: Sequence[str]):
    """Seed-0 items plus the seeds chosen by the selectors whose selection file exists.

    Returns (items, missing selectors). A selection naming an unknown sample or seed raises ValueError.
    """
    by_key = {(it["sample_id"], it["seed"]): it for it in items}
    wanted = {(sid, 0) for sid, _ in by_key}
    missing = []
    chosen_by = defaultdict(list)
    for selector in selectors:
        path = Path(selections_root) / selector / f"{bench}.jsonl"
        if not path.is_file():
            missing.append(selector)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["sample_id"], int(row["seed"]))
            if key not in by_key:
                raise ValueError(f"{path}: unknown sample/seed {key}")
            wanted.add(key)
            chosen_by[key].append(selector)
    out = [dict(by_key[k], chosen_by=chosen_by.get(k, [])) for k in sorted(wanted, key=lambda k: (k[0], k[1]))]
    return out, missing


def check_files(items: Sequence[Mapping[str, Any]]) -> None:
    missing = sorted({p for it in items for p in (it["source_image"], it["edited_image"], it.get("gt_image"))
                      if p and not Path(p).is_file()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images, first: {missing[:5]}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True, choices=bp.BENCHES)
    p.add_argument("--judge", choices=("gemma-4-31B-it", "Qwen2.5-VL-72B-Instruct-AWQ"))
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rollout-root", default=str(bp.ROLLOUT_ROOT))
    p.add_argument("--source-root", default=str(bp.SOURCE_ROOT))
    p.add_argument("--max-samples", type=int, default=None, help="Smoke test only (samples, each with 9 seeds).")
    p.add_argument("--kris-anomaly-prompt", default="original_code", choices=("original_code", "intended"))
    p.add_argument("--check-only", action="store_true", help="Build items and check image files; no model.")
    p.add_argument("--all-seeds", action="store_true", help="Judge all 9 seeds (not the table decision).")
    p.add_argument("--selections-root", default=str(PROJECT_DIR / "results" / "selections"))
    p.add_argument("--selectors", nargs="*", default=list(TABLE_SELECTORS))
    p.add_argument("--wait-hours", type=float, default=48.0, help="Max time to wait for missing selections.")
    p.add_argument("--poll-minutes", type=float, default=10.0)
    args = p.parse_args(argv)
    args.judge = args.judge or DEFAULT_JUDGE[args.bench]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    all_items = select_items(args.bench, Path(args.rollout_root), Path(args.source_root), args.max_samples)
    check_files(all_items)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"
    prompts = bp.load_prompts()
    config = {
        "started": datetime.now(timezone.utc).isoformat(), "bench": args.bench, "judge": args.judge,
        "model_path": args.model_path, "rollout_root": args.rollout_root, "source_root": args.source_root,
        "seeds": list(bp.SEEDS), "candidate_items": len(all_items), "max_samples": args.max_samples,
        "image_set": "all_seeds" if args.all_seeds else {"seed_0": True, "selectors": args.selectors},
        "kris_anomaly_prompt": args.kris_anomaly_prompt,
        "code_md5": {f.name: file_md5(f) for f in (HERE / "bench_protocols.py", HERE / "judges.py", Path(__file__))},
        "prompt_md5": hashlib.md5(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        "repo_files_md5": {str(p.relative_to(bp.PROJECT_DIR)): file_md5(p) for p in (
            bp.GEDIT_REPO / "viescore" / "vie_prompts.py", bp.IMGEDIT_REPO / "prompts.json",
            bp.KRIS_REPO / "utils" / "prompts.py")},
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    judge = None
    deadline = time.time() + args.wait_hours * 3600
    while True:
        if args.all_seeds:
            items, missing = [dict(it, chosen_by=[]) for it in all_items], []
        else:
            items, missing = wanted_items(all_items, Path(args.selections_root), args.bench, args.selectors)
        done = read_done(scores_path)
        todo = [it for it in items if (it["sample_id"], it["seed"]) not in done]
        print(json.dumps({"bench": args.bench, "judge": args.judge, "wanted": len(items), "done": len(done),
                          "todo": len(todo), "missing_selections": missing}), flush=True)
        if args.check_only:
            return 0
        if todo and judge is None:
            from judges import load_judge
            judge = load_judge(args.judge, args.model_path)
        with scores_path.open("a", encoding="utf-8") as stream:
            for n, item in enumerate(todo, 1):
                start = time.perf_counter()
                row = judge_item(item, prompts, judge.generate, args.kris_anomaly_prompt)
                row["judge"] = args.judge
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[{n}/{len(todo)}] {item['sample_id']} seed={item['seed']} {row['scores']} "
                      f"{row['parse']} {time.perf_counter() - start:.1f}s", flush=True)
                if n == 1 or n % 100 == 0:
                    import torch
                    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if not missing:
            print("all wanted images judged", flush=True)
            break
        if time.time() > deadline:
            print(f"stopped waiting; selections still missing: {missing} (resubmit to continue)", flush=True)
            break
        time.sleep(args.poll_minutes * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
