"""Shared helpers for model adapters: bench rows, image paths, needed dimension, row selection, resume.

One job scores the whole bench for one model variant; the bench is never split across jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

COMPARISON_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = COMPARISON_ROOT / "EvaluationConfig" / "metric_mapping.json"
DEFAULT_RESULTS = COMPARISON_ROOT / "results"
MODELS_DIR = Path("/datastore/tuanld/Truongtq/RewardModels")
# Same roots as AURA/AURA-TEST/build_aura_bench.py
SOURCE_IMAGE_ROOT = Path("/datastore/truongtq/ImageEditingBenchmark/AURABenchmark/AURA-BENCHMARK")
EDITED_IMAGE_ROOT = Path("/datastore/truongtq/ImageEditingBenchmark/EditingEvaluation/OutputEdited")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON ({error})") from error
    return rows


def load_bench_rows(path: Path) -> List[dict]:
    """Bench rows in file order; raises on a duplicate sample_id or a row without two edited images."""
    rows, seen = load_jsonl(path), set()
    for row in rows:
        for field in ("sample_id", "metric", "prompt", "src_image", "edited_images"):
            if field not in row:
                raise ValueError(f"bench row {row.get('sample_id')!r} misses {field!r}")
        if len(row["edited_images"]) != 2:
            raise ValueError(f"bench row {row['sample_id']!r} must have exactly two edited images")
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate bench sample_id {row['sample_id']!r}")
        seen.add(row["sample_id"])
    if not rows:
        raise ValueError(f"{path}: bench is empty")
    return rows


def image_paths(row: dict, source_root: Path = SOURCE_IMAGE_ROOT, edited_root: Path = EDITED_IMAGE_ROOT):
    """(source path, [edited A path, edited B path]) of one bench row."""
    return Path(source_root) / row["src_image"], [Path(edited_root) / p for p in row["edited_images"]]


def dimension_by_bench_metric(mapping: dict, model: str, pretrained: str) -> Dict[str, Optional[str]]:
    """bench metric -> scoring dimension (None = not evaluated) for one row of metric_mapping.json."""
    rows = [r for r in mapping["models"] if r["model"] == model and r["pretrained"] == pretrained]
    if len(rows) != 1:
        raise ValueError(f"expected one mapping row for {model} / {pretrained}, found {len(rows)}")
    return {c["bench_metric"]: rows[0]["scores"][c["column"]] for c in mapping["columns"]}


def select_rows(rows: Sequence[dict], per_metric_limit: Optional[int] = None,
                metrics: Optional[Iterable[str]] = None) -> List[dict]:
    """Rows in file order: all of them by default; optional metric filter and, for smoke tests only,
    the first per_metric_limit rows of each metric."""
    if per_metric_limit is not None and per_metric_limit < 1:
        raise ValueError("per_metric_limit must be at least 1")
    wanted = set(metrics) if metrics else None
    kept, counts = [], {}
    for row in rows:
        if wanted is not None and row["metric"] not in wanted:
            continue
        if per_metric_limit is not None and counts.get(row["metric"], 0) >= per_metric_limit:
            continue
        counts[row["metric"]] = counts.get(row["metric"], 0) + 1
        kept.append(row)
    return kept


def already_scored(path: Path) -> set:
    """(sample_id, dimension) pairs present in an existing score file, for resuming."""
    if not Path(path).is_file():
        return set()
    return {(r["sample_id"], r["dimension"]) for r in load_jsonl(path)}


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
