#!/usr/bin/env python
"""Build the AURA-Test comparison table from per-model score files.

Score file (written by each model adapter): results/<result folder>/scores.jsonl, one line per
(bench pair, scoring dimension):
    {"sample_id": "...", "dimension": "Visual Quality", "score_a": 7.1, "score_b": 5.0,
     "probabilities": [p_a, p_b, p_tie]}          # probabilities only for models with a Tie Head
score_a / score_b are the scores of edited_images[0] / edited_images[1] of that bench row.

For every model row and table column of EvaluationConfig/metric_mapping.json:
    dimension null        -> "-"
    no score file yet     -> "pending"
    otherwise             -> evaluation_metrics.summarize_column on the pairs of that column's bench metric

Usage:
    python scripts/comparison_table.py --output results/comparison_table [--resamples 2000] [--allow-partial]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import evaluation_metrics as em

COMPARISON_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = COMPARISON_ROOT / "EvaluationConfig" / "metric_mapping.json"
DEFAULT_RESULTS = COMPARISON_ROOT / "results"
SCORE_FILE_NAME = "scores.jsonl"
BENCH_FIELDS = ("sample_id", "metric", "human_label", "votes", "observed_agreement")
REFERENCE_FAMILY = "OurModel"
PAIRED_STATISTICS = ("accuracy", "margin_auc", "three_class_accuracy")


def result_folder_name(model: str, pretrained: str) -> str:
    """'EditScore' + 'Qwen3-VL-4B' -> 'EditScore_Qwen3-VL-4B'; base VLMs ('-') -> model name only."""
    name = model.replace(" ", "")
    return name if pretrained in ("-", "", None) else f"{name}_{pretrained}"


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error})") from error
    return rows


def load_bench(path: Path) -> Dict[str, dict]:
    """sample_id -> bench row; raises on a missing field or a duplicate sample_id."""
    bench: Dict[str, dict] = {}
    for row in load_jsonl(path):
        missing = [field for field in BENCH_FIELDS if field not in row]
        if missing:
            raise ValueError(f"bench row {row.get('sample_id')!r} misses {missing}")
        if row["sample_id"] in bench:
            raise ValueError(f"duplicate bench sample_id {row['sample_id']!r}")
        bench[row["sample_id"]] = row
    if not bench:
        raise ValueError(f"{path}: bench is empty")
    return bench


def load_scores(path: Path) -> Dict[str, Dict[str, dict]]:
    """dimension -> sample_id -> score record; raises on duplicates or missing fields."""
    scores: Dict[str, Dict[str, dict]] = {}
    for record in load_jsonl(path):
        for field in ("sample_id", "dimension", "score_a", "score_b"):
            if field not in record:
                raise ValueError(f"{path}: score record {record.get('sample_id')!r} misses {field!r}")
        by_sample = scores.setdefault(record["dimension"], {})
        if record["sample_id"] in by_sample:
            raise ValueError(f"{path}: duplicate score for {record['sample_id']!r} / {record['dimension']!r}")
        by_sample[record["sample_id"]] = record
    return scores


def join_pairs(bench: Dict[str, dict], dimension_scores: Dict[str, dict], bench_metric: str,
               allow_partial: bool = False) -> tuple:
    """PairScore list for every bench row of bench_metric, plus coverage information.

    A score for a sample_id that is not in the bench raises; missing rows raise unless allow_partial.
    """
    unknown = sorted(set(dimension_scores) - set(bench))
    if unknown:
        raise ValueError(f"scores for unknown sample_id(s): {unknown[:5]}")
    expected = [sid for sid, row in bench.items() if row["metric"] == bench_metric]
    missing = [sid for sid in expected if sid not in dimension_scores]
    if missing and not allow_partial:
        raise ValueError(f"{bench_metric}: {len(missing)} of {len(expected)} pairs have no score, e.g. {missing[:3]}")
    pairs = []
    for sid in expected:
        if sid not in dimension_scores:
            continue
        row, record = bench[sid], dimension_scores[sid]
        pairs.append(em.PairScore(
            sample_id=sid, metric=bench_metric, human_label=row["human_label"], votes=row["votes"],
            observed_agreement=row["observed_agreement"], score_a=record["score_a"], score_b=record["score_b"],
            probabilities=record.get("probabilities"),
        ))
    return pairs, {"expected": len(expected), "scored": len(pairs), "missing": len(missing)}


def statistic_function(name: str, pairs: Sequence[em.PairScore]):
    """Index-based statistic over a fixed pair list, used by the paired bootstrap."""
    import numpy as np

    labels = np.asarray([p.human_label for p in pairs])
    margins = np.asarray([p.margin for p in pairs])
    if name == "accuracy":
        return lambda idx: em.accuracy(labels[idx], margins[idx])
    if name == "margin_auc":
        return lambda idx: em.margin_auc(labels[idx], margins[idx])
    if name == "three_class_accuracy":
        if all(p.probabilities is not None for p in pairs):
            predicted = np.asarray(em.predicted_labels_from_probabilities([p.probabilities for p in pairs]))
        else:
            predicted = np.asarray(em.predicted_labels_from_margins(margins))
        return lambda idx: float((predicted[idx] == labels[idx]).mean())
    raise ValueError(f"unknown statistic {name!r}")


def paired_comparison(reference: Sequence[em.PairScore], other: Sequence[em.PairScore], resamples: int,
                      seed: int) -> Dict[str, dict]:
    """Paired bootstrap of reference minus other on the pairs both models scored."""
    other_by_id = {p.sample_id: p for p in other}
    common_reference = [p for p in reference if p.sample_id in other_by_id]
    common_other = [other_by_id[p.sample_id] for p in common_reference]
    if not common_reference:
        raise ValueError("no common pairs for the paired comparison")
    return {
        name: em.paired_bootstrap_difference(statistic_function(name, common_reference),
                                             statistic_function(name, common_other),
                                             len(common_reference), resamples, seed)
        for name in PAIRED_STATISTICS
    }


def reference_row(mapping: dict, rows_out: Sequence[dict]) -> Optional[dict]:
    """Row every other model is compared with: the mapping row marked "paired_reference" (Student-Stage2),
    else the first row of the OurModel family. More than one marked row raises."""
    marked = [i for i, row in enumerate(mapping["models"]) if row.get("paired_reference")]
    if len(marked) > 1:
        raise ValueError("more than one mapping row has paired_reference")
    if marked:
        return rows_out[marked[0]]
    return next((r for r in rows_out if r["family"] == REFERENCE_FAMILY), None)


def build_table(mapping: dict, bench: Dict[str, dict], results_dir: Path, resamples: int = 2000, seed: int = 0,
                allow_partial: bool = False) -> dict:
    columns = mapping["columns"]
    rows_out, joined = [], {}
    for row in mapping["models"]:
        folder = result_folder_name(row["model"], row["pretrained"])
        score_path = Path(results_dir) / folder / SCORE_FILE_NAME
        scores = load_scores(score_path) if score_path.is_file() else None
        cells = {}
        for column in columns:
            name, dimension = column["column"], row["scores"][column["column"]]
            if dimension is None:
                cells[name] = {"status": "not_applicable"}
            elif scores is None:
                cells[name] = {"status": "pending", "expected_file": str(score_path)}
            elif dimension not in scores:
                cells[name] = {"status": "pending", "reason": f"no scores for dimension {dimension!r}"}
            else:
                pairs, coverage = join_pairs(bench, scores[dimension], column["bench_metric"], allow_partial)
                if not pairs:
                    cells[name] = {"status": "pending", "coverage": coverage}
                    continue
                cells[name] = {"status": "done", "dimension": dimension, "coverage": coverage,
                               **em.summarize_column(pairs, column["bench_metric"], resamples, seed)}
                joined[(folder, name)] = pairs
        rows_out.append({"model": row["model"], "pretrained": row["pretrained"], "family": row["family"],
                         "result_folder": folder, "cells": cells})

    reference = reference_row(mapping, rows_out)
    if reference is not None:
        for row in rows_out:
            if row is reference:
                continue
            for name, cell in row["cells"].items():
                key_ref, key_other = (reference["result_folder"], name), (row["result_folder"], name)
                if cell["status"] == "done" and key_ref in joined:
                    cell["paired_with_our_model"] = paired_comparison(joined[key_ref], joined[key_other],
                                                                      resamples, seed)
    return {"columns": [c["column"] for c in columns], "resamples": resamples, "seed": seed, "rows": rows_out}


def format_percent(value: Optional[float]) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.2f}"


def format_correlation(value: Optional[float]) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def cell_text(cell: dict, fields: Iterable[str], formatter) -> str:
    if cell["status"] == "not_applicable":
        return "-"
    if cell["status"] != "done":
        return "pending"
    return " \\| ".join(formatter(cell.get(field)) for field in fields)


# One cell with every statistic of a column, main table first, then the secondary statistics.
COMBINED_FIELDS = (
    ("accuracy", "Acc", "format_percent"),
    ("margin_auc", "Margin AUC", "format_percent"),
    ("three_class_accuracy", "3-class Acc", "format_percent"),
    ("full_agreement_accuracy", "Full-agreement Acc", "format_percent"),
    ("tie_auc_from_margin", "Tie AUC", "format_percent"),
    ("plcc", "PLCC", "format_correlation"),
    ("srcc", "SRCC", "format_correlation"),
)


def combined_cell_text(cell: dict) -> str:
    """'-' / 'pending', or every statistic of COMBINED_FIELDS joined by ' | ' with its own number format."""
    if cell["status"] == "not_applicable":
        return "-"
    if cell["status"] != "done":
        return "pending"
    formatters = {"format_percent": format_percent, "format_correlation": format_correlation}
    return " \\| ".join(formatters[kind](cell.get(field)) for field, _, kind in COMBINED_FIELDS)


def render_combined_markdown(table: dict) -> list:
    names = " \\| ".join(label for _, label, _ in COMBINED_FIELDS)
    lines = [f"### Accuracy and all secondary statistics per cell: {names}",
             "",
             "Acc, Margin AUC, Full-agreement Acc: non-tie pairs; 3-class Acc, Tie AUC, PLCC, SRCC: all pairs. "
             "Percent except PLCC / SRCC. Score-only models predict Tie only when the two scores are equal, "
             "so equal scores on a non-tie pair count as wrong in Acc.",
             "",
             "| Model | Pretrained Model | " + " | ".join(table["columns"]) + " |",
             "|" + "---|" * (2 + len(table["columns"]))]
    for row in table["rows"]:
        cells = [combined_cell_text(row["cells"][c]) for c in table["columns"]]
        lines.append(f"| {row['model']} | {row['pretrained']} | " + " | ".join(cells) + " |")
    return lines + [""]


def render_markdown(table: dict) -> str:
    sections = [
        ("Accuracy \\| Margin AUC (%, non-tie pairs)", ("accuracy", "margin_auc"), format_percent),
        ("Three-class Accuracy (%, all pairs; models with a Tie Head: argmax P(A), P(B), P(Tie); others: Tie only if "
         "scores are equal)",
         ("three_class_accuracy",), format_percent),
        ("PLCC \\| SRCC with the human vote margin (all pairs)", ("plcc", "srcc"), format_correlation),
        ("Tie AUC from -|margin| (%, all pairs)", ("tie_auc_from_margin",), format_percent),
        ("Tie Head AUC from P(Tie) (%, all pairs; only models with a Tie Head, '-' for the others)",
         ("tie_auc_from_tie_probability",), format_percent),
        ("Accuracy on full-agreement non-tie pairs (%)", ("full_agreement_accuracy",), format_percent),
    ]
    lines = render_combined_markdown(table)
    header = "| Model | Pretrained Model | " + " | ".join(table["columns"]) + " |"
    divider = "|" + "---|" * (2 + len(table["columns"]))
    for title, fields, formatter in sections:
        lines += [f"### {title}", "", header, divider]
        for row in table["rows"]:
            cells = [cell_text(row["cells"][c], fields, formatter) for c in table["columns"]]
            lines.append(f"| {row['model']} | {row['pretrained']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--bench", type=Path, default=None, help="default: 'bench' field of the mapping")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, required=True, help="output path without extension (.json and .md)")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true", help="accept score files that miss bench pairs")
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    bench = load_bench(args.bench or Path(mapping["bench"]))
    table = build_table(mapping, bench, args.results_dir, args.resamples, args.seed, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(table) + "\n", encoding="utf-8")
    done = sum(c["status"] == "done" for r in table["rows"] for c in r["cells"].values())
    print(f"{done} cells computed; wrote {args.output.with_suffix('.json')} and {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
