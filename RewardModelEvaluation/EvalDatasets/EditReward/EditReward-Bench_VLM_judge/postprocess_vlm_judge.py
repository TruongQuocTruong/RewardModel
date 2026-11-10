"""Relabel the EditReward-Bench 2pair benchmark with the VLM-as-a-judge majority vote.

Inputs:
  pairs.csv                     one row per pair: pair_id = "<group_id>/<XvY>", original_overall, majority_overall
  EditReward-Bench_2Pair.json   the 133-pair 2pair benchmark (18 original ties)

Label convention in pairs.csv: 0 = candidate_1 preferred, 1 = candidate_2 preferred, 2 = tie.
Output ranking follows the evaluator: A = candidate_1, B = candidate_2 ("A>B", "B>A", "A=B").

The output keeps the same 115 pairs that are evaluated on the original benchmark (original ties removed),
with the same fields as EditReward-Bench_2Pair.json and metadata_ranking replaced by the VLM majority label.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

LABEL_TO_RANKING = {0: "A>B", 1: "B>A", 2: "A=B"}


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", type=Path, default=here / "pairs.csv")
    parser.add_argument("--benchmark", type=Path, default=here.parent / "EditReward-Bench_2Pair.json")
    parser.add_argument("--output", type=Path, default=here / "EditReward-Bench_2Pair_VLM_judge.json")
    parser.add_argument("--report", type=Path, default=here / "EditReward-Bench_2Pair_VLM_judge_report.json")
    args = parser.parse_args()

    with args.pairs_csv.open(encoding="utf-8", newline="") as handle:
        votes = {row["pair_id"].split("/")[0]: row for row in csv.DictReader(handle) if row["subset"] == "2pair"}
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))

    output, changed, original_ties = [], [], []
    for sample in benchmark:
        sample_id = sample["sample_id"]
        row = votes[sample_id]
        original = LABEL_TO_RANKING[int(row["original_overall"])]
        if original != sample["metadata_ranking"]:
            raise ValueError(f"{sample_id}: pairs.csv original {original} != benchmark {sample['metadata_ranking']}")
        if original == "A=B":
            original_ties.append(sample_id)
            continue
        if row["majority_overall"] in {"", "None", "null"}:
            raise ValueError(f"{sample_id}: VLM vote is unresolved")
        relabeled = dict(sample)
        relabeled["metadata_ranking"] = LABEL_TO_RANKING[int(row["majority_overall"])]
        if relabeled["metadata_ranking"] == "A=B":
            raise ValueError(f"{sample_id}: VLM majority is a tie on an originally non-tie pair")
        if relabeled["metadata_ranking"] != original:
            changed.append(sample_id)
        output.append(relabeled)

    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "benchmark_pairs": len(benchmark),
        "original_ties_removed": len(original_ties),
        "written_pairs": len(output),
        "changed_vs_original": len(changed),
        "ranking_counts_original": dict(
            collections.Counter(s["metadata_ranking"] for s in benchmark if s["sample_id"] not in original_ties)
        ),
        "ranking_counts_vlm_judge": dict(collections.Counter(s["metadata_ranking"] for s in output)),
        "changed_sample_ids": changed,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "changed_sample_ids"}, indent=2))


if __name__ == "__main__":
    main()
