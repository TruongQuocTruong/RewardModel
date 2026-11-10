"""Download EditReward-Bench from the EditScore paper (Hugging Face: EditScore/EditReward-Bench).

This bench is different from the 115-pair EditReward-Bench already in this folder, so it is
stored under the name EditScore-EditReward-Bench.

Step 1 only: download the raw files and check them. Conversion to our pairwise JSON format
is a separate script.

The repository is a `datasets.save_to_disk` folder (two .arrow shards, dataset_info.json,
state.json). After the download, the script checks:
  - every file exists and has the size reported by the Hugging Face API;
  - the folder loads with `datasets.load_from_disk`, has the expected number of rows and columns;
  - counts per task_type and per dimension (text columns only, images are not decoded).
A JSON report is written next to the raw files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPOSITORY_ID = "EditScore/EditReward-Bench"
EXPECTED_ROWS = 3072
EXPECTED_COLUMNS = {"key", "instruction", "input_image", "output_images", "task_type", "dimension"}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def remote_file_sizes(revision: str) -> tuple[dict[str, int], str]:
    api = HfApi()
    info = api.dataset_info(REPOSITORY_ID, revision=revision, files_metadata=True)
    return {sibling.rfilename: int(sibling.size) for sibling in info.siblings if sibling.size is not None}, info.sha


def check_file_sizes(output_dir: Path, expected: dict[str, int]) -> list[str]:
    problems = []
    for name, size in sorted(expected.items()):
        path = output_dir / name
        if not path.is_file():
            problems.append(f"missing file: {name}")
        elif path.stat().st_size != size:
            problems.append(f"size mismatch: {name} local={path.stat().st_size} remote={size}")
    return problems


def check_dataset(output_dir: Path) -> dict:
    from datasets import load_from_disk

    dataset = load_from_disk(str(output_dir))
    columns = set(dataset.column_names)
    text_only = dataset.select_columns(["key", "task_type", "dimension"])
    keys = text_only["key"]
    pair_keys = [tuple(pair) for pair in keys]
    image_keys = [key for pair in keys for key in pair]
    return {
        "rows": len(dataset),
        "columns": sorted(columns),
        "missing_columns": sorted(EXPECTED_COLUMNS - columns),
        "task_type_counts": dict(sorted(Counter(text_only["task_type"]).items())),
        "dimension_counts": dict(sorted(Counter(text_only["dimension"]).items())),
        "unique_pairs": len(set(pair_keys)),
        "unique_candidate_images": len(set(image_keys)),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_sizes, commit = remote_file_sizes(args.revision)
    print(f"Repository {REPOSITORY_ID} revision {args.revision} commit {commit}", flush=True)
    for name, size in sorted(expected_sizes.items()):
        print(f"  {name}: {size / 1e9:.3f} GB", flush=True)

    snapshot_download(
        repo_id=REPOSITORY_ID,
        repo_type="dataset",
        revision=commit,
        local_dir=str(output_dir),
        max_workers=args.max_workers,
    )

    problems = check_file_sizes(output_dir, expected_sizes)
    report = {"repository": REPOSITORY_ID, "commit": commit, "output_dir": str(output_dir),
              "file_sizes": expected_sizes, "file_problems": problems}
    if not problems:
        report["dataset"] = check_dataset(output_dir)
        if report["dataset"]["rows"] != EXPECTED_ROWS:
            problems.append(f"expected {EXPECTED_ROWS} rows, found {report['dataset']['rows']}")
        if report["dataset"]["missing_columns"]:
            problems.append(f"missing columns: {report['dataset']['missing_columns']}")

    report_path = output_dir.parent / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"Report written to {report_path}", flush=True)

    if problems:
        print("Download check FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print("Download check passed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
