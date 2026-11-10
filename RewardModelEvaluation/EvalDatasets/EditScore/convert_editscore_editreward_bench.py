"""Convert EditReward-Bench of the EditScore paper to our pairwise JSON format.

Input: the raw `datasets.save_to_disk` folder (Raw/) downloaded by
download_editscore_editreward_bench.py.

Label of the original bench: there is no label column; in every row output_images[0] is the
image the annotators preferred for that row's dimension (EditScore's calculate_statistics.py
counts a row as correct when score[0] > score[1]). There are no ties.

What this script does (only the dimensions given by --dimensions, default overall):
  - writes the images once, without re-encoding (original bytes):
      Extracted/images/<key>.<ext> for candidate images (key = id given by the bench),
      Extracted/images/source_<md5>.<ext> for source images (deduplicated by content);
  - puts the preferred image at position A or B by a fixed rule (parity of a SHA-256 hash of
    sample_id), so the label is not always A>B; the original order is kept in the record;
  - writes <output-prefix>_<dimension>_pairwise.json with the fields read by
    RewardModelEvaluation/OurModel/evaluate_reward_model.py
    (sample_id, source_image, candidate_1, candidate_2, edit_instruction, metadata_ranking)
    plus task_type, dimension, original_keys, preferred_key, swapped, image sizes;
  - writes a conversion report (counts per task_type, A>B / B>A balance, size checks).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = HERE / "Raw"
DEFAULT_EXTRACTED_DIR = HERE / "Extracted"
DEFAULT_OUTPUT_PREFIX = HERE / "EditScore-EditReward-Bench"
DIMENSIONS = ("overall", "prompt_following", "consistency")
FORMAT_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif", "GIF": "gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--extracted-dir", default=str(DEFAULT_EXTRACTED_DIR))
    parser.add_argument("--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--dimensions", nargs="+", default=["overall"], choices=DIMENSIONS)
    return parser.parse_args()


def image_info(data: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        extension = FORMAT_EXTENSIONS.get(image.format)
        if extension is None:
            raise ValueError(f"Unsupported image format: {image.format}")
        return extension, image.width, image.height


def raw_bytes(value: dict) -> bytes:
    data = value.get("bytes")
    if data is None:
        raise ValueError(f"Image without embedded bytes: {value.get('path')!r}")
    return data


class ImageWriter:
    """Writes each image once and checks that a reused key always has the same bytes."""

    def __init__(self, images_dir: Path):
        self.images_dir = images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.by_name: dict[str, tuple[str, str, int, int]] = {}  # name -> (md5, path, width, height)
        self.written = 0
        self.reused = 0

    def write(self, name: str, data: bytes) -> tuple[str, int, int]:
        digest = hashlib.md5(data).hexdigest()
        if name in self.by_name:
            known_digest, path, width, height = self.by_name[name]
            if known_digest != digest:
                raise ValueError(f"Image name {name} is used for two different images")
            self.reused += 1
            return path, width, height
        extension, width, height = image_info(data)
        path = self.images_dir / f"{name}.{extension}"
        if path.exists():
            if hashlib.md5(path.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Refusing to overwrite a different image: {path}")
        else:
            path.write_bytes(data)
        self.by_name[name] = (digest, str(path), width, height)
        self.written += 1
        return str(path), width, height


def is_swapped(sample_id: str) -> bool:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % 2 == 1


def main() -> int:
    args = parse_args()
    from datasets import Image as ImageFeature
    from datasets import Sequence, load_from_disk

    raw_dir = Path(args.raw_dir).resolve()
    extracted_dir = Path(args.extracted_dir).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    outputs = {dimension: Path(f"{output_prefix}_{dimension}_pairwise.json") for dimension in args.dimensions}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    dataset = load_from_disk(str(raw_dir))
    dataset = dataset.cast_column("input_image", ImageFeature(decode=False))
    dataset = dataset.cast_column("output_images", Sequence(ImageFeature(decode=False)))
    wanted = set(args.dimensions)
    indices = [i for i, dimension in enumerate(dataset["dimension"]) if dimension in wanted]
    print(f"Rows: {len(dataset)}, selected for {sorted(wanted)}: {len(indices)}", flush=True)

    writer = ImageWriter(extracted_dir / "images")
    records: dict[str, list[dict]] = {dimension: [] for dimension in args.dimensions}
    seen_ids: set[str] = set()
    size_mismatch = Counter()

    for done, index in enumerate(indices, start=1):
        row = dataset[index]
        dimension = row["dimension"]
        key_preferred, key_other = row["key"]
        sample_id = f"{dimension}_{key_preferred}_{key_other}"
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)

        source_bytes = raw_bytes(row["input_image"])
        source_name = "source_" + hashlib.md5(source_bytes).hexdigest()
        source_path, source_w, source_h = writer.write(source_name, source_bytes)
        preferred_path, preferred_w, preferred_h = writer.write(key_preferred, raw_bytes(row["output_images"][0]))
        other_path, other_w, other_h = writer.write(key_other, raw_bytes(row["output_images"][1]))
        for width, height in ((preferred_w, preferred_h), (other_w, other_h)):
            size_mismatch[(width, height) != (source_w, source_h)] += 1

        swapped = is_swapped(sample_id)
        if swapped:
            candidate_1, candidate_2, ranking = other_path, preferred_path, "B>A"
            size_1, size_2 = [other_w, other_h], [preferred_w, preferred_h]
        else:
            candidate_1, candidate_2, ranking = preferred_path, other_path, "A>B"
            size_1, size_2 = [preferred_w, preferred_h], [other_w, other_h]

        records[dimension].append(
            {
                "sample_id": sample_id,
                "source_image": source_path,
                "candidate_1": candidate_1,
                "candidate_2": candidate_2,
                "edit_instruction": row["instruction"].strip(),
                "metadata_ranking": ranking,
                "task_type": row["task_type"],
                "dimension": dimension,
                "original_keys": [key_preferred, key_other],
                "preferred_key": key_preferred,
                "swapped": swapped,
                "source_size": [source_w, source_h],
                "candidate_1_size": size_1,
                "candidate_2_size": size_2,
                "original_row_index": index,
            }
        )
        if done % 200 == 0:
            print(f"  {done}/{len(indices)} rows", flush=True)

    report = {
        "raw_dir": str(raw_dir),
        "extracted_dir": str(extracted_dir),
        "dimensions": {},
        "images_written": writer.written,
        "image_reuses": writer.reused,
        "candidate_size_differs_from_source": size_mismatch[True],
        "candidate_size_equals_source": size_mismatch[False],
        "swap_rule": "swapped = int(sha256(sample_id), 16) % 2 == 1; swapped rows have metadata_ranking B>A",
    }
    for dimension, rows in records.items():
        outputs[dimension].write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        report["dimensions"][dimension] = {
            "output": str(outputs[dimension]),
            "pairs": len(rows),
            "ranking_counts": dict(Counter(r["metadata_ranking"] for r in rows)),
            "task_type_counts": dict(sorted(Counter(r["task_type"] for r in rows).items())),
            "unique_instructions": len({r["edit_instruction"] for r in rows}),
            "unique_source_images": len({r["source_image"] for r in rows}),
        }
        print(f"Wrote {len(rows)} pairs to {outputs[dimension]}", flush=True)

    report_path = Path(f"{output_prefix}_{'_'.join(args.dimensions)}_conversion_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
