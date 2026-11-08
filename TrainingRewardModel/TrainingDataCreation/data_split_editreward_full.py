#!/usr/bin/env python3
"""Hold out a fraction of EditReward-Data source images as a validation split.

Every pair of a held-out source image goes to validation, all other pairs to
train, so no source image (and its edit instructions) is shared between the
two parts. Reads only the metadata fields ``unique_key``, ``source_img`` and
``vote_type``; no image is opened.

The output JSON has ``sample_partition`` ({unique_key: "train" | "validation"}),
the format read by ``train_overall_inspector.py`` through
``OVERALL_INSPECTOR_SPLIT``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VOTE_TO_CLASS = {"leftvote": "A", "rightvote": "B", "tie": "Tie", "tievote": "Tie"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default="/datastore/shared/datasets/EditReward-Data/extracted_data/metadata.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-source-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_source_fraction < 1.0:
        raise ValueError("--validation-source-fraction must be in (0, 1).")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing split: {output}")

    pairs = []
    seen = set()
    with open(args.metadata, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["unique_key"].strip()
            if sample_id in seen:
                raise ValueError(f"Line {line_number}: duplicate unique_key {sample_id}")
            seen.add(sample_id)
            vote = VOTE_TO_CLASS[row["vote_type"].strip().lower()]
            pairs.append((sample_id, row["source_img"].strip(), vote))

    sources = sorted({source for _, source, _ in pairs})
    count = round(args.validation_source_fraction * len(sources))
    held_out = set(random.Random(args.seed).sample(sources, count))
    partition = {
        sample_id: ("validation" if source in held_out else "train")
        for sample_id, source, _ in pairs
    }
    votes = {name: Counter() for name in ("train", "validation")}
    for sample_id, _, vote in pairs:
        votes[partition[sample_id]][vote] += 1

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": str(Path(args.metadata).resolve()),
        "method": "random source images held out as validation",
        "validation_source_fraction": args.validation_source_fraction,
        "seed": args.seed,
        "source_images": {"total": len(sources), "validation": len(held_out)},
        "pairs": {name: sum(counter.values()) for name, counter in votes.items()},
        "votes": {name: dict(sorted(counter.items())) for name, counter in votes.items()},
        "validation_source_images": sorted(held_out),
        "sample_partition": partition,
    }
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("source_images", "pairs", "votes")}, indent=1))


if __name__ == "__main__":
    main()
