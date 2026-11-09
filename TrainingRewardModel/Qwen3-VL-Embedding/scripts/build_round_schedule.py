#!/usr/bin/env python3
"""Write the per-round Unlabeled sample IDs of a finished Stage 2 run for --replay-round-selections.

Each round_XX_summary.json of a train_loop.py output folder records the IDs that round trained
(``selected_sample_ids``). Replaying them gives later runs the same Unlabeled data and step budget,
so a comparison isolates the change under test from the agreement filter's run-specific choices.

Usage:
    python scripts/build_round_schedule.py <stage2 output folder> <round_schedule.json> [--rounds 5]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    schedule = []
    for index in range(1, args.rounds + 1):
        summary = json.loads((args.run_dir / f"round_{index:02d}_summary.json").read_text())
        ids = summary["selected_sample_ids"]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Round {index} has an empty or duplicated selection.")
        schedule.append(ids)
    args.output.write_text(json.dumps(schedule))
    print(f"{args.output}: rounds {[len(ids) for ids in schedule]}")


if __name__ == "__main__":
    main()
