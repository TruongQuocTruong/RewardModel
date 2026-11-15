# -*- coding: utf-8 -*-

"""
Build AURA-Bench.jsonl: human gold labels from the user study.

Input:
- UserStudy/id_userstudy_pairs.jsonl: 2400 pairs (sample_id, metric, images, models, ...)
- UserStudy/IAACalculation/results_labeling_filtered_for_iaa.csv: votes kept after the
  IAA filtering (1561 samples; 1-1, 1-1-1, 4-rater 2-2, single-rater samples and
  40 random 3-rater 2-1 samples per metric removed)

Output: one line per kept sample, with image paths relative to
SOURCE_IMG_BENCH / EDITED_IMG_ROOT, vote counts and the majority label.
Label convention: "A" = edited_images[0] better, "B" = edited_images[1] better,
"Tie" = equal. A sample without a strict majority stops the script with an error.
"""

import argparse
import csv
import json
import os
from collections import Counter

SOURCE_IMG_BENCH = "/datastore/truongtq/ImageEditingBenchmark/AURABenchmark/AURA-BENCHMARK/"
EDITED_IMG_ROOT = "/datastore/truongtq/ImageEditingBenchmark/EditingEvaluation/OutputEdited/"

SRC_PREFIX = "subset_userstudy/"
EDITED_PREFIX = "OutputEdited/"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERSTUDY_DIR = "/datastore/truongtq/TrainingRewardModel/UserStudy"


def strip_prefix(path, prefix):
    if not path.startswith(prefix):
        raise ValueError(f"Path '{path}' does not start with '{prefix}'")
    return path[len(prefix):]


def map_selection(row):
    """Convert a raw selection into A, B or Tie (A = model_a = edited_images[0])."""
    selected = row['selected_model'].strip().lower()
    if selected in ['tie', 'both', 'none']:
        return 'Tie'
    if selected == row['model_a'].strip().lower():
        return 'A'
    if selected == row['model_b'].strip().lower():
        return 'B'
    raise ValueError(f"Unknown selection: {row}")


def load_votes(csv_path):
    """Return {sample_id: [vote rows]}, keeping the latest vote per participant."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: r['timestamp'])
    latest = {}
    for r in rows:
        latest[(r['sample_id'].strip(), r['participant_id'].strip())] = r

    votes = {}
    for (sample_id, _), r in latest.items():
        votes.setdefault(sample_id, []).append(r)
    return votes


def majority_label(counts):
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    return winners[0] if len(winners) == 1 else None


def observed_agreement(counts):
    n = sum(counts.values())
    if n < 2:
        return None
    return round(sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)), 4)


def parse_args():
    parser = argparse.ArgumentParser(description="Build AURA-Bench.jsonl from the filtered user study votes.")
    parser.add_argument("--pairs_path", type=str, default=os.path.join(USERSTUDY_DIR, "id_userstudy_pairs.jsonl"))
    parser.add_argument("--votes_csv", type=str,
                        default=os.path.join(USERSTUDY_DIR, "IAACalculation", "results_labeling_filtered_for_iaa.csv"))
    parser.add_argument("--output_path", type=str, default=os.path.join(BASE_DIR, "AURA-Bench.jsonl"))
    parser.add_argument("--check_exists", action="store_true",
                        help="Check that every source/edited image exists under its root.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.pairs_path, encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    votes = load_votes(args.votes_csv)

    missing_files = []
    label_counter = Counter()
    records = []

    for pair in pairs:
        sample_id = pair['sample_id']
        if sample_id not in votes:
            continue

        sample_votes = votes[sample_id]
        for r in sample_votes:
            if [r['model_a'].strip(), r['model_b'].strip()] != pair['models']:
                raise ValueError(f"Model order mismatch for {sample_id}: {r['model_a']}, {r['model_b']} vs {pair['models']}")

        counts = Counter({'A': 0, 'B': 0, 'Tie': 0})
        counts.update(map_selection(r) for r in sample_votes)
        label = majority_label(counts)
        if label is None:
            raise ValueError(f"No strict majority for {sample_id}: {dict(counts)}; remove it from the votes CSV")
        label_counter[label] += 1

        src_image = strip_prefix(pair['src_image'], SRC_PREFIX)
        edited_images = [strip_prefix(p, EDITED_PREFIX) for p in pair['edited_images']]

        if args.check_exists:
            for root, rel in [(SOURCE_IMG_BENCH, src_image)] + [(EDITED_IMG_ROOT, p) for p in edited_images]:
                if not os.path.isfile(os.path.join(root, rel)):
                    missing_files.append(os.path.join(root, rel))

        records.append({
            'sample_id': sample_id,
            'metric': pair['metric'],
            'image_id': pair['image_id'],
            'edit_type': pair['edit_type'],
            'prompt': pair['prompt'],
            'src_image': src_image,
            'edited_images': edited_images,
            'models': pair['models'],
            'num_raters': sum(counts.values()),
            'votes': dict(counts),
            'observed_agreement': observed_agreement(counts),
            'human_label': label,
            'preferred_model': {'A': pair['models'][0], 'B': pair['models'][1], 'Tie': 'Tie'}.get(label),
        })

    unmatched = set(votes) - {p['sample_id'] for p in pairs}
    if unmatched:
        raise ValueError(f"{len(unmatched)} voted samples not found in pairs file, e.g. {sorted(unmatched)[:3]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Pairs: {len(pairs)} | Samples with votes: {len(records)}")
    print(f"Per metric: {dict(Counter(r['metric'] for r in records))}")
    print(f"Labels: {dict(label_counter)}")
    print(f"Raters per sample: {dict(sorted(Counter(r['num_raters'] for r in records).items()))}")
    if args.check_exists:
        print(f"Missing image files: {len(missing_files)}")
        for p in missing_files[:10]:
            print(f"  {p}")
    print(f"Saved to: {args.output_path}")


if __name__ == '__main__':
    main()
