# -*- coding: utf-8 -*-

"""
Dataset statistics of AURA-Bench for the benchmark comparison table (Table 1):
#Images, #Instructions, Avg. #Words, Med. #Pixels (K), Editing Object #Categories / Med. Size (%), #Metrics.

Definitions
- Image: unique source image (image_id).
- Instruction: unique (image_id, edit_type, prompt); several model pairs and metrics share one instruction.
- Words: whitespace-separated tokens of each unique instruction, mean (min-max).
- Pixels (K): width x height / 1000 of each unique source image ("resolution" in jsons/<id>.json),
  median (min-max).
- Editing object: a detected object class of the source image whose name appears in the instruction
  (analyze_aura_bench.match_edited_objects). Instructions that name no detected object (global edits such as
  style transfer, weather, contrast) have no editing object.
- Object size (%): per (instruction, editing object class), area of the union of the masks of all detected
  instances of that class divided by the mask frame (mask "size"; boxes and masks are given in the original
  photo frame, which can differ from "resolution"). Reported as median (min-max).

Only metadata files are read (benchmark_meta_mapping is not needed; no image file is opened).
Outputs: benchmark_table_statistics.json and benchmark_comparison_table.md next to this script (--output_dir).
"""

import argparse
import ast
import json
import os

import numpy as np

from analyze_aura_bench import DEFAULT_LABEL_PATH, METRIC_ORDER, SOURCE_IMG_BENCH, match_edited_objects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rows copied from the comparison table of prior benchmarks (values as published).
TABLE_HEADER = ['Benchmark', '#Images', '#Instructions', 'Avg. #Words', 'Med. #Pixels (K)',
                'Editing Object: #Categories', 'Editing Object: Med. Size (%)', '#Metrics']
PRIOR_BENCHMARKS = [
    ['ICE-Bench [30]', '5139', '5139', '27.6 (3-233)', '1023 (38-1049)', '-', '-', '-'],
    ['ImgEdit [44]', '226', '737', '12.0 (3-44)', '1049 (786-8294)', '-', '-', '3'],
    ['GEdit-Bench [24]', '606', '606', '7.6 (2-26)', '270 (245-279)', '-', '-', '3'],
    ['MagicBrush [45]', '1053', '1053', '6.6 (2-16)', '250 (250-1049)', '80', '-', '5'],
    ['HATIE [33]', '18226', '49840', '7.0 (4-13)', '262', '76', '6.6 (0.5-92)', '5'],
    ['ImagenWorld [26]', '611', '611', '16.3 (4-51)', '620 (37-14746)', '-', '-', '4'],
    ['Omni IIE Bench [43]', '1240', '1723', '23.72 (7-105)', '284 (38-12193)', '-', '-', '-'],
    ['CompBench [19]', '2801', '2905', '9.97 (1-33)', '410', '-', '3.1 (0.1-58)', '-'],
]


# =========================================================
# COCO run-length encoding (compressed string form) without pycocotools
# =========================================================

def rle_counts(counts):
    """Decode a COCO compressed RLE string into run lengths (alternating background / foreground)."""
    if isinstance(counts, list):
        return [int(c) for c in counts]
    runs = []
    p = 0
    while p < len(counts):
        x, k, more = 0, 0, True
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    return runs


def parse_rle(mask):
    """Mask field of an object (dict or its string form) -> {'size': [h, w], 'counts': ...}."""
    return ast.literal_eval(mask) if isinstance(mask, str) else mask


def rle_decode(rle):
    """Binary mask (h, w) from a COCO RLE (column-major runs)."""
    h, w = rle['size']
    runs = rle_counts(rle['counts'])
    flat = np.repeat(np.arange(len(runs)) % 2 == 1, runs)
    if flat.size != h * w:
        raise ValueError(f"RLE covers {flat.size} pixels, expected {h * w}")
    return flat.reshape((w, h)).T


def union_area_percent(rles):
    """Area of the union of masks (same frame) as a percentage of the frame."""
    frames = {tuple(r['size']) for r in rles}
    if len(frames) != 1:
        raise ValueError(f"Masks have different frames: {frames}")
    union = np.zeros(next(iter(frames)), dtype=bool)
    for r in rles:
        union |= rle_decode(r)
    return float(union.mean() * 100)


# =========================================================
# Statistics
# =========================================================

def load_image_json(src_image, source_root):
    folder, _, filename = src_image.split('/')
    path = os.path.join(source_root, folder, 'jsons', os.path.splitext(filename)[0] + '.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def range_stats(values):
    v = np.asarray(values, dtype=float)
    return {'count': int(v.size), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'min': float(v.min()), 'max': float(v.max())}


def compute_statistics(records, image_jsons):
    instructions = sorted({(r['image_id'], r['edit_type'], r['prompt']) for r in records})

    pixels_k = [d['resolution']['width'] * d['resolution']['height'] / 1000 for d in image_jsons.values()]
    words = [len(prompt.split()) for _, _, prompt in instructions]

    categories = set()
    object_sizes = []
    instructions_with_object = 0
    for image_id, _, prompt in instructions:
        objects = image_jsons[image_id].get('objects', [])
        names = sorted({o['class_name'].strip().lower() for o in objects})
        matched = match_edited_objects(prompt, names)
        if not matched:
            continue
        instructions_with_object += 1
        categories.update(matched)
        for name in matched:
            rles = [parse_rle(o['mask']) for o in objects if o['class_name'].strip().lower() == name]
            object_sizes.append(union_area_percent(rles))

    return {
        'num_pairs': len(records),
        'num_images': len(image_jsons),
        'num_instructions': len(instructions),
        'num_unique_instruction_texts': len({p for _, _, p in instructions}),
        'instructions_per_image': len(instructions) / len(image_jsons),
        'words': range_stats(words),
        'pixels_k': range_stats(pixels_k),
        'instructions_with_editing_object': instructions_with_object,
        'editing_object_categories': len(categories),
        'editing_object_category_names': sorted(categories),
        'editing_object_size_percent': range_stats(object_sizes) if object_sizes else None,
        'num_metrics': len({r['metric'] for r in records}),
        'metrics': [m for m in METRIC_ORDER if any(r['metric'] == m for r in records)],
    }


def aura_row(stats):
    w, px, size = stats['words'], stats['pixels_k'], stats['editing_object_size_percent']
    size_text = f"{size['median']:.1f} ({size['min']:.1f}-{size['max']:.0f})" if size else '-'
    return ['**AURA-Bench**', str(stats['num_images']), str(stats['num_instructions']),
            f"{w['mean']:.1f} ({w['min']:.0f}-{w['max']:.0f})",
            f"{px['median']:.0f} ({px['min']:.0f}-{px['max']:.0f})",
            str(stats['editing_object_categories']), size_text, f"**{stats['num_metrics']}**"]


def caption(stats):
    """Table caption; every claim is taken from the computed statistics or the prior rows above."""
    size = stats['editing_object_size_percent']
    size_text = (f" with a median size of {size['median']:.1f}% of the image, larger than in HATIE (6.6%) "
                 f"and CompBench (3.1%)") if size else ""
    return (f"Table 1. Comparison of dataset statistics between AURA-Bench and existing benchmarks. "
            f"AURA-Bench evaluates edits along {stats['num_metrics']} metrics, the most among the compared "
            f"benchmarks, with {stats['num_pairs']} human-labelled pairs over {stats['num_instructions']} "
            f"instructions. Its source images have a high median resolution ({stats['pixels_k']['median']:.0f} K "
            f"pixels), and its editing objects span {stats['editing_object_categories']} categories"
            f"{size_text}. [Keys: (min-max) values, Med. = median]")


def markdown_table(rows):
    lines = ['| ' + ' | '.join(TABLE_HEADER) + ' |',
             '| :--- | ' + ' | '.join([':---:'] * (len(TABLE_HEADER) - 1)) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


# =========================================================
# Main
# =========================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AURA-Bench statistics for the benchmark comparison table.")
    parser.add_argument("--label_path", type=str, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--source_root", type=str, default=SOURCE_IMG_BENCH)
    parser.add_argument("--output_dir", type=str, default=BASE_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.label_path, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"No samples in {args.label_path}")

    image_jsons = {}
    for r in records:
        if r['image_id'] not in image_jsons:
            image_jsons[r['image_id']] = load_image_json(r['src_image'], args.source_root)

    stats = compute_statistics(records, image_jsons)
    table = markdown_table(PRIOR_BENCHMARKS + [aura_row(stats)])

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'benchmark_table_statistics.json')
    md_path = os.path.join(args.output_dir, 'benchmark_comparison_table.md')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(caption(stats) + "\n\n" + table + "\n")

    print(caption(stats) + "\n\n" + table)
    print(f"\nSaved: {json_path}\nSaved: {md_path}")
    return stats


if __name__ == '__main__':
    main()
