# -*- coding: utf-8 -*-

"""
Dataset statistics of AURA-Benchmark (10636 unlabelled instructions) and AURA-Test (1561 human-labelled
pairs) for the benchmark comparison table (Table 1):
#Images, #Instructions, Avg. #Words, Med. #Pixels (K), Editing Object #Categories / Med. Size (%), #Metrics.

Definitions (identical for both datasets)
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
- Metrics: distinct "metric" values of the samples; AURA-Benchmark has no metric field, so --num_metrics is used.

Only metadata files are read (no image file is opened).
Outputs in --output_dir: benchmark_table_statistics.json and benchmark_comparison_table.md
(caption, Markdown table and LaTeX rows).
"""

import argparse
import ast
import json
import os
import time

import numpy as np

from analyze_aura_bench import METRIC_ORDER, SOURCE_IMG_BENCH, match_edited_objects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AURA_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DEFAULT_TEST_PATH = os.path.join(AURA_DIR, "AURA-TEST", "AURA-Bench.jsonl")
DEFAULT_BENCHMARK_PATH = os.path.join(AURA_DIR, "AURA-BENCHMARK", "AURA-Benchmark.jsonl")

# Dataset key -> (Markdown name, LaTeX name)
DATASETS = {
    'benchmark': ('AURA-Benchmark', r'\benchmarkName'),
    'test': ('AURA-Test', r'\testName'),
}

TABLE_HEADER = ['Benchmark', '#Images', '#Instructions', 'Avg. #Words', 'Med. #Pixels (K)',
                'Editing Object: #Categories', 'Editing Object: Med. Size (%)', '#Metrics']
# Prior benchmarks as in the paper table (2026-11-15): (Markdown name, LaTeX name, cells).
PRIOR_BENCHMARKS = [
    ('ICE-Bench', r'ICE-Bench~\cite{pan2025ice}',
     ['5139', '5139', '27.6 (3-233)', '1023 (38-1049)', '-', '-', '6']),
    ('ImgEdit', r'ImgEdit~\cite{ye2025imgedit}',
     ['221', '787', '12.0 (3-44)', '1049 (786-8294)', '-', '-', '3']),
    ('GEdit-Bench', r'GEdit-Bench~\cite{liu2025step1x}',
     ['606', '606', '7.6 (2-26)', '270 (245-279)', '-', '-', '2']),
    ('MagicBrush', r'MagicBrush~\cite{zhang2023magicbrush}',
     ['1053', '1053', '6.6 (2-16)', '250 (250-1049)', '80', '-', '5$^{*}$']),
    ('HATIE', r'HATIE~\cite{ryu2025towards}',
     ['18226', '49840', '7.0 (4-13)', '262', '76', '6.6 (0.5-92)', '5']),
    ('ImagenWorld', r'ImagenWorld~\cite{mahdizadeh2026imagenworld}',
     ['611', '611', '16.3 (4-51)', '620 (37-14746)', '-', '-', '4']),
    ('Omni IIE Bench', r'Omni IIE Bench~\cite{yang2026omni}',
     ['1240', '1723', '23.72 (7-105)', '284 (38-12193)', '-', '-', '3']),
    ('CompBench', r'CompBench~\cite{jia2026compbench}',
     ['2801', '2905', '9.97 (1-33)', '410', '-', '3.1 (0.1-58)', '2']),
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
    h, w = next(iter(frames))
    if len(rles) == 1:  # a single mask: area from the run lengths, no decoding
        runs = rle_counts(rles[0]['counts'])
        if sum(runs) != h * w:
            raise ValueError(f"RLE covers {sum(runs)} pixels, expected {h * w}")
        return float(sum(runs[1::2]) / (h * w) * 100)
    union = np.zeros((h, w), dtype=bool)
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


def compute_statistics(records, image_jsons, num_metrics=None):
    """Table statistics of one dataset. num_metrics is used when the samples carry no 'metric' field."""
    instructions = sorted({(r['image_id'], r['edit_type'], r['prompt']) for r in records})

    pixels_k = [d['resolution']['width'] * d['resolution']['height'] / 1000 for d in image_jsons.values()]
    words = [len(prompt.split()) for _, _, prompt in instructions]

    categories = set()
    object_sizes = []
    instructions_with_object = 0
    size_cache = {}  # (image_id, class) -> union area %, shared by instructions on the same image
    for image_id, _, prompt in instructions:
        objects = image_jsons[image_id].get('objects', [])
        names = sorted({o['class_name'].strip().lower() for o in objects})
        matched = match_edited_objects(prompt, names)
        if not matched:
            continue
        instructions_with_object += 1
        categories.update(matched)
        for name in matched:
            if (image_id, name) not in size_cache:
                rles = [parse_rle(o['mask']) for o in objects if o['class_name'].strip().lower() == name]
                size_cache[(image_id, name)] = union_area_percent(rles)
            object_sizes.append(size_cache[(image_id, name)])

    metrics = [m for m in METRIC_ORDER if any(r.get('metric') == m for r in records)]
    labelled = any('metric' in r for r in records)
    if not labelled and num_metrics is None:
        raise ValueError("Samples have no 'metric' field: pass num_metrics")
    return {
        'num_samples': len(records),
        'human_labelled': labelled,
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
        'num_metrics': len({r['metric'] for r in records}) if labelled else int(num_metrics),
        'metrics': metrics,
    }


def table_cells(stats):
    """The 7 value cells of a dataset row (same formats as the prior benchmark rows)."""
    w, px, size = stats['words'], stats['pixels_k'], stats['editing_object_size_percent']
    size_text = f"{size['median']:.1f} ({size['min']:.1f}-{size['max']:.0f})" if size else '-'
    return [str(stats['num_images']), str(stats['num_instructions']),
            f"{w['mean']:.1f} ({w['min']:.0f}-{w['max']:.0f})",
            f"{px['median']:.0f} ({px['min']:.0f}-{px['max']:.0f})",
            str(stats['editing_object_categories']), size_text, str(stats['num_metrics'])]


def caption(all_stats):
    """Table caption; every claim follows from the computed statistics or the prior rows above."""
    parts = ["Table 1. Comparison of dataset statistics between AURA and existing benchmarks."]
    bench, test = all_stats.get('benchmark'), all_stats.get('test')
    if bench:
        parts.append(f"AURA-Benchmark contains {bench['num_instructions']} instructions over "
                     f"{bench['num_images']} images ({bench['instructions_per_image']:.1f} per image).")
    if test:
        parts.append(f"AURA-Test provides {test['num_samples']} human-labelled pairs over "
                     f"{test['num_instructions']} of these instructions.")
    any_stats = bench or test
    prior_max_metrics = max(int(c[6].split('$')[0]) for _, _, c in PRIOR_BENCHMARKS if c[6] != '-')
    if any_stats['num_metrics'] > prior_max_metrics:
        parts.append(f"Both are evaluated along {any_stats['num_metrics']} metrics, more than any prior benchmark.")
    elif any_stats['num_metrics'] == prior_max_metrics:
        parts.append(f"Both are evaluated along {any_stats['num_metrics']} metrics, the most among the compared "
                     f"benchmarks (tied with ICE-Bench).")
    sizes = [s['editing_object_size_percent']['median'] for s in (bench, test)
             if s and s['editing_object_size_percent']]
    if sizes and min(sizes) > 6.6:
        parts.append("Editing objects are larger (median size "
                     + " and ".join(f"{v:.1f}%" for v in sizes)
                     + ") than in HATIE (6.6%) and CompBench (3.1%).")
    parts.append("[Keys: (min-max) values, Med. = median]")
    return " ".join(parts)


def markdown_table(rows):
    lines = ['| ' + ' | '.join(TABLE_HEADER) + ' |',
             '| :--- | ' + ' | '.join([':---:'] * (len(TABLE_HEADER) - 1)) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def latex_rows(rows):
    return '\n'.join(' & '.join(row) + r' \\' for row in rows)


def load_dataset(path, source_root):
    with open(path, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"No samples in {path}")
    image_jsons = {}
    start = time.time()
    for r in records:
        if r['image_id'] not in image_jsons:
            image_jsons[r['image_id']] = load_image_json(r['src_image'], source_root)
            if len(image_jsons) % 100 == 0:
                print(f"  loaded {len(image_jsons)} image metadata files ({time.time() - start:.0f}s)", flush=True)
    return records, image_jsons


# =========================================================
# Main
# =========================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AURA-Benchmark / AURA-Test statistics for the comparison table.")
    parser.add_argument("--benchmark_path", type=str, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--datasets", type=str, default="benchmark,test",
                        help="Comma-separated subset of: benchmark, test.")
    parser.add_argument("--num_metrics", type=int, default=6,
                        help="Metrics of AURA-Benchmark (its samples carry no metric field).")
    parser.add_argument("--source_root", type=str, default=SOURCE_IMG_BENCH)
    parser.add_argument("--output_dir", type=str, default=BASE_DIR)
    args = parser.parse_args(argv)
    args.datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    unknown = set(args.datasets) - set(DATASETS)
    if not args.datasets or unknown:
        parser.error(f"--datasets must be a non-empty subset of {sorted(DATASETS)}, got {args.datasets}")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = {'benchmark': args.benchmark_path, 'test': args.test_path}

    all_stats = {}
    for key in ['benchmark', 'test']:
        if key not in args.datasets:
            continue
        print(f"{DATASETS[key][0]}: {paths[key]}", flush=True)
        records, image_jsons = load_dataset(paths[key], args.source_root)
        all_stats[key] = compute_statistics(records, image_jsons, num_metrics=args.num_metrics)

    md_rows = [[name] + cells for name, _, cells in PRIOR_BENCHMARKS]
    tex_rows = [[name] + cells for _, name, cells in PRIOR_BENCHMARKS]
    for key, stats in all_stats.items():
        cells = table_cells(stats)
        md_rows.append([f"**{DATASETS[key][0]}**"] + cells)
        tex_rows.append([DATASETS[key][1]] + cells)

    text = (caption(all_stats) + "\n\n" + markdown_table(md_rows)
            + "\n\nLaTeX rows:\n\n```latex\n" + latex_rows(tex_rows) + "\n```\n")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'benchmark_table_statistics.json')
    md_path = os.path.join(args.output_dir, 'benchmark_comparison_table.md')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({DATASETS[k][0]: v for k, v in all_stats.items()}, f, indent=2, ensure_ascii=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print("\n" + text)
    print(f"Saved: {json_path}\nSaved: {md_path}")
    return all_stats


if __name__ == '__main__':
    main()
