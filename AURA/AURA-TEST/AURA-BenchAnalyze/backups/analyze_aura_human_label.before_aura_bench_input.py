# -*- coding: utf-8 -*-

"""
Analyze AuraHumanLabel.jsonl (human gold labels from the user study).

Statistics and figures:
- edit_type distribution, edit_type x metric
- samples per metric and human label composition per metric
- GenericConcepts vs SpecialCases, special case types (lighting / occlusion / fisheye / topview)
- image categories (benchmark_meta_mapping.json)
- edited objects: detected object class names (jsons/<id>.json) matched against the instruction
- source image resolution and aspect ratio
- instruction word count
- raters per sample, observed agreement, model appearance and win rate

All figures and analysis_summary.json are saved next to this script.
Palette follows color_board.txt.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

SOURCE_IMG_BENCH = "/datastore/truongtq/ImageEditingBenchmark/AURABenchmark/BENCHMARK/EditVal-Benchmark/"
META_MAPPING_PATH = os.path.join(SOURCE_IMG_BENCH, "benchmark_meta_mapping.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "AuraHumanLabel.jsonl")

# =========================================================
# Palette (color_board.txt)
# =========================================================

PALETTE_17 = [
    '#E88574', '#EFA87C', '#E8BA8A', '#F7B57E', '#F7CB8E', '#B8BA7C',
    '#83C3BE', '#8DCAC6', '#93CDCA', '#75A6A3', '#56D1D9', '#41B7C6',
    '#3E909B', '#3B9E9B', '#40CCC2', '#47D5CA', '#3B8F80',
]
GRID_COLOR = '#DBEAF8'
TEXT_COLOR = '#1F1F1F'
MUTED_TEXT = '#5F6B6B'
BG_COLOR = '#FFFFFF'
NEUTRAL_GRAY = '#B9C0C4'

# Small categorical sets: alternate warm / cool so neighbours stay distinct
CATEGORICAL = ['#E88574', '#3E909B', '#F7B57E', '#56D1D9', '#B8BA7C', '#75A6A3']
LABEL_COLORS = {'A': '#E88574', 'B': '#3E909B', 'Tie': '#F7CB8E', 'No majority': NEUTRAL_GRAY}
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list('aura_teal', ['#FFFFFF', '#93CDCA', '#3E909B', '#1F5A61'])

METRIC_ORDER = ['Fidelity', 'Realism', 'Aesthetic', 'Background Consistency',
                'Foreground Consistency', 'Structure Consistency']
METRIC_SHORT = {
    'Fidelity': 'Fidelity', 'Realism': 'Realism', 'Aesthetic': 'Aesthetic',
    'Background Consistency': 'BC', 'Foreground Consistency': 'FC', 'Structure Consistency': 'SC',
}

STANDARD_RATIOS = {
    '1:1': 1.0, '5:4': 5 / 4, '4:3': 4 / 3, '3:2': 3 / 2, '16:9': 16 / 9, '2:1': 2.0,
    '4:5': 4 / 5, '3:4': 3 / 4, '2:3': 2 / 3, '9:16': 9 / 16,
}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'savefig.facecolor': BG_COLOR,
        'axes.edgecolor': GRID_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'axes.titlecolor': TEXT_COLOR,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'text.color': TEXT_COLOR,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': GRID_COLOR,
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 10,
        'legend.frameon': False,
        'axes.xmargin': 0.02,
        'axes.ymargin': 0.02,
    })


def save_fig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure: {path}")


def bar_labels(ax, bars, values, fmt='{}', horizontal=False):
    for bar, v in zip(bars, values):
        if horizontal:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, ' ' + fmt.format(v),
                    va='center', ha='left', fontsize=8.5, color=TEXT_COLOR)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v),
                    va='bottom', ha='center', fontsize=8.5, color=TEXT_COLOR)
    if not horizontal:
        ax.set_ylim(0, max(b.get_height() for b in bars) * 1.12)


# =========================================================
# Data loading
# =========================================================

def load_records(label_path):
    with open(label_path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_meta_mapping():
    with open(META_MAPPING_PATH, encoding='utf-8') as f:
        return {item['new_id']: item for item in json.load(f)}


def load_image_json(src_image):
    """GenericConcepts/images/000001.jpg -> GenericConcepts/jsons/000001.json (resolution, objects)."""
    folder, _, filename = src_image.split('/')
    json_path = os.path.join(SOURCE_IMG_BENCH, folder, 'jsons', os.path.splitext(filename)[0] + '.json')
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return {
        'width': data['resolution']['width'],
        'height': data['resolution']['height'],
        'class_names': sorted({o['class_name'].strip().lower() for o in data.get('objects', [])}),
    }


def match_edited_objects(prompt, class_names):
    """
    Detected object class names that appear in the instruction (singular or plural,
    whole words). A class contained in a longer matched class (car in car seat) is dropped.
    """
    text = prompt.lower()
    matched = [
        name for name in class_names
        if re.search(r'\b' + re.escape(name) + r'(s|es)?\b', text)
    ]
    return sorted(
        name for name in matched
        if not any(name != other and re.search(r'\b' + re.escape(name) + r'\b', other) for other in matched)
    )


def ratio_bucket(width, height):
    ratio = width / height
    return min(STANDARD_RATIOS, key=lambda k: abs(np.log(ratio / STANDARD_RATIOS[k])))


# =========================================================
# Figures
# =========================================================

def plot_edit_type(records, out_dir):
    counts = Counter(r['edit_type'] for r in records).most_common()
    names, values = zip(*counts)
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    bars = ax.bar(range(len(names)), values, color=PALETTE_17[:len(names)], width=0.62)
    bar_labels(ax, bars, values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', ' ') for n in names], rotation=40, ha='right')
    ax.set_ylabel('Number of samples')
    ax.set_title(f'Edit type distribution ({len(records)} samples)')
    save_fig(fig, out_dir, '01_edit_type_distribution.png')
    return dict(counts)


def plot_metric(records, out_dir):
    counts = Counter(r['metric'] for r in records)
    values = [counts[m] for m in METRIC_ORDER]
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    bars = ax.bar([METRIC_SHORT[m] for m in METRIC_ORDER], values, color=CATEGORICAL, width=0.62)
    bar_labels(ax, bars, values)
    ax.set_ylabel('Number of samples')
    ax.set_title('Samples per metric')
    save_fig(fig, out_dir, '02_samples_per_metric.png')
    return {m: counts[m] for m in METRIC_ORDER}


def plot_label_by_metric(records, out_dir):
    labels = ['A', 'B', 'Tie', 'No majority']
    table = {m: Counter() for m in METRIC_ORDER}
    for r in records:
        table[r['metric']][r['human_label'] or 'No majority'] += 1

    x = np.arange(len(METRIC_ORDER))
    bottom = np.zeros(len(METRIC_ORDER))
    fig, ax = plt.subplots(figsize=(5.1, 5))
    for lab in labels:
        vals = np.array([table[m][lab] for m in METRIC_ORDER], dtype=float)
        totals = np.array([sum(table[m].values()) for m in METRIC_ORDER], dtype=float)
        pct = vals / totals * 100
        ax.bar(x, pct, bottom=bottom, color=LABEL_COLORS[lab], width=0.62, label=lab,
               edgecolor=BG_COLOR, linewidth=1.5)
        for xi, b, p in zip(x, bottom, pct):
            if p >= 6:
                ax.text(xi, b + p / 2, f'{p:.0f}%', ha='center', va='center', fontsize=8.5, color=TEXT_COLOR)
        bottom += pct
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_SHORT[m] for m in METRIC_ORDER])
    ax.set_ylabel('Share of samples (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Human label per metric')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.09), fontsize=8.5,
              handlelength=1.2, columnspacing=1.0)
    save_fig(fig, out_dir, '03_human_label_by_metric.png')
    return {m: dict(table[m]) for m in METRIC_ORDER}


def plot_edit_type_by_metric(records, out_dir):
    edit_types = [e for e, _ in Counter(r['edit_type'] for r in records).most_common()]
    mat = np.zeros((len(edit_types), len(METRIC_ORDER)), dtype=int)
    for r in records:
        mat[edit_types.index(r['edit_type']), METRIC_ORDER.index(r['metric'])] += 1

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(mat, cmap=SEQUENTIAL_CMAP, aspect='auto')
    vmax = mat.max()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, mat[i, j], ha='center', va='center', fontsize=8.5,
                    color='#FFFFFF' if mat[i, j] > 0.6 * vmax else TEXT_COLOR)
    ax.set_xticks(range(len(METRIC_ORDER)))
    ax.set_xticklabels([METRIC_SHORT[m] for m in METRIC_ORDER], rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(edit_types)))
    ax.set_yticklabels([e.replace('_', ' ') for e in edit_types])
    ax.grid(False)
    ax.set_title('Samples per edit type and metric')
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='Number of samples')
    save_fig(fig, out_dir, '04_edit_type_by_metric.png')
    return {e: dict(zip(METRIC_ORDER, mat[i].tolist())) for i, e in enumerate(edit_types)}


def plot_source_type(records, meta, out_dir):
    def case_of(r):
        item = meta[r['image_id']]
        return item['dataset_root'] if item['type'] == 'SpecialCases' else 'GenericConcepts'

    sample_counts = Counter(case_of(r) for r in records)
    image_counts = Counter(case_of(r) for r in {r['image_id']: r for r in records}.values())
    order = ['GenericConcepts'] + sorted(k for k in sample_counts if k != 'GenericConcepts')

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6))
    for ax, counts, title in [(axes[0], sample_counts, 'Samples'), (axes[1], image_counts, 'Unique images')]:
        vals = [counts[k] for k in order]
        bars = ax.bar(order, vals, color=CATEGORICAL[:len(order)], width=0.62)
        bar_labels(ax, bars, vals)
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(['Generic' if k == 'GenericConcepts' else k for k in order], rotation=30, ha='right',
                           rotation_mode='anchor')
    save_fig(fig, out_dir, '05_source_and_special_case_type.png')

    special = [r for r in records if case_of(r) != 'GenericConcepts']
    special_edit = defaultdict(Counter)
    for r in special:
        special_edit[case_of(r)][r['edit_type']] += 1
    return {
        'samples': {k: sample_counts[k] for k in order},
        'unique_images': {k: image_counts[k] for k in order},
        'special_case_edit_types': {k: dict(v.most_common()) for k, v in special_edit.items()},
    }


def plot_category(records, meta, out_dir):
    sample_counts = Counter(c for r in records for c in meta[r['image_id']]['categories'])
    image_counts = Counter(c for i in {r['image_id'] for r in records} for c in meta[i]['categories'])
    names = [n for n, _ in sample_counts.most_common()]

    fig, ax = plt.subplots(figsize=(9, 2.9))
    y = np.arange(len(names))
    bars = ax.barh(y, [sample_counts[n] for n in names], color=PALETTE_17[::3][:len(names)], height=0.62)
    bar_labels(ax, bars, [f"{sample_counts[n]} ({image_counts[n]} img)" for n in names], horizontal=True)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Number of samples (multi-label)')
    ax.set_title('Image category distribution')
    ax.set_xlim(0, max(sample_counts.values()) * 1.2)
    save_fig(fig, out_dir, '06_image_category.png')
    return {n: {'samples': sample_counts[n], 'unique_images': image_counts[n]} for n in names}


def plot_edited_objects(records, image_info, out_dir, top_k=25):
    obj_counts = Counter()
    per_edit_type_hit = defaultdict(lambda: [0, 0])
    no_match = 0
    for r in records:
        matched = match_edited_objects(r['prompt'], image_info[r['image_id']]['class_names'])
        r['edited_objects'] = matched
        obj_counts.update(matched)
        per_edit_type_hit[r['edit_type']][1] += 1
        if matched:
            per_edit_type_hit[r['edit_type']][0] += 1
        else:
            no_match += 1

    top = obj_counts.most_common(top_k)
    names, values = zip(*top)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    y = np.arange(len(names))
    bars = ax.barh(y, values, color='#41B7C6', height=0.62)
    bar_labels(ax, bars, values, horizontal=True)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Samples whose instruction names the object')
    ax.set_title(f'Top {len(names)} edited objects ({len(records) - no_match}/{len(records)} matched)')
    save_fig(fig, out_dir, '07_edited_object_types.png')

    edit_types = sorted(per_edit_type_hit, key=lambda e: -per_edit_type_hit[e][0] / per_edit_type_hit[e][1])
    rates = [per_edit_type_hit[e][0] / per_edit_type_hit[e][1] * 100 for e in edit_types]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(edit_types))
    bars = ax.barh(y, rates, color='#3B8F80', height=0.62)
    bar_labels(ax, bars, rates, fmt='{:.0f}%', horizontal=True)
    ax.set_yticks(y)
    ax.set_yticklabels([e.replace('_', ' ') for e in edit_types])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Samples naming a detected object (%)')
    ax.set_title('Object match rate per edit type')
    save_fig(fig, out_dir, '08_object_match_rate_by_edit_type.png')

    return {
        'num_object_types': len(obj_counts),
        'samples_with_match': len(records) - no_match,
        'samples_without_match': no_match,
        'top_objects': dict(obj_counts.most_common(50)),
        'match_rate_by_edit_type': {e: round(per_edit_type_hit[e][0] / per_edit_type_hit[e][1], 4) for e in edit_types},
    }


def plot_resolution(records, image_info, meta, out_dir):
    image_ids = sorted({r['image_id'] for r in records})
    widths = np.array([image_info[i]['width'] for i in image_ids])
    heights = np.array([image_info[i]['height'] for i in image_ids])
    megapixels = widths * heights / 1e6
    is_special = np.array([meta[i]['type'] == 'SpecialCases' for i in image_ids])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    for mask, color, label in [(~is_special, '#41B7C6', 'GenericConcepts'), (is_special, '#E88574', 'SpecialCases')]:
        ax.scatter(widths[mask], heights[mask], s=36, color=color, label=f'{label} ({mask.sum()})',
                   edgecolor=BG_COLOR, linewidth=1, alpha=0.9)
    lim = max(widths.max(), heights.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=NEUTRAL_GRAY, linewidth=1, linestyle='--', zorder=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(axis='both')
    ax.set_xlabel('Width (px)')
    ax.set_ylabel('Height (px)')
    ax.set_title(f'Resolution ({len(image_ids)} images)')
    ax.legend(loc='upper left')

    ax = axes[1]
    bins = np.linspace(0, megapixels.max() * 1.02, 25)
    ax.hist([megapixels[~is_special], megapixels[is_special]], bins=bins, stacked=True, rwidth=0.62,
            color=['#41B7C6', '#E88574'], edgecolor=BG_COLOR, linewidth=1,
            label=['GenericConcepts', 'SpecialCases'])
    ax.set_xlabel('Megapixels')
    ax.set_ylabel('Number of images')
    ax.set_title('Image size (megapixels)')
    ax.legend()
    save_fig(fig, out_dir, '09_image_resolution.png')

    ratio_counts = Counter(ratio_bucket(w, h) for w, h in zip(widths, heights))
    sample_ratio_counts = Counter(ratio_bucket(image_info[r['image_id']]['width'], image_info[r['image_id']]['height'])
                                  for r in records)
    order = sorted(ratio_counts, key=lambda k: STANDARD_RATIOS[k])
    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    vals = [ratio_counts[k] for k in order]
    bars = ax.bar(order, vals, color='#3E909B', width=0.62)
    bar_labels(ax, bars, [f"{ratio_counts[k]} img\n{sample_ratio_counts[k]} smp" for k in order])
    ax.set_xlabel('Nearest standard aspect ratio (width:height)')
    ax.set_ylabel('Number of unique images')
    ax.set_title('Aspect ratio distribution')
    ax.set_ylim(0, max(vals) * 1.25)
    save_fig(fig, out_dir, '10_aspect_ratio.png')

    return {
        'num_unique_images': len(image_ids),
        'width': {'min': int(widths.min()), 'median': float(np.median(widths)), 'max': int(widths.max())},
        'height': {'min': int(heights.min()), 'median': float(np.median(heights)), 'max': int(heights.max())},
        'megapixels': {'min': round(float(megapixels.min()), 3), 'median': round(float(np.median(megapixels)), 3),
                       'max': round(float(megapixels.max()), 3)},
        'aspect_ratio_images': {k: ratio_counts[k] for k in order},
        'aspect_ratio_samples': {k: sample_ratio_counts[k] for k in order},
        'top_resolutions': {f'{w}x{h}': c for (w, h), c in Counter(zip(widths.tolist(), heights.tolist())).most_common(10)},
    }


def plot_word_count(records, out_dir):
    counts = np.array([len(r['prompt'].split()) for r in records])
    unique_prompts = {r['prompt'] for r in records}
    unique_counts = np.array([len(p.split()) for p in unique_prompts])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bins = np.arange(counts.min(), counts.max() + 2) - 0.5
    ax.hist(counts, bins=bins, color='#41B7C6', edgecolor=BG_COLOR, linewidth=1, rwidth=0.62)
    ax.axvline(np.mean(counts), color='#E88574', linewidth=1.2, alpha=0.8, label=f'Mean = {np.mean(counts):.1f}')
    ax.axvline(np.median(counts), color='#3B8F80', linewidth=0.9, alpha=0.45, linestyle='--', label=f'Median = {np.median(counts):.0f}')
    ax.set_xlabel('Words per instruction')
    ax.set_ylabel('Number of samples')
    ax.set_title(f'Instruction length ({len(records)} samples)')
    ax.legend(fontsize=8.5)
    save_fig(fig, out_dir, '11_instruction_word_count.png')

    by_type = defaultdict(list)
    for r, c in zip(records, counts):
        by_type[r['edit_type']].append(c)
    edit_types = sorted(by_type, key=lambda e: np.median(by_type[e]))
    fig, ax = plt.subplots(figsize=(9, 7))
    bp = ax.boxplot([by_type[e] for e in edit_types], orientation='horizontal', patch_artist=True, widths=0.6,
                    medianprops={'color': TEXT_COLOR, 'linewidth': 1.5},
                    flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': NEUTRAL_GRAY,
                                'markeredgecolor': NEUTRAL_GRAY})
    for patch in bp['boxes']:
        patch.set_facecolor('#93CDCA')
        patch.set_edgecolor('#3E909B')
    ax.set_yticks(range(1, len(edit_types) + 1))
    ax.set_yticklabels([e.replace('_', ' ') for e in edit_types])
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Words per instruction')
    ax.set_title('Instruction length per edit type')
    save_fig(fig, out_dir, '12_instruction_word_count_by_edit_type.png')

    return {
        'per_sample': {'min': int(counts.min()), 'mean': round(float(counts.mean()), 2),
                       'median': float(np.median(counts)), 'max': int(counts.max())},
        'per_unique_instruction': {'count': len(unique_prompts), 'mean': round(float(unique_counts.mean()), 2),
                                   'median': float(np.median(unique_counts))},
        'median_by_edit_type': {e: float(np.median(by_type[e])) for e in edit_types},
    }


def plot_agreement(records, out_dir):
    rater_counts = Counter(r['num_raters'] for r in records)
    agreement = Counter(r['observed_agreement'] for r in records if r['observed_agreement'] is not None)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.4))
    ax = axes[0]
    keys = sorted(rater_counts)
    bars = ax.bar([str(k) for k in keys], [rater_counts[k] for k in keys], color='#3E909B', width=0.62)
    bar_labels(ax, bars, [rater_counts[k] for k in keys])
    ax.set_xlabel('Raters per sample')
    ax.set_ylabel('Number of samples')
    ax.set_title('Raters per sample')

    ax = axes[1]
    keys = sorted(agreement)
    bars = ax.bar([f'{k:.2f}' for k in keys], [agreement[k] for k in keys], color='#E88574', width=0.62)
    bar_labels(ax, bars, [agreement[k] for k in keys])
    ax.set_xlabel('Observed agreement P_i')
    ax.set_ylabel('Number of samples')
    ax.set_title('Observed agreement')
    save_fig(fig, out_dir, '13_rater_agreement.png')

    return {
        'raters_per_sample': {str(k): v for k, v in sorted(rater_counts.items())},
        'observed_agreement': {f'{k:.4f}': v for k, v in sorted(agreement.items())},
        'mean_observed_agreement': round(float(np.mean([r['observed_agreement'] for r in records
                                                        if r['observed_agreement'] is not None])), 4),
    }


def plot_models(records, out_dir):
    appear = Counter(m for r in records for m in r['models'])
    wins = Counter(r['preferred_model'] for r in records if r['preferred_model'] not in (None, 'Tie'))
    ties = Counter(m for r in records if r['preferred_model'] in ('Tie', None) for m in r['models'])
    models = sorted(appear, key=lambda m: -wins[m] / appear[m])

    win = np.array([wins[m] / appear[m] * 100 for m in models])
    tie = np.array([ties[m] / appear[m] * 100 for m in models])
    lose = 100 - win - tie

    fig, ax = plt.subplots(figsize=(9, 3.3))
    y = np.arange(len(models))
    left = np.zeros(len(models))
    for vals, color, label in [(win, '#3E909B', 'Preferred'), (tie, '#F7CB8E', 'Tie / no majority'),
                               (lose, '#E88574', 'Not preferred')]:
        ax.barh(y, vals, left=left, color=color, height=0.62, label=label, edgecolor=BG_COLOR, linewidth=1.5)
        for yi, l, v in zip(y, left, vals):
            if v >= 6:
                ax.text(l + v / 2, yi, f'{v:.0f}%', ha='center', va='center', fontsize=8.5, color=TEXT_COLOR)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f'{m} ({appear[m]})' for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Share of pairs the model appears in (%)')
    ax.set_title('Model outcome under human labels')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=8.5)
    save_fig(fig, out_dir, '14_model_outcome.png')

    return {m: {'appearances': appear[m], 'preferred': wins[m], 'tie_or_no_majority': ties[m]} for m in models}


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Statistics and figures for AuraHumanLabel.jsonl.")
    parser.add_argument("--label_path", type=str, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output_dir", type=str, default=BASE_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    records = load_records(args.label_path)
    meta = load_meta_mapping()
    image_info = {r['image_id']: None for r in records}
    for r in records:
        if image_info[r['image_id']] is None:
            image_info[r['image_id']] = load_image_json(r['src_image'])

    print(f"Loaded {len(records)} samples, {len(image_info)} unique source images from {args.label_path}\n")

    summary = {
        'num_samples': len(records),
        'num_unique_images': len(image_info),
        'num_edit_types': len({r['edit_type'] for r in records}),
        'edit_type': plot_edit_type(records, args.output_dir),
        'metric': plot_metric(records, args.output_dir),
        'human_label_by_metric': plot_label_by_metric(records, args.output_dir),
        'edit_type_by_metric': plot_edit_type_by_metric(records, args.output_dir),
        'source_type': plot_source_type(records, meta, args.output_dir),
        'image_category': plot_category(records, meta, args.output_dir),
        'edited_objects': plot_edited_objects(records, image_info, args.output_dir),
        'resolution': plot_resolution(records, image_info, meta, args.output_dir),
        'instruction_word_count': plot_word_count(records, args.output_dir),
        'agreement': plot_agreement(records, args.output_dir),
        'models': plot_models(records, args.output_dir),
    }

    summary_path = os.path.join(args.output_dir, 'analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"Samples: {summary['num_samples']} | Unique images: {summary['num_unique_images']} | "
          f"Edit types: {summary['num_edit_types']}")
    print(f"Per metric: {summary['metric']}")
    print(f"Source type (samples): {summary['source_type']['samples']}")
    print(f"Edited objects: {summary['edited_objects']['samples_with_match']} samples matched, "
          f"{summary['edited_objects']['num_object_types']} object types")
    print(f"Instruction words: {summary['instruction_word_count']['per_sample']}")
    print(f"Megapixels: {summary['resolution']['megapixels']}")
    print("=" * 90)
    print(f"Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()
