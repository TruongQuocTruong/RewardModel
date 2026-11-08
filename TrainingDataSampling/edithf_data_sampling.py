# -*- coding: utf-8 -*-

import os
import csv
import random
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

source_dir = "/datastore/shared/datasets/EditHF-1M/extracted_data/source_img"
edited_dir = "/datastore/shared/datasets/EditHF-1M/extracted_data/target_img"
scores_metadata_path = "/datastore/shared/datasets/EditHF-1M/extracted_data/scores/train.csv"
rank_metadata_path = "/datastore/shared/datasets/EditHF-1M/extracted_data/rank/train.csv"

script_dir = os.path.dirname(os.path.abspath(__file__))
exclude_3k_csv = os.path.join(script_dir, "edithf_score_train_for_gemma_3k.csv")
output_csv = os.path.join(script_dir, "edithf_scoreandrank_train_unlabeled_5k.csv")
TARGET_TOTAL_SAMPLES = 5000
MAX_PAIRS_PER_IMAGE = 8 

list_models = [
    "Nano", "Qwen", "Seed", "FLUX", "Dream", "followyourshape",
    "OmniGen2", "reflex", "cds", "infedit", "zone"
]
list_models_lower = {m.lower(): m for m in list_models}

edit_tasks = (
    "Add_filter", "Clothing_edit", "derain", "Modify_expressions",
    "Replace_another", "Zoom_in", "Change_attribute", "comic",
    "desnow", "shadow", "Add_scene", "Change_conditions",
    "Outpainting", "Adjust_lighting", "Convert_style", "Change_object",
    "deblur", "Relocate_object", "Strong_transfer", "background",
    "Remove_object", "low-light", "Replace_Accessories"
)

error_tasks_in_rank_train = set([
    "derain", "Modify_expressions", "Replace_another", "Change_attribute", 
    "desnow", "shadow", "Add_scene", "Change_object", "deblur", 
    "background", "Remove_object", "low-light"
])

rank_valid_tasks = set(edit_tasks) - error_tasks_in_rank_train

random.seed(42)

output_columns = [
    "sample_id",
    "source_image",
    "edit_instruction",
    "edited_image_A",
    "edited_image_B",
]


def pair_key(image_a, image_b):
    """Return an order-independent key for a pair of edited images."""
    return tuple(sorted([
        os.path.normcase(os.path.normpath(str(image_a))),
        os.path.normcase(os.path.normpath(str(image_b))),
    ]))


# Load existing 3k pairs to prevent any pair duplication
existing_3k_pairs = set()
existing_3k_source_images = set()
if os.path.isfile(exclude_3k_csv):
    df_3k = pd.read_csv(exclude_3k_csv)
    for _, row in df_3k.iterrows():
        existing_3k_pairs.add(pair_key(row["edited_image_A"], row["edited_image_B"]))
        existing_3k_source_images.add(os.path.normpath(str(row["source_image"])))
    print(f"Loaded {len(existing_3k_pairs)} pairs and {len(existing_3k_source_images)} source images from {exclude_3k_csv}")

if os.path.isfile(output_csv):
    existing_df = pd.read_csv(output_csv)
    missing_columns = [
        column for column in output_columns if column not in existing_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Existing output is missing columns: {missing_columns}"
        )
    existing_df = existing_df[output_columns].copy()
else:
    existing_df = pd.DataFrame(columns=output_columns)

if len(existing_df) > TARGET_TOTAL_SAMPLES:
    raise ValueError(
        f"Existing output has {len(existing_df)} rows, which exceeds "
        f"the target of {TARGET_TOTAL_SAMPLES}."
    )

if existing_df["sample_id"].duplicated().any():
    raise ValueError("Existing output contains duplicate sample_id values.")

existing_output_pair_keys = {
    pair_key(row["edited_image_A"], row["edited_image_B"])
    for _, row in existing_df.iterrows()
}
existing_sample_ids = set(existing_df["sample_id"].astype(str))

# All disallowed pairs: pairs in 3k + pairs in existing output
forbidden_pairs = existing_3k_pairs | existing_output_pair_keys

print(f"Existing samples preserved: {len(existing_df)}")

# Read metadata from scores/train.csv and rank/train.csv
data_by_task = defaultdict(
    lambda: defaultdict(lambda: {"instruction": "", "models": {}})
)


def load_metadata_file(filepath, allowed_tasks):
    if not os.path.isfile(filepath):
        print(f"Warning: metadata file not found: {filepath}")
        return 0, 0
    print(f"Reading metadata from {filepath}...")
    used = 0
    skipped = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        score_col = None
        for col in ["visual", "score", "scores", "reward", "rank_score", "val"]:
            if col in reader.fieldnames:
                score_col = col
                break
        if score_col is None:
            for col in reader.fieldnames:
                if col not in ["filename", "\ufefffilename", "instruction"]:
                    score_col = col
                    break
        print(f"  Using score column: '{score_col}'")

        for row in reader:
            fn = row.get("filename") or row.get("\ufefffilename")
            if not fn:
                skipped += 1
                continue
            parts = fn.strip().replace("\\", "/").split("/")
            if len(parts) != 3:
                skipped += 1
                continue
            model, task, img_name = parts
            if task not in allowed_tasks:
                skipped += 1
                continue
            if model.lower() not in list_models_lower:
                skipped += 1
                continue
            matched_model = list_models_lower[model.lower()]

            try:
                score = float(row[score_col])
            except (ValueError, TypeError):
                skipped += 1
                continue

            item = data_by_task[task][img_name]
            if not item["instruction"]:
                item["instruction"] = str(row.get("instruction", "")).strip()

            if matched_model not in item["models"]:
                item["models"][matched_model] = (fn, score)
                used += 1
            else:
                skipped += 1

    print(f"  Rows used: {used}, Rows skipped: {skipped}")
    return used, skipped


# 1. Load from scores/train.csv for all edit_tasks (allows reusing source images for underrepresented tasks)
load_metadata_file(scores_metadata_path, edit_tasks)

# 2. Load from rank/train.csv ONLY for rank_valid_tasks (strictly excludes error_tasks_in_rank_train)
load_metadata_file(rank_metadata_path, rank_valid_tasks)

print("\nGenerating candidate pairs...")
# Group candidate pairs by task and then by source image (img_name)
pairs_by_task_and_img = defaultdict(lambda: defaultdict(list))
total_pairs_generated = 0
tie_count = 0
invalid_path_count = 0
existing_pair_count = 0

for task in edit_tasks:
    task_images = data_by_task.get(task, {})

    for img_name, item in task_images.items():
        models_available = list(item["models"].keys())
        if len(models_available) < 2:
            continue

        instruction = item["instruction"]
        src_path = os.path.normpath(os.path.join(source_dir, task, img_name))
        img_id = os.path.splitext(img_name)[0]

        for model_a, model_b in combinations(models_available, 2):
            total_pairs_generated += 1

            rel_a, score_a = item["models"][model_a]
            rel_b, score_b = item["models"][model_b]

            if abs(score_a - score_b) < 1e-6:
                tie_count += 1
                continue

            edited_image_a = os.path.normpath(os.path.join(edited_dir, rel_a))
            edited_image_b = os.path.normpath(os.path.join(edited_dir, rel_b))

            if not (
                os.path.isfile(src_path)
                and os.path.isfile(edited_image_a)
                and os.path.isfile(edited_image_b)
            ):
                invalid_path_count += 1
                continue

            if pair_key(edited_image_a, edited_image_b) in forbidden_pairs:
                existing_pair_count += 1
                continue

            if random.random() > 0.5:
                model_a, model_b = model_b, model_a
                edited_image_a, edited_image_b = edited_image_b, edited_image_a
                score_a, score_b = score_b, score_a

            pairs_by_task_and_img[task][img_name].append({
                "img_id": img_id,
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "score_a": score_a,
                "score_b": score_b,
                "source_image": src_path,
                "edit_instruction": instruction,
                "edited_image_A": edited_image_a,
                "edited_image_B": edited_image_b
            })

print("\n" + "=" * 60)
print("CANDIDATE PAIRS STATISTICS")
print("=" * 60)
print(f"Total pairs generated: {total_pairs_generated}")
print(f"Ties excluded: {tie_count}")
print(f"Pairs with missing files: {invalid_path_count}")
print(f"Forbidden (existing 3k/output) pairs excluded: {existing_pair_count}")

candidate_tasks = [t for t in edit_tasks if t in pairs_by_task_and_img and len(pairs_by_task_and_img[t]) > 0]
for task in sorted(candidate_tasks):
    n_pairs = sum(len(pairs) for pairs in pairs_by_task_and_img[task].values())
    n_imgs = len(pairs_by_task_and_img[task])
    print(f"  {task:25s}: {n_pairs:5d} pairs across {n_imgs:4d} unique source images")
print("=" * 60)

# Shuffle pairs within each image
for task in candidate_tasks:
    for img_name in pairs_by_task_and_img[task]:
        random.shuffle(pairs_by_task_and_img[task][img_name])

samples_needed = TARGET_TOTAL_SAMPLES - len(existing_df)
print(f"\nSampling {samples_needed} additional samples across {len(candidate_tasks)} tasks...")

# Strategy for sampling with minimal source_image repetition:
# For each task, we maintain an image pool (list of img_names).
# We take 1 pair per image round-robin until target is met,
# with MAX_PAIRS_PER_IMAGE cap to prevent excessive repetition.
task_image_pools = {}
for task in candidate_tasks:
    imgs = list(pairs_by_task_and_img[task].keys())
    random.shuffle(imgs)
    task_image_pools[task] = imgs

task_img_indices = {task: 0 for task in candidate_tasks}

sampled_pairs = []
existing_task_counts = Counter(
    os.path.basename(os.path.dirname(os.path.normpath(path)))
    for path in existing_df["source_image"].astype(str)
)
added_task_counts = Counter()
image_usage_count = Counter()

active_tasks = list(candidate_tasks)

while len(sampled_pairs) < samples_needed and active_tasks:
    task = min(
        active_tasks,
        key=lambda name: (
            existing_task_counts[name] + added_task_counts[name],
            name,
        ),
    )

    pool = task_image_pools[task]
    if not pool:
        active_tasks.remove(task)
        continue

    pair_found = False
    start_idx = task_img_indices[task]
    for step in range(len(pool)):
        idx = (start_idx + step) % len(pool)
        img_name = pool[idx]
        src_path = os.path.normpath(os.path.join(source_dir, task, img_name))
        if image_usage_count[src_path] < MAX_PAIRS_PER_IMAGE and pairs_by_task_and_img[task][img_name]:
            chosen_pair = pairs_by_task_and_img[task][img_name].pop()
            sampled_pairs.append(chosen_pair)
            added_task_counts[task] += 1
            image_usage_count[src_path] += 1
            task_img_indices[task] = (idx + 1) % len(pool)
            pair_found = True
            break

    if not pair_found:
        active_tasks.remove(task)

if len(sampled_pairs) < samples_needed:
    print("\nWARNING:")
    print(
        f"Only sampled {len(sampled_pairs)} / {samples_needed} "
        "additional samples."
    )
    print("Reason: Total valid candidates insufficient with current MAX_PAIRS_PER_IMAGE.")

pairwise_data = []

for entry in sampled_pairs:
    img_id = entry["img_id"]
    task_clean = entry["task"].lower()
    model_a_clean = entry["model_a"].lower()
    model_b_clean = entry["model_b"].lower()

    sample_id = (
        f"{img_id}_{task_clean}_{model_a_clean}_{model_b_clean}"
    )

    if sample_id in existing_sample_ids:
        raise ValueError(f"Generated duplicate sample_id: {sample_id}")
    existing_sample_ids.add(sample_id)

    pairwise_data.append({
        "sample_id": sample_id,
        "source_image": entry["source_image"],
        "edit_instruction": entry["edit_instruction"],
        "edited_image_A": entry["edited_image_A"],
        "edited_image_B": entry["edited_image_B"]
    })

random.shuffle(pairwise_data)

new_df = pd.DataFrame(pairwise_data, columns=output_columns)
out_df = pd.concat([existing_df, new_df], ignore_index=True)

if out_df["sample_id"].duplicated().any():
    raise ValueError("Final output contains duplicate sample_id values.")

final_pair_keys = [
    pair_key(row["edited_image_A"], row["edited_image_B"])
    for _, row in out_df.iterrows()
]
if len(final_pair_keys) != len(set(final_pair_keys)):
    raise ValueError("Final output contains duplicate image pairs.")

# Verify no overlap with 3k pairs
overlap_pairs = set(final_pair_keys) & existing_3k_pairs
if overlap_pairs:
    raise ValueError(f"Found {len(overlap_pairs)} image pairs overlapping with 3k dataset!")

temporary_output = output_csv + ".tmp"
out_df.to_csv(temporary_output, index=False)
os.replace(temporary_output, output_csv)

print("\n" + "=" * 60)
print("COMPLETED")
print(f"Saved {len(out_df)} samples to: {output_csv}")
print("=" * 60)

print("\nTask distribution in output:")
task_series = pd.Series([
    os.path.basename(os.path.dirname(os.path.normpath(p)))
    for p in out_df["source_image"]
]).value_counts().sort_index()
print(task_series.to_string())

print(f"\nUnique source images in output: {out_df['source_image'].nunique()}")
print(f"Max repetitions for a single source image: {out_df['source_image'].value_counts().max()}")
print(f"Min repetitions for a single source image: {out_df['source_image'].value_counts().min()}")
reused_src_count = len(set(out_df["source_image"].astype(str).apply(os.path.normpath)) & existing_3k_source_images)
print(f"Source images reused from 3k: {reused_src_count}")
