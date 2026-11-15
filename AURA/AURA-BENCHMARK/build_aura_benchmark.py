# -*- coding: utf-8 -*-

"""
Build AURA-Benchmark.jsonl: every (image, edit instruction) of the AURA benchmark with the
edited images of the 5 editing models. No human labels: reward models score these samples
to rank the editing models.

Input:
- SOURCE_IMG_BENCH/<GenericConcepts|SpecialCases>/prompts/<image_id>_prompt.json
  ({"<image file>": [{"prompt", "edit_type", "special_case"}, ...]})
- EDITED_IMG_ROOT/edited_<model>/<folder>/<image_id>/<edit_type>_<image_id>.png

Output: one line per (image, edit_type) with fields
sample_id, image_id, edit_type, prompt, src_image, edited_images, models.
src_image is relative to SOURCE_IMG_BENCH, edited_images are relative to EDITED_IMG_ROOT;
edited_images[i] belongs to models[i].
"""

import argparse
import glob
import json
import os
from collections import Counter

SOURCE_IMG_BENCH = "/datastore/truongtq/ImageEditingBenchmark/AURABenchmark/AURA-BENCHMARK/"
EDITED_IMG_ROOT = "/datastore/truongtq/ImageEditingBenchmark/EditingEvaluation/OutputEdited/"

FOLDERS = ["GenericConcepts", "SpecialCases"]
MODELS = ["banana", "flux", "gpt_image", "longcat", "qwen_image_edit"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Build AURA-Benchmark.jsonl (no human labels).")
    parser.add_argument("--output_path", type=str, default=os.path.join(BASE_DIR, "AURA-Benchmark.jsonl"))
    parser.add_argument("--check_exists", action="store_true",
                        help="Check that every source/edited image exists under its root.")
    return parser.parse_args()


def main():
    args = parse_args()

    records = []
    seen = set()
    for folder in FOLDERS:
        # image_id -> real file name (some prompt files store the key without extension, e.g. "000566")
        image_files = {os.path.splitext(n)[0]: n for n in os.listdir(os.path.join(SOURCE_IMG_BENCH, folder, "images"))}
        prompt_files = sorted(glob.glob(os.path.join(SOURCE_IMG_BENCH, folder, "prompts", "*_prompt.json")))
        if not prompt_files:
            raise FileNotFoundError(f"No prompt files in {os.path.join(SOURCE_IMG_BENCH, folder, 'prompts')}")
        for prompt_file in prompt_files:
            with open(prompt_file, encoding='utf-8') as f:
                content = json.load(f)
            if len(content) != 1:
                raise ValueError(f"{prompt_file}: expected exactly one image key, got {list(content)}")
            image_key, edits = next(iter(content.items()))
            image_id = os.path.splitext(image_key)[0]
            if image_id not in image_files:
                raise FileNotFoundError(f"{prompt_file}: no image for '{image_key}' in {folder}/images")
            image_file = image_files[image_id]
            if image_key != image_file:
                print(f"Note: {prompt_file} key '{image_key}' -> image file '{image_file}'")

            for edit in edits:
                edit_type = edit['edit_type']
                sample_id = f"{image_id}_{edit_type}"
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id {sample_id} ({prompt_file})")
                seen.add(sample_id)
                records.append({
                    'sample_id': sample_id,
                    'image_id': image_id,
                    'edit_type': edit_type,
                    'prompt': edit['prompt'].strip(),
                    'src_image': f"{folder}/images/{image_file}",
                    'edited_images': [
                        f"edited_{model}/{folder}/{image_id}/{edit_type}_{image_id}.png" for model in MODELS
                    ],
                    'models': list(MODELS),
                })

    missing_files = []
    if args.check_exists:
        for rec in records:
            paths = [os.path.join(SOURCE_IMG_BENCH, rec['src_image'])]
            paths += [os.path.join(EDITED_IMG_ROOT, p) for p in rec['edited_images']]
            missing_files.extend(p for p in paths if not os.path.isfile(p))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Samples: {len(records)} | Unique images: {len({r['image_id'] for r in records})}")
    print(f"Per folder: {dict(Counter(r['src_image'].split('/')[0] for r in records))}")
    print(f"Edit types ({len({r['edit_type'] for r in records})}): "
          f"{dict(Counter(r['edit_type'] for r in records).most_common())}")
    if args.check_exists:
        print(f"Missing image files: {len(missing_files)}")
        for p in missing_files[:10]:
            print(f"  {p}")
    print(f"Saved to: {args.output_path}")


if __name__ == '__main__':
    main()
