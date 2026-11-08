import csv
import json
import os
import random

# Đường dẫn file input CSV
OVERALL_HUMANGT_CSV = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling/editreward_overall_humanGT_3k.csv"
UNLABELED_CSV = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling/editreward_unlabeled_10k.csv"

# Đường dẫn file output JSONL
OUTPUT_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation"
OVERALL_HUMANGT_JSONL = os.path.join(OUTPUT_DIR, "editreward_overall_humanGT_3k.jsonl")
UNLABELED_JSONL = os.path.join(OUTPUT_DIR, "editreward_unlabeled_10k.jsonl")



def csv_to_jsonl(
    csv_path: str,
    jsonl_path: str,
    num_samples: int = None, # For full samples
    seed: int = 42,
    exclude_keys: list = None,
):
    if exclude_keys is None:
        exclude_keys = []
    exclude_set = set(exclude_keys)

    random.seed(seed)
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if len(records) > num_samples:
        sampled_records = random.sample(records, num_samples)
    else:
        sampled_records = records

    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in sampled_records:
            # Lọc bỏ các cột nằm trong danh sách exclude_keys
            filtered_row = {k: v for k, v in row.items() if k not in exclude_set}
            f.write(json.dumps(filtered_row, ensure_ascii=False) + "\n")

    print(f"Đã tạo {jsonl_path} với {len(sampled_records)} samples.")


def main():
    # Loại bỏ các cột định danh không cần thiết trong file có human ground truth.
    exclude_keys_editreward = ["key", "metadata_index", "pair_index"]

    csv_to_jsonl(
        OVERALL_HUMANGT_CSV,
        OVERALL_HUMANGT_JSONL,
        num_samples=3000,
        seed=42,
        exclude_keys=exclude_keys_editreward,
    )

    # File unlabeled giữ nguyên các cột.
    csv_to_jsonl(UNLABELED_CSV, UNLABELED_JSONL, num_samples=10000, seed=42)


if __name__ == "__main__":
    main()
