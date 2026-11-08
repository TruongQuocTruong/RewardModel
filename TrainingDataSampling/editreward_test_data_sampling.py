import csv
from collections import Counter, defaultdict
import json
import os
import random

from editreward_data_sampling import (
    ALLOWED_MODELS,
    HUMANGT_FIELDS,
    INPUT_METADATA,
    categorize_instruction,
    sample_stratified,
    source_key,
)

# ================= CẤU HÌNH =================
SAMPLING_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling"
CREATION_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation"
EDITREWARD_ROOT = "/datastore/shared/datasets/EditReward-Data/extracted_data"

OUTPUT_TEST_JSONL = os.path.join(SAMPLING_DIR, "editreward_overall_humanGT_test_1k.jsonl")

# Cùng các trường như editreward_overall_humanGT_3k.jsonl (HUMANGT_FIELDS bỏ
# các cột định danh), giá trị ghi dạng chuỗi như file HumanGT của pipeline.
EXCLUDE_KEYS = {"key", "metadata_index", "pair_index"}
OUTPUT_FIELDS = [field for field in HUMANGT_FIELDS if field not in EXCLUDE_KEYS]

RANDOM_SEED = 42
TEST_SAMPLES = 1_000

# Tập test chỉ giữ các pair có người chọn rõ bên thắng (không lấy tie).
NON_TIE_VOTES = {"leftvote", "rightvote"}

# Mỗi schema file đã lấy mẫu: (trường unique key, trường edited image trái/phải).
# HumanGT/Unlabeled dùng đường dẫn tương đối, GemmaGT dùng đường dẫn tuyệt đối.
SCHEMAS = (
    ("unique_key", "left_output_image", "right_output_image"),
    ("sample_id", "edited_image_A", "edited_image_B"),
)


def existing_editreward_files() -> list[str]:
    """Mọi file EditReward đã lấy mẫu trước đó (HumanGT, GemmaGT, Unlabeled)."""
    files = []
    for directory, extension in ((SAMPLING_DIR, ".csv"), (CREATION_DIR, ".jsonl")):
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if (
                name.startswith("editreward_")
                and name.endswith(extension)
                and path != OUTPUT_TEST_JSONL
            ):
                files.append(path)
    return files


def normalize_image(raw_path: str) -> str:
    """Đưa đường dẫn ảnh về dạng tương đối như trong metadata.jsonl."""
    value = raw_path.strip()
    if value.startswith("file://"):
        value = value[7:]
    if os.path.isabs(value):
        relative = os.path.relpath(os.path.normpath(value), EDITREWARD_ROOT)
        if relative.startswith(".."):
            raise ValueError(f"Ảnh nằm ngoài EditReward-Data: {raw_path}")
        return relative
    return os.path.normpath(value)


def reserved_from_files(paths: list[str]) -> tuple[set[str], set[str], set[str]]:
    """Trả về unique key, edited image và source image của các file đã có."""
    unique_keys, edited_images, sources = set(), set(), set()
    for path in paths:
        file_keys, file_images, file_sources = set(), set(), set()
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith(".csv"):
                rows = csv.DictReader(handle)
            else:
                rows = (json.loads(line) for line in handle if line.strip())
            for line_number, row in enumerate(rows, start=1):
                schema = next((s for s in SCHEMAS if all(row.get(f) for f in s)), None)
                source = row.get("source_img") or row.get("source_image")
                if schema is None or not source:
                    raise KeyError(
                        f"{path} dòng {line_number}: không khớp schema đã biết "
                        f"(các trường: {sorted(row)})"
                    )
                key_field, left_field, right_field = schema
                file_keys.add(row[key_field])
                file_images.add(normalize_image(row[left_field]))
                file_images.add(normalize_image(row[right_field]))
                file_sources.add(normalize_image(source))
        print(
            f"  - {os.path.basename(path)}: {len(file_keys):,} pair, "
            f"{len(file_images):,} edited image, {len(file_sources):,} source_img"
        )
        unique_keys |= file_keys
        edited_images |= file_images
        sources |= file_sources
    return unique_keys, edited_images, sources


def write_jsonl(output_path: str, fields: list[str], records: list[dict]) -> None:
    """Ghi đúng các trường của file HumanGT jsonl trong pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in records:
            row = {field: str(item.get(field, "")) for field in fields}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ================= MAIN SAMPLING PIPELINE =================
def main():
    if os.path.exists(OUTPUT_TEST_JSONL):
        raise FileExistsError(f"Không ghi đè file test đã có: {OUTPUT_TEST_JSONL}")

    random.seed(RANDOM_SEED)

    used_files = existing_editreward_files()
    if not used_files:
        raise RuntimeError("Không tìm thấy file EditReward đã lấy mẫu trước đó.")
    print(f"[*] Loại edited image của {len(used_files)} file đã có:")
    reserved_keys, reserved_images, used_sources = reserved_from_files(used_files)
    print(
        f"[*] Tổng bị loại: {len(reserved_keys):,} unique_key, "
        f"{len(reserved_images):,} edited image"
    )

    # Giống Unlabeled Data: source image và instruction được phép trùng với các
    # tập trước, nhưng cả hai edited image của pair test phải chưa từng dùng.
    print("[*] Đang đọc metadata và lọc các cặp theo models yêu cầu...")
    stats = Counter()
    pools_by_category = defaultdict(list)
    with open(INPUT_METADATA, "r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if not line.strip():
                continue
            item = json.loads(line)
            stats["metadata"] += 1

            if (
                item.get("left_model") not in ALLOWED_MODELS
                or item.get("right_model") not in ALLOWED_MODELS
            ):
                stats["loại_do_model"] += 1
                continue
            if not source_key(item):
                stats["loại_do_thiếu_source"] += 1
                continue
            if item.get("unique_key") in reserved_keys:
                stats["loại_do_trùng_pair_đã_dùng"] += 1
                continue
            if (
                normalize_image(item["left_output_image"]) in reserved_images
                or normalize_image(item["right_output_image"]) in reserved_images
            ):
                stats["loại_do_trùng_edited_image"] += 1
                continue
            if str(item.get("vote_type", "")).strip().lower() not in NON_TIE_VOTES:
                stats["loại_do_tie"] += 1
                continue
            pools_by_category[categorize_instruction(item.get("instruction", ""))].append(item)
            stats["hợp_lệ"] += 1

    print(f"[*] Thống kê lọc: {dict(stats)}")
    print("[*] Phân bố theo Editing Type:")
    for category in sorted(pools_by_category):
        print(f"  + {category:<30}: {len(pools_by_category[category]):>6,} pair")

    print(f"\n[*] Đang lấy {TEST_SAMPLES:,} pair cho tập test...")
    sampled_records = sample_stratified(pools_by_category, TEST_SAMPLES, set())
    if len(sampled_records) != TEST_SAMPLES:
        raise RuntimeError(
            f"Không đủ pair cho tập test: cần {TEST_SAMPLES:,}, "
            f"chỉ lấy được {len(sampled_records):,}."
        )

    test_keys = [item["unique_key"] for item in sampled_records]
    test_images = {
        normalize_image(item[field])
        for item in sampled_records
        for field in ("left_output_image", "right_output_image")
    }
    assert len(set(test_keys)) == len(test_keys)
    assert not reserved_keys.intersection(test_keys)
    assert not reserved_images.intersection(test_images)
    for item in sampled_records:
        for field in ("source_img", "left_output_image", "right_output_image"):
            image_path = os.path.join(EDITREWARD_ROOT, item[field])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    write_jsonl(OUTPUT_TEST_JSONL, OUTPUT_FIELDS, sampled_records)

    test_sources = {normalize_image(source_key(item)) for item in sampled_records}
    print(f"[*] Vote: {dict(Counter(item['vote_type'] for item in sampled_records))}")
    print(
        "[*] Editing Type: "
        f"{dict(Counter(categorize_instruction(item['instruction']) for item in sampled_records))}"
    )
    print(
        f"[*] Source image: {len(test_sources):,} khác nhau, "
        f"{len(test_sources & used_sources):,} đã xuất hiện ở tập trước "
        "(được phép, edited image đều mới)"
    )
    print(f"[✓] Test: {len(sampled_records):,} pair -> {OUTPUT_TEST_JSONL}")
    print("[✓] HOÀN TẤT: không pair và edited image nào trùng HumanGT, GemmaGT, Unlabeled.")


if __name__ == "__main__":
    main()
