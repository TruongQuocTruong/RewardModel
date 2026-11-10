import csv
from collections import defaultdict
import json
import os
import random
import re

# ================= CẤU HÌNH =================
INPUT_METADATA = "/datastore/shared/datasets/EditReward-Data/extracted_data/metadata.jsonl"

OUTPUT_HUMANGT_CSV = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling/editreward_overall_humanGT_GemmaGT_3k.csv"
# Tập Gemma 3k riêng (thiết kế cũ) không còn dùng: GemmaGT hiện được tạo trên đúng
# 3k cặp HumanGT. Bước lấy mẫu này được giữ lại chỉ để Unlabeled 10k lấy ra đúng như cũ
# (loại các source của tập này và dùng tiếp chuỗi random); file output không được giữ.
OUTPUT_FORGEMMA_CSV = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling/editreward_for_gemma_3k.csv"
OUTPUT_UNLABELED_CSV = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling/editreward_unlabeled_10k.csv"

RANDOM_SEED = 42

# Cả hai model trong một pair phải thuộc tập này.
ALLOWED_MODELS = {
    "qwen_edit",
    "step1x",
    "omnigen2",
    "ovis_u1",
    "flux_kontext",
    "bagel",
    "step1x_v2n",
    "flux_kontext_seed2025",
    "ovis_u1_seed2025",
    "step1x_v2n_seed42",
    "qwen_edit_seed2025",
    "bagel_think",
}

# Các trường dùng chung cho cả ba output. Những trường này không tiết lộ nhãn
# đánh giá của con người.
BASE_FIELDS = [
    "key",
    "unique_key",
    "metadata_index",
    "pair_index",
    "source_img",
    "instruction",
    "left_model",
    "left_output_image",
    "right_model",
    "right_output_image",
]

# Chỉ tập HumanGT chứa ground truth.
HUMANGT_FIELDS = [
    *BASE_FIELDS[:8],
    "left_overall_score",
    *BASE_FIELDS[8:],
    "right_overall_score",
    "vote_type",
]

SAMPLE_OUTPUTS = [
    ("HumanGT", OUTPUT_HUMANGT_CSV, 3_000, HUMANGT_FIELDS),
    ("Gemma", OUTPUT_FORGEMMA_CSV, 3_000, BASE_FIELDS),
    ("Unlabeled", OUTPUT_UNLABELED_CSV, 10_000, BASE_FIELDS),
]


# ================= HÀM PHÂN LOẠI EDITING TYPE =================
def categorize_instruction(instruction: str) -> str:
    """Phân loại instruction vào 7 nhóm tác vụ tương tự paper EditReward."""
    inst = instruction.lower().strip()

    if re.search(
        r"\b(remove|delete|erase|eliminate|clear out|take away|get rid of|drop)\b",
        inst,
    ):
        return "Subject Removal"
    if re.search(
        r"\b(add|insert|put|place|include|introduce|give him|give her|with a)\b",
        inst,
    ):
        return "Subject Addition"
    if re.search(
        r"\b(style|sketch|painting|watercolor|oil paint|anime|cartoon|comic|drawing|illustration|woodblock|cyberpunk|render|pixel art)\b",
        inst,
    ):
        return "Stylization & Creative"
    if re.search(
        r"\b(color|recolor|tint|black and white|monochrome|red|blue|green|yellow|purple|orange|pink|darken|brighten)\b",
        inst,
    ):
        return "Color Editing"
    if re.search(
        r"\b(background|backdrop|sky|weather|sunset|autumn|winter|spring|summer|snow|rain|beach|mountain|indoor|outdoor)\b",
        inst,
    ):
        return "Background Editing"
    if re.search(
        r"\b(material|texture|metal|wooden|wood|glass|plastic|leather|golden|silver|cloth|wear|shirt|hat|shoes|glasses|hair|expression|smile)\b",
        inst,
    ):
        return "Attribute & Material Editing"
    return "General & Other"


def source_key(item: dict) -> str:
    """Trả về khoá source image để kiểm soát trùng lặp giữa các output."""
    return item.get("source_img", "")


def sample_stratified(
    pools_by_category: dict[str, list[dict]], total_samples: int, reserved_sources: set[str]
) -> list[dict]:
    """Lấy mẫu phân tầng, không dùng source_img đã dành cho output trước đó.

    Một source image được phép có nhiều pair trong *cùng* output. Ngay khi một
    source được dùng trong output hiện tại, toàn bộ pair của nó sẽ được loại ở
    các output theo sau bằng cách cập nhật ``reserved_sources`` ở hàm main.
    """
    categories = sorted(pools_by_category)
    if not categories:
        return []

    k_per_category, remainder = divmod(total_samples, len(categories))
    selected = []

    for index, category in enumerate(categories):
        quota = k_per_category + (index < remainder)
        candidates = [
            item
            for item in pools_by_category[category]
            if source_key(item) not in reserved_sources
        ]
        selected.extend(random.sample(candidates, min(quota, len(candidates))))

    # Nếu một category không đủ quota, bù từ mọi category còn lại. Record đã
    # được chọn không được lặp, nhưng source trong output hiện tại có thể lặp.
    shortage = total_samples - len(selected)
    if shortage:
        selected_ids = {id(item) for item in selected}
        remaining = [
            item
            for items in pools_by_category.values()
            for item in items
            if source_key(item) not in reserved_sources and id(item) not in selected_ids
        ]
        selected.extend(random.sample(remaining, min(shortage, len(remaining))))

    random.shuffle(selected)
    return selected


def write_csv(output_path: str, fields: list[str], records: list[dict]) -> None:
    """Ghi đúng các cột được phép của mỗi output."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in records:
            writer.writerow({field: item.get(field, "") for field in fields})


# ================= MAIN SAMPLING PIPELINE =================
def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(INPUT_METADATA):
        print(f"[!] Không tìm thấy file: {INPUT_METADATA}")
        return

    print("[*] Đang đọc metadata và lọc các cặp theo models yêu cầu...")
    pools_by_category = defaultdict(list)
    total_valid = 0
    missing_source = 0

    with open(INPUT_METADATA, "r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if not line.strip():
                continue
            item = json.loads(line)

            if (
                item.get("left_model") not in ALLOWED_MODELS
                or item.get("right_model") not in ALLOWED_MODELS
            ):
                continue
            if not source_key(item):
                missing_source += 1
                continue

            pools_by_category[categorize_instruction(item.get("instruction", ""))].append(
                item
            )
            total_valid += 1

    print(f"[*] Tổng số pair hợp lệ: {total_valid:,}")
    if missing_source:
        print(f"[!] Bỏ qua {missing_source:,} pair không có source_img.")
    print("[*] Phân bố theo Editing Type:")
    for category in sorted(pools_by_category):
        print(f"  + {category:<30}: {len(pools_by_category[category]):>6,} pair")

    # reserved_sources được cập nhật sau từng tập, vì vậy một source image sẽ
    # thuộc tối đa một trong ba CSV. Đây là ràng buộc mạnh hơn việc chỉ giảm số
    # lần lặp: không có source_img nào giao nhau giữa các output.
    reserved_sources = set()
    sampled_source_sets = []

    for dataset_name, output_path, sample_count, fields in SAMPLE_OUTPUTS:
        print(f"\n[*] Đang lấy {sample_count:,} pair cho tập {dataset_name}...")
        sampled_records = sample_stratified(
            pools_by_category, sample_count, reserved_sources
        )
        if len(sampled_records) != sample_count:
            raise RuntimeError(
                f"Không đủ pair cho {dataset_name}: cần {sample_count:,}, "
                f"chỉ lấy được {len(sampled_records):,}."
            )

        source_images = {source_key(item) for item in sampled_records}
        assert not reserved_sources.intersection(source_images)
        reserved_sources.update(source_images)
        sampled_source_sets.append(source_images)

        write_csv(output_path, fields, sampled_records)
        print(
            f"[✓] {dataset_name}: {len(sampled_records):,} pair, "
            f"{len(source_images):,} source image -> {output_path}"
        )

    assert all(
        sampled_source_sets[i].isdisjoint(sampled_source_sets[j])
        for i in range(len(sampled_source_sets))
        for j in range(i + 1, len(sampled_source_sets))
    )
    print("\n[✓] HOÀN TẤT: source_img không trùng giữa ba file output.")


if __name__ == "__main__":
    main()
