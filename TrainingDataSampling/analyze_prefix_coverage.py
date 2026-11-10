#!/usr/bin/env python3
"""Bước 1: đo độ đa dạng của N cặp đầu (mặc định 912 và 1008) trong train_order của Inspector Baseline.

Chỉ phân tích, không chọn lại tập.  Chạy qua Slurm (cần GPU nếu chưa có vector đã lưu).

Vùng chọn: toàn bộ EditReward-Data (170772 cặp).  Mỗi N cặp đầu hiện tại được so với
R tập ngẫu nhiên cùng cỡ lấy từ toàn bộ dữ liệu.

Đặc trưng:
- Ảnh nguồn: chủ đề (gán nhãn không cần huấn luyện bằng Qwen3-VL-Embedding: so vector ảnh với
  vector câu mô tả nhãn), cụm k-means của vector ảnh, số điểm ảnh, tỉ lệ khung hình (đọc từ phần
  đầu file ảnh).  Thể loại ảnh và "có chữ" đã bỏ vì kiểm tra ảnh mẫu thấy gán nhãn sai nhiều.
- Instruction: loại chỉnh sửa (gán nhãn không cần huấn luyện), cụm k-means của vector
  instruction, số từ, số thao tác (đếm gần đúng qua "and", dấu phẩy, "then", câu).
- Cặp: model, cặp model, vote, hiệu điểm con người, đánh đổi giữa hai điểm con của con người,
  và từ GemmaGT (quy tắc đa số như train_student_hardlabel.py): độ nhất quán 3 lần chạy,
  đồng ý với HumanGT, đánh đổi giữa các tiêu chí con.

Vector được lưu lại trong thư mục output để bước chọn lại dùng lại.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

EDITREWARD_ROOT = "/datastore/shared/datasets/EditReward-Data/extracted_data"
DEFAULT_METADATA = f"{EDITREWARD_ROOT}/metadata.jsonl"
CREATION_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation"
SAMPLING_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling"
DEFAULT_GEMMA_FULL = f"{CREATION_DIR}/editreward_overall_finegrained_gemmaGT_full.jsonl"
DEFAULT_TRAIN_ORDER = (
    "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training/"
    "inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/train_order.jsonl"
)
DEFAULT_MODEL = "/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Embedding-2B"
EMBEDDER_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/src/models"
DEFAULT_OUTPUT_DIR = f"{SAMPLING_DIR}/full_data_analysis/prefix_coverage"

VOTE_TO_CLASS = {"leftvote": "A", "rightvote": "B", "tie": "Tie", "tievote": "Tie"}

# ----------------------------- nhãn gán không cần huấn luyện -----------------------------
IMAGE_QUERY_INSTRUCTION = "Find an image that matches the given description."
TOPIC_PROMPTS = {
    "person_portrait": "a portrait of one person",
    "people_group": "a group of several people",
    "animal": "an animal",
    "landscape_nature": "a natural landscape such as mountains, sea or forest",
    "city_architecture": "a city street or building architecture",
    "indoor_room": "an indoor room or interior scene",
    "food_drink": "food or drinks",
    "object_product": "a single object or product",
    "vehicle": "a car, bike, boat or other vehicle",
    "plant_flower": "plants or flowers",
    "text_heavy": "mostly written text, signs or letters",
}
INSTRUCTION_TYPE_PROMPTS = {
    "add_object": "Add a new object or person to the image.",
    "remove_object": "Remove an object or person from the image.",
    "replace_object": "Replace one object in the image with a different object.",
    "color_change": "Change the color of something in the image.",
    "material_texture": "Change the material or texture of an object.",
    "style_transfer": "Change the whole image into an artistic style such as painting, sketch or anime.",
    "background_change": "Change or replace the background of the image.",
    "text_edit": "Add, change or translate the text written in the image.",
    "pose_expression_action": "Change the pose, facial expression or action of a person or animal.",
    "appearance_clothing": "Change the clothes, hair or appearance of a person.",
    "lighting_weather_time": "Change the lighting, weather, season or time of day.",
    "camera_view_composition": "Change the camera viewpoint, zoom, crop or composition.",
    "enhance_restore": "Improve, restore, sharpen or colorize the photo quality.",
}

WORD_EDGES = [3, 6, 10, 15, 25]
WORD_LABELS = ["1-3", "4-6", "7-10", "11-15", "16-25", ">=26"]
OPS_EDGES = [1, 2, 3]
OPS_LABELS = ["1", "2", "3", ">=4"]
MP_EDGES = [0.25, 0.5, 1.0, 2.0, 4.0]
MP_LABELS = ["<0.25MP", "0.25-0.5MP", "0.5-1MP", "1-2MP", "2-4MP", ">=4MP"]
ASPECT_EDGES = [0.6, 0.8, 1.25, 1.67]
ASPECT_LABELS = ["tall<0.6", "portrait0.6-0.8", "square0.8-1.25", "landscape1.25-1.67", "wide>=1.67"]
ABS_DIFF_EDGES = [0, 1, 2, 3, 5]
ABS_DIFF_LABELS = ["0", "1", "2", "3", "4-5", ">=6"]


def bin_value(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def bin_ratio(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    """Như bin_value nhưng cận trên không thuộc nhóm (dùng cho số thực liên tục)."""
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_operations(instruction: str) -> int:
    """Đếm gần đúng số thao tác: tách theo câu, 'and', 'then', dấu phẩy, dấu chấm phẩy."""
    text = instruction.strip().lower()
    parts = re.split(r"(?:[.;!?]\s+|,\s*|\band\b|\bthen\b|\balso\b|\bwhile\b)", text)
    return max(1, sum(1 for part in parts if len(part.split()) >= 2))


# ----------------------------- GemmaGT -----------------------------
def gemma_features(record: dict) -> Dict[str, str]:
    """Quy tắc đa số như train_student_hardlabel.py: tiêu chí liên quan khi >= 2/3 lần chạy
    đánh dấu liên quan; lựa chọn = đa số tuyệt đối (> 50%) của các lần chạy liên quan, không có
    thì Tie.  Tổng hợp overall của Gemma (quy ước riêng của file này): đếm số tiêu chí liên quan
    chọn A và chọn B, nhiều hơn thì thắng, bằng nhau thì Tie."""
    runs = record["runs"]
    metrics = list(runs[0]["metric_groundtruth"].keys())
    n_a = n_b = n_relevant = n_unanimous = 0
    for metric in metrics:
        applicable = [run["metric_groundtruth"][metric] for run in runs
                      if int(run["metric_groundtruth"][metric]["targeted_metric"]) == 1]
        if len(applicable) / len(runs) < 2.0 / 3.0 - 1e-6:
            continue
        n_relevant += 1
        choices = Counter(item["choice"] for item in applicable)
        top, top_count = choices.most_common(1)[0]
        choice = top if top_count / len(applicable) > 0.5 else "Tie"
        if len(choices) == 1 and len(applicable) == len(runs):
            n_unanimous += 1
        n_a += choice == "A"
        n_b += choice == "B"
    overall = "A" if n_a > n_b else "B" if n_b > n_a else "Tie"
    consistency = n_unanimous / n_relevant if n_relevant else 0.0
    return {
        "gemma_overall": overall,
        "gemma_tradeoff": "yes" if (n_a > 0 and n_b > 0) else "no",
        "gemma_relevant_metrics": str(n_relevant),
        "gemma_consistency": bin_ratio(consistency, [0.5, 0.75, 1.0], ["<0.5", "0.5-0.75", "0.75-1", "1"]),
    }


# ----------------------------- vector -----------------------------
def load_embedder(model_path: str):
    import torch
    sys.path.insert(0, EMBEDDER_DIR)
    from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402
    return Qwen3VLEmbedder(model_name_or_path=model_path, torch_dtype=torch.bfloat16)


def embed(embedder, inputs: List[dict], batch_size: int, label: str) -> np.ndarray:
    import torch
    out = []
    start = time.time()
    for begin in range(0, len(inputs), batch_size):
        batch = inputs[begin:begin + batch_size]
        with torch.no_grad():
            vectors = embedder.process(batch)
        out.append(vectors.float().cpu().numpy())
        done = begin + len(batch)
        if done % (batch_size * 50) < batch_size or done == len(inputs):
            print(f"  [{label}] {done}/{len(inputs)}  {time.time() - start:.0f}s", flush=True)
    return np.concatenate(out, axis=0)


def cached_embeddings(path: Path, compute) -> np.ndarray:
    if path.exists():
        print(f"[*] Dùng lại vector đã lưu: {path}")
        return np.load(path).astype(np.float32)
    vectors = compute()
    np.save(path, vectors.astype(np.float16))
    return vectors.astype(np.float32)


def zero_shot(item_vectors: np.ndarray, label_vectors: np.ndarray, labels: List[str]) -> List[str]:
    scores = item_vectors @ label_vectors.T
    return [labels[i] for i in scores.argmax(axis=1)]


def nearest_examples(item_vectors: np.ndarray, label_vectors: np.ndarray, labels: List[str],
                     assigned: List[str], names: List[str], per_label: int) -> Dict[str, List[str]]:
    """Với mỗi nhãn: các phần tử được gán nhãn đó và có điểm cao nhất (để kiểm tra bằng mắt)."""
    scores = item_vectors @ label_vectors.T
    examples = {}
    for j, label in enumerate(labels):
        idx = [i for i, a in enumerate(assigned) if a == label]
        idx.sort(key=lambda i: -scores[i, j])
        examples[label] = [names[i] for i in idx[:per_label]]
    return examples


def contact_sheet(paths: List[str], out_path: Path, thumb: int = 160) -> None:
    from PIL import Image
    if not paths:
        return
    cols = min(8, len(paths))
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for k, rel in enumerate(paths):
        try:
            with Image.open(f"{EDITREWARD_ROOT}/{rel}") as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                sheet.paste(image, ((k % cols) * thumb, (k // cols) * thumb))
        except Exception as exc:  # ảnh lỗi: để ô trắng
            print(f"  [!] Không mở được {rel}: {exc}")
    sheet.save(out_path, quality=80)


# ----------------------------- thống kê -----------------------------
def normalize(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total else {}


def tvd(p: Counter, q: Counter) -> float:
    pn, qn = normalize(p), normalize(q)
    return 0.5 * sum(abs(pn.get(k, 0.0) - qn.get(k, 0.0)) for k in set(pn) | set(qn))


def norm_entropy(counter: Counter, n_values: int) -> float:
    """Entropy chuẩn hoá theo số giá trị có trong vùng chọn: 1 = đều tuyệt đối."""
    probs = [v for v in normalize(counter).values() if v > 0]
    if n_values <= 1:
        return 1.0
    return -sum(p * math.log(p) for p in probs) / math.log(n_values)


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def feature_counter(rows: Sequence[dict], name: str) -> Counter:
    if name == "model":
        counter = Counter()
        for row in rows:
            counter[row["left_model"]] += 1
            counter[row["right_model"]] += 1
        return counter
    return Counter(row[name] for row in rows)


def set_stats(rows: Sequence[dict], features: Sequence[str], pool_counters: Dict[str, Counter]) -> Dict[str, dict]:
    stats = {}
    for name in features:
        counter = feature_counter(rows, name)
        pool = pool_counters[name]
        stats[name] = {
            "tvd_vs_pool": tvd(counter, pool),
            "norm_entropy": norm_entropy(counter, len(pool)),
            "values_covered": sum(1 for k in pool if k in counter),
            "values_in_pool": len(pool),
        }
    stats["_unique_sources"] = len({row["source_img"] for row in rows})
    stats["_unique_instructions"] = len({row["instruction"] for row in rows})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--gemma-full", default=DEFAULT_GEMMA_FULL)
    parser.add_argument("--train-order", default=DEFAULT_TRAIN_ORDER)
    parser.add_argument("--first-n", default="912,1008", help="Các mốc N, cách nhau bởi dấu phẩy")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kmeans-k", default="50,100,200")
    parser.add_argument("--examples-per-label", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if (out_dir / "prefix_coverage.json").exists():
        raise FileExistsError(f"Kết quả đã tồn tại, không ghi đè: {out_dir}")
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (out_dir / "label_examples").mkdir(parents=True, exist_ok=True)
    ks = [int(k) for k in args.kmeans_k.split(",")]

    # ---------- dữ liệu ----------
    print("[*] Đọc metadata ...", flush=True)
    rows = read_jsonl(args.metadata)
    by_id = {row["unique_key"]: row for row in rows}
    assert len(by_id) == len(rows), "unique_key bị trùng"

    first_ns = [int(n) for n in args.first_n.split(",")]
    order_ids = [r["sample_id"] for r in read_jsonl(args.train_order)][: max(first_ns)]
    missing = [sid for sid in order_ids if sid not in by_id]
    assert not missing, f"{len(missing)} cặp của train_order không có trong metadata"

    print("[*] Đọc GemmaGT toàn bộ (chỉ giữ đặc trưng tóm tắt) ...", flush=True)
    gemma = {}
    with open(args.gemma_full, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                gemma[record["sample_id"]] = gemma_features(record)
    missing_gemma = sum(1 for row in rows if row["unique_key"] not in gemma)
    assert missing_gemma == 0, f"{missing_gemma} cặp không có GemmaGT"

    # ---------- ảnh nguồn ----------
    from PIL import Image
    sources = sorted({row["source_img"] for row in rows})
    instructions = sorted({row["instruction"] for row in rows})
    print(f"[*] {len(rows)} cặp, {len(sources)} ảnh nguồn, {len(instructions)} instruction khác nhau", flush=True)

    source_feat: Dict[str, dict] = {}
    for rel in sources:
        try:
            with Image.open(f"{EDITREWARD_ROOT}/{rel}") as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError(f"Không đọc được kích thước ảnh nguồn {rel}: {exc}") from exc
        source_feat[rel] = {
            "width": width, "height": height,
            "pixel_bin": bin_ratio(width * height / 1e6, MP_EDGES, MP_LABELS),
            "aspect_bin": bin_ratio(width / height, ASPECT_EDGES, ASPECT_LABELS),
        }

    # ---------- vector ----------
    embedder = None

    def get_embedder():
        nonlocal embedder
        if embedder is None:
            print(f"[*] Nạp model {args.model_path}", flush=True)
            embedder = load_embedder(args.model_path)
        return embedder

    emb_dir = out_dir / "embeddings"
    image_vec = cached_embeddings(emb_dir / "source_images.npy", lambda: embed(
        get_embedder(), [{"image": f"{EDITREWARD_ROOT}/{rel}"} for rel in sources], args.batch_size, "ảnh nguồn"))
    instr_vec = cached_embeddings(emb_dir / "instructions.npy", lambda: embed(
        get_embedder(), [{"text": text} for text in instructions], args.batch_size * 4, "instruction"))

    def text_vectors(prompts: Dict[str, str], instruction: str | None) -> np.ndarray:
        return embed(get_embedder(), [{"text": t, "instruction": instruction} for t in prompts.values()],
                     args.batch_size * 4, "nhãn")

    topic_vec = text_vectors(TOPIC_PROMPTS, IMAGE_QUERY_INSTRUCTION)
    itype_vec = text_vectors(INSTRUCTION_TYPE_PROMPTS, None)
    with open(emb_dir / "index.json", "w", encoding="utf-8") as handle:
        json.dump({"source_images": sources, "instructions": instructions,
                   "model_path": args.model_path, "dtype_saved": "float16"}, handle, ensure_ascii=False)

    topic = zero_shot(image_vec, topic_vec, list(TOPIC_PROMPTS))
    itype = zero_shot(instr_vec, itype_vec, list(INSTRUCTION_TYPE_PROMPTS))

    # ảnh mẫu cho từng nhãn để kiểm tra bằng mắt
    examples = {
        "topic": nearest_examples(image_vec, topic_vec, list(TOPIC_PROMPTS), topic, sources, args.examples_per_label),
        "instruction_type": nearest_examples(instr_vec, itype_vec, list(INSTRUCTION_TYPE_PROMPTS), itype,
                                             instructions, args.examples_per_label),
    }
    for group in ("topic",):
        for label, paths in examples[group].items():
            contact_sheet(paths, out_dir / "label_examples" / f"{group}__{label}.jpg")
    # ngoài ảnh có điểm cao nhất, lấy thêm ảnh ngẫu nhiên mỗi nhãn (điểm cao nhất dễ đúng hơn trung bình)
    rng_examples = random.Random(args.seed)
    random_examples = {}
    for group, assigned in (("topic", topic),):
        random_examples[group] = {}
        for label in sorted(set(assigned)):
            members = [sources[i] for i, a in enumerate(assigned) if a == label]
            pick = rng_examples.sample(members, min(args.examples_per_label, len(members)))
            random_examples[group][label] = pick
            contact_sheet(pick, out_dir / "label_examples" / f"{group}__{label}__random.jpg")
    random_examples["instruction_type"] = {}
    for label in sorted(set(itype)):
        members = [instructions[i] for i, a in enumerate(itype) if a == label]
        random_examples["instruction_type"][label] = rng_examples.sample(
            members, min(args.examples_per_label, len(members)))

    # k-means
    from sklearn.cluster import KMeans
    image_clusters, instr_clusters = {}, {}
    for k in ks:
        print(f"[*] k-means k={k}", flush=True)
        image_clusters[k] = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(image_vec)
        instr_clusters[k] = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(instr_vec)

    src_index = {rel: i for i, rel in enumerate(sources)}
    ins_index = {text: i for i, text in enumerate(instructions)}

    # ---------- đặc trưng từng cặp ----------
    pair_rows = []
    for row in rows:
        si, ii = src_index[row["source_img"]], ins_index[row["instruction"]]
        left, right = float(row["left_overall_score"]), float(row["right_overall_score"])
        sub = [row.get(k) for k in ("left_instruction_following_score", "right_instruction_following_score",
                                     "left_quality_aesthetic_score", "right_quality_aesthetic_score")]
        if any(v is None or v == "" for v in sub):
            tradeoff = "unknown"
        else:
            d_if, d_qa = float(sub[0]) - float(sub[1]), float(sub[2]) - float(sub[3])
            tradeoff = "yes" if d_if * d_qa < 0 else "no"
        vote = VOTE_TO_CLASS[row["vote_type"]]
        g = gemma[row["unique_key"]]
        feat = {
            "unique_key": row["unique_key"],
            "source_img": row["source_img"],
            "instruction": row["instruction"],
            "left_model": row["left_model"],
            "right_model": row["right_model"],
            "image_topic": topic[si],
            "image_pixels": source_feat[row["source_img"]]["pixel_bin"],
            "image_aspect": source_feat[row["source_img"]]["aspect_bin"],
            "instruction_type": itype[ii],
            "instruction_words": bin_value(len(row["instruction"].split()), WORD_EDGES, WORD_LABELS),
            "instruction_operations": bin_value(count_operations(row["instruction"]), OPS_EDGES, OPS_LABELS),
            "model_pair": " | ".join(sorted((row["left_model"], row["right_model"]))),
            "vote": vote,
            "human_abs_score_diff": bin_value(abs(left - right), ABS_DIFF_EDGES, ABS_DIFF_LABELS),
            "human_subscore_tradeoff": tradeoff,
            "gemma_agrees_human": "yes" if g["gemma_overall"] == vote else "no",
            **g,
        }
        for k in ks:
            feat[f"image_cluster_k{k}"] = str(int(image_clusters[k][si]))
            feat[f"instruction_cluster_k{k}"] = str(int(instr_clusters[k][ii]))
        pair_rows.append(feat)

    with open(out_dir / "pair_features.jsonl", "w", encoding="utf-8") as handle:
        for feat in pair_rows:
            handle.write(json.dumps(feat, ensure_ascii=False) + "\n")

    features = ["image_topic", "image_pixels", "image_aspect"]
    features += [f"image_cluster_k{k}" for k in ks]
    features += ["instruction_type", "instruction_words", "instruction_operations"]
    features += [f"instruction_cluster_k{k}" for k in ks]
    features += ["model", "model_pair", "vote", "human_abs_score_diff", "human_subscore_tradeoff",
                 "gemma_agrees_human", "gemma_consistency", "gemma_tradeoff", "gemma_relevant_metrics"]

    pool = pair_rows
    pool_counters = {name: feature_counter(pool, name) for name in features}
    feat_by_id = {f["unique_key"]: f for f in pair_rows}

    def summary(stats_list: List[dict], key: str, metric: str | None = None) -> dict:
        values = [s[key] if metric is None else s[key][metric] for s in stats_list]
        return {"mean": statistics_mean(values), "p05": percentile(values, 0.05), "p95": percentile(values, 0.95)}

    prefixes = {}
    for n in first_ns:
        prefix = [feat_by_id[sid] for sid in order_ids[:n]]
        prefix_stats = set_stats(prefix, features, pool_counters)
        rng = random.Random(args.seed)
        random_stats = [set_stats(rng.sample(pool, n), features, pool_counters)
                        for _ in range(args.random_repeats)]
        prefixes[str(n)] = {
            "pairs": n,
            "unique_sources": prefix_stats["_unique_sources"],
            "unique_instructions": prefix_stats["_unique_instructions"],
            "random_unique_sources": summary(random_stats, "_unique_sources"),
            "random_unique_instructions": summary(random_stats, "_unique_instructions"),
            "comparison": {
                name: {
                    "prefix": prefix_stats[name],
                    "random": {m: summary(random_stats, name, m)
                               for m in ("tvd_vs_pool", "norm_entropy", "values_covered")},
                    "share_prefix": normalize(feature_counter(prefix, name)),
                }
                for name in features
            },
        }

    result = {
        "inputs": vars(args),
        "counts": {
            "pairs_full": len(rows),
            "sources_full": len(sources),
            "instructions_full": len(instructions),
            "instructions_per_source_full": dict(Counter(
                Counter(row["source_img"] for row in
                        {(r["source_img"], r["instruction"]): r for r in rows}.values()).values())),
        },
        "image_size_full": {
            "width_min": min(v["width"] for v in source_feat.values()),
            "width_max": max(v["width"] for v in source_feat.values()),
            "height_min": min(v["height"] for v in source_feat.values()),
            "height_max": max(v["height"] for v in source_feat.values()),
        },
        "share_full": {name: normalize(pool_counters[name]) for name in features},
        "prefixes": prefixes,
        "label_examples_top": examples,
        "label_examples_random": random_examples,
    }
    with open(out_dir / "prefix_coverage.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    text = format_report(result, features)
    (out_dir / "prefix_coverage.txt").write_text(text, encoding="utf-8")
    print(text)


def format_report(result: dict, features: Sequence[str]) -> str:
    c = result["counts"]
    size = result["image_size_full"]
    lines = ["=" * 110]
    lines.append(f"Toàn bộ: {c['pairs_full']} cặp, {c['sources_full']} ảnh nguồn, {c['instructions_full']} instruction khác nhau")
    lines.append(f"Số instruction khác nhau trên mỗi ảnh nguồn (số ảnh nguồn): {c['instructions_per_source_full']}")
    lines.append(f"Kích thước ảnh nguồn: rộng {size['width_min']}-{size['width_max']}, cao {size['height_min']}-{size['height_max']}")
    lines.append("TVD = khoảng cách phân phối so với toàn bộ (0 = giống). Entropy chuẩn hoá: 1 = đều tuyệt đối giữa các giá trị.")
    for key, info in result["prefixes"].items():
        rs, ri = info["random_unique_sources"], info["random_unique_instructions"]
        lines.append("")
        lines.append("=" * 110)
        lines.append(f"{key} cặp đầu của train_order")
        lines.append(f"  Ảnh nguồn khác nhau: {info['unique_sources']} (ngẫu nhiên TB {rs['mean']:.1f}, p05-p95 {rs['p05']}-{rs['p95']})")
        lines.append(f"  Instruction khác nhau: {info['unique_instructions']} (ngẫu nhiên TB {ri['mean']:.1f}, p05-p95 {ri['p05']}-{ri['p95']})")
        lines.append(f"{'Đặc trưng':<28}{'TVD':>8}{'TVD ngẫu nhiên TB [p05,p95]':>30}{'Entropy':>9}"
                     f"{'Entropy ngẫu nhiên TB [p05,p95]':>34}{'Phủ':>10}{'Phủ ngẫu nhiên TB':>19}")
        for name in features:
            p, r = info["comparison"][name]["prefix"], info["comparison"][name]["random"]
            lines.append(
                f"{name:<28}{p['tvd_vs_pool']:>8.4f}"
                f"{r['tvd_vs_pool']['mean']:>12.4f} [{r['tvd_vs_pool']['p05']:.4f},{r['tvd_vs_pool']['p95']:.4f}]"
                f"{p['norm_entropy']:>9.4f}"
                f"{r['norm_entropy']['mean']:>16.4f} [{r['norm_entropy']['p05']:.4f},{r['norm_entropy']['p95']:.4f}]"
                f"{str(p['values_covered']) + '/' + str(p['values_in_pool']):>10}{r['values_covered']['mean']:>19.1f}")
    keys = list(result["prefixes"])
    for name in features:
        if "cluster" in name or name == "model_pair":
            continue
        full = result["share_full"][name]
        lines.append("")
        lines.append(f"  Tỉ lệ {name} ({' / '.join(k + ' cặp đầu' for k in keys)} / toàn bộ):")
        for value in sorted(full, key=lambda v: -full[v]):
            shares = "".join(f"{result['prefixes'][k]['comparison'][name]['share_prefix'].get(value, 0.0):>8.4f}" for k in keys)
            lines.append(f"    {value:<36}{shares}{full[value]:>8.4f}")
    return "\n".join(lines) + "\n"


def statistics_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
