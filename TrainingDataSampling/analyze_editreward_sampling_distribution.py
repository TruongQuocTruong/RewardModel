#!/usr/bin/env python3
"""So sánh phân bố của tập 3k (và 1008 cặp đầu) với toàn bộ EditReward-Data.

Chỉ đọc các file JSONL (không đọc ảnh):
- toàn bộ EditReward-Data: metadata.jsonl;
- tập 3k HumanGT (cùng các cặp với GemmaGT 3k);
- train_order.jsonl của Inspector Baseline (lấy N cặp đầu, mặc định 1008).

Mỗi tiêu chí được so bằng total variation distance (TVD) = 0.5 * tổng |p - q|,
0 = giống hệt, 1 = khác hoàn toàn.  Mốc so sánh: lấy ngẫu nhiên cùng số cặp từ
toàn bộ dữ liệu, lặp nhiều lần (seed cố định), báo trung bình và phân vị 95% của TVD.
TVD của tập 3k lớn hơn phân vị 95% của mốc ngẫu nhiên nghĩa là chênh lệch lớn hơn
mức do lấy mẫu ngẫu nhiên.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from editreward_data_sampling import ALLOWED_MODELS, categorize_instruction  # noqa: E402

DEFAULT_FULL = "/datastore/shared/datasets/EditReward-Data/extracted_data/metadata.jsonl"
DEFAULT_SUBSET = (
    "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation/"
    "editreward_overall_humanGT_3k.jsonl"
)
DEFAULT_TRAIN_ORDER = (
    "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training/"
    "inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/train_order.jsonl"
)
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "full_data_analysis" / "sampling_distribution")

VOTE_TO_CLASS = {"leftvote": "A", "rightvote": "B", "tie": "Tie", "tievote": "Tie"}


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bin_value(value: float | None, edges: Sequence[float], labels: Sequence[str]) -> str:
    """edges tăng dần; nhãn i ứng với value < edges[i]; nhãn cuối cho phần còn lại."""
    if value is None:
        return "missing"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


ABS_DIFF_EDGES = (0.5, 1.5, 2.5, 3.5, 5.5)
ABS_DIFF_LABELS = ("0", "1", "2", "3", "4-5", ">=6")
SIGNED_DIFF_EDGES = (-5.5, -2.5, -0.5, 0.5, 2.5, 5.5)
SIGNED_DIFF_LABELS = ("<=-6", "-5..-3", "-2..-1", "0", "1..2", "3..5", ">=6")
WORD_EDGES = (4, 7, 11, 16, 26)
WORD_LABELS = ("1-3", "4-6", "7-10", "11-15", "16-25", ">=26")
GROUP_EDGES = (2, 3, 4, 6, 11, 21)
GROUP_LABELS = ("1", "2", "3", "4-5", "6-10", "11-20", ">=21")


def pair_features(row: dict) -> Dict[str, str]:
    left, right = row.get("left_model", ""), row.get("right_model", "")
    left_score, right_score = to_float(row.get("left_overall_score")), to_float(row.get("right_overall_score"))
    diff = None if left_score is None or right_score is None else left_score - right_score
    instruction = row.get("instruction", "") or ""
    return {
        "vote": VOTE_TO_CLASS.get(str(row.get("vote_type", "")).lower(), "other"),
        "category": categorize_instruction(instruction),
        "model_pair": " | ".join(sorted((left, right))),
        "left_model": left,
        "right_model": right,
        "signed_score_diff": bin_value(diff, SIGNED_DIFF_EDGES, SIGNED_DIFF_LABELS),
        "abs_score_diff": bin_value(None if diff is None else abs(diff), ABS_DIFF_EDGES, ABS_DIFF_LABELS),
        "instruction_words": bin_value(float(len(instruction.split())), WORD_EDGES, WORD_LABELS),
        "category_x_vote": f"{categorize_instruction(instruction)} / "
        f"{VOTE_TO_CLASS.get(str(row.get('vote_type', '')).lower(), 'other')}",
    }


PAIR_FEATURES = (
    "vote",
    "category",
    "category_x_vote",
    "model",
    "model_pair",
    "signed_score_diff",
    "abs_score_diff",
    "instruction_words",
    "pairs_per_source_in_full",
)


def distributions(rows: Sequence[dict], features: Sequence[Dict[str, str]],
                  full_source_count: Counter) -> Dict[str, Counter]:
    out: Dict[str, Counter] = {name: Counter() for name in PAIR_FEATURES}
    for row, feat in zip(rows, features):
        for name in PAIR_FEATURES:
            if name == "model":
                # Mỗi cặp góp 2 lượt (trái + phải).
                out[name][feat["left_model"]] += 1
                out[name][feat["right_model"]] += 1
            elif name == "pairs_per_source_in_full":
                # Ảnh nguồn của cặp có bao nhiêu cặp trong toàn bộ dữ liệu.
                out[name][bin_value(float(full_source_count[row.get("source_img", "")]),
                                    GROUP_EDGES, GROUP_LABELS)] += 1
            else:
                out[name][feat[name]] += 1
    return out


def pairs_per_source_within(rows: Sequence[dict]) -> Counter:
    """Phân bố số cặp trên mỗi ảnh nguồn bên trong chính tập đó (đếm theo ảnh nguồn)."""
    per_source = Counter(row.get("source_img", "") for row in rows)
    return Counter(bin_value(float(n), GROUP_EDGES, GROUP_LABELS) for n in per_source.values())


def normalize(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()} if total else {}


def tvd(p: Counter, q: Counter) -> float:
    pn, qn = normalize(p), normalize(q)
    return 0.5 * sum(abs(pn.get(k, 0.0) - qn.get(k, 0.0)) for k in set(pn) | set(qn))


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def coverage(full: Counter, subset: Counter) -> Dict[str, object]:
    full_mass = sum(full.values())
    missing = [key for key in full if key not in subset]
    return {
        "values_in_full": len(full),
        "values_in_subset": sum(1 for key in full if key in subset),
        "values_only_in_subset": sorted(key for key in subset if key not in full),
        "full_mass_covered": (sum(v for k, v in full.items() if k in subset) / full_mass) if full_mass else None,
        "missing_values_largest_first": [
            {"value": key, "full_count": full[key]} for key in sorted(missing, key=lambda k: -full[k])[:30]
        ],
    }


def compare_subset(name: str, subset_rows: List[dict], subset_feats: List[Dict[str, str]],
                   full_rows: List[dict], full_feats: List[Dict[str, str]], full_dist: Dict[str, Counter],
                   full_source_count: Counter, repeats: int, seed: int) -> Dict[str, object]:
    sub_dist = distributions(subset_rows, subset_feats, full_source_count)
    sub_within = pairs_per_source_within(subset_rows)

    rng = random.Random(seed)
    indices = list(range(len(full_rows)))
    baseline: Dict[str, List[float]] = {feature: [] for feature in PAIR_FEATURES}
    baseline["pairs_per_source_within_set"] = []
    baseline_sources: List[int] = []
    for _ in range(repeats):
        picked = rng.sample(indices, len(subset_rows))
        rows = [full_rows[i] for i in picked]
        dist = distributions(rows, [full_feats[i] for i in picked], full_source_count)
        for feature in PAIR_FEATURES:
            baseline[feature].append(tvd(dist[feature], full_dist[feature]))
        baseline["pairs_per_source_within_set"].append(tvd(pairs_per_source_within(rows), sub_within))
        baseline_sources.append(len({row.get("source_img", "") for row in rows}))

    report: Dict[str, object] = {"pairs": len(subset_rows), "features": {}}
    for feature in PAIR_FEATURES:
        values = baseline[feature]
        report["features"][feature] = {
            "tvd_vs_full": tvd(sub_dist[feature], full_dist[feature]),
            "random_tvd_mean": statistics.fmean(values),
            "random_tvd_p95": percentile(values, 0.95),
            "coverage": coverage(full_dist[feature], sub_dist[feature]),
            "subset_share": normalize(sub_dist[feature]),
        }
    report["unique_sources"] = len({row.get("source_img", "") for row in subset_rows})
    report["random_unique_sources_mean"] = statistics.fmean(baseline_sources)
    report["random_unique_sources_range"] = [min(baseline_sources), max(baseline_sources)]
    report["pairs_per_source_within_set"] = normalize(sub_within)
    report["random_pairs_per_source_within_set_tvd_to_subset_mean"] = statistics.fmean(
        baseline["pairs_per_source_within_set"]
    )
    report["name"] = name
    return report


def per_category_tie_rate(rows: Iterable[dict], feats: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    totals, ties = Counter(), Counter()
    for _, feat in zip(rows, feats):
        totals[feat["category"]] += 1
        ties[feat["category"]] += feat["vote"] == "Tie"
    return {cat: {"pairs": totals[cat], "tie_rate": ties[cat] / totals[cat]} for cat in sorted(totals)}


def format_report(result: Dict[str, object]) -> str:
    lines = []
    full = result["full"]
    lines.append("=" * 100)
    lines.append(f"Toàn bộ EditReward-Data: {full['pairs']} cặp, {full['unique_sources']} ảnh nguồn")
    lines.append(f"Số tên model khác nhau: {len(full['model_counts'])}")
    for model, count in sorted(full["model_counts"].items(), key=lambda kv: -kv[1]):
        flag = "" if model in ALLOWED_MODELS else "   <- KHÔNG thuộc ALLOWED_MODELS"
        lines.append(f"  {model:<28} {count:>8} lượt{flag}")
    lines.append(f"Tên trong ALLOWED_MODELS không có trong dữ liệu: {full['allowed_models_absent']}")
    lines.append(f"Số cặp bị ALLOWED_MODELS loại: {full['pairs_removed_by_allowed_models']}")
    lines.append(f"Số cặp không có source_img: {full['pairs_without_source']}")

    for key in ("subset_3k", "first_n"):
        sub = result[key]
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"{sub['name']}: {sub['pairs']} cặp, {sub['unique_sources']} ảnh nguồn "
                     f"(ngẫu nhiên cùng cỡ: trung bình {sub['random_unique_sources_mean']:.0f}, "
                     f"khoảng {sub['random_unique_sources_range']})")
        lines.append(f"{'Tiêu chí':<28}{'TVD so full':>12}{'Ngẫu nhiên TB':>15}{'Ngẫu nhiên p95':>16}"
                     f"{'Vượt p95?':>11}{'Giá trị có/full':>17}{'Tỉ trọng full phủ':>19}")
        for feature, info in sub["features"].items():
            cov = info["coverage"]
            over = "CÓ" if info["tvd_vs_full"] > info["random_tvd_p95"] else "không"
            lines.append(
                f"{feature:<28}{info['tvd_vs_full']:>12.4f}{info['random_tvd_mean']:>15.4f}"
                f"{info['random_tvd_p95']:>16.4f}{over:>11}"
                f"{str(cov['values_in_subset']) + '/' + str(cov['values_in_full']):>17}"
                f"{cov['full_mass_covered']:>19.4f}"
            )
        for feature in ("vote", "category", "abs_score_diff", "instruction_words", "pairs_per_source_in_full"):
            lines.append(f"\n  Tỉ lệ {feature} (tập con / full):")
            full_share = result["full"]["share"][feature]
            sub_share = sub["features"][feature]["subset_share"]
            for value in sorted(set(full_share) | set(sub_share)):
                lines.append(f"    {value:<32}{sub_share.get(value, 0):>8.4f}{full_share.get(value, 0):>10.4f}")
        missing = sub["features"]["model_pair"]["coverage"]["missing_values_largest_first"]
        lines.append(f"\n  Cặp model có trong full nhưng không có trong tập con (lớn nhất trước, tối đa 30): "
                     f"{len(missing)} hiển thị")
        for item in missing:
            lines.append(f"    {item['value']:<60}{item['full_count']:>8}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("Tỉ lệ Tie theo loại chỉnh sửa (full / 3k / N cặp đầu):")
    for cat in sorted(result["tie_rate_by_category"]["full"]):
        cells = [result["tie_rate_by_category"][k].get(cat, {"tie_rate": float("nan"), "pairs": 0})
                 for k in ("full", "subset_3k", "first_n")]
        lines.append(f"  {cat:<32}" + "".join(f"{c['tie_rate']:>8.3f} ({c['pairs']:>6})" for c in cells))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", default=DEFAULT_FULL)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--train-order", default=DEFAULT_TRAIN_ORDER)
    parser.add_argument("--first-n", type=int, default=1008)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if (output_dir / "sampling_distribution.json").exists():
        raise FileExistsError(f"Kết quả đã tồn tại, không ghi đè: {output_dir}")

    full_rows = read_jsonl(args.full)
    by_id = {row["unique_key"]: row for row in full_rows}
    if len(by_id) != len(full_rows):
        raise ValueError("unique_key bị trùng trong dữ liệu full.")
    full_feats = [pair_features(row) for row in full_rows]
    full_source_count = Counter(row.get("source_img", "") for row in full_rows)
    full_dist = distributions(full_rows, full_feats, full_source_count)

    # Lấy hàng từ full theo unique_key để mọi tập dùng đúng một nguồn trường dữ liệu.
    subset_ids = [row["unique_key"] for row in read_jsonl(args.subset)]
    missing = [sid for sid in subset_ids if sid not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} cặp của tập 3k không có trong full, ví dụ {missing[:3]}")
    order_ids = [row["sample_id"] for row in read_jsonl(args.train_order)][: args.first_n]
    if not set(order_ids) <= set(subset_ids):
        print("CẢNH BÁO: N cặp đầu của train_order không nằm hết trong tập 3k.")

    def rows_feats(ids: Sequence[str]):
        rows = [by_id[sid] for sid in ids]
        return rows, [pair_features(row) for row in rows]

    sub_rows, sub_feats = rows_feats(subset_ids)
    first_rows, first_feats = rows_feats(order_ids)

    model_counts = full_dist["model"]
    removed = sum(1 for row in full_rows
                  if row.get("left_model") not in ALLOWED_MODELS or row.get("right_model") not in ALLOWED_MODELS)
    result = {
        "inputs": vars(args),
        "full": {
            "pairs": len(full_rows),
            "unique_sources": len(full_source_count),
            "model_counts": dict(model_counts),
            "allowed_models_absent": sorted(m for m in ALLOWED_MODELS if m not in model_counts),
            "pairs_removed_by_allowed_models": removed,
            "pairs_without_source": sum(1 for row in full_rows if not row.get("source_img")),
            "pairs_per_source_within_set": normalize(pairs_per_source_within(full_rows)),
            "share": {feature: normalize(full_dist[feature]) for feature in PAIR_FEATURES},
        },
        "subset_3k": compare_subset("Tập 3k", sub_rows, sub_feats, full_rows, full_feats, full_dist,
                                    full_source_count, args.random_repeats, args.seed),
        "first_n": compare_subset(f"{args.first_n} cặp đầu của train_order", first_rows, first_feats,
                                  full_rows, full_feats, full_dist, full_source_count,
                                  args.random_repeats, args.seed),
        "tie_rate_by_category": {
            "full": per_category_tie_rate(full_rows, full_feats),
            "subset_3k": per_category_tie_rate(sub_rows, sub_feats),
            "first_n": per_category_tie_rate(first_rows, first_feats),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sampling_distribution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = format_report(result)
    (output_dir / "sampling_distribution.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
