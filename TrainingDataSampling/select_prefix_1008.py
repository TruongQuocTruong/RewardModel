#!/usr/bin/env python3
"""Chọn 1008 cặp huấn luyện đầu tiên cho Inspector Baseline (tập ưu tiên mới).

Vùng chọn: toàn bộ EditReward-Data (170772 cặp).  Chạy qua Slurm.

Thứ tự ưu tiên:
1. Đa dạng (ở mức ảnh nguồn; mỗi ảnh nguồn có đúng 1 instruction):
   - mỗi ảnh nguồn tối đa 1 cặp và mỗi instruction tối đa 1 lần (một số instruction chung
     chung xuất hiện ở nhiều ảnh nguồn) -> 1008 ảnh nguồn và 1008 instruction khác nhau;
   - chọn tham lam từng ảnh nguồn: ưu tiên ảnh làm phủ thêm nhiều giá trị chưa có nhất, sau đó
     ảnh có các giá trị đang ít nhất (đưa phân bố về đều) trên các đặc trưng: cụm ảnh và cụm
     instruction (k = 50, 100, 200), chủ đề ảnh, số điểm ảnh, tỉ lệ khung hình, loại instruction,
     số từ instruction.
2. Chất lượng nhãn (điều kiện bắt buộc): Gemma đồng ý với HumanGT và độ nhất quán >= 0.7.
   - Quy tắc đa số mỗi tiêu chí như train_student_hardlabel.py.
   - Overall của Gemma: so số tiêu chí liên quan chọn A với số chọn B (quy ước giống
     analyze_prefix_coverage.py), bằng nhau -> Tie.
   - Độ nhất quán = tỉ lệ tiêu chí liên quan mà cả 3 lần chạy đều đánh dấu liên quan và
     cùng một lựa chọn.
3. Tỉ lệ vote A/B/Tie đều nhau (336 mỗi loại) và mỗi cặp model tối đa ceil(1008/66) = 16 cặp
   - điều kiện bắt buộc (không giới hạn thì nhóm Tie dồn vào các cặp cùng model khác seed).
4. Trong các cặp hợp lệ của ảnh nguồn đã chọn: ưu tiên cặp khó hơn (hiệu điểm con người 1,
   rồi 2; có đánh đổi giữa hai điểm con của con người; có đánh đổi giữa các tiêu chí con của
   Gemma), sau đó cặp model đang ít được chọn.

Đặc trưng cụm/nhãn lấy từ pair_features.jsonl của analyze_prefix_coverage.py (job 12219).
Output (không ghi đè nếu đã có):
- TrainingDataCreation/editreward_overall_humanGT_1008.jsonl (cùng định dạng file HumanGT 3k)
- TrainingDataCreation/editreward_overall_finegrained_gemmaGT_1008.jsonl (copy nguyên dòng từ
  file GemmaGT toàn bộ)
- TrainingDataSampling/prefix_selection_1008/selection_summary.json và selected_pairs.jsonl
Các dòng được ghi theo thứ tự chọn.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

EDITREWARD_ROOT = "/datastore/shared/datasets/EditReward-Data/extracted_data"
SAMPLING_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingDataSampling"
CREATION_DIR = "/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation"
DEFAULT_METADATA = f"{EDITREWARD_ROOT}/metadata.jsonl"
DEFAULT_GEMMA_FULL = f"{CREATION_DIR}/editreward_overall_finegrained_gemmaGT_full.jsonl"
DEFAULT_PAIR_FEATURES = f"{SAMPLING_DIR}/full_data_analysis/prefix_coverage/pair_features.jsonl"
DEFAULT_HUMAN_OUT = f"{CREATION_DIR}/editreward_overall_humanGT_1008.jsonl"
DEFAULT_GEMMA_OUT = f"{CREATION_DIR}/editreward_overall_finegrained_gemmaGT_1008.jsonl"
DEFAULT_REPORT_DIR = f"{SAMPLING_DIR}/prefix_selection_1008"

VOTE_TO_CLASS = {"leftvote": "A", "rightvote": "B", "tie": "Tie", "tievote": "Tie"}
VOTES = ("A", "B", "Tie")
HUMAN_FIELDS = ["unique_key", "source_img", "instruction", "left_model", "left_output_image",
                "left_overall_score", "right_model", "right_output_image", "right_overall_score",
                "vote_type"]
SOURCE_FEATURES = [
    "image_cluster_k50", "image_cluster_k100", "image_cluster_k200",
    "instruction_cluster_k50", "instruction_cluster_k100", "instruction_cluster_k200",
    "image_topic", "image_pixels", "image_aspect", "instruction_type", "instruction_words",
]


def gemma_quality(record: dict) -> Dict[str, object]:
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
    return {
        "gemma_overall": "A" if n_a > n_b else "B" if n_b > n_a else "Tie",
        "consistency": n_unanimous / n_relevant if n_relevant else 0.0,
    }


def hardness(feat: dict) -> int:
    score = {"1": 2, "2": 1}.get(feat["human_abs_score_diff"], 0)
    score += feat["human_subscore_tradeoff"] == "yes"
    score += feat["gemma_tradeoff"] == "yes"
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--gemma-full", default=DEFAULT_GEMMA_FULL)
    parser.add_argument("--pair-features", default=DEFAULT_PAIR_FEATURES)
    parser.add_argument("--human-out", default=DEFAULT_HUMAN_OUT)
    parser.add_argument("--gemma-out", default=DEFAULT_GEMMA_OUT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--size", type=int, default=1008)
    parser.add_argument("--min-consistency", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-model-pair", type=int, default=None,
                        help="Mặc định ceil(size / số cặp model)")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    for path in (Path(args.human_out), Path(args.gemma_out), report_dir / "selection_summary.json"):
        if path.exists():
            raise FileExistsError(f"Không ghi đè file đã có: {path}")
    if args.size % len(VOTES):
        raise ValueError("Số cặp phải chia hết cho 3 để A/B/Tie đều nhau.")
    quota = {vote: args.size // len(VOTES) for vote in VOTES}
    rng = random.Random(args.seed)

    print("[*] Đọc metadata ...", flush=True)
    meta = {}
    with open(args.metadata, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                meta[row["unique_key"]] = row

    print("[*] Đọc đặc trưng từng cặp ...", flush=True)
    feats = {}
    with open(args.pair_features, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                feats[row["unique_key"]] = row
    assert set(feats) == set(meta), "pair_features và metadata không cùng tập cặp"

    print("[*] Đọc GemmaGT toàn bộ ...", flush=True)
    gemma_line, gemma_q = {}, {}
    with open(args.gemma_full, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                gemma_line[record["sample_id"]] = line if line.endswith("\n") else line + "\n"
                gemma_q[record["sample_id"]] = gemma_quality(record)
    assert set(gemma_q) == set(meta), "GemmaGT và metadata không cùng tập cặp"

    # ---------- lọc chất lượng ----------
    eligible_by_source: Dict[str, List[str]] = defaultdict(list)
    vote_counts_all, vote_counts_eligible = Counter(), Counter()
    for key, row in meta.items():
        vote = VOTE_TO_CLASS[row["vote_type"]]
        vote_counts_all[vote] += 1
        q = gemma_q[key]
        if q["gemma_overall"] == vote and q["consistency"] >= args.min_consistency - 1e-9:
            eligible_by_source[row["source_img"]].append(key)
            vote_counts_eligible[vote] += 1
    sources_with_vote = {vote: sum(1 for keys in eligible_by_source.values()
                                   if any(VOTE_TO_CLASS[meta[k]["vote_type"]] == vote for k in keys))
                         for vote in VOTES}
    print(f"[*] Cặp hợp lệ theo vote: {dict(vote_counts_eligible)} / toàn bộ {dict(vote_counts_all)}")
    print(f"[*] Ảnh nguồn có >=1 cặp hợp lệ: {len(eligible_by_source)}; theo vote: {sources_with_vote}")
    for vote in VOTES:
        if sources_with_vote[vote] < quota[vote]:
            raise RuntimeError(f"Không đủ ảnh nguồn có cặp hợp lệ vote {vote}: "
                               f"{sources_with_vote[vote]} < {quota[vote]}")

    # ---------- chọn tham lam theo ảnh nguồn ----------
    sources = sorted(eligible_by_source)
    first_pair = {src: feats[eligible_by_source[src][0]] for src in sources}
    values = {f: sorted({first_pair[s][f] for s in sources}) for f in SOURCE_FEATURES}
    value_index = {f: {v: i for i, v in enumerate(values[f])} for f in SOURCE_FEATURES}
    codes = np.array([[value_index[f][first_pair[s][f]] for f in SOURCE_FEATURES] for s in sources])
    counts = [np.zeros(len(values[f])) for f in SOURCE_FEATURES]
    expected = [args.size / len(values[f]) for f in SOURCE_FEATURES]
    noise = np.array([rng.random() for _ in sources])
    available = np.ones(len(sources), dtype=bool)
    vote_of_pair = {k: VOTE_TO_CLASS[meta[k]["vote_type"]] for keys in eligible_by_source.values() for k in keys}

    n_model_pairs = len({f["model_pair"] for f in feats.values()})
    cap = args.max_per_model_pair or -(-args.size // n_model_pairs)
    print(f"[*] Giới hạn mỗi cặp model: {cap} (số cặp model: {n_model_pairs})", flush=True)

    selected: List[str] = []
    selected_vote = Counter()
    model_pair_count = Counter()
    used_instructions = set()

    def valid_pairs(src: str) -> List[str]:
        return [k for k in eligible_by_source[src]
                if selected_vote[vote_of_pair[k]] < quota[vote_of_pair[k]]
                and model_pair_count[feats[k]["model_pair"]] < cap]

    while len(selected) < args.size:
        coverage_gain = np.zeros(len(sources))
        balance = np.zeros(len(sources))
        for j in range(len(SOURCE_FEATURES)):
            current = counts[j][codes[:, j]]
            coverage_gain += current == 0
            balance += current / expected[j]
        score = coverage_gain * 1e6 - balance * 1e2 + noise
        score[~available] = -np.inf
        best, candidates = None, []
        for index in np.argsort(-score, kind="stable"):
            if not available[index]:
                break
            src = sources[index]
            if meta[eligible_by_source[src][0]]["instruction"] in used_instructions:
                available[index] = False
                continue
            candidates = valid_pairs(src)
            if candidates:
                best = int(index)
                break
        if best is None:
            raise RuntimeError(f"Hết ảnh nguồn hợp lệ khi đã chọn {len(selected)} cặp.")
        src = sources[best]

        # cặp trong ảnh nguồn: ưu tiên vote còn thiếu nhiều nhất, rồi khó hơn,
        # rồi cặp model ít được chọn, rồi theo mã cặp.
        candidates.sort(key=lambda k: (
            -(quota[vote_of_pair[k]] - selected_vote[vote_of_pair[k]]),
            -hardness(feats[k]),
            model_pair_count[feats[k]["model_pair"]],
            k,
        ))
        chosen = candidates[0]
        selected.append(chosen)
        selected_vote[vote_of_pair[chosen]] += 1
        model_pair_count[feats[chosen]["model_pair"]] += 1
        used_instructions.add(meta[chosen]["instruction"])
        for j in range(len(SOURCE_FEATURES)):
            counts[j][codes[best, j]] += 1
        available[best] = False

    # ---------- kiểm tra ----------
    assert len(set(selected)) == args.size
    assert len({meta[k]["source_img"] for k in selected}) == args.size
    assert len({meta[k]["instruction"] for k in selected}) == args.size
    assert max(model_pair_count.values()) <= cap
    assert all(selected_vote[v] == quota[v] for v in VOTES)
    assert all(gemma_q[k]["gemma_overall"] == vote_of_pair[k] for k in selected)
    assert all(gemma_q[k]["consistency"] >= args.min_consistency - 1e-9 for k in selected)

    # ---------- ghi file ----------
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(args.human_out, "x", encoding="utf-8") as handle:
        for k in selected:
            row = meta[k]
            out = {f: row[f] for f in HUMAN_FIELDS}
            out["left_overall_score"] = str(float(row["left_overall_score"]))
            out["right_overall_score"] = str(float(row["right_overall_score"]))
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    with open(args.gemma_out, "x", encoding="utf-8") as handle:
        for k in selected:
            handle.write(gemma_line[k])
    with open(report_dir / "selected_pairs.jsonl", "x", encoding="utf-8") as handle:
        for position, k in enumerate(selected):
            handle.write(json.dumps({"position": position, **feats[k],
                                     "gemma_consistency_value": gemma_q[k]["consistency"],
                                     "hardness": hardness(feats[k])}, ensure_ascii=False) + "\n")

    def dist(name: str) -> Dict[str, int]:
        return dict(Counter(feats[k][name] for k in selected).most_common())

    summary = {
        "inputs": vars(args),
        "eligible_pairs_by_vote": dict(vote_counts_eligible),
        "all_pairs_by_vote": dict(vote_counts_all),
        "eligible_sources": len(eligible_by_source),
        "eligible_sources_by_vote": sources_with_vote,
        "selected_pairs": len(selected),
        "selected_unique_sources": len({meta[k]["source_img"] for k in selected}),
        "selected_unique_instructions": len({meta[k]["instruction"] for k in selected}),
        "vote": dict(selected_vote),
        "values_covered": {f: f"{int((c > 0).sum())}/{len(c)}" for f, c in zip(SOURCE_FEATURES, counts)},
        "model_pair_cap": cap,
        "model_pairs_covered": len(model_pair_count),
        "model_pair_count_min_max": [min(model_pair_count.values()), max(model_pair_count.values())],
        "hardness": dict(Counter(hardness(feats[k]) for k in selected)),
        "human_abs_score_diff": dist("human_abs_score_diff"),
        "human_subscore_tradeoff": dist("human_subscore_tradeoff"),
        "gemma_tradeoff": dist("gemma_tradeoff"),
        "image_topic": dist("image_topic"),
        "image_pixels": dist("image_pixels"),
        "image_aspect": dist("image_aspect"),
        "instruction_type": dist("instruction_type"),
        "instruction_words": dist("instruction_words"),
    }
    (report_dir / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[✓] {args.human_out}\n[✓] {args.gemma_out}")


if __name__ == "__main__":
    main()
