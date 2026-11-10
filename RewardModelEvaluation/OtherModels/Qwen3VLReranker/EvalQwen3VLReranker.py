"""Zero-shot evaluation of the original Qwen3-VL-Reranker with its unmodified source code.

The model is loaded with the official ``Qwen3VLReranker`` class from
``Models/Qwen3-VL-Reranker-8B/scripts/qwen3_vl_reranker.py`` (no LoRA, no Tie head).
Each candidate image is scored independently:

    Instruct = Overall Inspector evaluation instruction (same text as ``base_zero_shot``)
    Query    = source image + edit instruction
    Document = candidate image (A or B)

Two scores are stored per image:
- logit  = score_linear(last hidden state of the last token) = logit(yes) - logit(no)
- sigmoid = sigmoid(logit), exactly what the official ``compute_scores`` returns.

Datasets (no other benchmark is read):
- EditReward-Test (1k): 1000 non-tie pairs.
- EditReward-Bench (115): non-tie pairs of EditReward-Bench, human labels.
- EditReward (VLM-Reannotate)-Bench (115): same pairs, VLM majority labels.
- EditReward-Bench (133) with Tie samples: human labels, 18 ties.
- EditReward (VLM-Reannotate)-Bench (133) with Tie samples: VLM majority labels of all 133 pairs.

Accuracy and margin AUC use the non-tie pairs. Tie AP and AUC treat Tie as the
positive class with confidence -|d|, d = score A - score B. Every metric is
computed twice: d from logits and d from sigmoid scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/datastore/truongtq/TrainingRewardModel")
EVAL_ROOT = ROOT / "RewardModelEvaluation"
DATA_DIR = EVAL_ROOT / "EvalDatasets"
MODEL_DIR = ROOT / "Models" / "Qwen3-VL-Reranker-8B"
BASE_ZERO_SHOT = (
    EVAL_ROOT
    / "OurModel/evaluation_results/inspector_checkpoints"
    / "inspector_pipeline_overall_humanGT_300_maxpixelsdefault/base_zero_shot.json"
)

sys.path.insert(0, str(MODEL_DIR / "scripts"))
from qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402  (official source, unmodified)

TEST = "EditReward-Test(1k)"
BENCH = "EditReward-Bench"
RANKING_TO_LABEL = {"A>B": 1, "B>A": -1, "A=B": 0}
VOTE_TO_LABEL = {"leftvote": 1, "rightvote": -1}
VLM_VOTE_TO_LABEL = {"0": 1, "1": -1, "2": 0}  # 0 = candidate_1 (A), 1 = candidate_2 (B), 2 = tie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(MODEL_DIR))
    parser.add_argument("--test-dataset", default=str(DATA_DIR / "EditReward" / "EditReward-Test_pairwise_1k.jsonl"))
    parser.add_argument("--editreward-root", default="/datastore/shared/datasets/EditReward-Data")
    parser.add_argument("--bench", default=str(DATA_DIR / "EditReward" / "EditReward-Bench_pairwise.json"))
    parser.add_argument("--vlm-bench", default=str(DATA_DIR / "EditReward" / "EditReward-Bench_pairwise_VLM_judge.json"))
    parser.add_argument("--vlm-votes", default=str(DATA_DIR / "EditReward" / "EditReward-Bench_VLM_judge" / "pairs.csv"))
    parser.add_argument("--base-zero-shot", default=str(BASE_ZERO_SHOT))
    parser.add_argument(
        "--output-dir",
        default=str(EVAL_ROOT / "OtherModels/evaluation_results/qwen3_vl_reranker_original"),
    )
    return parser.parse_args()


def existing_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return str(path.resolve())


def resolve_editreward_image(raw: str, root: Path) -> str:
    """Absolute path, or relative to the EditReward-Data root (or its extracted_data folder)."""
    path = Path(raw.strip().removeprefix("file://"))
    if path.is_absolute():
        return existing_file(path)
    for candidate in (root / path, root / "extracted_data" / path):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Image does not exist under {root}: {raw}")


def load_test(path: Path, root: Path) -> List[Dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            {
                "sample_id": row["unique_key"],
                "source_image": resolve_editreward_image(row["source_img"], root),
                "edit_instruction": row["instruction"].strip(),
                "image_a": resolve_editreward_image(row["left_output_image"], root),
                "image_b": resolve_editreward_image(row["right_output_image"], root),
                "label": VOTE_TO_LABEL[row["vote_type"].lower()],
            }
        )
    return records


def load_bench(path: Path) -> List[Dict]:
    return [
        {
            "sample_id": row["sample_id"],
            "source_image": existing_file(Path(row["source_image"])),
            "edit_instruction": row["edit_instruction"].strip(),
            "image_a": existing_file(Path(row["candidate_1"])),
            "image_b": existing_file(Path(row["candidate_2"])),
            "label": RANKING_TO_LABEL[row["metadata_ranking"]],
        }
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def load_vlm_labels(vlm_bench: Path, vlm_votes: Path, bench: List[Dict]) -> Dict[str, Dict[str, int]]:
    """VLM labels: the 115-pair judge file, and the vote majority for all 133 pairs."""
    human = {r["sample_id"]: r["label"] for r in bench}
    labels_115 = {
        row["sample_id"]: RANKING_TO_LABEL[row["metadata_ranking"]]
        for row in json.loads(vlm_bench.read_text(encoding="utf-8"))
    }
    labels_133 = {}
    with open(vlm_votes, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["subset"] != "2pair":
                continue
            sample_id = row["group_id"]
            if VLM_VOTE_TO_LABEL[row["original_overall"]] != human[sample_id]:
                raise ValueError(f"{sample_id}: pairs.csv original label differs from EditReward-Bench.")
            labels_133[sample_id] = VLM_VOTE_TO_LABEL[row["majority_overall"]]
    if set(labels_133) != set(human):
        raise ValueError("pairs.csv does not cover exactly the EditReward-Bench pairs.")
    if any(labels_133[k] != v for k, v in labels_115.items()):
        raise ValueError("VLM judge file disagrees with the pairs.csv majority labels.")
    return {"115": labels_115, "133": labels_133}


@torch.no_grad()
def score_image(model: Qwen3VLReranker, instruction: str, record: Dict, image: str) -> tuple[float, float]:
    """The two steps of the official compute_scores, keeping the logit before sigmoid."""
    pair = model.format_mm_instruction(
        record["edit_instruction"], record["source_image"], None,
        None, image, None,
        instruction=instruction,
    )
    inputs = model.tokenize([pair]).to(model.model.device)
    logit = model.score_linear(model.model(**inputs).last_hidden_state[:, -1])
    return float(logit.float().item()), float(torch.sigmoid(logit).float().item())


def margin_auc(labels: np.ndarray, margins: np.ndarray) -> float:
    """Within-pair margin AUC: fraction of (i, j) with w_i + w_j > 0, w = label * margin."""
    w = labels * margins
    combined = w[:, None] + w[None, :]
    return float(((combined > 0).sum() + 0.5 * (combined == 0).sum()) / combined.size)


def pairwise_metrics(labels: np.ndarray, margins: np.ndarray) -> Dict:
    keep = labels != 0
    y, m = labels[keep], margins[keep]
    return {
        "pairs": int(keep.sum()),
        "accuracy_percent": round(float((np.sign(m) == y).mean()) * 100, 2),
        "margin_auc_percent": round(margin_auc(y, m) * 100, 2),
    }


def tie_metrics(labels: np.ndarray, margins: np.ndarray) -> Dict:
    tie = (labels == 0).astype(int)
    confidence = -np.abs(margins)
    return {
        "pairs": int(len(labels)),
        "tie_samples": int(tie.sum()),
        "confidence_definition": "-|d|",
        "random_ap_baseline_percent": round(float(tie.mean()) * 100, 2),
        "average_precision_percent": round(float(average_precision_score(tie, confidence)) * 100, 2),
        "roc_auc_percent": round(float(roc_auc_score(tie, confidence)) * 100, 2),
    }


def compare_with_base_zero_shot(predictions: Dict[str, List[Dict]], path: Path) -> Dict:
    base = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    names = {TEST: "EditReward-Data-Test", BENCH: "EditReward-Bench"}
    report = {}
    for ours, theirs in names.items():
        reference = {x["sample_id"]: x for x in base[theirs]}
        rows = predictions[ours]
        if {r["sample_id"] for r in rows} != set(reference):
            raise ValueError(f"{ours}: sample ids differ from base_zero_shot.json.")
        scores = np.array([[r["logit_A"], r["logit_B"]] for r in rows])
        ref_scores = np.array([[reference[r["sample_id"]]["score_A"], reference[r["sample_id"]]["score_B"]] for r in rows])
        margins = scores[:, 0] - scores[:, 1]
        ref_margins = np.array([reference[r["sample_id"]]["margin_A_minus_B"] for r in rows])
        report[ours] = {
            "max_abs_score_difference": float(np.abs(scores - ref_scores).max()),
            "mean_abs_score_difference": float(np.abs(scores - ref_scores).mean()),
            "margin_sign_agreement_percent": round(float((np.sign(margins) == np.sign(ref_margins)).mean()) * 100, 2),
            "margin_pearson": float(np.corrcoef(margins, ref_margins)[0, 1]),
        }
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = json.loads(Path(args.base_zero_shot).read_text(encoding="utf-8"))["prompt"]
    instruction = base_prompt["evaluation_instruction"]

    datasets = {
        TEST: load_test(Path(args.test_dataset), Path(args.editreward_root)),
        BENCH: load_bench(Path(args.bench)),
    }
    if any(r["label"] == 0 for r in datasets[TEST]):
        raise ValueError("EditReward-Test (1k) must not contain ties.")
    vlm_labels = load_vlm_labels(Path(args.vlm_bench), Path(args.vlm_votes), datasets[BENCH])
    print({name: len(rows) for name, rows in datasets.items()}, flush=True)

    # Recommended settings of the official README: bf16; all other arguments keep their defaults.
    model = Qwen3VLReranker(model_name_or_path=args.model_path, torch_dtype=torch.bfloat16)

    # The split compute_scores must reproduce the official process() output.
    check = datasets[BENCH][0]
    official = model.process(
        {
            "instruction": instruction,
            "query": {"text": check["edit_instruction"], "image": check["source_image"]},
            "documents": [{"image": check["image_a"]}, {"image": check["image_b"]}],
        }
    )
    split = [score_image(model, instruction, check, check[k])[1] for k in ("image_a", "image_b")]
    if not np.allclose(official, split, atol=5e-3):  # one bf16 step of sigmoid
        raise AssertionError(f"Split scoring {split} differs from official process() {official}.")
    print(f"Official process() check passed: {official}", flush=True)

    predictions: Dict[str, List[Dict]] = {}
    for name, records in datasets.items():
        rows = []
        for index, record in enumerate(records, start=1):
            logit_a, sigmoid_a = score_image(model, instruction, record, record["image_a"])
            logit_b, sigmoid_b = score_image(model, instruction, record, record["image_b"])
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "label": record["label"],
                    "logit_A": logit_a,
                    "logit_B": logit_b,
                    "sigmoid_A": sigmoid_a,
                    "sigmoid_B": sigmoid_b,
                }
            )
            if index % 100 == 0 or index == len(records):
                print(f"{name}: {index}/{len(records)}", flush=True)
        predictions[name] = rows

    bench_ids = [r["sample_id"] for r in predictions[BENCH]]
    label_sets = {
        TEST: np.array([r["label"] for r in predictions[TEST]]),
        "EditReward-Bench(115)": np.array([r["label"] for r in predictions[BENCH]]),
        "EditReward(VLM-Reannotate)-Bench(115)": np.array([vlm_labels["115"].get(k, 0) for k in bench_ids]),
        "EditReward-Bench(133) w Tie samples": np.array([r["label"] for r in predictions[BENCH]]),
        "EditReward(VLM-Reannotate)-Bench(133) w Tie samples": np.array([vlm_labels["133"][k] for k in bench_ids]),
    }
    metrics = {}
    for score in ("logit", "sigmoid"):
        margins = {
            name: np.array([r[f"{score}_A"] - r[f"{score}_B"] for r in rows])
            for name, rows in predictions.items()
        }
        metrics[f"d_from_{score}"] = {
            TEST: pairwise_metrics(label_sets[TEST], margins[TEST]),
            "EditReward-Bench(115)": pairwise_metrics(label_sets["EditReward-Bench(115)"], margins[BENCH]),
            "EditReward(VLM-Reannotate)-Bench(115)": pairwise_metrics(
                label_sets["EditReward(VLM-Reannotate)-Bench(115)"], margins[BENCH]
            ),
            "EditReward-Bench(133) w Tie samples": tie_metrics(
                label_sets["EditReward-Bench(133) w Tie samples"], margins[BENCH]
            ),
            "EditReward(VLM-Reannotate)-Bench(133) w Tie samples": tie_metrics(
                label_sets["EditReward(VLM-Reannotate)-Bench(133) w Tie samples"], margins[BENCH]
            ),
        }
    if metrics["d_from_logit"]["EditReward(VLM-Reannotate)-Bench(115)"]["pairs"] != 115:
        raise AssertionError("VLM-Reannotate Bench (115) must contain 115 non-tie pairs.")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_path,
        "source_code": str(MODEL_DIR / "scripts/qwen3_vl_reranker.py"),
        "settings": {
            "torch_dtype": "bfloat16",
            "max_length": model.max_length,
            "min_pixels": model.min_pixels,
            "max_pixels": model.max_pixels,
            "instruction": instruction,
            "official_process_check": official,
        },
        "metrics": metrics,
        "comparison_with_base_zero_shot": compare_with_base_zero_shot(predictions, Path(args.base_zero_shot)),
    }
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print(json.dumps(summary["comparison_with_base_zero_shot"], indent=2), flush=True)


if __name__ == "__main__":
    main()
