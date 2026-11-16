"""Judge protocols of GEdit-Bench-EN, ImgEdit (Basic singleturn) and KRIS-Bench for Best-of-N rollouts.

Pure Python (no torch): builds the judge calls of one rollout image and parses the judge answers,
following the original repositories cloned next to this folder (read only):
  Step1X-Edit/GEdit-Bench  run_gedit_score.py, viescore/__init__.py, viescore/utils.py, viescore/vie_prompts.py
  imgedit/Benchmark/Basic  basic_bench.py, prompts.json, step1_get_avgscore.py
  Kris_Bench               metrics_common.py, metrics_knowledge.py, metrics_view_change.py, utils/prompts.py

A judge call is {"name", "content": [("text", str) | ("image", path, resize_rule)], "max_new_tokens"}.
resize_rule: "gedit_512_area" (GEdit evaluation geometry) or None (original resolution, as the
original scripts send the file bytes).
"""

from __future__ import annotations

import json
import math
import random
import re
import runpy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parent.parent
GEDIT_REPO = PROJECT_DIR / "Step1X-Edit" / "GEdit-Bench"
IMGEDIT_REPO = PROJECT_DIR / "imgedit" / "Benchmark" / "Basic"
KRIS_REPO = PROJECT_DIR / "Kris_Bench"

ROLLOUT_ROOT = Path("/datastore/nghiatg/Truongtq/RewardGuidedBenchsRolloutsGen")
SOURCE_ROOT = Path("/datastore/anhndt/Truongtq/OtherEditingBenchmark")
SEEDS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
BENCHES = ("GEdit-Bench-EN", "ImgEdit", "KRIS-Bench")

COLUMNS = {
    "GEdit-Bench-EN": ("semantic_consistency", "perceptual_quality", "overall"),
    "ImgEdit": ("instruction_adherence", "editing_quality", "detail_preservation"),
    "KRIS-Bench": ("visual_consistency", "visual_quality", "instruction_following"),
}

# Position (0-based) of each table column in the 3 scores ImgEdit judges return, per edit_type.
# The order of the score names in prompts.json differs for style / action / extract.
IMGEDIT_COLUMN_POSITIONS = {
    **{t: {"instruction_adherence": 0, "editing_quality": 1, "detail_preservation": 2}
       for t in ("replace", "add", "adjust", "remove", "background", "compose")},
    **{t: {"instruction_adherence": 0, "editing_quality": 2, "detail_preservation": 1}
       for t in ("style", "action", "extract")},
}

KRIS_KNOWLEDGE_CATEGORIES = (
    "abstract_reasoning", "mathematics", "practical_knowledge", "medicine", "rule-based_reasoning",
    "biology", "geography", "chemistry", "humanities", "physics",
)
KRIS_COMMON_CATEGORIES = (
    "count_change", "color_change", "anomaly_correction", "position_movement", "size_adjustment",
    "part_completion", "multi-instruction_execution",
)
KRIS_VIEW_CATEGORIES = ("viewpoint_change",)


# --------------------------------------------------------------------------- prompts
def load_prompts() -> Dict[str, Any]:
    vie = runpy.run_path(str(GEDIT_REPO / "viescore" / "vie_prompts.py"))
    kris = runpy.run_path(str(KRIS_REPO / "utils" / "prompts.py"))
    imgedit = json.loads((IMGEDIT_REPO / "prompts.json").read_text(encoding="utf-8"))
    return {
        # viescore/__init__.py, task == "tie"
        "gedit_sc": "\n".join([vie["_context_no_delimit"], vie["_prompts_0shot_two_image_edit_rule"],
                               vie["_prompts_0shot_tie_rule_SC"]]),
        "gedit_pq": "\n".join([vie["_context_no_delimit"], vie["_prompts_0shot_rule_PQ"]]),
        "imgedit": imgedit,
        "kris": {k: v for k, v in kris.items() if k.startswith("prompt_")},
    }


# --------------------------------------------------------------------------- samples
def _gedit_samples(rollout_root: Path, source_root: Path) -> List[Dict[str, Any]]:
    data = json.loads((rollout_root / "gedit_300.json").read_text(encoding="utf-8"))
    return [{"sample_id": f"{d['task_type']}/{d['key']}", "category": d["task_type"],
             "instruction": d["instruction"],
             "source_image": str(source_root / "GEdit-Bench-EN" / d["image"]),
             "rollout_dir": str(rollout_root / "Results/FLUX2-9B/GEdit-Bench-EN" / d["task_type"] / d["key"])}
            for d in data]


def _imgedit_samples(rollout_root: Path, source_root: Path) -> List[Dict[str, Any]]:
    data = json.loads((rollout_root / "imgedit_singleturn_300.json").read_text(encoding="utf-8"))
    rows = []
    for key, d in data.items():
        folder = Path(d["id"]).parent  # compose ids are nested: compose/human/12.jpg -> compose/human/<key>
        rows.append({"sample_id": key, "category": d["edit_type"], "instruction": d["prompt"],
                     "source_image": str(source_root / "ImgEdit-Bench/Benchmark/singleturn" / d["id"]),
                     "rollout_dir": str(rollout_root / "Results/FLUX2-9B/ImgEdit-Bench/singleturn" / folder / key)})
    return rows


def _kris_samples(rollout_root: Path, source_root: Path) -> List[Dict[str, Any]]:
    data = json.loads((rollout_root / "kris_300.json").read_text(encoding="utf-8"))
    rows = []
    for d in data:
        row = {"sample_id": f"{d['category']}/{d['id']}", "category": d["category"],
               "instruction": d["instruction"], "explanation": d.get("explanation") or "",
               "source_image": str(source_root / "KRIS_Bench" / d["image"]),
               "rollout_dir": str(rollout_root / "Results/FLUX2-9B/KRIS-Bench" / d["category"] / d["export_key"])}
        if d.get("gt_image"):
            row["gt_image"] = str(source_root / "KRIS_Bench" / d["gt_image"])
        rows.append(row)
    return rows


def load_samples(bench: str, rollout_root: Path = ROLLOUT_ROOT, source_root: Path = SOURCE_ROOT) -> List[Dict[str, Any]]:
    loaders = {"GEdit-Bench-EN": _gedit_samples, "ImgEdit": _imgedit_samples, "KRIS-Bench": _kris_samples}
    if bench not in loaders:
        raise ValueError(f"Unknown bench: {bench!r}")
    rows = loaders[bench](Path(rollout_root), Path(source_root))
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample_id in {bench}")
    for r in rows:
        r["bench"] = bench
        if bench == "KRIS-Bench":
            kris_category_group(r["category"])  # raises on an unknown category
        if bench == "ImgEdit" and r["category"] not in IMGEDIT_COLUMN_POSITIONS:
            raise ValueError(f"Unknown ImgEdit edit_type: {r['category']!r}")
    return rows


def expand_seeds(samples: Sequence[Mapping[str, Any]], seeds: Sequence[int] = SEEDS) -> List[Dict[str, Any]]:
    """One judge item per (sample, seed); the edited image is <rollout_dir>/seed_<s>.png."""
    items = []
    for s in samples:
        for seed in seeds:
            item = dict(s)
            item["seed"] = int(seed)
            item["edited_image"] = str(Path(s["rollout_dir"]) / f"seed_{seed}.png")
            items.append(item)
    return items


def kris_category_group(category: str) -> str:
    if category in KRIS_KNOWLEDGE_CATEGORIES:
        return "knowledge"
    if category in KRIS_VIEW_CATEGORIES:
        return "view"
    if category in KRIS_COMMON_CATEGORIES:
        return "common"
    raise ValueError(f"KRIS category not handled by this protocol: {category!r}")


# --------------------------------------------------------------------------- judge calls
def judge_calls(item: Mapping[str, Any], prompts: Mapping[str, Any], kris_anomaly_prompt: str = "original_code") -> List[Dict[str, Any]]:
    bench = item["bench"]
    src, edt = item["source_image"], item["edited_image"]
    if bench == "GEdit-Bench-EN":
        # viescore: SC sees [source, edited] + prompt; PQ sees only the edited image. Images first.
        return [
            {"name": "SC", "max_new_tokens": 512,
             "content": [("image", src, "gedit_512_area"), ("image", edt, "gedit_512_area"),
                         ("text", prompts["gedit_sc"].replace("<instruction>", item["instruction"]))]},
            {"name": "PQ", "max_new_tokens": 512,
             "content": [("image", edt, "gedit_512_area"), ("text", prompts["gedit_pq"])]},
        ]
    if bench == "ImgEdit":
        # basic_bench.py: text, original image, edited image.
        text = prompts["imgedit"][item["category"]].replace("<edit_prompt>", item["instruction"])
        return [{"name": "ImgEdit", "max_new_tokens": 512,
                 "content": [("text", text), ("image", src, None), ("image", edt, None)]}]
    if bench == "KRIS-Bench":
        p = prompts["kris"]
        group = kris_category_group(item["category"])
        instr, expl = item["instruction"], item.get("explanation", "")

        def with_images(text, original=True, gt=False):
            content = [("text", text)]
            if original:
                content += [("text", "This is the original image:"), ("image", src, None)]
            content += [("text", "This is the edited image:"), ("image", edt, None)]
            if gt:
                if not item.get("gt_image"):
                    raise ValueError(f"viewpoint_change sample without gt_image: {item['sample_id']}")
                content += [("text", "This is the ground truth image:"), ("image", item["gt_image"], None)]
            return content

        calls = [{"name": "consistency", "max_new_tokens": 1000,
                  "content": with_images(p["prompt_consist"].format(instruct=instr))}]
        if group == "knowledge":
            calls.append({"name": "dual", "max_new_tokens": 1000,
                          "content": with_images(p["prompt_dual_evaluation"].format(instruct=instr, explanation=expl))})
        elif group == "view":
            calls.append({"name": "instruction", "max_new_tokens": 1000,
                          "content": with_images(p["prompt_view_instruction_following"].format(instruct=instr), gt=True)})
        else:
            # metrics_common.py checks "abnormality_correction" in category, which never matches
            # "anomaly_correction": the original code always uses the normal prompt.
            use_abnormal = kris_anomaly_prompt == "intended" and item["category"] == "anomaly_correction"
            if kris_anomaly_prompt not in ("original_code", "intended"):
                raise ValueError(f"Unknown kris_anomaly_prompt: {kris_anomaly_prompt!r}")
            text = (p["prompt_abnormal_instruction_following"].format(instruct=instr, explanation=expl)
                    if use_abnormal else p["prompt_instruction_following"].format(instruct=instr))
            calls.append({"name": "instruction", "max_new_tokens": 1000, "content": with_images(text)})
        # The original scripts send prompt_quality WITHOUT .format(): the judge sees the literal "{{".
        calls.append({"name": "quality", "max_new_tokens": 1000,
                      "content": with_images(p["prompt_quality"], original=False)})
        return calls
    raise ValueError(f"Unknown bench: {bench!r}")


def gedit_resize_size(width: int, height: int) -> tuple:
    """run_gedit_score.calculate_dimensions(512 * 512, width / height)."""
    if width <= 0 or height <= 0:
        raise ValueError("Image size must be positive")
    ratio = width / height
    new_width = math.sqrt(512 * 512 * ratio)
    return int(new_width), int(new_width / ratio)


# --------------------------------------------------------------------------- parsing: GEdit (viescore/utils.py)
def _is_int_between_0_and_10(s: str) -> bool:
    try:
        return 0 <= int(s) <= 10
    except ValueError:
        return False


def _fix_json(input_str: str) -> str:
    fixed = re.sub(r'(\w+):', r'"\1":', input_str)

    def format_value(match):
        key, value, comma = match.groups()
        value = value.strip()
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = f'[{value}]'
        elif re.match(r'^(true|false|null)$', value, re.IGNORECASE):
            pass
        else:
            value = f'"{value}"'
        return f'{key}: {value}{comma}'

    return re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed)


def viescore_output_to_dict(text: str, give_up_parsing: bool = False, rng: Optional[random.Random] = None):
    """viescore.utils.mllm_output_to_dict (delimiter, JSON, list-of-scores, single number; guess if giving up)."""
    delimiter = '||V^=^V||'
    if text.count(delimiter) == 2:
        start, end = text.find(delimiter) + len(delimiter), text.rfind(delimiter)
    else:
        start, end = text.find('{'), text.rfind('}') + 1
        if start == -1 or end == 0:
            start, end = text.find('['), text.rfind(']') + 1
            if give_up_parsing:
                guess = (rng or random).randint(0, 10)
                text = json.dumps({'score': [guess], "reasoning": f"guess_if_cannot_parse | {text}"})
                start, end = 0, len(text)
            elif re.match(r'^\[\d+, ?\d+\]$', text[start:end]):
                text = json.dumps({'score': json.loads(text[start:end]), "reasoning": "list of scores"})
                start, end = 0, len(text)
            elif _is_int_between_0_and_10(text):
                text = json.dumps({'score': [int(text)], "reasoning": "a number"})
                start, end = 0, len(text)
            else:
                return False
    if start != -1 and end != -1 and start != end:
        json_str = text[start:end].strip().replace("\n", "")
        try:
            data = json.loads(json_str)
            if not isinstance(data['score'], list):
                data['score'] = [data['score']]
        except Exception:
            try:
                return json.loads(_fix_json(json_str))
            except Exception:
                return False
        return data
    return False


def gedit_min_score(parsed: Any) -> Optional[float]:
    """viescore: SC = min(score list), PQ = min(score list)."""
    if not isinstance(parsed, dict) or 'score' not in parsed:
        return None
    scores = parsed['score'] if isinstance(parsed['score'], list) else [parsed['score']]
    try:
        values = [float(s) for s in scores]
    except (TypeError, ValueError):
        return None
    if not values or any(math.isnan(v) for v in values):
        return None
    return min(values)


def gedit_columns(sc: Optional[float], pq: Optional[float]) -> Dict[str, Optional[float]]:
    if sc is None or pq is None:
        return {"semantic_consistency": sc, "perceptual_quality": pq, "overall": None}
    if sc < 0 or pq < 0:
        raise ValueError("GEdit scores must be non-negative")
    return {"semantic_consistency": sc, "perceptual_quality": pq, "overall": math.sqrt(sc * pq)}


# --------------------------------------------------------------------------- parsing: ImgEdit (step1_get_avgscore.py)
def imgedit_scores(text: str) -> List[int]:
    """Every line 'Name: <digits>' in order, as step1_get_avgscore.extract_scores_and_average."""
    scores = []
    for line in text.splitlines():
        parts = line.strip().split(': ')
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    return scores


def imgedit_columns(scores: Sequence[int], edit_type: str) -> Dict[str, Optional[float]]:
    positions = IMGEDIT_COLUMN_POSITIONS.get(edit_type)
    if positions is None:
        raise ValueError(f"Unknown ImgEdit edit_type: {edit_type!r}")
    average = round(sum(scores) / len(scores), 2) if scores else None
    if len(scores) != 3:
        cols = {c: None for c in positions}
    else:
        cols = {c: float(scores[i]) for c, i in positions.items()}
    cols["original_average"] = average
    return cols


# --------------------------------------------------------------------------- parsing: KRIS (metrics_common / metrics_knowledge)
_DEFAULT_PATTERNS = [r"([1-5])\s*/\s*5", r"([1-5])\s+out\s+of\s+5", r"\b([1-5])\b"]


def kris_score(text: str, score_key: str) -> Optional[int]:
    """JSON object containing score_key first, then '<key>: n', 'n/5', 'n out of 5', first standalone 1-5."""
    m = re.search(r"\{[^{}]*" + re.escape(score_key) + r"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            value = json.loads(m.group(0)).get(score_key)
            if value is not None:
                return int(value)
        except Exception:
            pass
    prefix = score_key.replace("_score", "")
    for pat in [prefix + r"[_\s]*score\s*[:：]?\s*([1-5])"] + _DEFAULT_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return int(m.group(1))
    return None


def kris_dual_scores(text: str) -> Dict[str, Optional[int]]:
    """metrics_knowledge.extract_dual_scores (JSON block, then 'instruction_score: n' / 'knowledge_score: n')."""
    m = re.search(r"\{[^{}]*instruction_score[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return {k: (int(data[k]) if data.get(k) is not None else None)
                    for k in ("instruction_score", "knowledge_score")}
        except Exception:
            pass
    out = {}
    for k in ("instruction", "knowledge"):
        m = re.search(k + r"[_\s]*score\s*[:：]\s*([1-5])", text, re.IGNORECASE)
        out[f"{k}_score"] = int(m.group(1)) if m else None
    return out


def kris_columns(answers: Mapping[str, str]) -> Dict[str, Optional[float]]:
    cols: Dict[str, Optional[float]] = {
        "visual_consistency": kris_score(answers["consistency"], "consistency_score"),
        "visual_quality": kris_score(answers["quality"], "quality_score"),
    }
    if "dual" in answers:
        dual = kris_dual_scores(answers["dual"])
        cols["instruction_following"] = dual["instruction_score"]
        cols["knowledge_plausibility"] = dual["knowledge_score"]
    else:
        cols["instruction_following"] = kris_score(answers["instruction"], "instruction_score")
    return {k: (float(v) if v is not None else None) for k, v in cols.items()}
