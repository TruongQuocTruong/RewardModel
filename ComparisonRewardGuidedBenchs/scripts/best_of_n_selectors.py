"""Selectors: score the 9 rollouts of each sample; the highest score is the Best-of-N choice.

Every selector returns one record per (sample, seed): {"score": float, ...extra}. Selectors never see
judge prompts or judge scores.

  Qwen2.5-VL-7B               pointwise prompt (SELECTOR_PROMPT_QWEN), score = expected rating from the
                              probabilities of the rating tokens "1".."5" at the first generated position
  ImageReward                 ImageReward-v1.0 score(instruction, edited image)
  CLIP                        cosine(CLIP text(instruction), CLIP image(edited image)), ViT-L/14
  FIRE-Reward-Stage2-Teacher  Teacher adapter of stage2_ckpt/checkpoint-621.pt (default); each forward = one pair
                              (source, instruction, image A, image B) as in training; in eval mode the score of
                              an image depends only on (source, instruction, image), so the 9 seeds are scored in
                              5 forwards (0-5, 10-15, 20-25, 30-35, 40-0). All six metric scores, relevance and
                              Tie probabilities are kept; the per-image score rule is applied in select_best.py.
                              --model-path selects the checkpoint (e.g. the updated Stage 2 checkpoint-927).
  EditScore                   EditScore-Qwen3-VL-8B-Instruct LoRA on Qwen3-VL-8B, official EditScore.evaluate
                              (score_range 25, temperature 0.7, seed 42, 1 pass; edited image resized to the
                              source size as in EditScore/evaluation.py); score = "overall" = sqrt(SC * PQ),
                              SC = min(editing success, overediting) / 2.5, PQ = min(naturalness, artifacts) / 2.5.
  EditReward                  EditReward-Qwen2.5-VL-7B, official EditRewardInferencer.reward (reward_dim from the repo config = overall_detail, ranknet_multi_head,
                              config EditReward-Qwen2.5-7B-VL.yaml with the local backbone path, pooling "mean" of
                              the two heads); score = mu of the pooled output. The 9 seeds go in one batch.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TABLE4_SCRIPTS = REPO_ROOT / "ComparisonRewardModelsOnOtherBenchs" / "scripts"
MODELS = Path("/datastore/tuanld/Truongtq/RewardModels")

SELECTORS = ("Qwen2.5-VL-7B", "ImageReward", "CLIP", "FIRE-Reward-Stage2-Teacher", "EditScore", "EditReward")
EDITSCORE_REPO = MODELS / "EditScore"          # clone of the EditScore repo (not modified)
EDITREWARD_REPO = MODELS / "EditReward"        # clone of the EditReward repo (not modified)
# The EditScore LoRA weights were downloaded for the AURA-Bench comparison (lora_download_manifest.json there).
EDITSCORE_LORA = REPO_ROOT / "ComparisonRewardModelsOnAURA" / "EditScore" / "weights" / "EditScore-Qwen3-VL-8B-Instruct"
EDITSCORE_BACKBONE = MODELS / "Qwen3-VL-8B"
EDITREWARD_BACKBONE = MODELS / "Qwen2.5-VL-7B"
EDITREWARD_CONFIG = EDITREWARD_REPO / "EditReward" / "config" / "EditReward-Qwen2.5-7B-VL.yaml"
EDITSCORE_SETTINGS = {"backbone": "qwen3vl", "score_range": 25, "temperature": 0.7, "seed": 42, "num_pass": 1}
DEFAULT_PATHS = {
    "Qwen2.5-VL-7B": str(MODELS / "Qwen2.5-VL-7B"),
    "ImageReward": str(MODELS / "ImageReward"),
    "CLIP": str(MODELS / "CLIP-ViT-L-14"),
    "FIRE-Reward-Stage2-Teacher": str(REPO_ROOT / "TrainingRewardModel/Qwen3-VL-Embedding/output_training/"
                                      "stage2_ckpt/checkpoint-621.pt"),  # was best_stage2_ckpt/ (folder renamed)
    "EditScore": str(EDITSCORE_LORA),
    "EditReward": str(MODELS / "EditReward-Qwen2.5-VL-7B"),
}
BERT_TOKENIZER_PATH = str(MODELS / "bert-base-uncased")
FIRE_BASE_MODEL = str(REPO_ROOT / "Models" / "Qwen3-VL-Reranker-8B")

# Pending user approval (AGENTS.md, open decision 3). Not a judge prompt.
SELECTOR_PROMPT_QWEN = (
    "The first image is the source image and the second image is the result of editing it with the "
    "instruction: \"{instruction}\".\n"
    "Rate the edited image from 1 (worst) to 5 (best), considering how well it follows the instruction, "
    "how well it keeps the parts that should not change, and its visual quality.\n"
    "Answer with a single digit."
)
RATING_TOKENS = ("1", "2", "3", "4", "5")


def pair_seeds(seeds: Sequence[int]) -> List[tuple]:
    """Pair consecutive seeds; an odd last seed is paired with the first one."""
    seeds = list(seeds)
    if not seeds:
        return []
    if len(seeds) == 1:
        return [(seeds[0], seeds[0])]
    pairs = [(seeds[i], seeds[i + 1]) for i in range(0, len(seeds) - 1, 2)]
    if len(seeds) % 2:
        pairs.append((seeds[-1], seeds[0]))
    return pairs


def expected_rating(probabilities: Sequence[float]) -> float:
    """Expected rating over RATING_TOKENS from (unnormalised) probabilities."""
    total = float(sum(probabilities))
    if not total > 0:
        raise ValueError("Rating token probabilities sum to zero")
    return sum((i + 1) * p for i, p in enumerate(probabilities)) / total


# --------------------------------------------------------------------------- model wrappers
class QwenSelector:
    name = "Qwen2.5-VL-7B"

    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch, self.device = torch, device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        ids = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in RATING_TOKENS]
        if any(i is None or i == self.processor.tokenizer.unk_token_id for i in ids):
            raise ValueError("Rating digits are not single tokens")
        self.rating_ids = ids

    def score_sample(self, sample: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        rows = []
        for item in items:
            with Image.open(item["source_image"]) as s, Image.open(item["edited_image"]) as e:
                images = [s.convert("RGB"), e.convert("RGB")]
            messages = [{"role": "user", "content": [
                {"type": "image", "image": images[0]}, {"type": "image", "image": images[1]},
                {"type": "text", "text": SELECTOR_PROMPT_QWEN.format(instruction=item["instruction"])}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            vision, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=vision, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[0, -1].float()
            probs = self.torch.softmax(logits, dim=-1)[self.rating_ids].cpu().tolist()
            rows.append({"seed": item["seed"], "score": expected_rating(probs), "rating_probabilities": probs})
        return rows


def patch_transformers_for_imagereward() -> None:
    """ImageReward 1.5 (BLIP med.py) imports three helpers that transformers 4.57 keeps only in pytorch_utils."""
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


class ImageRewardSelector:
    name = "ImageReward"

    def __init__(self, model_dir: str, device: str = "cuda"):
        import transformers
        patch_transformers_for_imagereward()
        import ImageReward as RM  # package image-reward
        original = transformers.BertTokenizer.from_pretrained.__func__

        def local_bert(cls, name, *args, **kwargs):  # ImageReward hard-codes "bert-base-uncased"
            if name == "bert-base-uncased":
                name, kwargs["local_files_only"] = BERT_TOKENIZER_PATH, True
            return original(cls, name, *args, **kwargs)

        transformers.BertTokenizer.from_pretrained = classmethod(local_bert)
        model_dir = Path(model_dir)
        self.model = RM.load(str(model_dir / "ImageReward.pt"), device=device,
                             med_config=str(model_dir / "med_config.json"))

    def score_sample(self, sample, items):
        return [{"seed": it["seed"], "score": float(self.model.score(it["instruction"], it["edited_image"]))}
                for it in items]


class ClipSelector:
    name = "CLIP"

    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.device = torch, device
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)

    def score_sample(self, sample, items):
        from PIL import Image
        images = []
        for it in items:
            with Image.open(it["edited_image"]) as im:
                images.append(im.convert("RGB"))
        inputs = self.processor(text=[sample["instruction"]], images=images, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with self.torch.inference_mode():
            text = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        text = text / text.norm(dim=-1, keepdim=True)
        image = image / image.norm(dim=-1, keepdim=True)
        cos = (image @ text.T)[:, 0].float().cpu().tolist()
        return [{"seed": it["seed"], "score": c} for it, c in zip(items, cos)]


class FireSelector:
    name = "FIRE-Reward-Stage2-Teacher"

    def __init__(self, checkpoint: str, device: str = "cuda", base_model_path: str = FIRE_BASE_MODEL):
        import torch
        if str(TABLE4_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(TABLE4_SCRIPTS))
        from score_bench import Unit, load_scorer
        self.Unit = Unit
        self.score_units, self.description = load_scorer(
            "stage2_teacher", Path(checkpoint).resolve(), Path(base_model_path).resolve(), torch.device(device))

    def score_sample(self, sample, items):
        by_seed = {it["seed"]: it for it in items}
        out: Dict[int, Dict[str, Any]] = {}
        for a, b in pair_seeds([it["seed"] for it in items]):
            unit = self.Unit(sample["source_image"], sample["instruction"],
                             by_seed[a]["edited_image"], by_seed[b]["edited_image"])
            result = self.score_units([unit])[0]
            for seed, side in ((a, "A"), (b, "B")):
                if seed not in out:
                    out[seed] = {"seed": seed, "score": result[f"image_scores_{side}"],
                                 "metric_scores": result[f"metric_scores_{side}"],
                                 "relevance_probs": result["relevance_probs"], "tie_probs": result["tie_probs"],
                                 "paired_with": b if side == "A" else a}
        return [out[it["seed"]] for it in items]


def capped_size(size, max_pixels):
    """(w, h) scaled down (aspect kept) so that w * h <= max_pixels; unchanged when max_pixels is None or already small."""
    w, h = size
    if max_pixels is None or w * h <= max_pixels:
        return (w, h)
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    scale = math.sqrt(max_pixels / (w * h))
    return (max(1, int(w * scale)), max(1, int(h * scale)))


def editscore_guessed(output: Mapping[str, Any]) -> bool:
    """EditScore guesses a random score when it cannot parse the answer (mllm_output_to_dict give_up_parsing)."""
    return any(str(output.get(k, "")).startswith("guess_if_cannot_parse") for k in ("SC_reasoning", "PQ_reasoning"))


class EditScoreSelector:
    name = "EditScore"

    def __init__(self, lora_path: str, device: str = "cuda", backbone_path: str = str(EDITSCORE_BACKBONE),
                 max_source_pixels: int | None = None):
        if str(EDITSCORE_REPO) not in sys.path:
            sys.path.insert(0, str(EDITSCORE_REPO))
        from editscore import EditScore
        for path in (lora_path, backbone_path):
            if not Path(path).is_dir():
                raise FileNotFoundError(f"missing local weights: {path}")
        self.max_source_pixels = max_source_pixels
        self.scorer = EditScore(model_name_or_path=str(backbone_path), lora_path=str(lora_path), **EDITSCORE_SETTINGS)
        self.description = {**EDITSCORE_SETTINGS, "backbone_path": str(backbone_path), "lora_path": str(lora_path),
                            "repo": str(EDITSCORE_REPO), "max_source_pixels": max_source_pixels,
                            "score": "EditScore.evaluate overall = sqrt(SC * PQ)"}

    def score_sample(self, sample, items):
        from PIL import Image
        with Image.open(sample["source_image"]) as s:
            source = s.convert("RGB")
        source = source.resize(capped_size(source.size, self.max_source_pixels))
        rows = []
        for it in items:
            with Image.open(it["edited_image"]) as e:
                edited = e.convert("RGB").resize(source.size)  # EditScore/evaluation.py
            out = self.scorer.evaluate([source, edited], it["instruction"])
            rows.append({"seed": it["seed"], "score": float(out["overall"]),
                         "prompt_following": float(out["prompt_following"]), "consistency": float(out["consistency"]),
                         "perceptual_quality": float(out["perceptual_quality"]), "guessed": editscore_guessed(out),
                         "SC_reasoning": out.get("SC_reasoning"), "PQ_reasoning": out.get("PQ_reasoning"),
                         "scored_size": list(source.size)})
        return rows


def editreward_local_config(yaml_text: str, backbone_path: str, output_dir: str) -> str:
    """Repo yaml with model_name_or_path = local backbone and output_dir set; nothing else changes."""
    for key, value in (("model_name_or_path", backbone_path), ("output_dir", output_dir)):
        pattern = re.compile(rf"^{key}:.*$", re.MULTILINE)
        if len(pattern.findall(yaml_text)) != 1:
            raise ValueError(f"expected exactly one '{key}:' line in the EditReward config")
        yaml_text = pattern.sub(f'{key}: "{value}"', yaml_text)
    return yaml_text


def editreward_reward_dim(yaml_text: str) -> str:
    """reward_dim of the repo yaml (the prompt the model was trained with); EditRewardInferencer does not read it
    from the config and defaults to "dim1" (instruction-following prompt), so it must be passed explicitly."""
    found = re.findall(r"^reward_dim:\s*[\"']?([A-Za-z0-9_]+)[\"']?\s*(?:#.*)?$", yaml_text, re.MULTILINE)
    if len(found) != 1:
        raise ValueError("expected exactly one 'reward_dim:' line in the EditReward config")
    if found[0] not in {"dim1", "dim2", "overall", "overall_detail"}:
        raise ValueError(f"unknown EditReward reward_dim: {found[0]!r}")
    return found[0]


def editreward_rows(rewards, items) -> List[Dict[str, Any]]:
    """rewards[i] = [mu, log sigma] of items[i]; score = mu."""
    if len(rewards) != len(items):
        raise ValueError(f"EditReward returned {len(rewards)} outputs for {len(items)} images")
    rows = []
    for it, values in zip(items, rewards):
        values = [float(v) for v in values]
        if len(values) != 2 or any(math.isnan(v) for v in values):
            raise ValueError(f"expected [mu, log sigma] without NaN, got {values}")
        rows.append({"seed": it["seed"], "score": values[0], "log_sigma": values[1]})
    return rows


class EditRewardSelector:
    name = "EditReward"

    def __init__(self, checkpoint_path: str, device: str = "cuda", backbone_path: str = str(EDITREWARD_BACKBONE),
                 work_dir: str | None = None):
        import torch
        for path in (checkpoint_path, backbone_path):
            if not Path(path).is_dir():
                raise FileNotFoundError(f"missing local weights: {path}")
        if work_dir is None:
            raise ValueError("EditReward needs work_dir for its local config copy")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_path = work_dir / "editreward_config.yaml"
        repo_yaml = EDITREWARD_CONFIG.read_text(encoding="utf-8")
        reward_dim = editreward_reward_dim(repo_yaml)
        config_path.write_text(editreward_local_config(repo_yaml, str(backbone_path), str(work_dir)), encoding="utf-8")
        if str(EDITREWARD_REPO) not in sys.path:
            sys.path.insert(0, str(EDITREWARD_REPO))
        from EditReward.inference_edit import EditRewardInferencer
        self.torch = torch
        self.inferencer = EditRewardInferencer(config_path=str(config_path), checkpoint_path=str(checkpoint_path),
                                               device=device, reward_dim=reward_dim,
                                               rm_head_type="ranknet_multi_head")
        self.description = {"checkpoint_path": str(checkpoint_path), "backbone_path": str(backbone_path),
                            "repo_config": str(EDITREWARD_CONFIG), "local_config": str(config_path),
                            "reward_dim": reward_dim,
                            "pooling_strategy": getattr(self.inferencer.model, "pooling_strategy", None),
                            "score": "mu of EditRewardInferencer.reward (heads pooled by the config)"}

    def score_sample(self, sample, items):
        n = len(items)
        with self.torch.inference_mode():
            rewards = self.inferencer.reward([sample["instruction"]] * n, [sample["source_image"]] * n,
                                             [it["edited_image"] for it in items])
        return editreward_rows(rewards.float().cpu().tolist(), items)


def load_selector(name: str, path: str | None = None, device: str = "cuda", work_dir: str | None = None,
                  max_source_pixels: int | None = None):
    path = path or DEFAULT_PATHS[name]
    classes = {"Qwen2.5-VL-7B": QwenSelector, "ImageReward": ImageRewardSelector, "CLIP": ClipSelector,
               "FIRE-Reward-Stage2-Teacher": FireSelector, "EditScore": EditScoreSelector,
               "EditReward": EditRewardSelector}
    if name not in classes:
        raise ValueError(f"Unknown selector: {name!r}")
    if name == "EditReward":
        return EditRewardSelector(path, device, work_dir=work_dir)
    if name == "EditScore":
        return EditScoreSelector(path, device, max_source_pixels=max_source_pixels)
    return classes[name](path, device)
