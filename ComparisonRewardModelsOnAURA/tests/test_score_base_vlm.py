"""Tests for scripts/score_base_vlm.py with a fake next-token judge (no weights, no GPU, no real images)."""

import json

import pytest

import bench_inputs as bi
import score_base_vlm as sb

PROMPTS = {"system_prompt": "SYS",
           "metrics": {"Fidelity": {"images": ["source", "edited"], "user_prompt": "F: {instruction}"},
                       "Aesthetic": {"images": ["source", "edited"], "user_prompt": "A: {instruction}"}}}
ROWS = [
    {"sample_id": "F1", "metric": "Fidelity", "prompt": "add a hat", "src_image": "s", "edited_images": ["a", "b"],
     "human_label": "A", "votes": {"A": 2, "B": 1, "Tie": 0}},
    {"sample_id": "A1", "metric": "Aesthetic", "prompt": "blur", "src_image": "s", "edited_images": ["a", "b"]},
    {"sample_id": "X1", "metric": "Unmapped", "prompt": "p", "src_image": "s", "edited_images": ["a", "b"]},
]
KEYS = {"Fidelity": "Fidelity", "Aesthetic": "Aesthetic", "Unmapped": None}


def one_hot(digit, mass=1.0):
    return [mass if d == digit else 0.0 for d in range(1, 10)]


class FakeJudge:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, messages):
        self.calls.append(messages)
        return {"digit_probs": self.outputs.pop(0), "top_token": "7", "num_input_tokens": 100}


def run(tmp_path, judge, rows=ROWS):
    path = tmp_path / "scores.jsonl"
    counts = sb.score_rows(judge, PROMPTS, rows, KEYS, path, load_images=lambda r: ("SRC", ["IMG_A", "IMG_B"]),
                           log=lambda m: None)
    return counts, bi.load_jsonl(path) if path.exists() else []


def test_build_messages_puts_system_prompt_then_source_edited_and_instruction():
    messages = sb.build_messages(PROMPTS, "Fidelity", "add a hat", "SRC", "EDIT")
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "SYS"}]}
    assert messages[1]["content"] == [{"type": "image", "image": "SRC"}, {"type": "image", "image": "EDIT"},
                                      {"type": "text", "text": "F: add a hat"}]


def test_build_messages_rejects_unknown_prompt_key():
    with pytest.raises(ValueError, match="no base VLM prompt"):
        sb.build_messages(PROMPTS, "Realism", "x", "S", "E")


def test_expected_digit_score_is_mean_of_renormalised_digit_probabilities():
    result = sb.expected_digit_score([0, 0, 0, 0, 0.1, 0, 0, 0, 0.1])
    assert result["score"] == pytest.approx(7.0) and result["digit_mass"] == pytest.approx(0.2)
    assert sum(result["digit_probs"]) == pytest.approx(1.0)
    assert sb.expected_digit_score(one_hot(1))["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("probs", [[0.0] * 9, [0.1] * 8, [float("nan")] + [0.1] * 8, [-0.1] + [0.1] * 8])
def test_expected_digit_score_rejects_zero_wrong_length_nan_or_negative(probs):
    with pytest.raises(ValueError):
        sb.expected_digit_score(probs)


def test_score_rows_scores_image_a_then_b_and_skips_unmapped_metrics(tmp_path):
    judge = FakeJudge([one_hot(8), one_hot(3), one_hot(5), one_hot(5)])
    counts, records = run(tmp_path, judge)
    assert counts == {"scored": 2, "skipped_done": 0, "not_applicable": 1}
    assert [(r["sample_id"], r["dimension"]) for r in records] == [("F1", "Fidelity"), ("A1", "Aesthetic")]
    assert (records[0]["score_a"], records[0]["score_b"], records[0]["margin"]) == pytest.approx((8.0, 3.0, 5.0))
    assert [c[1]["content"][1]["image"] for c in judge.calls[:2]] == ["IMG_A", "IMG_B"]
    assert records[1]["margin"] == pytest.approx(0.0)


def test_score_record_is_self_contained(tmp_path):
    _, records = run(tmp_path, FakeJudge([one_hot(8, 0.9), one_hot(3, 0.5)]), ROWS[:1])
    record = records[0]
    for field in sb.BENCH_FIELDS_TO_KEEP:
        assert record[field] == ROWS[0].get(field), field
    assert record["digit_mass_a"] == pytest.approx(0.9) and record["digit_probs_b"][2] == pytest.approx(1.0)
    assert record["top_token_a"] == "7" and record["num_input_tokens_b"] == 100


def test_score_rows_resumes_without_rescoring_done_rows(tmp_path):
    (tmp_path / "scores.jsonl").write_text(json.dumps({"sample_id": "F1", "dimension": "Fidelity"}) + "\n")
    counts, records = run(tmp_path, FakeJudge([one_hot(5), one_hot(5)]))
    assert counts["skipped_done"] == 1 and counts["scored"] == 1 and len(records) == 2


def test_score_rows_with_empty_row_list_writes_nothing(tmp_path):
    counts, records = run(tmp_path, FakeJudge([]), [])
    assert counts["scored"] == 0 and records == []


def test_score_rows_rejects_mapping_key_without_prompt(tmp_path):
    with pytest.raises(ValueError, match="no base VLM prompt"):
        sb.score_rows(FakeJudge([]), PROMPTS, ROWS[:1], {"Fidelity": "Realism"}, tmp_path / "s.jsonl",
                      load_images=lambda r: ("S", ["A", "B"]))


def test_load_pair_images_keeps_original_sizes(tmp_path):
    from PIL import Image

    Image.new("RGB", (40, 30)).save(tmp_path / "src.png")
    Image.new("L", (80, 60)).save(tmp_path / "a.png")
    Image.new("RGB", (20, 10)).save(tmp_path / "b.png")
    source, edited = sb.load_pair_images({"src_image": "src.png", "edited_images": ["a.png", "b.png"]}, tmp_path, tmp_path)
    assert source.size == (40, 30) and [im.size for im in edited] == [(80, 60), (20, 10)]
    assert all(im.mode == "RGB" for im in edited)


class FakeTokenizer:
    def __init__(self, split_digit=None):
        self.split_digit = split_digit

    def encode(self, text, add_special_tokens=False):
        return [1, 2] if text == self.split_digit else [100 + int(text)]


def test_digit_token_ids_one_token_per_digit_and_rejects_split_digit():
    assert sb.digit_token_ids(FakeTokenizer()) == [101 + i for i in range(9)]
    with pytest.raises(ValueError, match="expected 1"):
        sb.digit_token_ids(FakeTokenizer(split_digit="7"))


def test_check_thinking_disabled_requires_closed_think_block_only_when_disabled():
    sb.check_thinking_disabled("<|im_start|>assistant\n<think>\n\n</think>\n\n", {"enable_thinking": False})
    sb.check_thinking_disabled("<|im_start|>assistant\n", {})
    with pytest.raises(RuntimeError):
        sb.check_thinking_disabled("<|im_start|>assistant\n<think>\n", {"enable_thinking": False})


def test_original_base_vlms_use_reward_models_dir_and_qwen35_disables_thinking_in_gemma4_env():
    # 2026-11-16 (user): Qwen3-VL-Reranker-8B and Gemma-4-31B live elsewhere; their paths have their own test.
    for name in ("Qwen2.5-VL-7B", "Qwen3-VL-8B", "Qwen3.5-9B"):
        assert str(sb.VARIANTS[name]["model_path"]).startswith(str(bi.MODELS_DIR))
    assert sb.VARIANTS["Qwen3.5-9B"] == dict(sb.VARIANTS["Qwen3.5-9B"], conda_env="gemma4",
                                             chat_template_kwargs={"enable_thinking": False})
    assert {sb.VARIANTS[m]["conda_env"] for m in ("Qwen2.5-VL-7B", "Qwen3-VL-8B")} == {"qwen_vl"}


def test_real_prompt_file_covers_every_mapped_metric_of_every_base_vlm():
    prompts = sb.load_prompts()
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    for model in sb.VARIANTS:
        keys = {k for k in bi.dimension_by_bench_metric(mapping, model, "-").values() if k is not None}
        assert keys and keys <= set(prompts["metrics"])


def test_load_prompts_rejects_missing_instruction_or_wrong_images(tmp_path):
    bad = {"system_prompt": "S", "metrics": {"Fidelity": {"images": ["edited"], "user_prompt": "{instruction}"}}}
    (tmp_path / "p.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="expected images"):
        sb.load_prompts(tmp_path / "p.json")
    bad["metrics"]["Fidelity"] = {"images": ["source", "edited"], "user_prompt": "no placeholder"}
    (tmp_path / "p.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="placeholder"):
        sb.load_prompts(tmp_path / "p.json")


def test_parse_args_rejects_unknown_model_and_has_no_split_options():
    with pytest.raises(SystemExit):
        sb.parse_args(["--model", "Qwen2.5-VL-72B"])
    with pytest.raises(SystemExit):
        sb.parse_args(["--model", "Qwen3-VL-8B", "--num-shards", "2"])


def test_launcher_picks_gemma4_only_for_qwen35_and_has_no_shard_options():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_base_vlm.slurm").read_text()
    assert "Qwen3.5-9B) CONDA_ENV=gemma4" in launcher and "Qwen2.5-VL-7B|Qwen3-VL-8B) CONDA_ENV=qwen_vl" in launcher
    assert "SHARD" not in launcher.upper()


def test_check_thinking_disabled_accepts_closed_gemma_thought_channel_and_rejects_open_one():
    sb.check_thinking_disabled("<|turn>model\n<|channel>thought\n<channel|>", {"enable_thinking": False})
    with pytest.raises(RuntimeError):
        sb.check_thinking_disabled("<|turn>model\n<|channel>thought\n", {"enable_thinking": False})


def test_pipeline_backbone_and_gemma_variants_use_their_local_weights_and_envs():
    reranker, gemma = sb.VARIANTS["Qwen3-VL-Reranker-8B"], sb.VARIANTS["Gemma-4-31B"]
    assert reranker["model_path"] == bi.COMPARISON_ROOT.parent / "Models" / "Qwen3-VL-Reranker-8B"
    assert (reranker["conda_env"], reranker["chat_template_kwargs"]) == ("qwen_vl", {})
    assert str(gemma["model_path"]) == "/datastore/shared/models/gemma-4-31B-it"
    assert (gemma["conda_env"], gemma["chat_template_kwargs"]) == ("gemma4", {"enable_thinking": False})


@pytest.mark.parametrize("model", ["Qwen3-VL-Reranker-8B", "Gemma-4-31B"])
def test_new_base_vlm_rows_score_all_six_metrics_with_their_own_prompt(model):
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    row = next(r for r in mapping["models"] if r["model"] == model)
    assert (row["family"], row["pretrained"]) == ("BaseVLM", "-")
    assert row["scores"] == {c: c for c in ("Aesthetic", "Realism", "Fidelity", "BC", "FC", "SC")}
    assert sb.parse_args(["--model", model]).model == model


def test_launcher_sets_env_of_new_models_and_logs_the_actual_gpu_type():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_base_vlm.slurm").read_text()
    assert "Qwen3-VL-Reranker-8B) CONDA_ENV=qwen_vl" in launcher and "Gemma-4-31B) CONDA_ENV=gemma4" in launcher
    assert "gpu:2g.35gb:1\" \\" not in launcher and '"${GRES:-unknown}"' in launcher
