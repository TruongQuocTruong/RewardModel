"""Tests for scripts/score_spatialreward.py with a fake generator (no weights, no GPU, no real images)."""

import json
from types import SimpleNamespace

import pytest

import bench_inputs as bi
import comparison_table as ct
import score_spatialreward as ss

ROWS = [
    {"sample_id": "F1", "metric": "Fidelity", "prompt": "add a hat", "src_image": "s", "edited_images": ["a", "b"],
     "human_label": "A", "votes": {"A": 3, "B": 0, "Tie": 0}},
    {"sample_id": "A1", "metric": "Aesthetic", "prompt": "blur", "src_image": "s", "edited_images": ["a", "b"]},
    {"sample_id": "X1", "metric": "Unmapped", "prompt": "p", "src_image": "s", "edited_images": ["a", "b"]},
]
DIMENSIONS = {"Fidelity": "Semantic Consistency", "Aesthetic": "Perceptual Quality", "Unmapped": None}
SC_TEXT = '{"edit_region": [{"id": 0, "label": "hat", "bbox_2d": [1, 2, 3, 4]}], "reasoning": "ok", "score": [20, 15]}'
PQ_TEXT = '{"reasoning": "fine", "score": [25, 10]}'


@pytest.fixture(scope="module")
def module():
    pytest.importorskip("qwen_vl_utils")
    pytest.importorskip("datasets")
    return ss.load_dataset_module()


class FakeGenerator:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages):
        self.calls.append(messages)
        return {"text": self.texts.pop(0), "num_input_tokens": 900, "num_output_tokens": 60}


def run(tmp_path, module, generator, rows=ROWS):
    counts = ss.score_rows(generator, module, rows, DIMENSIONS, tmp_path / "scores.jsonl", tmp_path / "raw.jsonl",
                           load_images=lambda r: ("SRC", ["IMG_A", "IMG_B"]), log=lambda m: None)
    path = tmp_path / "scores.jsonl"
    return counts, (bi.load_jsonl(path) if path.exists() else []), tmp_path / "raw.jsonl"


def test_settings_follow_run_sh_and_local_weights():
    assert (ss.MAX_PIXELS, ss.MIN_PIXELS, ss.TEMPERATURE, ss.MAX_NEW_TOKENS, ss.SEED) == (589824, 3136, 0.0, 4096, 42)
    run_sh = (ss.SPATIALREWARD_REPO / "eval" / "EditReward-Bench" / "run.sh").read_text()
    assert "MAX_PIXELS=589824" in run_sh and "MIN_PIXELS=3136" in run_sh and "TEMPERATURE=0.0" in run_sh
    assert "--max_tokens 4096" in run_sh
    assert ss.MODEL_PATH == bi.MODELS_DIR / "SpatialReward-8B"


def test_mapping_dimensions_are_spatialreward_dimensions_and_result_folder_matches_the_table():
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    dimensions = {d for d in bi.dimension_by_bench_metric(mapping, "SpatialReward", "Qwen3-VL-8B").values() if d}
    assert dimensions == set(ss.DIMENSIONS)
    assert ct.result_folder_name(ss.MODEL_NAME, ss.PRETRAINED) == "SpatialReward_Qwen3-VL-8B"


def test_build_messages_sc_uses_source_then_edited_and_repo_prompt(module):
    messages = ss.build_messages(module, "Semantic Consistency", "add a hat", "SRC", "EDIT")
    content = messages[0]["content"]
    assert [c.get("image") for c in content[:2]] == ["SRC", "EDIT"]
    assert all(c["max_pixels"] == 589824 and c["min_pixels"] == 3136 for c in content[:2])
    assert content[2]["text"] == module.SC_PROMPT_TEMPLATE.format(instruction="add a hat")
    assert "Editing instruction: add a hat" in content[2]["text"]


def test_build_messages_pq_uses_only_the_edited_image(module):
    content = ss.build_messages(module, "Perceptual Quality", "ignored", "SRC", "EDIT")[0]["content"]
    assert [c["type"] for c in content] == ["image", "text"] and content[0]["image"] == "EDIT"
    assert content[1]["text"] == module.PQ_PROMPT


def test_build_messages_without_pixels_matches_the_repo_messages(module):
    content = ss.build_messages(module, "Perceptual Quality", "x", "S", "E", pixels_in_messages=False)[0]["content"]
    assert content[0] == {"type": "image", "image": "E"}


def test_build_messages_rejects_unknown_dimension(module):
    with pytest.raises(ValueError, match="unknown SpatialReward dimension"):
        ss.build_messages(module, "Visual Quality", "x", "S", "E")


def test_pixels_in_messages_make_process_vision_info_apply_max_pixels():
    qwen_vl_utils = pytest.importorskip("qwen_vl_utils")
    from PIL import Image

    big = Image.new("RGB", (2000, 1500))
    with_limit = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [ss.image_entry(big)]}], image_patch_size=16)[0][0]
    without = qwen_vl_utils.process_vision_info(
        [{"role": "user", "content": [ss.image_entry(big, pixels_in_messages=False)]}], image_patch_size=16)[0][0]
    assert with_limit.size[0] * with_limit.size[1] <= ss.MAX_PIXELS
    assert without.size[0] * without.size[1] > ss.MAX_PIXELS


def test_dimension_score_equals_the_repo_min_aggregation(module):
    helper = SimpleNamespace(score_aggregation="min", weighted_power_params=None)
    scores = module.EditScoreDataset.calculate_scores(helper, {"score": [20, 15]}, {"score": [25, 10]})
    assert ss.dimension_score([20, 15]) == pytest.approx(min(scores["prompt_following"], scores["consistency"]))
    assert ss.dimension_score([25, 10]) == pytest.approx(scores["perceptual_quality"]) == pytest.approx(4.0)


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], [float("nan"), 1.0]])
def test_dimension_score_rejects_wrong_length_or_nan(bad):
    with pytest.raises(ValueError):
        ss.dimension_score(bad)


def test_score_rows_writes_min_scores_and_skips_unmapped_metrics(tmp_path, module):
    generator = FakeGenerator([SC_TEXT, '{"edit_region": [], "reasoning": "r", "score": [5, 25]}', PQ_TEXT, PQ_TEXT])
    counts, records, raw = run(tmp_path, module, generator)
    assert counts["scored"] == 2 and counts["not_applicable"] == 1 and counts["parse_failed_images"] == 0
    f1, a1 = records
    assert (f1["dimension"], f1["score_a"], f1["score_b"], f1["margin"]) == ("Semantic Consistency", 6.0, 2.0, 4.0)
    assert f1["edit_region_a"][0]["label"] == "hat" and f1["sub_score_names"] == ["editing success", "overediting"]
    assert (a1["dimension"], a1["score_a"], a1["margin"]) == ("Perceptual Quality", 4.0, 0.0)
    assert [c[0]["content"][1]["image"] for c in generator.calls[:2]] == ["IMG_A", "IMG_B"]
    assert generator.calls[2][0]["content"][0]["image"] == "IMG_A"
    assert bi.load_jsonl(raw)[0]["output_a"] == SC_TEXT


def test_unparseable_output_gets_the_repo_default_and_is_flagged(tmp_path, module):
    generator = FakeGenerator(["no json at all", '{"score": [10, 20], "reasoning": "x" "broken"}', PQ_TEXT, PQ_TEXT])
    counts, records, _ = run(tmp_path, module, generator)
    assert records[0]["status_a"] == "parse_failed" and records[0]["sub_scores_a"] == [12.5, 12.5]
    assert records[0]["status_b"] == "partial_json" and records[0]["sub_scores_b"] == [10.0, 20.0]
    assert counts["parse_failed_images"] == 1 and counts["partial_json_images"] == 1


def test_output_with_think_block_is_parsed_like_the_repo(tmp_path, module):
    generator = FakeGenerator(["<think>long</think>\n" + SC_TEXT, SC_TEXT])
    _, records, _ = run(tmp_path, module, generator, ROWS[:1])
    assert records[0]["status_a"] == "parsed" and records[0]["score_a"] == 6.0


def test_score_record_is_self_contained(tmp_path, module):
    _, records, _ = run(tmp_path, module, FakeGenerator([SC_TEXT, SC_TEXT]), ROWS[:1])
    for field in ss.BENCH_FIELDS_TO_KEEP:
        assert records[0][field] == ROWS[0].get(field), field
    assert records[0]["num_input_tokens_a"] == 900 and records[0]["num_output_tokens_b"] == 60


def test_score_rows_resumes_without_rescoring_done_rows(tmp_path, module):
    (tmp_path / "scores.jsonl").write_text(json.dumps({"sample_id": "F1", "dimension": "Semantic Consistency"}) + "\n")
    counts, records, _ = run(tmp_path, module, FakeGenerator([PQ_TEXT, PQ_TEXT]))
    assert counts["skipped_done"] == 1 and counts["scored"] == 1 and len(records) == 2


def test_score_rows_with_empty_row_list_writes_nothing(tmp_path, module):
    counts, records, _ = run(tmp_path, module, FakeGenerator([]), [])
    assert counts["scored"] == 0 and records == []


def test_score_rows_rejects_mapping_with_a_dimension_spatialreward_does_not_have(tmp_path, module):
    with pytest.raises(ValueError, match="not a SpatialReward dimension"):
        ss.score_rows(FakeGenerator([]), module, ROWS[:1], {"Fidelity": "Visual Quality"}, tmp_path / "s.jsonl",
                      tmp_path / "r.jsonl", load_images=lambda r: ("S", ["A", "B"]))


def test_parse_args_has_no_split_options():
    with pytest.raises(SystemExit):
        ss.parse_args(["--num-shards", "2"])


def test_launcher_uses_omnigen2_rl_env_normal_partition_and_has_no_shard_options():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_spatialreward.slurm").read_text()
    assert "CONDA_ENV=omnigen2-rl" in launcher and "scripts/score_spatialreward.py" in launcher
    assert "#SBATCH --partition=normal" in launcher and "#SBATCH --time=3-00:00:00" in launcher
    assert "SHARD" not in launcher.upper()
