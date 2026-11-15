"""Tests for scripts/score_editreward.py with a fake head runner (no weights, no GPU, no real images)."""

import json

import pytest

import bench_inputs as bi
import score_editreward as se

ROWS = [
    {"sample_id": "F1", "metric": "Fidelity", "prompt": "add a hat", "src_image": "s.png", "edited_images": ["a", "b"],
     "human_label": "A", "votes": {"A": 3, "B": 0, "Tie": 0}},
    {"sample_id": "R1", "metric": "Realism", "prompt": "blur", "src_image": "s.png", "edited_images": ["a", "b"]},
    {"sample_id": "B1", "metric": "Background Consistency", "prompt": "p", "src_image": "s.png", "edited_images": ["a", "b"]},
]
DIMENSIONS = {"Fidelity": "Instruction Following", "Realism": "Visual Quality", "Background Consistency": None}


class FakeHeads:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, prompts, sources, edited, batch_key):
        self.calls.append((prompts, sources, edited, batch_key))
        return self.outputs.pop(0)


def paths(row):
    return f"SRC/{row['src_image']}", [f"ED/{p}" for p in row["edited_images"]]


def run(tmp_path, heads, rows=ROWS):
    path = tmp_path / "scores.jsonl"
    counts = se.score_rows(heads, rows, DIMENSIONS, path, image_paths=paths, log=lambda m: None)
    return counts, bi.load_jsonl(path)


def test_instruction_following_uses_head_0_and_visual_quality_head_1():
    assert se.HEAD_OF_DIMENSION == {"Instruction Following": ("batch_dim1", 0), "Visual Quality": ("batch_dim2", 1)}


def test_pair_scores_sends_both_edited_images_in_one_batch_with_the_same_source_and_instruction():
    heads = FakeHeads([[[1.5, -0.2], [0.5, 0.1]]])
    result = se.pair_scores(heads, "S", ["A", "B"], "make it red", "Instruction Following")
    assert heads.calls == [(["make it red"] * 2, ["S", "S"], ["A", "B"], "batch_dim1")]
    assert result == [(1.5, -0.2), (0.5, 0.1)]


@pytest.mark.parametrize("outputs", [[[1.0, 0.0]], [[1.0], [2.0]], [[float("nan"), 0.0], [1.0, 0.0]]])
def test_pair_scores_rejects_wrong_count_shape_or_nan(outputs):
    with pytest.raises(ValueError):
        se.pair_scores(FakeHeads([outputs]), "S", ["A", "B"], "x", "Visual Quality")


def test_pair_scores_rejects_unknown_dimension():
    with pytest.raises(ValueError, match="unknown EditReward dimension"):
        se.pair_scores(FakeHeads([]), "S", ["A", "B"], "x", "Attribute Preservation")


def test_score_rows_writes_mu_as_score_and_skips_metrics_without_dimension(tmp_path):
    heads = FakeHeads([[[2.0, -1.0], [0.5, -0.5]], [[0.1, 0.0], [0.3, 0.2]]])
    counts, records = run(tmp_path, heads)
    assert counts == {"scored": 2, "skipped_done": 0, "not_applicable": 1}
    assert [(r["sample_id"], r["dimension"], r["head"]) for r in records] == [
        ("F1", "Instruction Following", "head_0"), ("R1", "Visual Quality", "head_1")]
    assert (records[0]["score_a"], records[0]["score_b"], records[0]["margin"]) == pytest.approx((2.0, 0.5, 1.5))
    assert (records[0]["log_sigma_a"], records[0]["log_sigma_b"]) == pytest.approx((-1.0, -0.5))
    assert heads.calls[1][1:] == (["SRC/s.png"] * 2, ["ED/a", "ED/b"], "batch_dim2")


def test_score_record_is_self_contained(tmp_path):
    _, records = run(tmp_path, FakeHeads([[[1.0, 0.0], [0.0, 0.0]]]), ROWS[:1])
    for field in se.BENCH_FIELDS_TO_KEEP:
        assert records[0][field] == ROWS[0].get(field), field


def test_score_rows_resumes_without_rescoring_done_rows(tmp_path):
    (tmp_path / "scores.jsonl").write_text(json.dumps({"sample_id": "F1", "dimension": "Instruction Following"}) + "\n")
    counts, records = run(tmp_path, FakeHeads([[[0.1, 0.0], [0.3, 0.2]]]))
    assert counts["skipped_done"] == 1 and counts["scored"] == 1 and len(records) == 2


def test_score_rows_with_empty_row_list_writes_nothing(tmp_path):
    counts, records = run(tmp_path, FakeHeads([]), [])
    assert counts["scored"] == 0 and records == []


def test_score_rows_rejects_mapping_with_a_dimension_editreward_does_not_have(tmp_path):
    with pytest.raises(ValueError, match="not an EditReward dimension"):
        se.score_rows(FakeHeads([]), ROWS[:1], {"Fidelity": "Semantic Consistency"}, tmp_path / "s.jsonl",
                      image_paths=paths)


class FakeModel:
    def __init__(self, result):
        self.pooling_strategy, self.result, self.kwargs = "mean", result, None

    def __call__(self, return_dict=True, **kwargs):
        self.kwargs = kwargs
        return self.result


class FakeTensor(list):
    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakeInferencer:
    def __init__(self, result):
        self.model = FakeModel(result)

    def prepare_batch(self, image_src, image_paths, prompts):
        return {"batch_dim1": ("dim1", image_src, image_paths, prompts), "batch_dim2": ("dim2",)}


def test_make_run_head_disables_head_averaging_and_passes_only_the_needed_batch():
    torch = pytest.importorskip("torch")
    inferencer = FakeInferencer([FakeTensor([[1.0, 0.0], [2.0, 0.0]])])
    run_head = se.make_run_head(inferencer, torch)
    assert inferencer.model.pooling_strategy is None
    assert run_head(["p", "p"], ["S", "S"], ["A", "B"], "batch_dim1") == [[1.0, 0.0], [2.0, 0.0]]
    assert list(inferencer.model.kwargs) == ["batch_dim1"] and inferencer.model.kwargs["batch_dim1"][0] == "dim1"


def test_make_run_head_rejects_averaged_or_multi_head_output():
    torch = pytest.importorskip("torch")
    run_head = se.make_run_head(FakeInferencer({"logits": "averaged"}), torch)
    with pytest.raises(RuntimeError, match="one head"):
        run_head(["p"] * 2, ["S"] * 2, ["A", "B"], "batch_dim2")


def test_local_config_text_replaces_only_backbone_and_output_dir():
    text = 'rm_head_type: "ranknet_multi_head"\nmodel_name_or_path: "Qwen/Qwen2.5-VL-7B-Instruct"\noutput_dir: \nbf16: True\n'
    out = se.local_config_text(text, "/m/Qwen2.5-VL-7B", "/r/out")
    assert out == 'rm_head_type: "ranknet_multi_head"\nmodel_name_or_path: "/m/Qwen2.5-VL-7B"\noutput_dir: "/r/out"\nbf16: True\n'


def test_local_config_text_rejects_missing_or_duplicate_key():
    with pytest.raises(ValueError, match="model_name_or_path"):
        se.local_config_text("output_dir: x\n", "/m", "/o")
    with pytest.raises(ValueError, match="output_dir"):
        se.local_config_text("model_name_or_path: a\noutput_dir: x\noutput_dir: y\n", "/m", "/o")


def test_real_repo_configs_convert_and_keep_multi_head_settings():
    for variant in se.VARIANTS.values():
        out = se.local_config_text(variant["config"].read_text(), variant["backbone_path"], "/r/out")
        assert f'model_name_or_path: "{variant["backbone_path"]}"' in out
        assert 'rm_head_type: "ranknet_multi_head"' in out and "use_special_tokens: true" in out


def test_variants_use_the_main_readme_inferencer_not_the_unified_one():
    for variant in se.VARIANTS.values():
        assert (variant["module"], variant["inferencer"]) == ("EditReward.inference_edit", "EditRewardInferencer")


def test_variants_use_local_weights():
    for variant in se.VARIANTS.values():
        assert str(variant["backbone_path"]).startswith(str(bi.MODELS_DIR))
        assert str(variant["checkpoint_path"]).startswith(str(bi.MODELS_DIR))
        assert variant["config"].is_file()


def test_mapping_dimensions_of_every_variant_are_editreward_dimensions():
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    for pretrained in se.VARIANTS:
        dimensions = {d for d in bi.dimension_by_bench_metric(mapping, "EditReward", pretrained).values() if d}
        assert dimensions == set(se.HEAD_OF_DIMENSION)


def test_parse_args_rejects_unknown_variant_and_split_options():
    with pytest.raises(SystemExit):
        se.parse_args(["--pretrained", "Qwen3-VL-8B"])
    with pytest.raises(SystemExit):
        se.parse_args(["--pretrained", "MiMo-VL-7B", "--num-shards", "2"])


def test_launcher_uses_edit_reward_env_and_has_no_shard_options():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_editreward.slurm").read_text()
    assert "CONDA_ENV=edit_reward" in launcher and 'ARGS=(--pretrained "$PRETRAINED"' in launcher
    assert "SHARD" not in launcher.upper()
