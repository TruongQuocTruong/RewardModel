"""Tests for scripts/score_lmm4edit.py with a fake scorer (no weights, no GPU, no real images)."""

import json

import pytest

import bench_inputs as bi
import comparison_table as ct
import score_lmm4edit as sl

ROWS = [
    {"sample_id": "B1", "metric": "Background Consistency", "prompt": "p1", "src_image": "s.png",
     "edited_images": ["a.png", "b.png"], "human_label": "B", "votes": {"A": 0, "B": 3, "Tie": 0}},
    {"sample_id": "A1", "metric": "Aesthetic", "prompt": "p2", "src_image": "s.png", "edited_images": ["a.png", "b.png"]},
    {"sample_id": "F1", "metric": "Fidelity", "prompt": "p3", "src_image": "s.png", "edited_images": ["a.png", "b.png"]},
    {"sample_id": "A2", "metric": "Aesthetic", "prompt": "p4", "src_image": "s.png", "edited_images": ["a.png", "b.png"]},
    {"sample_id": "X1", "metric": "Unmapped", "prompt": "p", "src_image": "s.png", "edited_images": ["a.png", "b.png"]},
]
DIMENSIONS = {"Background Consistency": "Attribute Preservation", "Aesthetic": "Perceptual Quality",
              "Fidelity": "Editing Alignment", "Unmapped": None}


class FakeScorer:
    def __init__(self, dimension, log):
        self.dimension, self.log, self.released = dimension, log, False
        log.append(("load", dimension))

    def __call__(self, source, edited, instruction):
        assert not self.released
        self.log.append(("score", self.dimension, source, edited, instruction))
        return {"ED/a.png": 61.5, "ED/b.png": 40.0}[edited]

    def release(self):
        self.released = True
        self.log.append(("release", self.dimension))


def paths(row):
    return f"SRC/{row['src_image']}", [f"ED/{p}" for p in row["edited_images"]]


def run(tmp_path, rows=ROWS, log=None):
    log = [] if log is None else log
    path = tmp_path / "scores.jsonl"
    counts = sl.score_rows(lambda d: FakeScorer(d, log), rows, DIMENSIONS, path, image_paths=paths, log=lambda m: None)
    return counts, (bi.load_jsonl(path) if path.exists() else []), log


def test_modes_follow_the_mapping_and_point_to_the_downloaded_weights():
    assert {d: m for d, (m, _) in sl.MODES.items()} == {
        "Perceptual Quality": "visual", "Editing Alignment": "alignment", "Attribute Preservation": "preservation"}
    for mode, folder in sl.MODES.values():
        assert folder.name == mode
        for name in sl.WEIGHT_FILES:
            assert (folder / name).is_file(), folder / name
        assert json.loads((folder / "regression_config.json").read_text())["dimension"] == mode


def test_mapping_dimensions_are_lmm4edit_modes_and_result_folder_matches_the_table():
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    dimensions = {d for d in bi.dimension_by_bench_metric(mapping, "LMM4Edit", "Qwen3-VL-8B").values() if d}
    assert dimensions == set(sl.MODES)
    assert ct.result_folder_name(sl.MODEL_NAME, sl.PRETRAINED) == "LMM4Edit_Qwen3-VL-8B"


def test_backbone_is_the_local_qwen3_vl_8b():
    assert sl.BACKBONE_PATH == bi.MODELS_DIR / "Qwen3-VL-8B"
    assert sl.MAX_PIXELS_PER_IMAGE == 1048576


def test_score_rows_loads_each_mode_once_in_order_and_releases_before_the_next(tmp_path):
    counts, records, log = run(tmp_path)
    assert [e for e in log if e[0] != "score"] == [
        ("load", "Perceptual Quality"), ("release", "Perceptual Quality"),
        ("load", "Editing Alignment"), ("release", "Editing Alignment"),
        ("load", "Attribute Preservation"), ("release", "Attribute Preservation")]
    assert [r["sample_id"] for r in records] == ["A1", "A2", "F1", "B1"]
    assert counts["scored"] == 4 and counts["not_applicable"] == 1
    assert [m["mode"] for m in counts["modes_loaded"]] == ["visual", "alignment", "preservation"]


def test_score_rows_scores_image_a_then_b_with_source_path_and_row_instruction(tmp_path):
    _, records, log = run(tmp_path, ROWS[:1])
    assert [e[2:] for e in log if e[0] == "score"] == [("SRC/s.png", "ED/a.png", "p1"), ("SRC/s.png", "ED/b.png", "p1")]
    assert (records[0]["score_a"], records[0]["score_b"], records[0]["margin"]) == pytest.approx((61.5, 40.0, 21.5))
    assert records[0]["mode"] == "preservation" and records[0]["dimension"] == "Attribute Preservation"


def test_score_record_is_self_contained(tmp_path):
    _, records, _ = run(tmp_path, ROWS[:1])
    for field in sl.BENCH_FIELDS_TO_KEEP:
        assert records[0][field] == ROWS[0].get(field), field


def test_score_rows_resume_skips_done_rows_and_does_not_load_a_finished_mode(tmp_path):
    done = [{"sample_id": "A1", "dimension": "Perceptual Quality"}, {"sample_id": "A2", "dimension": "Perceptual Quality"}]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done))
    counts, records, log = run(tmp_path)
    assert ("load", "Perceptual Quality") not in log
    assert counts["skipped_done"] == 2 and counts["scored"] == 2 and len(records) == 4


def test_score_rows_releases_the_model_when_scoring_fails(tmp_path):
    log = []

    def failing_paths(row):
        raise OSError("unreadable image")

    with pytest.raises(OSError):
        sl.score_rows(lambda d: FakeScorer(d, log), ROWS[:1], DIMENSIONS, tmp_path / "s.jsonl",
                      image_paths=failing_paths, log=lambda m: None)
    assert log[-1] == ("release", "Attribute Preservation")


def test_score_rows_with_empty_row_list_loads_nothing(tmp_path):
    counts, records, log = run(tmp_path, [])
    assert counts["scored"] == 0 and records == [] and log == []


def test_rows_by_dimension_rejects_dimension_lmm4edit_does_not_have():
    with pytest.raises(ValueError, match="not an LMM4Edit dimension"):
        sl.rows_by_dimension(ROWS[:1], {"Background Consistency": "Semantic Consistency"})


def test_final_score_multiplies_by_the_output_scale_and_rejects_nan():
    assert sl.final_score(0.4567, 100.0) == pytest.approx(45.67)
    assert sl.final_score(-0.1, 100.0) == pytest.approx(-10.0)
    with pytest.raises(ValueError):
        sl.final_score(float("nan"), 100.0)


def test_check_checkpoint_mode_accepts_same_or_missing_dimension_and_rejects_mismatch():
    sl.check_checkpoint_mode("visual", "visual")
    sl.check_checkpoint_mode(None, "alignment")
    with pytest.raises(ValueError, match="mismatch"):
        sl.check_checkpoint_mode("visual", "preservation")


def test_repo_query_and_image_resize_are_the_ones_the_scorer_uses(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers without Qwen3-VL")
    from PIL import Image

    lmm4edit = sl.load_lmm4edit_module()
    query = lmm4edit.build_query("  make it blue ", "alignment")
    assert "Editing instruction: make it blue." in query and "editing instruction alignment" in query
    Image.new("RGB", (2000, 1000)).save(tmp_path / "big.png")
    Image.new("RGB", (64, 48)).save(tmp_path / "small.png")
    big = lmm4edit.resolve_image(tmp_path / "big.png", sl.MAX_PIXELS_PER_IMAGE)
    assert big.size[0] * big.size[1] <= sl.MAX_PIXELS_PER_IMAGE and big.size[0] % 16 == 0 and big.size[1] % 16 == 0
    assert lmm4edit.resolve_image(tmp_path / "small.png", sl.MAX_PIXELS_PER_IMAGE).size == (64, 48)


def test_parse_args_has_no_split_options():
    with pytest.raises(SystemExit):
        sl.parse_args(["--num-shards", "2"])


def test_launcher_uses_omnigen2_rl_env_normal_partition_and_has_no_shard_options():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_lmm4edit.slurm").read_text()
    assert "CONDA_ENV=omnigen2-rl" in launcher and "scripts/score_lmm4edit.py" in launcher
    assert "#SBATCH --partition=normal" in launcher and "#SBATCH --gres=gpu:2g.35gb:1" in launcher
    assert "SHARD" not in launcher.upper()
