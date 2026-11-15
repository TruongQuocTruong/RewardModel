"""Tests for scripts/score_edithf.py with a fake scorer (no weights, no GPU, no real images)."""

import json

import pytest

import bench_inputs as bi
import score_edithf as sh

ROWS = [
    {"sample_id": "B1", "metric": "Background Consistency", "prompt": "p1", "src_image": "s", "edited_images": ["a", "b"],
     "human_label": "B"},
    {"sample_id": "A1", "metric": "Aesthetic", "prompt": "p2", "src_image": "s", "edited_images": ["a", "b"]},
    {"sample_id": "F1", "metric": "Fidelity", "prompt": "p3", "src_image": "s", "edited_images": ["a", "b"]},
    {"sample_id": "A2", "metric": "Aesthetic", "prompt": "p4", "src_image": "s", "edited_images": ["a", "b"]},
    {"sample_id": "X1", "metric": "Unmapped", "prompt": "p", "src_image": "s", "edited_images": ["a", "b"]},
]
DIMENSIONS = {"Background Consistency": "Attribute Preservation", "Aesthetic": "Visual Quality",
              "Fidelity": "Instruction Alignment", "Unmapped": None}


class FakeScorer:
    def __init__(self, dimension, log):
        self.dimension, self.log, self.released = dimension, log, False
        log.append(("load", dimension))

    def __call__(self, source, edited, instruction):
        assert not self.released
        self.log.append(("score", self.dimension, edited, instruction))
        return {"IMG_A": 70.0, "IMG_B": 40.0}[edited]

    def release(self):
        self.released = True
        self.log.append(("release", self.dimension))


def run(tmp_path, rows=ROWS, log=None):
    log = [] if log is None else log
    path = tmp_path / "scores.jsonl"
    counts = sh.score_rows(lambda d: FakeScorer(d, log), rows, DIMENSIONS, path,
                           load_images=lambda r: ("SRC", ["IMG_A", "IMG_B"]), log=lambda m: None)
    return counts, (bi.load_jsonl(path) if path.exists() else []), log


def test_modes_follow_the_mapping_and_editing_folder_is_alignment():
    assert {d: m for d, (m, _) in sh.MODES.items()} == {
        "Visual Quality": "visual", "Instruction Alignment": "alignment", "Attribute Preservation": "preservation"}
    assert sh.MODES["Instruction Alignment"][1].name == "lora_checkpoints_editing"
    for _, lora in sh.MODES.values():
        assert (lora / "mlp_head.pth").is_file() and (lora / "vision_lora").is_dir() and (lora / "llm_lora").is_dir()


def test_score_rows_loads_each_mode_once_in_order_and_releases_before_the_next(tmp_path):
    counts, records, log = run(tmp_path)
    assert [e for e in log if e[0] != "score"] == [
        ("load", "Visual Quality"), ("release", "Visual Quality"),
        ("load", "Instruction Alignment"), ("release", "Instruction Alignment"),
        ("load", "Attribute Preservation"), ("release", "Attribute Preservation")]
    assert [r["sample_id"] for r in records] == ["A1", "A2", "F1", "B1"]
    assert counts["scored"] == 4 and counts["not_applicable"] == 1
    assert [m["mode"] for m in counts["modes_loaded"]] == ["visual", "alignment", "preservation"]


def test_score_rows_scores_image_a_then_b_with_the_row_instruction(tmp_path):
    _, records, log = run(tmp_path, ROWS[:1])
    assert [e[2:] for e in log if e[0] == "score"] == [("IMG_A", "p1"), ("IMG_B", "p1")]
    assert (records[0]["score_a"], records[0]["score_b"], records[0]["margin"]) == pytest.approx((70.0, 40.0, 30.0))
    assert records[0]["mode"] == "preservation" and records[0]["human_label"] == "B"
    for field in sh.BENCH_FIELDS_TO_KEEP:
        assert records[0][field] == ROWS[0].get(field), field


def test_score_rows_resume_skips_done_rows_and_does_not_load_a_finished_mode(tmp_path):
    done = [{"sample_id": "A1", "dimension": "Visual Quality"}, {"sample_id": "A2", "dimension": "Visual Quality"}]
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done))
    counts, records, log = run(tmp_path)
    assert ("load", "Visual Quality") not in log
    assert counts["skipped_done"] == 2 and counts["scored"] == 2 and len(records) == 4


def test_score_rows_releases_the_model_when_scoring_fails(tmp_path):
    log = []

    def failing_images(row):
        raise OSError("unreadable image")

    with pytest.raises(OSError):
        sh.score_rows(lambda d: FakeScorer(d, log), ROWS[:1], DIMENSIONS, tmp_path / "s.jsonl",
                      load_images=failing_images, log=lambda m: None)
    assert log[-1] == ("release", "Attribute Preservation")


def test_score_rows_with_empty_row_list_loads_nothing(tmp_path):
    counts, records, log = run(tmp_path, [])
    assert counts["scored"] == 0 and records == [] and log == []


def test_rows_by_dimension_rejects_dimension_edithf_does_not_have():
    with pytest.raises(ValueError, match="not an EditHF dimension"):
        sh.rows_by_dimension(ROWS[:1], {"Background Consistency": "Semantic Consistency"})


def test_regression_score_is_absolute_value_times_100_and_rejects_nan():
    assert sh.regression_score(-0.4567) == pytest.approx(45.67)
    assert sh.regression_score(0.0) == 0.0
    with pytest.raises(ValueError):
        sh.regression_score(float("nan"))


def test_expand_query_matches_infer_single_text_with_the_repo_build_query():
    pytest.importorskip("torchvision")
    edithf = sh.load_edithf_module()
    query = sh.expand_query(edithf, "make it blue", "alignment", 4, [1, 1])
    image = edithf.IMG_START_TOKEN + edithf.IMG_CONTEXT_TOKEN * 4 + edithf.IMG_END_TOKEN
    expected = edithf.build_query("make it blue", "alignment").replace("<image>", image) + sh.QUERY_SUFFIX
    assert query == expected and query.endswith("The quality score is 00.00")
    assert query.count(edithf.IMG_CONTEXT_TOKEN) == 8


def test_expand_query_rejects_unknown_mode():
    pytest.importorskip("torchvision")
    with pytest.raises(ValueError):
        sh.expand_query(sh.load_edithf_module(), "x", "overall", 4, [1, 1])


def test_mapping_dimensions_are_edithf_modes():
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    dimensions = {d for d in bi.dimension_by_bench_metric(mapping, "EditHF", "InternVL3.5-8B").values() if d}
    assert dimensions == set(sh.MODES)


def test_parse_args_has_no_split_options():
    with pytest.raises(SystemExit):
        sh.parse_args(["--num-shards", "2"])


def test_launcher_uses_edit_reward_env_and_has_no_shard_options():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_edithf.slurm").read_text()
    assert "CONDA_ENV=edit_reward" in launcher and "SHARD" not in launcher.upper()
