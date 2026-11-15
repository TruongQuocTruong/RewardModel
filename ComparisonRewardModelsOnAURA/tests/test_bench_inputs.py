"""Tests for scripts/bench_inputs.py (fake bench, temporary files)."""

import json

import pytest

import bench_inputs as bi


def bench_row(sid, metric):
    return {"sample_id": sid, "metric": metric, "prompt": "p", "src_image": f"src/{sid}.jpg",
            "edited_images": [f"a/{sid}.png", f"b/{sid}.png"]}


ROWS = [bench_row("F1", "Fidelity"), bench_row("R1", "Realism"), bench_row("F2", "Fidelity"),
        bench_row("R2", "Realism"), bench_row("F3", "Fidelity")]
MAPPING = {"columns": [{"column": "Realism", "bench_metric": "Realism"}, {"column": "BC", "bench_metric": "Background Consistency"}],
           "models": [{"model": "EditScore", "pretrained": "Qwen3-VL-4B",
                       "scores": {"Realism": "Perceptual Quality", "BC": "Semantic Consistency"}},
                      {"model": "EditReward", "pretrained": "MiMo-VL-7B", "scores": {"Realism": "Visual Quality", "BC": None}}]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_bench_rows_keeps_file_order(tmp_path):
    assert [r["sample_id"] for r in bi.load_bench_rows(write_jsonl(tmp_path / "b.jsonl", ROWS))] == ["F1", "R1", "F2", "R2", "F3"]


def test_load_bench_rows_rejects_duplicates_missing_fields_and_wrong_image_count(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        bi.load_bench_rows(write_jsonl(tmp_path / "a.jsonl", ROWS + ROWS[:1]))
    with pytest.raises(ValueError, match="misses"):
        bi.load_bench_rows(write_jsonl(tmp_path / "b.jsonl", [{"sample_id": "x"}]))
    bad = dict(bench_row("x", "Fidelity"), edited_images=["only_one.png"])
    with pytest.raises(ValueError, match="exactly two"):
        bi.load_bench_rows(write_jsonl(tmp_path / "c.jsonl", [bad]))
    with pytest.raises(ValueError, match="empty"):
        bi.load_bench_rows(write_jsonl(tmp_path / "d.jsonl", []))


def test_image_paths_join_source_and_edited_roots(tmp_path):
    source, edited = bi.image_paths(bench_row("F1", "Fidelity"), tmp_path / "S", tmp_path / "E")
    assert source == tmp_path / "S" / "src/F1.jpg"
    assert edited == [tmp_path / "E" / "a/F1.png", tmp_path / "E" / "b/F1.png"]


def test_dimension_by_bench_metric_uses_bench_metric_names_and_keeps_none():
    assert bi.dimension_by_bench_metric(MAPPING, "EditScore", "Qwen3-VL-4B") == {
        "Realism": "Perceptual Quality", "Background Consistency": "Semantic Consistency"}
    assert bi.dimension_by_bench_metric(MAPPING, "EditReward", "MiMo-VL-7B")["Background Consistency"] is None


def test_dimension_by_bench_metric_raises_for_unknown_model():
    with pytest.raises(ValueError, match="expected one mapping row"):
        bi.dimension_by_bench_metric(MAPPING, "EditScore", "Qwen2.5-VL-72B-missing")


def test_select_rows_default_returns_every_row_in_file_order():
    assert [r["sample_id"] for r in bi.select_rows(ROWS)] == ["F1", "R1", "F2", "R2", "F3"]


def test_select_rows_has_no_shard_parameters():
    with pytest.raises(TypeError):
        bi.select_rows(ROWS, num_shards=2, shard_index=0)
    assert not hasattr(bi, "output_name")


def test_select_rows_per_metric_limit_takes_first_rows_of_each_metric():
    assert [r["sample_id"] for r in bi.select_rows(ROWS, per_metric_limit=1)] == ["F1", "R1"]


def test_select_rows_metric_filter():
    assert [r["sample_id"] for r in bi.select_rows(ROWS, metrics=["Realism"])] == ["R1", "R2"]


@pytest.mark.parametrize("limit", [0, -1])
def test_select_rows_rejects_non_positive_per_metric_limit(limit):
    with pytest.raises(ValueError):
        bi.select_rows(ROWS, per_metric_limit=limit)


def test_already_scored_reads_sample_and_dimension_pairs(tmp_path):
    assert bi.already_scored(tmp_path / "missing.jsonl") == set()
    path = write_jsonl(tmp_path / "s.jsonl", [{"sample_id": "F1", "dimension": "Semantic Consistency"}])
    assert bi.already_scored(path) == {("F1", "Semantic Consistency")}
