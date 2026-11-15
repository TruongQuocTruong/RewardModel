"""Tests for scripts/score_editscore.py with a fake EditScore model (no weights, no GPU, no images)."""

import json
import sys
from types import SimpleNamespace

import pytest

import bench_inputs as bi
import score_editscore as se

sys.path.insert(0, str(se.EDITSCORE_REPO))
from editscore.utils import mllm_output_to_dict  # original parser of the cloned repo  # noqa: E402


class FakeModel:
    """Returns queued texts; records what was prepared."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.prepared = []
        self.seeds = []

    def prepare_input(self, images, prompt):
        self.prepared.append((images, prompt))
        return {"images": images, "prompt": prompt}

    def inference(self, inputs, seed=None):
        self.seeds.append(seed)
        return self.texts.pop(0)


def fake_scorer(texts):
    return SimpleNamespace(model=FakeModel(texts), seed=42, score_range=25,
                           SC_prompt="SC rules for <instruction>", PQ_prompt="PQ rules")


def answer(*scores):
    return json.dumps({"reasoning": "fine", "score": list(scores)})


def test_semantic_consistency_uses_both_images_instruction_and_min_of_two_scores_scaled_to_ten():
    scorer = fake_scorer([answer(20, 15)])
    result = se.score_image(scorer, mllm_output_to_dict, "SRC", "EDIT", "make it red", "Semantic Consistency")
    images, prompt = scorer.model.prepared[0]
    assert images == ["SRC", "EDIT"] and prompt == "SC rules for make it red"
    assert result["score"] == pytest.approx(15 / 2.5)
    assert result["sub_scores"] == [20.0, 15.0] and result["status"] == "parsed" and result["tries"] == 1


def test_perceptual_quality_uses_only_the_edited_image_and_no_instruction():
    scorer = fake_scorer([answer(10, 25)])
    result = se.score_image(scorer, mllm_output_to_dict, "SRC", "EDIT", "make it red", "Perceptual Quality")
    assert scorer.model.prepared[0] == ("EDIT", "PQ rules")
    assert result["score"] == pytest.approx(4.0)


def test_every_generation_uses_the_fixed_seed():
    scorer = fake_scorer(["garbage", answer(5, 5)])
    se.score_image(scorer, mllm_output_to_dict, "S", "E", "x", "Perceptual Quality")
    assert scorer.model.seeds == [42, 42]


def test_unparsable_output_is_retried_then_guessed_and_flagged(monkeypatch):
    scorer = fake_scorer(["no json", "still no json", "never json"])
    result = se.score_image(scorer, mllm_output_to_dict, "S", "E", "x", "Semantic Consistency")
    assert result["tries"] == 3 and result["status"] == "guessed"
    assert len(result["texts"]) == 3 and 0 <= result["score"] <= 10


def test_refusal_gives_up_immediately_as_in_editscore():
    scorer = fake_scorer([se.REFUSAL])
    result = se.score_image(scorer, mllm_output_to_dict, "S", "E", "x", "Semantic Consistency")
    assert result["tries"] == 1 and result["status"] == "guessed"


def test_unknown_dimension_raises():
    with pytest.raises(ValueError, match="unknown EditScore dimension"):
        se.score_image(fake_scorer([]), mllm_output_to_dict, "S", "E", "x", "Visual Quality")


def rows():
    return [
        {"sample_id": "F1", "metric": "Fidelity", "prompt": "add a hat", "src_image": "s", "edited_images": ["a", "b"]},
        {"sample_id": "A1", "metric": "Aesthetic", "prompt": "blur", "src_image": "s", "edited_images": ["a", "b"]},
        {"sample_id": "X1", "metric": "Unmapped", "prompt": "p", "src_image": "s", "edited_images": ["a", "b"]},
    ]


DIMENSIONS = {"Fidelity": "Semantic Consistency", "Aesthetic": "Perceptual Quality", "Unmapped": None}


def test_score_rows_writes_one_record_per_row_with_image_a_then_b(tmp_path):
    scorer = fake_scorer([answer(25, 20), answer(10, 5), answer(15, 15), answer(5, 25)])
    counts = se.score_rows(scorer, mllm_output_to_dict, rows(), DIMENSIONS, tmp_path / "s.jsonl", tmp_path / "r.jsonl",
                           load_images=lambda row: ("SRC", ["IMG_A", "IMG_B"]), log=lambda m: None)
    records = bi.load_jsonl(tmp_path / "s.jsonl")
    assert counts == {"scored": 2, "skipped_done": 0, "not_applicable": 1, "guessed_images": 0}
    assert [(r["sample_id"], r["dimension"]) for r in records] == [("F1", "Semantic Consistency"), ("A1", "Perceptual Quality")]
    assert (records[0]["score_a"], records[0]["score_b"]) == (pytest.approx(8.0), pytest.approx(2.0))
    assert scorer.model.prepared[0][0] == ["SRC", "IMG_A"] and scorer.model.prepared[1][0] == ["SRC", "IMG_B"]
    assert bi.load_jsonl(tmp_path / "r.jsonl")[0]["outputs_a"] == [answer(25, 20)]


def test_score_record_is_self_contained_with_bench_fields_sub_score_names_and_reasoning(tmp_path):
    row = dict(rows()[0], image_id="000140", edit_type="add_object", models=["flux", "banana"], num_raters=3,
               votes={"A": 2, "B": 1, "Tie": 0}, observed_agreement=0.3333, human_label="A", preferred_model="flux")
    scorer = fake_scorer([answer(25, 20), answer(10, 5)])
    se.score_rows(scorer, mllm_output_to_dict, [row], DIMENSIONS, tmp_path / "s.jsonl", tmp_path / "r.jsonl",
                  load_images=lambda r: ("SRC", ["A", "B"]), log=lambda m: None)
    record = bi.load_jsonl(tmp_path / "s.jsonl")[0]
    for field in se.BENCH_FIELDS_TO_KEEP:
        assert record[field] == row[field], field
    assert record["margin"] == pytest.approx(record["score_a"] - record["score_b"])
    assert record["sub_score_names"] == ["editing success", "overediting"]
    assert record["reasoning_a"] == "fine" and record["tries_a"] == 1 and record["tries_b"] == 1


def test_score_rows_resumes_without_rescoring_done_rows(tmp_path):
    score_path = tmp_path / "s.jsonl"
    score_path.write_text(json.dumps({"sample_id": "F1", "dimension": "Semantic Consistency"}) + "\n")
    scorer = fake_scorer([answer(15, 15), answer(5, 25)])
    counts = se.score_rows(scorer, mllm_output_to_dict, rows(), DIMENSIONS, score_path, tmp_path / "r.jsonl",
                           load_images=lambda row: ("SRC", ["A", "B"]), log=lambda m: None)
    assert counts["skipped_done"] == 1 and counts["scored"] == 1
    assert len(bi.load_jsonl(score_path)) == 2


def test_score_rows_rejects_mapping_with_a_dimension_editscore_does_not_have(tmp_path):
    with pytest.raises(ValueError, match="not an EditScore dimension"):
        se.score_rows(fake_scorer([]), mllm_output_to_dict, rows()[:1], {"Fidelity": "Visual Quality"},
                      tmp_path / "s.jsonl", tmp_path / "r.jsonl", load_images=lambda row: ("S", ["A", "B"]))


def test_score_rows_with_empty_row_list_writes_nothing(tmp_path):
    counts = se.score_rows(fake_scorer([]), mllm_output_to_dict, [], DIMENSIONS, tmp_path / "s.jsonl", tmp_path / "r.jsonl")
    assert counts["scored"] == 0 and bi.load_jsonl(tmp_path / "s.jsonl") == []


def test_load_pair_images_resizes_edited_images_to_source_size(tmp_path):
    from PIL import Image

    Image.new("RGB", (40, 30)).save(tmp_path / "src.png")
    Image.new("RGB", (80, 80)).save(tmp_path / "a.png")
    Image.new("L", (20, 10)).save(tmp_path / "b.png")
    row = {"src_image": "src.png", "edited_images": ["a.png", "b.png"]}
    source, edited = se.load_pair_images(row, tmp_path, tmp_path)
    assert source.size == (40, 30) and [im.size for im in edited] == [(40, 30), (40, 30)]
    assert all(im.mode == "RGB" for im in edited)


def test_every_variant_points_to_local_weights_with_the_matching_backbone():
    assert se.VARIANTS["Qwen2.5-VL-7B"]["backbone"] == "qwen25vl"
    assert {se.VARIANTS[k]["backbone"] for k in ("Qwen3-VL-4B", "Qwen3-VL-8B")} == {"qwen3vl"}
    for variant in se.VARIANTS.values():
        assert str(variant["model_path"]).startswith(str(bi.MODELS_DIR))
        assert str(variant["lora_path"]).startswith(str(se.EDITSCORE_REPO / "weights"))


def test_mapping_dimensions_of_every_variant_are_editscore_dimensions():
    mapping = json.loads(bi.DEFAULT_MAPPING.read_text())
    for pretrained in se.VARIANTS:
        dimensions = set(bi.dimension_by_bench_metric(mapping, "EditScore", pretrained).values())
        assert dimensions <= set(se.DIMENSIONS)


def test_parse_args_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        se.parse_args(["--pretrained", "Qwen2.5-VL-72B"])


@pytest.mark.parametrize("split", [["--num-shards", "2"], ["--num-shards", "4", "--shard-index", "1"], ["--shard-index", "1"]])
def test_parse_args_rejects_any_split_of_the_bench(split):
    with pytest.raises(SystemExit):
        se.parse_args(["--pretrained", "Qwen3-VL-4B"] + split)


def test_parse_args_accepts_the_whole_bench_flags_passed_by_already_queued_jobs():
    args = se.parse_args(["--pretrained", "Qwen3-VL-4B", "--num-shards", "1", "--shard-index", "0"])
    assert (args.num_shards, args.shard_index) == (1, 0)


def test_launcher_passes_no_shard_options():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_editscore.slurm").read_text()
    assert "SHARD" not in launcher.upper()


def test_launcher_uses_an_env_whose_peft_reads_the_editscore_lora_config():
    launcher = (bi.COMPARISON_ROOT / "script_slurm" / "run_score_editscore.slurm").read_text()
    assert "conda activate omnigen2-rl" in launcher and "conda activate qwen_vl" not in launcher
