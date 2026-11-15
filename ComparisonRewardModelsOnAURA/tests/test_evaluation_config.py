"""Checks for EvaluationConfig/*.json (the agreed model -> metric mapping, metrics and base VLM prompts)."""

import json
import subprocess

import pytest

COLUMNS = ["Aesthetic", "Realism", "Fidelity", "BC", "FC", "SC"]
BENCH_METRICS = {"Aesthetic": "Aesthetic", "Realism": "Realism", "Fidelity": "Fidelity",
                 "BC": "Background Consistency", "FC": "Foreground Consistency", "SC": "Structure Consistency"}


@pytest.fixture
def mapping(comparison_root):
    return json.loads((comparison_root / "EvaluationConfig" / "metric_mapping.json").read_text())


@pytest.fixture
def prompts(comparison_root):
    return json.loads((comparison_root / "EvaluationConfig" / "base_vlm_metric_prompts.json").read_text())


def test_mapping_columns_follow_table_order_and_bench_metric_names(mapping):
    assert [c["column"] for c in mapping["columns"]] == COLUMNS
    assert {c["column"]: c["bench_metric"] for c in mapping["columns"]} == BENCH_METRICS


def test_every_model_row_has_exactly_the_six_columns(mapping):
    for row in mapping["models"]:
        assert list(row["scores"]) == COLUMNS, row["model"]


def test_table_has_fifteen_rows_in_agreed_order(mapping):
    # 2026-11-13 (user): our rows are Student-Stage1 (also stands for Teacher-Stage1, an identical copy),
    # Student-Stage2 and Teacher-Stage2; no "Our Model" row and no Inspector Baseline (overall-only model).
    rows = [(r["model"], r["pretrained"]) for r in mapping["models"]]
    assert len(rows) == 15
    assert rows[0] == ("Qwen2.5-VL-7B", "-")
    assert rows[-3:] == [("Student-Stage1", "Qwen3-VL-Reranker-8B"), ("Student-Stage2", "Qwen3-VL-Reranker-8B"),
                         ("Teacher-Stage2", "Qwen3-VL-Reranker-8B")]
    assert not {"Our Model", "Inspector Baseline", "Teacher-Stage1"} & {r["model"] for r in mapping["models"]}


def test_student_stage2_is_the_only_paired_reference(mapping):
    assert [r["model"] for r in mapping["models"] if r.get("paired_reference")] == ["Student-Stage2"]


def test_tie_head_auc_is_recorded_as_secondary_statistic(mapping):
    assert "Tie Head AUC" in mapping["evaluation_metrics"]["secondary_table"]


def test_every_used_dimension_is_defined_for_its_model_family(mapping):
    dimensions = mapping["reward_model_dimensions"]
    for row in mapping["models"]:
        if row["family"] in ("BaseVLM", "OurModel"):
            continue
        for column, dimension in row["scores"].items():
            if dimension is not None:
                assert dimension in dimensions[row["family"]], (row["model"], column, dimension)


def test_mapping_matches_user_decision_per_model(mapping):
    expected = {
        "EditHF": ["Visual Quality"] * 2 + ["Instruction Alignment"] + ["Attribute Preservation"] * 3,
        "LMM4Edit": ["Perceptual Quality"] * 2 + ["Editing Alignment"] + ["Attribute Preservation"] * 3,
        "EditReward": ["Visual Quality"] * 2 + ["Instruction Following"] + [None] * 3,
        "EditScore": ["Perceptual Quality"] * 2 + ["Semantic Consistency"] * 4,
        "SpatialReward": ["Perceptual Quality"] * 2 + ["Semantic Consistency"] * 4,
    }
    for row in mapping["models"]:
        if row["family"] in expected and "not_run" not in row:
            assert [row["scores"][c] for c in COLUMNS] == expected[row["family"]], (row["model"], row["pretrained"])


def test_editscore_72b_is_marked_not_run(mapping):
    rows = [r for r in mapping["models"] if r["pretrained"] == "Qwen2.5-VL-72B"]
    assert len(rows) == 1 and "not_run" in rows[0]
    assert all(v is None for v in rows[0]["scores"].values())


def test_agreed_evaluation_metrics_are_recorded(mapping):
    metrics = mapping["evaluation_metrics"]
    assert set(metrics["main_table"]) == {"Accuracy", "Margin AUC", "Three-class Accuracy"}
    assert {"PLCC", "SRCC", "Tie AUC", "Accuracy on full-agreement pairs"} <= set(metrics["secondary_table"])
    assert "confidence_interval" in metrics["uncertainty"] and "paired_comparison" in metrics["uncertainty"]


def test_base_vlm_prompts_cover_all_six_metrics(prompts):
    assert sorted(prompts["metrics"]) == sorted(COLUMNS)


def test_every_metric_gives_source_then_edited_image_and_the_instruction(prompts):
    for metric, spec in prompts["metrics"].items():
        assert spec["images"] == ["source", "edited"], metric
        assert spec["uses_instruction"] is True, metric
        assert spec["user_prompt"].count("{instruction}") == 1, metric
        assert "make the sky blue" in spec["user_prompt"].format(instruction="make the sky blue")


def test_human_rater_view_is_recorded_without_changing_model_input(prompts):
    for metric in ("Aesthetic", "Realism"):
        assert prompts["metrics"][metric]["human_raters_saw"] == "edited image only, no instruction"


def test_score_scale_is_single_digit_so_each_score_is_one_token(prompts):
    scale = prompts["score_scale"]
    assert 0 <= scale["min"] < scale["max"] <= 9
    for spec in prompts["metrics"].values():
        assert f"({scale['min']}-{scale['max']})" in spec["user_prompt"]


def test_user_prompt_with_unknown_placeholder_raises_key_error(prompts):
    template = prompts["metrics"]["Fidelity"]["user_prompt"] + " {unexpected}"
    with pytest.raises(KeyError):
        template.format(instruction="x")


def test_slurm_submission_is_blocked_inside_tests():
    with pytest.raises(RuntimeError, match="not allowed"):
        subprocess.run(["sbatch", "job.slurm"])
