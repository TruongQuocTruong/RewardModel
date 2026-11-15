"""Tests for scripts/comparison_table.py with a tiny fake bench, mapping and score files."""

import json

import pytest

import comparison_table as ct

BENCH = [
    {"sample_id": "F1", "metric": "Fidelity", "human_label": "A", "votes": {"A": 3, "B": 0, "Tie": 0}, "observed_agreement": 1.0},
    {"sample_id": "F2", "metric": "Fidelity", "human_label": "B", "votes": {"A": 0, "B": 2, "Tie": 1}, "observed_agreement": 0.3333},
    {"sample_id": "F3", "metric": "Fidelity", "human_label": "Tie", "votes": {"A": 0, "B": 0, "Tie": 3}, "observed_agreement": 1.0},
    {"sample_id": "R1", "metric": "Realism", "human_label": "A", "votes": {"A": 3, "B": 0, "Tie": 0}, "observed_agreement": 1.0},
    {"sample_id": "R2", "metric": "Realism", "human_label": "Tie", "votes": {"A": 1, "B": 0, "Tie": 2}, "observed_agreement": 0.3333},
]
MAPPING = {
    "columns": [{"column": "Realism", "bench_metric": "Realism"}, {"column": "Fidelity", "bench_metric": "Fidelity"}],
    "models": [
        {"model": "Qwen3-VL-8B", "pretrained": "-", "family": "BaseVLM", "scores": {"Realism": "Realism", "Fidelity": "Fidelity"}},
        {"model": "EditReward", "pretrained": "MiMo-VL-7B", "family": "EditReward",
         "scores": {"Realism": "Visual Quality", "Fidelity": None}},
        {"model": "Our Model", "pretrained": "Qwen3-VL-Reranker-8B", "family": "OurModel",
         "scores": {"Realism": "Realism", "Fidelity": "Fidelity"}},
    ],
}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def score(sid, dimension, a, b, probabilities=None):
    record = {"sample_id": sid, "dimension": dimension, "score_a": a, "score_b": b}
    if probabilities is not None:
        record["probabilities"] = probabilities
    return record


@pytest.fixture
def bench(tmp_path):
    return ct.load_bench(write_jsonl(tmp_path / "bench.jsonl", BENCH))


def test_result_folder_name_joins_model_and_pretrained_and_drops_spaces():
    assert ct.result_folder_name("EditScore", "Qwen3-VL-4B") == "EditScore_Qwen3-VL-4B"
    assert ct.result_folder_name("Qwen2.5-VL-7B", "-") == "Qwen2.5-VL-7B"
    assert ct.result_folder_name("Our Model", "Qwen3-VL-Reranker-8B") == "OurModel_Qwen3-VL-Reranker-8B"


def test_load_bench_rejects_duplicate_ids_missing_fields_and_empty_file(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        ct.load_bench(write_jsonl(tmp_path / "a.jsonl", BENCH + BENCH[:1]))
    with pytest.raises(ValueError, match="misses"):
        ct.load_bench(write_jsonl(tmp_path / "b.jsonl", [{"sample_id": "x", "metric": "Fidelity"}]))
    with pytest.raises(ValueError, match="empty"):
        ct.load_bench(write_jsonl(tmp_path / "c.jsonl", []))


def test_load_jsonl_reports_the_line_of_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{broken\n')
    with pytest.raises(ValueError, match=":2:"):
        ct.load_jsonl(path)


def test_load_scores_rejects_duplicate_sample_and_dimension(tmp_path):
    path = write_jsonl(tmp_path / "s.jsonl", [score("F1", "Fidelity", 1, 0), score("F1", "Fidelity", 2, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        ct.load_scores(path)


def test_join_pairs_keeps_only_rows_of_the_bench_metric_and_orients_scores_as_a_then_b(bench):
    scores = {"F1": score("F1", "Fidelity", 9, 1), "F2": score("F2", "Fidelity", 2, 8), "F3": score("F3", "Fidelity", 5, 5)}
    pairs, coverage = ct.join_pairs(bench, scores, "Fidelity")
    assert [p.sample_id for p in pairs] == ["F1", "F2", "F3"]
    assert [p.margin for p in pairs] == [8, -6, 0]
    assert coverage == {"expected": 3, "scored": 3, "missing": 0}


def test_join_pairs_raises_on_missing_pairs_unless_partial_is_allowed(bench):
    scores = {"F1": score("F1", "Fidelity", 9, 1)}
    with pytest.raises(ValueError, match="2 of 3 pairs have no score"):
        ct.join_pairs(bench, scores, "Fidelity")
    pairs, coverage = ct.join_pairs(bench, scores, "Fidelity", allow_partial=True)
    assert len(pairs) == 1 and coverage["missing"] == 2


def test_join_pairs_raises_on_scores_for_unknown_sample_ids(bench):
    with pytest.raises(ValueError, match="unknown sample_id"):
        ct.join_pairs(bench, {"X9": score("X9", "Fidelity", 1, 0)}, "Fidelity")


def test_build_table_marks_not_applicable_pending_and_done_cells(tmp_path, bench):
    write_jsonl(tmp_path / "results" / "Qwen3-VL-8B" / "scores.jsonl", [
        score("R1", "Realism", 6, 2), score("R2", "Realism", 4, 4),
        score("F1", "Fidelity", 7, 3), score("F2", "Fidelity", 1, 5), score("F3", "Fidelity", 3, 3),
    ])
    table = ct.build_table(MAPPING, bench, tmp_path / "results", resamples=20)
    base, edit_reward, ours = table["rows"]
    assert base["cells"]["Fidelity"]["status"] == "done"
    assert base["cells"]["Fidelity"]["accuracy"] == 1.0
    assert base["cells"]["Fidelity"]["three_class_accuracy"] == 1.0  # equal scores on the Tie pair
    assert edit_reward["cells"]["Fidelity"]["status"] == "not_applicable"
    assert edit_reward["cells"]["Realism"]["status"] == "pending"
    assert ours["cells"]["Realism"]["status"] == "pending"


def test_build_table_reports_dimension_missing_from_score_file_as_pending(tmp_path, bench):
    write_jsonl(tmp_path / "results" / "EditReward_MiMo-VL-7B" / "scores.jsonl", [score("R1", "Instruction Following", 1, 0)])
    table = ct.build_table(MAPPING, bench, tmp_path / "results", resamples=10)
    assert "no scores for dimension" in table["rows"][1]["cells"]["Realism"]["reason"]


def test_build_table_adds_paired_comparison_against_our_model(tmp_path, bench):
    write_jsonl(tmp_path / "results" / "Qwen3-VL-8B" / "scores.jsonl", [
        score("R1", "Realism", 1, 2), score("R2", "Realism", 4, 3),
        score("F1", "Fidelity", 1, 3), score("F2", "Fidelity", 5, 1), score("F3", "Fidelity", 2, 3),
    ])
    write_jsonl(tmp_path / "results" / "OurModel_Qwen3-VL-Reranker-8B" / "scores.jsonl", [
        score("R1", "Realism", 3, 1, [0.8, 0.1, 0.1]), score("R2", "Realism", 2, 2, [0.2, 0.2, 0.6]),
        score("F1", "Fidelity", 3, 1, [0.7, 0.2, 0.1]), score("F2", "Fidelity", 1, 4, [0.1, 0.8, 0.1]),
        score("F3", "Fidelity", 2, 2, [0.25, 0.25, 0.5]),
    ])
    table = ct.build_table(MAPPING, bench, tmp_path / "results", resamples=50)
    ours, base = table["rows"][2], table["rows"][0]
    assert ours["cells"]["Fidelity"]["three_class_rule"] == "argmax of P(A), P(B), P(Tie)"
    assert ours["cells"]["Fidelity"]["three_class_accuracy"] == 1.0
    comparison = base["cells"]["Fidelity"]["paired_with_our_model"]
    assert set(comparison) == {"accuracy", "margin_auc", "three_class_accuracy"}
    assert comparison["three_class_accuracy"]["mean"] > 0
    assert "paired_with_our_model" not in ours["cells"]["Fidelity"]


def test_render_markdown_shows_dash_pending_and_acc_margin_auc_cells(tmp_path, bench):
    write_jsonl(tmp_path / "results" / "Qwen3-VL-8B" / "scores.jsonl", [
        score("R1", "Realism", 6, 2), score("R2", "Realism", 4, 4),
        score("F1", "Fidelity", 7, 3), score("F2", "Fidelity", 1, 5), score("F3", "Fidelity", 3, 3),
    ])
    text = ct.render_markdown(ct.build_table(MAPPING, bench, tmp_path / "results", resamples=10))
    assert "| Qwen3-VL-8B | - | 100.00 \\| 100.00 | 100.00 \\| 100.00 |" in text
    assert "| EditReward | MiMo-VL-7B | pending | - |" in text
    assert "Three-class Accuracy" in text


def test_main_writes_json_and_markdown(tmp_path):
    bench_path = write_jsonl(tmp_path / "bench.jsonl", BENCH)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(MAPPING))
    output = tmp_path / "out" / "table"
    assert ct.main(["--mapping", str(mapping_path), "--bench", str(bench_path), "--results-dir",
                    str(tmp_path / "results"), "--output", str(output), "--resamples", "5"]) == 0
    assert json.loads(output.with_suffix(".json").read_text())["columns"] == ["Realism", "Fidelity"]
    assert output.with_suffix(".md").read_text().startswith("### Accuracy")


def test_parse_args_rejects_zero_resamples():
    with pytest.raises(SystemExit):
        ct.parse_args(["--output", "x", "--resamples", "0"])


def test_combined_cell_text_lists_every_statistic_in_order_with_percent_and_correlation_formats():
    cell = {"status": "done", "accuracy": 0.5, "margin_auc": 0.625, "three_class_accuracy": 0.4,
            "full_agreement_accuracy": 1.0, "tie_auc_from_margin": 0.75, "plcc": 0.1234, "srcc": -0.5}
    assert ct.combined_cell_text(cell) == "50.00 \\| 62.50 \\| 40.00 \\| 100.00 \\| 75.00 \\| 0.123 \\| -0.500"
    assert [f for f, _, _ in ct.COMBINED_FIELDS] == ["accuracy", "margin_auc", "three_class_accuracy",
                                                     "full_agreement_accuracy", "tie_auc_from_margin", "plcc", "srcc"]


def test_combined_cell_text_dash_pending_and_missing_statistic():
    assert ct.combined_cell_text({"status": "not_applicable"}) == "-"
    assert ct.combined_cell_text({"status": "pending"}) == "pending"
    cell = {"status": "done", "accuracy": 0.5, "margin_auc": float("nan"), "three_class_accuracy": 0.4,
            "full_agreement_accuracy": None, "tie_auc_from_margin": 0.5, "plcc": float("nan"), "srcc": 0.0}
    assert ct.combined_cell_text(cell) == "50.00 \\| - \\| 40.00 \\| - \\| 50.00 \\| - \\| 0.000"


def test_render_markdown_starts_with_one_combined_table_holding_all_statistics(tmp_path, bench):
    write_jsonl(tmp_path / "results" / "Qwen3-VL-8B" / "scores.jsonl", [
        score("R1", "Realism", 6, 2), score("R2", "Realism", 4, 4),
        score("F1", "Fidelity", 7, 3), score("F2", "Fidelity", 1, 5), score("F3", "Fidelity", 3, 3),
    ])
    table = ct.build_table(MAPPING, bench, tmp_path / "results", resamples=10)
    text = ct.render_markdown(table)
    first_section = text.split("\n### ")[0]
    fidelity = table["rows"][0]["cells"]["Fidelity"]
    assert first_section.startswith("### Accuracy and all secondary statistics per cell: Acc \\| Margin AUC")
    assert "| EditReward | MiMo-VL-7B | pending | - |" in first_section
    assert ct.combined_cell_text(fidelity) in first_section
    assert ct.combined_cell_text(fidelity).startswith("100.00 \\| 100.00 \\| 100.00 \\| 100.00 \\| ")


def _mapping_with_three_our_rows(reference_index=None):
    rows = [MAPPING["models"][0]] + [
        {"model": name, "pretrained": "Qwen3-VL-Reranker-8B", "family": "OurModel",
         "scores": {"Realism": "realism", "Fidelity": "fidelity"}}
        for name in ("Inspector Baseline", "Student-Stage1", "Student-Stage2")]
    if reference_index is not None:
        rows[reference_index] = {**rows[reference_index], "paired_reference": True}
    return {**MAPPING, "models": rows}


def test_reference_row_is_the_marked_row_not_the_first_our_model_row():
    mapping = _mapping_with_three_our_rows(reference_index=3)
    rows_out = [{"model": r["model"], "family": r["family"]} for r in mapping["models"]]
    assert ct.reference_row(mapping, rows_out)["model"] == "Student-Stage2"


def test_reference_row_falls_back_to_first_our_model_row_without_mark():
    mapping = _mapping_with_three_our_rows()
    rows_out = [{"model": r["model"], "family": r["family"]} for r in mapping["models"]]
    assert ct.reference_row(mapping, rows_out)["model"] == "Inspector Baseline"


def test_reference_row_rejects_two_marked_rows():
    mapping = _mapping_with_three_our_rows(reference_index=3)
    mapping["models"][1] = {**mapping["models"][1], "paired_reference": True}
    with pytest.raises(ValueError, match="more than one"):
        ct.reference_row(mapping, [{"model": r["model"], "family": r["family"]} for r in mapping["models"]])


def test_pending_reference_gives_no_paired_comparison_and_other_our_rows_are_compared_once_it_is_done(tmp_path, bench):
    mapping = _mapping_with_three_our_rows(reference_index=3)
    stage1 = [score("R1", "realism", 3, 1, [0.8, 0.1, 0.1]), score("R2", "realism", 2, 2, [0.2, 0.2, 0.6]),
              score("F1", "fidelity", 3, 1, [0.7, 0.2, 0.1]), score("F2", "fidelity", 1, 4, [0.1, 0.8, 0.1]),
              score("F3", "fidelity", 2, 2, [0.25, 0.25, 0.5])]
    write_jsonl(tmp_path / "results" / "Student-Stage1_Qwen3-VL-Reranker-8B" / "scores.jsonl", stage1)
    table = ct.build_table(mapping, bench, tmp_path / "results", resamples=10)
    assert "paired_with_our_model" not in table["rows"][2]["cells"]["Fidelity"]
    write_jsonl(tmp_path / "results" / "Student-Stage2_Qwen3-VL-Reranker-8B" / "scores.jsonl", stage1)
    table = ct.build_table(mapping, bench, tmp_path / "results", resamples=10)
    assert "paired_with_our_model" in table["rows"][2]["cells"]["Fidelity"]
    assert "paired_with_our_model" not in table["rows"][3]["cells"]["Fidelity"]


def test_render_markdown_has_tie_head_auc_section_with_dash_for_score_only_models(tmp_path, bench):
    mapping = _mapping_with_three_our_rows(reference_index=3)
    write_jsonl(tmp_path / "results" / "Qwen3-VL-8B" / "scores.jsonl", [
        score("R1", "Realism", 6, 2), score("R2", "Realism", 4, 4),
        score("F1", "Fidelity", 7, 3), score("F2", "Fidelity", 1, 5), score("F3", "Fidelity", 3, 3)])
    write_jsonl(tmp_path / "results" / "InspectorBaseline_Qwen3-VL-Reranker-8B" / "scores.jsonl", [
        score("R1", "realism", 3, 1, [0.8, 0.1, 0.1]), score("R2", "realism", 2, 2, [0.2, 0.2, 0.6]),
        score("F1", "fidelity", 3, 1, [0.7, 0.2, 0.1]), score("F2", "fidelity", 1, 4, [0.1, 0.8, 0.1]),
        score("F3", "fidelity", 2, 2, [0.25, 0.25, 0.5])])
    text = ct.render_markdown(ct.build_table(mapping, bench, tmp_path / "results", resamples=10))
    section = text.split("### Tie Head AUC from P(Tie)")[1].split("\n### ")[0]
    assert "| Qwen3-VL-8B | - | - | - |" in section
    assert "| Inspector Baseline | Qwen3-VL-Reranker-8B | 100.00 | 100.00 |" in section
    assert "| Student-Stage2 | Qwen3-VL-Reranker-8B | pending | pending |" in section
