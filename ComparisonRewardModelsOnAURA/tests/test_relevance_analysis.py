"""Tests for scripts/relevance_analysis.py on hand-made Student score records."""

import json

import pytest

import relevance_analysis as ra
from score_our_model import STUDENT_HEADS


def record(sid, metric, head, label, a, b, p_tie, relevance_own, relevance_other=0.1):
    relevance = {h: relevance_other for h in STUDENT_HEADS}
    relevance[head] = relevance_own
    return {"sample_id": sid, "metric": metric, "dimension": head, "human_label": label,
            "score_a": a, "score_b": b, "probabilities": [(1 - p_tie) / 2, (1 - p_tie) / 2, p_tie],
            "relevance_probs": relevance}


BC = "Background Consistency"
RECORDS = [
    record("b1", BC, "background_consistency", "A", 2.0, 1.0, 0.1, 0.9),   # high relevance, correct
    record("b2", BC, "background_consistency", "B", 2.0, 1.0, 0.2, 0.8),   # high relevance, wrong
    record("b3", BC, "background_consistency", "Tie", 1.0, 1.0, 0.7, 0.6),  # high relevance, tie
    record("b4", BC, "background_consistency", "A", 3.0, 1.0, 0.3, 0.2, relevance_other=0.4),  # low, correct
    record("a1", "Aesthetic", "aesthetic", "B", 0.0, 1.0, 0.1, 0.7),
]


def test_own_relevance_statistics_per_column_are_computed_by_hand():
    report = ra.analyse(RECORDS)
    bc = report["columns"]["BC"]
    assert bc["pairs"] == 4 and bc["head"] == "background_consistency"
    assert bc["mean_own_relevance"] == pytest.approx((0.9 + 0.8 + 0.6 + 0.2) / 4)
    assert bc["share_own_relevance_at_least_threshold"] == 0.75
    assert bc["share_own_metric_most_relevant"] == 0.75  # b4: own 0.2 < others 0.4
    assert set(report["columns"]) == {"BC", "Aesthetic"}


def test_own_metric_tied_with_an_earlier_head_counts_as_most_relevant():
    # Realism comes after aesthetic in the head order; a tie must not be broken by that order.
    tied = record("r1", "Realism", "realism", "A", 1.0, 0.0, 0.1, 1.0)
    tied["relevance_probs"]["aesthetic"] = 1.0
    below = record("r2", "Realism", "realism", "A", 1.0, 0.0, 0.1, 0.99)
    below["relevance_probs"]["aesthetic"] = 1.0
    assert ra.analyse([tied, below])["columns"]["Realism"]["share_own_metric_most_relevant"] == 0.5


def test_groups_split_by_own_relevance_with_accuracy_margin_auc_and_tie_head_auc():
    bc = ra.analyse(RECORDS)["columns"]["BC"]
    high, low = bc["own_relevance_at_least_threshold"], bc["own_relevance_below_threshold"]
    assert (high["pairs"], high["non_tie_pairs"], high["tie_pairs"]) == (3, 2, 1)
    assert high["accuracy"] == 0.5 and high["margin_auc"] == 0.5
    assert high["tie_head_auc"] == 1.0  # the tie pair has the highest P(Tie)
    assert (low["pairs"], low["accuracy"], low["margin_auc"]) == (1, 1.0, 1.0)
    assert low["tie_head_auc"] is None  # no tie pair in the group
    assert bc["all_pairs"]["accuracy"] == pytest.approx(2 / 3)


def test_threshold_boundary_counts_as_relevant():
    rows = [record("x", BC, "background_consistency", "A", 1.0, 0.0, 0.1, 0.5)]
    bc = ra.analyse(rows)["columns"]["BC"]
    assert bc["own_relevance_at_least_threshold"]["pairs"] == 1
    assert bc["own_relevance_below_threshold"]["pairs"] == 0
    assert bc["own_relevance_below_threshold"]["accuracy"] is None


def test_mean_relevance_per_head_covers_all_six_heads():
    per_head = ra.analyse(RECORDS)["columns"]["Aesthetic"]["mean_relevance_per_head"]
    assert list(per_head) == list(STUDENT_HEADS)
    assert per_head["aesthetic"] == 0.7 and per_head["realism"] == 0.1


def test_group_with_only_tie_pairs_has_no_accuracy_but_no_error():
    rows = [record("t", BC, "background_consistency", "Tie", 1.0, 1.0, 0.6, 0.9)]
    group = ra.analyse(rows)["columns"]["BC"]["all_pairs"]
    assert group["accuracy"] is None and group["margin_auc"] is None and group["tie_head_auc"] is None


@pytest.mark.parametrize("change", [
    {"metric": "Overall"},
    {"dimension": "aesthetic"},
    {"relevance_probs": {"background_consistency": 0.5}},
    {"relevance_probs": {h: float("nan") for h in STUDENT_HEADS}},
    {"relevance_probs": {h: 1.5 for h in STUDENT_HEADS}},
])
def test_invalid_records_raise(change):
    with pytest.raises(ValueError):
        ra.analyse([{**RECORDS[0], **change}])


def test_empty_file_and_duplicate_sample_id_raise():
    with pytest.raises(ValueError, match="empty"):
        ra.analyse([])
    with pytest.raises(ValueError, match="duplicate"):
        ra.analyse([RECORDS[0], RECORDS[0]])


def test_markdown_has_three_tables_and_dash_for_missing_values():
    text = ra.render_markdown(ra.analyse(RECORDS))
    assert text.count("### ") == 3
    assert "| BC | 4 | 0.625 | 75.00 | 75.00 |" in text
    assert "| BC | < 0.5 | 1 (1 / 0) | 100.00 | 100.00 | - |" in text


def test_main_writes_json_and_markdown(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    output = tmp_path / "out" / "relevance_analysis"
    assert ra.main(["--scores", str(scores), "--output", str(output)]) == 0
    assert json.loads(output.with_suffix(".json").read_text())["pairs"] == 5
    assert output.with_suffix(".md").read_text().startswith("### Relevance")


def test_parse_args_rejects_threshold_outside_zero_one():
    with pytest.raises(SystemExit):
        ra.parse_args(["--scores", "s", "--output", "o", "--threshold", "1.0"])
