"""Tests for scripts/evaluation_metrics.py: every metric against hand-computed values."""

import math

import numpy as np
import pytest

import evaluation_metrics as em


def pair(sample_id, label, score_a, score_b, votes=None, agreement=1.0, metric="Fidelity", probabilities=None):
    votes = votes or {"A": {"A": 3, "B": 0, "Tie": 0}, "B": {"A": 0, "B": 3, "Tie": 0},
                      "Tie": {"A": 0, "B": 0, "Tie": 3}}[label]
    return em.PairScore(sample_id, metric, label, votes, agreement, score_a, score_b, probabilities)


# ---------------------------------------------------------------- PairScore validation

def test_pair_margin_is_score_a_minus_score_b():
    assert pair("s", "A", 7.5, 2.0).margin == pytest.approx(5.5)


def test_vote_margin_is_a_minus_b_over_all_votes_including_tie_votes():
    assert pair("s", "A", 1, 0, votes={"A": 2, "B": 0, "Tie": 1}).vote_margin == pytest.approx(2 / 3)
    assert pair("s", "Tie", 1, 0, votes={"A": 1, "B": 1, "Tie": 1}).vote_margin == 0.0


def test_pair_rejects_unknown_label():
    with pytest.raises(ValueError, match="human_label"):
        pair("s", "C", 1, 0, votes={"A": 1, "B": 0, "Tie": 0})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "3"])
def test_pair_rejects_non_finite_or_non_numeric_scores(bad):
    with pytest.raises(ValueError, match="score_a"):
        pair("s", "A", bad, 0)


def test_pair_rejects_empty_or_negative_votes():
    with pytest.raises(ValueError, match="empty"):
        pair("s", "A", 1, 0, votes={"A": 0, "B": 0, "Tie": 0})
    with pytest.raises(ValueError, match="non-negative"):
        pair("s", "A", 1, 0, votes={"A": -1, "B": 2, "Tie": 0})


@pytest.mark.parametrize("probabilities", [(0.5, 0.5), (0.5, 0.6, 0.1), (0.5, float("nan"), 0.5), (1.2, -0.2, 0.0)])
def test_pair_rejects_invalid_tie_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        pair("s", "A", 1, 0, probabilities=probabilities)


# ---------------------------------------------------------------- Accuracy

def test_accuracy_counts_sign_agreement_on_non_tie_pairs_only():
    labels = ["A", "B", "A", "Tie", "B"]
    margins = [2.0, -1.0, -0.5, 10.0, 3.0]
    assert em.accuracy(labels, margins) == pytest.approx(2 / 4)


def test_accuracy_counts_zero_margin_as_wrong():
    assert em.accuracy(["A", "B"], [0.0, -1.0]) == pytest.approx(0.5)


def test_accuracy_with_a_single_pair():
    assert em.accuracy(["B"], [-0.1]) == 1.0


def test_accuracy_raises_when_only_tie_pairs():
    with pytest.raises(ValueError, match="non-tie"):
        em.accuracy(["Tie", "Tie"], [1.0, 2.0])


def test_accuracy_raises_on_empty_input_and_length_mismatch():
    with pytest.raises(ValueError):
        em.accuracy([], [])
    with pytest.raises(ValueError, match="length"):
        em.accuracy(["A", "B"], [1.0])


def test_accuracy_raises_on_nan_margin():
    with pytest.raises(ValueError, match="finite"):
        em.accuracy(["A"], [float("nan")])


# ---------------------------------------------------------------- Margin AUC

def test_margin_auc_is_one_when_every_winner_has_positive_margin():
    assert em.margin_auc(["A", "B"], [1.0, -2.0]) == 1.0


def test_margin_auc_matches_hand_computation_with_one_wrong_pair():
    # winner-minus-loser values: +2, +1, -1 ; negatives: -2, -1, +1
    # pairs (pos, neg): 9 comparisons; wins = 2>-2,2>-1,2>1, 1>-2,1>-1, (1 vs 1 = 0.5), -1>-2, (-1 vs -1 = 0.5) -> 7
    assert em.margin_auc(["A", "B", "A"], [2.0, -1.0, -1.0]) == pytest.approx(7 / 9)


def test_margin_auc_ignores_tie_pairs():
    with_ties = em.margin_auc(["A", "Tie", "B"], [1.0, -50.0, -3.0])
    assert with_ties == em.margin_auc(["A", "B"], [1.0, -3.0])


def test_margin_auc_is_half_when_all_margins_are_zero():
    assert em.margin_auc(["A", "B", "A"], [0.0, 0.0, 0.0]) == 0.5


def test_margin_auc_raises_when_no_non_tie_pair():
    with pytest.raises(ValueError, match="non-tie"):
        em.margin_auc(["Tie"], [0.3])


# ---------------------------------------------------------------- Tie AUC

def test_tie_auc_from_absolute_margin_ranks_small_margins_as_ties():
    labels = ["Tie", "A", "B", "Tie"]
    margins = np.asarray([0.1, 2.0, -3.0, 5.0])
    # tie confidence -|m|: ties -0.1, -5.0 ; others -2.0, -3.0 -> wins: (-0.1>-2, -0.1>-3) = 2 of 4
    assert em.tie_auc(labels, -np.abs(margins)) == pytest.approx(0.5)


def test_tie_auc_raises_without_tie_pairs_or_without_non_tie_pairs():
    with pytest.raises(ValueError):
        em.tie_auc(["A", "B"], [0.1, 0.2])
    with pytest.raises(ValueError):
        em.tie_auc(["Tie", "Tie"], [0.1, 0.2])


# ---------------------------------------------------------------- correlations

def test_pearson_matches_numpy_corrcoef():
    x, y = [1.0, 2.0, 4.0, 3.0], [0.5, 1.0, 2.5, 1.0]
    assert em.pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_pearson_is_nan_for_constant_model_margin():
    assert math.isnan(em.pearson([1.0, 1.0, 1.0], [1.0, -1.0, 0.0]))


def test_pearson_raises_on_single_pair_and_on_nan():
    with pytest.raises(ValueError, match="two"):
        em.pearson([1.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        em.pearson([1.0, float("nan")], [1.0, 2.0])


def test_spearman_uses_average_ranks_for_tied_values():
    x = [10.0, 20.0, 20.0, 40.0]
    y = [1.0, 2.0, 3.0, 4.0]
    expected = np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1]
    assert em.spearman(x, y) == pytest.approx(expected)


def test_spearman_is_one_for_any_monotonic_transformation():
    x = [0.1, 0.5, 2.0, 7.0]
    assert em.spearman(x, np.exp(x)) == pytest.approx(1.0)


# ---------------------------------------------------------------- three-class accuracy (tie prediction)

def test_predicted_labels_take_argmax_of_three_probabilities():
    probabilities = [(0.7, 0.2, 0.1), (0.2, 0.3, 0.5), (0.1, 0.8, 0.1)]
    assert em.predicted_labels_from_probabilities(probabilities) == ["A", "Tie", "B"]


def test_predicted_labels_reject_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        em.predicted_labels_from_probabilities([(0.5, 0.5)])


def test_predicted_labels_from_margins_give_tie_only_for_exactly_zero_margin():
    assert em.predicted_labels_from_margins([0.5, -1e-9, 0.0, -0.0]) == ["A", "B", "Tie", "Tie"]


def test_predicted_labels_from_margins_reject_nan_and_wrong_shape():
    with pytest.raises(ValueError, match="finite"):
        em.predicted_labels_from_margins([float("nan")])
    with pytest.raises(ValueError, match="one-dimensional"):
        em.predicted_labels_from_margins([[1.0, 2.0]])


def test_three_class_report_gives_accuracy_and_tie_recall_and_precision():
    labels = ["A", "Tie", "B", "Tie"]
    predicted = ["A", "Tie", "Tie", "A"]
    report = em.three_class_report(labels, predicted)
    assert report["accuracy"] == pytest.approx(2 / 4)
    assert report["per_class"]["Tie"] == {"true": 2, "predicted": 2, "recall": 0.5, "precision": 0.5}
    assert report["per_class"]["B"]["precision"] is None


def test_three_class_report_raises_on_empty_mismatched_or_unknown_predictions():
    with pytest.raises(ValueError):
        em.three_class_report([], [])
    with pytest.raises(ValueError, match="length"):
        em.three_class_report(["A"], ["A", "B"])
    with pytest.raises(ValueError, match="predictions"):
        em.three_class_report(["A"], ["X"])


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_interval_is_deterministic_for_a_fixed_seed_and_contains_the_estimate():
    values = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    statistic = lambda idx: float(values[idx].mean())
    first = em.bootstrap_interval(statistic, len(values), resamples=500, seed=3)
    second = em.bootstrap_interval(statistic, len(values), resamples=500, seed=3)
    assert first == second
    assert first["low"] <= values.mean() <= first["high"]


def test_bootstrap_interval_skips_resamples_where_the_statistic_is_undefined():
    labels = np.asarray(["Tie"] + ["A"] * 9)
    margins = np.asarray([0.0] + [1.0] * 9)
    statistic = lambda idx: em.tie_auc(labels[idx], -np.abs(margins[idx]))
    result = em.bootstrap_interval(statistic, len(labels), resamples=200, seed=0)
    assert 0 < result["valid_resamples"] < 200


def test_bootstrap_interval_reports_none_when_never_defined():
    result = em.bootstrap_interval(lambda idx: em.accuracy(["Tie"] * len(idx), [0.0] * len(idx)), 3, resamples=20)
    assert result["low"] is None and result["valid_resamples"] == 0


def test_bootstrap_interval_rejects_no_items():
    with pytest.raises(ValueError):
        em.bootstrap_interval(lambda idx: 0.0, 0)


def test_paired_bootstrap_difference_is_positive_when_ours_is_always_better():
    ours = np.ones(20)
    other = np.zeros(20)
    result = em.paired_bootstrap_difference(lambda i: float(ours[i].mean()), lambda i: float(other[i].mean()), 20,
                                            resamples=100)
    assert result["mean"] == pytest.approx(1.0) and result["p_not_better"] == 0.0


# ---------------------------------------------------------------- one table cell

def bench_pairs(with_probabilities=False):
    rows = [
        ("f1", "A", 0.9, 0.1, {"A": 3, "B": 0, "Tie": 0}, 1.0, (0.8, 0.1, 0.1)),
        ("f2", "B", 0.2, 0.7, {"A": 0, "B": 2, "Tie": 1}, 0.3333, (0.1, 0.7, 0.2)),
        ("f3", "A", 0.4, 0.6, {"A": 2, "B": 1, "Tie": 0}, 0.3333, (0.3, 0.5, 0.2)),
        ("f4", "Tie", 0.5, 0.5, {"A": 0, "B": 0, "Tie": 3}, 1.0, (0.2, 0.2, 0.6)),
        ("f5", "Tie", 0.3, 0.9, {"A": 1, "B": 0, "Tie": 2}, 0.3333, (0.2, 0.3, 0.5)),
    ]
    pairs = [pair(s, l, a, b, votes=v, agreement=g, probabilities=p if with_probabilities else None)
             for s, l, a, b, v, g, p in rows]
    pairs.append(pair("r1", "A", 0.0, 1.0, metric="Realism"))  # other metric, must be ignored
    return pairs


def test_summarize_column_uses_only_pairs_of_the_requested_metric():
    report = em.summarize_column(bench_pairs(), "Fidelity", resamples=50)
    assert (report["pairs"], report["non_tie_pairs"], report["tie_pairs"]) == (5, 3, 2)
    assert report["accuracy"] == pytest.approx(2 / 3)


def test_summarize_column_full_agreement_accuracy_uses_only_unanimous_non_tie_pairs():
    report = em.summarize_column(bench_pairs(), "Fidelity", resamples=50)
    assert report["full_agreement_non_tie_pairs"] == 1
    assert report["full_agreement_accuracy"] == 1.0


def test_summarize_column_correlates_margin_with_vote_margin_over_all_pairs():
    pairs = [p for p in bench_pairs() if p.metric == "Fidelity"]
    report = em.summarize_column(pairs, "Fidelity", resamples=50)
    margins = [p.margin for p in pairs]
    vote_margins = [1.0, -2 / 3, 1 / 3, 0.0, 1 / 3]
    assert report["plcc"] == pytest.approx(np.corrcoef(margins, vote_margins)[0, 1])


def test_summarize_column_three_class_accuracy_of_score_only_model_predicts_tie_only_on_equal_scores():
    # changed 2026-11-11 with the user's decision (option 2): score-only models get their native three-class
    # accuracy instead of "-". Margins 0.8, -0.5, -0.2, 0.0, -0.6 -> A, B, B, Tie, B vs A, B, A, Tie, Tie -> 3 of 5
    report = em.summarize_column(bench_pairs(), "Fidelity", resamples=50)
    assert report["has_tie_prediction"] is False
    assert report["three_class_rule"] == "Tie only if score_A == score_B"
    assert report["three_class_accuracy"] == pytest.approx(3 / 5)
    assert report["three_class"]["per_class"]["Tie"] == {"true": 2, "predicted": 1, "recall": 0.5, "precision": 1.0}
    assert report["tie_auc_from_tie_probability"] is None


def test_summarize_column_reports_three_class_accuracy_when_tie_probabilities_are_given():
    report = em.summarize_column(bench_pairs(with_probabilities=True), "Fidelity", resamples=50)
    # argmax predictions: A, B, B, Tie, Tie vs labels A, B, A, Tie, Tie -> 4 of 5
    assert report["three_class_accuracy"] == pytest.approx(4 / 5)
    assert report["three_class_rule"] == "argmax of P(A), P(B), P(Tie)"
    assert report["three_class"]["per_class"]["Tie"]["recall"] == 1.0
    assert report["tie_auc_from_tie_probability"] == pytest.approx(1.0)
    assert report["three_class_accuracy_interval"]["valid_resamples"] == 50


def test_summarize_column_rejects_mixed_presence_of_tie_probabilities():
    pairs = bench_pairs(with_probabilities=True)
    pairs[0] = pair("f1", "A", 0.9, 0.1)
    with pytest.raises(ValueError, match="some pairs but not all"):
        em.summarize_column(pairs, "Fidelity", resamples=10)


def test_summarize_column_rejects_unknown_metric_and_duplicate_sample_ids():
    with pytest.raises(ValueError, match="no pairs"):
        em.summarize_column(bench_pairs(), "Aesthetic", resamples=10)
    pairs = bench_pairs()
    pairs.append(pair("f1", "A", 1.0, 0.0))
    with pytest.raises(ValueError, match="duplicate"):
        em.summarize_column(pairs, "Fidelity", resamples=10)


def test_summarize_column_with_a_single_non_tie_pair_returns_none_for_undefined_metrics():
    report = em.summarize_column([pair("x", "A", 1.0, 0.0)], "Fidelity", resamples=10)
    assert report["accuracy"] == 1.0
    assert report["plcc"] is None and report["tie_auc_from_margin"] is None
