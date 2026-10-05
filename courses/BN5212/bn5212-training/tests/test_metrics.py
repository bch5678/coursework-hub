"""Selection metrics.

These only pick checkpoints, but a wrong AUROC would pick the wrong checkpoint,
so they are pinned against hand-computable cases. Values are cross-checked
against the benchmark project's implementation on shared runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from bn5212_training.metrics import (
    auprc,
    auroc,
    brier,
    confusion_counts,
    cross_fitted_decisions,
    decision_summary,
    is_better,
    selection_metrics,
    youden_threshold,
)


def test_perfect_ranking_scores_one():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_inverted_ranking_scores_zero():
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_all_equal_scores_give_one_half():
    """Every pair is a tie, so the model has no ranking information."""
    assert auroc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auroc_counts_ties_as_half_a_pair():
    # One positive above one negative, tied with the other negative.
    value = auroc(np.array([0, 0, 1]), np.array([0.1, 0.5, 0.5]))
    assert value == pytest.approx(0.75)


def test_single_class_split_returns_none_rather_than_a_number():
    """A faked AUROC on a single-label split would be a silent reporting error."""
    assert auroc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])) is None
    assert auroc(np.array([0, 0]), np.array([0.2, 0.5])) is None
    assert auprc(np.array([0, 0]), np.array([0.2, 0.5])) is None


def test_auprc_of_a_perfect_ranking_is_one():
    assert auprc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auprc_matches_a_hand_computed_case():
    # Ranked: pos, neg, pos. Precision at the two positives is 1.0 and 2/3;
    # each contributes half the recall.
    value = auprc(np.array([1, 0, 1]), np.array([0.9, 0.6, 0.3]))
    assert value == pytest.approx((1.0 * 0.5) + ((2 / 3) * 0.5))


def test_brier_is_the_mean_squared_error():
    assert brier(np.array([1, 0]), np.array([0.75, 0.25])) == pytest.approx(0.0625)


def test_selection_metrics_reports_counts():
    result = selection_metrics(np.array([0, 1, 1]), np.array([0.2, 0.7, 0.8]))
    assert result["n"] == 3
    assert result["n_positive"] == 2
    assert result["prevalence"] == pytest.approx(2 / 3)


def test_is_better_respects_metric_direction():
    assert is_better("auroc", 0.8, 0.7)
    assert not is_better("auroc", 0.6, 0.7)
    assert is_better("loss", 0.2, 0.3)
    assert not is_better("loss", 0.4, 0.3)


def test_missing_candidate_never_wins_and_missing_incumbent_always_loses():
    """An undefined AUROC must not be promoted to best checkpoint."""
    assert not is_better("auroc", None, 0.5)
    assert is_better("auroc", 0.5, None)


def test_confusion_counts_treat_a_score_at_the_threshold_as_positive():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.5, 0.5, 0.9])
    assert confusion_counts(labels, scores, 0.5) == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_youden_threshold_on_a_hand_computed_case():
    # At 0.6: sensitivity 2/2, specificity 2/3. No other observed score does better.
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.6, 0.9])
    assert youden_threshold(labels, scores) == 0.6


def test_youden_threshold_needs_both_classes():
    with pytest.raises(ValueError, match="Both classes"):
        youden_threshold(np.array([0, 0]), np.array([0.1, 0.2]))


def test_youden_threshold_agrees_with_the_benchmark_project():
    """The test split is thresholded by benchmark-evaluation; the rule must match."""
    benchmark = pytest.importorskip("bn5212_benchmark.metrics")
    rng = np.random.default_rng(0)
    for _ in range(50):
        labels = rng.integers(0, 2, 30)
        labels[:2] = [0, 1]
        scores = np.round(rng.random(30), 2)  # rounding forces ties
        assert youden_threshold(labels, scores) == benchmark.choose_threshold(labels, scores)[0]


def test_a_folds_own_labels_never_choose_its_threshold():
    rng = np.random.default_rng(1)
    labels = np.tile([0, 0, 0, 1], 15)
    scores = np.clip(0.3 + 0.3 * labels + rng.normal(0, 0.2, 60), 0, 1)
    folds = np.repeat(np.arange(3), 20)

    decisions = cross_fitted_decisions(labels, scores, folds)
    flipped = labels.copy()
    flipped[folds == 0] = 1 - flipped[folds == 0]
    # Rewriting fold 0's labels changes the other folds' thresholds, not its own.
    assert np.array_equal(cross_fitted_decisions(flipped, scores, folds)[folds == 0],
                          decisions[folds == 0])

    summary = decision_summary(labels, decisions)
    assert summary["tn"] + summary["fp"] + summary["fn"] + summary["tp"] == 60
    assert summary["sensitivity"] == summary["tp"] / 15
