from __future__ import annotations

import pandas as pd
import pytest

from bn5212_benchmark.aggregation import aggregate_predictions
from bn5212_benchmark.metrics import binary_metrics, choose_threshold, patient_bootstrap_intervals


def test_binary_metrics_known_example():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.5)
    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 1
    assert result["tp"] == 1
    assert result["auroc"] == pytest.approx(0.75)
    assert result["specificity"] == pytest.approx(1.0)
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_threshold_selection_uses_validation_objective():
    threshold, details = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8], "youden_j")
    assert threshold == pytest.approx(0.7)
    assert details["objective"] == pytest.approx(1.0)
    assert details["method"] == "youden_j"


def test_threshold_requires_two_validation_classes():
    with pytest.raises(ValueError, match="both classes"):
        choose_threshold([0, 0], [0.1, 0.2])
    threshold, details = choose_threshold([0, 0], [0.1, 0.2], "fixed_0.5")
    assert threshold == 0.5
    assert details["method"] == "fixed_0.5"


def test_class_dependent_metrics_are_undefined_when_class_is_absent():
    result = binary_metrics([0, 0], [0.1, 0.2], threshold=0.5)
    assert result["auroc"] is None
    assert result["sensitivity"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["specificity"] == 1.0


def test_patient_bootstrap_is_reproducible():
    frame = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "y_score": [0.1, 0.3, 0.7, 0.9],
        }
    )
    first = patient_bootstrap_intervals(frame, 0.5, samples=50, seed=7)
    second = patient_bootstrap_intervals(frame, 0.5, samples=50, seed=7)
    assert first == second
    assert first["accuracy"]["estimate"] == pytest.approx(1.0)
    assert first["accuracy"]["successful_replicates"] == 50


def test_admission_aggregation_uses_sample_weights():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "subject_id": ["p1", "p1", "p2"],
            "hadm_id": ["h1", "h1", "h2"],
            "label": [1, 1, 0],
            "sample_weight": [0.25, 0.75, 1.0],
            "split": ["test", "test", "test"],
            "y_score": [0.0, 1.0, 0.2],
        }
    )
    result = aggregate_predictions(frame, "admission").set_index("hadm_id")
    assert len(result) == 2
    assert result.loc["h1", "y_score"] == pytest.approx(0.75)
    assert result.loc["h1", "source_samples"] == 2
