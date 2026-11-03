import numpy as np
import pandas as pd
import pytest

from bn5212_training.paired_comparison import paired_auroc


def predictions():
    return pd.DataFrame({
        "sample_id": list("abcdef"), "hadm_id": list("abcdef"),
        "subject_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
        "label": [0, 1, 0, 1, 0, 1], "y_score": [.1, .9, .2, .8, .3, .7],
    })


def test_identical_predictions_have_zero_paired_uncertainty_even_if_shuffled():
    frame = predictions()
    result = paired_auroc(frame, frame.sample(frac=1, random_state=5), samples=30)
    assert result["delta_auroc"] == result["lower"] == result["upper"] == 0
    assert result["successful_replicates"] == 30
    assert result["patients"] == 3 and result["n"] == 6


def test_patient_cluster_resampling_retains_both_admissions():
    frame = predictions()
    baseline = frame.assign(y_score=1 - frame["y_score"])
    result = paired_auroc(frame, baseline, samples=30)
    assert result["delta_auroc"] == result["lower"] == result["upper"] == 1
    assert result["successful_replicates"] == 30


@pytest.mark.parametrize("column,value", [
    ("label", 1), ("subject_id", "other"), ("sample_id", "other"),
    ("hadm_id", "other"), ("y_score", np.nan), ("y_score", 1.5),
])
def test_mismatched_or_invalid_predictions_are_rejected(column, value):
    frame = predictions()
    changed = frame.copy()
    changed.loc[0, column] = value
    with pytest.raises(ValueError):
        paired_auroc(frame, changed, samples=5)


def test_single_class_has_no_auroc_or_valid_replicates():
    frame = predictions().assign(label=0)
    result = paired_auroc(frame, frame, samples=5)
    assert result["delta_auroc"] is None and result["lower"] is None
    assert result["successful_replicates"] == 0
