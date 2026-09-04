from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bn5212_benchmark.io import load_dataset_index, load_predictions


def test_dataset_contract_and_exact_predictions(dataset_run: Path, prediction_files: tuple[Path, Path]):
    index, spec = load_dataset_index(dataset_run)
    val = load_predictions(prediction_files[0], index, "val")
    assert spec["task"] == "in_hospital_mortality"
    assert len(val) == 4
    assert set(["sample_id", "subject_id", "hadm_id", "label", "y_score"]) <= set(val.columns)


def test_missing_prediction_is_rejected(dataset_run: Path, prediction_files: tuple[Path, Path], tmp_path: Path):
    index, _ = load_dataset_index(dataset_run)
    truncated = pd.read_csv(prediction_files[0]).iloc[:-1]
    path = tmp_path / "truncated.csv"
    truncated.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing=1"):
        load_predictions(path, index, "val")


def test_changed_dataset_artifact_is_rejected(dataset_run: Path):
    index_path = dataset_run / "index.csv"
    index_path.write_text(index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset_index(dataset_run)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_are_rejected(dataset_run: Path, tmp_path: Path, score: float):
    index, _ = load_dataset_index(dataset_run)
    val = index[index["split"].eq("val")][["sample_id"]].copy()
    val["y_score"] = [0.1, 0.2, 0.3, score]
    path = tmp_path / "invalid.csv"
    val.to_csv(path, index=False)
    with pytest.raises(ValueError, match="y_score"):
        load_predictions(path, index, "val")
