from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bn5212_benchmark.baselines import run_prevalence_baseline
from bn5212_benchmark.compare import build_leaderboard
from bn5212_benchmark.evaluation import evaluate_prediction_files


def test_evaluation_writes_complete_offline_report(
    dataset_run: Path,
    prediction_files: tuple[Path, Path],
    tmp_path: Path,
):
    output = evaluate_prediction_files(
        run_dir=dataset_run,
        val_predictions=prediction_files[0],
        test_predictions=prediction_files[1],
        output_dir=tmp_path / "perfect_model",
        model_name="perfect_model",
        model_version="test-1",
        bootstrap_samples=50,
        bootstrap_seed=9,
    )
    expected = {
        "predictions.csv",
        "metrics.json",
        "metrics.csv",
        "evaluation_config.json",
        "run_manifest.json",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "confusion_matrix.png",
        "report.html",
        "SUCCESS.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold_selection"]["source_split"] == "val"
    assert metrics["splits"]["test"]["auroc"] == pytest.approx(1.0)
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "perfect_model" in report
    assert "data:image/png;base64," in report
    assert "Test 95% patient bootstrap CI" in report
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_prediction_files(
            run_dir=dataset_run,
            val_predictions=prediction_files[0],
            test_predictions=prediction_files[1],
            output_dir=output,
            model_name="perfect_model",
            model_version="test-1",
            bootstrap_samples=0,
        )


def test_prevalence_baseline_and_html_leaderboard(
    dataset_run: Path,
    prediction_files: tuple[Path, Path],
    tmp_path: Path,
):
    baseline = run_prevalence_baseline(
        run_dir=dataset_run,
        output_dir=tmp_path / "prevalence",
        bootstrap_samples=20,
    )
    model = evaluate_prediction_files(
        run_dir=dataset_run,
        val_predictions=prediction_files[0],
        test_predictions=prediction_files[1],
        output_dir=tmp_path / "model",
        model_name="model",
        model_version="1",
        bootstrap_samples=20,
    )
    baseline_metrics = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    assert baseline_metrics["splits"]["test"]["auroc"] == pytest.approx(0.5)
    csv_path, html_path = build_leaderboard(
        [baseline, model],
        output_csv=tmp_path / "leaderboard.csv",
        output_html=tmp_path / "leaderboard.html",
    )
    leaderboard = pd.read_csv(csv_path)
    assert leaderboard.iloc[0]["model"] == "model"
    assert "prevalence_baseline" in html_path.read_text(encoding="utf-8")
