"""Atomic end-to-end evaluation from model probability files."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregation import aggregate_predictions
from .io import load_dataset_index, load_predictions, sha256, write_json
from .metrics import binary_metrics, choose_threshold, patient_bootstrap_intervals
from .plots import save_evaluation_plots
from .report import METRIC_LABELS, write_run_report


def _git_commit(start: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def _warnings(val_metrics: dict[str, Any], test_metrics: dict[str, Any], intervals: dict[str, Any]) -> list[str]:
    warnings = []
    if val_metrics["auroc"] is None:
        warnings.append("Validation AUROC is undefined because the split contains one outcome class.")
    if test_metrics["auroc"] is None:
        warnings.append("Test AUROC is undefined because the split contains one outcome class.")
    for name, value in intervals.items():
        if value.get("successful_replicates", 0) == 0:
            warnings.append(f"No valid patient-bootstrap replicates were available for {name}.")
    return warnings


def _metrics_rows(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    intervals = metrics.get("test_patient_bootstrap", {})
    for split in ["val", "test"]:
        for name in METRIC_LABELS:
            item = {"split": split, "metric": name, "value": metrics["splits"][split].get(name), "lower": None, "upper": None}
            if split == "test" and name in intervals:
                item["lower"] = intervals[name]["lower"]
                item["upper"] = intervals[name]["upper"]
            rows.append(item)
    return pd.DataFrame(rows)


def evaluate_prediction_files(
    *,
    run_dir: str | Path,
    val_predictions: str | Path,
    test_predictions: str | Path,
    output_dir: str | Path,
    model_name: str,
    model_version: str,
    evaluation_unit: str = "admission",
    threshold_method: str = "youden_j",
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 5212,
    checkpoint: str | Path | None = None,
) -> Path:
    if not model_name.strip() or not model_version.strip():
        raise ValueError("model_name and model_version cannot be empty")
    run_dir = Path(run_dir).resolve()
    val_predictions = Path(val_predictions).resolve()
    test_predictions = Path(test_predictions).resolve()
    output_dir = Path(output_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable evaluation run: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))

    try:
        dataset_index, spec = load_dataset_index(run_dir)
        val = load_predictions(val_predictions, dataset_index, "val")
        test = load_predictions(test_predictions, dataset_index, "test")
        val_evaluation = aggregate_predictions(val, evaluation_unit)
        test_evaluation = aggregate_predictions(test, evaluation_unit)
        threshold, threshold_details = choose_threshold(
            val_evaluation["label"], val_evaluation["y_score"], threshold_method
        )
        val_metrics = binary_metrics(val_evaluation["label"], val_evaluation["y_score"], threshold)
        test_metrics = binary_metrics(test_evaluation["label"], test_evaluation["y_score"], threshold)
        intervals = patient_bootstrap_intervals(
            test_evaluation,
            threshold,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        metrics = {
            "schema_version": "1.0",
            "model": {"name": model_name.strip(), "version": model_version.strip()},
            "dataset": {
                "task": spec.get("task", "unknown"),
                "index_sha256": spec["index_sha256"],
                "split_unit": spec.get("split_unit", "subject_id"),
                "dataset_versions": spec.get("dataset_versions", {}),
            },
            "evaluation_unit": evaluation_unit,
            "threshold_selection": {"threshold": threshold, **threshold_details, "source_split": "val"},
            "splits": {"val": val_metrics, "test": test_metrics},
            "bootstrap": {"unit": "subject_id", "samples": bootstrap_samples, "seed": bootstrap_seed, "confidence": 0.95},
            "test_patient_bootstrap": intervals,
        }
        metrics["warnings"] = _warnings(val_metrics, test_metrics, intervals)

        combined = pd.concat([val, test], ignore_index=True)
        combined.insert(0, "model_name", model_name.strip())
        combined.insert(1, "model_version", model_version.strip())
        combined.to_csv(staging / "predictions.csv", index=False, lineterminator="\n")
        write_json(staging / "metrics.json", metrics)
        _metrics_rows(metrics).to_csv(staging / "metrics.csv", index=False, lineterminator="\n")
        write_json(
            staging / "evaluation_config.json",
            {
                "model_name": model_name.strip(),
                "model_version": model_version.strip(),
                "evaluation_unit": evaluation_unit,
                "threshold_method": threshold_method,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            },
        )
        plot_paths = save_evaluation_plots(test_evaluation, threshold, staging)
        project_root = Path(__file__).resolve().parents[5]
        manifest = {
            "schema_version": "1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(project_root),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "dataset_run_dir": str(run_dir),
            "dataset_index_sha256": spec["index_sha256"],
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path else None,
            "inputs": {
                "val_predictions": {"path": str(val_predictions), "sha256": sha256(val_predictions)},
                "test_predictions": {"path": str(test_predictions), "sha256": sha256(test_predictions)},
            },
        }
        write_json(staging / "run_manifest.json", manifest)
        write_run_report(staging / "report.html", metrics, manifest, plot_paths)
        artifacts = {path.name: sha256(path) for path in staging.iterdir() if path.is_file()}
        write_json(staging / "SUCCESS.json", {"status": "complete", "artifacts_sha256": artifacts})
        os.replace(staging, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
