"""Dataset and prediction contracts used by every benchmark model."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_INDEX_COLUMNS = {
    "sample_id",
    "subject_id",
    "hadm_id",
    "label",
    "sample_weight",
    "split",
}
SPLITS = {"train", "val", "test"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return value


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_success_manifest(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    success_path = run_dir / "SUCCESS.json"
    if not success_path.is_file():
        raise ValueError("Dataset run is incomplete: missing SUCCESS.json")
    success = read_json(success_path)
    if success.get("status") != "complete":
        raise ValueError("Dataset SUCCESS.json does not report a complete run")
    artifacts = success.get("artifacts_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError("Dataset SUCCESS.json has no artifact checksum map")
    for name, expected in artifacts.items():
        artifact = run_dir / name
        if not artifact.is_file():
            raise ValueError(f"Dataset artifact is missing: {name}")
        if sha256(artifact) != expected:
            raise ValueError(f"Dataset artifact checksum mismatch: {name}")
    return success


def load_dataset_index(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    verify_success_manifest(run_dir)
    spec_path = run_dir / "dataset_spec.json"
    index_path = run_dir / "index.csv"
    if not spec_path.is_file() or not index_path.is_file():
        raise ValueError("Dataset run must contain dataset_spec.json and index.csv")
    spec = read_json(spec_path)
    if spec.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported dataset schema: {spec.get('schema_version')!r}")
    if sha256(index_path) != spec.get("index_sha256"):
        raise ValueError("index.csv checksum does not match dataset_spec.json")

    string_columns = ["sample_id", "subject_id", "hadm_id", "study_id", "dicom_id"]
    frame = pd.read_csv(index_path, dtype={name: str for name in string_columns})
    missing = REQUIRED_INDEX_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"index.csv is missing required columns: {sorted(missing)}")
    if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("index.csv must contain unique, nonempty sample_id values")
    if set(frame["split"].unique()) != SPLITS:
        raise ValueError("index.csv must contain nonempty train, val and test splits")
    if frame.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage detected across dataset splits")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("index.csv labels must be binary 0/1 values")
    weights = pd.to_numeric(frame["sample_weight"], errors="coerce")
    if weights.isna().any() or not weights.map(lambda value: math.isfinite(value) and value > 0).all():
        raise ValueError("index.csv sample_weight values must be finite and positive")
    frame["label"] = labels.astype(int)
    frame["sample_weight"] = weights.astype(float)
    return frame, spec


def load_predictions(
    path: str | Path,
    dataset_index: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    if split not in {"val", "test"}:
        raise ValueError("Prediction evaluation is restricted to val or test")
    path = Path(path)
    predictions = pd.read_csv(path, dtype={"sample_id": str})
    missing = {"sample_id", "y_score"} - set(predictions.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if predictions.empty or predictions["sample_id"].isna().any():
        raise ValueError(f"{path.name} contains no usable predictions")
    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate sample_id values")
    score = pd.to_numeric(predictions["y_score"], errors="coerce")
    if score.isna().any() or not score.map(math.isfinite).all():
        raise ValueError(f"{path.name} y_score must contain only finite numbers")
    if not score.between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{path.name} y_score must be a probability in [0, 1]")
    predictions = predictions[["sample_id"]].assign(y_score=score.astype(float))

    expected = dataset_index.loc[dataset_index["split"].eq(split)].copy()
    expected_ids = set(expected["sample_id"])
    actual_ids = set(predictions["sample_id"])
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids
    if missing_ids or extra_ids:
        raise ValueError(
            f"{path.name} does not exactly match split={split}: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    identity_columns = [
        name
        for name in ["sample_id", "subject_id", "hadm_id", "study_id", "dicom_id"]
        if name in expected.columns
    ]
    canonical = expected[identity_columns + ["label", "sample_weight", "split"]].merge(
        predictions,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    return canonical.sort_values("sample_id").reset_index(drop=True)
