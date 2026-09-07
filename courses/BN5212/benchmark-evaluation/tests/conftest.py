from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture()
def dataset_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "dataset_run"
    run_dir.mkdir()
    splits = ["train"] * 6 + ["val"] * 4 + ["test"] * 4
    labels = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index:02d}" for index in range(14)],
            "subject_id": [f"patient-{index:02d}" for index in range(14)],
            "hadm_id": [f"admission-{index:02d}" for index in range(14)],
            "study_id": [f"study-{index:02d}" for index in range(14)],
            "dicom_id": [f"dicom-{index:02d}" for index in range(14)],
            "label": labels,
            "sample_weight": 1.0,
            "split": splits,
        }
    )
    index_path = run_dir / "index.csv"
    frame.to_csv(index_path, index=False, lineterminator="\n")
    spec = {
        "schema_version": "1.0",
        "task": "in_hospital_mortality",
        "index": "index.csv",
        "index_sha256": file_sha256(index_path),
        "split_unit": "subject_id",
        "split_seed": 5212,
        "dataset_versions": {"mimic_iv": "synthetic", "mimic_cxr": "synthetic"},
    }
    spec_path = run_dir / "dataset_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    success = {
        "status": "complete",
        "artifacts_sha256": {
            "index.csv": file_sha256(index_path),
            "dataset_spec.json": file_sha256(spec_path),
        },
    }
    (run_dir / "SUCCESS.json").write_text(json.dumps(success, indent=2) + "\n", encoding="utf-8")
    return run_dir


@pytest.fixture()
def prediction_files(dataset_run: Path, tmp_path: Path) -> tuple[Path, Path]:
    index = pd.read_csv(dataset_run / "index.csv", dtype={"sample_id": str})
    files = []
    scores = {
        "val": [0.10, 0.85, 0.20, 0.75],
        "test": [0.15, 0.90, 0.30, 0.80],
    }
    for split in ["val", "test"]:
        frame = index.loc[index["split"].eq(split), ["sample_id"]].copy()
        frame["y_score"] = scores[split]
        path = tmp_path / f"{split}_predictions.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        files.append(path)
    return files[0], files[1]
