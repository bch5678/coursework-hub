"""Prospective transfer labels at the radiograph, independent of mortality."""
import json

import pandas as pd
import pytest

from scripts.make_synthetic_data import make_synthetic
from src.data.cohort import label_icu_transfer
from src.data.config import load_config, validate_config
from src.data.io import Audit
from src.data.pipeline import build


def test_transfer_labels_exclude_icu_at_or_before_cxr_and_respect_admission_end():
    admission = pd.Timestamp("2145-01-01")
    cxr = admission + pd.Timedelta(hours=6)
    cutoff = admission + pd.Timedelta(hours=96)
    frame = pd.DataFrame({
        "subject_id": [str(number) for number in range(7)],
        "hadm_id": [str(number) for number in range(7)],
        "study_time": cxr, "event_cutoff": cutoff,
    })
    stays = pd.DataFrame({
        "subject_id": ["1", "2", "3", "4", "5", "6", "6"],
        "hadm_id": ["1", "2", "3", "4", "5", "6", "6"],
        "intime": [cxr, cxr - pd.Timedelta(hours=1), cxr + pd.Timedelta(hours=1),
                   cutoff, cutoff + pd.Timedelta(hours=1), admission, cxr + pd.Timedelta(hours=2)],
    })
    result = label_icu_transfer(frame, stays, {}, Audit()).set_index("hadm_id")
    assert set(result.index) == {"0", "3", "4", "5"}
    assert result["label"].to_dict() == {"0": 0, "3": 1, "4": 0, "5": 0}
    assert result["label_name"].eq("icu_transfer").all()


def test_transfer_pipeline_uses_derived_labels_and_patient_disjoint_splits(tmp_path):
    config = make_synthetic(tmp_path / "raw")
    settings = json.loads(config.read_text())
    settings["label"] = {"kind": "icu_transfer", "column": "icu_transfer",
                         "positive_values": ["1"], "negative_values": ["0"]}
    config.write_text(json.dumps(settings))
    path = config.parent / "raw/mimiciv/icu/icustays.csv.gz"
    stays = pd.read_csv(path, dtype=str)
    stays = stays[stays["subject_id"].astype(int).mod(3).ne(0)].copy()
    stays["intime"] = (pd.to_datetime(stays["intime"]) + pd.Timedelta(hours=18)).astype(str)
    stays.to_csv(path, index=False)
    output, report = build(config)
    index = pd.read_csv(output / "index.csv", dtype={"hadm_id": str})
    expected = set(stays["hadm_id"])
    assert index["label"].tolist() == index["hadm_id"].isin(expected).astype(int).tolist()
    assert index.groupby("subject_id").split.nunique().max() == 1
    assert index["label_name"].eq("icu_transfer").all()
    assert json.loads((output / "dataset_spec.json").read_text())["task"] == "icu_transfer"
    assert any(row["stage"] == "not_previously_in_icu_at_cxr" for row in report["cohort_flow"])


def test_transfer_rejects_icu_stay_unit(tmp_path):
    config = make_synthetic(tmp_path / "raw")
    cfg = load_config(config)
    cfg["cohort"].update(unit="icu_stay", selection="first_per_icu_stay")
    cfg["label"] = {"kind": "icu_transfer", "column": "icu_transfer",
                    "positive_values": ["1"], "negative_values": ["0"]}
    with pytest.raises(ValueError, match="admission unit"):
        validate_config(cfg)
