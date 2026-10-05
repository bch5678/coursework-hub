"""Pretraining the clinical encoder on stays outside the study cohort.

What has to hold for the pretrained encoder to be admissible: no study patient
reaches the external cohort, the external stays obey the cohort's own
eligibility rule, the encoder arrives in a study model unchanged and stays that
way, and the statistics its inputs are scaled with come along with it.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bn5212_training.clinical import MIMIC_BENCHMARK_VARIABLES
from bn5212_training.config import config_from_dict
from bn5212_training.crossval import _patient_table
from bn5212_training.model import build_model, load_checkpoint
from bn5212_training.pretrain import (
    build_external_cohort,
    load_pretrained_normalizer,
    pretrain_clinical,
    score_study_cohort,
)
from bn5212_training.trainer import train

HOURS = 8


def _mimic_tables(root: Path, rows: list[dict]) -> Path:
    """icustays, admissions and patients for the given stays, one admission each."""
    (root / "icu").mkdir(parents=True)
    (root / "hosp").mkdir()
    frame = pd.DataFrame(rows)
    frame[["subject_id", "hadm_id", "stay_id", "intime"]].to_csv(
        root / "icu" / "icustays.csv", index=False
    )
    frame[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
           "hospital_expire_flag"]].to_csv(root / "hosp" / "admissions.csv", index=False)
    frame[["subject_id", "anchor_age", "anchor_year"]].drop_duplicates("subject_id").to_csv(
        root / "hosp" / "patients.csv", index=False
    )
    return root


def _stay(subject, *, stay_hours=120, died_at=None, age=60):
    intime = pd.Timestamp("2150-01-01 08:00:00")
    discharge = intime + pd.Timedelta(hours=stay_hours)
    death = None if died_at is None else intime + pd.Timedelta(hours=died_at)
    return {
        "subject_id": str(subject), "hadm_id": f"h{subject}", "stay_id": f"s{subject}",
        "intime": intime, "admittime": intime, "dischtime": discharge if death is None else death,
        "deathtime": death, "hospital_expire_flag": int(death is not None),
        "anchor_age": age, "anchor_year": 2150,
    }


def test_no_study_patient_reaches_the_external_cohort(dataset_run, tmp_path):
    study = sorted(pd.read_csv(Path(dataset_run) / "index.csv", dtype=str)["subject_id"].unique())
    outsiders = [f"9{number:04d}" for number in range(40)]
    mimic = _mimic_tables(tmp_path / "mimic", [_stay(s) for s in study + outsiders])

    summary = build_external_cohort(dataset_run, mimic, tmp_path / "external", max_stays=None)
    index = pd.read_csv(tmp_path / "external" / "index.csv", dtype=str)
    assert not (set(index["subject_id"]) & set(study))
    assert set(index["subject_id"]) == set(outsiders)
    assert summary["study_patients_excluded"] == len(study)
    # A patient sits on one side of the external split only.
    assert index.groupby("subject_id")["split"].nunique().eq(1).all()


def test_external_stays_obey_the_cohort_eligibility_rule(dataset_run, tmp_path):
    """The outcome must still be open 48 h after ICU admission, as in the cohort."""
    rows = [
        _stay("90001"),                             # eligible survivor
        _stay("90002", died_at=100),                # eligible death, after the window
        _stay("90003", died_at=30),                 # died inside the window
        _stay("90004", stay_hours=40),              # discharged inside the window
        _stay("90005", age=15),                     # a minor
    ]
    build_external_cohort(dataset_run, _mimic_tables(tmp_path / "mimic", rows),
                          tmp_path / "external", max_stays=None)
    index = pd.read_csv(tmp_path / "external" / "index.csv", dtype=str)
    assert sorted(index["subject_id"]) == ["90001", "90002"]
    assert index.set_index("subject_id")["label"].to_dict() == {"90001": "0", "90002": "1"}


def test_patients_beyond_a_truncated_chartevents_file_are_dropped(dataset_run, tmp_path):
    """A cut-off chartevents holds no vital signs for the patients after the cut.

    Found on the development machine: the file ended mid-row at 72% of the
    patients, and every later ICU stay looked like a stay with lab values only.
    """
    rows = [_stay(str(90001 + number)) for number in range(6)]
    mimic = _mimic_tables(tmp_path / "mimic", rows)
    header = "subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valuenum,valueuom,warning"
    events = [f"{90001 + number},h,s,1,2150-01-01 09:00:00,2150-01-01 09:00:00,220045,80,80,bpm,0"
              for number in range(4)]
    # The last row is cut off mid-way, as an interrupted unzip leaves it.
    (mimic / "icu" / "chartevents.csv").write_text(
        "\n".join([header, *events]) + "\n90004,h,s,1,2150-01-01 10:00", encoding="utf-8"
    )

    summary = build_external_cohort(dataset_run, mimic, tmp_path / "external", max_stays=None)
    index = pd.read_csv(tmp_path / "external" / "index.csv", dtype=str)
    # 90004 is the last patient with a complete row and may itself be incomplete.
    assert sorted(index["subject_id"]) == ["90001", "90002", "90003"]
    assert summary["chartevents_truncated_at_subject_id"] == 90004


def test_a_complete_chartevents_file_drops_nobody(dataset_run, tmp_path):
    rows = [_stay(str(90001 + number)) for number in range(3)]
    mimic = _mimic_tables(tmp_path / "mimic", rows)
    lines = ["subject_id,hadm_id,stay_id,caregiver_id,charttime,storetime,itemid,value,valuenum,valueuom,warning"]
    lines += [f"{90001 + number},h,s,1,2150-01-01 09:00:00,2150-01-01 09:00:00,220045,80,80,bpm,0"
              for number in range(3)]
    (mimic / "icu" / "chartevents.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = build_external_cohort(dataset_run, mimic, tmp_path / "external", max_stays=None)
    assert summary["stays"] == 3
    assert summary["chartevents_truncated_at_subject_id"] is None


def _external_cohort(directory: Path, stays: int = 80) -> Path:
    """A small external cohort with a clinical table whose signal follows the label."""
    rng = np.random.default_rng(0)
    directory.mkdir(parents=True)
    index, events = [], []
    for number in range(stays):
        label = int(number % 4 == 0)
        index.append({
            "subject_id": f"x{number}", "hadm_id": f"xh{number}", "stay_id": f"xs{number}",
            "admittime": "2150-01-01 00:00:00", "study_time": "2150-01-03 00:00:00",
            "hours_since_admission": 48.0, "label": label,
            "split": "pretrain_val" if number % 5 == 0 else "pretrain_train",
        })
        for variable in MIMIC_BENCHMARK_VARIABLES:
            for hour in range(HOURS):
                if rng.random() < 0.7:
                    events.append({"hadm_id": f"xh{number}", "stay_id": f"xs{number}",
                                   "hour": hour, "variable": variable,
                                   "value": rng.normal(50 + 10 * label, 5)})
    pd.DataFrame(index).to_csv(directory / "index.csv", index=False)
    pd.DataFrame(events).to_csv(directory / "clinical_features.csv", index=False)
    (directory / "cohort.json").write_text(
        json.dumps({"study_patients_excluded": 0, "study_index_sha256": "fixture"}),
        encoding="utf-8",
    )
    return directory


def _clinical_config(base_config: dict, **encoder):
    payload = json.loads(json.dumps(base_config))
    payload["experiment"] = "pretrain"
    payload["modalities"] = ["clinical"]
    payload["fusion"] = {"name": "clinical_only", "embed_dim": 32, "pooling": "mean"}
    payload["clinical_encoder"] = {"name": "linear_projection", "embed_dim": 32, **encoder}
    payload["optim"] = {"epochs": 2, "warmup_epochs": 0, "early_stopping_patience": 3}
    payload["selection_metric"] = "loss"
    return config_from_dict(payload)


@pytest.fixture
def pretrained(base_config, tmp_path):
    cohort = _external_cohort(tmp_path / "external")
    checkpoint = tmp_path / "external" / "clinical_encoder.pt"
    report = pretrain_clinical(
        cohort, _clinical_config(base_config), checkpoint, epochs=3, batch_size=16, device="cpu"
    )
    return checkpoint, report


def test_pretraining_reports_only_external_validation(pretrained):
    checkpoint, report = pretrained
    assert checkpoint.is_file()
    external = report["external_validation"]
    assert external["n"] == 16 and external["fitting_stays"] == 64
    assert 1 <= external["selected_epoch"] <= 3


def test_the_pretrained_encoder_arrives_unchanged_and_frozen(base_config, pretrained):
    checkpoint, _ = pretrained
    cfg = _clinical_config(base_config, pretrained=str(checkpoint), freeze=True)
    model = build_model(cfg, num_variables=len(MIMIC_BENCHMARK_VARIABLES), num_timesteps=HOURS)

    stored = torch.load(checkpoint, weights_only=False)["encoder_state"]
    for name, value in model.clinical_encoder.state_dict().items():
        assert torch.equal(value, stored[name]), name
    assert not any(p.requires_grad for p in model.clinical_encoder.parameters())
    assert any(p.requires_grad for p in model.head.parameters())
    # Frozen means its dropout stays off while the rest of the model trains.
    model.train()
    assert model.training and not model.clinical_encoder.training


def test_training_does_not_move_a_frozen_pretrained_encoder(base_config, pretrained, tmp_path):
    checkpoint, _ = pretrained
    cfg = _clinical_config(base_config, pretrained=str(checkpoint), freeze=True)
    subjects = sorted(_patient_table(cfg)["subject_id"])
    result = train(cfg, output_dir=tmp_path / "runs", run_id="frozen",
                   fit_subjects=subjects[:-6], select_subjects=subjects[-6:])

    saved = Path(result["run_dir"]) / "checkpoint_best.pt"
    stored = torch.load(checkpoint, weights_only=False)["encoder_state"]
    # Remove the pretraining file: a study checkpoint must load without it.
    checkpoint.unlink()
    model, payload = load_checkpoint(saved)
    for name, value in model.clinical_encoder.state_dict().items():
        assert torch.equal(value, stored[name]), name
    # The inputs were scaled with the external statistics, not refitted on the fold.
    assert "external" in payload["clinical_normalizer"]["fitted_on"]


def test_the_normalizer_travels_with_the_encoder(pretrained):
    checkpoint, _ = pretrained
    normalizer = load_pretrained_normalizer(checkpoint)
    assert normalizer.variable_names == tuple(sorted(MIMIC_BENCHMARK_VARIABLES))
    # Fitted on values drawn around 50-60, so it cannot be the identity default.
    assert np.all(normalizer.mean > 40) and np.all(normalizer.std > 1)


def test_an_encoder_of_another_shape_is_refused(base_config, pretrained):
    checkpoint, _ = pretrained
    cfg = _clinical_config(base_config, name="variable_projection", pretrained=str(checkpoint))
    with pytest.raises(ValueError, match="different clinical encoder"):
        build_model(cfg, num_variables=len(MIMIC_BENCHMARK_VARIABLES), num_timesteps=HOURS)


def test_freezing_needs_pretrained_weights(base_config):
    with pytest.raises(ValueError, match="needs clinical_encoder.pretrained"):
        _clinical_config(base_config, freeze=True)


def test_scoring_the_study_cohort_covers_each_split_once(base_config, pretrained, tmp_path):
    checkpoint, _ = pretrained
    cfg = _clinical_config(base_config)
    result = score_study_cohort(
        checkpoint, cfg, tmp_path / "scored", n_splits=3, bootstrap_samples=50, device="cpu"
    )
    root = Path(result["run_dir"])
    index = pd.read_csv(Path(cfg.data.run_dir) / "index.csv", dtype={"sample_id": str})
    pooled = pd.read_csv(root / "out_of_fold_predictions.csv", dtype={"sample_id": str})
    assert set(pooled["sample_id"]) == set(
        index.loc[index["split"].isin(["train", "val"]), "sample_id"]
    )
    assert pooled["fold"].notna().all()
    for split in ("val", "test"):
        frame = pd.read_csv(root / f"predictions_{split}.csv", dtype={"sample_id": str})
        assert set(frame["sample_id"]) == set(index.loc[index["split"].eq(split), "sample_id"])
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert "no parameter was fitted" in manifest["evaluation"]
