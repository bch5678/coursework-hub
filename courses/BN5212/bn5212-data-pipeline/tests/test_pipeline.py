import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.make_synthetic_data import make_synthetic, write_dicom
from src.data.cohort import align_images
from src.data.config import load_config, validate_config
from src.data.dataset import MimicCXRDataset, make_dataloader
from src.data.images import decode_image, image_array
from src.data.io import Audit, deduplicate, safe_path, sha256
from src.data.metadata import parse_study_time
from src.data.pipeline import build
from src.data.splits import assign_splits
from src.data.tables import load_hospital_tables
from test_dataloader import check_run


@pytest.fixture(scope="module")
def png_run(tmp_path_factory):
    config = make_synthetic(tmp_path_factory.mktemp("png") / "fixture")
    output, report = build(config)
    return config, output, report


def test_end_to_end_png(png_run):
    config, output, report = png_run
    check_run(output)
    assert report["counts"] == {"images": 68, "studies": 68, "admissions": 68, "patients": 35}
    frame = pd.read_csv(output / "index.csv")
    assert frame.groupby("hadm_id").size().max() == 1
    assert frame.label.sum() == 9
    assert frame.hours_since_admission.between(6, 6.001).all()
    assert "synthetic-corrupt" not in set(frame.dicom_id)
    audit = {r["stage"]: r for r in report["cohort_flow"]}
    for stage in ["deduplicate_admissions", "deduplicate_cxr", "valid_study_timestamp", "present_in_local_subset", "allowed_view", "strictly_before_outcome", "early_imaging_window", "decodable_nonconstant_image"]:
        assert audit[stage]["excluded"] > 0, stage
    success = json.loads((output / "SUCCESS.json").read_text())
    for filename, checksum in success["artifacts_sha256"].items():
        assert sha256(output / filename) == checksum


def test_immutable_output(png_run):
    with pytest.raises(FileExistsError):
        build(png_run[0])


def test_split_reproducible_with_input_reorder(png_run):
    config, output, _ = png_run
    frame = pd.read_csv(output / "index.csv", dtype={"subject_id": str}).drop(columns="split")
    cfg = load_config(config)
    _, first = assign_splits(frame, cfg, Audit())
    _, second = assign_splits(frame.sample(frac=1, random_state=99), cfg, Audit())
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("subject_id").split.nunique().max() == 1


def test_dataloader_seed_pickle_and_workers(png_run):
    output = png_run[1]
    a = make_dataloader(output, "train", seed=42)
    b = make_dataloader(output, "train", seed=42)
    assert next(iter(a))["sample_id"] == next(iter(b))["sample_id"]
    dataset = pickle.loads(pickle.dumps(a.dataset))
    assert torch.equal(dataset[0]["image"], a.dataset[0]["image"])
    worker_loader = make_dataloader(output, "val", num_workers=2)
    batch = next(iter(worker_loader))
    assert batch["image"].shape[1:] == (1, 32, 32)


def test_timestamp_precision_and_invalids():
    date = pd.Series(["21450101"] * 8)
    times = pd.Series(["91502.123456", "0.0", "235959.1", "240000", "126000", "", "hello", "1200.12"])
    parsed = parse_study_time(date, times)
    assert parsed[0] == pd.Timestamp("2145-01-01 09:15:02.123456")
    assert parsed[1] == pd.Timestamp("2145-01-01 00:00:00")
    assert parsed[2] == pd.Timestamp("2145-01-01 23:59:59.100000")
    assert parsed[3:7].isna().all()
    assert parsed[7] == pd.Timestamp("2145-01-01 00:12:00.120000")


def test_no_dod_used_for_mortality(png_run):
    cfg = load_config(png_run[0])
    admissions, _ = load_hospital_tables(cfg, Audit())
    assert "dod" not in admissions
    assert not admissions.age_eligible[admissions.subject_id.eq("90000035")].any()


def test_overlapping_admissions_are_excluded(png_run):
    cfg = load_config(png_run[0]); audit = Audit()
    a, _ = load_hospital_tables(cfg, audit)
    one = a[(a.subject_id == "90000000")].iloc[[0]].copy()
    another = one.copy(); another["hadm_id"] = "89999999"
    a = pd.concat([one, another], ignore_index=True)
    cxr = pd.DataFrame([{"subject_id": "90000000", "study_id": "1", "dicom_id": "x", "study_time": one.iloc[0].admittime + pd.Timedelta(hours=6)}])
    result = align_images(cxr, a, cfg, audit)
    assert result.empty
    assert any(r["stage"] == "unique_inpatient_interval" and r["excluded"] == 1 for r in audit.flow)


def test_half_open_interval_and_death_boundary(png_run):
    cfg = load_config(png_run[0]); audit = Audit()
    a, _ = load_hospital_tables(cfg, audit)
    a = a[(a.subject_id == "90000000") & (a.hospital_expire_flag == "1")]
    row = a.iloc[0]
    cxr = pd.DataFrame([{"subject_id": row.subject_id, "study_id": str(i + 1), "dicom_id": str(i + 1), "study_time": t}
                        for i, t in enumerate([row.admittime, row.deathtime, row.dischtime, row.admittime - pd.Timedelta(microseconds=1)])])
    cfg["alignment"]["max_hours_after_admission"] = None
    result = align_images(cxr, a, cfg, audit)
    assert result.dicom_id.tolist() == ["1"]


def test_configurable_labels_and_multiview_weights(tmp_path):
    config = make_synthetic(tmp_path / "custom")
    cfg = json.loads(config.read_text())
    cfg["label"] = {"kind": "admission_binary", "column": "custom_outcome", "positive_values": ["yes"], "negative_values": ["no"]}
    cfg["cohort"]["selection"] = "all_images"
    config.write_text(json.dumps(cfg))
    output, _ = build(config)
    frame = pd.read_csv(output / "index.csv")
    assert frame.groupby("hadm_id").size().max() == 2
    np.testing.assert_allclose(frame.groupby("hadm_id").sample_weight.sum(), 1)
    assert frame.label_name.eq("custom_outcome").all()


def test_dicom_end_to_end(tmp_path):
    config = make_synthetic(tmp_path / "dicom", image_format="dicom")
    output, report = build(config)
    check_run(output)
    assert report["counts"]["images"] == 68
    frame = pd.read_csv(output / "index.csv")
    assert frame.image_path.str.endswith(".dcm").all()


def test_monochrome_inversion_and_channels(tmp_path):
    a, b = tmp_path / "a.dcm", tmp_path / "b.dcm"
    write_dicom(a, 90000000, "21450101", "060000", invert=False)
    write_dicom(b, 90000000, "21450101", "060000", invert=True)
    np.testing.assert_allclose(np.asarray(decode_image(a), dtype=float) + np.asarray(decode_image(b), dtype=float), 255, atol=1)
    assert image_array(a, 32, 3).shape == (3, 32, 32)


def test_conflicting_duplicates_and_paths(tmp_path):
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        deduplicate(pd.DataFrame({"key": ["x", "x"], "label": [0, 1]}), ["key"], "test")
    for path in ["../secret", "..\\secret", "/outside", "C:/outside"]:
        with pytest.raises(ValueError):
            safe_path(tmp_path, path)


def test_invalid_split_configuration(png_run):
    cfg = load_config(png_run[0]); cfg["split"]["test"] = 0.8
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_minimum_patient_count(png_run):
    with pytest.raises(ValueError, match="three eligible"):
        assign_splits(pd.DataFrame({"subject_id": ["1", "2"], "label": [0, 1]}), load_config(png_run[0]), Audit())
