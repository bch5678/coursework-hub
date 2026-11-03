"""The preparation stage builds/reuses the same data and refuses stale inputs."""
from __future__ import annotations

from dataclasses import replace
import json
import importlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from bn5212_training import upstream
from bn5212_training.config import config_from_dict
from bn5212_training.prepare import prepare_inputs


def test_preparation_reuses_the_index_and_existing_cache(base_config):
    cfg = config_from_dict(base_config)
    prepared, report = prepare_inputs(cfg)
    original = (Path(cfg.data.run_dir) / "index.csv").read_bytes()
    cache_path = Path(prepared.data.image_cache)
    timestamp = cache_path.stat().st_mtime_ns
    repeated, second = prepare_inputs(prepared)
    assert (Path(cfg.data.run_dir) / "index.csv").read_bytes() == original
    assert prepared == repeated
    assert cache_path.stat().st_mtime_ns == timestamp
    assert report["stages"]["image_cache"] == "built"
    assert second["stages"]["image_cache"] == "reused"


def test_preparation_checks_the_original_baseline_hash(base_config):
    with pytest.raises(ValueError, match="Baseline index hash mismatch"):
        prepare_inputs(config_from_dict(base_config), expected_index_sha256="0" * 64)


def test_preparation_does_not_overwrite_an_incomplete_run(base_config, tmp_path):
    cfg = config_from_dict(base_config)
    existing = tmp_path / "incomplete"
    existing.mkdir()
    cfg = replace(cfg, data=replace(cfg.data, run_dir=str(existing)))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        prepare_inputs(cfg)
    assert not (existing / "SUCCESS.json").exists()


def test_new_run_requires_raw_pipeline_inputs(base_config, tmp_path):
    cfg = config_from_dict(base_config)
    cfg = replace(cfg, data=replace(cfg.data, run_dir=str(tmp_path / "new_run")))
    with pytest.raises(ValueError, match="raw radiographs"):
        prepare_inputs(cfg)


def test_download_manifest_runs_before_dataset_preparation(base_config, tmp_path, monkeypatch):
    upstream.load_pipeline()
    downloader = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.download")
    observed = []
    manifest = tmp_path / "authorized.local.json"
    manifest.write_text(json.dumps({"destination_root": str(tmp_path), "downloads": []}))

    def download(path):
        observed.append(Path(path))

    monkeypatch.setattr(downloader, "download_manifest", download)
    _, report = prepare_inputs(config_from_dict(base_config), download_manifest=manifest)
    assert observed == [manifest]
    assert list(report["stages"])[0] == "downloads"
    assert report["stages"]["dataset"] == "reused"


def test_clinical_extraction_and_provenance_reuse(base_config, tmp_path):
    cfg = config_from_dict(base_config)
    frame = pd.read_csv(Path(cfg.data.run_dir) / "index.csv")
    row = frame.iloc[0]
    root = tmp_path / "mimic"
    (root / "icu").mkdir(parents=True)
    stamp = pd.Timestamp(row["admittime"]) + pd.Timedelta(hours=1)
    pd.DataFrame([{
        "hadm_id": row["hadm_id"], "charttime": stamp, "storetime": stamp,
        "itemid": 220045, "value": "88", "valuenum": 88,
    }]).to_csv(root / "icu" / "chartevents.csv", index=False)
    source = tmp_path / "clinical.csv"
    cfg = replace(cfg, data=replace(cfg.data, clinical_provider="table", clinical_source=str(source)))
    prepared, report = prepare_inputs(cfg, mimic_root=root)
    assert report["stages"]["clinical"] == "extracted"
    observations = pd.read_csv(source)
    assert "Heart Rate" in set(observations["variable"])
    assert (observations["value"] == 88).any()
    timestamp = source.stat().st_mtime_ns
    _, repeated = prepare_inputs(prepared)
    assert repeated["stages"]["clinical"] == "reused"
    assert source.stat().st_mtime_ns == timestamp
    sidecar = source.with_suffix(".csv.preparation.json")
    metadata = json.loads(sidecar.read_text())
    metadata["index_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different data/settings"):
        prepare_inputs(prepared)


def test_pipeline_template_matches_the_documented_old_icu_protocol():
    upstream.load_pipeline()
    loader = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.config")
    path = Path(__file__).resolve().parents[1] / "configs/preparation/icu_mortality_pipeline.json"
    cfg = loader.load_config(path)
    assert cfg["cohort"]["unit"] == "icu_stay"
    assert cfg["cohort"]["selection"] == "first_per_icu_stay"
    assert cfg["alignment"]["max_hours_after_admission"] is None
    assert cfg["alignment"]["icu_observation_hours"] == 48
    assert cfg["split"]["seed"] == 5212
    assert cfg["loader"]["channels"] == 3


def test_preparation_reproduces_missing_external_encoder(base_config, tmp_path):
    from bn5212_training.pretrain import _read_checkpoint

    project = Path(__file__).resolve().parents[1]
    generator_path = project.parent / "bn5212-data-pipeline/scripts/make_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("pretraining_prepare_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    raw = tmp_path / "study"
    pipeline_config = generator.make_synthetic(raw)
    settings = json.loads(pipeline_config.read_text())
    settings["cohort"].update(unit="icu_stay", selection="first_per_icu_stay")
    settings["alignment"].update(max_hours_after_admission=None, icu_observation_hours=48)
    pipeline_config.write_text(json.dumps(settings))

    external_root = tmp_path / "external_raw"
    (external_root / "icu").mkdir(parents=True)
    (external_root / "hosp").mkdir()
    timestamp = pd.Timestamp("2150-01-01 08:00:00")
    rows = [{
        "subject_id": str(91000000 + number), "hadm_id": str(81000000 + number),
        "stay_id": str(61000000 + number), "intime": timestamp, "admittime": timestamp,
        "dischtime": timestamp + pd.Timedelta(hours=120),
        "deathtime": timestamp + pd.Timedelta(hours=100) if number % 2 else None,
        "hospital_expire_flag": number % 2, "anchor_age": 60, "anchor_year": 2150,
    } for number in range(40)]
    frame = pd.DataFrame(rows)
    frame[["subject_id", "hadm_id", "stay_id", "intime"]].to_csv(external_root / "icu/icustays.csv", index=False)
    frame[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]].to_csv(
        external_root / "hosp/admissions.csv", index=False)
    frame[["subject_id", "anchor_age", "anchor_year"]].to_csv(external_root / "hosp/patients.csv", index=False)
    events = [{"subject_id": row["subject_id"], "hadm_id": row["hadm_id"],
               "charttime": timestamp + pd.Timedelta(hours=1), "storetime": timestamp + pd.Timedelta(hours=1),
               "itemid": 220045, "value": str(70 + number), "valuenum": 70 + number}
              for number, row in enumerate(rows)]
    pd.DataFrame(events).to_csv(external_root / "icu/chartevents.csv", index=False)
    cfg = config_from_dict(base_config)
    checkpoint = tmp_path / "pretrained/clinical.pt"
    cfg = replace(cfg, data=replace(cfg.data, run_dir=str(raw / "processed")),
                  clinical_encoder=replace(cfg.clinical_encoder, pretrained=str(checkpoint), freeze=True))
    _, report = prepare_inputs(cfg, pipeline_config=pipeline_config, mimic_root=external_root,
                               pretrain=True, pretrain_epochs=1)
    assert report["stages"]["clinical_pretraining"] == "trained"
    assert _read_checkpoint(checkpoint)["encoder"]["embed_dim"] == 32
    study = pd.read_csv(raw / "processed/index.csv", dtype=str)
    external = pd.read_csv(checkpoint.parent / "external_cohort/index.csv", dtype=str)
    assert not set(study["subject_id"]) & set(external["subject_id"])
