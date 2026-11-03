"""CXR-anchored transfer features and refusal of the old mortality dataset."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from bn5212_training import upstream
from bn5212_training.clinical import build_provider
from bn5212_training.config import config_from_dict, load_config
from bn5212_training.prepare import prepare_inputs
from bn5212_training.transfer import LAB_LABELS, VARIABLES, extract_transfer_features


def test_transfer_labs_use_precxr_window_and_both_clocks_even_without_hadm(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    admission = pd.Timestamp("2145-01-01 12:00:00")
    cxr = admission + pd.Timedelta(hours=6)
    pd.DataFrame([{
        "subject_id": "90000001", "hadm_id": "80000001", "admittime": admission,
        "study_time": cxr, "hours_since_admission": 6,
        "age_at_admission": 70, "gender": "F",
    }]).to_csv(run / "index.csv", index=False)
    root = tmp_path / "mimic"
    (root / "hosp").mkdir(parents=True)
    pd.DataFrame([{"itemid": 100 + index, "label": aliases[0], "fluid": "Blood"}
                  for index, aliases in enumerate(LAB_LABELS.values())]).to_csv(root / "hosp/d_labitems.csv", index=False)
    events = []
    for delta, stored_delta, value in ((-47, -46, 10), (-1, -0.5, 20), (0, 0, 999),
                                      (-1, 1, 999), (-49, -48, 999), (-1, None, 999)):
        events.append({"subject_id": "90000001", "hadm_id": None, "itemid": 100,
                       "charttime": cxr + pd.Timedelta(hours=delta),
                       "storetime": cxr + pd.Timedelta(hours=stored_delta) if stored_delta is not None else None,
                       "value": str(value), "valuenum": value})
    pd.DataFrame(events).to_csv(root / "hosp/labevents.csv", index=False)
    output = tmp_path / "features.csv"
    extract_transfer_features(run, root, output)
    observations = pd.read_csv(output)
    labs = observations[observations["variable"].eq("White Blood Cells")]
    assert labs["value"].tolist() == [10, 20]
    assert labs["hour"].tolist() == [1, 47]
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/icu_transfer/cross_attention_adapter.json")
    provider = build_provider(replace(cfg.data, clinical_source=str(output)))
    values, mask = provider.features({"hadm_id": "80000001", "hours_since_admission": 6})
    assert values.shape == (18, 48)
    assert provider.variable_names == tuple(VARIABLES)
    assert values[0, 47] == 20 and mask[0, 47]
    assert values[VARIABLES.index("Age"), 47] == 70
    assert values[VARIABLES.index("Female"), 47] == 1


def test_transfer_preparation_refuses_mortality_data_and_old_weights(base_config):
    cfg = config_from_dict(base_config)
    cfg = replace(cfg, data=replace(cfg.data, task="icu_transfer"))
    with pytest.raises(ValueError, match="Expected dataset task"):
        prepare_inputs(cfg)


def test_transfer_configs_and_task_preset_share_consistent_paths():
    import importlib

    project = Path(__file__).resolve().parents[1]
    cfg = load_config(project / "configs/icu_transfer/cross_attention_adapter.json")
    upstream.load_pipeline()
    settings = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.config").load_config(
        project / "configs/preparation/icu_transfer_pipeline.json")
    assert Path(settings["paths"]["output_dir"]) == (project / cfg.data.run_dir).resolve()
    assert settings["label"]["kind"] == cfg.data.task == "icu_transfer"
    assert cfg.data.clinical_variables == VARIABLES
    assert cfg.data.clinical_window_anchor == "cxr"
    assert cfg.clinical_encoder.pretrained is None and not cfg.clinical_encoder.freeze


@pytest.mark.parametrize("staged", [False, True], ids=["multimodal_only", "frozen_from_unimodal"])
def test_icu_transfer_prepare_train_and_benchmark(base_config, tmp_path, staged):
    project = Path(__file__).resolve().parents[1]
    generator_path = project.parent / "bn5212-data-pipeline/scripts/make_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("transfer_e2e_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    raw = tmp_path / "raw"
    pipeline = generator.make_synthetic(raw)
    settings = json.loads(pipeline.read_text())
    settings["label"] = {"kind": "icu_transfer", "column": "icu_transfer",
                         "positive_values": ["1"], "negative_values": ["0"]}
    pipeline.write_text(json.dumps(settings))
    stays_path = raw / "raw/mimiciv/icu/icustays.csv.gz"
    stays = pd.read_csv(stays_path, dtype=str)
    stays = stays[stays["subject_id"].astype(int).mod(3).ne(0)].copy()
    stays["intime"] = (pd.to_datetime(stays["intime"]) + pd.Timedelta(hours=18)).astype(str)
    stays.to_csv(stays_path, index=False)
    hosp = raw / "raw/mimiciv/hosp"
    pd.DataFrame([{"itemid": 100 + number, "label": aliases[0], "fluid": "Blood"}
                  for number, aliases in enumerate(LAB_LABELS.values())]).to_csv(hosp / "d_labitems.csv", index=False)
    admissions = pd.read_csv(hosp / "admissions.csv.gz").drop_duplicates("hadm_id")
    labs = [{"subject_id": row.subject_id, "hadm_id": None,
             "charttime": pd.Timestamp(row.admittime) + pd.Timedelta(hours=5),
             "storetime": pd.Timestamp(row.admittime) + pd.Timedelta(hours=5, minutes=5),
             "itemid": 100, "value": "10", "valuenum": 10} for row in admissions.itertuples()]
    pd.DataFrame(labs).to_csv(hosp / "labevents.csv", index=False)
    payload = load_config(project / "configs/icu_transfer/cross_attention_adapter.json").to_dict()
    payload["data"].update(run_dir=str(raw / "processed"), clinical_source=str(tmp_path / "clinical.csv"),
                           image_cache=str(tmp_path / "images.json"), batch_size=4)
    payload["image_encoder"] = base_config["image_encoder"]
    payload["clinical_encoder"]["embed_dim"] = 32
    payload["fusion"].update(embed_dim=32, depth=1, num_heads=4)
    payload["optim"].update(epochs=1, warmup_epochs=0, amp=False)
    config = tmp_path / "training.json"
    config.write_text(json.dumps(payload))
    command = ["bash", str(project / "scripts/run_icu_transfer.sh"),
               "--base-config", str(config), "--pipeline-config", str(pipeline),
               "--output-dir", str(tmp_path / "outputs"), "--run-id", "transfer_e2e",
               "--device", "cpu", "--bootstrap-samples", "5"]
    if staged:
        command += ["--include-unimodal", "--freeze-from-unimodal"]
    env = dict(os.environ, PYTHON_BIN=sys.executable, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    batch = tmp_path / "outputs/transfer_e2e"
    names = ("clinical_only", "cxr_only", "concat", "joint", "cross_attention_adapter") if staged else ("concat", "joint", "cross_attention_adapter")
    for name in names:
        assert (batch / f"training/{name}/fit/checkpoint_best.pt").is_file()
        metrics = json.loads((batch / f"benchmark/{name}/metrics.json").read_text())
        assert metrics["dataset"]["task"] == "icu_transfer"
        assert metrics["threshold_selection"]["source_split"] == "val"
    assert (batch / "benchmark_leaderboard.csv").is_file()

    configs = [json.loads((batch / f"configs/{name}.json").read_text())
               for name in names]
    for section in ("data", "image_encoder", "clinical_encoder", "head", "optim", "seed"):
        assert all(config[section] == configs[0][section] for config in configs)

    if not staged:
        assert set(pd.read_csv(batch / "benchmark_leaderboard.csv")["model"]) == set(names)
        assert not (batch / "training/clinical_only").exists()
        assert not (batch / "training/cxr_only").exists()
        assert not (batch / "paired_auroc_gains.csv").exists()
        assert configs[1]["fusion"]["name"] == "joint_self_attention"
        assert configs[0]["fusion"]["concat_project"] is False
        for name, config in zip(names, configs):
            assert not config["freeze_encoders"]
            assert config["clinical_checkpoint"] is None and config["image_checkpoint"] is None
            manifest = json.loads((batch / f"training/{name}/fit/run_manifest.json").read_text())
            assert manifest["trainable_parameters"]["clinical_encoder"] > 0
            assert manifest["trainable_parameters"]["image_encoder"] > 0
        resumed = subprocess.run(command + ["--resume"], env=env, capture_output=True, text=True, timeout=90)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert resumed.stdout.count("Training already complete") == 3
        return

    gains = pd.read_csv(batch / "paired_auroc_gains.csv")
    assert len(gains) == 6
    assert set(gains["baseline"]) == {"clinical_only", "cxr_only"}
    import torch
    from bn5212_training.manifest import sha256
    sources = {
        "clinical": torch.load(batch / "training/clinical_only/fit/checkpoint_best.pt", weights_only=False),
        "image": torch.load(batch / "training/cxr_only/fit/checkpoint_best.pt", weights_only=False),
    }
    assert configs[1]["fusion"]["pooling"] == "patch_mean"
    for name in ("concat", "joint", "cross_attention_adapter"):
        checkpoint = torch.load(batch / f"training/{name}/fit/checkpoint_best.pt", weights_only=False)
        assert checkpoint["config"]["freeze_encoders"]
        assert checkpoint["clinical_normalizer"] == sources["clinical"]["clinical_normalizer"]
        for branch, source in sources.items():
            prefix = f"{branch}_encoder."
            for key, value in source["state_dict"].items():
                if key.startswith(prefix):
                    torch.testing.assert_close(checkpoint["state_dict"][key], value, rtol=0, atol=0)
        manifest = json.loads((batch / f"training/{name}/fit/run_manifest.json").read_text())
        assert manifest["trainable_parameters"]["clinical_encoder"] == 0
        assert manifest["trainable_parameters"]["image_encoder"] == 0
        for branch in sources:
            record = manifest["encoder_sources"][branch]
            assert record["sha256"] == sha256(record["path"])
    assert configs[2]["fusion"]["concat_project"] is False
    resumed = subprocess.run(command + ["--resume"], env=env, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert resumed.stdout.count("Training already complete") == 5
