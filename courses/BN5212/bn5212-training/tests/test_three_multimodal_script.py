"""The single Bash entry point saves three trained models and benchmark reports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize("exists", [False, True], ids=["missing_directory", "incomplete_directory"])
def test_missing_dataset_reports_the_resolved_path_without_starting_training(tmp_path, exists):
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "run_three_multimodal.sh"
    dataset = tmp_path / "unpublished data"
    if exists:
        dataset.mkdir()
    output = tmp_path / "should_not_exist"
    environment = dict(os.environ, PYTHON_BIN=sys.executable)
    result = subprocess.run(
        ["bash", str(script), "--run-dir", str(dataset), "--output-dir", str(output),
         "--run-id", "missing_data"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert str(dataset) in result.stderr
    assert "--run-dir" in result.stderr
    assert "SUCCESS.json" in result.stderr
    assert "do not create SUCCESS.json manually" in result.stderr
    assert "training + validation" not in result.stdout
    assert not output.exists()


def test_three_models_train_evaluate_and_resume(base_config, tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "run_three_multimodal.sh"
    config = tmp_path / "shared config.json"
    config.write_text(json.dumps(base_config), encoding="utf-8")
    batch = tmp_path / "outputs with spaces" / "three_models"
    arguments = [
        "bash", str(script), "--base-config", str(config),
        "--output-dir", str(batch.parent), "--run-id", batch.name,
        "--device", "cpu", "--set", "optim.epochs=1", "--bootstrap-samples", "5",
    ]
    environment = dict(os.environ, PYTHON_BIN=sys.executable, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(arguments, cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    models = ("concat", "joint", "cross_attention_adapter")
    positions = []
    timestamps = {}
    configs = []
    for model in models:
        positions += [result.stdout.index(f"=== {model}: training + validation ==="),
                      result.stdout.index(f"=== {model}: benchmark test ===")]
        train_root = batch / "training" / model / "fit"
        benchmark_root = batch / "benchmark" / model
        checkpoint = train_root / "checkpoint_best.pt"
        timestamps[str(checkpoint)] = checkpoint.stat().st_mtime_ns
        assert (train_root / "TRAIN_SUCCESS").is_file()
        configs.append(json.loads((train_root / "config.json").read_text()))
        for filename in ("metrics.json", "metrics.csv", "predictions.csv", "report.html", "SUCCESS.json"):
            assert (benchmark_root / filename).is_file()
        metrics = json.loads((benchmark_root / "metrics.json").read_text())
        assert metrics["threshold_selection"]["source_split"] == "val"
        assert metrics["model"]["name"] == model
        assert metrics["splits"]["test"]["n"] > 0
        assert "auroc" in metrics["test_patient_bootstrap"]
        manifest = json.loads((benchmark_root / "run_manifest.json").read_text())
        assert manifest["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        timestamps[str(benchmark_root / "metrics.json")] = (benchmark_root / "metrics.json").stat().st_mtime_ns
        for stage in ("train", "benchmark"):
            assert (batch / "logs" / f"{model}.{stage}.log").stat().st_size > 0
    assert positions == sorted(positions)
    for section in ("data", "image_encoder", "clinical_encoder", "head", "optim", "seed"):
        assert all(config[section] == configs[0][section] for config in configs)
    leaderboard = pd.read_csv(batch / "benchmark_leaderboard.csv")
    assert set(leaderboard["model"]) == set(models)
    assert (batch / "benchmark_leaderboard.html").is_file()
    assert (batch / "validation_comparison.csv").is_file()

    resumed = subprocess.run(arguments + ["--resume"], cwd=tmp_path, env=environment,
                             capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert resumed.stdout.count("Training already complete") == 3
    assert resumed.stdout.count("Benchmark already complete") == 3
    assert all(Path(path).stat().st_mtime_ns == timestamp for path, timestamp in timestamps.items())
    duplicate = subprocess.run(arguments, cwd=tmp_path, env=environment,
                               capture_output=True, text=True, timeout=30)
    assert duplicate.returncode != 0
    changed = subprocess.run(arguments + ["--resume", "--set", "optim.lr=0.02"],
                             cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0
    assert "Resume requires" in changed.stderr


def test_preparation_training_and_benchmark_run_in_one_command(base_config, tmp_path):
    project = Path(__file__).resolve().parents[1]
    generator_path = project.parent / "bn5212-data-pipeline/scripts/make_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("prepare_e2e_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    raw = tmp_path / "raw_fixture"
    pipeline_config = generator.make_synthetic(raw)
    payload = json.loads(json.dumps(base_config))
    payload["data"]["run_dir"] = str(raw / "processed")
    config = tmp_path / "training.json"
    config.write_text(json.dumps(payload))
    script = project / "scripts/run_three_multimodal.sh"
    batch = tmp_path / "outputs" / "prepare_e2e"
    args = [
        "bash", str(script), "--base-config", str(config), "--pipeline-config", str(pipeline_config),
        "--output-dir", str(batch.parent), "--run-id", batch.name,
        "--device", "cpu", "--set", "optim.epochs=1", "--bootstrap-samples", "5",
    ]
    env = dict(os.environ, PYTHON_BIN=sys.executable, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    outcome = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=150)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    assert (raw / "processed/SUCCESS.json").is_file()
    report_path = project / "data/prepared_configs/prepare_e2e.preparation.json"
    report = json.loads(report_path.read_text())
    assert report["stages"] == {"dataset": "built", "image_cache": "built"}
    index_path = raw / "processed/index.csv"
    timestamp = index_path.stat().st_mtime_ns
    for model in ("concat", "joint", "cross_attention_adapter"):
        assert (batch / "training" / model / "fit/checkpoint_best.pt").is_file()
        assert (batch / "benchmark" / model / "SUCCESS.json").is_file()
    resumed = subprocess.run(args + ["--resume"], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert index_path.stat().st_mtime_ns == timestamp
    repeated = json.loads(report_path.read_text())
    assert repeated["stages"] == {"dataset": "reused", "image_cache": "reused"}
