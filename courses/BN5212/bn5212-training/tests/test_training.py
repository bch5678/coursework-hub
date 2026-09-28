"""End-to-end runs on the synthetic fixture.

The point of these tests is the claim the framework is built on: all four
experiments in the project plan are the same loop with a different fusion name,
and the files they emit are exactly what benchmark-evaluation accepts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from bn5212_training.config import config_from_dict
from bn5212_training.data import TrainingDataset, make_loader
from bn5212_training.model import build_model, load_checkpoint
from bn5212_training.trainer import train


def _configure(base: dict, *, experiment: str, modalities: list[str], fusion: str, **fusion_extra):
    payload = json.loads(json.dumps(base))
    payload["experiment"] = experiment
    payload["modalities"] = modalities
    payload["fusion"] = {"name": fusion, "embed_dim": 32, "depth": 1, "num_heads": 2, **fusion_extra}
    if fusion == "clinical_only":
        payload["fusion"]["pooling"] = "mean"
    return config_from_dict(payload)


EXPERIMENTS = [
    ("clinical_only", ["clinical"], "clinical_only"),
    ("cxr_only", ["cxr"], "image_only"),
    ("concat_fusion", ["clinical", "cxr"], "concat_mlp"),
    ("metra_joint", ["clinical", "cxr"], "joint_self_attention"),
    ("cross_attention", ["clinical", "cxr"], "cross_attention"),
]


@pytest.mark.parametrize("experiment,modalities,fusion", EXPERIMENTS)
def test_every_experiment_runs_and_emits_the_same_artefacts(
    base_config, tmp_path, experiment, modalities, fusion
):
    cfg = _configure(base_config, experiment=experiment, modalities=modalities, fusion=fusion)
    result = train(cfg, output_dir=tmp_path, run_id="unit")
    root = Path(result["run_dir"])

    for name in (
        "config.json",
        "run_manifest.json",
        "metrics_val.json",
        "checkpoint_best.pt",
        "predictions_val.csv",
        "predictions_test.csv",
        "summary.csv",
    ):
        assert (root / name).is_file(), f"{experiment} did not write {name}"
    assert (root / "figures" / "training_curves.png").is_file()
    # Every figure ships a table view alongside it.
    assert (root / "figures" / "training_curves.csv").is_file()


def test_predictions_match_the_frozen_split_exactly(base_config, tmp_path):
    cfg = _configure(
        base_config, experiment="cxr_only", modalities=["cxr"], fusion="image_only"
    )
    result = train(cfg, output_dir=tmp_path, run_id="unit")
    index = pd.read_csv(Path(cfg.data.run_dir) / "index.csv", dtype={"sample_id": str})

    for split in ("val", "test"):
        predictions = pd.read_csv(
            Path(result["run_dir"]) / f"predictions_{split}.csv", dtype={"sample_id": str}
        )
        assert list(predictions.columns) == ["sample_id", "y_score"]
        expected = set(index.loc[index["split"].eq(split), "sample_id"])
        assert set(predictions["sample_id"]) == expected
        assert not predictions["sample_id"].duplicated().any()
        assert predictions["y_score"].between(0.0, 1.0).all()
        assert predictions["y_score"].notna().all()


def test_the_benchmark_project_accepts_our_prediction_files(base_config, tmp_path):
    """The real integration check: the downstream contract, enforced upstream."""
    benchmark_io = pytest.importorskip(
        "bn5212_benchmark.io", reason="benchmark-evaluation is not installed in this environment"
    )
    cfg = _configure(
        base_config, experiment="cxr_only", modalities=["cxr"], fusion="image_only"
    )
    result = train(cfg, output_dir=tmp_path, run_id="unit")
    index, _spec = benchmark_io.load_dataset_index(cfg.data.run_dir)
    for split in ("val", "test"):
        frame = benchmark_io.load_predictions(
            Path(result["run_dir"]) / f"predictions_{split}.csv", index, split
        )
        assert len(frame) == int(index["split"].eq(split).sum())


def test_validation_metrics_are_recorded_but_test_metrics_are_not(base_config, tmp_path):
    """Test predictions are exported; test labels are never scored here."""
    cfg = _configure(
        base_config, experiment="cxr_only", modalities=["cxr"], fusion="image_only"
    )
    result = train(cfg, output_dir=tmp_path, run_id="unit")
    metrics = json.loads((Path(result["run_dir"]) / "metrics_val.json").read_text(encoding="utf-8"))
    assert "auroc" in metrics
    assert not any("test" in key for key in metrics)

    manifest = json.loads(
        (Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert "validation_metrics" in manifest
    assert "test_metrics" not in manifest


def test_manifest_records_the_provenance_the_benchmark_requires(base_config, tmp_path):
    cfg = _configure(
        base_config, experiment="metra_joint", modalities=["clinical", "cxr"], fusion="joint_self_attention"
    )
    result = train(cfg, output_dir=tmp_path, run_id="unit")
    manifest = json.loads(
        (Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seed"] == cfg.seed
    assert manifest["config_fingerprint"]
    assert manifest["checkpoint_sha256"]
    assert manifest["dataset"]["index_sha256"]
    # The recomputed hash proves the index was not swapped mid-project.
    assert manifest["dataset"]["index_sha256"] == manifest["dataset"]["index_sha256_recomputed"]
    assert "torch" in manifest["environment"]


def test_checkpoint_round_trip_reproduces_the_same_scores(base_config, tmp_path):
    cfg = _configure(
        base_config, experiment="cxr_only", modalities=["cxr"], fusion="image_only"
    )
    result = train(cfg, output_dir=tmp_path, run_id="unit")
    checkpoint = Path(result["run_dir"]) / "checkpoint_best.pt"

    model, payload = load_checkpoint(checkpoint)
    assert payload["config"]["experiment"] == "cxr_only"
    dataset = TrainingDataset(cfg.data.run_dir, "test", load_image=True)
    loader = make_loader(dataset, batch_size=4, shuffle=False, num_workers=0)

    from bn5212_training.predict import positive_probability

    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in loader:
            scores.extend(positive_probability(model(batch)).tolist())

    exported = pd.read_csv(
        Path(result["run_dir"]) / "predictions_test.csv", dtype={"sample_id": str}
    )
    # The loader is unshuffled for test, so order matches the exported file.
    assert exported["y_score"].to_numpy() == pytest.approx(scores, abs=1e-5)


def test_clinical_only_never_decodes_an_image(base_config):
    cfg = _configure(
        base_config, experiment="clinical_only", modalities=["clinical"], fusion="clinical_only"
    )
    from bn5212_training.clinical import build_provider

    provider = build_provider(cfg.data)
    dataset = TrainingDataset(
        cfg.data.run_dir, "train", provider=provider, load_image=cfg.uses_image
    )
    sample = dataset[0]
    assert "image" not in sample
    assert sample["clinical"].shape == (len(provider.variable_names), cfg.data.clinical_timesteps)


def test_a_fusion_that_needs_a_disabled_modality_fails_fast(base_config):
    cfg = _configure(
        base_config,
        experiment="broken",
        modalities=["clinical"],
        fusion="joint_self_attention",
    )
    with pytest.raises(ValueError, match="needs the cxr modality"):
        build_model(cfg, image_size=32, channels=1, num_variables=3, num_timesteps=8)


def test_the_clinical_branch_needs_a_provider(base_config):
    cfg = _configure(
        base_config, experiment="clinical_only", modalities=["clinical"], fusion="clinical_only"
    )
    with pytest.raises(ValueError, match="num_variables"):
        build_model(cfg, image_size=32, channels=1, num_variables=0, num_timesteps=0)


def test_the_same_seed_reproduces_the_same_predictions(base_config, tmp_path):
    cfg = _configure(
        base_config, experiment="cxr_only", modalities=["cxr"], fusion="image_only"
    )
    first = train(cfg, output_dir=tmp_path / "a", run_id="unit")
    second = train(cfg, output_dir=tmp_path / "b", run_id="unit")
    left = pd.read_csv(Path(first["run_dir"]) / "predictions_test.csv")
    right = pd.read_csv(Path(second["run_dir"]) / "predictions_test.csv")
    assert left["y_score"].to_numpy() == pytest.approx(right["y_score"].to_numpy(), abs=1e-6)
