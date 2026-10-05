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


# --- DataLoader workers ---------------------------------------------------
# On Windows a worker starts by spawn and unpickles the dataset from scratch, so
# it must be able to *import* the pipeline package rather than inherit it from a
# sys.modules entry the parent made. These cover both modality paths.


@pytest.mark.parametrize(
    "modalities,fusion,load_image",
    [(["clinical"], "clinical_only", False), (["cxr"], "image_only", True)],
)
def test_dataloader_workers_can_unpickle_the_dataset(
    base_config, modalities, fusion, load_image
):
    from torch.utils.data import DataLoader

    from bn5212_training.clinical import build_provider
    from bn5212_training.data import TrainingDataset, seed_worker

    cfg = _configure(
        base_config, experiment="workers", modalities=modalities, fusion=fusion
    )
    provider = build_provider(cfg.data) if cfg.uses_clinical else None
    dataset = TrainingDataset(
        cfg.data.run_dir, "val", provider=provider, load_image=load_image
    )
    loader = DataLoader(
        dataset, batch_size=2, num_workers=2, shuffle=False, worker_init_fn=seed_worker
    )
    batches = [batch for batch in loader]
    assert batches, "worker processes produced no batches"
    assert sum(len(b["sample_id"]) for b in batches) == len(dataset)


def test_the_pipeline_package_is_importable_not_just_preloaded():
    """A spawned worker imports it; a sys.modules entry alone would not survive."""
    import importlib
    import sys

    from bn5212_training import upstream

    sys.modules.pop(upstream.PACKAGE_ALIAS, None)
    for name in [n for n in sys.modules if n.startswith(upstream.PACKAGE_ALIAS + ".")]:
        sys.modules.pop(name, None)
    module = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.dataset")
    assert hasattr(module, "MimicCXRDataset")


# --- checkpoint selection and warmup --------------------------------------


def test_warmup_epochs_cannot_win_the_checkpoint(base_config, tmp_path):
    """A noisy validation split can hand its best score to a barely-trained epoch.

    During warmup the learning rate is still ramping, so those epochs are not
    comparable with the rest; selecting one freezes an undertrained model.
    """
    payload = json.loads(json.dumps(base_config))
    payload["experiment"] = "warmup"
    payload["modalities"] = ["clinical"]
    payload["fusion"] = {"name": "clinical_only", "embed_dim": 32, "pooling": "mean"}
    payload["optim"] = {
        "epochs": 8,
        "warmup_epochs": 4,
        "early_stopping_patience": 20,
    }
    payload["selection_metric"] = "loss"
    cfg = config_from_dict(payload)

    result = train(cfg, output_dir=tmp_path, run_id="unit")
    selected = result["validation_metrics"]["selected_epoch"]
    # Epoch numbers are 1-based, so warmup covers epochs 1..4.
    assert selected > cfg.optim.warmup_epochs, f"selected epoch {selected} is inside warmup"


def test_selection_threshold_can_be_set_explicitly(base_config, tmp_path):
    payload = json.loads(json.dumps(base_config))
    payload["experiment"] = "warmup_explicit"
    payload["modalities"] = ["clinical"]
    payload["fusion"] = {"name": "clinical_only", "embed_dim": 32, "pooling": "mean"}
    payload["optim"] = {
        "epochs": 10,
        "warmup_epochs": 0,
        "min_epochs_before_selection": 6,
        "early_stopping_patience": 20,
    }
    payload["selection_metric"] = "loss"
    cfg = config_from_dict(payload)

    result = train(cfg, output_dir=tmp_path, run_id="unit")
    assert result["validation_metrics"]["selected_epoch"] > 6


def test_checkpoint_round_trip_restores_encoder_buffers(base_config, tmp_path):
    """An encoder can hold the normalisation as a buffer sized by the channels.

    Rebuilding from the checkpoint therefore has to know those statistics, or
    load_state_dict fails on a shape mismatch the moment the run is not
    single-channel.
    """
    from bn5212_training.model import load_checkpoint, save_checkpoint

    cfg = _configure(
        base_config, experiment="buffers", modalities=["cxr"], fusion="image_only"
    )
    model = build_model(
        cfg, image_size=32, channels=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    geometry = {
        "image_size": 32, "channels": 1, "num_variables": 0, "num_timesteps": 0,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, epoch=1, geometry=geometry)

    restored, payload = load_checkpoint(path)
    assert payload["geometry"]["mean"] == [0.485, 0.456, 0.406]
    for (name, original), (_, copy) in zip(
        model.state_dict().items(), restored.state_dict().items()
    ):
        assert original.shape == copy.shape, name


def test_checkpoints_without_recorded_statistics_still_load(base_config, tmp_path):
    """Checkpoints written before the statistics were recorded stay loadable."""
    import torch

    from bn5212_training.model import load_checkpoint, save_checkpoint

    cfg = _configure(
        base_config, experiment="legacy", modalities=["cxr"], fusion="image_only"
    )
    model = build_model(cfg, image_size=32, channels=1)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, epoch=1, geometry={"image_size": 32, "channels": 1})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["geometry"].pop("mean", None)
    payload["geometry"].pop("std", None)
    torch.save(payload, path)

    restored, _ = load_checkpoint(path)
    assert restored is not None
