"""Train and replay adapter predictions through the existing data interfaces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bn5212_training import upstream
from bn5212_training.cache import build_image_cache
from bn5212_training.config import config_from_dict, load_config
from bn5212_training.predict import build_inference_loader, predict_from_checkpoint
from bn5212_training.trainer import train


def _clinical_table(run_dir, destination):
    index = pd.read_csv(Path(run_dir) / "index.csv", dtype={"hadm_id": str})
    generator = np.random.default_rng(5212)
    observations = []
    for admission in index["hadm_id"].unique():
        for variable in ("age", "lab_a", "lab_b"):
            for hour in range(8):
                if hour == 0 or generator.random() > 0.3:
                    observations.append({
                        "hadm_id": admission, "hour": hour,
                        "variable": variable, "value": float(generator.normal()),
                    })
    pd.DataFrame(observations).to_csv(destination, index=False)


def test_v3_models_share_the_saved_experiment_settings():
    configs = Path(__file__).resolve().parents[1] / "configs/icu_transfer"
    models = [load_config(configs / name) for name in
              ("concat.json", "joint_attention.json", "cross_attention_adapter.json")]
    reference = models[0]
    for model in models:
        for section in ("data", "image_encoder", "clinical_encoder", "augmentation",
                        "head", "optim", "seed", "selection_metric"):
            assert getattr(model, section) == getattr(reference, section)
        assert model.optim.lr == 0.0003
        assert model.optim.min_epochs_before_selection == 0
        assert model.data.task == "icu_transfer"
        assert len(model.data.clinical_variables) == 18
        assert model.fusion.concat_image_pooling == "patch_mean"
        assert model.fusion.concat_project is False


@pytest.mark.parametrize(
    "use_cache,workers,device",
    [(False, 0, "cpu"), (True, 0, "cpu"), (True, 2, "cpu"), (True, 0, "cuda")],
    ids=["decoded_cpu", "cached_cpu", "cached_workers", "cached_cuda_amp"],
)
def test_adapter_training_and_standalone_prediction_reuse_data(
    base_config, tmp_path, monkeypatch, use_cache, workers, device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    payload = json.loads(json.dumps(base_config))
    clinical_source = tmp_path / "clinical_features.csv"
    _clinical_table(payload["data"]["run_dir"], clinical_source)
    payload["data"].update(
        clinical_provider="table", clinical_source=str(clinical_source), num_workers=workers,
    )
    payload["experiment"] = "adapter_data_integration"
    payload["fusion"]["name"] = "cross_attention_adapter"
    payload["device"] = device
    payload["optim"].update(epochs=2, amp=device == "cuda")
    if use_cache:
        cache_path = tmp_path / "images.json"
        build_image_cache(payload["data"]["run_dir"], cache_path, progress_every=0)
        payload["data"]["image_cache"] = str(cache_path)

        def refuse_decode(*args, **kwargs):
            raise AssertionError("Cached training/prediction must not decode source images")

        monkeypatch.setattr(upstream.dataset_module(), "image_array", refuse_decode)

    cfg = config_from_dict(payload)
    result = train(cfg, output_dir=tmp_path / "outputs", run_id="smoke")
    assert len(result["history"]) == 2
    assert all(np.isfinite(record[key]) for record in result["history"]
               for key in ("train_loss", "val_loss"))
    root = Path(result["run_dir"])
    checkpoint = root / "checkpoint_best.pt"
    assert checkpoint.is_file()
    dataset = build_inference_loader(cfg, "test")
    assert (dataset.image_cache is not None) == use_cache
    standalone_path = predict_from_checkpoint(
        checkpoint, split="test", output_path=root / "predictions_test_reloaded.csv",
        device=device,
    )
    expected = pd.read_csv(root / "predictions_test.csv")
    actual = pd.read_csv(standalone_path)
    pd.testing.assert_series_equal(actual["sample_id"], expected["sample_id"])
    np.testing.assert_allclose(actual["y_score"], expected["y_score"], atol=1e-6, rtol=1e-6)
    assert np.isfinite(actual["y_score"]).all()
    assert actual["y_score"].between(0.0, 1.0).all()
    index = pd.read_csv(Path(cfg.data.run_dir) / "index.csv")
    assert set(actual["sample_id"]) == set(index.loc[index["split"].eq("test"), "sample_id"])
    patient_groups = [set(index.loc[index["split"].eq(split), "subject_id"])
                      for split in ("train", "val", "test")]
    for position, first in enumerate(patient_groups):
        assert all(not first & second for second in patient_groups[position + 1:])
    for figure in ("training_curves.png", "attention_matrix.png", "attention_patch_maps.png"):
        assert (root / "figures" / figure).is_file()
    report = {
        "data": "synthetic fixture; not a clinical performance experiment",
        "device": device,
        "amp": cfg.optim.amp,
        "uses_image_cache": use_cache,
        "num_workers": workers,
        "clinical_provider": cfg.data.clinical_provider,
        "epochs": len(result["history"]),
        "split_samples": index["split"].value_counts().to_dict(),
        "reload_max_abs_error": float(np.max(np.abs(actual["y_score"] - expected["y_score"]))),
        "history": result["history"],
    }
    (root / "smoke_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
