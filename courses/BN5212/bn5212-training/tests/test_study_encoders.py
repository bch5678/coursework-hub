from dataclasses import replace

import numpy as np
import pytest
import torch

from bn5212_training.clinical import ClinicalNormalizer
from bn5212_training.config import config_from_dict
from bn5212_training.model import build_model, load_checkpoint, save_checkpoint
from bn5212_training.study_encoders import read_study_checkpoint
from bn5212_training.trainer import _prepare_clinical, train


@pytest.fixture
def staged(tmp_path):
    (tmp_path / "index.csv").write_text("immutable fixture index\n")
    cfg = config_from_dict({
        "experiment": "stage", "modalities": ["clinical", "cxr"],
        "data": {"run_dir": str(tmp_path), "clinical_timesteps": 8},
        "image_encoder": {"embed_dim": 16, "depth": 1, "num_heads": 4, "patch_size": 16},
        "clinical_encoder": {"embed_dim": 16},
        "fusion": {"name": "cross_attention_adapter", "embed_dim": 16, "num_heads": 4, "depth": 1},
    })
    geometry = dict(image_size=32, channels=1, num_variables=3, num_timesteps=8)
    normalizer = ClinicalNormalizer(np.array([2., 4., 6.]), np.array([1., 2., 3.]), ("a", "b", "c"))
    paths, states = {}, {}
    for branch, modality, fusion in (("clinical", "clinical", "clinical_only"), ("image", "cxr", "image_only")):
        source = replace(cfg, modalities=(modality,), fusion=replace(cfg.fusion, name=fusion))
        model = build_model(source, **geometry)
        with torch.no_grad():
            for p in getattr(model, f"{branch}_encoder").parameters():
                p.add_(.123)
        states[branch] = {k: v.clone() for k, v in getattr(model, f"{branch}_encoder").state_dict().items()}
        paths[branch] = save_checkpoint(tmp_path / f"{branch}.pt", model, epoch=1,
                                       geometry=geometry, normalizer=normalizer if branch == "clinical" else None)
    cfg = replace(cfg, clinical_checkpoint=str(paths["clinical"]), image_checkpoint=str(paths["image"]),
                  freeze_encoders=True)
    return cfg, geometry, states, normalizer


def test_fusion_reuses_exact_encoders_and_only_fusion_head_receive_gradients(staged):
    cfg, geometry, states, _ = staged
    model = build_model(cfg, **geometry).train()
    for branch in ("image", "clinical"):
        encoder = getattr(model, f"{branch}_encoder")
        assert not encoder.training
        assert all(not p.requires_grad for p in encoder.parameters())
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, states[branch][key])
    model({"image": torch.randn(2, 1, 32, 32), "clinical": torch.randn(2, 3, 8),
           "clinical_mask": torch.ones(2, 3, 8, dtype=torch.bool)}).square().sum().backward()
    for encoder in (model.image_encoder, model.clinical_encoder):
        assert all(p.grad is None for p in encoder.parameters())
    assert any(p.grad is not None for p in model.fusion.parameters())
    assert any(p.grad is not None for p in model.head.parameters())


def test_source_normalizer_is_restored_without_refitting(staged):
    cfg, _, _, normalizer = staged
    _, actual = _prepare_clinical(cfg)
    assert actual.to_dict() == normalizer.to_dict()


@pytest.mark.parametrize("change", ["index", "seed", "architecture", "scope", "modality", "task"])
def test_incompatible_sources_are_rejected(staged, change):
    cfg, _, _, _ = staged
    if change == "index":
        from pathlib import Path
        (Path(cfg.data.run_dir) / "index.csv").write_text("changed index")
    elif change == "seed":
        cfg = replace(cfg, seed=cfg.seed + 1)
    elif change == "architecture":
        cfg = replace(cfg, clinical_encoder=replace(cfg.clinical_encoder, missing_indicator=False))
    elif change == "task":
        cfg = replace(cfg, data=replace(cfg.data, task="different_task"))
    else:
        payload = torch.load(cfg.clinical_checkpoint, weights_only=False)
        if change == "scope":
            payload["training_scope"] = "fold"
        else:
            payload["config"]["modalities"] = ["clinical", "cxr"]
        torch.save(payload, cfg.clinical_checkpoint)
    with pytest.raises(ValueError):
        read_study_checkpoint(cfg.clinical_checkpoint, cfg, "clinical")


def test_fixed_split_weights_are_rejected_in_cross_validation(staged):
    cfg, _, _, _ = staged
    with pytest.raises(ValueError, match="CV folds"):
        train(cfg, fit_subjects=["a"], select_subjects=["b"], score_subjects=["c"])


def test_fusion_checkpoint_remains_portable_without_source_files(staged, tmp_path):
    from pathlib import Path
    cfg, geometry, _, normalizer = staged
    model = build_model(cfg, **geometry).eval()
    batch = {"image": torch.randn(2, 1, 32, 32), "clinical": torch.randn(2, 3, 8),
             "clinical_mask": torch.ones(2, 3, 8, dtype=torch.bool)}
    expected = model(batch)
    output = save_checkpoint(tmp_path / "fused.pt", model, epoch=1, geometry=geometry, normalizer=normalizer)
    Path(cfg.clinical_checkpoint).unlink()
    Path(cfg.image_checkpoint).unlink()
    restored, _ = load_checkpoint(output)
    torch.testing.assert_close(restored.eval()(batch), expected)
