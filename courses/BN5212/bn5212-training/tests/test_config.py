"""Config validation.

A silently ignored typo in a config would produce a run that looks fine and
answers a different question, so unknown keys are errors rather than warnings.
"""
from __future__ import annotations

import json

import pytest

from bn5212_training.config import config_from_dict, load_config


def _minimal(**overrides):
    payload = {
        "experiment": "unit",
        "modalities": ["cxr"],
        "data": {"run_dir": "/tmp/run"},
        "image_encoder": {"embed_dim": 64},
        "fusion": {"name": "image_only", "embed_dim": 64},
    }
    payload.update(overrides)
    return payload


def test_minimal_config_builds():
    cfg = config_from_dict(_minimal())
    assert cfg.uses_image and not cfg.uses_clinical
    assert cfg.modalities == ("cxr",)


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown top-level"):
        config_from_dict(_minimal(epochs=10))


def test_unknown_section_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown optim"):
        config_from_dict(_minimal(optim={"learning_rate": 1e-3}))


def test_unknown_modality_is_rejected():
    with pytest.raises(ValueError, match="Unknown modalities"):
        config_from_dict(_minimal(modalities=["cxr", "notes"]))


def test_empty_modalities_is_rejected():
    with pytest.raises(ValueError, match="At least one modality"):
        config_from_dict(_minimal(modalities=[]))


def test_embedding_width_mismatch_is_caught_before_training():
    """A mismatch here would only surface as a shape error deep inside a run."""
    with pytest.raises(ValueError, match="embed_dim"):
        config_from_dict(
            _minimal(image_encoder={"embed_dim": 64}, fusion={"name": "image_only", "embed_dim": 128})
        )


def test_clinical_width_must_match_fusion():
    with pytest.raises(ValueError, match="clinical_encoder.embed_dim"):
        config_from_dict(
            _minimal(
                modalities=["clinical"],
                clinical_encoder={"embed_dim": 32},
                fusion={"name": "clinical_only", "embed_dim": 64},
            )
        )


def test_invalid_head_width_is_rejected():
    with pytest.raises(ValueError, match="num_outputs"):
        config_from_dict(_minimal(head={"num_outputs": 3}))


def test_invalid_selection_metric_is_rejected():
    with pytest.raises(ValueError, match="selection_metric"):
        config_from_dict(_minimal(selection_metric="accuracy"))


def test_fingerprint_is_stable_and_sensitive():
    first = config_from_dict(_minimal())
    same = config_from_dict(_minimal())
    different = config_from_dict(_minimal(seed=99))
    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != different.fingerprint()


def test_dotted_overrides_reach_the_right_section(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_minimal()), encoding="utf-8")
    cfg = load_config(path, {"optim.epochs": 7, "data.run_dir": "/elsewhere", "seed": 1})
    assert cfg.optim.epochs == 7
    assert cfg.data.run_dir == "/elsewhere"
    assert cfg.seed == 1


def test_override_of_unknown_section_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_minimal()), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown section"):
        load_config(path, {"trainer.epochs": 3})


def test_shipped_configs_are_all_valid():
    """configs/ also holds the clinical item map, which is not an experiment."""
    from pathlib import Path

    from bn5212_training.fusion import FUSIONS

    directory = Path(__file__).resolve().parents[1] / "configs"
    experiments = [
        path
        for path in sorted(directory.rglob("*.json"))
        if "experiment" in json.loads(path.read_text(encoding="utf-8"))
    ]
    assert len(experiments) >= 10, "expected the real and synthetic config sets"
    for path in experiments:
        cfg = load_config(path)
        assert cfg.experiment
        # Every shipped config must name a fusion the registry actually has.
        assert cfg.fusion.name in FUSIONS, f"{path.name} names an unknown fusion"
