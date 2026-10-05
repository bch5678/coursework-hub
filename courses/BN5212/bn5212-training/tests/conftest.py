"""Shared fixtures.

The dataset fixture is built by running the real data pipeline over its own
synthetic generator, so the tests exercise the actual upstream contract -- hash
verification, patient isolation, the sample dict -- without touching MIMIC. No
course data, credentials or restricted images are involved.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from bn5212_training import upstream


def _load_generator(pipeline_root: Path):
    path = pipeline_root / "scripts" / "make_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("bn5212_make_synthetic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def dataset_run(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A completed pipeline run directory built from synthetic records."""
    try:
        pipeline_root = upstream.data_pipeline_root()
    except FileNotFoundError as error:  # pragma: no cover - depends on checkout
        pytest.skip(str(error))

    generator = _load_generator(pipeline_root)
    destination = tmp_path_factory.mktemp("fixture") / "png"
    config_path = generator.make_synthetic(destination)

    upstream.load_pipeline()
    build = importlib.import_module("bn5212_pipeline.data.pipeline").build
    output, _summary = build(config_path)
    return str(output)


@pytest.fixture
def base_config(dataset_run: str) -> dict:
    """Minimal multimodal config sized for the 32x32 synthetic images."""
    return {
        "experiment": "test",
        "modalities": ["clinical", "cxr"],
        "data": {
            "run_dir": dataset_run,
            "clinical_provider": "synthetic",
            "clinical_timesteps": 8,
            "batch_size": 4,
        },
        "image_encoder": {"name": "vit", "embed_dim": 32, "patch_size": 16, "depth": 1, "num_heads": 2},
        "clinical_encoder": {"name": "linear_projection", "embed_dim": 32},
        "fusion": {"name": "joint_self_attention", "embed_dim": 32, "depth": 1, "num_heads": 2},
        "head": {"num_outputs": 1},
        "optim": {"epochs": 1, "warmup_epochs": 0, "early_stopping_patience": 3},
        "seed": 5212,
        "device": "cpu",
    }
