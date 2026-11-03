"""Restore fixed-split single-modality encoders for controlled fusion experiments."""
from pathlib import Path

import torch

from .config import config_from_dict
from .manifest import sha256


def read_study_checkpoint(path, cfg, branch):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "bn5212-training/1":
        raise ValueError("Study encoder requires a bn5212-training checkpoint")
    if payload.get("training_scope") != "fixed":
        raise ValueError("Study encoder must come from fixed train/val training")
    source = config_from_dict(payload["config"])
    expected_modality = "clinical" if branch == "clinical" else "cxr"
    if source.modalities != (expected_modality,):
        raise ValueError("Study encoder source must be the corresponding single-modality model")
    if source.data != cfg.data or source.seed != cfg.seed:
        raise ValueError("Study encoder data settings and seed must match the fusion experiment")
    if getattr(source, f"{branch}_encoder") != getattr(cfg, f"{branch}_encoder"):
        raise ValueError("Study encoder architecture/settings do not match")
    index = Path(cfg.data.run_dir) / "index.csv"
    if not index.is_file() or payload.get("dataset_index_sha256") != sha256(index):
        raise ValueError("Study encoder dataset index hash does not match")
    if branch == "clinical" and cfg.data.clinical_source:
        if payload.get("clinical_source_sha256") != sha256(cfg.data.clinical_source):
            raise ValueError("Study encoder clinical table hash does not match")
    return payload


def restore_study_encoder(encoder, path, cfg, branch):
    payload = read_study_checkpoint(path, cfg, branch)
    prefix = f"{branch}_encoder."
    state = {name[len(prefix):]: value for name, value in payload["state_dict"].items()
             if name.startswith(prefix)}
    encoder.load_state_dict(state, strict=True)
