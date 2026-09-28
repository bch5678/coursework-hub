"""Deterministic inference and the prediction hand-off to benchmark-evaluation.

The only artefact the benchmark accepts is a CSV with exactly two columns,
sample_id and y_score, where y_score is P(in-hospital mortality) in [0, 1] and the
sample_id set matches the frozen split exactly. Everything here exists to produce
that file and to fail loudly rather than emit a file the benchmark will reject.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .clinical import ClinicalNormalizer, build_provider
from .config import TrainingConfig
from .data import TrainingDataset, make_loader
from .model import BN5212Model, load_checkpoint


def positive_probability(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to the positive-class probability.

    Mirrors benchmark-evaluation's logits_to_positive_probability so the training
    and evaluation projects cannot disagree about what the positive class is.
    """
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(f"Expected logits shaped [B], [B,1] or [B,2], got {tuple(logits.shape)}")


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


@torch.inference_mode()
def run_inference(
    model: BN5212Model, loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Run eval-mode inference and return aligned ids, labels and probabilities."""
    model.eval()
    sample_ids: list[str] = []
    subject_ids: list[str] = []
    hadm_ids: list[str] = []
    scores: list[float] = []
    labels: list[int] = []

    for batch in loader:
        batch_on_device = move_to_device(batch, device)
        probability = positive_probability(model(batch_on_device)).detach().float().cpu()
        if probability.ndim != 1 or len(probability) != len(batch["sample_id"]):
            raise ValueError("Model output batch size does not match sample_id")
        if not torch.isfinite(probability).all():
            raise ValueError("Model produced non-finite probabilities")
        sample_ids.extend(batch["sample_id"])
        subject_ids.extend(batch.get("subject_id", [""] * len(probability)))
        hadm_ids.extend(batch.get("hadm_id", [""] * len(probability)))
        scores.extend(probability.tolist())
        labels.extend(batch["label"].detach().cpu().tolist())

    return {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "subject_id": np.asarray(subject_ids, dtype=object),
        "hadm_id": np.asarray(hadm_ids, dtype=object),
        "y_score": np.asarray(scores, dtype=np.float64),
        "label": np.asarray(labels, dtype=np.int64),
    }


def write_predictions(result: Mapping[str, np.ndarray], path: str | Path) -> Path:
    """Write the two-column file the benchmark consumes, after validating it."""
    path = Path(path)
    frame = pd.DataFrame(
        {"sample_id": result["sample_id"], "y_score": result["y_score"]}
    )
    if frame.empty:
        raise ValueError("Refusing to write an empty prediction file")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Prediction file would contain duplicate sample_id values")
    if not np.isfinite(frame["y_score"]).all():
        raise ValueError("Prediction file would contain non-finite y_score values")
    if not frame["y_score"].between(0.0, 1.0).all():
        raise ValueError("y_score must be a probability in [0, 1]")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def write_detailed_predictions(
    result: Mapping[str, np.ndarray], path: str | Path, *, split: str, model_version: str
) -> Path:
    """Richer per-sample file for our own inspection; not the benchmark contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": result["sample_id"],
            "subject_id": result["subject_id"],
            "hadm_id": result["hadm_id"],
            "label": result["label"],
            "y_score": result["y_score"],
            "split": split,
            "model_version": model_version,
        }
    ).to_csv(path, index=False, lineterminator="\n")
    return path


def build_inference_loader(
    cfg: TrainingConfig,
    split: str,
    *,
    normalizer: ClinicalNormalizer | None = None,
) -> TrainingDataset:
    """Dataset for a split, reusing the frozen normalisation from training."""
    provider = None
    if cfg.uses_clinical:
        names = None if normalizer is None else list(normalizer.variable_names)
        provider = build_provider(cfg.data, variable_names=names)
    return TrainingDataset(
        cfg.data.run_dir,
        split,
        data_pipeline_path=cfg.data.data_pipeline_path,
        provider=provider,
        normalizer=normalizer,
        load_image=cfg.uses_image,
    )


def predict_from_checkpoint(
    checkpoint: str | Path,
    *,
    split: str,
    output_path: str | Path,
    run_dir: str | None = None,
    device: str = "auto",
) -> Path:
    """Standalone prediction entry point for an existing checkpoint."""
    target = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, map_location=target)
    cfg = model.cfg
    if run_dir:
        cfg = replace(cfg, data=replace(cfg.data, run_dir=run_dir))
    normalizer = (
        ClinicalNormalizer.from_dict(payload["clinical_normalizer"])
        if payload.get("clinical_normalizer")
        else None
    )
    dataset = build_inference_loader(cfg, split, normalizer=normalizer)
    loader = make_loader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        seed=cfg.seed,
    )
    model.to(target)
    result = run_inference(model, loader, target)
    return write_predictions(result, output_path)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
