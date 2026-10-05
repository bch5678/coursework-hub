"""Loss construction.

sample_weight (1 / images in that admission) weights the per-image loss, so an
admission with many radiographs does not count as several patients.
positive_class_weight: None reproduces MeTra; "auto" or a number reweights.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainingConfig

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def positive_class_weight(labels: np.ndarray) -> float:
    """negatives / positives, the standard BCE pos_weight for imbalance."""
    labels = np.asarray(labels)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        raise ValueError("Cannot derive a positive class weight: no positive labels")
    return float(negatives) / float(positives)


def build_loss(cfg: TrainingConfig, train_labels: np.ndarray) -> LossFn:
    """Return loss(logits, labels, sample_weight) -> scalar."""
    weight = cfg.optim.positive_class_weight
    pos_weight = None
    if weight is not None:
        resolved = positive_class_weight(train_labels) if weight == "auto" else float(weight)
        if resolved <= 0:
            raise ValueError("optim.positive_class_weight must be positive")
        pos_weight = resolved
    use_sample_weight = cfg.optim.use_sample_weight
    num_outputs = cfg.head.num_outputs

    def loss_fn(
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_outputs == 1:
            if logits.shape != labels.shape:
                logits = logits.reshape(labels.shape)
            per_sample = F.binary_cross_entropy_with_logits(
                logits,
                labels.to(logits.dtype),
                pos_weight=(
                    None
                    if pos_weight is None
                    else torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
                ),
                reduction="none",
            )
        else:
            class_weight = None
            if pos_weight is not None:
                class_weight = torch.tensor(
                    [1.0, pos_weight], device=logits.device, dtype=logits.dtype
                )
            per_sample = F.cross_entropy(
                logits, labels.long(), weight=class_weight, reduction="none"
            )

        if use_sample_weight and sample_weight is not None:
            weights = sample_weight.to(per_sample.dtype)
            # Normalise by the weight total so the effective learning rate does
            # not change with the average number of images per admission.
            return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
        return per_sample.mean()

    return loss_fn
