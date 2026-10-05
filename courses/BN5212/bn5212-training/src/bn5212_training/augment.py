"""Training-time image augmentation.

No horizontal flip by default: a mirrored chest film shows dextrocardia.
Transforms receive the already-normalised tensor, so brightness and contrast
are affine operations in normalised units.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import AugmentationConfig


class ChestRadiographAugmentation(nn.Module):
    """Geometry and intensity jitter that preserves left/right anatomy."""

    def __init__(self, cfg: AugmentationConfig, image_size: int) -> None:
        super().__init__()
        if not 0 < cfg.crop_scale_min <= cfg.crop_scale_max <= 1.0:
            raise ValueError("augmentation crop scale must satisfy 0 < min <= max <= 1")
        if cfg.rotation_degrees < 0:
            raise ValueError("augmentation rotation_degrees must be non-negative")
        self.cfg = cfg
        self.image_size = int(image_size)

        from torchvision.transforms import v2

        steps: list[nn.Module] = []
        if cfg.rotation_degrees:
            # fill=0 is the mean intensity in normalised units, so a rotated
            # corner blends in instead of leaving a hard wedge the model could
            # key on as a label-correlated artefact.
            steps.append(v2.RandomRotation(cfg.rotation_degrees, fill=0.0))
        if cfg.crop_scale_min < 1.0:
            steps.append(
                v2.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(cfg.crop_scale_min, cfg.crop_scale_max),
                    ratio=(0.95, 1.05),
                    antialias=True,
                )
            )
        if cfg.horizontal_flip:
            steps.append(v2.RandomHorizontalFlip(p=0.5))
        self.geometry = v2.Compose(steps) if steps else None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.geometry is not None:
            image = self.geometry(image)
        cfg = self.cfg
        if cfg.contrast:
            # Scale around the image's own mean so average intensity is preserved.
            factor = 1.0 + float(torch.empty(()).uniform_(-cfg.contrast, cfg.contrast))
            mean = image.mean()
            image = (image - mean) * factor + mean
        if cfg.brightness:
            image = image + float(torch.empty(()).uniform_(-cfg.brightness, cfg.brightness))
        return image


def build_train_transform(cfg: AugmentationConfig, image_size: int):
    """Return the training transform, or None when augmentation is off."""
    if not cfg.enabled:
        return None
    return ChestRadiographAugmentation(cfg, image_size)
