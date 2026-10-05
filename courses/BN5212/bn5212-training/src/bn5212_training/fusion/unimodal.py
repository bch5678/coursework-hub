"""Single-modality fusion: pool one token set and ignore the other."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FusionConfig
from ..registry import FUSIONS
from .base import FusionModule, masked_mean


class _PoolingFusion(FusionModule):
    def __init__(self, embed_dim: int, pooling: str, dropout: float = 0.0) -> None:
        super().__init__(embed_dim)
        if pooling not in {"cls", "mean"}:
            raise ValueError("fusion.pooling must be cls or mean")
        self.pooling = pooling
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _pool(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # "cls" reuses the encoder's leading token; image encoders put it at index 0.
        pooled = tokens[:, 0] if self.pooling == "cls" else masked_mean(tokens, mask)
        return self.dropout(self.norm(pooled))


class ImageOnlyFusion(_PoolingFusion):
    uses_image = True
    uses_clinical = False

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        return self._pool(image_tokens, None)


class ClinicalOnlyFusion(_PoolingFusion):
    uses_image = False
    uses_clinical = True

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        return self._pool(clinical_tokens, clinical_mask)


@FUSIONS.register("image_only")
def build_image_only(cfg: FusionConfig, **_: object) -> ImageOnlyFusion:
    return ImageOnlyFusion(cfg.embed_dim, cfg.pooling, cfg.dropout)


@FUSIONS.register("clinical_only")
def build_clinical_only(cfg: FusionConfig, **_: object) -> ClinicalOnlyFusion:
    # Clinical tokens carry no CLS token, so masked mean is the meaningful pool.
    pooling = "mean" if cfg.pooling == "cls" else cfg.pooling
    return ClinicalOnlyFusion(cfg.embed_dim, pooling, cfg.dropout)
