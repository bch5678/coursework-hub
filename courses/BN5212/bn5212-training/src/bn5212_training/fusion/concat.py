"""Simple late fusion: pool each modality, concatenate, then one MLP.

The reference that attention-based fusion has to beat.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FusionConfig
from ..registry import FUSIONS
from .base import FusionModule, masked_mean


class ConcatMLPFusion(FusionModule):
    uses_image = True
    uses_clinical = True

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.clinical_norm = nn.LayerNorm(embed_dim)
        self.project = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        image = self.image_norm(image_tokens[:, 0])
        clinical = self.clinical_norm(masked_mean(clinical_tokens, clinical_mask))
        return self.project(torch.cat([image, clinical], dim=-1))


@FUSIONS.register("concat_mlp")
def build_concat_mlp(cfg: FusionConfig, **_: object) -> ConcatMLPFusion:
    return ConcatMLPFusion(cfg.embed_dim, cfg.dropout)
