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

    def __init__(self, embed_dim: int, dropout: float = 0.0, image_pooling: str = "cls", project: bool = True) -> None:
        super().__init__(embed_dim)
        if image_pooling not in {"cls", "patch_mean"}:
            raise ValueError("fusion.concat_image_pooling must be cls or patch_mean")
        self.image_pooling = image_pooling
        self.raw_concat = not project
        self.image_norm = nn.LayerNorm(embed_dim) if project else nn.Identity()
        self.clinical_norm = nn.LayerNorm(embed_dim) if project else nn.Identity()
        self.project = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        ) if project else nn.Identity()

    @property
    def output_dim(self):
        return 2 * self.embed_dim if self.raw_concat else self.embed_dim

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        if self.image_pooling == "patch_mean":
            if image_tokens.shape[1] < 2:
                raise ValueError("patch_mean requires CLS followed by image patches")
            image = image_tokens[:, 1:].mean(dim=1)
        else:
            image = image_tokens[:, 0]
        image = self.image_norm(image)
        clinical = self.clinical_norm(masked_mean(clinical_tokens, clinical_mask))
        return self.project(torch.cat([image, clinical], dim=-1))


@FUSIONS.register("concat_mlp")
def build_concat_mlp(cfg: FusionConfig, **_: object) -> ConcatMLPFusion:
    return ConcatMLPFusion(cfg.embed_dim, cfg.dropout, cfg.concat_image_pooling, cfg.concat_project)
