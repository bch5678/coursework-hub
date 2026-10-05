"""Fusion strategies.

Importing this package registers every strategy in the FUSIONS registry. A new
strategy is a new module here plus one line below -- nothing else changes.
"""
from __future__ import annotations

from ..registry import FUSIONS
from . import concat, cross_attention, joint_self_attention, unimodal  # noqa: F401
from .base import FusionModule, build_key_padding_mask, build_transformer_encoder, masked_mean

__all__ = [
    "FUSIONS",
    "FusionModule",
    "build_fusion",
    "build_key_padding_mask",
    "build_transformer_encoder",
    "masked_mean",
]


def build_fusion(cfg, *, num_image_tokens: int = 0, num_clinical_tokens: int = 0) -> FusionModule:
    return FUSIONS.build(
        cfg.name,
        cfg,
        num_image_tokens=num_image_tokens,
        num_clinical_tokens=num_clinical_tokens,
    )
