"""MeTra's fusion: one shared self-attention space over both modalities.

Reproduces the fusion described in Sec. 6.3 of the project plan and in the MeTra
paper (Nature Scientific Reports 2023, s41598-023-37835-1):

    Z = [CLS; I; C] + learnable positional embeddings
    Z' = TransformerEncoder(Z)          # multi-head self-attention
    prediction = head(Z'[CLS])

Because attention runs over the concatenated sequence, all four interactions
happen in the same weights and are not separable: image->image,
clinical->clinical, clinical->image and image->clinical. That entanglement is
exactly what the proposed cross-attention module is testing against.

This is an independent implementation of the published architecture, not a copy
of the authors' repository (github.com/FirasGit/MeTra states no license).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FusionConfig
from ..registry import FUSIONS
from .base import FusionModule, build_key_padding_mask, build_transformer_encoder, masked_mean


class JointSelfAttentionFusion(FusionModule):
    """Concatenate modality tokens and run joint multi-head self-attention."""

    uses_image = True
    uses_clinical = True

    def __init__(
        self,
        embed_dim: int,
        num_image_tokens: int,
        num_clinical_tokens: int,
        *,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pooling: str = "cls",
    ) -> None:
        super().__init__(embed_dim)
        if pooling not in {"cls", "mean"}:
            raise ValueError("fusion.pooling must be cls or mean")
        self.num_image_tokens = int(num_image_tokens)
        self.num_clinical_tokens = int(num_clinical_tokens)
        self.sequence_length = 1 + self.num_image_tokens + self.num_clinical_tokens
        self.pooling = pooling

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # One learnable embedding per position, added element-wise (MeTra).
        self.position = nn.Parameter(torch.zeros(1, self.sequence_length, embed_dim))
        # A learnable per-modality offset keeps the origin of a token recoverable
        # after concatenation; without it the encoder must infer modality from
        # position alone.
        self.modality = nn.Parameter(torch.zeros(1, 2, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.modality, std=0.02)

        self.input_dropout = nn.Dropout(dropout)
        self.encoder = build_transformer_encoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        batch = image_tokens.shape[0]
        if image_tokens.shape[1] != self.num_image_tokens:
            raise ValueError(
                f"Expected {self.num_image_tokens} image tokens, got {image_tokens.shape[1]}"
            )
        if clinical_tokens.shape[1] != self.num_clinical_tokens:
            raise ValueError(
                f"Expected {self.num_clinical_tokens} clinical tokens, got {clinical_tokens.shape[1]}"
            )

        image = image_tokens + self.modality[:, 0:1]
        clinical = clinical_tokens + self.modality[:, 1:2]
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, image, clinical], dim=1) + self.position
        sequence = self.input_dropout(sequence)

        padding = build_key_padding_mask(
            clinical_mask, self.num_clinical_tokens, batch, sequence.device
        )
        if padding is not None:
            keep = torch.zeros(
                batch, 1 + self.num_image_tokens, dtype=torch.bool, device=sequence.device
            )
            padding = torch.cat([keep, padding], dim=1)

        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        if self.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            valid = None if padding is None else ~padding
            pooled = masked_mean(encoded, valid)
        return self.norm(pooled)


@FUSIONS.register("joint_self_attention")
def build_joint_self_attention(
    cfg: FusionConfig,
    *,
    num_image_tokens: int,
    num_clinical_tokens: int,
    **_: object,
) -> JointSelfAttentionFusion:
    return JointSelfAttentionFusion(
        cfg.embed_dim,
        num_image_tokens,
        num_clinical_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    )
