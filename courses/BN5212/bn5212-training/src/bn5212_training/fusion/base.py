"""The single interface every fusion strategy implements.

    forward(image_tokens=[B,N,D] | None,
            clinical_tokens=[B,M,D] | None,
            clinical_mask=[B,M] | None) -> [B, output_dim]

Unimodal experiments pass None for the modality they do not use.
"""
from __future__ import annotations

import abc

import torch
import torch.nn as nn


class FusionModule(nn.Module, abc.ABC):
    """Reduce image and/or clinical tokens to one pooled patient vector."""

    #: Whether forward() requires the corresponding token tensor.
    uses_image: bool = False
    uses_clinical: bool = False

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    @property
    def output_dim(self) -> int:
        """Width of the pooled representation handed to the prediction head."""
        return self.embed_dim

    @abc.abstractmethod
    def forward(
        self,
        image_tokens: torch.Tensor | None = None,
        clinical_tokens: torch.Tensor | None = None,
        clinical_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def check_inputs(
        self,
        image_tokens: torch.Tensor | None,
        clinical_tokens: torch.Tensor | None,
    ) -> None:
        if self.uses_image and image_tokens is None:
            raise ValueError(f"{type(self).__name__} requires image tokens")
        if self.uses_clinical and clinical_tokens is None:
            raise ValueError(f"{type(self).__name__} requires clinical tokens")

    def last_attention(self) -> torch.Tensor | None:
        """Most recent attention weights, when the strategy records them.

        Cross-attention returns [B, heads, M, N] so that RQ3 can ask which image
        patches a given clinical variable attended to. Strategies that do not
        record attention return None.
        """
        return None


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over the token axis, ignoring padded positions.

    tokens: [B, L, D]; mask: [B, L] with True for real tokens.
    """
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    total = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / total


def build_transformer_encoder(
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
) -> nn.TransformerEncoder:
    """Pre-norm encoder stack, batch-first, matching the MeTra description."""
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


def build_key_padding_mask(
    mask: torch.Tensor | None,
    length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert a validity mask to the ignore-mask that nn attention expects.

    `mask` is [B, L] with True for real tokens; the returned tensor is True where
    a position must be ignored. A row that would be fully ignored produces NaN
    inside softmax, so the first position of such a row is kept.
    """
    if mask is None:
        return None
    if mask.shape != (batch_size, length):
        raise ValueError(f"Expected mask shaped {(batch_size, length)}, got {tuple(mask.shape)}")
    valid = mask.to(torch.bool).clone()
    empty = ~valid.any(dim=1)
    if bool(empty.any()):
        valid[empty, 0] = True
    return (~valid).to(device)
