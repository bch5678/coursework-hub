"""Clinical queries over image patches, exposed through the fusion plug-in API."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FusionConfig
from ..registry import FUSIONS
from .base import FusionModule, masked_mean


class CrossAttentionAdapterBlock(nn.Module):
    """Pre-norm cross-attention and FFN with clinical residual connections."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, clinical: torch.Tensor, patches: torch.Tensor, record_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        context = self.context_norm(patches)
        attended, weights = self.attention(
            self.query_norm(clinical), context, context,
            need_weights=record_attention, average_attn_weights=False,
        )
        attended = self.attention_dropout(attended)
        clinical = clinical + attended if self.residual else attended
        return clinical + self.ffn(self.ffn_norm(clinical)), weights


class CrossAttentionAdapter(FusionModule):
    """Map [B, M, D] clinical queries and [B, 1+N, D] images to [B, D].

    The first image token is the encoder's CLS/global summary and is excluded.
    Missing clinical queries contribute neither pooled features nor recorded
    attention. An entirely missing clinical row produces a zero patient vector.
    """

    uses_image = True
    uses_clinical = True

    def __init__(
        self,
        embed_dim: int,
        *,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__(embed_dim)
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if depth < 1:
            raise ValueError("fusion.depth must be >= 1")
        if not mlp_ratio > 0 or int(embed_dim * mlp_ratio) < 1:
            raise ValueError("fusion.mlp_ratio must produce a positive FFN width")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("fusion.dropout must be between 0 and 1")
        self.blocks = nn.ModuleList(
            CrossAttentionAdapterBlock(embed_dim, num_heads, mlp_ratio, dropout, residual)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.record_attention = False
        self._attention: torch.Tensor | None = None

    def last_attention(self) -> torch.Tensor | None:
        """Detached final-layer weights [B, heads, M, N], without image CLS."""
        return self._attention

    def forward(
        self, image_tokens: torch.Tensor | None = None,
        clinical_tokens: torch.Tensor | None = None,
        clinical_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._attention = None
        self.check_inputs(image_tokens, clinical_tokens)
        for name, tokens in (("image", image_tokens), ("clinical", clinical_tokens)):
            if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
                raise ValueError(f"Expected {name} tokens [B, L, {self.embed_dim}]")
        if image_tokens.shape[0] != clinical_tokens.shape[0]:
            raise ValueError("Image and clinical batch sizes must match")
        if image_tokens.shape[1] < 2:
            raise ValueError("Image tokens must contain CLS followed by at least one patch")
        if clinical_tokens.shape[1] < 1:
            raise ValueError("At least one clinical token is required")
        if clinical_mask is None:
            valid = torch.ones(
                clinical_tokens.shape[:2], dtype=torch.bool, device=clinical_tokens.device
            )
        else:
            if clinical_mask.shape != clinical_tokens.shape[:2]:
                raise ValueError("clinical_mask must have shape [B, M]")
            valid = clinical_mask.to(device=clinical_tokens.device, dtype=torch.bool)
        queried = clinical_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        patches = image_tokens[:, 1:]
        weights = None
        for block in self.blocks:
            queried, weights = block(queried, patches, self.record_attention)
            queried = queried.masked_fill(~valid.unsqueeze(-1), 0.0)
        if weights is not None:
            self._attention = weights.detach().masked_fill(~valid[:, None, :, None], 0.0)
        pooled = self.norm(masked_mean(queried, valid))
        return pooled.masked_fill(~valid.any(dim=1, keepdim=True), 0.0)


@FUSIONS.register("cross_attention_adapter")
def build_cross_attention_adapter(cfg: FusionConfig, **_: object) -> CrossAttentionAdapter:
    if cfg.direction != "clinical_to_image":
        raise ValueError("cross_attention_adapter requires fusion.direction=clinical_to_image")
    return CrossAttentionAdapter(
        cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        residual=cfg.residual,
    )
