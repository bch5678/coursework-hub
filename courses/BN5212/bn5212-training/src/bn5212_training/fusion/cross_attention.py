"""Explicit cross-modal attention (the project's proposed fusion).

    Q = C, K = V = I   ->   C' = softmax(QK^T / sqrt(d)) V

Each clinical token queries the image tokens directly, so clinical->image is the
only cross-modal path in the module and it is separable from the within-modality
paths that joint self-attention mixes in. `direction` covers the project plan's
Optional 1 and 2 (clinical_to_image, image_to_clinical, bidirectional).

OWNERSHIP: this is a reference implementation supplied by the training framework
so that all four experiments run end to end today. The multimodal owner owns this
file. Replacing it requires no change anywhere else in the framework -- keep the
FusionModule signature and re-register the same name.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import FusionConfig
from ..registry import FUSIONS
from .base import FusionModule, build_key_padding_mask, masked_mean

DIRECTIONS = ("clinical_to_image", "image_to_clinical", "bidirectional")


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a position-wise feed-forward layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        record_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed_query = self.query_norm(query)
        normed_context = self.context_norm(context)
        attended, weights = self.attention(
            normed_query,
            normed_context,
            normed_context,
            key_padding_mask=key_padding_mask,
            need_weights=record_attention,
            average_attn_weights=False,
        )
        query = query + attended if self.residual else attended
        query = query + self.mlp(self.mlp_norm(query))
        return query, weights


class CrossAttentionFusion(FusionModule):
    uses_image = True
    uses_clinical = True

    def __init__(
        self,
        embed_dim: int,
        *,
        direction: str = "clinical_to_image",
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__(embed_dim)
        if direction not in DIRECTIONS:
            raise ValueError(f"fusion.direction must be one of {DIRECTIONS}")
        self.direction = direction
        # Attention weights are only materialised when explicitly requested, so
        # training keeps the fast attention path.
        self.record_attention = False
        self._attention: dict[str, torch.Tensor | None] = {}

        def stack() -> nn.ModuleList:
            return nn.ModuleList(
                _CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout, residual)
                for _ in range(depth)
            )

        self.clinical_blocks = stack() if direction != "image_to_clinical" else None
        self.image_blocks = stack() if direction != "clinical_to_image" else None
        self.norm = nn.LayerNorm(self._output_dim())

    def _output_dim(self) -> int:
        return 2 * self.embed_dim if self.direction == "bidirectional" else self.embed_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim()

    def last_attention(self) -> torch.Tensor | None:
        """Clinical->image weights [B, heads, M, N] from the final block."""
        return self._attention.get("clinical_to_image")

    def attention_maps(self) -> dict[str, torch.Tensor | None]:
        """All recorded attention tensors, keyed by direction."""
        return dict(self._attention)

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        self._attention = {}
        batch = image_tokens.shape[0]
        clinical_padding = build_key_padding_mask(
            clinical_mask, clinical_tokens.shape[1], batch, clinical_tokens.device
        )
        valid = None if clinical_padding is None else ~clinical_padding
        pooled: list[torch.Tensor] = []

        if self.clinical_blocks is not None:
            # Q = C, K = V = I. Image tokens are never padded, so no key mask.
            queried = clinical_tokens
            weights = None
            for block in self.clinical_blocks:
                queried, weights = block(queried, image_tokens, None, self.record_attention)
            self._attention["clinical_to_image"] = weights
            pooled.append(masked_mean(queried, valid))

        if self.image_blocks is not None:
            # Q = I, K = V = C. Padded clinical positions must be ignored here.
            queried = image_tokens
            weights = None
            for block in self.image_blocks:
                queried, weights = block(
                    queried, clinical_tokens, clinical_padding, self.record_attention
                )
            self._attention["image_to_clinical"] = weights
            pooled.append(queried[:, 0])

        return self.norm(torch.cat(pooled, dim=-1) if len(pooled) > 1 else pooled[0])


@FUSIONS.register("cross_attention")
def build_cross_attention(cfg: FusionConfig, **_: object) -> CrossAttentionFusion:
    return CrossAttentionFusion(
        cfg.embed_dim,
        direction=cfg.direction,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        residual=cfg.residual,
    )
