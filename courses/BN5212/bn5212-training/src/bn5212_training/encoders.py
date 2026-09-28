"""Modality encoders. Both produce tokens in one shared embedding space.

Image encoder:    [B, C, H, W]        -> [B, N, D]   (CLS token at index 0)
Clinical encoder: [B, K, T] + mask    -> [B, M, D], [B, M]

The encoders are identical across all four experiments. Swapping the fusion
module must be the only difference between MeTra and the proposed method, so
nothing experiment-specific belongs in this file.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ClinicalEncoderConfig, ImageEncoderConfig
from .registry import CLINICAL_ENCODERS, IMAGE_ENCODERS


class ImageEncoder(nn.Module):
    """Base class fixing the token contract: CLS first, then patch tokens."""

    embed_dim: int
    num_tokens: int

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class VisionTransformer(ImageEncoder):
    """Plain ViT, implemented in pure torch so no download is needed for tests.

    MeTra uses an ImageNet-pretrained ViT-B/16 at 384x384. Use the `timm_vit`
    encoder for that; this one is for small/offline runs and for the synthetic
    fixture, where a pretrained 86M-parameter backbone would be meaningless.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        embed_dim: int = 192,
        patch_size: int = 16,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} is not divisible by patch_size {patch_size}"
            )
        grid = image_size // patch_size
        self.embed_dim = int(embed_dim)
        self.num_patches = grid * grid
        self.num_tokens = self.num_patches + 1

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(image).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.position
        return self.norm(self.blocks(self.dropout(tokens)))


class TimmVisionTransformer(ImageEncoder):
    """ImageNet-pretrained backbone from timm, for the MeTra-faithful setting."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "The timm_vit encoder needs timm. Install it with "
                "'pip install -e .[pretrained]'. Use image_encoder.name='vit' "
                "for an offline run."
            ) from error
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            img_size=image_size,
            drop_rate=dropout,
        )
        self.embed_dim = int(self.backbone.num_features)
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        with torch.no_grad():
            probe = self.backbone.forward_features(
                torch.zeros(1, in_channels, image_size, image_size)
            )
        if probe.ndim != 3:
            raise ValueError(
                f"{model_name} produced {probe.ndim}D features; the framework needs "
                "a token sequence [B, N, D]. Pick a ViT-style model."
            )
        self.num_tokens = int(probe.shape[1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(image)


class SimpleCNN(ImageEncoder):
    """Small conv stack, used as the cheap CXR reference and for fast tests."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        embed_dim: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        widths = [32, 64, 128]
        layers: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(channels, width, 3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.project = nn.Conv2d(channels, embed_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        with torch.no_grad():
            probe = self.project(
                self.features(torch.zeros(1, in_channels, image_size, image_size))
            )
        self.num_tokens = int(probe.shape[2] * probe.shape[3]) + 1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grid = self.project(self.features(image)).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(grid.shape[0], -1, -1)
        return self.norm(self.dropout(torch.cat([cls, grid], dim=1)))


@IMAGE_ENCODERS.register("vit")
def build_vit(cfg: ImageEncoderConfig, *, image_size: int, in_channels: int) -> VisionTransformer:
    return VisionTransformer(
        image_size,
        in_channels,
        embed_dim=cfg.embed_dim,
        patch_size=cfg.patch_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )


@IMAGE_ENCODERS.register("timm_vit")
def build_timm_vit(
    cfg: ImageEncoderConfig, *, image_size: int, in_channels: int
) -> TimmVisionTransformer:
    return TimmVisionTransformer(
        image_size,
        in_channels,
        model_name=cfg.timm_model,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        freeze=cfg.freeze,
    )


@IMAGE_ENCODERS.register("simple_cnn")
def build_simple_cnn(cfg: ImageEncoderConfig, *, image_size: int, in_channels: int) -> SimpleCNN:
    return SimpleCNN(image_size, in_channels, embed_dim=cfg.embed_dim, dropout=cfg.dropout)


class ClinicalEncoder(nn.Module):
    """Project the [B, K, T] clinical window into tokens plus a token mask.

    Tokenization decides what a clinical token *means*, which matters beyond
    accuracy: RQ3 asks which image regions a given clinical variable attends to,
    and that question is only answerable when one token is one variable. Hence
    `per_variable` (M = K) is the default. `per_timestep` (M = T) is available for
    ablations but makes the attention maps per-hour rather than per-variable.
    """

    def __init__(
        self,
        num_variables: int,
        num_timesteps: int,
        *,
        tokenization: str = "per_variable",
        embed_dim: int = 192,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        missing_indicator: bool = True,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if tokenization not in {"per_variable", "per_timestep"}:
            raise ValueError("clinical_encoder.tokenization must be per_variable or per_timestep")
        self.tokenization = tokenization
        self.num_variables = int(num_variables)
        self.num_timesteps = int(num_timesteps)
        self.embed_dim = int(embed_dim)
        self.missing_indicator = bool(missing_indicator)

        span = self.num_timesteps if tokenization == "per_variable" else self.num_variables
        self.num_tokens = self.num_variables if tokenization == "per_variable" else self.num_timesteps
        in_features = span * (2 if missing_indicator else 1)

        if depth <= 1:
            # MeTra projects the clinical window with a single linear layer.
            self.project: nn.Module = nn.Linear(in_features, embed_dim)
        else:
            self.project = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
        # Token identity: which variable (or which hour) this token came from.
        self.identity = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.identity, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[-2:] != (self.num_variables, self.num_timesteps):
            raise ValueError(
                f"Expected clinical values [B, {self.num_variables}, {self.num_timesteps}], "
                f"got {tuple(values.shape)}"
            )
        if mask.shape != values.shape:
            raise ValueError("Clinical mask must have the same shape as the values")

        observed = mask.to(values.dtype)
        if self.tokenization == "per_variable":
            series, indicator = values, observed
            token_mask = mask.any(dim=2)
        else:
            # One token per hour: transpose so the variable axis becomes features.
            series, indicator = values.transpose(1, 2), observed.transpose(1, 2)
            token_mask = mask.any(dim=1)

        features = torch.cat([series, indicator], dim=-1) if self.missing_indicator else series
        tokens = self.project(features) + self.identity
        # Zero out padded tokens so they cannot leak a learned bias downstream.
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return self.norm(self.dropout(tokens)), token_mask


@CLINICAL_ENCODERS.register("linear_projection")
def build_linear_projection(
    cfg: ClinicalEncoderConfig, *, num_variables: int, num_timesteps: int
) -> ClinicalEncoder:
    return ClinicalEncoder(
        num_variables,
        num_timesteps,
        tokenization=cfg.tokenization,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        missing_indicator=cfg.missing_indicator,
        depth=1,
    )


@CLINICAL_ENCODERS.register("mlp_projection")
def build_mlp_projection(
    cfg: ClinicalEncoderConfig, *, num_variables: int, num_timesteps: int
) -> ClinicalEncoder:
    return ClinicalEncoder(
        num_variables,
        num_timesteps,
        tokenization=cfg.tokenization,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        missing_indicator=cfg.missing_indicator,
        depth=2,
    )
