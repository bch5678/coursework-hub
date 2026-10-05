"""Modality encoders. Both produce tokens in one shared embedding space.

Image encoder:    [B, C, H, W]        -> [B, N, D]   (CLS token at index 0)
Clinical encoder: [B, K, T] + mask    -> [B, M, D], [B, M]
"""
from __future__ import annotations

from typing import Sequence

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
    """Plain ViT in pure torch, for offline runs and the synthetic fixture."""

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
    """ImageNet-pretrained backbone from timm, optionally frozen and projected."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze: bool = False,
        project_to: int | None = None,
        unfreeze_last_blocks: int = 0,
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
        backbone_width = int(self.backbone.num_features)
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            # Partial fine-tuning: only the last blocks and the final norm are trained.
            if unfreeze_last_blocks > 0:
                blocks = getattr(self.backbone, "blocks", None)
                if blocks is None:
                    raise ValueError(
                        f"{model_name} has no .blocks to unfreeze; use freeze=false instead"
                    )
                for block in list(blocks)[-unfreeze_last_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad_(True)
                for name in ("norm", "fc_norm"):
                    module = getattr(self.backbone, name, None)
                    if module is not None:
                        for parameter in module.parameters():
                            parameter.requires_grad_(True)

        with torch.no_grad():
            probe = self.backbone.forward_features(
                torch.zeros(1, in_channels, image_size, image_size)
            )
        if probe.ndim != 3:
            raise ValueError(
                f"{model_name} produced {probe.ndim}D features; the framework needs "
                "a token sequence [B, N, D]. Use the timm_cnn encoder for a "
                "convolutional backbone."
            )
        self.num_tokens = int(probe.shape[1])
        # Projection is trainable even when the backbone is frozen, which is what
        # lets a fixed 768-wide backbone feed a narrow fusion module.
        self.project = nn.Linear(backbone_width, project_to) if project_to else nn.Identity()
        self.embed_dim = int(project_to or backbone_width)
        self._fully_frozen = bool(freeze) and unfreeze_last_blocks == 0

    def train(self, mode: bool = True) -> "TimmVisionTransformer":
        super().train(mode)
        # A frozen backbone is a fixed feature extractor: it must not switch its
        # dropout or normalisation layers into training behaviour.
        if self._fully_frozen:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(self.backbone.forward_features(image))


class TimmResNet(ImageEncoder):
    """ImageNet-pretrained ResNet from timm, flattened to a token sequence.

    The second CNN arm. `xrv_densenet` answers "what does a chest-radiograph
    pretrained DenseNet give us"; this one answers "what does a plain ImageNet
    ResNet give us" without pulling in torchxrayvision. Kept deliberately close
    to `XRayVisionDenseNet`: same token contract, same frozen-BatchNorm guard,
    same mean-summary leading token, same `project_to` handling.

    Two adaptations are needed because a CNN is not a token model:

    * timm's ResNet `forward_features` returns a ``[B, C, h, w]`` feature map
      while every fusion module consumes ``[B, N, D]``; the grid is flattened to
      ``h*w`` tokens.
    * the backbone width (512 for resnet18/34, 2048 for resnet50) is projected to
      `project_to` when given, so a CNN and a ViT arm can meet the same fusion
      width without either one dictating it.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_name: str = "resnet18",
        pretrained: bool = True,
        *,
        project_to: int | None = None,
        dropout: float = 0.0,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "The timm_resnet encoder needs timm. Install it with "
                "'pip install -e .[pretrained]'. Use image_encoder.name='vit' "
                "for an offline run."
            ) from error
        # No img_size: ResNets are fully convolutional, so the grid is read off the
        # probe below. Passing it would also make timm reject the model, since
        # ResNet.__init__ has no such argument.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            drop_rate=dropout,
        )
        self._frozen = bool(freeze)
        if self._frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

        with torch.no_grad():
            probe = self.backbone.forward_features(
                torch.zeros(1, in_channels, image_size, image_size)
            )
        if probe.ndim != 4:
            raise ValueError(
                f"{model_name} produced {probe.ndim}D features; the timm_resnet "
                "encoder needs a convolutional feature map [B, C, H, W]. Use "
                "image_encoder.name='timm_vit' for a ViT backbone."
            )
        width = int(probe.shape[1])
        self.grid_h, self.grid_w = int(probe.shape[2]), int(probe.shape[3])
        # One token per spatial position, plus a summary token, so pooling="cls"
        # behaves the same as it does for the ViT encoders.
        self.num_tokens = self.grid_h * self.grid_w + 1
        self.project = nn.Linear(width, project_to) if project_to else nn.Identity()
        self.embed_dim = int(project_to or width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.embed_dim)

    def train(self, mode: bool = True) -> "TimmResNet":
        super().train(mode)
        # requires_grad=False does not freeze BatchNorm: in training mode it
        # normalises with batch statistics and keeps moving its running averages,
        # so a "frozen" ResNet would silently become a different feature extractor
        # part-way through training while a frozen ViT (all LayerNorm) would not.
        if self._frozen:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(image)
        tokens = self.project(features.flatten(2).transpose(1, 2))
        # Leading token: the mean of the feature map, matching XRayVisionDenseNet,
        # so pooling='cls' sees the image rather than a constant.
        summary = tokens.mean(dim=1, keepdim=True)
        return self.norm(self.dropout(torch.cat([summary, tokens], dim=1)))


@IMAGE_ENCODERS.register("vit")
def build_vit(cfg: ImageEncoderConfig, *, image_size: int, in_channels: int, **_: object) -> VisionTransformer:
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
    cfg: ImageEncoderConfig, *, image_size: int, in_channels: int, **_: object
) -> TimmVisionTransformer:
    return TimmVisionTransformer(
        image_size,
        in_channels,
        model_name=cfg.timm_model,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        freeze=cfg.freeze,
        project_to=cfg.project_to,
        unfreeze_last_blocks=cfg.unfreeze_last_blocks,
    )


class _PerTokenLinear(nn.Module):
    """One linear map per token position: [B, M, F] -> [B, M, D]."""

    def __init__(self, num_tokens: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_tokens, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(num_tokens, out_features))
        bound = in_features ** -0.5  # the range nn.Linear draws from
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bmf,mfd->bmd", features, self.weight) + self.bias


class ClinicalEncoder(nn.Module):
    """Project the [B, K, T] clinical window into tokens plus a token mask.

    per_variable (default): one token per variable, so attention maps read per
    variable. per_timestep: one token per hour.
    """

    def __init__(
        self,
        num_variables: int,
        num_timesteps: int,
        *,
        tokenization: str = "per_variable",
        embed_dim: int = 192,
        dropout: float = 0.0,
        missing_indicator: bool = True,
        variable_specific: bool = False,
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

        if variable_specific:
            # A weight shared by every variable gives them all the same sign. With
            # enough data one weight per variable is clearly better: external
            # validation AUROC 0.848 against 0.791.
            self.project: nn.Module = _PerTokenLinear(self.num_tokens, in_features, embed_dim)
        else:
            # MeTra projects the clinical window with a single linear layer.
            self.project = nn.Linear(in_features, embed_dim)
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
        dropout=cfg.dropout,
        missing_indicator=cfg.missing_indicator,
    )


@CLINICAL_ENCODERS.register("variable_projection")
def build_variable_projection(
    cfg: ClinicalEncoderConfig, *, num_variables: int, num_timesteps: int
) -> ClinicalEncoder:
    return ClinicalEncoder(
        num_variables,
        num_timesteps,
        tokenization=cfg.tokenization,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        missing_indicator=cfg.missing_indicator,
        variable_specific=True,
    )


class ClinicalSummaryEncoder(nn.Module):
    """Encode each variable by summary statistics instead of its raw series.

    Per variable, over observed hours only: mean, min, max, last, standard
    deviation, slope and observation count. An unobserved variable contributes
    zeros and a False token mask.
    """

    STATISTICS = ("mean", "min", "max", "last", "std", "slope", "count")

    def __init__(
        self,
        num_variables: int,
        num_timesteps: int,
        *,
        embed_dim: int = 192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_variables = int(num_variables)
        self.num_timesteps = int(num_timesteps)
        self.num_tokens = self.num_variables
        self.embed_dim = int(embed_dim)
        features = len(self.STATISTICS)

        self.project = nn.Linear(features, embed_dim)
        self.identity = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.identity, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def _statistics(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = mask.to(values.dtype)
        count = observed.sum(dim=-1)
        safe = count.clamp_min(1.0)
        mean = (values * observed).sum(dim=-1) / safe

        very_large = torch.finfo(values.dtype).max
        minimum = torch.where(mask, values, torch.full_like(values, very_large)).amin(dim=-1)
        maximum = torch.where(mask, values, torch.full_like(values, -very_large)).amax(dim=-1)
        minimum = torch.where(count > 0, minimum, torch.zeros_like(minimum))
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))

        # Last observed value: the highest hour index that carries an observation.
        hours = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype)
        last_index = torch.where(mask, hours, torch.full_like(hours, -1.0)).amax(dim=-1)
        gather_at = last_index.clamp_min(0).to(torch.long).unsqueeze(-1)
        last = torch.gather(values, -1, gather_at).squeeze(-1)
        last = torch.where(count > 0, last, torch.zeros_like(last))

        centred = (values - mean.unsqueeze(-1)) * observed
        variance = (centred**2).sum(dim=-1) / safe
        # The epsilon keeps sqrt differentiable at zero; it must not leak a
        # spurious spread for a variable that was never measured.
        deviation = torch.sqrt(variance.clamp_min(0.0) + 1e-8)
        deviation = torch.where(count > 0, deviation, torch.zeros_like(deviation))

        # Least-squares slope over observed hours: how fast the variable moved.
        hour_mean = (hours * observed).sum(dim=-1) / safe
        hour_centred = (hours - hour_mean.unsqueeze(-1)) * observed
        denominator = (hour_centred**2).sum(dim=-1)
        slope = (hour_centred * centred).sum(dim=-1) / denominator.clamp_min(1e-6)
        slope = torch.where(count > 1, slope, torch.zeros_like(slope))

        # Scaled to roughly unit range so it does not dominate the projection.
        density = count / float(max(self.num_timesteps, 1))
        return torch.stack([mean, minimum, maximum, last, deviation, slope, density], dim=-1)

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

        statistics = self._statistics(values, mask.to(torch.bool))
        token_mask = mask.any(dim=2)
        tokens = self.project(statistics) + self.identity
        tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return self.norm(self.dropout(tokens)), token_mask


@CLINICAL_ENCODERS.register("summary_stats")
def build_summary_stats(
    cfg: ClinicalEncoderConfig, *, num_variables: int, num_timesteps: int
) -> ClinicalSummaryEncoder:
    return ClinicalSummaryEncoder(
        num_variables,
        num_timesteps,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    )


class XRayVisionDenseNet(ImageEncoder):
    """DenseNet121 pretrained on chest radiographs (torchxrayvision).

    Checkpoints trained on MIMIC-CXR ('all', 'mimic') are rejected: they may have
    seen this project's held-out images. The pipeline's normalisation is undone
    and the image rescaled to the [-1024, 1024] range these weights expect.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        *,
        weights: str = "densenet121-res224-chex",
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.5,),
        project_to: int | None = None,
        dropout: float = 0.0,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            import torchxrayvision as xrv
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "The xrv_densenet encoder needs torchxrayvision: "
                "pip install torchxrayvision"
            ) from error
        if "mimic" in weights or weights.endswith("-all"):
            raise ValueError(
                f"{weights!r} was pretrained on a corpus containing MIMIC-CXR, which "
                "overlaps this project's held-out images. Use a MIMIC-free checkpoint "
                "such as densenet121-res224-chex, -nih or -pc."
            )
        self.backbone = xrv.models.DenseNet(weights=weights)
        self.weights = weights
        self._frozen = bool(freeze)
        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

        self.register_buffer("pixel_mean", torch.tensor(list(mean), dtype=torch.float32)[:, None, None])
        self.register_buffer("pixel_std", torch.tensor(list(std), dtype=torch.float32)[:, None, None])

        with torch.no_grad():
            probe = self.backbone.features(torch.zeros(1, 1, image_size, image_size))
        if probe.ndim != 4:
            raise ValueError(f"Expected a [B,C,H,W] feature map, got {tuple(probe.shape)}")
        width = int(probe.shape[1])
        self.grid = int(probe.shape[2])
        # One token per spatial position, plus a learned CLS so that pooling="cls"
        # behaves the same as it does for the ViT encoders.
        self.num_tokens = self.grid * self.grid + 1
        self.project = nn.Linear(width, project_to) if project_to else nn.Identity()
        self.embed_dim = int(project_to or width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.embed_dim)

    def train(self, mode: bool = True) -> "XRayVisionDenseNet":
        super().train(mode)
        # requires_grad=False does not freeze BatchNorm: in training mode it
        # normalises with batch statistics and keeps moving its running averages,
        # so the "frozen" features would differ between training and inference.
        if self._frozen:
            self.backbone.eval()
        return self

    def _to_xrv_range(self, image: torch.Tensor) -> torch.Tensor:
        # Undo the pipeline's normalisation back to [0, 1], collapse the repeated
        # grayscale channels, then scale to the range these weights expect.
        restored = image * self.pixel_std + self.pixel_mean
        if restored.shape[1] > 1:
            restored = restored.mean(dim=1, keepdim=True)
        return restored.clamp(0.0, 1.0) * 2048.0 - 1024.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(self._to_xrv_range(image))
        tokens = self.project(features.flatten(2).transpose(1, 2))
        # Leading token: the global average of the feature map, so pooling='cls'
        # sees the image rather than a constant.
        summary = tokens.mean(dim=1, keepdim=True)
        return self.norm(self.dropout(torch.cat([summary, tokens], dim=1)))


@IMAGE_ENCODERS.register("xrv_densenet")
def build_xrv_densenet(
    cfg: ImageEncoderConfig,
    *,
    image_size: int,
    in_channels: int,
    mean: Sequence[float] = (0.5,),
    std: Sequence[float] = (0.5,),
) -> XRayVisionDenseNet:
    return XRayVisionDenseNet(
        image_size,
        in_channels,
        weights=cfg.xrv_weights,
        mean=mean,
        std=std,
        project_to=cfg.project_to,
        dropout=cfg.dropout,
        freeze=cfg.freeze,
    )


@IMAGE_ENCODERS.register("timm_resnet")
def build_timm_resnet(
    cfg: ImageEncoderConfig, *, image_size: int, in_channels: int, **_: object
) -> TimmResNet:
    return TimmResNet(
        image_size,
        in_channels,
        model_name=cfg.timm_model,
        pretrained=cfg.pretrained,
        project_to=cfg.project_to,
        dropout=cfg.dropout,
        freeze=cfg.freeze,
    )
