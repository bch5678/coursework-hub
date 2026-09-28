"""Assemble encoders, fusion and prediction head into one model.

The model also satisfies the benchmark project's ModelAdapter protocol
(`eval()` plus `predict_logits(batch)`), so benchmark-evaluation can run
inference against a checkpoint from this framework without an extra shim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .clinical import ClinicalNormalizer
from .config import TrainingConfig, config_from_dict
from .encoders import ClinicalEncoder, ImageEncoder
from .fusion import FusionModule, build_fusion
from .registry import CLINICAL_ENCODERS, IMAGE_ENCODERS


class PredictionHead(nn.Module):
    """Maps the pooled patient representation to mortality logits."""

    def __init__(
        self, input_dim: int, num_outputs: int = 1, hidden_dim: int = 0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if hidden_dim > 0:
            layers += [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            input_dim = hidden_dim
        elif dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_dim, num_outputs))
        self.net = nn.Sequential(*layers)
        self.num_outputs = int(num_outputs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        # [B] for the single-logit BCE setup, [B, 2] for the cross-entropy setup.
        return logits.squeeze(-1) if self.num_outputs == 1 else logits


class BN5212Model(nn.Module):
    def __init__(
        self,
        cfg: TrainingConfig,
        *,
        image_size: int = 224,
        channels: int = 1,
        num_variables: int = 0,
        num_timesteps: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder: ImageEncoder | None = None
        self.clinical_encoder: ClinicalEncoder | None = None
        num_image_tokens = 0
        num_clinical_tokens = 0

        if cfg.uses_image:
            self.image_encoder = IMAGE_ENCODERS.build(
                cfg.image_encoder.name,
                cfg.image_encoder,
                image_size=image_size,
                in_channels=channels,
            )
            if self.image_encoder.embed_dim != cfg.fusion.embed_dim:
                raise ValueError(
                    f"Image encoder produces width {self.image_encoder.embed_dim} but "
                    f"fusion.embed_dim is {cfg.fusion.embed_dim}. Set both encoder and "
                    "fusion embed_dim to the backbone width (768 for ViT-B/16)."
                )
            num_image_tokens = self.image_encoder.num_tokens

        if cfg.uses_clinical:
            if num_variables <= 0 or num_timesteps <= 0:
                raise ValueError(
                    "The clinical branch needs num_variables and num_timesteps > 0; "
                    "no clinical feature provider was supplied."
                )
            self.clinical_encoder = CLINICAL_ENCODERS.build(
                cfg.clinical_encoder.name,
                cfg.clinical_encoder,
                num_variables=num_variables,
                num_timesteps=num_timesteps,
            )
            num_clinical_tokens = self.clinical_encoder.num_tokens

        self.fusion: FusionModule = build_fusion(
            cfg.fusion,
            num_image_tokens=num_image_tokens,
            num_clinical_tokens=num_clinical_tokens,
        )
        if self.fusion.uses_image and not cfg.uses_image:
            raise ValueError(
                f"fusion {cfg.fusion.name!r} needs the cxr modality, which is not enabled"
            )
        if self.fusion.uses_clinical and not cfg.uses_clinical:
            raise ValueError(
                f"fusion {cfg.fusion.name!r} needs the clinical modality, which is not enabled"
            )

        self.head = PredictionHead(
            self.fusion.output_dim,
            num_outputs=cfg.head.num_outputs,
            hidden_dim=cfg.head.hidden_dim,
            dropout=cfg.head.dropout,
        )
        self.num_image_tokens = num_image_tokens
        self.num_clinical_tokens = num_clinical_tokens

    def _drop_modality(
        self, tokens: torch.Tensor, probability: float, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zero a modality for randomly chosen samples (MeTra's vision dropout).

        Applied in training mode only, so validation and test inference always see
        every modality that the experiment declares.
        """
        if not self.training or probability <= 0.0:
            return tokens, mask
        keep = (torch.rand(tokens.shape[0], device=tokens.device) >= probability).to(tokens.dtype)
        tokens = tokens * keep[:, None, None]
        if mask is not None:
            mask = mask & keep.bool()[:, None]
        return tokens, mask

    def encode(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        image_tokens = None
        clinical_tokens = None
        clinical_mask = None

        if self.image_encoder is not None:
            if "image" not in batch:
                raise KeyError("Batch has no 'image' but the cxr modality is enabled")
            image_tokens = self.image_encoder(batch["image"])
            image_tokens, _ = self._drop_modality(
                image_tokens, self.cfg.optim.image_dropout_prob
            )

        if self.clinical_encoder is not None:
            if "clinical" not in batch or "clinical_mask" not in batch:
                raise KeyError(
                    "Batch has no 'clinical'/'clinical_mask' but the clinical modality "
                    "is enabled. Check data.clinical_provider."
                )
            clinical_tokens, clinical_mask = self.clinical_encoder(
                batch["clinical"], batch["clinical_mask"]
            )
            clinical_tokens, clinical_mask = self._drop_modality(
                clinical_tokens, self.cfg.optim.clinical_dropout_prob, clinical_mask
            )

        return image_tokens, clinical_tokens, clinical_mask

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        image_tokens, clinical_tokens, clinical_mask = self.encode(batch)
        fused = self.fusion(
            image_tokens=image_tokens,
            clinical_tokens=clinical_tokens,
            clinical_mask=clinical_mask,
        )
        return self.head(fused)

    # --- benchmark-evaluation ModelAdapter protocol ---------------------------
    def predict_logits(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return self.forward(batch)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def parameter_counts(self) -> dict[str, int]:
        """Per-branch parameter counts, for the size-vs-performance comparison."""

        def count(module: nn.Module | None) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        return {
            "image_encoder": count(self.image_encoder),
            "clinical_encoder": count(self.clinical_encoder),
            "fusion": count(self.fusion),
            "head": count(self.head),
            "total": count(self),
        }


def build_model(
    cfg: TrainingConfig,
    *,
    image_size: int = 224,
    channels: int = 1,
    num_variables: int = 0,
    num_timesteps: int = 0,
) -> BN5212Model:
    return BN5212Model(
        cfg,
        image_size=image_size,
        channels=channels,
        num_variables=num_variables,
        num_timesteps=num_timesteps,
    )


def save_checkpoint(
    path: str | Path,
    model: BN5212Model,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None = None,
    normalizer: ClinicalNormalizer | None = None,
    geometry: Mapping[str, int] | None = None,
) -> Path:
    """Persist everything needed to rebuild the model without the config file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "bn5212-training/1",
            "state_dict": model.state_dict(),
            "config": model.cfg.to_dict(),
            "geometry": dict(geometry or {}),
            "epoch": int(epoch),
            "metrics": dict(metrics or {}),
            "clinical_normalizer": None if normalizer is None else normalizer.to_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[BN5212Model, dict[str, Any]]:
    """Rebuild the model and return it with the checkpoint payload."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != "bn5212-training/1":
        raise ValueError(
            f"{Path(path).name} is not a bn5212-training checkpoint "
            f"(format={payload.get('format')!r})"
        )
    cfg = config_from_dict(payload["config"])
    geometry = payload.get("geometry") or {}
    model = build_model(
        cfg,
        image_size=int(geometry.get("image_size", 224)),
        channels=int(geometry.get("channels", 1)),
        num_variables=int(geometry.get("num_variables", 0)),
        num_timesteps=int(geometry.get("num_timesteps", 0)),
    )
    model.load_state_dict(payload["state_dict"])
    return model, payload
