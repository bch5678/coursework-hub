"""Declarative experiment configuration.

Every experiment in the project plan is one config file. Encoders, prediction
head, optimiser and data stay identical across experiments; only `fusion.name`
changes. That is what keeps MeTra and the proposed cross-attention comparable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

MODALITIES = ("clinical", "cxr")


def _require_keys(payload: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown {where} config keys: {sorted(unknown)}")


@dataclass(frozen=True)
class DataConfig:
    """Where the frozen dataset run lives and how batches are assembled."""

    run_dir: str
    data_pipeline_path: str | None = None
    batch_size: int | None = None          # None -> dataset_spec.loader.batch_size
    num_workers: int | None = None
    # Clinical branch
    clinical_provider: str = "synthetic"   # synthetic | table
    clinical_source: str | None = None     # path used by the table provider
    clinical_timesteps: int = 48
    clinical_window_hours: float | None = None  # None -> admittime..study_time
    clinical_signal: float = 0.0           # synthetic provider only; 0 = pure noise


@dataclass(frozen=True)
class ImageEncoderConfig:
    name: str = "vit"                      # vit | timm_vit | simple_cnn
    embed_dim: int = 192
    patch_size: int = 16
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pretrained: bool = False               # timm_vit only
    timm_model: str = "vit_base_patch16_224"
    freeze: bool = False


@dataclass(frozen=True)
class ClinicalEncoderConfig:
    name: str = "linear_projection"        # linear_projection | mlp_projection
    tokenization: str = "per_variable"     # per_variable | per_timestep
    embed_dim: int = 192
    hidden_dim: int = 256
    dropout: float = 0.0
    missing_indicator: bool = True         # append the mask as extra channels


@dataclass(frozen=True)
class FusionConfig:
    name: str = "image_only"
    embed_dim: int = 192
    depth: int = 2
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pooling: str = "cls"                   # cls | mean
    residual: bool = True                  # cross-attention variants
    direction: str = "clinical_to_image"   # cross_attention: also image_to_clinical, bidirectional


@dataclass(frozen=True)
class HeadConfig:
    hidden_dim: int = 0                    # 0 -> single linear layer
    dropout: float = 0.0
    num_outputs: int = 1                   # 1 -> BCEWithLogits, 2 -> CrossEntropy


@dataclass(frozen=True)
class OptimConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    scheduler: str = "cosine"              # cosine | none
    grad_clip: float = 1.0
    # Optimiser steps every N batches. Effective batch = batch_size * N,
    # which lets a small-VRAM card keep a usable batch size.
    grad_accumulation_steps: int = 1
    amp: bool = False
    positive_class_weight: float | str | None = None  # None | "auto" | explicit weight
    use_sample_weight: bool = True
    early_stopping_patience: int = 8
    min_lr: float = 0.0                    # cosine annealing floor (MeTra uses 1e-7)
    # MeTra drops a whole modality at random during multimodal training so that
    # the model cannot rely on one branch alone. Probability of zeroing the
    # modality's tokens for a training sample; 0 disables it.
    image_dropout_prob: float = 0.0
    clinical_dropout_prob: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    experiment: str
    modalities: tuple[str, ...]
    data: DataConfig
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    clinical_encoder: ClinicalEncoderConfig = field(default_factory=ClinicalEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 5212
    device: str = "auto"
    selection_metric: str = "auroc"        # auroc | auprc | loss
    output_dir: str = "outputs"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("At least one modality must be enabled")
        unknown = set(self.modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("Duplicate modality entries")
        if self.head.num_outputs not in (1, 2):
            raise ValueError("head.num_outputs must be 1 or 2")
        if self.selection_metric not in {"auroc", "auprc", "loss"}:
            raise ValueError("selection_metric must be auroc, auprc or loss")
        # One embedding width is shared by both encoders and the fusion module so
        # that image and clinical tokens can live in the same attention space.
        if self.uses_image and self.image_encoder.embed_dim != self.fusion.embed_dim:
            raise ValueError(
                "image_encoder.embed_dim must equal fusion.embed_dim "
                f"({self.image_encoder.embed_dim} != {self.fusion.embed_dim})"
            )
        if self.uses_clinical and self.clinical_encoder.embed_dim != self.fusion.embed_dim:
            raise ValueError(
                "clinical_encoder.embed_dim must equal fusion.embed_dim "
                f"({self.clinical_encoder.embed_dim} != {self.fusion.embed_dim})"
            )

    @property
    def uses_image(self) -> bool:
        return "cxr" in self.modalities

    @property
    def uses_clinical(self) -> bool:
        return "clinical" in self.modalities

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modalities"] = list(self.modalities)
        return payload

    def fingerprint(self) -> str:
        """Stable hash of the full configuration, recorded in the run manifest."""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_SECTIONS = {
    "data": DataConfig,
    "image_encoder": ImageEncoderConfig,
    "clinical_encoder": ClinicalEncoderConfig,
    "fusion": FusionConfig,
    "head": HeadConfig,
    "optim": OptimConfig,
}


def config_from_dict(payload: dict[str, Any]) -> TrainingConfig:
    """Build a config from a plain dict, rejecting unknown keys."""
    payload = dict(payload)
    top_level = {f.name for f in fields(TrainingConfig)}
    _require_keys(payload, top_level, "top-level")
    if "data" not in payload:
        raise ValueError("Config must contain a data section with run_dir")

    sections: dict[str, Any] = {}
    for name, section_type in _SECTIONS.items():
        raw = payload.pop(name, None) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config section {name} must be an object")
        _require_keys(raw, {f.name for f in fields(section_type)}, name)
        sections[name] = section_type(**raw)

    modalities = payload.pop("modalities", None)
    if modalities is None:
        raise ValueError("Config must list modalities, e.g. [\"cxr\"]")
    payload["modalities"] = tuple(modalities)
    return TrainingConfig(**payload, **sections)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    """Read a JSON config file, applying shallow per-section overrides."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            payload[section] = value
            continue
        if section not in _SECTIONS:
            raise ValueError(f"Cannot override unknown section {section!r}")
        payload.setdefault(section, {})[key] = value
    return config_from_dict(payload)
