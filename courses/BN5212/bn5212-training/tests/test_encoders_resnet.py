"""The timm ResNet image encoder.

Kept in its own module rather than appended to `test_encoders.py` so that adding
a backbone arm never touches another author's test file. `test_encoders.py`
skips itself wholesale when torchxrayvision is missing (a module-level
`pytest.importorskip`), so anything placed there would not run on a machine
without that optional dependency anyway.

timm is optional, so this module skips when it is absent. Every encoder below is
built with `pretrained=False`: these assert wiring and shapes, and nothing here
should reach the network.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")

from bn5212_training.config import FusionConfig, ImageEncoderConfig  # noqa: E402
from bn5212_training.encoders import TimmResNet  # noqa: E402
from bn5212_training.fusion import build_fusion  # noqa: E402
from bn5212_training.registry import IMAGE_ENCODERS  # noqa: E402


def _encoder(image_size: int = 64, **overrides) -> TimmResNet:
    """Small inputs on purpose: these assert wiring, not accuracy."""
    kwargs = {
        "in_channels": 3,
        "model_name": "resnet18",
        "pretrained": False,
        "image_size": image_size,
    }
    kwargs.update(overrides)
    return TimmResNet(**kwargs)


def test_timm_resnet_is_registered():
    assert "timm_resnet" in IMAGE_ENCODERS


def test_timm_resnet_builds_from_the_config_dataclass():
    cfg = ImageEncoderConfig(name="timm_resnet", timm_model="resnet18", project_to=32)
    encoder = IMAGE_ENCODERS.build(cfg.name, cfg, image_size=64, in_channels=3)
    assert isinstance(encoder, TimmResNet)
    assert encoder.embed_dim == 32


def test_timm_resnet_emits_the_token_contract():
    encoder = _encoder(project_to=48)
    encoder.eval()
    with torch.no_grad():
        tokens = encoder(torch.zeros(2, 3, 64, 64))
    assert tokens.ndim == 3, "fusion modules need [B, N, D], not a feature map"
    assert tokens.shape == (2, encoder.num_tokens, 48)
    assert torch.isfinite(tokens).all()


def test_timm_resnet_flattens_the_feature_grid():
    small = _encoder(image_size=64)
    assert (small.grid_h, small.grid_w) == (2, 2), "64px resnet18 downsamples 32x"
    assert small.num_tokens == small.grid_h * small.grid_w + 1

    # The 224 setting the ICU configs use.
    reference = _encoder(image_size=224)
    assert (reference.grid_h, reference.grid_w) == (7, 7)
    assert reference.num_tokens == 50


def test_timm_resnet_projects_to_the_requested_width():
    """project_to is what lets a CNN and a ViT meet the same fusion width."""
    encoder = _encoder(project_to=32)
    assert encoder.embed_dim == 32
    assert not isinstance(encoder.project, torch.nn.Identity)

    # resnet50 ends on 2048 channels; the projection must not assume 512.
    wide = _encoder(model_name="resnet50", project_to=32)
    wide.eval()
    with torch.no_grad():
        assert wide(torch.zeros(1, 3, 64, 64)).shape == (1, wide.num_tokens, 32)


def test_timm_resnet_without_a_projection_keeps_the_backbone_width():
    encoder = _encoder()
    assert isinstance(encoder.project, torch.nn.Identity)
    assert encoder.embed_dim == 512  # resnet18's final width


def test_timm_resnet_rejects_a_vit_model():
    """A ViT returns [B, N, D] with no spatial axis, so flattening it is wrong.

    image_size must match the ViT's own patch grid, otherwise timm's patch_embed
    asserts first and this guard never runs.
    """
    with pytest.raises(ValueError, match="convolutional feature map"):
        _encoder(image_size=224, model_name="vit_base_patch16_224")


def test_resnet_tokens_drive_the_joint_self_attention_fusion():
    """End to end through the fusion the arm actually runs."""
    width = 32
    encoder = _encoder(project_to=width)
    encoder.eval()
    fusion = build_fusion(
        FusionConfig(name="joint_self_attention", embed_dim=width, depth=1, num_heads=2),
        num_image_tokens=encoder.num_tokens,
        num_clinical_tokens=3,
    )
    with torch.no_grad():
        out = fusion(
            image_tokens=encoder(torch.zeros(2, 3, 64, 64)),
            clinical_tokens=torch.randn(2, 3, width),
            clinical_mask=torch.ones(2, 3, dtype=torch.bool),
        )
    assert out.shape == (2, width)
    assert torch.isfinite(out).all()


def test_a_frozen_resnet_backbone_stays_in_inference_mode_during_training():
    """requires_grad=False alone does not stop BatchNorm from re-fitting.

    On real chest radiographs the drift is large enough to replace the pretrained
    features, and a frozen ViT (all LayerNorm) would not drift at all, so the two
    backbone arms would not share the same meaning of "frozen".
    """
    encoder = _encoder(freeze=True, project_to=32)
    batches = [
        m for m in encoder.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)
    ]
    assert batches, "resnet18 is expected to carry BatchNorm layers"
    before = [(b.running_mean.clone(), b.running_var.clone()) for b in batches]

    encoder.train()  # what trainer.train_one_epoch() does
    assert encoder.training, "the encoder itself stays in training mode"
    assert not encoder.backbone.training, "but the frozen backbone must not"
    for _ in range(3):
        encoder(torch.randn(2, 3, 64, 64))

    for layer, (mean, var) in zip(batches, before):
        assert torch.equal(layer.running_mean, mean)
        assert torch.equal(layer.running_var, var)


def test_an_unfrozen_resnet_backbone_still_updates_batch_norm():
    """The guard must be tied to freeze, not applied unconditionally."""
    encoder = _encoder(freeze=False)
    batches = [
        m for m in encoder.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)
    ]
    before = [b.running_mean.clone() for b in batches]

    encoder.train()
    assert encoder.backbone.training
    for _ in range(3):
        encoder(torch.randn(2, 3, 64, 64))

    assert any(not torch.equal(b.running_mean, m) for b, m in zip(batches, before))


def test_a_frozen_resnet_keeps_its_projection_trainable():
    """Freezing the backbone must not freeze the adapter added on top of it."""
    encoder = _encoder(freeze=True, project_to=32)
    assert not any(p.requires_grad for p in encoder.backbone.parameters())
    assert any(p.requires_grad for p in encoder.project.parameters())
