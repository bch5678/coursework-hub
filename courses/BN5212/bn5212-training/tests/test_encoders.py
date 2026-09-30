"""The image-encoder contract, and the CNN arm in particular.

`timm_resnet` exists so that "is the transformer backbone doing the work?" can be
answered by changing one config field. That claim only holds if the encoder
produces exactly the token sequence the fusion modules already consume and if it
leaves `fusion.embed_dim` where the ViT arm had it. Both are asserted here.

timm is an optional dependency (`pip install -e .[pretrained]`), so the whole
module skips when it is absent. Every encoder below is built with
`pretrained=False`: these tests are about shapes and wiring, and nothing here
should reach the network.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")

from bn5212_training.config import ImageEncoderConfig  # noqa: E402
from bn5212_training.encoders import IMAGE_ENCODERS, TimmResNet  # noqa: E402
from bn5212_training.fusion import build_fusion  # noqa: E402
from bn5212_training.config import FusionConfig  # noqa: E402

WIDTH = 768


def _encoder(image_size: int = 64, **overrides) -> TimmResNet:
    kwargs = {
        "in_channels": 3,
        "model_name": "resnet18",
        "pretrained": False,
        "embed_dim": WIDTH,
        "image_size": image_size,
    }
    kwargs.update(overrides)
    return TimmResNet(**kwargs)


def test_timm_resnet_is_registered():
    assert "timm_resnet" in IMAGE_ENCODERS.names()


def test_timm_resnet_builds_from_the_config_dataclass():
    cfg = ImageEncoderConfig(name="timm_resnet", timm_model="resnet18", embed_dim=WIDTH)
    encoder = IMAGE_ENCODERS.build(cfg.name, cfg, image_size=64, in_channels=3)
    assert isinstance(encoder, TimmResNet)
    assert encoder.embed_dim == WIDTH


def test_timm_resnet_emits_the_token_contract():
    encoder = _encoder()
    encoder.eval()
    with torch.no_grad():
        tokens = encoder(torch.zeros(2, 3, 64, 64))
    assert tokens.ndim == 3, "fusion modules need [B, N, D], not a feature map"
    assert tokens.shape == (2, encoder.num_tokens, WIDTH)
    assert torch.isfinite(tokens).all()


def test_timm_resnet_flattens_the_feature_grid():
    """num_tokens must follow the CNN grid, and a 224 input gives the 7x7 + CLS."""
    small = _encoder(image_size=64)
    assert small.num_patches == 2 * 2, "64px resnet18 downsamples 32x"
    assert small.num_tokens == small.num_patches + 1

    reference = _encoder(image_size=224)
    assert reference.num_patches == 7 * 7
    assert reference.num_tokens == 50


def test_timm_resnet_projects_up_to_the_fusion_width():
    """The whole point of the projection: fusion.embed_dim stays at the ViT value.

    resnet18 ends on 512 channels. Without the projection this encoder would
    report 512, `TrainingConfig` would reject the ViT arm's `fusion.embed_dim=768`
    and the backbone comparison would silently also be a fusion-width comparison.
    """
    encoder = _encoder()
    assert encoder.embed_dim == WIDTH
    assert not isinstance(encoder.project, torch.nn.Identity), "expected a 512->768 projection"


def test_timm_resnet_keeps_a_wider_backbone_projectable():
    """resnet50 ends on 2048 channels; the projection must not assume 512."""
    encoder = _encoder(model_name="resnet50")
    assert encoder.embed_dim == WIDTH
    encoder.eval()
    with torch.no_grad():
        tokens = encoder(torch.zeros(1, 3, 64, 64))
    assert tokens.shape == (1, encoder.num_tokens, WIDTH)


def test_timm_resnet_rejects_a_vit_model():
    """A ViT returns [B, N, D] with no spatial axis, so flattening it is wrong.

    image_size must match the ViT's own patch grid, otherwise timm's patch_embed
    asserts first and the guard below never runs.
    """
    with pytest.raises(ValueError, match="convolutional feature map"):
        _encoder(image_size=224, model_name="vit_base_patch16_224")


def test_resnet_tokens_drive_the_joint_self_attention_fusion():
    """End to end through MeTra's fusion, which is what the arm actually runs."""
    encoder = _encoder()
    encoder.eval()
    fusion = build_fusion(
        FusionConfig(name="joint_self_attention", embed_dim=WIDTH, depth=1, num_heads=2),
        num_image_tokens=encoder.num_tokens,
        num_clinical_tokens=3,
    )
    with torch.no_grad():
        image_tokens = encoder(torch.zeros(2, 3, 64, 64))
        out = fusion(
            image_tokens=image_tokens,
            clinical_tokens=torch.randn(2, 3, WIDTH),
            clinical_mask=torch.ones(2, 3, dtype=torch.bool),
        )
    assert out.shape == (2, WIDTH)
    assert torch.isfinite(out).all()


def test_freeze_leaves_the_projection_trainable():
    """Freezing the backbone must not freeze the adapter added on top of it."""
    encoder = _encoder(freeze=True)
    assert not any(p.requires_grad for p in encoder.backbone.parameters())
    assert encoder.cls_token.requires_grad
    if not isinstance(encoder.project, torch.nn.Identity):
        assert any(p.requires_grad for p in encoder.project.parameters())


def _batch_norm_layers(module):
    return [m for m in module.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def test_freeze_pins_batch_norm_statistics():
    """A frozen backbone must not keep re-fitting BatchNorm to our data.

    `requires_grad_(False)` alone does not achieve this: BatchNorm in training
    mode normalises with the current batch statistics and updates its running
    estimates regardless of gradients. On real chest radiographs the drift is
    large enough to replace the pretrained features (cosine 0.33 after 40 steps
    at 8 images per batch), and a ViT -- all LayerNorm -- would not drift at all,
    so the two backbone arms would not be comparable.
    """
    encoder = _encoder(freeze=True)
    layers = _batch_norm_layers(encoder.backbone)
    assert layers, "resnet18 is expected to carry BatchNorm layers"
    before = [(b.running_mean.clone(), b.running_var.clone()) for b in layers]

    encoder.train()  # what trainer.train_one_epoch() does
    assert encoder.training, "the encoder itself stays in training mode"
    assert not encoder.backbone.training, "but the frozen backbone must not"
    for _ in range(3):
        encoder(torch.randn(4, 3, 64, 64))

    for layer, (mean, var) in zip(layers, before):
        assert torch.equal(layer.running_mean, mean)
        assert torch.equal(layer.running_var, var)


def test_an_unfrozen_backbone_still_updates_batch_norm():
    """The guard must be tied to `freeze`, not applied unconditionally."""
    encoder = _encoder(freeze=False)
    layers = _batch_norm_layers(encoder.backbone)
    before = [b.running_mean.clone() for b in layers]

    encoder.train()
    assert encoder.backbone.training
    for _ in range(3):
        encoder(torch.randn(4, 3, 64, 64))

    assert any(not torch.equal(b.running_mean, m) for b, m in zip(layers, before))


def test_eval_mode_is_untouched_by_the_freeze_guard():
    """`eval()` must still propagate: only the frozen backbone is pinned."""
    encoder = _encoder(freeze=True)
    encoder.eval()
    assert not encoder.training
    assert not encoder.backbone.training
