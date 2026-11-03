"""The fusion contract: one signature, five strategies, swappable by name.

These tests are what make the plug-in claim real. If a new strategy passes them,
it can be dropped into any experiment without touching the trainer.
"""
from __future__ import annotations

import pytest
import torch

from bn5212_training.config import FusionConfig
from bn5212_training.fusion import FUSIONS, build_fusion

BATCH, IMAGE_TOKENS, CLINICAL_TOKENS, WIDTH = 3, 5, 4, 16


def _tokens():
    image = torch.randn(BATCH, IMAGE_TOKENS, WIDTH)
    clinical = torch.randn(BATCH, CLINICAL_TOKENS, WIDTH)
    mask = torch.ones(BATCH, CLINICAL_TOKENS, dtype=torch.bool)
    return image, clinical, mask


def _build(name: str, **extra):
    cfg = FusionConfig(name=name, embed_dim=WIDTH, depth=1, num_heads=2, **extra)
    return build_fusion(
        cfg, num_image_tokens=IMAGE_TOKENS, num_clinical_tokens=CLINICAL_TOKENS
    )


def test_every_registered_strategy_is_buildable():
    assert set(FUSIONS.names()) == {
        "clinical_only",
        "concat_mlp",
        "cross_attention_adapter",
        "image_only",
        "joint_self_attention",
    }


@pytest.mark.parametrize(
    "name", ["image_only", "clinical_only", "concat_mlp", "joint_self_attention", "cross_attention_adapter"]
)
def test_output_is_one_pooled_vector_per_sample(name):
    fusion = _build(name)
    image, clinical, mask = _tokens()
    output = fusion(image_tokens=image, clinical_tokens=clinical, clinical_mask=mask)
    assert output.shape == (BATCH, fusion.output_dim)
    assert torch.isfinite(output).all()


def test_unimodal_strategies_accept_a_missing_modality():
    image, clinical, mask = _tokens()
    assert _build("image_only")(image_tokens=image).shape == (BATCH, WIDTH)
    assert _build("clinical_only")(
        clinical_tokens=clinical, clinical_mask=mask
    ).shape == (BATCH, WIDTH)


def test_multimodal_strategies_refuse_a_missing_modality():
    image, clinical, mask = _tokens()
    for name in ("joint_self_attention", "cross_attention_adapter", "concat_mlp"):
        with pytest.raises(ValueError, match="requires"):
            _build(name)(image_tokens=image)
        with pytest.raises(ValueError, match="requires"):
            _build(name)(clinical_tokens=clinical, clinical_mask=mask)


def test_padded_clinical_tokens_do_not_change_other_samples():
    """A fully padded row must stay finite rather than producing NaN in softmax."""
    image, clinical, mask = _tokens()
    mask[0] = False
    for name in ("joint_self_attention", "cross_attention_adapter"):
        output = _build(name)(image_tokens=image, clinical_tokens=clinical, clinical_mask=mask)
        assert torch.isfinite(output).all()


def test_masked_mean_ignores_padded_positions():
    fusion = _build("clinical_only", pooling="mean")
    clinical = torch.zeros(1, CLINICAL_TOKENS, WIDTH)
    clinical[0, 0] = 1.0
    clinical[0, 1:] = 99.0
    mask = torch.zeros(1, CLINICAL_TOKENS, dtype=torch.bool)
    mask[0, 0] = True
    # Only the first token is valid, so the 99s must not reach the pooled vector.
    with_padding = fusion(clinical_tokens=clinical, clinical_mask=mask)
    only_valid = fusion(clinical_tokens=clinical[:, :1], clinical_mask=mask[:, :1])
    assert torch.allclose(with_padding, only_valid, atol=1e-5)


def test_joint_self_attention_validates_token_counts():
    fusion = _build("joint_self_attention")
    image, clinical, mask = _tokens()
    with pytest.raises(ValueError, match="image tokens"):
        fusion(
            image_tokens=image[:, :-1], clinical_tokens=clinical, clinical_mask=mask
        )


def test_unknown_strategy_lists_the_available_ones():
    with pytest.raises(KeyError, match="Available"):
        _build("no_such_fusion")


def test_concat_patch_pooling_ignores_cls_and_uses_image_regions():
    torch.manual_seed(5212)
    fusion = _build("concat_mlp", concat_image_pooling="patch_mean").eval()
    image, clinical, mask = _tokens()
    expected = fusion(image, clinical, mask)
    changed = image.clone()
    changed[:, 0] = torch.randn_like(changed[:, 0]) * 100
    torch.testing.assert_close(expected, fusion(changed, clinical, mask))
    changed[:, 1:] = torch.randn_like(changed[:, 1:])
    assert not torch.allclose(expected, fusion(changed, clinical, mask))
    with pytest.raises(ValueError, match="patches"):
        fusion(image[:, :1], clinical, mask)


def test_concat_rejects_unknown_image_pooling():
    with pytest.raises(ValueError, match="concat_image_pooling"):
        _build("concat_mlp", concat_image_pooling="unknown")


def test_document_concat_passes_raw_128_style_vector_to_head():
    fusion = _build("concat_mlp", concat_project=False, concat_image_pooling="patch_mean")
    image, clinical, mask = _tokens()
    mask[:, -1] = False
    expected = torch.cat([image[:, 1:].mean(1), clinical[:, :-1].mean(1)], dim=-1)
    torch.testing.assert_close(fusion(image, clinical, mask), expected)
    assert fusion.output_dim == 2 * WIDTH
    assert sum(p.numel() for p in fusion.parameters()) == 0


def test_image_only_patch_pooling_ignores_cls():
    fusion = _build("image_only", pooling="patch_mean").eval()
    image, _, _ = _tokens()
    expected = fusion(image_tokens=image)
    image[:, 0] = torch.randn_like(image[:, 0]) * 100
    torch.testing.assert_close(fusion(image_tokens=image), expected)
