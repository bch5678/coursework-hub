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
        "cross_attention",
        "image_only",
        "joint_self_attention",
    }


@pytest.mark.parametrize(
    "name", ["image_only", "clinical_only", "concat_mlp", "joint_self_attention", "cross_attention"]
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
    for name in ("joint_self_attention", "cross_attention", "concat_mlp"):
        with pytest.raises(ValueError, match="requires"):
            _build(name)(image_tokens=image)
        with pytest.raises(ValueError, match="requires"):
            _build(name)(clinical_tokens=clinical, clinical_mask=mask)


def test_padded_clinical_tokens_do_not_change_other_samples():
    """A fully padded row must stay finite rather than producing NaN in softmax."""
    image, clinical, mask = _tokens()
    mask[0] = False
    for name in ("joint_self_attention", "cross_attention"):
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


def test_bidirectional_cross_attention_doubles_the_output_width():
    assert _build("cross_attention", direction="clinical_to_image").output_dim == WIDTH
    assert _build("cross_attention", direction="image_to_clinical").output_dim == WIDTH
    assert _build("cross_attention", direction="bidirectional").output_dim == 2 * WIDTH


def test_cross_attention_records_weights_for_rq3():
    fusion = _build("cross_attention")
    image, clinical, mask = _tokens()
    assert fusion.last_attention() is None  # not recorded during training

    fusion.record_attention = True
    fusion(image_tokens=image, clinical_tokens=clinical, clinical_mask=mask)
    weights = fusion.last_attention()
    # [B, heads, clinical tokens, image tokens] -- one row per clinical variable.
    assert weights is not None
    assert weights.shape[0] == BATCH
    assert weights.shape[-2:] == (CLINICAL_TOKENS, IMAGE_TOKENS)
    attended = weights.sum(dim=-1)
    assert torch.allclose(attended, torch.ones_like(attended), atol=1e-4)


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
