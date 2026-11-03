"""Patch-only attention, missing queries and the existing model/checkpoint API."""
from __future__ import annotations

import pytest
import torch

from bn5212_training.config import FusionConfig, config_from_dict
from bn5212_training.fusion import build_fusion
from bn5212_training.model import build_model, load_checkpoint, save_checkpoint


def _adapter(**overrides):
    options = dict(name="cross_attention_adapter", embed_dim=16, depth=2, num_heads=4)
    options.update(overrides)
    return build_fusion(FusionConfig(**options))


def _inputs():
    return (
        torch.randn(3, 197, 16),
        torch.randn(3, 17, 16),
        torch.ones(3, 17, dtype=torch.bool),
    )


def test_patch_attention_excludes_cls_and_depends_on_both_modalities():
    torch.manual_seed(5212)
    adapter = _adapter().eval()
    image, clinical, mask = _inputs()
    expected = adapter(image, clinical, mask)
    changed_cls = image.clone()
    changed_cls[:, 0] = torch.randn_like(changed_cls[:, 0]) * 100
    torch.testing.assert_close(expected, adapter(changed_cls, clinical, mask))
    assert not torch.allclose(expected, adapter(torch.randn_like(image), clinical, mask))
    assert not torch.allclose(expected, adapter(image, torch.randn_like(clinical), mask))
    assert expected.shape == (3, 16)


def test_missing_queries_ignore_arbitrary_values_including_nan():
    adapter = _adapter().eval()
    image, clinical, mask = _inputs()
    mask[0] = False
    mask[1, 3:] = False
    expected = adapter(image, clinical, mask)
    clinical[~mask] = float("nan")
    actual = adapter(image, clinical, mask)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    compact = adapter(image[1:2], clinical[1:2, :3], mask[1:2, :3])
    torch.testing.assert_close(actual[1:2], compact)


def test_all_missing_stays_zero_after_learning_normalization_bias():
    adapter = _adapter()
    with torch.no_grad():
        adapter.norm.bias.fill_(2.0)
    image, clinical, mask = _inputs()
    mask[:] = False
    assert torch.equal(adapter(image, clinical, mask), torch.zeros(3, 16))


def test_optional_recording_masks_queries_detaches_and_clears_maps():
    adapter = _adapter().eval()
    image, clinical, mask = _inputs()
    mask[0] = False
    mask[1, 3:] = False
    assert adapter.last_attention() is None
    adapter.record_attention = True
    adapter(image, clinical, mask)
    weights = adapter.last_attention()
    assert weights.shape == (3, 4, 17, 196)
    assert not weights.requires_grad
    expected_mass = mask[:, None, :].expand(3, 4, 17).float()
    torch.testing.assert_close(weights.sum(dim=-1), expected_mass)
    adapter.record_attention = False
    adapter(image, clinical, mask)
    assert adapter.last_attention() is None


@pytest.mark.parametrize("residual", [True, False])
def test_backward_trains_adapter_and_respects_masks(residual):
    adapter = _adapter(residual=residual)
    image, clinical, mask = _inputs()
    image.requires_grad_()
    clinical.requires_grad_()
    mask[0] = False
    mask[1, 3:] = False
    output = adapter(image, clinical, mask)
    (output * torch.randn_like(output)).sum().backward()
    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert image.grad[:, 1:].abs().sum() > 0
    assert clinical.grad[mask].abs().sum() > 0
    assert image.grad[:, 0].count_nonzero() == 0
    assert clinical.grad[~mask].count_nonzero() == 0


@pytest.mark.parametrize(
    "options,match",
    [
        ({"depth": 0}, "depth"),
        ({"num_heads": 3}, "divisible"),
        ({"num_heads": 0}, "divisible"),
        ({"mlp_ratio": 0}, "FFN"),
        ({"dropout": 1.1}, "dropout"),
        ({"direction": "bidirectional"}, "clinical_to_image"),
    ],
)
def test_invalid_adapter_configuration_fails_early(options, match):
    with pytest.raises(ValueError, match=match):
        _adapter(**options)


@pytest.mark.parametrize("case", ["width", "batch", "patches", "clinical", "mask"])
def test_invalid_input_shapes_fail_clearly(case):
    adapter = _adapter()
    image, clinical, mask = _inputs()
    if case == "width":
        image = image[:, :, :-1]
    elif case == "batch":
        image = image[:1]
    elif case == "patches":
        image = image[:, :1]
    elif case == "clinical":
        clinical = clinical[:, :0]
    else:
        mask = mask[:, :-1]
    with pytest.raises(ValueError):
        adapter(image, clinical, mask)


def test_no_mask_means_every_clinical_token_is_valid():
    adapter = _adapter().eval()
    image, clinical, mask = _inputs()
    torch.testing.assert_close(adapter(image, clinical), adapter(image, clinical, mask))


def test_model_checkpoint_and_frozen_clinical_encoder(tmp_path):
    cfg = config_from_dict({
        "experiment": "adapter_round_trip",
        "modalities": ["clinical", "cxr"],
        "data": {"run_dir": str(tmp_path)},
        "image_encoder": {
            "embed_dim": 16, "patch_size": 16, "depth": 1, "num_heads": 4,
        },
        "clinical_encoder": {
            "embed_dim": 16, "freeze": True, "pretrained": "unused_fixture.pt",
        },
        "fusion": {
            "name": "cross_attention_adapter", "embed_dim": 16, "depth": 1,
            "num_heads": 4,
        },
    })
    geometry = dict(image_size=32, channels=1, num_variables=3, num_timesteps=8)
    model = build_model(cfg, **geometry, load_pretrained=False)
    batch = {
        "image": torch.randn(2, 1, 32, 32),
        "clinical": torch.randn(2, 3, 8),
        "clinical_mask": torch.ones(2, 3, 8, dtype=torch.bool),
    }
    model.train()
    assert not model.clinical_encoder.training
    model(batch).square().sum().backward()
    assert all(parameter.grad is None for parameter in model.clinical_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.fusion.parameters())
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    model.eval()
    with torch.no_grad():
        expected = model.predict_logits(batch)
    checkpoint = save_checkpoint(tmp_path / "adapter.pt", model, epoch=1, geometry=geometry)
    restored, _ = load_checkpoint(checkpoint)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.predict_logits(batch), expected)


def test_attention_receives_clinical_queries_and_image_patch_keys_values():
    adapter = _adapter(depth=1).eval()
    image, clinical, mask = _inputs()
    captured = []
    handle = adapter.blocks[0].attention.register_forward_pre_hook(
        lambda module, args: captured.append(args)
    )
    try:
        adapter(image, clinical, mask)
    finally:
        handle.remove()
    query, key, value = captured[0]
    block = adapter.blocks[0]
    torch.testing.assert_close(query, block.query_norm(clinical))
    torch.testing.assert_close(key, block.context_norm(image[:, 1:]))
    torch.testing.assert_close(value, key)
