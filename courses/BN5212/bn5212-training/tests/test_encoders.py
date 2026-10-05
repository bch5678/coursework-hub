"""Encoder behaviour: the projection, partial unfreezing and summary statistics.

The summary statistics are computed by hand in torch over a masked window, so
they are pinned against values worked out on paper. A silent error there would
change every clinical result without failing anything else.
"""
from __future__ import annotations

import pytest
import torch

from bn5212_training.config import AugmentationConfig, ClinicalEncoderConfig, ImageEncoderConfig
from bn5212_training.encoders import ClinicalSummaryEncoder
from bn5212_training.registry import CLINICAL_ENCODERS, IMAGE_ENCODERS


# --- clinical summary statistics -----------------------------------------

def _encoder(num_variables=1, num_timesteps=6, embed_dim=8):
    return ClinicalSummaryEncoder(num_variables, num_timesteps, embed_dim=embed_dim)


def test_statistics_match_hand_computed_values():
    encoder = _encoder()
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, True, True, False, False]]])
    stats = encoder._statistics(values, mask)[0, 0]
    names = dict(zip(ClinicalSummaryEncoder.STATISTICS, stats.tolist()))

    assert names["mean"] == pytest.approx(2.5)
    assert names["min"] == pytest.approx(1.0)
    assert names["max"] == pytest.approx(4.0)
    assert names["last"] == pytest.approx(4.0)      # hour 3, the latest observed
    assert names["std"] == pytest.approx(1.1180, abs=1e-3)   # population sd of 1..4
    assert names["slope"] == pytest.approx(1.0, abs=1e-4)    # +1 per hour
    assert names["count"] == pytest.approx(4 / 6)   # density over the window


def test_unobserved_hours_do_not_enter_the_statistics():
    """A gap in the middle must not be read as a zero measurement."""
    encoder = _encoder()
    with_gap = torch.tensor([[[10.0, 0.0, 10.0, 0.0, 0.0, 0.0]]])
    gap_mask = torch.tensor([[[True, False, True, False, False, False]]])
    stats = encoder._statistics(with_gap, gap_mask)[0, 0]
    names = dict(zip(ClinicalSummaryEncoder.STATISTICS, stats.tolist()))
    assert names["mean"] == pytest.approx(10.0)
    assert names["min"] == pytest.approx(10.0)
    assert names["std"] == pytest.approx(0.0, abs=1e-3)


def test_last_is_the_latest_observation_not_the_last_column():
    encoder = _encoder()
    values = torch.tensor([[[5.0, 7.0, 0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False, False, False, False]]])
    stats = encoder._statistics(values, mask)[0, 0]
    assert dict(zip(ClinicalSummaryEncoder.STATISTICS, stats.tolist()))["last"] == pytest.approx(7.0)


def test_a_never_observed_variable_yields_zeros_and_a_false_token():
    encoder = _encoder(num_variables=2)
    values = torch.zeros(1, 2, 6)
    values[0, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.zeros(1, 2, 6, dtype=torch.bool)
    mask[0, 0, :3] = True

    stats = encoder._statistics(values, mask)
    assert torch.allclose(stats[0, 1], torch.zeros(len(ClinicalSummaryEncoder.STATISTICS)))
    _tokens, token_mask = encoder(values, mask)
    assert token_mask[0, 0].item() is True
    assert token_mask[0, 1].item() is False


def test_slope_is_zero_with_a_single_observation():
    """One point defines no trend; a fitted slope there would be noise."""
    encoder = _encoder()
    values = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, False, False, False, False, False]]])
    stats = encoder._statistics(values, mask)[0, 0]
    assert dict(zip(ClinicalSummaryEncoder.STATISTICS, stats.tolist()))["slope"] == 0.0


def test_summary_encoder_emits_one_token_per_variable():
    cfg = ClinicalEncoderConfig(name="summary_stats", embed_dim=16)
    encoder = CLINICAL_ENCODERS.build("summary_stats", cfg, num_variables=5, num_timesteps=12)
    values = torch.randn(3, 5, 12)
    mask = torch.rand(3, 5, 12) > 0.5
    tokens, token_mask = encoder(values, mask)
    assert tokens.shape == (3, 5, 16)
    assert token_mask.shape == (3, 5)
    assert encoder.num_tokens == 5


def test_summary_encoder_output_is_finite_for_an_empty_window():
    encoder = _encoder(num_variables=3)
    tokens, token_mask = encoder(torch.zeros(2, 3, 6), torch.zeros(2, 3, 6, dtype=torch.bool))
    assert torch.isfinite(tokens).all()
    assert not token_mask.any()


# --- image encoder projection --------------------------------------------

def test_projection_sets_the_output_width():
    cfg = ImageEncoderConfig(name="vit", embed_dim=64, patch_size=8, depth=1, num_heads=2)
    encoder = IMAGE_ENCODERS.build("vit", cfg, image_size=32, in_channels=1)
    assert encoder.embed_dim == 64
    assert encoder(torch.zeros(2, 1, 32, 32)).shape[-1] == 64


def test_config_rejects_a_fusion_width_the_encoder_cannot_produce():
    from bn5212_training.config import config_from_dict

    payload = {
        "experiment": "x",
        "modalities": ["cxr"],
        "data": {"run_dir": "/tmp/run"},
        "image_encoder": {"embed_dim": 768},
        "fusion": {"name": "image_only", "embed_dim": 64},
    }
    with pytest.raises(ValueError, match="project_to"):
        config_from_dict(payload)


def test_config_accepts_the_width_once_a_projection_is_declared():
    from bn5212_training.config import config_from_dict

    payload = {
        "experiment": "x",
        "modalities": ["cxr"],
        "data": {"run_dir": "/tmp/run"},
        "image_encoder": {"embed_dim": 768, "project_to": 64},
        "fusion": {"name": "image_only", "embed_dim": 64},
    }
    assert config_from_dict(payload).image_encoder.project_to == 64


# --- augmentation ---------------------------------------------------------

def test_augmentation_changes_the_image():
    from bn5212_training.augment import build_train_transform

    transform = build_train_transform(AugmentationConfig(enabled=True), image_size=32)
    torch.manual_seed(0)
    image = torch.randn(1, 32, 32)
    out = transform(image)
    assert out.shape == image.shape
    assert not torch.allclose(out, image)
    assert torch.isfinite(out).all()


def test_augmentation_is_off_by_default():
    from bn5212_training.augment import build_train_transform

    assert build_train_transform(AugmentationConfig(), image_size=32) is None


def test_horizontal_flip_is_off_unless_requested():
    """A mirrored chest film is anatomically wrong, so it must be opt-in."""
    assert AugmentationConfig().horizontal_flip is False
    assert AugmentationConfig(enabled=True).horizontal_flip is False


def test_augmentation_rejects_an_impossible_crop_range():
    from bn5212_training.augment import ChestRadiographAugmentation

    with pytest.raises(ValueError, match="crop scale"):
        ChestRadiographAugmentation(
            AugmentationConfig(enabled=True, crop_scale_min=0.9, crop_scale_max=0.5), 32
        )


def test_intensity_only_augmentation_preserves_geometry():
    from bn5212_training.augment import build_train_transform

    cfg = AugmentationConfig(enabled=True, rotation_degrees=0, crop_scale_min=1.0)
    transform = build_train_transform(cfg, image_size=16)
    image = torch.zeros(1, 16, 16)
    image[0, 4:12, 4:12] = 1.0
    out = transform(image)
    # A pure affine intensity change keeps the two-level structure intact.
    assert len(torch.unique(out.round(decimals=4))) == 2


# --- optimiser parameter groups -------------------------------------------


def _image_config(**image_overrides):
    from bn5212_training.config import config_from_dict

    payload = {
        "experiment": "opt",
        "modalities": ["cxr"],
        "data": {"run_dir": "/tmp/run"},
        "image_encoder": {"name": "vit", "embed_dim": 32, "patch_size": 16,
                          "depth": 1, "num_heads": 2, **image_overrides},
        "fusion": {"name": "image_only", "embed_dim": 32},
    }
    return config_from_dict(payload)


def test_one_learning_rate_by_default():
    from bn5212_training.model import build_model
    from bn5212_training.trainer import _build_optimizer

    cfg = _image_config()
    model = build_model(cfg, image_size=32, channels=1)
    optimizer = _build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 1


def test_backbone_gets_its_own_rate_when_requested():
    """A head's learning rate would wash the pretrained features out."""
    from dataclasses import replace

    from bn5212_training.model import build_model
    from bn5212_training.trainer import _build_optimizer

    cfg = _image_config()
    cfg = replace(cfg, optim=replace(cfg.optim, lr=1e-3, backbone_lr=1e-5))
    model = build_model(cfg, image_size=32, channels=1)
    optimizer = _build_optimizer(model, cfg)

    # The from-scratch `vit` encoder has no .backbone, so nothing is separated.
    assert len(optimizer.param_groups) == 1

    class Fake(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.backbone = inner

    model.image_encoder = Fake(model.image_encoder)
    optimizer = _build_optimizer(model, cfg)
    rates = sorted(group["lr"] for group in optimizer.param_groups)
    assert rates == [1e-5, 1e-3]


def test_frozen_parameters_are_not_handed_to_the_optimiser():
    from dataclasses import replace

    from bn5212_training.model import build_model
    from bn5212_training.trainer import _build_optimizer

    cfg = replace(_image_config(), optim=replace(_image_config().optim, backbone_lr=1e-5))
    model = build_model(cfg, image_size=32, channels=1)

    class Fake(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.backbone = inner

    inner = model.image_encoder
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    model.image_encoder = Fake(inner)

    optimizer = _build_optimizer(model, cfg)
    handed = sum(len(group["params"]) for group in optimizer.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert handed == trainable


# --- chest-radiograph pretrained backbone ---------------------------------

xrv = pytest.importorskip("torchxrayvision", reason="torchxrayvision is optional")


def _xrv_encoder(**kwargs):
    from bn5212_training.encoders import XRayVisionDenseNet

    defaults = dict(weights="densenet121-res224-chex", mean=(0.5,), std=(0.5,), project_to=32)
    return XRayVisionDenseNet(224, 1, **{**defaults, **kwargs})


def test_weights_trained_on_mimic_are_refused():
    """Those checkpoints saw MIMIC-CXR, which overlaps our held-out images."""
    from bn5212_training.encoders import XRayVisionDenseNet

    for weights in ("densenet121-res224-all", "densenet121-res224-mimic"):
        with pytest.raises(ValueError, match="MIMIC-CXR"):
            XRayVisionDenseNet(224, 1, weights=weights)


def test_pipeline_normalisation_is_undone_before_the_backbone():
    """These weights expect roughly [-1024, 1024]; ours arrive standardised."""
    encoder = _xrv_encoder(mean=(0.485,), std=(0.229,))
    # A pixel that was 1.0 before normalisation.
    normalised = torch.full((1, 1, 224, 224), (1.0 - 0.485) / 0.229)
    converted = encoder._to_xrv_range(normalised)
    assert converted.max().item() == pytest.approx(1024.0, abs=1.0)

    zero_pixel = torch.full((1, 1, 224, 224), (0.0 - 0.485) / 0.229)
    assert encoder._to_xrv_range(zero_pixel).min().item() == pytest.approx(-1024.0, abs=1.0)


def test_three_repeated_channels_collapse_to_one():
    encoder = _xrv_encoder(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    image = torch.rand(2, 3, 224, 224)
    image[:, 1] = image[:, 0]
    image[:, 2] = image[:, 0]
    assert encoder._to_xrv_range(image).shape == (2, 1, 224, 224)


def test_spatial_map_becomes_tokens_with_a_cls():
    encoder = _xrv_encoder()
    tokens = encoder(torch.zeros(2, 1, 224, 224))
    assert tokens.shape == (2, encoder.num_tokens, 32)
    assert encoder.num_tokens == encoder.grid * encoder.grid + 1


def test_the_leading_token_depends_on_the_image():
    """A learned constant here would make pooling="cls" blind to the input.

    That failure is silent: the model trains, the loss moves, and every sample
    receives the same score, which shows up only as an AUROC pinned at 0.5.
    """
    encoder = _xrv_encoder().eval()
    # Structurally different images, not noise: random pixels share their global
    # statistics, so a pooled summary of them is near-constant for reasons that
    # have nothing to do with the bug being guarded against.
    images = torch.zeros(4, 1, 224, 224)
    images[1] = 1.0
    images[2, :, :112] = 1.0                      # bright upper half
    images[3, :, :, ::2] = 1.0                    # vertical stripes
    with torch.no_grad():
        tokens = encoder(images)
    leading = tokens[:, 0]
    spread = leading.std(dim=0).mean()
    assert spread > 0.05, f"the leading token barely varies between images ({spread:.4f})"
    # And it must actually differ pairwise, not just wobble.
    assert not torch.allclose(leading[0], leading[1], atol=1e-3)
    assert not torch.allclose(leading[2], leading[3], atol=1e-3)


def test_the_backbone_is_frozen_but_the_projection_is_not():
    encoder = _xrv_encoder(freeze=True)
    assert not any(p.requires_grad for p in encoder.backbone.parameters())
    assert all(p.requires_grad for p in encoder.project.parameters())


def test_a_frozen_backbone_stays_in_inference_mode_during_training():
    """requires_grad=False alone leaves BatchNorm updating its running averages.

    In training mode the frozen backbone would normalise with batch statistics
    and drift, so the features the head trained on would not be the features it
    is later scored with.
    """
    import torch

    encoder = _xrv_encoder(freeze=True)
    encoder.train()
    assert encoder.training and not encoder.backbone.training

    norms = [m for m in encoder.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [norm.running_mean.clone() for norm in norms]
    encoder(torch.randn(2, 1, 224, 224))
    assert all(torch.equal(norm.running_mean, kept) for norm, kept in zip(norms, before))

    # An unfrozen backbone is fine-tuned, so it does follow the training mode.
    assert _xrv_encoder(freeze=False).train().backbone.training


# --- variable-specific clinical projection --------------------------------

def _opposing_variables_auroc(encoder_name: str) -> float:
    """Held-out AUROC when the label needs variable 0 high and variable 1 low."""
    import torch

    from bn5212_training.config import config_from_dict
    from bn5212_training.metrics import auroc
    from bn5212_training.model import build_model

    torch.manual_seed(0)
    variables, hours, rows = 4, 8, 900
    values = torch.randn(rows, variables, hours)
    mask = torch.ones(rows, variables, hours, dtype=torch.bool)
    labels = (values[:, 0].mean(1) - values[:, 1].mean(1) > 0).float()

    cfg = config_from_dict({
        "experiment": "opposing", "modalities": ["clinical"],
        "data": {"run_dir": "unused"},
        "clinical_encoder": {"name": encoder_name, "embed_dim": 16},
        "fusion": {"name": "clinical_only", "embed_dim": 16, "pooling": "mean"},
    })
    model = build_model(cfg, num_variables=variables, num_timesteps=hours)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    for _ in range(40):
        for start in range(0, 600, 50):
            batch = slice(start, start + 50)
            logits = model({"clinical": values[batch], "clinical_mask": mask[batch]})
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model({"clinical": values[600:], "clinical_mask": mask[600:]}))
    return auroc(labels[600:].numpy(), scores.numpy())


def test_variable_projection_learns_variables_that_point_in_opposite_directions():
    """Clinical risk is this pattern: high heart rate and low blood pressure agree.

    The shared projection also learns it at this toy size, through its
    LayerNorm, so nothing is asserted about it here. It stopped doing so at the
    cohort's geometry of 17 variables over 48 hours, which is why this encoder
    exists; that measurement is recorded in the encoder's comment.
    """
    assert _opposing_variables_auroc("variable_projection") > 0.95


def test_variable_projection_keeps_the_token_contract():
    import torch

    from bn5212_training.config import ClinicalEncoderConfig
    from bn5212_training.registry import CLINICAL_ENCODERS

    encoder = CLINICAL_ENCODERS.build(
        "variable_projection", ClinicalEncoderConfig(name="variable_projection", embed_dim=16),
        num_variables=3, num_timesteps=5,
    )
    encoder.eval()
    values = torch.randn(2, 3, 5)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[0, 2] = False
    tokens, token_mask = encoder(values * mask, mask)
    assert tokens.shape == (2, 3, 16)
    assert token_mask.tolist() == [[True, True, False], [True, True, True]]

    # Each variable owns its weights: changing one variable moves only its token.
    changed = values.clone()
    changed[:, 0] += 1.0
    moved, _ = encoder(changed * mask, mask)
    assert not torch.allclose(moved[:, 0], tokens[:, 0])
    assert torch.allclose(moved[:, 1:], tokens[:, 1:])
