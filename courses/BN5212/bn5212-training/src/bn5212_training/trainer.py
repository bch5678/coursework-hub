"""The training loop shared by every experiment.

Optimiser, schedule, loss, seeding, checkpoint selection and prediction export
are identical across experiments. Model selection reads validation data only;
test predictions are exported and never scored here.
"""
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import plots
from .clinical import ClinicalNormalizer, build_provider
from .config import TrainingConfig
from .data import TrainingDataset, make_loader
from .losses import build_loss
from .manifest import build_manifest, sha256, write_manifest
from .metrics import is_better, selection_metrics
from .model import BN5212Model, build_model, load_checkpoint, save_checkpoint
from .predict import (
    move_to_device,
    positive_probability,
    resolve_device,
    run_inference,
    write_detailed_predictions,
    write_predictions,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _learning_rate_scale(epoch: int, cfg: TrainingConfig) -> float:
    """Linear warmup then cosine decay, expressed as a multiplier on the base LR."""
    warmup = max(0, cfg.optim.warmup_epochs)
    if epoch < warmup:
        return (epoch + 1) / (warmup + 1)
    if cfg.optim.scheduler == "none":
        return 1.0
    progress = (epoch - warmup) / max(1, cfg.optim.epochs - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    floor = cfg.optim.min_lr / cfg.optim.lr if cfg.optim.lr > 0 else 0.0
    return floor + (1.0 - floor) * cosine


@torch.inference_mode()
def evaluate_split(
    model: BN5212Model,
    loader: DataLoader,
    loss_fn: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Validation pass: mean loss plus the selection metrics."""
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    scores: list[float] = []
    labels: list[int] = []

    for batch in loader:
        on_device = move_to_device(batch, device)
        logits = model(on_device)
        loss = loss_fn(logits, on_device["label"], on_device.get("sample_weight"))
        count = len(batch["sample_id"])
        total_loss += float(loss) * count
        total_weight += count
        scores.extend(positive_probability(logits).detach().float().cpu().tolist())
        labels.extend(batch["label"].detach().cpu().tolist())

    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    result = selection_metrics(label_array, score_array)
    result["loss"] = total_loss / max(total_weight, 1.0)
    result["labels"] = label_array
    result["scores"] = score_array
    return result


def train_one_epoch(
    model: BN5212Model,
    loader: DataLoader,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainingConfig,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    use_amp = scaler is not None
    # Gradient accumulation decouples the effective batch size from what fits in
    # VRAM, which matters on an 8 GB card running ViT-B/16.
    accumulate = max(1, cfg.optim.grad_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    pending = 0

    def apply_step() -> None:
        if cfg.optim.grad_clip > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        on_device = move_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(on_device)
            loss = loss_fn(logits, on_device["label"], on_device.get("sample_weight"))
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite; lower the learning rate")

        # Scale so that accumulated gradients average rather than sum.
        scaled = loss / accumulate
        if use_amp:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        pending += 1
        if pending == accumulate:
            apply_step()
            pending = 0

        count = len(batch["sample_id"])
        total_loss += float(loss.detach()) * count
        total_count += count

    # A trailing partial group would otherwise be dropped on the floor.
    if pending:
        apply_step()

    return total_loss / max(total_count, 1)


def _build_optimizer(model: BN5212Model, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """AdamW, with a separate learning rate for unfrozen backbone weights when asked."""
    decay = cfg.optim.weight_decay
    if cfg.optim.backbone_lr is None or model.image_encoder is None:
        return torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=decay)

    backbone = getattr(model.image_encoder, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    backbone_params, other_params = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (backbone_params if id(parameter) in backbone_ids else other_params).append(parameter)

    groups = [{"params": other_params, "lr": cfg.optim.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.optim.backbone_lr})
    return torch.optim.AdamW(groups, lr=cfg.optim.lr, weight_decay=decay)


def _prepare_clinical(
    cfg: TrainingConfig,
    splits: Sequence[str] = ("train",),
    subjects: Sequence[str] | None = None,
) -> tuple[Any, ClinicalNormalizer | None]:
    """Build the provider and fit normalisation statistics on fitting data only.

    In a fold the statistics come from that fold's fitting patients.
    """
    if not cfg.uses_clinical:
        return None, None
    if cfg.clinical_encoder.pretrained:
        # The encoder was trained on inputs scaled with these statistics, and they
        # come from patients outside the cohort, so no fold can leak through them.
        from .pretrain import load_pretrained_normalizer

        normalizer = load_pretrained_normalizer(cfg.clinical_encoder.pretrained)
        provider = build_provider(cfg.data, variable_names=normalizer.variable_names)
        return provider, normalizer
    provider = build_provider(cfg.data)
    probe = TrainingDataset(
        cfg.data.run_dir,
        splits,
        data_pipeline_path=cfg.data.data_pipeline_path,
        provider=None,
        load_image=False,
        subjects=subjects,
    )
    normalizer = ClinicalNormalizer.fit(provider, probe.rows(), split="train")
    return provider, normalizer


def _attention_figures(
    model: BN5212Model,
    loader: DataLoader,
    device: torch.device,
    provider: Any,
    figures_dir: Path,
) -> list[str]:
    """Record one batch of cross-attention weights for the RQ3 figures."""
    fusion = model.fusion
    if not hasattr(fusion, "record_attention"):
        return []
    written: list[str] = []
    fusion.record_attention = True
    try:
        model.eval()
        with torch.inference_mode():
            batch = next(iter(loader))
            model(move_to_device(batch, device))
        weights = fusion.last_attention()
        if weights is None:
            return []
        # [B, heads, variables, patches] -> mean over samples and heads.
        matrix = weights.detach().float().cpu().numpy().mean(axis=(0, 1))
        names = list(provider.variable_names)[: matrix.shape[0]]
        written.append(
            str(plots.attention_matrix(matrix, names, figures_dir / "attention_matrix.png"))
        )
        # Drop the image CLS token, then fold the patch axis back to a square grid.
        patches = matrix[:, 1:] if matrix.shape[1] == model.num_image_tokens else matrix
        grid = int(math.isqrt(patches.shape[1]))
        if grid * grid == patches.shape[1] and grid > 1:
            written.append(
                str(
                    plots.attention_patch_maps(
                        patches, names, grid, figures_dir / "attention_patch_maps.png"
                    )
                )
            )
    except StopIteration:
        return written
    finally:
        fusion.record_attention = False
    return written


def train(
    cfg: TrainingConfig,
    *,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    fit_subjects: Sequence[str] | None = None,
    select_subjects: Sequence[str] | None = None,
    score_subjects: Sequence[str] | None = None,
    epoch_budget: int | None = None,
) -> dict[str, Any]:
    """Run one experiment end to end and return its summary.

    fit_subjects switches to fold mode (cross-validation over train+val). The
    three patient groups must be disjoint:

    * fit_subjects: the weights are fitted on these.
    * select_subjects: early stopping and checkpoint selection read these.
    * score_subjects: scored once by the selected checkpoint, exported as
      predictions_heldout, and never read by anything that shapes the model.

    epoch_budget replaces select_subjects: train exactly that many epochs of the
    configured schedule and keep the final weights.
    """
    started = time.time()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    fold_mode = fit_subjects is not None
    if not fold_mode and (select_subjects or score_subjects or epoch_budget is not None):
        raise ValueError("select_subjects, score_subjects and epoch_budget need fit_subjects")
    if fold_mode:
        if (select_subjects is None) == (epoch_budget is None):
            raise ValueError("fold mode needs exactly one of select_subjects or epoch_budget")
        if epoch_budget is not None and epoch_budget < 1:
            raise ValueError("epoch_budget must be at least 1")
        groups = {
            "fit_subjects": {str(s) for s in fit_subjects},
            "select_subjects": {str(s) for s in select_subjects or ()},
            "score_subjects": {str(s) for s in score_subjects or ()},
        }
        names = list(groups)
        for position, first in enumerate(names):
            for second in names[position + 1:]:
                if groups[first] & groups[second]:
                    raise ValueError(
                        f"{first} and {second} share {len(groups[first] & groups[second])} "
                        "patient(s); the three groups must be disjoint"
                    )

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(output_dir or cfg.output_dir) / cfg.experiment / run_id
    figures_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fold_splits = ("train", "val")
    provider, normalizer = _prepare_clinical(
        cfg, fold_splits if fold_mode else ("train",), fit_subjects
    )
    if normalizer is not None:
        normalizer.save(root / "clinical_normalizer.json")

    image_cache = None
    if cfg.uses_image and cfg.data.image_cache:
        from .cache import ImageCache

        probe = TrainingDataset(
            cfg.data.run_dir, "train",
            data_pipeline_path=cfg.data.data_pipeline_path, load_image=False,
        )
        image_cache = ImageCache(cfg.data.image_cache, probe.spec)

    # Augmentation is training-only: validation and test must see the image the
    # pipeline produced, or the metric measures a different input distribution.
    train_transform = None
    if cfg.uses_image and cfg.augmentation.enabled:
        from .augment import build_train_transform

        geometry = TrainingDataset(
            cfg.data.run_dir, "train",
            data_pipeline_path=cfg.data.data_pipeline_path, load_image=False,
        )
        train_transform = build_train_transform(cfg.augmentation, geometry.image_size)

    def build(splits, subjects, transform=None):
        return TrainingDataset(
            cfg.data.run_dir,
            splits,
            data_pipeline_path=cfg.data.data_pipeline_path,
            provider=provider,
            normalizer=normalizer,
            load_image=cfg.uses_image,
            subjects=subjects,
            image_cache=image_cache,
            transform=transform,
        )

    if fold_mode:
        # "val" is what the loop monitors: the selection patients, or under an
        # epoch budget the fitting patients themselves, whose numbers are then
        # in-sample and never reported. The frozen test split is not built.
        monitored = fit_subjects if select_subjects is None else select_subjects
        datasets = {
            "train": build(fold_splits, fit_subjects, train_transform),
            "val": build(fold_splits, monitored),
        }
        if score_subjects is not None:
            datasets["heldout"] = build(fold_splits, score_subjects)
    else:
        datasets = {"train": build("train", None, train_transform)}
        datasets.update({split: build(split, None) for split in ("val", "test")})
    loaders = {
        split: make_loader(
            dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=split == "train",
            seed=cfg.seed,
        )
        for split, dataset in datasets.items()
    }

    train_set = datasets["train"]
    loader_spec = train_set.spec["loader"]
    # Normalisation belongs in the geometry: an encoder can hold it as a buffer,
    # and rebuilding from a checkpoint has to reproduce the same shapes.
    geometry = {
        "image_size": train_set.image_size,
        "channels": train_set.channels,
        "num_variables": train_set.num_variables,
        "num_timesteps": train_set.num_timesteps,
        "mean": list(loader_spec["mean"]),
        "std": list(loader_spec["std"]),
    }
    model = build_model(cfg, **geometry).to(device)
    loss_fn = build_loss(cfg, train_set.label_array())
    optimizer = _build_optimizer(model, cfg)
    scaler = (
        torch.amp.GradScaler(device.type)
        if cfg.optim.amp and device.type == "cuda"
        else None
    )

    # Remember each group's base rate so the schedule scales them independently.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_epoch = -1
    best_path = root / "checkpoint_best.pt"
    patience_left = cfg.optim.early_stopping_patience
    selection_opens = (
        cfg.optim.warmup_epochs
        if cfg.optim.min_epochs_before_selection is None
        else cfg.optim.min_epochs_before_selection
    )

    # A budget truncates the run but not the schedule: the learning rate at
    # epoch k must be the one the selecting folds saw at epoch k.
    planned = cfg.optim.epochs if epoch_budget is None else min(epoch_budget, cfg.optim.epochs)
    for epoch in range(planned):
        scale = _learning_rate_scale(epoch, cfg)
        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", group["lr"]) * scale

        train_loss = train_one_epoch(
            model, loaders["train"], loss_fn, optimizer, device, cfg, scaler
        )
        validation = evaluate_split(model, loaders["val"], loss_fn, device)
        record = {
            "epoch": epoch + 1,
            "lr": cfg.optim.lr * scale,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_auroc": validation["auroc"],
            "val_auprc": validation["auprc"],
            "val_brier": validation["brier"],
        }
        history.append(record)

        candidate = validation.get(cfg.selection_metric)
        # Epochs inside the warmup ran at a reduced learning rate, so their
        # scores are not comparable with the rest. They neither win the
        # checkpoint nor count against the patience budget.
        warming_up = epoch < selection_opens and epoch_budget is None
        if warming_up or epoch_budget is not None:
            pass
        elif is_better(cfg.selection_metric, candidate, best_metric):
            best_metric = candidate
            best_epoch = epoch + 1
            patience_left = cfg.optim.early_stopping_patience
            save_checkpoint(
                best_path,
                model,
                epoch=best_epoch,
                metrics={key: record[key] for key in record if key != "epoch"},
                normalizer=normalizer,
                geometry=geometry,
            )
        else:
            patience_left -= 1

        print(
            f"[{cfg.experiment}] epoch {epoch + 1}/{cfg.optim.epochs} "
            f"train_loss={train_loss:.4f} val_loss={validation['loss']:.4f} "
            f"val_auroc={validation['auroc']}"
            + ("  (warmup)" if warming_up else "")
        )
        if patience_left <= 0:
            print(f"[{cfg.experiment}] early stopping after epoch {epoch + 1}")
            break

    save_checkpoint(
        root / "checkpoint_last.pt",
        model,
        epoch=len(history),
        metrics=history[-1] if history else {},
        normalizer=normalizer,
        geometry=geometry,
    )

    if epoch_budget is not None:
        # Nothing was selected: the deliverable is the model the budget produced.
        best_epoch = len(history)
        save_checkpoint(
            best_path,
            model,
            epoch=best_epoch,
            metrics={key: value for key, value in history[-1].items() if key != "epoch"},
            normalizer=normalizer,
            geometry=geometry,
        )

    # Every downstream artefact comes from the validation-selected checkpoint.
    if best_path.is_file():
        model, _ = load_checkpoint(best_path, map_location=device)
        model.to(device)

    validation = evaluate_split(model, loaders["val"], loss_fn, device)
    val_metrics = {
        key: validation[key]
        for key in ("loss", "auroc", "auprc", "brier", "prevalence", "n", "n_positive")
    }
    val_metrics["selected_epoch"] = best_epoch
    val_metrics["selection_metric"] = cfg.selection_metric
    (root / "metrics_val.json").write_text(
        json.dumps(val_metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )

    figure_paths = [
        str(plots.training_curves(history, figures_dir / "training_curves.png")),
        str(
            plots.roc_curve(
                validation["labels"],
                validation["scores"],
                figures_dir / "roc_curve_val.png",
                auroc=validation["auroc"],
            )
        ),
        str(
            plots.precision_recall_curve(
                validation["labels"],
                validation["scores"],
                figures_dir / "pr_curve_val.png",
                auprc=validation["auprc"],
            )
        ),
        str(
            plots.score_distribution(
                validation["labels"],
                validation["scores"],
                figures_dir / "score_distribution_val.png",
            )
        ),
    ]

    prediction_paths: dict[str, str] = {}
    exported = [name for name in ("val", "heldout", "test") if name in loaders]
    for split in exported:
        result = run_inference(model, loaders[split], device)
        prediction_paths[split] = str(
            write_predictions(result, root / f"predictions_{split}.csv")
        )
        write_detailed_predictions(
            result,
            root / f"predictions_{split}_detailed.csv",
            split=split,
            model_version=run_id,
        )

    if cfg.uses_clinical and cfg.uses_image and provider is not None:
        figure_paths.extend(_attention_figures(model, loaders["val"], device, provider, figures_dir))

    parameters = model.parameter_counts()
    summary_row = {
        "experiment": cfg.experiment,
        "run_id": run_id,
        "modalities": "+".join(cfg.modalities),
        "fusion": cfg.fusion.name,
        "val_auroc": validation["auroc"],
        "val_auprc": validation["auprc"],
        "val_brier": validation["brier"],
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "total_parameters": parameters["total"],
        "seed": cfg.seed,
    }
    plots.write_summary_table([summary_row], root / "summary")

    (root / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_manifest(
        root / "run_manifest.json",
        build_manifest(
            config=cfg.to_dict(),
            config_fingerprint=cfg.fingerprint(),
            run_dir=cfg.data.run_dir,
            seed=cfg.seed,
            experiment=cfg.experiment,
            extra={
                "run_id": run_id,
                "device": str(device),
                "duration_seconds": round(time.time() - started, 1),
                "parameters": parameters,
                "geometry": geometry,
                "split_sizes": {name: len(ds) for name, ds in datasets.items()},
                "fold_mode": fold_mode,
                "fit_patients": None if fit_subjects is None else len(fit_subjects),
                "select_patients": None if select_subjects is None else len(select_subjects),
                "score_patients": None if score_subjects is None else len(score_subjects),
                "epoch_budget": epoch_budget,
                "validation_metrics": val_metrics,
                "checkpoint_sha256": sha256(best_path) if best_path.is_file() else None,
                "clinical_provider": cfg.data.clinical_provider if cfg.uses_clinical else None,
                "image_cache": cfg.data.image_cache if cfg.uses_image else None,
                "augmentation": cfg.augmentation.__dict__ if (cfg.uses_image and cfg.augmentation.enabled) else None,
                # The statistics themselves live in clinical_normalizer.json; the
                # manifest only records that they were fitted on train.
                "clinical_normalizer": (
                    None
                    if normalizer is None
                    else {
                        "fitted_on": normalizer.fitted_on,
                        "num_variables": len(normalizer.variable_names),
                        "num_observations": normalizer.num_observations,
                    }
                ),
                "predictions": prediction_paths,
                "figures": figure_paths,
            },
        ),
    )

    return {
        "run_dir": str(root),
        "summary": summary_row,
        "validation_metrics": val_metrics,
        "predictions": prediction_paths,
        "history": history,
    }
