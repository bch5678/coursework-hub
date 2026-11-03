"""Patient-grouped nested cross-validation over the frozen train+val splits.

A single split leaves four deaths in validation, too few to compare models.

* Folds are grouped by subject_id: no patient is both fitted and scored.
* The held-out fold never chooses anything. Inner folds of the fitting patients
  pick the epoch budget, the model is refitted on all of them, and only then is
  the held-out fold scored. Stopping on the held-out fold itself reported an
  AUROC of 0.720 where this protocol gives 0.606.
* The test split is never fitted or selected on; the deliverable model exports
  label-free predictions for it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import plots
from .config import TrainingConfig
from .data import TrainingDataset
from .manifest import build_manifest, write_manifest
from .metrics import bootstrap_interval, selection_metrics
from .predict import predict_from_checkpoint

FOLD_SPLITS = ("train", "val")


def _patient_table(cfg: TrainingConfig, splits: Sequence[str] = FOLD_SPLITS) -> pd.DataFrame:
    """One row per patient with the label used for stratification."""
    frames = []
    for split in splits:
        dataset = TrainingDataset(
            cfg.data.run_dir,
            split,
            data_pipeline_path=cfg.data.data_pipeline_path,
            load_image=False,
        )
        frames.append(dataset.frame[["subject_id", "label"]])
    pooled = pd.concat(frames, ignore_index=True)
    pooled["subject_id"] = pooled["subject_id"].astype(str)
    # A patient counts as positive if any of their admissions ended in death,
    # matching how the pipeline stratifies its own split.
    return pooled.groupby("subject_id", as_index=False)["label"].max()


def make_folds(
    patients: pd.DataFrame, *, n_splits: int = 5, seed: int = 5212, stratify: bool = True
) -> list[np.ndarray]:
    """Deterministic patient-grouped folds, stratified when the classes allow it.

    Assignment is by a seeded hash of subject_id rather than a shuffle, so the
    folds are reproducible from the seed alone and do not depend on row order.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(patients) < n_splits:
        raise ValueError(f"{len(patients)} patients cannot fill {n_splits} folds")

    groups: list[pd.DataFrame] = []
    positives = int(patients["label"].sum())
    negatives = int(len(patients) - positives)
    if stratify and min(positives, negatives) >= n_splits:
        groups = [group for _, group in patients.groupby("label")]
    else:
        groups = [patients]

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for group in groups:
        ordered = group.assign(
            order=group["subject_id"].map(
                lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
            )
        ).sort_values(["order", "subject_id"])
        # Deal round-robin so each fold gets a near-equal share of every class.
        for position, subject in enumerate(ordered["subject_id"]):
            folds[position % n_splits].append(subject)

    empty = [index for index, fold in enumerate(folds) if not fold]
    if empty:
        raise ValueError(f"Folds {empty} came out empty; reduce n_splits")
    return [np.asarray(sorted(fold), dtype=object) for fold in folds]


def _log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped)))


def fold_summary(frame: pd.DataFrame, index: int, held_out_patients: int) -> dict[str, Any]:
    """Metrics of one held-out fold, from its scored rows."""
    labels = frame["label"].to_numpy()
    scores = frame["y_score"].to_numpy()
    metrics = selection_metrics(labels, scores)
    return {
        "fold": index,
        "held_out_patients": int(held_out_patients),
        "scored_rows": int(len(frame)),
        "positives": int(frame["label"].sum()),
        **{key: metrics.get(key) for key in ("auroc", "auprc", "brier")},
        "loss": _log_loss(labels, scores),
    }


def write_pooled_report(
    root: Path,
    cfg: TrainingConfig,
    pooled: pd.DataFrame,
    per_fold: list[dict[str, Any]],
    *,
    run_id: str,
    n_splits: int,
    inner_splits: int | None,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Pooled metrics, intervals, tables and figures for held-out predictions.

    pooled carries sample_id, subject_id, label, y_score and fold, one row per
    sample, each scored by a model that never fitted on that patient.
    """
    if pooled["sample_id"].duplicated().any():
        raise ValueError("A sample was scored by more than one fold; the folds overlap")

    labels = pooled["label"].to_numpy()
    scores = pooled["y_score"].to_numpy()
    subjects = pooled["subject_id"].to_numpy()
    pooled_metrics = selection_metrics(labels, scores)
    pooled_metrics["auroc_ci"] = bootstrap_interval(
        labels, scores, subjects, metric="auroc", samples=bootstrap_samples, seed=cfg.seed
    )
    pooled_metrics["auprc_ci"] = bootstrap_interval(
        labels, scores, subjects, metric="auprc", samples=bootstrap_samples, seed=cfg.seed
    )

    fold_frame = pd.DataFrame(per_fold)
    spread = {
        key: {
            "mean": float(fold_frame[key].mean()),
            "std": float(fold_frame[key].std(ddof=1)) if len(fold_frame) > 1 else 0.0,
            "min": float(fold_frame[key].min()),
            "max": float(fold_frame[key].max()),
        }
        for key in ("auroc", "auprc", "brier")
        if fold_frame[key].notna().all()
    }

    root.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(root / "out_of_fold_predictions.csv", index=False, lineterminator="\n")
    fold_frame.to_csv(root / "per_fold_metrics.csv", index=False, lineterminator="\n")
    (root / "cv_metrics.json").write_text(
        json.dumps(
            {
                "n_splits": n_splits,
                "inner_splits": inner_splits,
                "pooled": pooled_metrics,
                "per_fold_spread": spread,
                "per_fold": per_fold,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    figures = root / "figures"
    plots.roc_curve(labels, scores, figures / "roc_out_of_fold.png", auroc=pooled_metrics["auroc"])
    plots.precision_recall_curve(
        labels, scores, figures / "pr_out_of_fold.png", auprc=pooled_metrics["auprc"]
    )
    plots.score_distribution(labels, scores, figures / "score_distribution_out_of_fold.png")

    interval = pooled_metrics["auroc_ci"]
    plots.write_summary_table(
        [
            {
                "experiment": cfg.experiment,
                "run_id": run_id,
                "n_splits": n_splits,
                "modalities": "+".join(cfg.modalities),
                "fusion": cfg.fusion.name,
                "oof_n": pooled_metrics["n"],
                "oof_positives": pooled_metrics["n_positive"],
                "oof_auroc": pooled_metrics["auroc"],
                "auroc_lower": interval["lower"],
                "auroc_upper": interval["upper"],
                "oof_auprc": pooled_metrics["auprc"],
                "oof_brier": pooled_metrics["brier"],
                "fold_auroc_std": spread.get("auroc", {}).get("std"),
            }
        ],
        root / "summary",
    )
    return pooled_metrics


def run_cross_validation(
    cfg: TrainingConfig,
    *,
    n_splits: int = 5,
    inner_splits: int = 4,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    bootstrap_samples: int = 2000,
    keep_fold_checkpoints: bool = False,
    train_final_model: bool = True,
) -> dict[str, Any]:
    """Score every train+val patient out of fold and pool the predictions.

    Per outer fold, inner_splits models early-stop on different slices of the
    fitting patients; their median epoch is the budget for a refit on all fitting
    patients, and that refit scores the held-out fold. Fold checkpoints are
    discarded unless keep_fold_checkpoints is set.
    """
    from .trainer import train  # imported here to keep module import light

    run_id = run_id or f"cv{n_splits}-{cfg.fingerprint()[:8]}"
    root = Path(output_dir or cfg.output_dir) / cfg.experiment / run_id
    root.mkdir(parents=True, exist_ok=True)

    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2")

    patients = _patient_table(cfg)
    folds = make_folds(patients, n_splits=n_splits, seed=cfg.seed, stratify=True)
    all_subjects = set(patients["subject_id"])

    def discard_checkpoints(run_dir: str) -> None:
        if not keep_fold_checkpoints:
            for checkpoint in Path(run_dir).glob("checkpoint_*.pt"):
                checkpoint.unlink()

    out_of_fold: list[pd.DataFrame] = []
    per_fold: list[dict[str, Any]] = []
    selected_epochs: list[int] = []
    for index, held_out in enumerate(folds):
        fitting = patients[~patients["subject_id"].isin(set(held_out))]
        # A different salt per outer fold, so the inner folds are not the same
        # deal of the same hash order every time.
        inner = make_folds(
            fitting, n_splits=inner_splits, seed=cfg.seed * 100 + index + 1, stratify=True
        )
        fold_cfg = replace(
            cfg,
            experiment=f"{cfg.experiment}_fold{index}",
            # Each fold keeps the frozen file layout: the model fits on the train
            # split restricted to the fitting patients and is scored on the
            # held-out patients wherever they live.
            notes=f"{cfg.notes} | cross-validation fold {index} of {n_splits}",
        )
        epochs: list[int] = []
        for position, selecting in enumerate(inner):
            result = train(
                fold_cfg,
                output_dir=root / "folds",
                run_id=f"inner{position}",
                fit_subjects=sorted(set(fitting["subject_id"]) - set(selecting)),
                select_subjects=list(selecting),
            )
            epochs.append(int(result["validation_metrics"]["selected_epoch"]))
            discard_checkpoints(result["run_dir"])

        budget = max(1, int(round(float(np.median(epochs)))))
        result = train(
            fold_cfg,
            output_dir=root / "folds",
            run_id="refit",
            fit_subjects=sorted(fitting["subject_id"]),
            epoch_budget=budget,
            score_subjects=list(held_out),
        )
        frame = pd.read_csv(
            Path(result["run_dir"]) / "predictions_heldout_detailed.csv",
            dtype={"sample_id": str, "subject_id": str, "hadm_id": str},
        ).drop(columns=["split", "model_version"])
        discard_checkpoints(result["run_dir"])
        frame["fold"] = index
        out_of_fold.append(frame)
        selected_epochs.extend(epochs)

        per_fold.append(
            {
                **fold_summary(frame, index, len(held_out)),
                # Chosen on the inner folds; the held-out fold had no say.
                "selected_epoch": budget,
                "inner_selected_epochs": "/".join(str(epoch) for epoch in epochs),
            }
        )

    pooled = pd.concat(out_of_fold, ignore_index=True)
    pooled_metrics = write_pooled_report(
        root, cfg, pooled, per_fold,
        run_id=run_id, n_splits=n_splits, inner_splits=inner_splits,
        bootstrap_samples=bootstrap_samples,
    )

    # Threshold selection in the benchmark needs validation scores from models
    # that never fitted on those patients. The out-of-fold rows of the frozen val
    # split are exactly that, whereas the deliverable has fitted on them.
    val_ids = set(
        TrainingDataset(
            cfg.data.run_dir, "val",
            data_pipeline_path=cfg.data.data_pipeline_path, load_image=False,
        ).frame["sample_id"].astype(str)
    )
    pooled.loc[pooled["sample_id"].isin(val_ids), ["sample_id", "y_score"]].to_csv(
        root / "predictions_val.csv", index=False, lineterminator="\n"
    )

    final = None
    if train_final_model:
        # The deliverable is a separate fit on every patient, for the median epoch
        # the inner folds selected, under the same schedule.
        budget = max(1, int(round(float(np.median(selected_epochs)))))
        final_cfg = replace(
            cfg,
            experiment=f"{cfg.experiment}_final",
            notes=(f"{cfg.notes} | final model fitted on all {len(all_subjects)} "
                   f"train+val patients for {budget} epochs, the median epoch the "
                   f"{n_splits}x{inner_splits} inner folds selected. Its performance "
                   "estimate is the out-of-fold result, not anything measured on "
                   "these patients."),
        )
        subjects = sorted(all_subjects)
        outcome = train(
            final_cfg,
            output_dir=root,
            run_id="final_model",
            fit_subjects=subjects,
            epoch_budget=budget,
        )
        # train() nests its output under experiment/run_id; lift the result to
        # one predictable place so the deliverable is not four levels down.
        produced = Path(outcome["run_dir"])
        destination = root / "final_model"
        if produced != destination:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(produced), str(destination))
            parent = produced.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        # Label-free hand-off for the benchmark project; nothing here scores it.
        test_predictions = predict_from_checkpoint(
            destination / "checkpoint_best.pt",
            split="test",
            output_path=root / "predictions_test.csv",
            device=cfg.device,
        )
        final = {
            "run_dir": str(destination),
            "checkpoint": str(destination / "checkpoint_best.pt"),
            "epochs": budget,
            "patients": len(subjects),
            "test_predictions": str(test_predictions),
            "note": "performance comes from the out-of-fold estimate, not from this fit",
        }

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
                "evaluation": "patient-grouped cross-validation over train+val",
                "n_splits": n_splits,
                "inner_splits": inner_splits,
                "selection": (
                    "inner folds of the fitting patients choose the epoch budget; "
                    "the held-out fold is scored by a refit on all fitting patients"
                ),
                "fold_sizes": [int(len(fold)) for fold in folds],
                "pooled_metrics": pooled_metrics,
                "per_fold": per_fold,
                # True of fitting and selection. The deliverable exports label-free
                # test predictions afterwards; no score is computed on them here.
                "test_split_untouched": True,
                "fold_checkpoints_kept": keep_fold_checkpoints,
                "final_model": final,
            },
        ),
    )
    return {
        "run_dir": str(root),
        "pooled": pooled_metrics,
        "per_fold": per_fold,
        "final_model": final,
    }
