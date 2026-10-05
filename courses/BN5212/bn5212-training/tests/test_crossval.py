"""Patient-grouped cross-validation.

The properties tested here are what make the pooled result trustworthy: no
patient is both fitted and scored, the patients being scored never choose the
checkpoint, every sample is scored exactly once, and the frozen test split
stays out of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bn5212_training.config import config_from_dict
from bn5212_training.crossval import make_folds, run_cross_validation
from bn5212_training.metrics import bootstrap_interval, paired_bootstrap_difference


def _patients(n_positive: int, n_negative: int) -> pd.DataFrame:
    rows = [{"subject_id": f"p{i}", "label": 1} for i in range(n_positive)]
    rows += [{"subject_id": f"n{i}", "label": 0} for i in range(n_negative)]
    return pd.DataFrame(rows)


def test_folds_partition_every_patient_exactly_once():
    folds = make_folds(_patients(10, 40), n_splits=5, seed=5212)
    pooled = np.concatenate(folds)
    assert len(pooled) == 50
    assert len(set(pooled)) == 50, "a patient appears in two folds"


def test_folds_are_deterministic_from_the_seed():
    first = make_folds(_patients(10, 40), n_splits=5, seed=5212)
    same = make_folds(_patients(10, 40), n_splits=5, seed=5212)
    other = make_folds(_patients(10, 40), n_splits=5, seed=99)
    assert [list(f) for f in first] == [list(f) for f in same]
    assert [list(f) for f in first] != [list(f) for f in other]


def test_folds_do_not_depend_on_row_order():
    frame = _patients(10, 40)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert [list(f) for f in make_folds(frame, n_splits=5, seed=5212)] == [
        list(f) for f in make_folds(shuffled, n_splits=5, seed=5212)
    ]


def test_stratification_spreads_the_positives():
    """With few events, one fold holding them all would make its AUROC undefined."""
    folds = make_folds(_patients(10, 40), n_splits=5, seed=5212)
    positives_per_fold = [sum(1 for s in fold if str(s).startswith("p")) for fold in folds]
    assert min(positives_per_fold) >= 1
    assert max(positives_per_fold) - min(positives_per_fold) <= 1


def test_too_few_patients_for_the_fold_count_is_rejected():
    with pytest.raises(ValueError, match="cannot fill"):
        make_folds(_patients(1, 2), n_splits=5)


def test_at_least_two_folds_are_required():
    with pytest.raises(ValueError, match="at least 2"):
        make_folds(_patients(10, 40), n_splits=1)


def test_bootstrap_resamples_patients_not_rows():
    """Rows of one patient are not independent; resampling them narrows the CI."""
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 40)
    # Overlapping distributions: a perfectly separable set would put every
    # replicate at AUROC 1.0 and both intervals would collapse to a point.
    scores = np.concatenate([rng.normal(0.45, 0.2, 40), rng.normal(0.58, 0.2, 40)])
    per_row = np.arange(80)                     # each row its own patient
    per_pair = np.repeat(np.arange(40), 2)      # two rows per patient, label-pure

    grouped = bootstrap_interval(labels, scores, per_pair, samples=800, seed=1)
    ungrouped = bootstrap_interval(labels, scores, per_row, samples=800, seed=1)
    assert 0.0 < (ungrouped["upper"] - ungrouped["lower"]) < 1.0
    assert (grouped["upper"] - grouped["lower"]) > (ungrouped["upper"] - ungrouped["lower"])


def test_bootstrap_reports_replicate_count_and_point_estimate():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = bootstrap_interval(labels, scores, np.array(["a", "b", "c", "d"]), samples=200)
    assert result["estimate"] == 1.0
    assert 0 < result["replicates"] <= 200
    assert result["lower"] <= result["estimate"] <= result["upper"]


def _cv_config(base_config: dict) -> object:
    payload = json.loads(json.dumps(base_config))
    payload["experiment"] = "cv"
    payload["modalities"] = ["clinical"]
    payload["fusion"] = {"name": "clinical_only", "embed_dim": 32, "pooling": "mean"}
    payload["optim"] = {"epochs": 1, "warmup_epochs": 0, "early_stopping_patience": 2}
    return config_from_dict(payload)


def test_cross_validation_scores_each_sample_exactly_once(base_config, tmp_path):
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    root = Path(result["run_dir"])
    pooled = pd.read_csv(root / "out_of_fold_predictions.csv", dtype={"sample_id": str})
    assert not pooled["sample_id"].duplicated().any()

    index = pd.read_csv(Path(cfg.data.run_dir) / "index.csv", dtype={"sample_id": str})
    expected = set(index.loc[index["split"].isin(["train", "val"]), "sample_id"])
    assert set(pooled["sample_id"]) == expected, "out-of-fold coverage is not complete"


def test_cross_validation_never_scores_a_patient_it_fitted_on(base_config, tmp_path):
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    root = Path(result["run_dir"])
    pooled = pd.read_csv(
        root / "out_of_fold_predictions.csv", dtype={"subject_id": str, "sample_id": str}
    )
    # Each patient is scored by exactly one fold; that fold did not fit on them.
    per_patient = pooled.groupby("subject_id")["fold"].nunique()
    assert per_patient.eq(1).all(), "a patient was scored by more than one fold"


def test_cross_validation_leaves_the_frozen_test_split_alone(base_config, tmp_path):
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    root = Path(result["run_dir"])
    index = pd.read_csv(Path(cfg.data.run_dir) / "index.csv", dtype={"sample_id": str})
    test_ids = set(index.loc[index["split"].eq("test"), "sample_id"])
    pooled = pd.read_csv(root / "out_of_fold_predictions.csv", dtype={"sample_id": str})
    assert not (set(pooled["sample_id"]) & test_ids), "test samples leaked into the folds"

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_split_untouched"] is True


def test_cross_validation_writes_the_reporting_artefacts(base_config, tmp_path):
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    root = Path(result["run_dir"])
    for name in ("cv_metrics.json", "per_fold_metrics.csv", "out_of_fold_predictions.csv",
                 "summary.csv", "summary.md", "run_manifest.json"):
        assert (root / name).is_file(), name
    for name in ("roc_out_of_fold.png", "pr_out_of_fold.png",
                 "score_distribution_out_of_fold.png"):
        assert (root / "figures" / name).is_file(), name

    metrics = json.loads((root / "cv_metrics.json").read_text(encoding="utf-8"))
    assert metrics["pooled"]["auroc_ci"]["replicates"] > 0
    assert len(metrics["per_fold"]) == 3


def test_fold_checkpoints_are_discarded_by_default(base_config, tmp_path):
    """A frozen backbone repeats unchanged in every fold checkpoint."""
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    root = Path(result["run_dir"])
    # Only the folds are cleared; the deliverable model keeps its checkpoint.
    assert not list((root / "folds").rglob("checkpoint_*.pt"))
    assert (root / "out_of_fold_predictions.csv").is_file()

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fold_checkpoints_kept"] is False


def test_fold_checkpoints_can_be_kept_on_request(base_config, tmp_path):
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit",
        bootstrap_samples=50, inner_splits=2, keep_fold_checkpoints=True,
    )
    root = Path(result["run_dir"])
    # Per outer fold: two inner models that choose the budget, and the refit.
    assert len(list((root / "folds").rglob("checkpoint_best.pt"))) == 9


def test_a_deliverable_model_is_fitted_on_every_patient(base_config, tmp_path):
    """Cross-validation estimates performance; it does not leave a model to ship.

    The folds each hold out a different group, so none of them was fitted on all
    the data. The deliverable is a separate fit over every train+val patient,
    with its epoch budget taken from what the folds used, since there is no
    held-out split left to stop on.
    """
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    final = result["final_model"]
    assert final is not None
    assert Path(final["checkpoint"]).is_file()

    from bn5212_training.model import load_checkpoint

    model, payload = load_checkpoint(final["checkpoint"])
    assert payload["config"]["experiment"].endswith("_final")

    # It must have seen every patient the folds drew from.
    patients = pd.read_csv(
        Path(result["run_dir"]) / "out_of_fold_predictions.csv", dtype={"subject_id": str}
    )["subject_id"].nunique()
    assert final["patients"] == patients


def test_the_deliverable_can_be_skipped(base_config, tmp_path):
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit",
        bootstrap_samples=50, inner_splits=2, train_final_model=False,
    )
    assert result["final_model"] is None


def test_the_deliverables_reported_score_is_the_out_of_fold_one(base_config, tmp_path):
    """Scoring the final model on its own training data would flatter it."""
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, output_dir=tmp_path, run_id="unit", bootstrap_samples=50, inner_splits=2
    )
    manifest = json.loads(
        (Path(result["run_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert "out-of-fold" in manifest["final_model"]["note"]
    # The headline metrics in the run stay the pooled ones.
    assert manifest["pooled_metrics"]["auroc"] == result["pooled"]["auroc"]


def test_the_held_out_fold_never_chooses_the_checkpoint(base_config, tmp_path, monkeypatch):
    """Early stopping on the fold being scored inflates the pooled estimate.

    Measured on the ICU cohort: stopping each fold on its own held-out patients
    reported AUROC 0.720, while any epoch budget fixed in advance gave 0.64-0.67.
    """
    from bn5212_training import trainer

    calls = []
    original = trainer.train

    def spy(cfg, **kwargs):
        calls.append(kwargs)
        return original(cfg, **kwargs)

    monkeypatch.setattr(trainer, "train", spy)
    run_cross_validation(
        _cv_config(base_config), n_splits=3, inner_splits=2, output_dir=tmp_path,
        run_id="unit", bootstrap_samples=50, train_final_model=False,
    )

    # Per outer fold: two inner fits that choose the budget, then the refit that
    # scores the held-out patients.
    assert len(calls) == 9
    for start in range(0, len(calls), 3):
        *inner, refit = calls[start:start + 3]
        scored = set(refit["score_subjects"])
        assert scored and refit.get("select_subjects") is None
        assert refit["epoch_budget"] >= 1
        assert not (set(refit["fit_subjects"]) & scored), "the scored patients were fitted on"
        for call in inner:
            fitted, selecting = set(call["fit_subjects"]), set(call["select_subjects"])
            assert selecting and call.get("score_subjects") is None
            assert not (selecting & scored), "the scored patients took part in selection"
            assert not (fitted & scored), "the scored patients shaped an inner model"
            assert not (fitted & selecting), "selection ran on fitted patients"


def test_overlapping_patient_groups_are_rejected(base_config, tmp_path):
    from bn5212_training.crossval import _patient_table
    from bn5212_training.trainer import train

    cfg = _cv_config(base_config)
    subjects = sorted(_patient_table(cfg)["subject_id"])
    fitted, selecting, scored = subjects[:-8], subjects[-8:-4], subjects[-4:]
    with pytest.raises(ValueError, match="disjoint"):
        train(cfg, output_dir=tmp_path, fit_subjects=fitted,
              select_subjects=scored, score_subjects=scored)
    with pytest.raises(ValueError, match="disjoint"):
        train(cfg, output_dir=tmp_path, fit_subjects=fitted + scored,
              select_subjects=selecting, score_subjects=scored)
    with pytest.raises(ValueError, match="exactly one"):
        train(cfg, output_dir=tmp_path, fit_subjects=fitted, score_subjects=scored)


def test_an_epoch_budget_truncates_the_run_but_keeps_the_schedule(base_config, tmp_path):
    """The deliverable must train the way the selecting folds did, only shorter."""
    from dataclasses import replace

    from bn5212_training.crossval import _patient_table
    from bn5212_training.trainer import train

    cfg = _cv_config(base_config)
    cfg = replace(cfg, optim=replace(cfg.optim, epochs=6, warmup_epochs=1))
    subjects = sorted(_patient_table(cfg)["subject_id"])
    selected = train(cfg, output_dir=tmp_path, run_id="selected",
                     fit_subjects=subjects[:-6], select_subjects=subjects[-6:])
    budgeted = train(cfg, output_dir=tmp_path, run_id="budgeted",
                     fit_subjects=subjects, epoch_budget=3)

    assert len(budgeted["history"]) == 3
    assert budgeted["validation_metrics"]["selected_epoch"] == 3
    assert [row["lr"] for row in budgeted["history"]] == [
        row["lr"] for row in selected["history"][:3]
    ]


def test_the_benchmark_hand_off_matches_the_frozen_splits(base_config, tmp_path):
    """Validation scores must be out-of-fold: the deliverable has fitted on val."""
    cfg = _cv_config(base_config)
    result = run_cross_validation(
        cfg, n_splits=3, inner_splits=2, output_dir=tmp_path, run_id="unit",
        bootstrap_samples=50,
    )
    root = Path(result["run_dir"])
    index = pd.read_csv(Path(cfg.data.run_dir) / "index.csv", dtype={"sample_id": str})
    pooled = pd.read_csv(root / "out_of_fold_predictions.csv", dtype={"sample_id": str})

    for split in ("val", "test"):
        frame = pd.read_csv(root / f"predictions_{split}.csv", dtype={"sample_id": str})
        assert list(frame.columns) == ["sample_id", "y_score"]
        assert set(frame["sample_id"]) == set(index.loc[index["split"].eq(split), "sample_id"])

    handed = pd.read_csv(root / "predictions_val.csv", dtype={"sample_id": str})
    merged = handed.merge(pooled, on="sample_id", suffixes=("", "_oof"))
    assert np.allclose(merged["y_score"], merged["y_score_oof"])


def test_a_paired_difference_is_tighter_than_two_separate_intervals():
    """Two models scored on the same patients share most of their sampling noise."""
    rng = np.random.default_rng(3)
    labels = np.repeat([0, 1], 60)
    base = np.concatenate([rng.normal(0.4, 0.2, 60), rng.normal(0.6, 0.2, 60)])
    first = base + rng.normal(0, 0.02, 120)
    second = base + rng.normal(0, 0.02, 120)
    patients = np.arange(120)

    paired = paired_bootstrap_difference(labels, first, second, patients, samples=600, seed=1)
    alone = bootstrap_interval(labels, first, patients, samples=600, seed=1)
    assert paired["lower"] <= paired["estimate"] <= paired["upper"]
    assert (paired["upper"] - paired["lower"]) < (alone["upper"] - alone["lower"])
    # Near-identical models: the difference must not be called real.
    assert paired["lower"] < 0 < paired["upper"]


def test_a_paired_difference_detects_a_model_that_is_clearly_better():
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1], 60)
    strong = np.concatenate([rng.normal(0.2, 0.1, 60), rng.normal(0.8, 0.1, 60)])
    noise = rng.normal(0.5, 0.2, 120)
    result = paired_bootstrap_difference(labels, strong, noise, np.arange(120), samples=400)
    assert result["lower"] > 0


def test_paired_difference_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap_difference(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.5]), np.array(["a", "b"])
        )


def test_the_interval_figure_is_written_with_its_table(tmp_path):
    from bn5212_training import plots

    rows = [
        {"label": "Clinical-only", "estimate": 0.61, "lower": 0.46, "upper": 0.76},
        {"label": "CXR-only", "estimate": 0.55, "lower": 0.42, "upper": 0.68},
    ]
    path = plots.interval_comparison(rows, tmp_path / "auroc.png", title="t", xlabel="AUROC")
    assert path.is_file() and path.stat().st_size > 0
    table = pd.read_csv(path.with_suffix(".csv"))
    assert list(table["label"]) == ["Clinical-only", "CXR-only"]


def test_the_roc_comparison_labels_each_curve_with_its_own_auroc(tmp_path):
    from bn5212_training import plots
    from bn5212_training.metrics import auroc

    rng = np.random.default_rng(5)
    labels = np.repeat([0, 1], 30)
    curves = {
        "strong": (labels, np.concatenate([rng.normal(0.3, 0.1, 30), rng.normal(0.7, 0.1, 30)])),
        "weak": (labels, rng.normal(0.5, 0.2, 60)),
    }
    path = plots.roc_comparison(curves, tmp_path / "roc.png", title="t")
    assert path.is_file() and path.stat().st_size > 0
    table = pd.read_csv(path.with_suffix(".csv"))
    assert set(table["model"]) == {"strong", "weak"}
    # Every curve runs corner to corner, and nothing but curve geometry is written.
    assert list(table.columns) == ["model", "false_positive_rate", "true_positive_rate"]
    for _, part in table.groupby("model"):
        assert part.iloc[0, 1:].tolist() == [0.0, 0.0] and part.iloc[-1, 1:].tolist() == [1.0, 1.0]
    assert auroc(*curves["strong"]) > auroc(*curves["weak"])


def test_a_fifth_curve_is_rejected_rather_than_given_a_generated_colour(tmp_path):
    from bn5212_training import plots

    labels, scores = np.array([0, 1]), np.array([0.2, 0.8])
    with pytest.raises(ValueError, match="At most"):
        plots.roc_comparison({str(i): (labels, scores) for i in range(5)},
                             tmp_path / "roc.png", title="t")
