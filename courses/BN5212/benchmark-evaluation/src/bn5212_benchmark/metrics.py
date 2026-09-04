"""Binary outcome metrics, validation threshold selection and clustered intervals."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _arrays(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or score.ndim != 1 or len(truth) != len(score) or len(truth) == 0:
        raise ValueError("y_true and y_score must be nonempty one-dimensional arrays of equal length")
    if not set(np.unique(truth)) <= {0, 1}:
        raise ValueError("y_true must contain only binary 0/1 labels")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("y_score must contain finite probabilities in [0, 1]")
    return truth, score


def confusion_counts(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> dict[str, int]:
    truth, score = _arrays(y_true, y_score)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and inside [0, 1]")
    predicted = score >= threshold
    return {
        "tn": int(((truth == 0) & ~predicted).sum()),
        "fp": int(((truth == 0) & predicted).sum()),
        "fn": int(((truth == 1) & ~predicted).sum()),
        "tp": int(((truth == 1) & predicted).sum()),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
) -> dict[str, int | float | None]:
    truth, score = _arrays(y_true, y_score)
    counts = confusion_counts(truth, score, threshold)
    tn, fp, fn, tp = (counts[name] for name in ["tn", "fp", "fn", "tp"])
    precision = _safe_ratio(tp, tp + fp)
    sensitivity = _safe_ratio(tp, tp + fn) if tp + fn else None
    specificity = _safe_ratio(tn, tn + fp) if tn + fp else None
    metrics: dict[str, int | float | None] = {
        "n": int(len(truth)),
        "positives": int(truth.sum()),
        "prevalence": float(truth.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else None,
        "auprc": float(average_precision_score(truth, score)) if truth.sum() > 0 else None,
        "accuracy": _safe_ratio(tp + tn, len(truth)),
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": (
            _safe_ratio(2 * precision * sensitivity, precision + sensitivity)
            if sensitivity is not None
            else None
        ),
        "brier": float(brier_score_loss(truth, score)),
        **counts,
    }
    return metrics


def choose_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    method: str = "youden_j",
) -> tuple[float, dict[str, float | str]]:
    truth, score = _arrays(y_true, y_score)
    if method == "fixed_0.5":
        return 0.5, {"method": method, "objective": 0.0}
    if method not in {"youden_j", "f1"}:
        raise ValueError("threshold method must be one of: fixed_0.5, youden_j, f1")
    if len(np.unique(truth)) != 2:
        raise ValueError("Validation split must contain both classes to select a data-driven threshold")

    # Data-driven strategies choose an observed validation score. A fixed 0.5
    # decision rule is available explicitly as ``fixed_0.5`` above.
    candidates = np.unique(score)
    ranked: list[tuple[float, float, float]] = []
    for threshold in candidates:
        counts = confusion_counts(truth, score, float(threshold))
        sensitivity = _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        specificity = _safe_ratio(counts["tn"], counts["tn"] + counts["fp"])
        precision = _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        objective = (
            sensitivity + specificity - 1
            if method == "youden_j"
            else _safe_ratio(2 * precision * sensitivity, precision + sensitivity)
        )
        ranked.append((float(objective), -abs(float(threshold) - 0.5), float(threshold)))
    objective, _, threshold = max(ranked)
    return threshold, {"method": method, "objective": objective}


def patient_bootstrap_intervals(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    samples: int = 1000,
    seed: int = 5212,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int | None]]:
    required = {"subject_id", "label", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Bootstrap predictions are missing columns: {sorted(missing)}")
    if samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be inside (0, 1)")
    if samples == 0:
        return {}

    groups = [group for _, group in predictions.groupby("subject_id", sort=True)]
    if not groups:
        raise ValueError("Cannot bootstrap an empty prediction table")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(samples):
        sampled = [groups[index] for index in rng.integers(0, len(groups), size=len(groups))]
        replicate = pd.concat(sampled, ignore_index=True)
        current = binary_metrics(replicate["label"], replicate["y_score"], threshold)
        for name, value in current.items():
            if name in {"n", "positives", "tn", "fp", "fn", "tp", "threshold"}:
                continue
            if value is not None and math.isfinite(float(value)):
                values.setdefault(name, []).append(float(value))

    alpha = (1 - confidence) / 2
    result: dict[str, dict[str, float | int | None]] = {}
    for name, observed in binary_metrics(predictions["label"], predictions["y_score"], threshold).items():
        if name in {"n", "positives", "tn", "fp", "fn", "tp", "threshold"}:
            continue
        draws = values.get(name, [])
        result[name] = {
            "estimate": None if observed is None else float(observed),
            "lower": float(np.quantile(draws, alpha)) if draws else None,
            "upper": float(np.quantile(draws, 1 - alpha)) if draws else None,
            "successful_replicates": len(draws),
        }
    return result
