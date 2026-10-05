"""Validation metrics for checkpoint selection and cross-validation.

Test metrics come from the benchmark-evaluation project; this module never
touches the test split. Implemented in numpy, so no scikit-learn is needed.
"""
from __future__ import annotations

import numpy as np


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """Average ranks, so tied scores do not bias AUROC."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    assigned = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        assigned[start : stop + 1] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    ranks[order] = assigned
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUROC. None when only one class is present (never faked)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    num_pos = int(positives.sum())
    num_neg = int((labels == 0).sum())
    if num_pos == 0 or num_neg == 0:
        return None
    ranks = _rank_with_ties(scores)
    return float(
        (ranks[positives].sum() - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    )


def auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Average precision, the step-wise sum used by average_precision_score."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if int((labels == 1).sum()) == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    true_positives = np.cumsum(ordered == 1)
    predicted = np.arange(1, len(ordered) + 1)
    precision = true_positives / predicted
    recall = true_positives / true_positives[-1]
    recall_gain = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * recall_gain).sum())


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean((scores - labels) ** 2))


def selection_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "brier": brier(labels, scores),
        "prevalence": float(labels.mean()) if len(labels) else None,
        "n": int(len(labels)),
        "n_positive": int((labels == 1).sum()),
    }


def bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str = "auroc",
    samples: int = 2000,
    seed: int = 5212,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Patient-level bootstrap interval for a ranking metric.

    Patients are resampled, not rows. Replicates holding a single class are
    skipped, and the number of usable replicates is returned.
    """
    function = {"auroc": auroc, "auprc": auprc}.get(metric)
    if function is None:
        raise ValueError(f"Unsupported bootstrap metric {metric!r}")
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)

    unique = np.unique(groups)
    rows_of = {key: np.flatnonzero(groups == key) for key in unique}
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(samples)):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([rows_of[key] for key in drawn])
        value = function(labels[index], scores[index])
        if value is not None:
            estimates.append(value)

    point = function(labels, scores)
    if not estimates:
        return {"estimate": point, "lower": None, "upper": None, "replicates": 0}
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": point,
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "replicates": len(estimates),
    }


def paired_bootstrap_difference(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str = "auroc",
    samples: int = 2000,
    seed: int = 5212,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Patient-level bootstrap interval for metric(a) - metric(b).

    Both models are scored on the same resampled patients in every replicate.
    """
    function = {"auroc": auroc, "auprc": auprc}.get(metric)
    if function is None:
        raise ValueError(f"Unsupported bootstrap metric {metric!r}")
    labels = np.asarray(labels).astype(int)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    if not (len(labels) == len(scores_a) == len(scores_b) == len(groups)):
        raise ValueError("labels, both score arrays and groups must be aligned row by row")

    unique = np.unique(groups)
    rows_of = {key: np.flatnonzero(groups == key) for key in unique}
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(int(samples)):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([rows_of[key] for key in drawn])
        first = function(labels[index], scores_a[index])
        second = function(labels[index], scores_b[index])
        if first is not None and second is not None:
            differences.append(first - second)

    whole_a, whole_b = function(labels, scores_a), function(labels, scores_b)
    point = None if whole_a is None or whole_b is None else whole_a - whole_b
    if not differences:
        return {"estimate": point, "lower": None, "upper": None, "replicates": 0}
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": point,
        "lower": float(np.quantile(differences, tail)),
        "upper": float(np.quantile(differences, 1.0 - tail)),
        "replicates": len(differences),
    }


def confusion_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Counts at a threshold; a score equal to the threshold is predicted positive."""
    labels = np.asarray(labels).astype(int)
    predicted = np.asarray(scores, dtype=np.float64) >= threshold
    return {
        "tn": int(((labels == 0) & ~predicted).sum()),
        "fp": int(((labels == 0) & predicted).sum()),
        "fn": int(((labels == 1) & ~predicted).sum()),
        "tp": int(((labels == 1) & predicted).sum()),
    }


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The observed score maximising sensitivity + specificity - 1.

    Same rule as benchmark-evaluation, ties included: the candidate nearer 0.5
    wins, then the larger one.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are needed to choose a threshold")
    ranked = []
    for threshold in np.unique(scores):
        counts = confusion_counts(labels, scores, float(threshold))
        sensitivity = counts["tp"] / (counts["tp"] + counts["fn"])
        specificity = counts["tn"] / (counts["tn"] + counts["fp"])
        ranked.append((sensitivity + specificity - 1.0, -abs(float(threshold) - 0.5), float(threshold)))
    return max(ranked)[2]


def cross_fitted_decisions(labels: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Positive/negative decisions for out-of-fold scores, without peeking.

    Each fold is thresholded at the Youden point of the other folds, so the
    labels of the samples being classified never choose their own threshold.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    folds = np.asarray(folds)
    decisions = np.zeros(len(labels), dtype=bool)
    for fold in np.unique(folds):
        held = folds == fold
        decisions[held] = scores[held] >= youden_threshold(labels[~held], scores[~held])
    return decisions


def decision_summary(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    """Confusion counts and the rates read off them."""
    labels = np.asarray(labels).astype(int)
    counts = confusion_counts(labels, np.asarray(decisions, dtype=np.float64), 0.5)
    positives, negatives = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
    flagged = counts["tp"] + counts["fp"]
    return {
        **counts,
        "sensitivity": counts["tp"] / positives if positives else None,
        "specificity": counts["tn"] / negatives if negatives else None,
        "precision": counts["tp"] / flagged if flagged else None,
        "accuracy": (counts["tp"] + counts["tn"]) / max(len(labels), 1),
    }


def is_better(metric: str, candidate: float | None, incumbent: float | None) -> bool:
    """Direction-aware comparison for checkpoint selection."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return candidate < incumbent if metric == "loss" else candidate > incumbent
