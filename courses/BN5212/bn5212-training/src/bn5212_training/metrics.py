"""Validation metrics used for checkpoint and early-stopping decisions.

SCOPE: selection only. The reported benchmark numbers -- test metrics, confidence
intervals, threshold selection, admission-level aggregation -- come from the
benchmark-evaluation project, which is the single source of truth for results.
Computing them twice would invite two different tables in the final report, so
this module deliberately stays minimal and never touches the test split.

Implemented in numpy so the training project needs no scikit-learn.
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


def is_better(metric: str, candidate: float | None, incumbent: float | None) -> bool:
    """Direction-aware comparison for checkpoint selection."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return candidate < incumbent if metric == "loss" else candidate > incumbent
