"""Static benchmark plots used both as files and inside the HTML report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


PLOT_FILES = {
    "roc": "roc_curve.png",
    "precision_recall": "precision_recall_curve.png",
    "calibration": "calibration_curve.png",
    "confusion_matrix": "confusion_matrix.png",
}


def _finish(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _unavailable_plot(path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(6.4, 4.8))
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    _finish(path)


def save_evaluation_plots(predictions: pd.DataFrame, threshold: float, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    truth = predictions["label"].to_numpy(dtype=int)
    score = predictions["y_score"].to_numpy(dtype=float)
    paths = {name: output_dir / filename for name, filename in PLOT_FILES.items()}

    if len(np.unique(truth)) == 2:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        RocCurveDisplay.from_predictions(truth, score, ax=ax, name="Test set")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7c8798", linewidth=1)
        ax.set_title("Receiver operating characteristic")
        _finish(paths["roc"])
    else:
        _unavailable_plot(paths["roc"], "Receiver operating characteristic", "AUROC requires both outcome classes.")

    if truth.sum() > 0:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        PrecisionRecallDisplay.from_predictions(truth, score, ax=ax, name="Test set")
        ax.axhline(float(truth.mean()), linestyle="--", color="#7c8798", linewidth=1, label="Prevalence")
        ax.legend(loc="best")
        ax.set_title("Precision-recall curve")
        _finish(paths["precision_recall"])
    else:
        _unavailable_plot(paths["precision_recall"], "Precision-recall curve", "AUPRC requires at least one positive outcome.")

    if len(np.unique(score)) > 1:
        bins = min(10, max(2, len(score) // 5))
        observed, predicted = calibration_curve(truth, score, n_bins=bins, strategy="quantile")
        _, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7c8798", linewidth=1, label="Perfect calibration")
        ax.plot(predicted, observed, marker="o", color="#1769aa", label="Test set")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate", title="Calibration")
        ax.legend(loc="best")
        _finish(paths["calibration"])
    else:
        _unavailable_plot(paths["calibration"], "Calibration", "A constant prediction does not form a calibration curve.")

    predicted_label = (score >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted_label, labels=[0, 1])
    _, ax = plt.subplots(figsize=(5.6, 4.8))
    ConfusionMatrixDisplay(matrix, display_labels=["Survived", "Died"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix at threshold {threshold:.3f}")
    _finish(paths["confusion_matrix"])
    return paths
