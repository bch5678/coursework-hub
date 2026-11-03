"""PNG figures and plain-text tables for a training run.

Every figure is also written as CSV next to it. Nothing here plots or scores
the test split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .metrics import auroc as rank_auroc  # noqa: E402

# Validated categorical slots, assigned in fixed order and never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8887"
SURFACE = "#fcfcfb"
GRID = "#e3e2de"
# Single-hue sequential ramp, light -> dark, for magnitude (attention weights).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "bn5212_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

LINE_WIDTH = 2.0
MARKER_SIZE = 5.0  # ~10px diameter


def _style_axes(axes: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axes.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)
    for side, spine in axes.spines.items():
        spine.set_visible(side in {"left", "bottom"})
        spine.set_color(GRID)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def training_curves(history: Sequence[Mapping[str, float]], path: str | Path) -> Path:
    """Loss and validation ranking metrics, on separate panels.

    Loss and AUROC live on different scales, so they get their own axes rather
    than a second y-axis on one plot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(history))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)
    epochs = frame["epoch"]

    for index, (column, label) in enumerate([("train_loss", "Train"), ("val_loss", "Validation")]):
        if column in frame:
            top.plot(
                epochs,
                frame[column],
                color=SERIES[index],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=label,
            )
    _style_axes(top, "Loss", "", "Weighted BCE")
    top.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")

    for index, (column, label) in enumerate([("val_auroc", "AUROC"), ("val_auprc", "AUPRC")]):
        if column in frame and frame[column].notna().any():
            bottom.plot(
                epochs,
                frame[column],
                color=SERIES[index],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=label,
            )
    bottom.axhline(0.5, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
    bottom.annotate(
        "chance (AUROC 0.5)",
        xy=(epochs.iloc[0] if len(epochs) else 0, 0.5),
        xytext=(2, 4),
        textcoords="offset points",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    _style_axes(bottom, "Validation ranking metrics (selection only)", "Epoch", "Score")
    bottom.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")
    # Keep headroom above the chance line so its label never sits on the frame.
    bottom.set_ylim(top=max(0.62, bottom.get_ylim()[1]))
    # Epochs are whole numbers; matplotlib would otherwise offer 1.5, 2.5, ...
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

    frame.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    return _save(figure, path)


def _roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    true_positive = np.concatenate([[0.0], np.cumsum(ordered == 1)])
    false_positive = np.concatenate([[0.0], np.cumsum(ordered == 0)])
    positives = max(true_positive[-1], 1.0)
    negatives = max(false_positive[-1], 1.0)
    return false_positive / negatives, true_positive / positives


def roc_curve(labels: np.ndarray, scores: np.ndarray, path: str | Path, *, auroc=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    false_positive, true_positive = _roc_points(labels, scores)

    figure, axes = plt.subplots(figsize=(5.6, 5.0))
    axes.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
    axes.plot(false_positive, true_positive, color=SERIES[0], linewidth=LINE_WIDTH)
    title = "Validation ROC" if auroc is None else f"Validation ROC  -  AUROC {auroc:.3f}"
    _style_axes(axes, title, "False positive rate", "True positive rate")
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    pd.DataFrame({"false_positive_rate": false_positive, "true_positive_rate": true_positive}).to_csv(
        path.with_suffix(".csv"), index=False, lineterminator="\n"
    )
    return _save(figure, path)


def roc_comparison(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: str | Path, *, title: str
) -> Path:
    """ROC curves of several models scored on the same samples, on one axis.

    curves maps a model name to its (labels, scores). The palette has four
    validated slots and they are never cycled, so a fifth model is rejected
    rather than given a generated colour.
    """
    if len(curves) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} curves fit one figure; split the comparison")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6.0, 5.4))
    axes.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
    points = []
    for index, (name, (labels, scores)) in enumerate(curves.items()):
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores, dtype=float)
        false_positive, true_positive = _roc_points(labels, scores)
        # Computed here from the same scores, so the legend cannot drift from
        # the curve; rank-based, which is what the result tables report.
        axes.plot(
            false_positive, true_positive, color=SERIES[index], linewidth=LINE_WIDTH,
            label=f"{name}  (AUROC {rank_auroc(labels, scores):.3f})",
        )
        points.append(pd.DataFrame({
            "model": name,
            "false_positive_rate": false_positive,
            "true_positive_rate": true_positive,
        }))
    _style_axes(axes, title, "False positive rate", "True positive rate")
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower right")
    pd.concat(points, ignore_index=True).to_csv(
        path.with_suffix(".csv"), index=False, lineterminator="\n"
    )
    return _save(figure, path)


def precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray, path: str | Path, *, auprc=None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    true_positive = np.cumsum(ordered == 1)
    predicted = np.arange(1, len(ordered) + 1)
    precision = true_positive / predicted
    recall = true_positive / max(true_positive[-1], 1)
    prevalence = float(labels.mean()) if len(labels) else 0.0

    figure, axes = plt.subplots(figsize=(5.6, 5.0))
    axes.axhline(prevalence, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
    axes.annotate(
        f"prevalence {prevalence:.3f}",
        xy=(0.02, prevalence),
        xytext=(0, 4),
        textcoords="offset points",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    axes.plot(recall, precision, color=SERIES[0], linewidth=LINE_WIDTH)
    title = "Validation precision-recall"
    if auprc is not None:
        title = f"{title}  -  AUPRC {auprc:.3f}"
    _style_axes(axes, title, "Recall", "Precision")
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
        path.with_suffix(".csv"), index=False, lineterminator="\n"
    )
    return _save(figure, path)


def score_distribution(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> Path:
    """Predicted probability by true outcome - shows separation and calibration drift."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    bins = np.linspace(0.0, 1.0, 21)

    figure, axes = plt.subplots(figsize=(6.4, 4.4))
    for index, (value, label) in enumerate([(0, "Survived"), (1, "Died in hospital")]):
        subset = scores[labels == value]
        if subset.size:
            axes.hist(
                subset,
                bins=bins,
                color=SERIES[index],
                alpha=0.75,
                label=f"{label} (n={subset.size})",
                edgecolor=SURFACE,
                linewidth=1.0,
            )
    _style_axes(axes, "Validation predicted probability by outcome", "P(in-hospital mortality)", "Samples")
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    counts = pd.DataFrame({"bin_left": bins[:-1], "bin_right": bins[1:]})
    for value, name in [(0, "survived"), (1, "died")]:
        counts[name] = np.histogram(scores[labels == value], bins=bins)[0]
    counts.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    return _save(figure, path)


def attention_matrix(
    weights: np.ndarray, variable_names: Sequence[str], path: str | Path
) -> Path:
    """Clinical variable x image patch attention, averaged over heads and samples.

    This is the RQ3 figure: each row is one clinical variable, each column one
    image patch, so a row shows where that variable looked.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(f"Expected a 2D [variables, patches] matrix, got {weights.shape}")
    height = max(3.0, 0.32 * len(variable_names) + 1.6)

    figure, axes = plt.subplots(figsize=(8.0, height))
    image = axes.imshow(weights, aspect="auto", cmap=SEQUENTIAL, interpolation="nearest")
    axes.set_yticks(range(len(variable_names)))
    axes.set_yticklabels(variable_names, fontsize=8)
    _style_axes(axes, "Clinical-to-image cross-attention", "Image patch token", "")
    axes.grid(False)
    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    bar.set_label("Mean attention weight", color=TEXT_SECONDARY, fontsize=8)

    pd.DataFrame(weights, index=list(variable_names)).to_csv(
        path.with_suffix(".csv"), index_label="variable", lineterminator="\n"
    )
    return _save(figure, path)


def attention_patch_maps(
    weights: np.ndarray,
    variable_names: Sequence[str],
    grid_size: int,
    path: str | Path,
    *,
    max_panels: int = 12,
) -> Path:
    """Per-variable attention folded back onto the image patch grid."""
    path = Path(path)
    weights = np.asarray(weights, dtype=float)
    count = min(len(variable_names), max_panels)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))

    # One colour scale across every panel. Letting each panel normalise itself
    # would make a variable with uniformly weak attention look identical to one
    # with strong attention, and the whole point is to compare variables.
    panel_data = [
        weights[position][: grid_size * grid_size].reshape(grid_size, grid_size)
        for position in range(count)
    ]
    low = float(min(panel.min() for panel in panel_data))
    high = float(max(panel.max() for panel in panel_data))
    if high <= low:
        high = low + 1e-12

    figure, axes_grid = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.9 * rows))
    panels = np.atleast_1d(axes_grid).ravel()
    drawn = None
    for position in range(len(panels)):
        axes = panels[position]
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_color(GRID)
        if position >= count:
            axes.set_visible(False)
            continue
        drawn = axes.imshow(
            panel_data[position],
            cmap=SEQUENTIAL,
            interpolation="nearest",
            vmin=low,
            vmax=high,
        )
        axes.set_title(variable_names[position], fontsize=7.5, color=TEXT_PRIMARY, pad=5)

    figure.suptitle(
        "Where each clinical variable attends on the radiograph",
        fontsize=11,
        color=TEXT_PRIMARY,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.subplots_adjust(hspace=0.45)
    if drawn is not None:
        bar = figure.colorbar(
            drawn, ax=list(panels), orientation="horizontal", fraction=0.03, pad=0.05
        )
        bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        bar.set_label("Mean attention weight (shared scale)", color=TEXT_SECONDARY, fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor(SURFACE)
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def interval_comparison(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    title: str,
    xlabel: str,
    reference: float | None = 0.5,
    reference_label: str = "chance",
) -> Path:
    """One estimate with its interval per model, on a shared axis.

    rows carry label, estimate, lower and upper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    positions = np.arange(len(frame))[::-1]

    figure, axes = plt.subplots(figsize=(7.6, 0.62 * len(frame) + 1.5))
    if reference is not None:
        axes.axvline(reference, color=TEXT_MUTED, linewidth=1.0)
        axes.annotate(
            reference_label,
            xy=(reference, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -10),
            textcoords="offset points",
            color=TEXT_MUTED,
            fontsize=7.5,
        )
    axes.hlines(
        positions, frame["lower"], frame["upper"], color=SERIES[0], linewidth=LINE_WIDTH,
        capstyle="round",
    )
    axes.plot(
        frame["estimate"], positions, linestyle="none", marker="o", color=SERIES[0],
        markersize=MARKER_SIZE + 2.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
    )
    low = min(float(frame["lower"].min()), reference if reference is not None else np.inf)
    high = float(frame["upper"].max())
    pad = 0.04 * (high - low)
    # Room on the right for the value labels, which stay in text ink.
    axes.set_xlim(low - pad, high + 0.42 * (high - low))
    for position, record in zip(positions, frame.to_dict(orient="records")):
        axes.annotate(
            f"{record['estimate']:.3f}  [{record['lower']:.3f}, {record['upper']:.3f}]",
            xy=(high, position),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=8,
        )
    _style_axes(axes, title, xlabel, "")
    # The label gutter is not part of the scale, so it carries no ticks or grid.
    axes.set_xticks([tick for tick in axes.get_xticks() if low - pad <= tick <= high + pad])
    axes.set_yticks(positions)
    axes.set_yticklabels(frame["label"], color=TEXT_PRIMARY, fontsize=9)
    axes.set_ylim(-0.6, len(frame) - 0.4)
    axes.grid(False, axis="y")

    frame.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    return _save(figure, path)


def write_summary_table(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    """Write the run summary as CSV plus an aligned Markdown table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")

    def render(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    header = list(frame.columns)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for record in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(render(record[name]) for name in header) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.with_suffix(".md")
