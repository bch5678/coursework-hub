"""Redraw six v3 figures with ICU-transfer labels; leave raw outputs intact."""
from pathlib import Path
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "courses/BN5212/bn5212-training/outputs/three_multimodal/transfer_multimodal_v3_20261030_161749"
MODELS = {"concat": "Concat", "joint": "Joint attention", "cross_attention_adapter": "Cross-attention Adapter"}
LABELS = ["No ICU transfer", "ICU transfer"]
manifest = {"task": "icu_transfer", "run_id": RUN.name, "models": {}}

for model, name in MODELS.items():
    source = RUN / "benchmark" / model
    prediction_path = source / "predictions.csv"
    metrics_path = source / "metrics.json"
    rows = list(csv.DictReader(prediction_path.open()))
    metrics = json.loads(metrics_path.read_text())
    threshold = metrics["threshold_selection"]["threshold"]
    assert metrics["threshold_selection"]["source_split"] == "val"
    split_rows = {split: [r for r in rows if r["split"] == split] for split in ("val", "test")}
    for split, records in split_rows.items():
        assert len(records) == metrics["splits"][split]["n"]
        assert sum(int(r["label"]) for r in records) == metrics["splits"][split]["positives"]

    test = split_rows["test"]
    matrix = np.zeros((2, 2), dtype=int)
    for row in test:
        matrix[int(row["label"]), int(float(row["y_score"]) >= threshold)] += 1
    expected = metrics["splits"]["test"]
    assert matrix.tolist() == [[expected["tn"], expected["fp"]], [expected["fn"], expected["tp"]]]
    fig, ax = plt.subplots(figsize=(6.8, 5.7), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("True outcome")
    ax.set_title(f"{name} | Test confusion matrix\nValidation-selected threshold = {threshold:.4f}", pad=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=20,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "#172b4d")
    fig.savefig(OUT / f"{model}_confusion_matrix_test.png", dpi=180)
    plt.close(fig)

    val = split_rows["val"]
    # Confirm these are the exact validation scores used by the original training plot.
    original_path = RUN / "training" / model / "fit/predictions_val.csv"
    original = {r["sample_id"]: float(r["y_score"]) for r in csv.DictReader(original_path.open())}
    assert set(original) == {r["sample_id"] for r in val}
    assert all(abs(original[r["sample_id"]] - float(r["y_score"])) < 1e-12 for r in val)
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    histograms = {}
    for label, color in enumerate(["#2a78d6", "#eb6834"]):
        scores = [float(r["y_score"]) for r in val if int(r["label"]) == label]
        ax.hist(scores, bins=bins, color=color, alpha=0.70, edgecolor="white",
                label=f"{LABELS[label]} (n={len(scores)})")
        histograms[str(label)] = np.histogram(scores, bins=bins)[0].tolist()
    ax.set_title(f"{name} | Validation score distribution", pad=12)
    ax.set_xlabel("P(future ICU transfer during this admission)")
    ax.set_ylabel("Admissions")
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(OUT / f"{model}_score_distribution_val.png", dpi=180)
    plt.close(fig)
    manifest["models"][model] = {
        "source_predictions": str(prediction_path.relative_to(ROOT)),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "threshold_from_validation": threshold, "test_confusion_matrix": matrix.tolist(),
        "validation_histogram_bins": bins.tolist(), "validation_histogram_counts": histograms,
        "validation_scores_match_original_training_export": True,
    }

(OUT / "redraw_verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Redrew 6 figures; confusion matrices match recorded metrics; validation scores match original exports.")
