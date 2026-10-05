"""Export the final results from local run directories into results/.

    python scripts/export_results.py

Three models are reported: clinical-only, CXR-only and clinical + CXR. Only
aggregates are written (pooled metrics, intervals, curves and counts), and the
export fails if an identifier column would be written. Test-split numbers are
read from benchmark-evaluation, the only project that scores the test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bn5212_training import plots
from bn5212_training.metrics import cross_fitted_decisions, decision_summary
from compare_crossval import interval_figure, load_runs, paired_rows, summary_rows

IDENTIFIERS = {"sample_id", "subject_id", "hadm_id", "stay_id", "study_id", "dicom_id"}

CLINICAL, CXR, MULTIMODAL = "Clinical-only", "CXR-only", "Clinical + CXR"

# label -> (cross-validation run, benchmark-evaluation result)
RUNS: dict[str, tuple[str, str]] = {
    CLINICAL: ("clinical_only/v3", "pretrained-v3/clinical_only"),
    CXR: ("cxr_only/v2", "nested-v2/cxr_only"),
    # The multimodal model is the project's proposed fusion, cross-attention.
    MULTIMODAL: ("cross_attention/v3", "pretrained-v3/cross_attention"),
}


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    leaked = IDENTIFIERS & {key for row in rows for key in row}
    if leaked:
        raise ValueError(f"Refusing to export identifier columns: {sorted(leaked)}")
    print(plots.write_summary_table(rows, path))


def _test_metrics(benchmark: Path, result: str) -> dict[str, Any]:
    """Test-split numbers as benchmark-evaluation computed them."""
    metrics = json.loads((benchmark / result / "metrics.json").read_text(encoding="utf-8"))
    test, interval = metrics["splits"]["test"], metrics["test_patient_bootstrap"]["auroc"]
    return {
        "test_n": test["n"],
        "test_events": test["positives"],
        "test_auroc": test["auroc"],
        "test_auroc_lower": interval["lower"],
        "test_auroc_upper": interval["upper"],
        "test_auprc": test["auprc"],
        "test_brier": test["brier"],
    }


def _confusion(runs, benchmark: Path) -> list[dict[str, Any]]:
    """Confusion counts per model: cross-validation, then the test split.

    Cross-validation decisions threshold each fold at the Youden point of the
    other folds. Test counts are benchmark-evaluation's, whose threshold is the
    Youden point of the out-of-fold validation predictions.
    """
    rows = []
    for label, (_, predictions, _) in runs.items():
        truth = predictions["label"].to_numpy()
        decisions = cross_fitted_decisions(
            truth, predictions["y_score"].to_numpy(), predictions["fold"].to_numpy()
        )
        rows.append({"model": label, "evaluation": "cross-validation", "n": len(truth),
                     "threshold_from": "other folds", **decision_summary(truth, decisions)})
    for label in runs:
        path = benchmark / RUNS[label][1] / "metrics.json"
        test = json.loads(path.read_text(encoding="utf-8"))["splits"]["test"]
        rows.append({"model": label, "evaluation": "test", "n": test["n"],
                     "threshold_from": "validation (out-of-fold)",
                     **{key: test[key] for key in ("tn", "fp", "fn", "tp", "sensitivity",
                                                   "specificity", "precision", "accuracy")}})
    return rows


def _pretraining(pretrain: Path) -> list[dict[str, Any]]:
    """The external cohort the clinical encoder was pretrained on, in counts."""
    cohort = json.loads((pretrain / "cohort.json").read_text(encoding="utf-8"))
    rows = [{"item": f"cohort flow: {step['stage']}", "value": step["stays"]} for step in cohort["flow"]]
    rows += [
        {"item": "study patients excluded", "value": cohort["study_patients_excluded"]},
        {"item": "external stays", "value": cohort["stays"]},
        {"item": "external patients", "value": cohort["patients"]},
        {"item": "external deaths", "value": cohort["deaths"]},
    ]
    for name, part in cohort["per_split"].items():
        rows.append({"item": f"{name}: stays / deaths", "value": f"{part['stays']} / {part['deaths']}"})
    external = json.loads((pretrain / "variable_projection.json").read_text(encoding="utf-8"))
    for metric in ("auroc", "auprc"):
        rows.append({"item": f"external validation {metric.upper()}",
                     "value": round(external["external_validation"][metric], 4)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", default="outputs/nested_cv")
    parser.add_argument("--benchmark", default="../benchmark-evaluation/outputs")
    parser.add_argument("--pretrain", default="data/pretrain/clinical_v1")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    outputs, benchmark, results = Path(args.outputs), Path(args.benchmark), Path(args.results)
    figures = results / "figures"

    runs = load_runs([f"{label}={outputs / run}" for label, (run, _) in RUNS.items()])
    rows = [{**row, **_test_metrics(benchmark, RUNS[row["model"]][1])} for row in summary_rows(runs)]
    _write(rows, results / "summary")
    print(interval_figure(rows, figures / "auroc.png"))
    print(plots.roc_comparison(
        {label: (run[1]["label"].to_numpy(), run[1]["y_score"].to_numpy()) for label, run in runs.items()},
        figures / "roc.png",
        title=f"Out-of-fold ROC ({rows[0]['events']} events)",
    ))
    _write(paired_rows(runs, [(CLINICAL, CXR), (MULTIMODAL, CLINICAL), (MULTIMODAL, CXR)]),
           results / "paired")

    confusion = _confusion(runs, benchmark)
    _write(confusion, results / "confusion")
    # One row of panels per evaluation, one column per model.
    print(plots.confusion_matrices(
        [(f"{row['model']}\n{row['evaluation']}, n={row['n']}", row) for row in confusion],
        figures / "confusion.png",
        columns=len(runs),
        title="Confusion matrices, normalised by actual class",
    ))

    _write(_pretraining(Path(args.pretrain)), results / "pretraining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
