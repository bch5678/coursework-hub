"""Export the reported results from local run directories into results/.

    python scripts/export_results.py

Only aggregates are written: pooled metrics, intervals, curves and counts. The
export fails if an identifier column would be written. Test-split numbers are
read from benchmark-evaluation, the only project that scores the test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


from bn5212_training import plots
from compare_crossval import interval_figure, load_runs, paired_rows, summary_rows

IDENTIFIERS = {"sample_id", "subject_id", "hadm_id", "stay_id", "study_id", "dicom_id"}

CLINICAL = "Clinical-only (pretrained encoder)"
CLINICAL_COHORT = "Clinical-only (cohort only)"
CLINICAL_EXTERNAL = "External clinical model (no cohort fitting)"
CXR = "CXR-only"
CONCAT, METRA, CROSS = "Concat fusion", "MeTra joint self-attention", "Cross-attention"

# label -> (cross-validation run, benchmark-evaluation result)
RUNS: dict[str, tuple[str, str]] = {
    CLINICAL: ("clinical_only/v3", "pretrained-v3/clinical_only"),
    CLINICAL_COHORT: ("clinical_only/v2", "nested-v2/clinical_only"),
    CLINICAL_EXTERNAL: ("clinical_external/v3", "pretrained-v3/clinical_external"),
    CXR: ("cxr_only/v2", "nested-v2/cxr_only"),
    CONCAT: ("concat_fusion/v3", "pretrained-v3/concat_fusion"),
    METRA: ("metra_joint/v3", "pretrained-v3/metra_joint"),
    CROSS: ("cross_attention/v3", "pretrained-v3/cross_attention"),
}
# The same five models with the clinical encoder trained on the cohort alone.
COHORT_ONLY = {
    "Clinical-only": "clinical_only/v2",
    CXR: "cxr_only/v2",
    CONCAT: "concat_fusion/v2",
    METRA: "metra_joint/v2",
    CROSS: "cross_attention/v2",
}
ABLATIONS = {
    "image_encoder": {
        "ImageNet ViT-B/16, frozen (kept)": "cxr_only/v2",
        "ViT-Tiny, last 2 blocks fine-tuned": "cxr_tiny_unfreeze2/v2",
        "DenseNet121 CheXpert, frozen": "cxr_xrv/v2",
        "ViT-Tiny, frozen": "cxr_tiny_frozen/v2",
    },
    "clinical_encoder_cohort_only": {
        "linear_projection": "clinical_only/v2",
        "variable_projection": "clinical_only/v2-varproj",
        "summary_stats": "clinical_only/v2-summary",
    },
}


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    leaked = IDENTIFIERS & {key for row in rows for key in row}
    if leaked:
        raise ValueError(f"Refusing to export identifier columns: {sorted(leaked)}")
    print(plots.write_summary_table(rows, path))


def _test_metrics(benchmark: Path, result: str) -> dict[str, Any]:
    """Test-split numbers as benchmark-evaluation computed them."""
    path = benchmark / result / "metrics.json"
    if not path.is_file():
        return {}
    metrics = json.loads(path.read_text(encoding="utf-8"))
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


def _group(outputs: Path, labels: Mapping[str, str] | list[str]):
    names = labels if isinstance(labels, Mapping) else {label: RUNS[label][0] for label in labels}
    return load_runs([f"{label}={outputs / run}" for label, run in names.items()])


def _pretraining(pretrain: Path) -> list[dict[str, Any]]:
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
    for encoder in ("variable_projection", "linear_projection"):
        report = pretrain / f"{encoder}.json"
        if report.is_file():
            external = json.loads(report.read_text(encoding="utf-8"))["external_validation"]
            rows.append({"item": f"{encoder}: external validation AUROC", "value": round(external["auroc"], 4)})
            rows.append({"item": f"{encoder}: external validation AUPRC", "value": round(external["auprc"], 4)})
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

    # Unimodal baselines: the scope of this project.
    unimodal = _group(outputs, [CLINICAL, CLINICAL_COHORT, CLINICAL_EXTERNAL, CXR])
    rows = [{**row, **_test_metrics(benchmark, RUNS[row["model"]][1])} for row in summary_rows(unimodal)]
    _write(rows, results / "unimodal")
    print(interval_figure(rows, figures / "unimodal_auroc.png"))
    print(plots.roc_comparison(
        {label: (unimodal[label][1]["label"].to_numpy(), unimodal[label][1]["y_score"].to_numpy())
         for label in (CLINICAL, CLINICAL_COHORT, CXR)},
        figures / "unimodal_roc.png",
        title=f"Out-of-fold ROC ({rows[0]['events']} events)",
    ))
    _write(
        paired_rows(unimodal, [(CLINICAL, CLINICAL_COHORT), (CLINICAL, CXR), (CLINICAL_COHORT, CXR)]),
        results / "unimodal_paired",
    )

    # Every model the framework trains, for context.
    everything = _group(outputs, [CLINICAL, CXR, CONCAT, METRA, CROSS])
    rows = [{**row, **_test_metrics(benchmark, RUNS[row["model"]][1])} for row in summary_rows(everything)]
    _write(rows, results / "all_models")
    print(interval_figure(rows, figures / "all_models_auroc.png"))
    _write(
        paired_rows(everything, [(CXR, CLINICAL), (CONCAT, CLINICAL), (METRA, CLINICAL),
                                 (CROSS, CLINICAL), (CROSS, METRA), (CROSS, CONCAT)]),
        results / "all_models_paired",
    )

    _write(summary_rows(_group(outputs, COHORT_ONLY)), results / "all_models_cohort_only")
    for name, runs in ABLATIONS.items():
        _write(summary_rows(_group(outputs, runs)), results / f"ablation_{name}")

    if (Path(args.pretrain) / "cohort.json").is_file():
        _write(_pretraining(Path(args.pretrain)), results / "pretraining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
