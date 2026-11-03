"""Lab-only clinical windows before CXR for prospective ICU-transfer prediction."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .extract import ExtractionStats, _aggregate, _extract_source, build_cohort, find_table

LAB_LABELS = {
    "White Blood Cells": ["White Blood Cells", "WBC"],
    "Hemoglobin": ["Hemoglobin"],
    "Hematocrit": ["Hematocrit"],
    "Platelet Count": ["Platelet Count"],
    "Sodium": ["Sodium", "Sodium, Whole Blood"],
    "Potassium": ["Potassium", "Potassium, Whole Blood"],
    "Chloride": ["Chloride", "Chloride, Whole Blood"],
    "Bicarbonate": ["Bicarbonate"],
    "Urea Nitrogen": ["Urea Nitrogen"],
    "Creatinine": ["Creatinine"],
    "Glucose": ["Glucose"],
    "Lactate": ["Lactate"],
    "pH": ["pH"],
    "INR": ["INR", "INR(PT)"],
    "Bilirubin, Total": ["Bilirubin, Total"],
    "Albumin": ["Albumin"],
}
VARIABLES = [*LAB_LABELS, "Age", "Female"]


def resolve_lab_items(root: Path) -> dict:
    dictionary = pd.read_csv(find_table(root, "d_labitems"))
    if not {"itemid", "label", "fluid"} <= set(dictionary):
        raise ValueError("d_labitems must contain itemid, label and fluid")
    blood = dictionary[dictionary["fluid"].str.strip().str.casefold().eq("blood")].copy()
    blood["normalized"] = blood["label"].str.replace(r"\s+", "", regex=True).str.casefold()
    variables = {}
    for name, aliases in LAB_LABELS.items():
        normalized = {"".join(alias.split()).casefold() for alias in aliases}
        selected = blood[blood["normalized"].isin(normalized)]
        if selected.empty:
            raise ValueError(f"No blood lab item resolved for {name}; inspect d_labitems before training")
        variables[name] = {"source": "labevents", "lab_itemids": sorted(selected["itemid"].astype(int).unique().tolist())}
    return {"schema_version": "1.0", "hour_aggregation_default": "mean", "variables": variables}


def extract_transfer_features(run_dir: Path, root: Path, output: Path, *, hours: int = 48,
                              chunksize: int = 1_000_000) -> dict:
    cohort = build_cohort(run_dir, unit="cxr_lookback", icu_hours=hours)
    item_map = resolve_lab_items(root)
    stats = ExtractionStats()
    observations = _extract_source(
        find_table(root, "labevents"), "labevents", cohort, item_map, stats, chunksize,
        match_on_subject=True, strict_availability=True,
    )
    clinical = _aggregate(observations, item_map)
    index = pd.read_csv(run_dir / "index.csv", dtype={"hadm_id": str}).drop_duplicates("hadm_id")
    demographic_rows = []
    for row in index.itertuples(index=False):
        for name, value in (("Age", row.age_at_admission),
                            ("Female", {"F": 1.0, "M": 0.0}.get(str(row.gender).strip().upper()))):
            if pd.notna(value):
                demographic_rows.append({"hadm_id": row.hadm_id, "stay_id": "", "hour": hours - 1,
                                         "variable": name, "value": float(value)})
    columns = ["hadm_id", "stay_id", "hour", "variable", "value"]
    clinical = pd.concat([clinical[columns], pd.DataFrame(demographic_rows, columns=columns)], ignore_index=True)
    clinical = clinical.sort_values(["hadm_id", "hour", "variable"])
    stats.rows_kept = len(clinical)
    for name in VARIABLES:
        rows = clinical[clinical["variable"].eq(name)]
        covered = rows["hadm_id"].nunique()
        stats.per_variable[name] = {
            "observations": len(rows), "units_with_any": int(covered),
            "unit_missing_rate": float(1 - covered / len(cohort)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(output, index=False)
    output.with_suffix(".items.json").write_text(json.dumps(item_map, indent=2) + "\n", encoding="utf-8")
    report = {"task": "icu_transfer", "window": f"[CXR-{hours}h,CXR)", "hours": hours,
              "variable_names": VARIABLES, "units": len(cohort), "lab_rows": len(observations),
              "source_matching": "subject_id", "storetime_required": True, **stats.as_dict()}
    output.with_suffix(".qa.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
