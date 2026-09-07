"""Convert sample predictions to the pre-declared benchmark evaluation unit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_predictions(predictions: pd.DataFrame, unit: str = "admission") -> pd.DataFrame:
    if unit not in {"sample", "admission"}:
        raise ValueError("evaluation unit must be sample or admission")
    if unit == "sample":
        result = predictions.copy()
        result["source_samples"] = 1
        return result

    required = {"hadm_id", "subject_id", "label", "sample_weight", "split", "y_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Admission aggregation is missing columns: {sorted(missing)}")
    if predictions.groupby("hadm_id")["label"].nunique().gt(1).any():
        raise ValueError("An admission has inconsistent labels")
    if predictions.groupby("hadm_id")["subject_id"].nunique().gt(1).any():
        raise ValueError("An admission is linked to multiple patients")
    if predictions.groupby("hadm_id")["split"].nunique().gt(1).any():
        raise ValueError("An admission crosses dataset splits")

    rows = []
    for hadm_id, group in predictions.groupby("hadm_id", sort=True):
        weight = group["sample_weight"].to_numpy(dtype=float)
        if not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("Admission aggregation requires finite positive sample weights")
        rows.append(
            {
                "sample_id": f"admission:{hadm_id}",
                "subject_id": group["subject_id"].iloc[0],
                "hadm_id": hadm_id,
                "label": int(group["label"].iloc[0]),
                "sample_weight": 1.0,
                "split": group["split"].iloc[0],
                "y_score": float(np.average(group["y_score"].to_numpy(dtype=float), weights=weight)),
                "source_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows)
