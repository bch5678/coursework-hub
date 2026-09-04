"""Non-learned baselines that validate the full evaluation pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .evaluation import evaluate_prediction_files
from .io import load_dataset_index


def run_prevalence_baseline(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 5212,
) -> Path:
    index, _ = load_dataset_index(run_dir)
    train = index[index["split"].eq("train")]
    prevalence = float(train["label"].mean())
    with tempfile.TemporaryDirectory(prefix="bn5212-prevalence-") as temporary:
        temporary = Path(temporary)
        files = {}
        for split in ["val", "test"]:
            predictions = index.loc[index["split"].eq(split), ["sample_id"]].copy()
            predictions["y_score"] = prevalence
            files[split] = temporary / f"{split}_predictions.csv"
            predictions.to_csv(files[split], index=False, lineterminator="\n")
        return evaluate_prediction_files(
            run_dir=run_dir,
            val_predictions=files["val"],
            test_predictions=files["test"],
            output_dir=output_dir,
            model_name="prevalence_baseline",
            model_version="1.0",
            threshold_method="fixed_0.5",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
