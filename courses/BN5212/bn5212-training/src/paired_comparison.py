"""Paired AUROC gains on the same held-out patients; no model selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def paired_auroc(model: pd.DataFrame, baseline: pd.DataFrame, *, samples=2000, seed=5212):
    if samples < 1:
        raise ValueError("samples must be positive")
    columns = ["sample_id", "subject_id", "hadm_id", "label", "y_score"]
    frames = []
    for frame in (model, baseline):
        frame = frame[columns].copy().sort_values("sample_id").reset_index(drop=True)
        if frame.empty or frame["sample_id"].duplicated().any() or frame.isna().any().any():
            raise ValueError("Predictions must be nonempty, unique and complete")
        if not frame["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary")
        if not np.isfinite(frame["y_score"]).all() or not frame["y_score"].between(0, 1).all():
            raise ValueError("Scores must be finite probabilities")
        frames.append(frame)
    model, baseline = frames
    if not model[columns[:-1]].equals(baseline[columns[:-1]]):
        raise ValueError("Paired comparison requires identical samples, patients, admissions and labels")
    labels = model["label"].to_numpy()
    first = model["y_score"].to_numpy()
    second = baseline["y_score"].to_numpy()
    groups = list(model.groupby("subject_id", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(labels[indices])) == 2:
            draws.append(roc_auc_score(labels[indices], first[indices]) -
                         roc_auc_score(labels[indices], second[indices]))
    estimate = (float(roc_auc_score(labels, first) - roc_auc_score(labels, second))
                if len(np.unique(labels)) == 2 else None)
    return dict(delta_auroc=estimate,
                lower=float(np.quantile(draws, .025)) if draws else None,
                upper=float(np.quantile(draws, .975)) if draws else None,
                successful_replicates=len(draws), bootstrap_samples=samples,
                seed=seed, confidence=.95, n=len(labels), patients=len(groups))


def compare_batch(batch: Path, *, samples=2000, seed=5212):
    from bn5212_benchmark.aggregation import aggregate_predictions
    from bn5212_benchmark.compare import _load_result

    names = ("clinical_only", "cxr_only", "concat", "joint", "cross_attention_adapter")
    predictions = {}
    signatures = set()
    for name in names:
        directory = batch / "benchmark" / name
        result = _load_result(directory)  # verify completed results and artifact hashes
        signatures.add((result["dataset_index_sha256"], result["evaluation_unit"]))
        raw = pd.read_csv(directory / "predictions.csv", dtype={
            "sample_id": str, "subject_id": str, "hadm_id": str})
        predictions[name] = aggregate_predictions(raw[raw["split"].eq("test")],
                                                  result["evaluation_unit"])
    if len(signatures) != 1:
        raise ValueError("Paired comparison requires a shared dataset index and evaluation unit")
    rows = []
    for model in names[2:]:
        for baseline in names[:2]:
            rows.append(dict(model=model, baseline=baseline, split="test",
                             evaluation_unit=next(iter(signatures))[1],
                             dataset_index_sha256=next(iter(signatures))[0],
                             **paired_auroc(predictions[model], predictions[baseline],
                                            samples=samples, seed=seed)))
    output = batch / "paired_auroc_gains.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5212)
    args = parser.parse_args()
    print(compare_batch(args.batch, samples=args.samples, seed=args.seed))
