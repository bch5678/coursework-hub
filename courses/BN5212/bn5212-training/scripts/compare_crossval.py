"""Put several cross-validation runs side by side.

    python scripts/compare_crossval.py "Clinical-only=<run>" "CXR-only=<run>" --reference "Clinical-only"

Writes a table, the paired differences against the reference, and a figure of
the out-of-fold AUROC intervals. Runs that did not score the same samples in
the same folds are rejected. The test split is scored by benchmark-evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from bn5212_training import plots
from bn5212_training.metrics import paired_bootstrap_difference

Run = tuple[Path, pd.DataFrame, dict]


def load_runs(specs: Sequence[str]) -> dict[str, Run]:
    """Read LABEL=RUN_DIR specs and refuse runs that are not comparable."""
    runs: dict[str, Run] = {}
    for spec in specs:
        label, separator, location = spec.partition("=")
        if not separator:
            location, label = spec, Path(spec).parent.name
        root = Path(location)
        predictions = pd.read_csv(
            root / "out_of_fold_predictions.csv", dtype={"sample_id": str, "subject_id": str}
        ).sort_values("sample_id", ignore_index=True)
        metrics = json.loads((root / "cv_metrics.json").read_text(encoding="utf-8"))
        runs[label] = (root, predictions, metrics)

    first_label = next(iter(runs))
    anchor = runs[first_label][1]
    for label, (_, predictions, _) in runs.items():
        same = (
            len(predictions) == len(anchor)
            and predictions["sample_id"].equals(anchor["sample_id"])
            and predictions["label"].equals(anchor["label"])
            and predictions["fold"].equals(anchor["fold"])
        )
        if not same:
            raise SystemExit(
                f"{label} did not score the same samples in the same folds as "
                f"{first_label}; the runs are not comparable"
            )
    return runs


def summary_rows(runs: Mapping[str, Run]) -> list[dict[str, Any]]:
    """One row of pooled and per-fold metrics per run. Aggregates only."""
    rows = []
    for label, (_, _, metrics) in runs.items():
        pooled, spread = metrics["pooled"], metrics["per_fold_spread"]
        folds = [fold["auroc"] for fold in metrics["per_fold"]]
        rows.append({
            "model": label,
            "n": pooled["n"],
            "events": pooled["n_positive"],
            "oof_auroc": pooled["auroc"],
            "auroc_lower": pooled["auroc_ci"]["lower"],
            "auroc_upper": pooled["auroc_ci"]["upper"],
            "fold_auroc_mean": spread["auroc"]["mean"],
            "fold_auroc_std": spread["auroc"]["std"],
            "fold_auroc_min": min(folds),
            "fold_auroc_max": max(folds),
            "oof_auprc": pooled["auprc"],
            "auprc_lower": pooled["auprc_ci"]["lower"],
            "auprc_upper": pooled["auprc_ci"]["upper"],
            "oof_brier": pooled["brier"],
        })
    return rows


def interval_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    return plots.interval_comparison(
        [
            {"label": row["model"], "estimate": row["oof_auroc"],
             "lower": row["auroc_lower"], "upper": row["auroc_upper"]}
            for row in rows
        ],
        path,
        title=f"Out-of-fold AUROC, 95% patient bootstrap interval ({rows[0]['events']} events)",
        xlabel="AUROC",
    )


def paired_rows(
    runs: Mapping[str, Run],
    pairs: Sequence[tuple[str, str]],
    *,
    samples: int = 2000,
    seed: int = 5212,
) -> list[dict[str, Any]]:
    """AUROC of the first model minus the second, bootstrapped over shared patients."""
    rows = []
    for first, second in pairs:
        if first not in runs or second not in runs:
            raise SystemExit(f"Unknown label in pair {first!r} - {second!r}; have {list(runs)}")
        a, b = runs[first][1], runs[second][1]
        result = paired_bootstrap_difference(
            a["label"].to_numpy(), a["y_score"].to_numpy(), b["y_score"].to_numpy(),
            a["subject_id"].to_numpy(), samples=samples, seed=seed,
        )
        rows.append({
            "comparison": f"{first} - {second}",
            "auroc_difference": result["estimate"],
            "lower": result["lower"],
            "upper": result["upper"],
            "interval_excludes_zero": bool(result["lower"] > 0 or result["upper"] < 0),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", metavar="LABEL=RUN_DIR")
    parser.add_argument("--reference", help="Label of the run the others are differenced against")
    parser.add_argument("--also-pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="An extra A-minus-B difference, e.g. the proposed fusion against MeTra")
    parser.add_argument("--output", default="outputs/crossval_comparison")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5212)
    args = parser.parse_args()

    runs = load_runs(args.runs)
    rows = summary_rows(runs)
    output = Path(args.output)
    print(plots.write_summary_table(rows, output))
    print(interval_figure(rows, output.with_name(output.name + "_auroc.png")))

    pairs = [(label, args.reference) for label in runs if args.reference and label != args.reference]
    pairs += [tuple(pair) for pair in args.also_pair]
    if pairs:
        differences = paired_rows(runs, pairs, samples=args.bootstrap_samples, seed=args.seed)
        print(plots.write_summary_table(differences, output.with_name(output.name + "_paired")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
