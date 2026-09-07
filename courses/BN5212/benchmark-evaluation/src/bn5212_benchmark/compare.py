"""Create a CSV and offline HTML leaderboard from completed evaluation runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_json, sha256
from .report import write_leaderboard_report


def _load_result(result_dir: str | Path) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    success = read_json(result_dir / "SUCCESS.json")
    if success.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation result: {result_dir}")
    for name, expected in success.get("artifacts_sha256", {}).items():
        if sha256(result_dir / name) != expected:
            raise ValueError(f"Changed evaluation artifact: {result_dir / name}")
    metrics = read_json(result_dir / "metrics.json")
    test = metrics["splits"]["test"]
    return {
        "model": metrics["model"]["name"],
        "version": metrics["model"]["version"],
        "n": test["n"],
        "evaluation_unit": metrics["evaluation_unit"],
        **{name: test.get(name) for name in ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]},
        "result_dir": str(result_dir),
        "dataset_index_sha256": metrics["dataset"]["index_sha256"],
    }


def build_leaderboard(
    result_dirs: list[str | Path],
    *,
    output_csv: str | Path,
    output_html: str | Path,
) -> tuple[Path, Path]:
    if not result_dirs:
        raise ValueError("At least one evaluation result directory is required")
    rows = [_load_result(path) for path in result_dirs]
    dataset_hashes = {row["dataset_index_sha256"] for row in rows}
    if len(dataset_hashes) != 1:
        raise ValueError("Cannot compare models evaluated on different dataset indexes")
    evaluation_units = {row["evaluation_unit"] for row in rows}
    if len(evaluation_units) != 1:
        raise ValueError("Cannot compare models evaluated using different evaluation units")
    rows.sort(
        key=lambda row: (
            float("inf") if row["auroc"] is None else -row["auroc"],
            row["model"],
            row["version"],
        )
    )
    output_csv, output_html = Path(output_csv), Path(output_html)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, lineterminator="\n")
    write_leaderboard_report(output_html, rows)
    return output_csv, output_html
