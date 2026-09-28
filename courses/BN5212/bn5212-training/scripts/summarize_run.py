"""Print the cohort a pipeline run produced.

Run this right after the pipeline finishes. The event count it reports is what
decides how strong a conclusion the experiments can support: AUROC computed on a
handful of positives carries a confidence interval far wider than the effect
sizes this project is trying to separate.

    python scripts/summarize_run.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main(run_dir: str) -> int:
    root = Path(run_dir)
    index_path = root / "index.csv"
    if not index_path.is_file():
        print(f"no index.csv under {root}")
        return 1

    frame = pd.read_csv(index_path, dtype={"subject_id": str, "hadm_id": str})
    print(f"run_dir: {root.resolve()}")
    print(f"rows (images): {len(frame)}")
    print(f"patients:      {frame['subject_id'].nunique()}")
    print(f"admissions:    {frame['hadm_id'].nunique()}")
    print(f"deaths:        {int(frame['label'].sum())}")
    print(f"prevalence:    {frame['label'].mean():.4f}")

    print("\nper split")
    summary = frame.groupby("split").agg(
        rows=("label", "size"),
        patients=("subject_id", "nunique"),
        deaths=("label", "sum"),
        prevalence=("label", "mean"),
    )
    print(summary.to_string())

    smallest = int(summary["deaths"].min())
    if smallest < 25:
        print(
            f"\nWARNING: the smallest split holds {smallest} deaths. "
            "AUROC on that few positives has a very wide confidence interval; "
            "report CIs and consider cross-validation or an outcome with more events."
        )

    print("\nviews:", frame["view"].value_counts().to_dict())
    print(
        "hours_since_admission: min %.1f  median %.1f  max %.1f"
        % (
            frame["hours_since_admission"].min(),
            frame["hours_since_admission"].median(),
            frame["hours_since_admission"].max(),
        )
    )

    qa_path = root / "qa_report.json"
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        warnings = qa.get("warnings")
        if warnings:
            print("\npipeline warnings:")
            for item in warnings[:10]:
                print(" -", item)

    flow_path = root / "cohort_flow.csv"
    if flow_path.is_file():
        print("\ncohort flow (where rows were lost)")
        flow = pd.read_csv(flow_path)
        print(flow.to_string(index=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
