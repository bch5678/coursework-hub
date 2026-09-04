"""Command-line entry points."""
from __future__ import annotations

import argparse

from .baselines import run_prevalence_baseline
from .compare import build_leaderboard
from .evaluation import evaluate_prediction_files


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation and test probability files")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--evaluation-unit", choices=["sample", "admission"], default="admission")
    parser.add_argument("--threshold-method", choices=["fixed_0.5", "youden_j", "f1"], default="youden_j")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=5212)
    args = parser.parse_args()
    output = evaluate_prediction_files(
        run_dir=args.run_dir,
        val_predictions=args.val_predictions,
        test_predictions=args.test_predictions,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_version=args.model_version,
        evaluation_unit=args.evaluation_unit,
        threshold_method=args.threshold_method,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        checkpoint=args.checkpoint,
    )
    print(output)


def prevalence_main() -> None:
    parser = argparse.ArgumentParser(description="Run the train-prevalence probability baseline")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=5212)
    args = parser.parse_args()
    output = run_prevalence_baseline(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(output)


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed BN5212 evaluation runs")
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()
    csv_path, html_path = build_leaderboard(
        args.result_dirs,
        output_csv=args.output_csv,
        output_html=args.output_html,
    )
    print(csv_path)
    print(html_path)
