"""Command-line entry points.

    bn5212-train      --config configs/cxr_only.json --run-dir <frozen run>
    bn5212-predict    --checkpoint <ckpt> --split test --output test.csv
    bn5212-summarize  outputs/**/summary.csv --output-csv comparison.csv
    bn5212-crossval   --config configs/clinical_only.json --folds 5
    bn5212-build-image-cache --run-dir <run> --output cache/images.json
    bn5212-extract-clinical --run-dir <run> --mimic-root <mimic> --output clinical.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .plots import write_summary_table
from .predict import predict_from_checkpoint


def _parse_override(item: str) -> tuple[str, Any]:
    key, separator, raw = item.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"--set expects section.key=value, got {item!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # bare strings such as --set fusion.name=cross_attention
    return key.strip(), value


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train one BN5212 experiment")
    parser.add_argument("--config", required=True, help="Experiment config JSON")
    parser.add_argument("--run-dir", help="Override data.run_dir (the frozen dataset run)")
    parser.add_argument("--data-pipeline-path", help="Override the bn5212-data-pipeline location")
    parser.add_argument("--output-dir", help="Override the output root")
    parser.add_argument("--run-id", help="Name this run instead of using a UTC timestamp")
    parser.add_argument("--seed", type=int, help="Override the training seed")
    parser.add_argument("--device", help="cpu, cuda or auto")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override any config field, e.g. --set optim.epochs=3",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = dict(_parse_override(item) for item in args.set)
    if args.run_dir:
        overrides["data.run_dir"] = args.run_dir
    if args.data_pipeline_path:
        overrides["data.data_pipeline_path"] = args.data_pipeline_path
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device

    cfg = load_config(args.config, overrides)
    # Imported here so that --help stays fast and does not need torch.
    from .trainer import train

    result = train(cfg, output_dir=args.output_dir, run_id=args.run_id)
    print(json.dumps(result["summary"], indent=2, default=str))
    print(f"\nRun directory: {result['run_dir']}")
    print("Hand the prediction files to benchmark-evaluation:")
    for split, path in result["predictions"].items():
        print(f"  {split}: {path}")


def predict_main() -> None:
    parser = argparse.ArgumentParser(
        description="Write sample_id,y_score predictions from a trained checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", help="Score a different frozen dataset run")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    path = predict_from_checkpoint(
        args.checkpoint,
        split=args.split,
        output_path=args.output,
        run_dir=args.run_dir,
        device=args.device,
    )
    print(path)


def build_image_cache_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode every radiograph of a frozen run once into a reusable cache. "
            "Without it each epoch re-decodes full-resolution DICOMs, which "
            "dominates multi-fold image runs."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True, help="Metadata JSON; the array goes beside it")
    parser.add_argument("--data-pipeline-path")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    from .cache import build_image_cache

    report = build_image_cache(
        args.run_dir,
        args.output,
        data_pipeline_path=args.data_pipeline_path,
        splits=args.splits,
    )
    print(json.dumps(report, indent=2))
    print()
    print(f"Use it with:  --set data.image_cache={report['metadata']}")


def crossval_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patient-grouped cross-validation over the frozen train+val splits. "
            "Use this when one split leaves too few events to compare models; "
            "the frozen test split is not touched."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", help="Override data.run_dir")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help="Inner folds per outer fold; early stopping runs on these, never on the held-out fold",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--keep-fold-checkpoints",
        action="store_true",
        help="Retain each fold's checkpoints; they repeat a frozen backbone and are large",
    )
    parser.add_argument(
        "--no-final-model",
        action="store_true",
        help="Skip the deliverable model fitted on all train+val patients",
    )
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    overrides: dict[str, Any] = dict(_parse_override(item) for item in args.set)
    if args.run_dir:
        overrides["data.run_dir"] = args.run_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device:
        overrides["device"] = args.device

    cfg = load_config(args.config, overrides)
    from .crossval import run_cross_validation

    result = run_cross_validation(
        cfg,
        n_splits=args.folds,
        inner_splits=args.inner_folds,
        output_dir=args.output_dir,
        run_id=args.run_id,
        bootstrap_samples=args.bootstrap_samples,
        keep_fold_checkpoints=args.keep_fold_checkpoints,
        train_final_model=not args.no_final_model,
    )
    pooled = result["pooled"]
    interval = pooled["auroc_ci"]
    print()
    print(f"out-of-fold n={pooled['n']}  positives={pooled['n_positive']}")
    print(f"AUROC {pooled['auroc']:.3f}  95% CI [{interval['lower']:.3f}, {interval['upper']:.3f}]")
    print(f"AUPRC {pooled['auprc']:.3f}   Brier {pooled['brier']:.3f}")
    print()
    print(f"Run directory: {result['run_dir']}")
    if result.get("final_model"):
        print(f"Deliverable checkpoint: {result['final_model']['checkpoint']}")
        print("Hand these to benchmark-evaluation:")
        print(f"  val:  {Path(result['run_dir']) / 'predictions_val.csv'}  (out-of-fold)")
        print(f"  test: {result['final_model']['test_predictions']}")
        print(f"  fitted on {result['final_model']['patients']} patients "
              f"for {result['final_model']['epochs']} epochs")


def extract_clinical_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract clinical time series from MIMIC-IV into the long-format table "
            "described in docs/CLINICAL_FEATURE_SPEC.md"
        )
    )
    # --run-dir is not needed to inspect itemids, and inspecting them is exactly
    # what you do before the first pipeline run exists.
    parser.add_argument("--run-dir", help="Frozen dataset run (supplies the cohort and cutoffs)")
    parser.add_argument("--mimic-root", required=True, help="MIMIC-IV root containing hosp/ and icu/")
    parser.add_argument("--output", help="Destination CSV (required unless --inspect-items)")
    parser.add_argument(
        "--item-map",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "clinical_items.json"),
    )
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument(
        "--cohort-unit",
        choices=["admission", "icu_stay"],
        default="admission",
        help="icu_stay counts hours from the ICU intime over a fixed window, as MeTra does",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--no-labs", action="store_true", help="Skip labevents")
    parser.add_argument(
        "--inspect-items",
        action="store_true",
        help="Print what each configured itemid resolves to in this dataset and exit",
    )
    args = parser.parse_args()

    from .extract import extract_clinical_features, inspect_items, load_item_map

    item_map = load_item_map(args.item_map)
    if args.inspect_items:
        report = inspect_items(args.mimic_root, item_map)
        print(report.to_string(index=False))
        missing = report.loc[~report["found"], "itemid"].tolist()
        if missing:
            print(f"\n{len(missing)} configured itemid(s) not present in this dataset: {missing}")
            print("Edit the item map before extracting, or accept them as unavailable.")
        return

    if not args.output:
        parser.error("--output is required unless --inspect-items is given")
    if not args.run_dir:
        parser.error("--run-dir is required to extract (it supplies the cohort and cutoffs)")

    report = extract_clinical_features(
        run_dir=args.run_dir,
        mimic_root=args.mimic_root,
        item_map_path=args.item_map,
        output_path=args.output,
        max_hours=args.max_hours,
        chunksize=args.chunksize,
        include_labs=not args.no_labs,
        unit=args.cohort_unit,
    )
    print(json.dumps({key: report[key] for key in (
        "output", "qa_report", "cohort_unit", "units_in_cohort", "rows_kept",
        "source_coverage", "variables_with_no_data", "dropped",
    )}, indent=2, ensure_ascii=False))


def summarize_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect validation summaries from several training runs. "
            "Final test results come from benchmark-evaluation, not from this table."
        )
    )
    parser.add_argument("run_dirs", nargs="+", help="Training run directories")
    parser.add_argument("--output", default="outputs/training_comparison")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for directory in args.run_dirs:
        summary = Path(directory) / "summary.csv"
        if not summary.is_file():
            print(f"skipping {directory}: no summary.csv")
            continue
        import pandas as pd

        rows.extend(pd.read_csv(summary).to_dict(orient="records"))
    if not rows:
        raise SystemExit("No run summaries found")
    path = write_summary_table(rows, args.output)
    print(path)
