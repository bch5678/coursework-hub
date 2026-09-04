"""Auditable orchestration with atomic publication of a completed run."""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cohort import align_images, select_images
from .config import load_config
from .dataset import INDEX_COLUMNS
from .io import Audit, safe_path, sha256, write_json
from .metadata import load_cxr
from .splits import assign_splits, validate_index
from .tables import load_hospital_tables


def build(config_path):
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    output = Path(cfg["paths"]["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    log_path = staging / "pipeline.log"
    logger = logging.getLogger("bn5212.pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    audit = Audit()
    try:
        logger.info("Starting validated pipeline")
        admissions, source_files = load_hospital_tables(cfg, audit)
        logger.info("Loaded %d valid admission intervals", len(admissions))
        cxr, metadata_files = load_cxr(cfg, audit)
        logger.info("Found %d timestamped local candidate images", len(cxr))
        cxr.to_csv(staging / "cxr_metadata_snapshot.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f", lineterminator="\n")
        image_root = Path(cfg["paths"]["image_root"])
        inventory = []
        for relative in cxr.image_path:
            stat = safe_path(image_root, relative).stat()
            inventory.append({"image_path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        pd.DataFrame(inventory, columns=["image_path", "bytes", "mtime_ns"]).to_csv(staging / "input_image_inventory.csv", index=False, lineterminator="\n")
        source_files += metadata_files
        frame = select_images(align_images(cxr, admissions, cfg, audit), cfg, audit)
        frame, assignments = assign_splits(frame, cfg, audit)
        validate_index(frame)
        frame = frame[INDEX_COLUMNS].sort_values(["split", "subject_id", "admittime", "study_time", "sample_id"])
        frame.to_csv(staging / "index.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f", lineterminator="\n")
        for split in ["train", "val", "test"]:
            frame[frame.split.eq(split)].to_csv(staging / f"index_{split}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f", lineterminator="\n")
        assignments.to_csv(staging / "split_assignments.csv", index=False, lineterminator="\n")
        pd.DataFrame(audit.flow).to_csv(staging / "cohort_flow.csv", index=False, lineterminator="\n")
        summary = {
            "schema_version": "1.0", "warnings": audit.warnings, "cohort_flow": audit.flow,
            "counts": {"images": len(frame), "studies": int(frame.study_id.nunique()), "admissions": int(frame.hadm_id.nunique()), "patients": int(frame.subject_id.nunique())},
            "by_split": frame.groupby("split").agg(images=("sample_id", "size"), patients=("subject_id", "nunique"), admissions=("hadm_id", "nunique"), positives=("label", "sum"), prevalence=("label", "mean")).round(6).astype(object).where(pd.notna, None).to_dict("index"),
        }
        for split, values in summary["by_split"].items():
            if values["positives"] in (0, values["images"]):
                audit.warnings.append(f"{split} has only one outcome class; AUROC is undefined for this split")
        write_json(staging / "qa_report.json", summary)
        shutil.copyfile(config_path, staging / "config.snapshot.json")
        index_hash = sha256(staging / "index.csv")
        spec = {"schema_version": "1.0", "task": cfg["label"]["kind"], "index": "index.csv", "index_sha256": index_hash,
                "image_root": cfg["paths"]["image_root"], "image_path_semantics": "POSIX relative path below image_root",
                "dataset_versions": cfg["dataset_versions"], "split_unit": "subject_id", "split_seed": cfg["split"]["seed"],
                "sample_contract": {"image": "float32 [C,H,W], normalized", "label": "int64 scalar (0/1)", "sample_weight": "float32 scalar; sums to 1 per admission", "identifiers": ["sample_id", "subject_id", "hadm_id", "study_id", "dicom_id"]},
                "loader": cfg["loader"]}
        write_json(staging / "dataset_spec.json", spec)
        source_manifest = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in source_files]
        if cfg["audit"]["hash_images"]:
            root = Path(cfg["paths"]["image_root"])
            source_manifest.extend({"path": str(root / p), "sha256": sha256(root / p), "bytes": (root / p).stat().st_size} for p in frame.image_path)
        write_json(staging / "run_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "platform": platform.platform(), "config_sha256": sha256(config_path), "inputs": source_manifest,
                   "packages": {name: version(name) for name in ["numpy", "pandas", "Pillow", "pydicom", "torch"]},
                   "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
                                   for p in Path(__file__).resolve().parent.glob("*.py")}})
        logger.info("Validated %d images across %d patients; preparing atomic publication", len(frame), frame.subject_id.nunique())
        artifacts = {p.name: sha256(p) for p in staging.iterdir() if p.is_file() and p.name != "SUCCESS.json"}
        write_json(staging / "SUCCESS.json", {"status": "complete", "artifacts_sha256": artifacts})
        handler.close(); logger.removeHandler(handler)
        os.replace(staging, output)
        return output, summary
    except Exception:
        write_json(staging / "FAILED.json", {"status": "failed", "cohort_flow": audit.flow, "warnings": audit.warnings})
        logger.exception("Pipeline failed; incomplete staging directory retained for diagnosis")
        handler.close(); logger.removeHandler(handler)
        raise
