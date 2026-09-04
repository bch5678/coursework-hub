"""Strict configuration; relative paths are relative to the JSON file."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    validate_config(cfg)
    for key, value in cfg["paths"].items():
        if value is None:
            continue
        expanded = os.path.expandvars(value)
        if "${" in expanded or (expanded.startswith("%") and expanded.endswith("%")):
            raise ValueError(f"Unresolved environment variable in paths.{key}")
        p = Path(expanded).expanduser()
        cfg["paths"][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return cfg


def validate_config(cfg: dict) -> None:
    expected = {"schema_version", "dataset_versions", "paths", "cohort", "alignment", "label", "split", "cleaning", "loader", "audit"}
    if set(cfg) != expected or cfg["schema_version"] != "1.0":
        raise ValueError("Configuration must match schema 1.0; check missing or unknown top-level keys")
    keys = {
        "dataset_versions": {"mimic_iv", "mimic_cxr"},
        "paths": {"mimic_iv_root", "image_root", "cxr_metadata", "output_dir"},
        "cohort": {"min_age", "views", "selection"},
        "alignment": {"max_hours_after_admission", "minimum_hours_before_end"},
        "label": {"kind", "column", "positive_values", "negative_values"},
        "split": {"seed", "train", "val", "test", "stratify"},
        "cleaning": {"verify_images", "invalid_image_policy", "max_invalid_image_fraction"},
        "loader": {"image_size", "channels", "batch_size", "num_workers", "mean", "std"},
        "audit": {"hash_images"},
    }
    for section, required in keys.items():
        if set(cfg[section]) != required:
            raise ValueError(f"Missing/unknown keys in {section}; expected {sorted(required)}")
    if cfg["dataset_versions"] != {"mimic_iv": "3.1", "mimic_cxr": "2.1.0"}:
        raise ValueError("This project targets MIMIC-IV 3.1 and MIMIC-CXR 2.1.0")
    for key, value in cfg["paths"].items():
        if key == "cxr_metadata" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a nonempty path")
    split = cfg["split"]
    ratios = [split[k] for k in ("train", "val", "test")]
    if not all(isinstance(r, (float, int)) and math.isfinite(r) and 0 < r < 1 for r in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-8):
        raise ValueError("Split ratios must be positive and sum to 1")
    if not isinstance(split["seed"], int) or split["seed"] < 0:
        raise ValueError("split.seed must be a nonnegative integer")
    if cfg["cohort"]["selection"] not in {"first_per_admission", "first_per_study", "all_images"}:
        raise ValueError("Unknown cohort.selection")
    if not isinstance(cfg["cohort"]["views"], list) or not cfg["cohort"]["views"]:
        raise ValueError("cohort.views must be a nonempty list")
    if cfg["cohort"]["min_age"] < 0:
        raise ValueError("min_age must be nonnegative")
    a = cfg["alignment"]
    if a["max_hours_after_admission"] is not None and (not math.isfinite(a["max_hours_after_admission"]) or a["max_hours_after_admission"] <= 0):
        raise ValueError("max_hours_after_admission must be positive or null")
    if not math.isfinite(a["minimum_hours_before_end"]) or a["minimum_hours_before_end"] < 0:
        raise ValueError("minimum_hours_before_end must be nonnegative")
    label = cfg["label"]
    if label["kind"] not in {"in_hospital_mortality", "admission_binary"}:
        raise ValueError("Unknown label.kind")
    if label["kind"] == "in_hospital_mortality" and (label["column"] != "hospital_expire_flag" or label["positive_values"] != ["1"] or label["negative_values"] != ["0"]):
        raise ValueError("Mortality requires hospital_expire_flag and the standard 1/0 mapping")
    pos, neg = map(lambda x: set(map(str, x)), (label["positive_values"], label["negative_values"]))
    if not pos or not neg or pos & neg:
        raise ValueError("Binary label mappings must be nonempty and disjoint")
    cleaning = cfg["cleaning"]
    if cleaning["invalid_image_policy"] not in {"drop", "error"} or not 0 <= cleaning["max_invalid_image_fraction"] <= 1:
        raise ValueError("Invalid image cleaning policy")
    for section, key in [("split", "stratify"), ("cleaning", "verify_images"), ("audit", "hash_images")]:
        if not isinstance(cfg[section][key], bool):
            raise ValueError(f"{section}.{key} must be boolean")
    loader = cfg["loader"]
    for key, minimum in [("image_size", 8), ("batch_size", 1), ("num_workers", 0)]:
        if not isinstance(loader[key], int) or loader[key] < minimum:
            raise ValueError(f"Invalid loader.{key}")
    if loader["channels"] not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if len(loader["mean"]) != loader["channels"] or len(loader["std"]) != loader["channels"]:
        raise ValueError("mean/std lengths must equal channels")
    if any(not math.isfinite(x) for x in loader["mean"] + loader["std"]) or any(x <= 0 for x in loader["std"]):
        raise ValueError("mean/std must be finite and std positive")
