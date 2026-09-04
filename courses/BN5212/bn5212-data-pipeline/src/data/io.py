"""CSV utilities, identifiers, safe paths and audit checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def read_csv(path, required=()) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing required columns {sorted(missing)}")
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame


def table_path(root: Path, name: str) -> Path:
    candidates = [root / "hosp" / f"{name}.csv.gz", root / "hosp" / f"{name}.csv", root / f"{name}.csv.gz", root / f"{name}.csv"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {name}.csv[.gz] in root/hosp or root; found {len(found)}")
    return found[0]


def safe_path(root: Path, relative: str) -> Path:
    # Treat both separators consistently on Windows and Linux.
    relative = str(relative).replace("\\", "/")
    if not relative or Path(relative).is_absolute() or ":" in relative or ".." in relative.split("/"):
        raise ValueError("Image/archive path must be relative and cannot traverse directories")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Resolved path escapes configured root")
    return target


def valid_ids(frame: pd.DataFrame, columns) -> pd.Series:
    valid = pd.Series(True, index=frame.index)
    for key in columns:
        valid &= frame[key].str.fullmatch(r"[1-9][0-9]*", na=False)
    return valid


def deduplicate(frame: pd.DataFrame, keys, name: str) -> pd.DataFrame:
    frame = frame.drop_duplicates().copy()
    if frame.duplicated(keys, keep=False).any():
        raise ValueError(f"Conflicting duplicate {name} keys; resolve source records before proceeding")
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Audit:
    def __init__(self):
        self.flow = []
        self.warnings = []

    def filter(self, frame, mask, stage, unit="rows"):
        mask = mask.fillna(False)
        kept = frame.loc[mask].copy()
        self.flow.append({"stage": stage, "unit": unit, "before": len(frame), "after": len(kept), "excluded": len(frame) - len(kept)})
        return kept
