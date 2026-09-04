"""CXR sidecar adapter or direct DICOM-header extraction for course subsets."""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from .io import deduplicate, read_csv, safe_path, valid_ids

REQUIRED = ["subject_id", "study_id", "dicom_id", "StudyDate", "StudyTime", "ViewPosition"]


def parse_study_time(date: pd.Series, time: pd.Series) -> pd.Series:
    date = date.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    time = time.astype(str).str.strip()
    valid = date.str.fullmatch(r"[0-9]{8}") & time.str.fullmatch(r"[0-9]{1,6}(\.[0-9]{1,6})?")
    parts = time.str.split(".", n=1, expand=True).reindex(columns=[0, 1])
    integral = parts[0].str.zfill(6)
    fraction = parts[1].fillna("0").str.pad(6, side="right", fillchar="0")
    valid &= (integral.str[:2].astype("string") <= "23") & (integral.str[2:4] <= "59") & (integral.str[4:6] <= "59")
    full = (date + " " + integral + "." + fraction).where(valid)
    return pd.to_datetime(full, format="%Y%m%d %H%M%S.%f", errors="coerce")


def scan_dicom(root: Path, audit) -> tuple[pd.DataFrame, list[Path]]:
    import pydicom
    rows, paths, bad = [], [], 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".dcm":
            continue
        paths.append(path)
        relative = path.relative_to(root).as_posix()
        match = re.search(r"(?:^|/)p([1-9][0-9]*)/s([1-9][0-9]*)/([^/]+)\.dcm$", relative, re.I)
        if not match:
            raise ValueError("DICOM folder layout needs p<subject_id>/s<study_id>/<dicom_id>.dcm; use a metadata CSV for other layouts")
        subject, study, dicom = match.groups()
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["StudyDate", "StudyTime", "ViewPosition", "PatientID"])
            patient_id = str(getattr(ds, "PatientID", "")).strip()
            if patient_id and patient_id not in {subject, "p" + subject}:
                raise ValueError("DICOM PatientID disagrees with folder subject_id")
            time = str(getattr(ds, "StudyTime", ""))
            if not re.fullmatch(r"[0-9]{6}(\.[0-9]{1,6})?", time):
                time = ""  # Reduced-precision headers are not precise enough for alignment.
            rows.append({"subject_id": subject, "study_id": study, "dicom_id": dicom,
                         "StudyDate": str(getattr(ds, "StudyDate", "")), "StudyTime": time,
                         "ViewPosition": str(getattr(ds, "ViewPosition", "")), "image_path": relative})
        except (OSError, ValueError, EOFError, pydicom.errors.InvalidDicomError):
            bad += 1
    audit.flow.append({"stage": "read_dicom_headers", "unit": "images", "before": len(paths), "after": len(rows), "excluded": bad})
    if bad:
        audit.warnings.append(f"Excluded {bad}/{len(paths)} unreadable or inconsistent DICOM headers")
    return pd.DataFrame(rows, columns=REQUIRED + ["image_path"]), paths


def load_cxr(cfg, audit):
    root = Path(cfg["paths"]["image_root"])
    if not root.is_dir():
        raise ValueError("Configured image_root does not exist")
    source = cfg["paths"]["cxr_metadata"]
    input_files = []
    if source:
        input_files.append(Path(source))
        frame = read_csv(source, REQUIRED)
        frame = frame[REQUIRED + (["image_path"] if "image_path" in frame else [])]
    else:
        frame, headers = scan_dicom(root, audit)
        excluded = audit.flow[-1]["excluded"]
        if excluded and (cfg["cleaning"]["invalid_image_policy"] == "error" or excluded / max(1, len(headers)) > cfg["cleaning"]["max_invalid_image_fraction"]):
            raise ValueError("DICOM header failure threshold exceeded; check cohort audit")
    if frame.empty:
        raise ValueError("No CXR metadata; for JPG/PNG specify paths.cxr_metadata")
    frame = audit.filter(frame, valid_ids(frame, ["subject_id", "study_id"]) & frame.dicom_id.str.fullmatch(r"[A-Za-z0-9_-]+", na=False), "valid_cxr_ids", "images")
    before = len(frame)
    frame = deduplicate(frame, ["dicom_id"], "CXR image")
    audit.flow.append({"stage": "deduplicate_cxr", "unit": "images", "before": before, "after": len(frame), "excluded": before - len(frame)})
    frame["study_time"] = parse_study_time(frame.StudyDate, frame.StudyTime)
    frame["view"] = frame.ViewPosition.str.upper().str.strip()
    # A study must have one subject and a consistent acquisition timestamp.
    grouped = frame.groupby("study_id")
    inconsistent = grouped.subject_id.nunique().gt(1) | grouped.study_time.nunique(dropna=False).gt(1)
    frame = audit.filter(frame, ~frame.study_id.isin(inconsistent[inconsistent].index), "consistent_study_metadata", "images")
    frame = audit.filter(frame, frame.study_time.notna(), "valid_study_timestamp", "images")
    frame = audit.filter(frame, frame.view.isin([str(x).upper() for x in cfg["cohort"]["views"]]), "allowed_view", "images")
    resolved = []
    for row in frame.itertuples(index=False):
        if hasattr(row, "image_path") and row.image_path:
            path = safe_path(root, row.image_path)
            resolved.append(path.relative_to(root).as_posix() if path.is_file() else "")
            continue
        base = f"files/p{row.subject_id[:2]}/p{row.subject_id}/s{row.study_id}/{row.dicom_id}"
        candidates = [safe_path(root, base + suffix) for suffix in [".dcm", ".jpg", ".jpeg", ".png"]]
        found = [p for p in candidates if p.is_file()]
        if len(found) > 1:
            raise ValueError("Multiple image formats for one dicom_id; specify image_path explicitly")
        resolved.append(found[0].relative_to(root).as_posix() if found else "")
    frame["image_path"] = resolved
    frame = audit.filter(frame, frame.image_path.ne(""), "present_in_local_subset", "images")
    if frame.image_path.duplicated().any():
        raise ValueError("Multiple image IDs refer to the same resolved file")
    return frame, input_files
