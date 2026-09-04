"""Reproducible grouped splits. Freeze the resulting assignments for experiments."""
import hashlib
import numpy as np
import pandas as pd


def allocate_counts(n, ratios):
    if n < 3:
        raise ValueError("At least three patients are required for train/val/test")
    ideal = np.asarray(ratios) * n
    counts = np.floor(ideal).astype(int)
    for index in np.argsort(-(ideal - counts), kind="stable")[:n - counts.sum()]:
        counts[index] += 1
    for index in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[index] += 1
    return counts


def assign_splits(frame, cfg, audit):
    subject = frame.groupby("subject_id", sort=True).label.max().rename("patient_stratum").reset_index()
    if len(subject) < 3:
        raise ValueError("At least three eligible patients are needed; cannot safely create all splits")
    c = cfg["split"]
    ratios = [c[x] for x in ("train", "val", "test")]
    stratify = c["stratify"] and subject.patient_stratum.nunique() == 2 and subject.groupby("patient_stratum").size().min() >= 3
    if c["stratify"] and not stratify:
        audit.warnings.append("Too few patients/classes for stratification; used deterministic unstratified patient split")
    groups = [group for _, group in subject.groupby("patient_stratum")] if stratify else [subject]
    result = []
    for group in groups:
        group = group.copy()
        group["order"] = group.subject_id.map(lambda x: hashlib.sha256(f"{c['seed']}:{x}".encode()).hexdigest())
        group = group.sort_values(["order", "subject_id"])
        sizes = allocate_counts(len(group), ratios)
        group["split"] = np.repeat(["train", "val", "test"], sizes)
        result.append(group.drop(columns="order"))
    assignments = pd.concat(result).sort_values("subject_id").reset_index(drop=True)
    return frame.merge(assignments[["subject_id", "split"]], on="subject_id", validate="many_to_one"), assignments


def validate_index(frame):
    if frame.empty or frame.sample_id.duplicated().any() or frame.dicom_id.duplicated().any():
        raise ValueError("Empty index or duplicate sample/image IDs")
    if not set(frame.label.unique()) <= {0, 1} or frame.label.isna().any():
        raise ValueError("Invalid labels")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("All three splits must be nonempty")
    for key in ["subject_id", "hadm_id", "study_id", "dicom_id", "image_path"]:
        if frame.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"Cross-split leakage for {key}")
    if frame.groupby("hadm_id").label.nunique().gt(1).any():
        raise ValueError("Inconsistent labels for the same admission")
    if not (frame.study_time.ge(frame.admittime) & frame.study_time.lt(frame.event_cutoff) & frame.study_time.lt(frame.dischtime)).all():
        raise ValueError("Invalid prediction timestamp")
