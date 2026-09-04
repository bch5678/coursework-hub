"""Patient linkage, strict interval matching, configurable labels and sampling."""
from __future__ import annotations

import bisect
from pathlib import Path
import pandas as pd

from .images import decode_image
from .io import safe_path


def align_images(cxr, admissions, cfg, audit):
    # Search each patient's sorted admissions; handles overlaps without a large Cartesian join.
    groups = {}
    for subject, group in admissions.groupby("subject_id", sort=False):
        group = group.sort_values(["admittime", "hadm_id"])
        groups[subject] = (group.admittime.tolist(), group.dischtime.tolist(), group.hadm_id.tolist())
    assignments, counts = [], []
    for row in cxr.itertuples(index=False):
        group = groups.get(row.subject_id)
        matches = []
        if group:
            starts, ends, hadms = group
            upto = bisect.bisect_right(starts, row.study_time)
            matches = [hadms[i] for i in range(upto) if row.study_time < ends[i]]
        assignments.append(matches[0] if len(matches) == 1 else "")
        counts.append(len(matches))
    cxr = cxr.copy()
    cxr["hadm_id"], cxr["matches"] = assignments, counts
    cxr = audit.filter(cxr, cxr.matches.ne(0), "matched_inpatient_interval", "images")
    cxr = audit.filter(cxr, cxr.matches.eq(1), "unique_inpatient_interval", "images")
    frame = cxr.merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
    frame = audit.filter(frame, frame.age_eligible, "adult_with_valid_demographics", "images")
    expire = pd.to_numeric(frame.hospital_expire_flag, errors="coerce")
    consistent = (expire.isin([0, 1]) & ~frame.death_time_invalid
                  & (~expire.eq(0) | frame.deathtime.isna())
                  & (frame.deathtime.isna() | ((frame.deathtime >= frame.admittime) & (frame.deathtime <= frame.dischtime))))
    frame = audit.filter(frame, consistent, "valid_mortality_and_death_time", "images")
    missing_death = (pd.to_numeric(frame.hospital_expire_flag).eq(1) & frame.deathtime.isna()).sum()
    if missing_death:
        audit.warnings.append(f"{int(missing_death)} candidate images have positive mortality but missing deathtime; discharge is the only available cutoff")
    frame["event_cutoff"] = frame.dischtime.where(frame.deathtime.isna(), frame.deathtime)
    frame["hours_since_admission"] = (frame.study_time - frame.admittime).dt.total_seconds() / 3600
    before_event = (frame.event_cutoff - frame.study_time).dt.total_seconds() / 3600
    frame = audit.filter(frame, (before_event > 0) & before_event.ge(cfg["alignment"]["minimum_hours_before_end"]), "strictly_before_outcome", "images")
    max_hours = cfg["alignment"]["max_hours_after_admission"]
    if max_hours is not None:
        frame = audit.filter(frame, frame.hours_since_admission.le(max_hours), "early_imaging_window", "images")
    label = cfg["label"]
    values = frame[label["column"]]
    if label["kind"] == "in_hospital_mortality":
        values = pd.to_numeric(values).astype(int).astype(str)
    mapping = {str(x): 1 for x in label["positive_values"]} | {str(x): 0 for x in label["negative_values"]}
    frame["label"] = values.map(mapping)
    frame = audit.filter(frame, frame.label.notna(), "known_binary_label", "images")
    frame["label"] = frame.label.astype(int)
    frame["label_name"] = label["kind"] if label["kind"] == "in_hospital_mortality" else label["column"]
    return frame


def select_images(frame, cfg, audit):
    if cfg["cleaning"]["verify_images"]:
        good, failures = [], {}
        for row in frame.itertuples(index=False):
            try:
                decode_image(safe_path(Path(cfg["paths"]["image_root"]), row.image_path))
                good.append(True)
            except Exception as exc:
                good.append(False)
                name = type(exc).__name__
                failures[name] = failures.get(name, 0) + 1
        fraction = 1 - sum(good) / len(good) if good else 0
        if failures:
            audit.warnings.append(f"Image decoding failures by exception type: {failures}")
        if failures and (cfg["cleaning"]["invalid_image_policy"] == "error" or fraction > cfg["cleaning"]["max_invalid_image_fraction"]):
            raise ValueError("Image decoding failure threshold exceeded; verify files and DICOM codecs")
        frame = audit.filter(frame, pd.Series(good, index=frame.index, dtype=bool), "decodable_nonconstant_image", "images")
    else:
        audit.warnings.append("Image decoding verification disabled; DataLoader may fail on corrupt files")
    # View preference breaks equal-time ties; random IDs never stand in for timestamps.
    ranking = {str(view).upper(): i for i, view in enumerate(cfg["cohort"]["views"])}
    frame["view_rank"] = frame.view.map(ranking)
    frame = frame.sort_values(["subject_id", "hadm_id", "study_time", "view_rank", "study_id", "dicom_id"])
    selection = cfg["cohort"]["selection"]
    keys = {"first_per_admission": ["hadm_id"], "first_per_study": ["study_id"], "all_images": ["dicom_id"]}[selection]
    frame = audit.filter(frame, ~frame.duplicated(keys), selection, "images")
    if frame.empty:
        raise ValueError("No eligible images; inspect paths, timestamps, views and cohort settings")
    frame["sample_id"] = "cxr_" + frame.dicom_id
    frame["sample_weight"] = 1.0 / frame.groupby("hadm_id").dicom_id.transform("size")
    return frame.reset_index(drop=True)
