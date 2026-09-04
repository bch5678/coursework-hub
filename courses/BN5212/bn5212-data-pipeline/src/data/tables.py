"""Clean MIMIC-IV admissions and patients without imputing outcome labels."""
from pathlib import Path
import pandas as pd

from .io import deduplicate, read_csv, table_path, valid_ids


def load_hospital_tables(cfg, audit):
    root = Path(cfg["paths"]["mimic_iv_root"])
    admissions_path, patients_path = table_path(root, "admissions"), table_path(root, "patients")
    required = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag", cfg["label"]["column"]]
    a = read_csv(admissions_path, required)
    a = a[list(dict.fromkeys(required))]
    a = audit.filter(a, valid_ids(a, ["subject_id", "hadm_id"]), "valid_admission_ids", "admissions")
    before = len(a)
    a = deduplicate(a, ["hadm_id"], "admission")
    audit.flow.append({"stage": "deduplicate_admissions", "unit": "admissions", "before": before, "after": len(a), "excluded": before - len(a)})
    a["death_time_invalid"] = False
    for key in ["admittime", "dischtime", "deathtime"]:
        parsed = pd.to_datetime(a[key], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if key == "deathtime":
            a["death_time_invalid"] = a[key].ne("") & parsed.isna()
        a[key] = parsed
    a = audit.filter(a, a.admittime.notna() & a.dischtime.notna() & (a.admittime < a.dischtime), "valid_admission_interval", "admissions")
    p = read_csv(patients_path, ["subject_id", "anchor_age", "anchor_year", "gender"])
    p = p[["subject_id", "anchor_age", "anchor_year", "gender"]]
    p = audit.filter(p, valid_ids(p, ["subject_id"]), "valid_patient_ids", "patients")
    p = deduplicate(p, ["subject_id"], "patient")
    for key in ["anchor_age", "anchor_year"]:
        p[key] = pd.to_numeric(p[key], errors="coerce")
    a = a.merge(p, on="subject_id", how="left", validate="many_to_one")
    # Ages >89 at the anchor year are deidentified to 91, not an exact age.
    a["age_at_admission"] = a.anchor_age + a.admittime.dt.year - a.anchor_year
    a["age_is_topcoded"] = a.anchor_age.eq(91)
    a["age_eligible"] = (a.anchor_age.between(0, 91) & a.anchor_year.between(2000, 2300)
                           & a.anchor_age.mod(1).eq(0) & a.anchor_year.mod(1).eq(0)
                           & a.age_at_admission.ge(cfg["cohort"]["min_age"]))
    return a, [admissions_path, patients_path]
