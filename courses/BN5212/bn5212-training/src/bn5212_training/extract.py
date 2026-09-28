"""Extract clinical time series from MIMIC-IV into the long-format table.

Produces exactly the file docs/CLINICAL_FEATURE_SPEC.md describes, which
TableClinicalProvider then consumes. It writes a side-car file and never touches
the frozen run directory, index.csv or the pipeline's Dataset, so the data
project's module stays untouched.

Three properties this module is built around:

* chartevents is ~30 GB compressed. It is streamed in chunks and filtered on
  itemid and cohort hadm_id inside each chunk, so peak memory stays flat and the
  output is only the cohort's own rows.
* Leakage is blocked on two clocks. A measurement is kept only when both its
  charttime and its storetime are before the prediction time: a value charted at
  hour 3 but stored at hour 50 was not available when the radiograph was taken.
  The dataloader spec calls this out explicitly.
* An admission with several radiographs is cut at its earliest study_time, so no
  sample of that admission can see another sample's future.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

CHARTEVENTS_COLUMNS = ["hadm_id", "charttime", "storetime", "itemid", "value", "valuenum"]
LABEVENTS_COLUMNS = ["hadm_id", "charttime", "storetime", "itemid", "value", "valuenum"]

CONVERSIONS = {
    "fahrenheit_to_celsius": lambda v: (v - 32.0) * 5.0 / 9.0,
    "inches_to_cm": lambda v: v * 2.54,
    "pounds_to_kg": lambda v: v * 0.45359237,
    # Charted FiO2 is sometimes a percentage and sometimes a fraction.
    "percent_to_fraction_if_above_one": lambda v: np.where(v > 1.0, v / 100.0, v),
}


@dataclass
class ExtractionStats:
    rows_scanned: int = 0
    rows_kept: int = 0
    dropped_out_of_cohort: int = 0
    dropped_after_cutoff: int = 0
    dropped_late_storetime: int = 0
    dropped_out_of_range: int = 0
    dropped_unparsable: int = 0
    per_variable: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_scanned": self.rows_scanned,
            "rows_kept": self.rows_kept,
            "dropped": {
                "out_of_cohort": self.dropped_out_of_cohort,
                "after_study_time": self.dropped_after_cutoff,
                "stored_after_study_time": self.dropped_late_storetime,
                "out_of_physiologic_range": self.dropped_out_of_range,
                "unparsable_value": self.dropped_unparsable,
            },
            "per_variable": self.per_variable,
        }


def find_table(root: Path, name: str) -> Path:
    """Locate a MIMIC table under hosp/, icu/ or the root, csv or csv.gz."""
    candidates = [
        root / part / f"{name}{suffix}"
        for part in ("", "hosp", "icu")
        for suffix in (".csv.gz", ".csv")
    ]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(
            f"Could not find {name}.csv[.gz] under {root} (looked in ., hosp/, icu/)"
        )
    if len({path.name for path in found}) > 1 or len(found) > 1:
        # Several candidates means two MIMIC versions could be mixed by accident.
        raise ValueError(
            f"Ambiguous {name} table; found {[str(p) for p in found]}. "
            "Keep one copy so the run cannot mix dataset versions."
        )
    return found[0]


def load_item_map(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported item map schema: {payload.get('schema_version')!r}")
    if not payload.get("variables"):
        raise ValueError("Item map declares no variables")
    return payload


def build_cohort(
    run_dir: str | Path,
    *,
    unit: str = "admission",
    mimic_root: str | Path | None = None,
    icu_hours: float = 48.0,
) -> pd.DataFrame:
    """One observation window per study unit.

    admission: the window runs from admittime to the earliest radiograph of that
    admission, which is when the prediction is made.

    icu_stay: the window runs from the ICU intime for icu_hours, and the
    prediction is made at the end of it, as in MeTra. An admission can hold more
    than one ICU stay, so the window is keyed by stay_id; ICU stays of a patient
    do not overlap, so an event falls in at most one window.

    Columns: unit_id, hadm_id, stay_id, window_start, prediction_time, cutoff_hours.
    """
    index_path = Path(run_dir) / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found; point --run-dir at a completed run")
    frame = pd.read_csv(index_path, dtype={"hadm_id": str, "stay_id": str})
    required = {"hadm_id", "admittime", "study_time", "hours_since_admission"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"index.csv is missing columns: {sorted(missing)}")
    frame["admittime"] = pd.to_datetime(frame["admittime"])
    frame["study_time"] = pd.to_datetime(frame["study_time"])
    frame["hours_since_admission"] = pd.to_numeric(frame["hours_since_admission"])

    if unit == "admission":
        cohort = frame.groupby("hadm_id", as_index=False).agg(
            window_start=("admittime", "min"),
            prediction_time=("study_time", "min"),
            cutoff_hours=("hours_since_admission", "min"),
        )
        cohort["stay_id"] = ""
        cohort["unit_id"] = cohort["hadm_id"]
        return cohort[["unit_id", "hadm_id", "stay_id", "window_start", "prediction_time", "cutoff_hours"]]

    if unit != "icu_stay":
        raise ValueError(f"Unknown cohort unit {unit!r}; expected admission or icu_stay")
    if "stay_id" not in frame.columns or frame["stay_id"].fillna("").eq("").all():
        raise ValueError(
            "index.csv carries no stay_id, so this run was not built with "
            "cohort.unit=icu_stay. Rebuild the run or extract with --cohort-unit admission."
        )
    if mimic_root is None:
        raise ValueError("icu_stay extraction needs --mimic-root to read icustays")

    stays = pd.read_csv(
        find_table(Path(mimic_root), "icustays"),
        usecols=["hadm_id", "stay_id", "intime"],
        dtype={"hadm_id": str, "stay_id": str},
    )
    stays["intime"] = pd.to_datetime(stays["intime"], errors="coerce")
    selected = frame[["hadm_id", "stay_id"]].drop_duplicates()
    cohort = selected.merge(stays, on=["hadm_id", "stay_id"], how="left", validate="one_to_one")
    unknown = cohort["intime"].isna()
    if bool(unknown.any()):
        raise ValueError(
            f"{int(unknown.sum())} stay_id values in index.csv are absent from icustays; "
            "check that --mimic-root matches the run"
        )
    cohort["window_start"] = cohort["intime"]
    cohort["prediction_time"] = cohort["intime"] + pd.to_timedelta(float(icu_hours), unit="h")
    cohort["cutoff_hours"] = float(icu_hours)
    cohort["unit_id"] = cohort["stay_id"]
    return cohort[["unit_id", "hadm_id", "stay_id", "window_start", "prediction_time", "cutoff_hours"]]


def _resolve_lookup(item_map: dict[str, Any], source: str) -> tuple[dict[int, str], dict[int, str]]:
    """itemid -> variable, and itemid -> conversion name, for one source table."""
    key = "itemids" if source == "chartevents" else "lab_itemids"
    variable_of: dict[int, str] = {}
    conversion_of: dict[int, str] = {}
    for name, spec in item_map["variables"].items():
        for itemid in spec.get(key, []) or []:
            variable_of[int(itemid)] = name
        for itemid, conversion in (spec.get("convert") or {}).items():
            if conversion not in CONVERSIONS:
                raise ValueError(f"{name}: unknown conversion {conversion!r}")
            # A converted itemid belongs to this variable even when it is only
            # listed under convert (e.g. Temperature in Fahrenheit).
            if source == "chartevents":
                variable_of.setdefault(int(itemid), name)
                conversion_of[int(itemid)] = conversion
    return variable_of, conversion_of


def inspect_items(mimic_root: str | Path, item_map: dict[str, Any]) -> pd.DataFrame:
    """Report the label each configured itemid actually carries in this dataset.

    A wrong itemid then shows up as a mismatched label instead of silently
    extracting a different signal.
    """
    root = Path(mimic_root)
    dictionaries: dict[str, pd.DataFrame] = {}
    for source, table in (("chartevents", "d_items"), ("labevents", "d_labitems")):
        try:
            dictionaries[source] = pd.read_csv(find_table(root, table))
        except FileNotFoundError:
            dictionaries[source] = pd.DataFrame(columns=["itemid", "label"])

    rows = []
    for source in ("chartevents", "labevents"):
        variable_of, _ = _resolve_lookup(item_map, source)
        dictionary = dictionaries[source]
        labels = (
            dict(zip(dictionary["itemid"].astype(int), dictionary["label"].astype(str)))
            if not dictionary.empty
            else {}
        )
        for itemid, variable in sorted(variable_of.items()):
            rows.append(
                {
                    "variable": variable,
                    "source": source,
                    "itemid": itemid,
                    "label_in_dataset": labels.get(itemid, "<NOT FOUND>"),
                    "found": itemid in labels,
                }
            )
    return pd.DataFrame(rows)


def _numeric_values(chunk: pd.DataFrame, spec_of: dict[str, dict[str, Any]]) -> pd.Series:
    """Prefer valuenum; fall back to the categorical map for scored variables."""
    values = pd.to_numeric(chunk["valuenum"], errors="coerce")
    unresolved = values.isna()
    if not unresolved.any() or "value" not in chunk.columns:
        return values

    text = chunk.loc[unresolved, "value"].astype(str).str.strip().str.lower()
    variables = chunk.loc[unresolved, "variable"]
    mapped = pd.Series(np.nan, index=text.index, dtype=float)
    for variable in variables.unique():
        categorical = (spec_of.get(variable) or {}).get("categorical_map")
        if not categorical:
            continue
        selector = variables.eq(variable)
        mapped.loc[selector] = text.loc[selector].map(
            {key.lower(): float(value) for key, value in categorical.items()}
        )
    values.loc[unresolved] = mapped
    return values


def _iter_chunks(path: Path, columns: list[str], chunksize: int) -> Iterator[pd.DataFrame]:
    available = pd.read_csv(path, nrows=0).columns
    use = [name for name in columns if name in available]
    if "storetime" not in use:
        # Older extracts may not carry storetime; the caller degrades gracefully.
        pass
    reader = pd.read_csv(
        path,
        usecols=use,
        chunksize=chunksize,
        dtype={"hadm_id": "string", "itemid": "Int64", "value": "string"},
        low_memory=False,
    )
    for chunk in reader:
        yield chunk


def _extract_source(
    path: Path,
    source: str,
    cohort: pd.DataFrame,
    item_map: dict[str, Any],
    stats: ExtractionStats,
    chunksize: int,
) -> pd.DataFrame:
    variable_of, conversion_of = _resolve_lookup(item_map, source)
    if not variable_of:
        return pd.DataFrame(columns=["unit_id", "hadm_id", "stay_id", "variable", "hour", "value", "charttime", "source"])

    spec_of = item_map["variables"]
    wanted_admissions = set(cohort["hadm_id"])
    wanted_items = set(variable_of)
    columns = CHARTEVENTS_COLUMNS if source == "chartevents" else LABEVENTS_COLUMNS
    collected: list[pd.DataFrame] = []

    for chunk in _iter_chunks(path, columns, chunksize):
        stats.rows_scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(wanted_items)]
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(str)
        before = len(chunk)
        chunk = chunk[chunk["hadm_id"].isin(wanted_admissions)]
        stats.dropped_out_of_cohort += before - len(chunk)
        if chunk.empty:
            continue

        chunk["variable"] = chunk["itemid"].astype(int).map(variable_of)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])

        joined = chunk.merge(cohort, on="hadm_id", how="inner")
        elapsed = (joined["charttime"] - joined["window_start"]).dt.total_seconds() / 3600.0
        in_window = elapsed.ge(0) & elapsed.lt(joined["cutoff_hours"])
        stats.dropped_after_cutoff += int((~in_window).sum())
        joined = joined[in_window]
        elapsed = elapsed[in_window]
        if joined.empty:
            continue

        # Second clock: a value must also have been stored before prediction time.
        if "storetime" in joined.columns:
            stored = pd.to_datetime(joined["storetime"], errors="coerce")
            available = stored.isna() | stored.le(joined["prediction_time"])
            stats.dropped_late_storetime += int((~available).sum())
            joined = joined[available]
            elapsed = elapsed[available]
            if joined.empty:
                continue

        values = _numeric_values(joined, spec_of)
        for itemid, conversion in conversion_of.items():
            selector = joined["itemid"].astype(int).eq(itemid)
            if selector.any():
                values.loc[selector] = CONVERSIONS[conversion](values.loc[selector].to_numpy())

        unparsable = values.isna()
        stats.dropped_unparsable += int(unparsable.sum())
        keep = ~unparsable

        for variable in joined.loc[keep, "variable"].unique():
            low, high = (spec_of[variable].get("valid_range") or [-np.inf, np.inf])[:2]
            selector = keep & joined["variable"].eq(variable)
            out_of_range = selector & (~values.between(low, high))
            stats.dropped_out_of_range += int(out_of_range.sum())
            keep = keep & ~out_of_range

        if not keep.any():
            continue
        collected.append(
            pd.DataFrame(
                {
                    "unit_id": joined.loc[keep, "unit_id"].to_numpy(),
                    "hadm_id": joined.loc[keep, "hadm_id"].to_numpy(),
                    "stay_id": joined.loc[keep, "stay_id"].to_numpy(),
                    "variable": joined.loc[keep, "variable"].to_numpy(),
                    "hour": np.floor(elapsed[keep].to_numpy()).astype(int),
                    "value": values[keep].to_numpy(dtype=float),
                    "charttime": joined.loc[keep, "charttime"].to_numpy(),
                    "source": source,
                }
            )
        )

    if not collected:
        return pd.DataFrame(
            columns=["unit_id", "hadm_id", "stay_id", "variable", "hour", "value", "charttime", "source"]
        )
    return pd.concat(collected, ignore_index=True)


def _aggregate(frame: pd.DataFrame, item_map: dict[str, Any]) -> pd.DataFrame:
    """One row per (hadm_id, variable, hour), using each variable's rule."""
    if frame.empty:
        return pd.DataFrame(columns=["unit_id", "hadm_id", "stay_id", "hour", "variable", "value"])
    default = item_map.get("hour_aggregation_default", "mean")
    pieces: list[pd.DataFrame] = []
    for variable, group in frame.groupby("variable", sort=False):
        rule = (item_map["variables"].get(variable) or {}).get("aggregation", default)
        keys = ["unit_id", "hadm_id", "stay_id", "variable", "hour"]
        if rule == "last":
            # Ordinal scores must not be averaged into a value that never occurred.
            ordered = group.sort_values("charttime")
            reduced = ordered.groupby(keys, as_index=False)["value"].last()
        elif rule == "median":
            reduced = group.groupby(keys, as_index=False)["value"].median()
        else:
            reduced = group.groupby(keys, as_index=False)["value"].mean()
        pieces.append(reduced)
    return pd.concat(pieces, ignore_index=True)


def _derive(frame: pd.DataFrame, item_map: dict[str, Any]) -> pd.DataFrame:
    """Compute variables defined as an operation over other variables."""
    derived_specs = {
        name: spec
        for name, spec in item_map["variables"].items()
        if spec.get("source") == "derived"
    }
    if not derived_specs or frame.empty:
        return frame
    pieces = [frame]
    for name, spec in derived_specs.items():
        parts = spec.get("derived_from") or []
        subset = frame[frame["variable"].isin(parts)]
        if subset.empty:
            continue
        wide = subset.pivot_table(
            index=["unit_id", "hadm_id", "stay_id", "hour"], columns="variable", values="value", aggfunc="last"
        )
        # Only emit the derived value where every component exists for that hour.
        complete = wide.dropna(subset=[part for part in parts if part in wide.columns])
        if complete.empty or not all(part in complete.columns for part in parts):
            continue
        total = complete[parts].sum(axis=1)
        low, high = (spec.get("valid_range") or [-np.inf, np.inf])[:2]
        total = total[total.between(low, high)]
        if total.empty:
            continue
        pieces.append(
            total.reset_index().rename(columns={0: "value"}).assign(variable=name)[
                ["unit_id", "hadm_id", "stay_id", "variable", "hour", "value"]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def extract_clinical_features(
    *,
    run_dir: str | Path,
    mimic_root: str | Path,
    item_map_path: str | Path,
    output_path: str | Path,
    max_hours: int = 48,
    chunksize: int = 1_000_000,
    include_labs: bool = True,
    unit: str = "admission",
) -> dict[str, Any]:
    """Run the extraction and write the long-format table plus a QA report."""
    item_map = load_item_map(item_map_path)
    root = Path(mimic_root)
    cohort = build_cohort(run_dir, unit=unit, mimic_root=root, icu_hours=float(max_hours))
    cohort["cutoff_hours"] = cohort["cutoff_hours"].clip(upper=float(max_hours))
    # In ICU mode the study unit is the stay, so coverage and per-variable counts
    # are reported per stay rather than per admission.
    unit_key = "stay_id" if unit == "icu_stay" else "hadm_id"
    stats = ExtractionStats()

    frames = [
        _extract_source(
            find_table(root, "chartevents"), "chartevents", cohort, item_map, stats, chunksize
        )
    ]
    if include_labs:
        try:
            lab_path = find_table(root, "labevents")
        except FileNotFoundError:
            lab_path = None
        if lab_path is not None:
            frames.append(
                _extract_source(lab_path, "labevents", cohort, item_map, stats, chunksize)
            )

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame(columns=["unit_id", "hadm_id", "stay_id", "variable", "hour", "value", "charttime", "source"])

    # chartevents lives in the ICU module, so it only exists for admissions that
    # included an ICU stay. On an admission-level cohort a large share of patients
    # can have no vital signs at all, which would quietly train the clinical
    # branch on mostly-missing input. Surface it here rather than let it hide.
    units = int(len(cohort))
    coverage = {}
    for source in ("chartevents", "labevents"):
        subset = combined[combined["source"].eq(source)] if not combined.empty else combined
        covered = int(subset["unit_id"].nunique()) if not subset.empty else 0
        coverage[source] = {
            "units_with_any": covered,
            "unit_coverage": round(covered / units, 4) if units else None,
        }
    coverage["any_source"] = {
        "units_with_any": int(combined["unit_id"].nunique()) if not combined.empty else 0,
    }

    aggregated = _aggregate(combined, item_map)
    aggregated = _derive(aggregated, item_map)
    aggregated = aggregated[["hadm_id", "stay_id", "hour", "variable", "value"]].sort_values(
        ["hadm_id", "stay_id", "hour", "variable"]
    )
    stats.rows_kept = len(aggregated)

    for name in item_map["variables"]:
        subset = aggregated[aggregated["variable"].eq(name)]
        covered = int(subset[unit_key].nunique())
        stats.per_variable[name] = {
            "observations": int(len(subset)),
            "units_with_any": covered,
            "unit_missing_rate": round(1.0 - covered / units, 4) if units else None,
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False, lineterminator="\n")

    report = {
        "output": str(output_path),
        "run_dir": str(Path(run_dir).resolve()),
        "mimic_root": str(root.resolve()),
        "item_map": str(Path(item_map_path).resolve()),
        "max_hours": int(max_hours),
        "cohort_unit": unit,
        "units_in_cohort": units,
        "source_coverage": coverage,
        "variables_with_no_data": sorted(
            name for name, info in stats.per_variable.items() if info["observations"] == 0
        ),
        **stats.as_dict(),
    }
    chartevents_coverage = coverage["chartevents"]["unit_coverage"]
    if chartevents_coverage is not None and chartevents_coverage < 0.5:
        report["warnings"] = [
            f"Only {chartevents_coverage:.1%} of cohort units have any chartevents row. "
            "chartevents belongs to the MIMIC-IV icu module, so admissions without an ICU "
            "stay have no vital signs at all. Consider an ICU-level cohort, or treat the "
            "clinical branch as lab-only for the remaining admissions."
        ]
    report_path = output_path.with_name(output_path.stem.split(".")[0] + "_qa.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    report["qa_report"] = str(report_path)
    return report
