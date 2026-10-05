"""Clinical extraction against MIMIC-shaped synthetic tables.

The extraction cannot be checked against real MIMIC on a development machine, so
every rule it enforces is pinned here: the two leakage clocks, unit conversion,
physiologic ranges, ordinal scores, hour aggregation, and the round trip through
the provider that consumes the output.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bn5212_training.clinical import TableClinicalProvider
from bn5212_training.extract import (
    build_cohort,
    extract_clinical_features,
    inspect_items,
    load_item_map,
)

ITEM_MAP_PATH = Path(__file__).resolve().parents[1] / "configs" / "clinical_items.json"
HEART_RATE = 220045
TEMPERATURE_F = 223761
GCS_EYE = 220739
GCS_VERBAL = 223900
GCS_MOTOR = 223901


@pytest.fixture
def cohort(dataset_run):
    return build_cohort(dataset_run)


def _write_mimic(root: Path, events: list[dict]) -> Path:
    """Write a minimal MIMIC-IV layout containing only what the extractor reads."""
    icu = root / "icu"
    icu.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        events,
        columns=["hadm_id", "charttime", "storetime", "itemid", "value", "valuenum"],
    ).to_csv(icu / "chartevents.csv", index=False)
    pd.DataFrame(
        [
            {"itemid": HEART_RATE, "label": "Heart Rate"},
            {"itemid": TEMPERATURE_F, "label": "Temperature Fahrenheit"},
            {"itemid": GCS_EYE, "label": "GCS - Eye Opening"},
            {"itemid": GCS_VERBAL, "label": "GCS - Verbal Response"},
            {"itemid": GCS_MOTOR, "label": "GCS - Motor Response"},
        ]
    ).to_csv(icu / "d_items.csv", index=False)
    return root


def _event(hadm_id, admittime, hour, itemid, valuenum, *, value="", store_hour=None):
    charttime = admittime + timedelta(hours=hour)
    store = admittime + timedelta(hours=hour if store_hour is None else store_hour)
    return {
        "hadm_id": hadm_id,
        "charttime": charttime.strftime("%Y-%m-%d %H:%M:%S"),
        "storetime": store.strftime("%Y-%m-%d %H:%M:%S"),
        "itemid": itemid,
        "value": value,
        "valuenum": valuenum,
    }


def _event_at(hadm_id, admittime, charttime, itemid, valuenum, *, value=""):
    """Like _event but pinned to an absolute charttime, for boundary tests."""
    return {
        "hadm_id": hadm_id,
        "charttime": charttime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "storetime": charttime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "itemid": itemid,
        "value": value,
        "valuenum": valuenum,
    }


def _run(tmp_path, dataset_run, events, **kwargs):
    root = _write_mimic(tmp_path / "mimic", events)
    output = tmp_path / "clinical_features.csv"
    report = extract_clinical_features(
        run_dir=dataset_run,
        mimic_root=root,
        item_map_path=ITEM_MAP_PATH,
        output_path=output,
        include_labs=False,
        **kwargs,
    )
    frame = pd.read_csv(output, dtype={"hadm_id": str})
    return frame, report


def test_cohort_describes_one_window_per_study_unit(cohort):
    assert {"unit_id", "hadm_id", "stay_id", "window_start", "prediction_time", "cutoff_hours"} <= set(cohort.columns)
    assert (cohort["cutoff_hours"] >= 0).all()
    assert cohort["hadm_id"].is_unique


def test_output_matches_the_documented_column_contract(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [_event(row.hadm_id, row.window_start, 0, HEART_RATE, 88)]
    frame, _ = _run(tmp_path, dataset_run, events)
    assert list(frame.columns) == ["hadm_id", "stay_id", "hour", "variable", "value"]
    assert frame.iloc[0]["variable"] == "Heart Rate"
    assert frame.iloc[0]["value"] == pytest.approx(88.0)


def test_the_cutoff_is_the_radiograph_instant_not_the_hour_bin(tmp_path, dataset_run, cohort):
    """A value one second before the radiograph was available; one second after was not.

    The boundary is study_time itself, not the start of the hour containing it, so
    the hour bin holding study_time can legitimately be partly populated.
    """
    row = cohort.iloc[0]
    events = [
        _event_at(row.hadm_id, row.window_start, row.prediction_time - timedelta(seconds=1), HEART_RATE, 80),
        _event_at(row.hadm_id, row.window_start, row.prediction_time + timedelta(seconds=1), HEART_RATE, 140),
        _event_at(row.hadm_id, row.window_start, row.prediction_time + timedelta(hours=5), HEART_RATE, 150),
    ]
    frame, report = _run(tmp_path, dataset_run, events)
    assert frame.loc[frame["variable"].eq("Heart Rate"), "value"].tolist() == [pytest.approx(80.0)]
    assert report["dropped"]["after_study_time"] == 2


def test_measurements_stored_after_study_time_are_dropped(tmp_path, dataset_run, cohort):
    """Charted early but stored late: not available when the image was taken."""
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 1, HEART_RATE, 80, store_hour=1),
        _event(row.hadm_id, row.window_start, 2, HEART_RATE, 99, store_hour=500),
    ]
    frame, report = _run(tmp_path, dataset_run, events)
    values = frame.loc[frame["variable"].eq("Heart Rate"), "value"].tolist()
    assert values == [pytest.approx(80.0)]
    assert report["dropped"]["stored_after_study_time"] == 1


def test_events_before_admission_are_dropped(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, -3, HEART_RATE, 70),
        _event(row.hadm_id, row.window_start, 1, HEART_RATE, 75),
    ]
    frame, _ = _run(tmp_path, dataset_run, events)
    assert frame.loc[frame["variable"].eq("Heart Rate"), "hour"].tolist() == [1]


def test_admissions_outside_the_cohort_are_ignored(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 0, HEART_RATE, 80),
        _event("99999999", row.window_start, 0, HEART_RATE, 80),
    ]
    frame, report = _run(tmp_path, dataset_run, events)
    assert set(frame["hadm_id"]) == {row.hadm_id}
    assert report["dropped"]["out_of_cohort"] == 1


def test_fahrenheit_is_converted_to_celsius(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [_event(row.hadm_id, row.window_start, 0, TEMPERATURE_F, 98.6)]
    frame, _ = _run(tmp_path, dataset_run, events)
    temperature = frame.loc[frame["variable"].eq("Temperature"), "value"]
    assert temperature.iloc[0] == pytest.approx(37.0, abs=0.01)


def test_physiologically_impossible_values_are_dropped(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 0, HEART_RATE, 0),
        _event(row.hadm_id, row.window_start, 1, HEART_RATE, 9999),
        _event(row.hadm_id, row.window_start, 2, HEART_RATE, 72),
    ]
    frame, report = _run(tmp_path, dataset_run, events)
    assert frame.loc[frame["variable"].eq("Heart Rate"), "value"].tolist() == [pytest.approx(72.0)]
    assert report["dropped"]["out_of_physiologic_range"] == 2


def test_categorical_scores_are_mapped_from_text(tmp_path, dataset_run, cohort):
    """MIMIC charts GCS components as text when valuenum is absent."""
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 0, GCS_EYE, None, value="Spontaneously"),
        _event(row.hadm_id, row.window_start, 0, GCS_VERBAL, None, value="Oriented"),
        _event(row.hadm_id, row.window_start, 0, GCS_MOTOR, None, value="Obeys Commands"),
    ]
    frame, _ = _run(tmp_path, dataset_run, events)
    scores = dict(zip(frame["variable"], frame["value"]))
    assert scores["Glasgow coma scale eye opening"] == pytest.approx(4.0)
    assert scores["Glasgow coma scale verbal response"] == pytest.approx(5.0)
    assert scores["Glasgow coma scale motor response"] == pytest.approx(6.0)


def test_gcs_total_is_derived_only_when_all_components_exist(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 0, GCS_EYE, 4),
        _event(row.hadm_id, row.window_start, 0, GCS_VERBAL, 5),
        _event(row.hadm_id, row.window_start, 0, GCS_MOTOR, 6),
        _event(row.hadm_id, row.window_start, 1, GCS_EYE, 3),  # incomplete hour
    ]
    frame, _ = _run(tmp_path, dataset_run, events)
    total = frame[frame["variable"].eq("Glasgow coma scale total")]
    assert total["hour"].tolist() == [0]
    assert total["value"].iloc[0] == pytest.approx(15.0)


def test_repeated_values_in_one_hour_are_averaged(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 1, HEART_RATE, 70),
        _event(row.hadm_id, row.window_start, 1, HEART_RATE, 90),
    ]
    frame, _ = _run(tmp_path, dataset_run, events)
    assert frame.loc[frame["variable"].eq("Heart Rate"), "value"].iloc[0] == pytest.approx(80.0)


def test_ordinal_scores_take_the_last_value_not_the_mean(tmp_path, dataset_run, cohort):
    """Averaging GCS would invent a score that was never observed."""
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, 1, GCS_MOTOR, 6),
        _event(row.hadm_id, row.window_start, 1, GCS_MOTOR, 3),
    ]
    frame, _ = _run(tmp_path, dataset_run, events)
    value = frame.loc[frame["variable"].eq("Glasgow coma scale motor response"), "value"].iloc[0]
    assert value == pytest.approx(3.0)


def test_qa_report_records_coverage_and_empty_variables(tmp_path, dataset_run, cohort):
    row = cohort.iloc[0]
    events = [_event(row.hadm_id, row.window_start, 0, HEART_RATE, 80)]
    _, report = _run(tmp_path, dataset_run, events)
    assert report["per_variable"]["Heart Rate"]["observations"] == 1
    assert "Capillary refill rate" in report["variables_with_no_data"]
    assert Path(report["qa_report"]).is_file()
    saved = json.loads(Path(report["qa_report"]).read_text(encoding="utf-8"))
    assert saved["units_in_cohort"] == len(cohort)


def test_extracted_table_is_readable_by_the_training_provider(tmp_path, dataset_run, cohort):
    """The round trip that matters: extraction output feeds the clinical branch."""
    row = cohort.iloc[0]
    events = [
        _event(row.hadm_id, row.window_start, hour, HEART_RATE, 70 + hour)
        for hour in range(int(row.cutoff_hours))
    ]
    _, report = _run(tmp_path, dataset_run, events)

    provider = TableClinicalProvider(report["output"], num_timesteps=48)
    values, mask = provider.features(
        {"hadm_id": row.hadm_id, "hours_since_admission": row.cutoff_hours}
    )
    position = list(provider.variable_names).index("Heart Rate")
    assert mask[position, : int(row.cutoff_hours)].all()
    assert values[position, 0] == pytest.approx(70.0)
    assert not mask[position, int(row.cutoff_hours) :].any()


def test_inspect_items_flags_an_itemid_missing_from_the_dataset(tmp_path, dataset_run):
    root = _write_mimic(tmp_path / "mimic", [])
    report = inspect_items(root, load_item_map(ITEM_MAP_PATH))
    heart_rate = report[report["itemid"].eq(HEART_RATE)].iloc[0]
    assert heart_rate["found"]
    assert heart_rate["label_in_dataset"] == "Heart Rate"
    # The fixture dictionary only defines five items, so the rest must be flagged.
    assert (~report["found"]).any()
    assert "<NOT FOUND>" in set(report["label_in_dataset"])


def test_report_warns_when_few_units_have_chartevents(tmp_path, dataset_run, cohort):
    """chartevents is ICU-only, so an admission-level cohort can be mostly empty.

    Without this warning the clinical branch would quietly train on input that is
    missing for most patients, which looks like a modelling failure rather than a
    cohort definition problem.
    """
    row = cohort.iloc[0]
    events = [_event(row.hadm_id, row.window_start, 0, HEART_RATE, 80)]
    _, report = _run(tmp_path, dataset_run, events)

    coverage = report["source_coverage"]["chartevents"]
    assert coverage["units_with_any"] == 1
    assert coverage["unit_coverage"] == pytest.approx(1 / len(cohort), abs=1e-4)
    assert "warnings" in report
    assert "icu module" in report["warnings"][0]


def test_no_warning_when_every_unit_has_chartevents(tmp_path, dataset_run, cohort):
    events = [
        _event(row.hadm_id, row.window_start, 0, HEART_RATE, 80)
        for row in cohort.itertuples()
    ]
    _, report = _run(tmp_path, dataset_run, events)
    assert report["source_coverage"]["chartevents"]["unit_coverage"] == pytest.approx(1.0)
    assert "warnings" not in report


def test_shipped_item_map_is_internally_consistent():
    item_map = load_item_map(ITEM_MAP_PATH)
    from bn5212_training.clinical import MIMIC_BENCHMARK_VARIABLES

    declared = set(item_map["variables"])
    assert declared == set(MIMIC_BENCHMARK_VARIABLES), (
        "the item map and the default variable list must describe the same variables"
    )
    for name, spec in item_map["variables"].items():
        assert spec.get("source") in {"chartevents", "labevents", "derived"}, name
        low, high = spec["valid_range"]
        assert low < high, name
        if spec["source"] == "derived":
            assert set(spec["derived_from"]) <= declared, name
