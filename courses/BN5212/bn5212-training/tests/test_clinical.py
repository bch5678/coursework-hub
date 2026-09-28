"""Leakage rules for the clinical branch.

Two things must hold no matter what the extraction hands us: nothing at or after
the prediction time reaches the model, and normalisation statistics come from the
train split alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from bn5212_training.clinical import (
    MIMIC_BENCHMARK_VARIABLES,
    ClinicalNormalizer,
    SyntheticClinicalProvider,
    TableClinicalProvider,
)

VARIABLES = ("Heart Rate", "Respiratory rate", "Glucose")


def _row(hadm_id="1", hours=48.0, label=0):
    return {"hadm_id": hadm_id, "hours_since_admission": hours, "label": label}


def test_values_are_zero_wherever_the_mask_is_false():
    provider = SyntheticClinicalProvider(VARIABLES, 12)
    values, mask = provider.features(_row())
    assert values.shape == mask.shape == (3, 12)
    assert np.all(values[~mask] == 0.0)


def test_observations_at_or_after_the_prediction_time_are_dropped():
    provider = SyntheticClinicalProvider(VARIABLES, 24)
    _, mask = provider.features(_row(hours=6.0))
    assert not mask[:, 6:].any(), "history beyond study_time must not be visible"
    _, full = provider.features(_row(hours=24.0))
    assert full[:, 6:].any(), "a longer window should expose more history"


def test_a_patient_with_no_history_still_yields_a_usable_window():
    provider = SyntheticClinicalProvider(VARIABLES, 12)
    _, mask = provider.features(_row(hours=0.0))
    assert mask.any(), "an all-false mask would produce NaN inside attention"


def test_the_synthetic_provider_is_deterministic():
    first = SyntheticClinicalProvider(VARIABLES, 12).features(_row("42"))
    second = SyntheticClinicalProvider(VARIABLES, 12).features(_row("42"))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    other = SyntheticClinicalProvider(VARIABLES, 12).features(_row("43"))
    assert not np.array_equal(first[0], other[0])


def test_the_default_variable_list_matches_the_benchmark_extraction():
    # MeTra reports 15 of these 17 after dropping two that were fully missing.
    assert len(MIMIC_BENCHMARK_VARIABLES) == 17
    assert "Heart Rate" in MIMIC_BENCHMARK_VARIABLES
    assert "Glasgow coma scale total" in MIMIC_BENCHMARK_VARIABLES


def test_normalizer_refuses_to_fit_on_anything_but_train():
    provider = SyntheticClinicalProvider(VARIABLES, 8)
    with pytest.raises(ValueError, match="train split only"):
        ClinicalNormalizer.fit(provider, [_row()], split="val")


def test_normalizer_standardises_observed_values():
    provider = SyntheticClinicalProvider(VARIABLES, 24)
    rows = [_row(str(index)) for index in range(80)]
    normalizer = ClinicalNormalizer.fit(provider, rows, split="train")

    scaled = []
    for row in rows:
        values, mask = provider.features(row)
        result = normalizer.apply(values, mask)
        assert np.all(result[~mask] == 0.0)
        scaled.append(result[mask])
    pooled = np.concatenate(scaled)
    assert abs(float(pooled.mean())) < 0.15
    assert 0.8 < float(pooled.std()) < 1.25


def test_normalizer_round_trips_through_json(tmp_path):
    provider = SyntheticClinicalProvider(VARIABLES, 8)
    normalizer = ClinicalNormalizer.fit(provider, [_row(str(i)) for i in range(5)], split="train")
    path = tmp_path / "norm.json"
    normalizer.save(path)
    restored = ClinicalNormalizer.load(path)
    assert restored.variable_names == normalizer.variable_names
    assert np.allclose(restored.mean, normalizer.mean)
    assert np.allclose(restored.std, normalizer.std)


def test_a_constant_variable_is_not_amplified():
    """Dividing by a near-zero scale would blow a constant column up."""
    provider = SyntheticClinicalProvider(VARIABLES, 4)
    normalizer = ClinicalNormalizer.fit(provider, [_row("only")], split="train")
    assert np.all(normalizer.std > 0)


def test_table_provider_reads_the_long_format(tmp_path):
    path = tmp_path / "clinical.csv"
    path.write_text(
        "hadm_id,hour,variable,value\n"
        "10,0,Heart Rate,80\n"
        "10,1,Heart Rate,85\n"
        "10,2,Glucose,140\n"
        "11,0,Heart Rate,70\n",
        encoding="utf-8",
    )
    provider = TableClinicalProvider(path, variable_names=VARIABLES, num_timesteps=12)
    values, mask = provider.features(_row("10", hours=12.0))
    assert values[0, 0] == pytest.approx(80.0)
    assert values[0, 1] == pytest.approx(85.0)
    assert values[2, 2] == pytest.approx(140.0)
    assert not mask[1].any(), "a variable with no rows stays unobserved"


def test_table_provider_enforces_the_cutoff_even_if_the_file_ignores_it():
    """Defence in depth: the extraction may be wrong, the loader must not pass it on."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/clinical.csv"
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("hadm_id,hour,variable,value\n")
            for hour in range(12):
                stream.write(f"10,{hour},Heart Rate,{80 + hour}\n")
        provider = TableClinicalProvider(path, variable_names=VARIABLES, num_timesteps=12)
        _, mask = provider.features(_row("10", hours=4.0))
        assert mask[0, :4].any()
        assert not mask[0, 4:].any(), "rows past study_time must be dropped"


def test_table_provider_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthetic"):
        TableClinicalProvider(tmp_path / "absent.csv")


def test_table_provider_rejects_undeclared_variables(tmp_path):
    path = tmp_path / "clinical.csv"
    path.write_text(
        "hadm_id,hour,variable,value\n10,0,Mystery Variable,1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside the declared set"):
        TableClinicalProvider(path, variable_names=VARIABLES, num_timesteps=12)


def test_table_provider_rejects_missing_columns(tmp_path):
    path = tmp_path / "clinical.csv"
    path.write_text("hadm_id,hour,value\n10,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        TableClinicalProvider(path, num_timesteps=12)
