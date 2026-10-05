"""Clinical time-series features for the clinical branch.

Two leakage rules are enforced here:

1. Every observation is strictly before the prediction time.
2. Normalisation statistics are fitted on fitting data only, then frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

# The 17 variables of the standard MIMIC benchmark extraction that MeTra builds
# on. MeTra reports 15 after dropping two that were 100% missing in its cohort,
# so the real K depends on the extraction and is read from the data, not fixed.
MIMIC_BENCHMARK_VARIABLES: tuple[str, ...] = (
    "Capillary refill rate",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glasgow coma scale eye opening",
    "Glasgow coma scale motor response",
    "Glasgow coma scale total",
    "Glasgow coma scale verbal response",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
)


class ClinicalFeatureProvider(Protocol):
    """Returns the clinical window of one admission as dense values plus a mask."""

    variable_names: Sequence[str]
    num_timesteps: int

    def features(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return (values [K, T] float32, mask [K, T] bool) for one index row.

        values must be 0.0 wherever mask is False. row carries at least
        hadm_id, admittime, study_time and hours_since_admission.
        """
        ...


def _guard_empty_window(mask: np.ndarray) -> None:
    """Keep one position valid when an admission has no observations at all.

    Attention over an all-padded sequence produces NaN.
    """
    if not mask.any():
        mask[0, 0] = True


def _cutoff_hours(row: Mapping[str, Any], num_timesteps: int) -> float:
    """Hours of history available strictly before the prediction time."""
    hours = row.get("hours_since_admission")
    if hours is None or (isinstance(hours, float) and math.isnan(hours)):
        return float(num_timesteps)
    return max(0.0, float(hours))


class SyntheticClinicalProvider:
    """Deterministic pseudo-data for offline development and tests.

    Pure noise by default (AUROC ~0.5). signal > 0 shifts values by the label to
    prove the loop can learn; never a benchmark result.
    """

    def __init__(
        self,
        variable_names: Sequence[str] = MIMIC_BENCHMARK_VARIABLES,
        num_timesteps: int = 48,
        *,
        seed: int = 5212,
        signal: float = 0.0,
        observed_fraction: float = 0.6,
    ) -> None:
        self.variable_names = tuple(variable_names)
        self.num_timesteps = int(num_timesteps)
        self.seed = int(seed)
        self.signal = float(signal)
        self.observed_fraction = float(observed_fraction)

    def _generator(self, key: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def features(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        shape = (len(self.variable_names), self.num_timesteps)
        rng = self._generator(str(row.get("hadm_id", "")))
        values = rng.standard_normal(shape).astype(np.float32)
        mask = rng.random(shape) < self.observed_fraction

        if self.signal:
            label = float(row.get("label", 0) or 0)
            values = values + np.float32(self.signal * label)

        # Same cutoff rule as the real provider, so tests exercise it too.
        limit = int(min(self.num_timesteps, math.ceil(_cutoff_hours(row, self.num_timesteps))))
        if limit < self.num_timesteps:
            mask[:, limit:] = False
        _guard_empty_window(mask)
        values = np.where(mask, values, np.float32(0.0)).astype(np.float32)
        return values, mask


class TableClinicalProvider:
    """Reads the long-format table produced by the data pipeline.

    Required columns: hadm_id, hour, variable, value. One row per observation,
    with hour an integer hour index counted from admittime. See
    docs/CLINICAL_FEATURE_SPEC.md for the full contract.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        variable_names: Sequence[str] | None = None,
        num_timesteps: int = 48,
    ) -> None:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Clinical feature table not found: {path}. Either point "
                "data.clinical_source at the file from the data owner, or set the "
                "clinical_provider to synthetic for offline development."
            )
        frame = self._read(path)
        required = {"hadm_id", "hour", "variable", "value"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        # An ICU-level extraction keys rows by stay_id, because one admission can
        # hold several ICU stays with different windows and therefore different
        # hour-0 references. An admission-level extraction leaves stay_id empty.
        self.key = "hadm_id"
        if "stay_id" in frame.columns:
            frame["stay_id"] = frame["stay_id"].fillna("").astype(str)
            if frame["stay_id"].str.strip().ne("").any():
                self.key = "stay_id"

        frame = frame.astype({"hadm_id": str, "variable": str})
        frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        bad = frame["hour"].isna() | frame["value"].isna()
        if bool(bad.any()):
            raise ValueError(
                f"{path.name} has {int(bad.sum())} rows with a non-numeric hour or value"
            )
        frame["hour"] = frame["hour"].astype(int)
        if bool((frame["hour"] < 0).any()):
            raise ValueError(f"{path.name} contains negative hour indices")

        self.num_timesteps = int(num_timesteps)
        frame = frame[frame["hour"] < self.num_timesteps]
        self.variable_names = tuple(
            variable_names if variable_names is not None else sorted(frame["variable"].unique())
        )
        if not self.variable_names:
            raise ValueError(f"{path.name} declares no clinical variables")
        unknown = set(frame["variable"].unique()) - set(self.variable_names)
        if unknown:
            raise ValueError(
                f"{path.name} has variables outside the declared set: {sorted(unknown)}"
            )

        self._row_of = {name: position for position, name in enumerate(self.variable_names)}
        # Last observation per (hadm_id, variable, hour) wins, matching the
        # most-recent-measurement rule the extraction is specified to apply.
        frame = frame.drop_duplicates(subset=[self.key, "variable", "hour"], keep="last")
        self._by_unit = {
            str(key): group for key, group in frame.groupby(self.key, sort=False)
        }

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def has_observations(self, row: Mapping[str, Any]) -> bool:
        """Whether the table holds any row for this admission or ICU stay."""
        return str(row.get(self.key, "") or "") in self._by_unit

    def features(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        shape = (len(self.variable_names), self.num_timesteps)
        values = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        group = self._by_unit.get(str(row.get(self.key, "") or ""))
        if group is None or group.empty:
            mask[:, 0] = True
            return values, mask

        # An ICU-level table counts hours from the ICU intime and already covers
        # the whole window, so hours_since_admission must not truncate it.
        if self.key == "stay_id":
            limit = self.num_timesteps
        else:
            limit = int(min(self.num_timesteps, math.ceil(_cutoff_hours(row, self.num_timesteps))))
        group = group[group["hour"] < max(limit, 1)]
        for variable, hour, value in zip(group["variable"], group["hour"], group["value"]):
            position = self._row_of.get(variable)
            if position is None:
                continue
            values[position, hour] = np.float32(value)
            mask[position, hour] = True
        _guard_empty_window(mask)
        values = np.where(mask, values, np.float32(0.0)).astype(np.float32)
        return values, mask


def build_provider(cfg, *, variable_names: Sequence[str] | None = None) -> ClinicalFeatureProvider:
    """Construct the provider named by data.clinical_provider."""
    if cfg.clinical_provider == "synthetic":
        return SyntheticClinicalProvider(
            variable_names or MIMIC_BENCHMARK_VARIABLES,
            cfg.clinical_timesteps,
            signal=cfg.clinical_signal,
        )
    if cfg.clinical_provider == "table":
        if not cfg.clinical_source:
            raise ValueError("data.clinical_source is required for the table provider")
        return TableClinicalProvider(
            cfg.clinical_source,
            variable_names=variable_names,
            num_timesteps=cfg.clinical_timesteps,
        )
    raise ValueError(
        f"Unknown clinical_provider {cfg.clinical_provider!r}; expected synthetic or table"
    )


@dataclass
class ClinicalNormalizer:
    """Per-variable standardisation fitted on observed train values only."""

    mean: np.ndarray
    std: np.ndarray
    variable_names: tuple[str, ...]
    fitted_on: str = "train"
    num_observations: int = 0

    @classmethod
    def fit(
        cls,
        provider: ClinicalFeatureProvider,
        rows: Sequence[Mapping[str, Any]],
        *,
        split: str = "train",
    ) -> ClinicalNormalizer:
        if split != "train":
            raise ValueError(
                "Normalisation statistics must be fitted on the train split only; "
                f"refusing to fit on {split!r}"
            )
        names = tuple(provider.variable_names)
        totals = np.zeros(len(names), dtype=np.float64)
        squares = np.zeros(len(names), dtype=np.float64)
        counts = np.zeros(len(names), dtype=np.int64)
        for row in rows:
            values, mask = provider.features(row)
            observed = mask.astype(np.float64)
            totals += (values * observed).sum(axis=1)
            squares += ((values.astype(np.float64) ** 2) * observed).sum(axis=1)
            counts += mask.sum(axis=1)

        safe = np.maximum(counts, 1)
        mean = totals / safe
        variance = np.maximum(squares / safe - mean**2, 0.0)
        std = np.sqrt(variance)
        # A variable that is never observed, or constant, is left untouched rather
        # than amplified by dividing through a near-zero scale.
        mean = np.where(counts > 0, mean, 0.0)
        std = np.where((counts > 0) & (std > 1e-6), std, 1.0)
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            variable_names=names,
            num_observations=int(counts.sum()),
        )

    def apply(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scaled = (values - self.mean[:, None]) / self.std[:, None]
        return np.where(mask, scaled, np.float32(0.0)).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "variable_names": list(self.variable_names),
            "fitted_on": self.fitted_on,
            "num_observations": self.num_observations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClinicalNormalizer:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            variable_names=tuple(payload["variable_names"]),
            fitted_on=str(payload.get("fitted_on", "train")),
            num_observations=int(payload.get("num_observations", 0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ClinicalNormalizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
