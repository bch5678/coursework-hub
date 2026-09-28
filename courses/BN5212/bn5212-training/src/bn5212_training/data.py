"""Batch assembly on top of the frozen pipeline Dataset.

The pipeline owns the cohort, the split and the image transform; this module only
composes on top of it. It never re-reads index.csv by itself and never rebuilds a
split, so the hash and patient-isolation checks in MimicCXRDataset still run for
every training job.

Batch keys:
    image           float32 [B, C, H, W]   (absent when the image is not loaded)
    clinical        float32 [B, K, T]      (only when the clinical branch is on)
    clinical_mask   bool    [B, K, T]
    label           int64   [B]
    sample_weight   float32 [B]
    sample_id, subject_id, hadm_id, study_id, dicom_id   list[str] of length B
"""
from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import upstream
from .clinical import ClinicalFeatureProvider, ClinicalNormalizer
from .config import TrainingConfig

SPLITS = ("train", "val", "test")


class TrainingDataset(Dataset):
    """Pipeline samples plus the clinical window, assembled per index row."""

    def __init__(
        self,
        run_dir: str,
        split: str,
        *,
        data_pipeline_path: str | None = None,
        provider: ClinicalFeatureProvider | None = None,
        normalizer: ClinicalNormalizer | None = None,
        load_image: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        dataset_module = upstream.dataset_module(data_pipeline_path)
        # Instantiating the pipeline Dataset is what verifies SUCCESS.json, the
        # dataset_spec schema and the index.csv checksum.
        self.base = dataset_module.MimicCXRDataset(run_dir, split, transform=transform)
        self.split = split
        self.spec = self.base.spec
        self.frame = self.base.frame
        self.load_image = bool(load_image)
        self.provider = provider
        self.normalizer = normalizer

        if provider is not None and normalizer is not None:
            declared = tuple(provider.variable_names)
            if declared != tuple(normalizer.variable_names):
                raise ValueError(
                    "Clinical normaliser was fitted on different variables than the "
                    f"provider supplies: {normalizer.variable_names} vs {declared}"
                )

    @property
    def image_size(self) -> int:
        return int(self.spec["loader"]["image_size"])

    @property
    def channels(self) -> int:
        return int(self.spec["loader"]["channels"])

    @property
    def num_variables(self) -> int:
        return 0 if self.provider is None else len(self.provider.variable_names)

    @property
    def num_timesteps(self) -> int:
        return 0 if self.provider is None else int(self.provider.num_timesteps)

    def rows(self) -> list[dict[str, Any]]:
        """Index rows for this split, used to fit train-only statistics."""
        return self.frame.to_dict(orient="records")

    def label_array(self) -> np.ndarray:
        return self.frame["label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        if self.load_image:
            sample = self.base[index]
        else:
            # Clinical-only training does not need to decode a JPG/DICOM per step.
            sample = {
                "label": torch.tensor(int(row.label), dtype=torch.long),
                "sample_weight": torch.tensor(float(row.sample_weight), dtype=torch.float32),
                "sample_id": row.sample_id,
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "study_id": row.study_id,
                "dicom_id": row.dicom_id,
            }

        if self.provider is not None:
            values, mask = self.provider.features(row.to_dict())
            if self.normalizer is not None:
                values = self.normalizer.apply(values, mask)
            sample["clinical"] = torch.from_numpy(np.ascontiguousarray(values))
            sample["clinical_mask"] = torch.from_numpy(np.ascontiguousarray(mask))
        return sample


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: TrainingDataset,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
) -> DataLoader:
    """DataLoader with the same reproducibility wiring as the pipeline default."""
    settings = dataset.spec["loader"]
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    num_workers = settings["num_workers"] if num_workers is None else num_workers
    shuffle = dataset.split == "train" if shuffle is None else shuffle
    seed = dataset.spec["split_seed"] if seed is None else seed
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def build_datasets(
    cfg: TrainingConfig,
    *,
    splits: Sequence[str] = SPLITS,
    provider: ClinicalFeatureProvider | None = None,
    normalizer: ClinicalNormalizer | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, TrainingDataset]:
    """Build one dataset per split, sharing the provider and frozen statistics."""
    return {
        split: TrainingDataset(
            cfg.data.run_dir,
            split,
            data_pipeline_path=cfg.data.data_pipeline_path,
            provider=provider if cfg.uses_clinical else None,
            normalizer=normalizer if cfg.uses_clinical else None,
            load_image=cfg.uses_image,
            transform=transform if split == "train" else None,
        )
        for split in splits
    }


def make_loaders(
    cfg: TrainingConfig, datasets: dict[str, TrainingDataset]
) -> dict[str, DataLoader]:
    return {
        split: make_loader(
            dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=split == "train",
            seed=cfg.seed,
        )
        for split, dataset in datasets.items()
    }
