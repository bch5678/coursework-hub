"""Batch assembly on top of the frozen pipeline Dataset.

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
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import upstream
from .clinical import ClinicalFeatureProvider, ClinicalNormalizer

SPLITS = ("train", "val", "test")


class TrainingDataset(Dataset):
    """Pipeline samples plus the clinical window, assembled per index row."""

    def __init__(
        self,
        run_dir: str,
        split: str | Sequence[str],
        *,
        data_pipeline_path: str | None = None,
        provider: ClinicalFeatureProvider | None = None,
        normalizer: ClinicalNormalizer | None = None,
        load_image: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        subjects: Sequence[str] | None = None,
        image_cache: Any | None = None,
    ) -> None:
        # A cross-validation fold draws its patients from train and val at once,
        # so several frozen splits can back one dataset.
        splits = (split,) if isinstance(split, str) else tuple(split)
        if not splits or any(name not in SPLITS for name in splits):
            raise ValueError(f"split must be one or more of {SPLITS}")
        if len(set(splits)) != len(splits):
            raise ValueError("Duplicate split names")
        dataset_module = upstream.dataset_module(data_pipeline_path)

        wanted = None if subjects is None else {str(s) for s in subjects}
        bases, frames, owners = [], [], []
        for position, name in enumerate(splits):
            # Instantiating the pipeline Dataset is what verifies SUCCESS.json,
            # the dataset_spec schema and the index.csv checksum.
            base = dataset_module.MimicCXRDataset(run_dir, name, transform=transform)
            frame = base.frame
            keep = np.ones(len(frame), dtype=bool)
            if wanted is not None:
                keep = frame["subject_id"].astype(str).isin(wanted).to_numpy()
            if keep.any():
                frames.append(frame.loc[keep])
                # The pipeline Dataset still indexes its own unfiltered frame, so
                # remember which base and row each of our rows came from.
                owners.extend((position, int(row)) for row in np.flatnonzero(keep))
            bases.append(base)

        if not frames:
            raise ValueError(f"No rows left in splits {splits} for the requested subjects")
        self.splits = splits
        self.split = splits[0] if len(splits) == 1 else "+".join(splits)
        self.spec = bases[0].spec
        self.frame = pd.concat(frames, ignore_index=True)
        self._owners = owners
        self.load_image = bool(load_image)
        # Only the image path needs the pipeline objects. Dropping them otherwise
        # keeps them out of what a DataLoader worker has to unpickle.
        self.bases = bases if (self.load_image and image_cache is None) else None
        self.image_cache = image_cache if self.load_image else None
        self.transform = transform
        loader = self.spec["loader"]
        self._mean = torch.tensor(loader["mean"], dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(loader["std"], dtype=torch.float32)[:, None, None]
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
        if self.load_image and self.image_cache is None:
            owner, position = self._owners[index]
            sample = self.bases[owner][position]
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
            if self.load_image:
                # Normalisation stays here rather than in the cache, so mean/std
                # can change without rebuilding it.
                image = torch.from_numpy(self.image_cache.get(row.sample_id))
                image = (image - self._mean) / self._std
                if self.transform is not None:
                    image = self.transform(image)
                sample["image"] = image

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

