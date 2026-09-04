"""Stable PyTorch sample contract for image-only and future multimodal models."""
from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .images import image_array
from .io import safe_path, sha256

INDEX_COLUMNS = ["sample_id", "subject_id", "hadm_id", "study_id", "dicom_id", "image_path", "study_time",
                 "admittime", "dischtime", "deathtime", "hours_since_admission", "view", "gender",
                 "age_at_admission", "age_is_topcoded", "label", "label_name", "sample_weight", "split"]


class MimicCXRDataset(Dataset):
    """Map-style Dataset returning image, label, weight and traceable IDs."""
    def __init__(self, run_dir, split, transform=None):
        self.run_dir = Path(run_dir).resolve()
        if not (self.run_dir / "SUCCESS.json").is_file():
            raise ValueError("Dataset run is incomplete (missing SUCCESS.json)")
        with (self.run_dir / "dataset_spec.json").open(encoding="utf-8") as stream:
            self.spec = json.load(stream)
        if self.spec.get("schema_version") != "1.0":
            raise ValueError("Unsupported dataset specification")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        index_path = self.run_dir / "index.csv"
        if sha256(index_path) != self.spec["index_sha256"]:
            raise ValueError("index.csv checksum does not match dataset_spec.json")
        self.frame = pd.read_csv(index_path, dtype={key: str for key in ["sample_id", "subject_id", "hadm_id", "study_id", "dicom_id", "image_path"]})
        if list(self.frame.columns) != INDEX_COLUMNS:
            raise ValueError("Unexpected unified index schema")
        self.frame = self.frame[self.frame.split.eq(split)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Split {split} is empty")
        self.root = Path(self.spec["image_root"])
        self.transform = transform
        self.image_size = int(self.spec["loader"]["image_size"])
        self.channels = int(self.spec["loader"]["channels"])
        self.mean = torch.tensor(self.spec["loader"]["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(self.spec["loader"]["std"], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = safe_path(self.root, row.image_path)
        image = torch.from_numpy(image_array(path, self.image_size, self.channels))
        image = (image - self.mean) / self.std
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "sample_weight": torch.tensor(float(row.sample_weight), dtype=torch.float32),
            "sample_id": row.sample_id,
            "subject_id": row.subject_id,
            "hadm_id": row.hadm_id,
            "study_id": row.study_id,
            "dicom_id": row.dicom_id,
        }


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(run_dir, split, *, transform=None, batch_size=None, num_workers=None, shuffle=None, seed=None):
    import torch
    dataset = MimicCXRDataset(run_dir, split, transform=transform)
    settings = dataset.spec["loader"]
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    num_workers = settings["num_workers"] if num_workers is None else num_workers
    shuffle = split == "train" if shuffle is None else shuffle
    seed = dataset.spec["split_seed"] if seed is None else seed
    generator = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                                       worker_init_fn=seed_worker, generator=generator)
