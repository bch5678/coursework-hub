"""Decode every radiograph once and reuse it across epochs.

The cache stores what the pipeline's image_array() returns, as float16;
normalisation still happens at load time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import upstream

CACHE_FORMAT = "bn5212-image-cache/1"
SPLITS = ("train", "val", "test")


def _loader_signature(spec: dict[str, Any]) -> dict[str, Any]:
    loader = spec["loader"]
    return {"image_size": int(loader["image_size"]), "channels": int(loader["channels"])}


def build_image_cache(
    run_dir: str | Path,
    output_path: str | Path,
    *,
    data_pipeline_path: str | None = None,
    splits: Sequence[str] = SPLITS,
    progress_every: int = 50,
) -> dict[str, Any]:
    """Decode every image of the given splits into one memory-mapped array."""
    import importlib

    upstream.load_pipeline(data_pipeline_path)
    dataset_module = upstream.dataset_module(data_pipeline_path)
    images = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.images")
    io_module = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.io")

    run_dir = Path(run_dir)
    sample_ids: list[str] = []
    paths: list[Path] = []
    spec = None
    for split in splits:
        base = dataset_module.MimicCXRDataset(run_dir, split)
        spec = base.spec
        root = Path(spec["image_root"])
        for row in base.frame.itertuples(index=False):
            sample_ids.append(str(row.sample_id))
            paths.append(io_module.safe_path(root, row.image_path))
    if spec is None or not sample_ids:
        raise ValueError(f"No samples found in {run_dir} for splits {tuple(splits)}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample_id across splits; the run is inconsistent")

    signature = _loader_signature(spec)
    size, channels = signature["image_size"], signature["channels"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array_path = output_path.with_suffix(".npy")

    shape = (len(sample_ids), channels, size, size)
    memmap = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float16, shape=shape)
    started = time.time()
    for position, path in enumerate(paths):
        memmap[position] = images.image_array(path, size, channels).astype(np.float16)
        if progress_every and (position + 1) % progress_every == 0:
            done = position + 1
            rate = done / max(time.time() - started, 1e-6)
            remaining = (len(paths) - done) / max(rate, 1e-6)
            print(
                f"  cached {done}/{len(paths)}  {rate:.1f} img/s  ~{remaining / 60:.1f} min left",
                flush=True,
            )
    memmap.flush()
    del memmap

    metadata = {
        "format": CACHE_FORMAT,
        "run_dir": str(run_dir.resolve()),
        "index_sha256": spec.get("index_sha256"),
        "loader": signature,
        "splits": list(splits),
        "sample_ids": sample_ids,
        "array": array_path.name,
        "dtype": "float16",
        "shape": list(shape),
        "built_seconds": round(time.time() - started, 1),
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "metadata": str(output_path),
        "array": str(array_path),
        "images": len(sample_ids),
        "megabytes": round(array_path.stat().st_size / 1024**2, 1),
        "seconds": metadata["built_seconds"],
    }


class ImageCache:
    """Read-side view of a built cache, validated against the run it came from."""

    def __init__(self, metadata_path: str | Path, spec: dict[str, Any] | None = None) -> None:
        self.path = Path(metadata_path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Image cache not found: {self.path}. Build it with bn5212-build-image-cache."
            )
        self.meta = json.loads(self.path.read_text(encoding="utf-8"))
        if self.meta.get("format") != CACHE_FORMAT:
            raise ValueError(f"{self.path.name} is not a {CACHE_FORMAT} file")
        if spec is not None:
            # A cache built at a different size or channel count would silently
            # feed the model the wrong tensors, so refuse rather than reshape.
            if self.meta["loader"] != _loader_signature(spec):
                raise ValueError(
                    f"Image cache was built for {self.meta['loader']} but this run needs "
                    f"{_loader_signature(spec)}; rebuild the cache."
                )
            if self.meta.get("index_sha256") != spec.get("index_sha256"):
                raise ValueError(
                    "Image cache was built from a different index.csv; rebuild it so the "
                    "cached images match this frozen run."
                )
        self.position_of = {name: index for index, name in enumerate(self.meta["sample_ids"])}
        self._array: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        # Opened lazily and per process, so a DataLoader worker maps it itself
        # instead of inheriting a handle through pickle.
        if self._array is None:
            self._array = np.load(self.path.with_suffix(".npy"), mmap_mode="r")
        return self._array

    def __contains__(self, sample_id: object) -> bool:
        return str(sample_id) in self.position_of

    def get(self, sample_id: str) -> np.ndarray:
        position = self.position_of.get(str(sample_id))
        if position is None:
            raise KeyError(
                f"{sample_id} is not in the image cache; rebuild it for this run."
            )
        return np.asarray(self.array[position], dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None  # memmaps do not survive pickling to a worker
        return state

    def __len__(self) -> int:
        return len(self.position_of)

