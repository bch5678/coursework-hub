"""The decoded-image cache.

A cache that returns anything other than what decoding would return is worse
than no cache: it changes results invisibly. The first test is therefore an
equivalence check against the pipeline's own decode path, and the rest cover the
guards that stop a stale or mismatched cache from being used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bn5212_training.cache import ImageCache, build_image_cache
from bn5212_training.data import TrainingDataset


@pytest.fixture(scope="module")
def built_cache(dataset_run, tmp_path_factory):
    output = tmp_path_factory.mktemp("cache") / "images.json"
    report = build_image_cache(dataset_run, output, progress_every=0)
    return output, report


def test_cache_covers_every_sample_of_the_run(built_cache, dataset_run):
    output, report = built_cache
    cache = ImageCache(output)
    import pandas as pd

    index = pd.read_csv(Path(dataset_run) / "index.csv", dtype={"sample_id": str})
    assert len(cache) == len(index) == report["images"]
    assert set(cache.position_of) == set(index["sample_id"])


def test_cached_images_match_a_fresh_decode(built_cache, dataset_run):
    """The whole point: reading the cache must equal decoding the DICOM."""
    output, _report = built_cache
    cache = ImageCache(output)
    decoded = TrainingDataset(dataset_run, "val", load_image=True)
    cached = TrainingDataset(dataset_run, "val", load_image=True, image_cache=cache)

    assert len(decoded) == len(cached)
    for index in range(len(decoded)):
        left, right = decoded[index], cached[index]
        assert left["sample_id"] == right["sample_id"]
        assert left["image"].shape == right["image"].shape
        # float16 storage costs about three decimal digits; the source is 8-bit
        # quantised before the resize, so this is the only difference expected.
        assert torch.allclose(left["image"], right["image"], atol=2e-3), left["sample_id"]


def test_cache_is_usable_from_dataloader_workers(built_cache, dataset_run):
    from torch.utils.data import DataLoader

    from bn5212_training.data import seed_worker

    output, _report = built_cache
    dataset = TrainingDataset(
        dataset_run, "val", load_image=True, image_cache=ImageCache(output)
    )
    loader = DataLoader(dataset, batch_size=2, num_workers=2, worker_init_fn=seed_worker)
    seen = sum(len(batch["sample_id"]) for batch in loader)
    assert seen == len(dataset)


def test_cache_built_for_another_geometry_is_refused(built_cache, dataset_run):
    output, _report = built_cache
    spec = TrainingDataset(dataset_run, "val", load_image=False).spec
    wrong = json.loads(json.dumps(spec))
    wrong["loader"]["image_size"] = spec["loader"]["image_size"] * 2
    with pytest.raises(ValueError, match="rebuild the cache"):
        ImageCache(output, wrong)


def test_cache_built_from_another_index_is_refused(built_cache, dataset_run):
    output, _report = built_cache
    spec = TrainingDataset(dataset_run, "val", load_image=False).spec
    stale = json.loads(json.dumps(spec))
    stale["index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different index.csv"):
        ImageCache(output, stale)


def test_matching_spec_is_accepted(built_cache, dataset_run):
    output, _report = built_cache
    spec = TrainingDataset(dataset_run, "val", load_image=False).spec
    assert len(ImageCache(output, spec)) > 0


def test_unknown_sample_raises_rather_than_returning_noise(built_cache):
    output, _report = built_cache
    cache = ImageCache(output)
    with pytest.raises(KeyError, match="rebuild"):
        cache.get("cxr_not_in_this_run")


def test_missing_cache_file_names_the_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="bn5212-build-image-cache"):
        ImageCache(tmp_path / "absent.json")


def test_cache_stores_float16_and_the_expected_shape(built_cache, dataset_run):
    output, report = built_cache
    meta = json.loads(Path(output).read_text(encoding="utf-8"))
    spec = TrainingDataset(dataset_run, "val", load_image=False).spec
    size = int(spec["loader"]["image_size"])
    channels = int(spec["loader"]["channels"])
    assert meta["dtype"] == "float16"
    assert meta["shape"] == [report["images"], channels, size, size]
    array = np.load(Path(output).with_suffix(".npy"), mmap_mode="r")
    assert array.dtype == np.float16
