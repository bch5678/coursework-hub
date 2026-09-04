#!/usr/bin/env python3
"""Server smoke test: validates the full published-run contract and loads batches."""
import argparse
import json
from pathlib import Path


def check_run(run_dir):
    import torch
    import pandas as pd
    from src.data.dataset import make_dataloader
    from src.data.io import sha256
    from src.data.splits import validate_index
    run_dir = Path(run_dir)
    success = json.loads((run_dir / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["status"] == "complete"
    for name, checksum in success["artifacts_sha256"].items():
        assert sha256(run_dir / name) == checksum, f"Changed artifact: {name}"
    frame = pd.read_csv(run_dir / "index.csv", dtype={"subject_id": str, "hadm_id": str, "study_id": str, "dicom_id": str})
    for key in ["study_time", "admittime", "dischtime", "deathtime"]:
        frame[key] = pd.to_datetime(frame[key], errors="coerce")
    frame["event_cutoff"] = frame.deathtime.fillna(frame.dischtime)
    validate_index(frame)
    seen = {}
    for split in ["train", "val", "test"]:
        loader = make_dataloader(run_dir, split, batch_size=4, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        assert batch["image"].dtype == torch.float32 and batch["image"].ndim == 4
        assert tuple(batch["image"].shape[1:]) == (loader.dataset.channels, loader.dataset.image_size, loader.dataset.image_size)
        assert batch["label"].dtype == torch.int64 and set(batch["label"].tolist()) <= {0, 1}
        assert torch.isfinite(batch["image"]).all() and torch.isfinite(batch["sample_weight"]).all()
        seen[split] = set(loader.dataset.frame.subject_id)
        print(f"{split}: {len(loader.dataset)} samples; first batch {tuple(batch['image'].shape)}")
    assert not (seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"])
    print("DataLoader contract and patient-level isolation: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    check_run(parser.parse_args().run_dir)
