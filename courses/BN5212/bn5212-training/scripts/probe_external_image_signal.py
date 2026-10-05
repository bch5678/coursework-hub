"""Can the image branch be pretrained on radiographs outside the study cohort? No.

    python scripts/probe_external_image_signal.py count | cache | frozen | finetune

Only patients outside the study cohort are read.

count     Non-cohort inpatient AP films: 299, with 7 in-hospital deaths.
cache     Decode the non-cohort AP films; label = death within 180 days.
frozen    Logistic regression on frozen features, patient-grouped 5-fold CV:
          ViT-B/16 0.542, CheXpert DenseNet121 0.502, pathology scores 0.558.
finetune  CXR-only model with the last two ViT blocks unfrozen: 0.54-0.58 per
          epoch, the best epoch read off after the fact.
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bn5212_training import upstream
from bn5212_training.augment import build_train_transform
from bn5212_training.config import load_config
from bn5212_training.crossval import make_folds
from bn5212_training.extract import find_table
from bn5212_training.metrics import auroc, bootstrap_interval
from bn5212_training.model import build_model
from bn5212_training.trainer import _build_optimizer, set_seed

HORIZON_DAYS = 180


def _external_films(run_dir: Path, mimic_root: Path) -> pd.DataFrame:
    """AP films of patients outside the study cohort, with both candidate labels."""
    films = pd.read_csv(run_dir / "cxr_metadata_snapshot.csv", dtype=str)
    study = set(pd.read_csv(run_dir / "index.csv", dtype=str)["subject_id"])
    films = films[~films["subject_id"].isin(study) & films["view"].eq("AP")].copy()
    films["study_time"] = pd.to_datetime(films["study_time"])
    assert not (set(films["subject_id"]) & study)

    patients = pd.read_csv(find_table(mimic_root, "patients"), usecols=["subject_id", "dod"], dtype=str)
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    films = films.merge(patients, on="subject_id", how="left")
    days = (films["dod"] - films["study_time"].dt.normalize()).dt.days
    films["label"] = (days.ge(0) & days.le(HORIZON_DAYS)).astype(int)

    admissions = pd.read_csv(
        find_table(mimic_root, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"],
        dtype=str,
    )
    for column in ("admittime", "dischtime", "deathtime"):
        admissions[column] = pd.to_datetime(admissions[column], errors="coerce")
    inside = films[["dicom_id", "subject_id", "study_time"]].merge(admissions, on="subject_id")
    cutoff = inside["dischtime"].where(inside["deathtime"].isna(), inside["deathtime"])
    inside = inside[(inside["study_time"] >= inside["admittime"]) & (inside["study_time"] < cutoff)]
    inside = inside.drop_duplicates("dicom_id", keep=False).set_index("dicom_id")
    films["in_hospital_death"] = films["dicom_id"].map(inside["hospital_expire_flag"]).astype(float)
    return films.reset_index(drop=True)


def count(args: argparse.Namespace) -> None:
    films = _external_films(Path(args.run_dir), Path(args.mimic_root))
    inpatient = films[films["in_hospital_death"].notna()]
    died = inpatient[inpatient["in_hospital_death"].eq(1)]
    print(f"non-cohort AP films: {len(films)}, patients {films['subject_id'].nunique()}")
    print(f"  taken during an admission: {len(inpatient)} films, "
          f"{inpatient['subject_id'].nunique()} patients, "
          f"{died['subject_id'].nunique()} patients who died in hospital")
    proxy = films[films["label"].eq(1)]
    print(f"  death within {HORIZON_DAYS} days of the film: {len(proxy)} films, "
          f"{proxy['subject_id'].nunique()} patients")


def cache(args: argparse.Namespace) -> None:
    run_dir, output = Path(args.run_dir), Path(args.output)
    films = _external_films(run_dir, Path(args.mimic_root))
    upstream.load_pipeline(None)
    images = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.images")
    root = Path(json.loads((run_dir / "dataset_spec.json").read_text(encoding="utf-8"))["image_root"])

    output.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.float16, shape=(len(films), 3, 224, 224)
    )
    for position, row in enumerate(films.itertuples(index=False)):
        array[position] = images.image_array(root / row.image_path, 224, 3).astype(np.float16)
        if (position + 1) % 100 == 0:
            print(f"decoded {position + 1}/{len(films)}", flush=True)
    array.flush()
    films.to_csv(output / "index.csv", index=False)
    print(f"cached {len(films)} films of {films['subject_id'].nunique()} patients in {output}")


def _load(output: Path):
    index = pd.read_csv(output / "index.csv", dtype={"subject_id": str})
    array = np.load(output / "images.npy", mmap_mode="r")
    labels = index["label"].to_numpy()
    subjects = index["subject_id"].to_numpy(dtype=object)
    patients = pd.DataFrame({"subject_id": subjects, "label": labels}).groupby(
        "subject_id", as_index=False)["label"].max()
    return index, array, labels, subjects, make_folds(patients, n_splits=5, seed=5212)


def _logistic(train_x, train_y, test_x, penalty: float) -> np.ndarray:
    x = torch.tensor(train_x, dtype=torch.float64)
    y = torch.tensor(train_y, dtype=torch.float64)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weight + bias, y)
        loss = loss + penalty * (weight ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(test_x, dtype=torch.float64) @ weight + bias).numpy()


def frozen(args: argparse.Namespace) -> None:
    import timm
    import torchxrayvision as xrv

    _, array, labels, subjects, folds = _load(Path(args.output))
    device = torch.device(args.device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device).eval()
    dense = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device).eval()

    cls, pooled, pathology = [], [], []
    with torch.inference_mode():
        for start in range(0, len(labels), 32):
            image = torch.from_numpy(np.asarray(array[start:start + 32], dtype=np.float32)).to(device)
            cls.append(vit.forward_features((image - mean) / std)[:, 0].float().cpu())
            gray = image.mean(dim=1, keepdim=True).clamp(0, 1) * 2048.0 - 1024.0
            pooled.append(torch.relu(dense.features(gray)).mean(dim=(2, 3)).float().cpu())
            pathology.append(dense(gray).float().cpu())
    representations = {
        "ImageNet ViT-B/16 CLS (768)": torch.cat(cls).numpy(),
        "CheXpert DenseNet121 pooled (1024)": torch.cat(pooled).numpy(),
        "CheXpert DenseNet121 pathology scores (18)": torch.cat(pathology).numpy(),
    }
    print(f"external AP films {len(labels)}, event films {int(labels.sum())}")
    for name, features in representations.items():
        scores = np.zeros(len(labels))
        for held in folds:
            test = np.isin(subjects, list(held))
            centre = features[~test].mean(0)
            scale = np.maximum(features[~test].std(0), 1e-6)
            scores[test] = _logistic(
                (features[~test] - centre) / scale, labels[~test],
                (features[test] - centre) / scale, penalty=0.05,
            )
        interval = bootstrap_interval(labels, scores, subjects, samples=2000)
        print(f"{name:44s} AUROC {auroc(labels, scores):.3f}  "
              f"95% CI [{interval['lower']:.3f}, {interval['upper']:.3f}]")


def finetune(args: argparse.Namespace) -> None:
    index, array, labels, subjects, folds = _load(Path(args.output))
    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        image_encoder=replace(cfg.image_encoder, unfreeze_last_blocks=2),
        optim=replace(cfg.optim, backbone_lr=1e-5),
    )
    device = torch.device(args.device)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    augment = build_train_transform(cfg.augmentation, 224)
    # A patient's films share one label, so they are weighted to sum to one.
    weights = (1.0 / index.groupby("subject_id")["label"].transform("size")).to_numpy(np.float32)

    def batch(rows, train: bool) -> torch.Tensor:
        images = []
        for row in rows:
            image = (torch.from_numpy(np.asarray(array[row], dtype=np.float32)) - mean) / std
            images.append(augment(image) if train and augment is not None else image)
        return torch.stack(images).to(device)

    held_out = np.zeros((args.epochs, len(labels)))
    for number, held in enumerate(folds):
        set_seed(cfg.seed)
        test_rows = np.flatnonzero(np.isin(subjects, list(held)))
        train_rows = np.flatnonzero(~np.isin(subjects, list(held)))
        model = build_model(cfg, image_size=224, channels=3).to(device)
        optimizer = _build_optimizer(model, cfg)
        scaler = torch.amp.GradScaler(device.type)
        generator = np.random.default_rng(cfg.seed)
        for epoch in range(args.epochs):
            model.train()
            order = generator.permutation(train_rows)
            for start in range(0, len(order), 16):
                rows = order[start:start + 16]
                with torch.amp.autocast(device.type):
                    logits = model({"image": batch(rows, True)})
                    target = torch.from_numpy(labels[rows]).float().to(device)
                    per_film = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    weight = torch.from_numpy(weights[rows]).to(device)
                    loss = (per_film * weight).sum() / weight.sum()
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            model.eval()
            with torch.inference_mode(), torch.amp.autocast(device.type):
                scores = [
                    torch.sigmoid(model({"image": batch(test_rows[s:s + 32], False)})).float().cpu()
                    for s in range(0, len(test_rows), 32)
                ]
            held_out[epoch, test_rows] = torch.cat(scores).numpy()
        print(f"fold {number + 1}/{len(folds)} done", flush=True)

    print("held-out AUROC after each epoch (the best one is an upper bound, not an estimate):")
    for epoch in range(args.epochs):
        print(f"  epoch {epoch + 1:2d}: {auroc(labels, held_out[epoch]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["count", "cache", "frozen", "finetune"])
    parser.add_argument("--run-dir", default="data/runs/icu_mortality_v1")
    parser.add_argument("--mimic-root", default="data/mimiciv")
    parser.add_argument("--output", default="data/pretrain/image_v1")
    parser.add_argument("--config", default="configs/icu/cxr_only.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"count": count, "cache": cache, "frozen": frozen, "finetune": finetune}[args.command](args)


if __name__ == "__main__":
    main()
