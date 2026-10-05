"""Pretrain the clinical encoder on ICU stays outside the study cohort.

Every study patient (train, val and test) is excluded by subject_id. External
stays pass the cohort's eligibility rule, and every choice about the encoder is
made on the external validation split, never on the cohort.

    python -m bn5212_training.pretrain cohort --run-dir <run> --mimic-root <mimic> --output <dir>
    python -m bn5212_training.pretrain fit    --cohort <dir> --config configs/icu/clinical_only.json
    python -m bn5212_training.pretrain score  --checkpoint <ckpt> --config <config> --output <out>

Between cohort and fit, extract the features with bn5212-extract-clinical
(--run-dir <dir> --cohort-unit icu_stay --output <dir>/clinical_features.csv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .clinical import ClinicalNormalizer, TableClinicalProvider, build_provider
from .config import ClinicalEncoderConfig, TrainingConfig, config_from_dict, load_config
from .extract import find_table
from .manifest import sha256
from .metrics import selection_metrics

EXTERNAL_SPLITS = ("pretrain_train", "pretrain_val")
CHECKPOINT_FORMAT = "bn5212-clinical-pretrain/1"
# What must agree between the pretrained encoder and the encoder it is loaded into.
_ENCODER_SHAPE = ("name", "tokenization", "embed_dim", "hidden_dim", "missing_indicator")


def _unit_interval(seed: int, key: str) -> float:
    """Deterministic value in [0, 1) from the seed and a patient id."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _last_covered_subject(root: Path) -> int | None:
    """subject_id of the last complete chartevents row, or None.

    chartevents is in subject_id order, so a truncated file holds no vital signs
    for the patients after the cut; they must not enter pretraining.
    """
    try:
        path = find_table(root, "chartevents")
    except FileNotFoundError:
        return None
    if path.suffix != ".csv":
        return None  # a compressed file cannot be read from its end
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 8192))
        pieces = stream.read().decode("utf-8", errors="replace").split("\n")
    # The first piece may start mid-row and the last is either empty (the file
    # ends cleanly) or a row that was cut off.
    complete = [piece for piece in pieces[1:-1] if piece.strip()]
    if not complete:
        return None
    subject = complete[-1].split(",")[0]
    return int(subject) if subject.isdigit() else None


def build_external_cohort(
    run_dir: str | Path,
    mimic_root: str | Path,
    output_dir: str | Path,
    *,
    hours: float = 48.0,
    min_age: int = 18,
    max_stays: int | None = 20_000,
    val_fraction: float = 0.15,
    seed: int = 5212,
) -> dict[str, Any]:
    """Select ICU stays of patients who are not in the study cohort.

    Writes index.csv in the shape the clinical extraction reads, so the same
    extraction code and item map produce the features for both cohorts.
    """
    run_dir, root, output_dir = Path(run_dir), Path(mimic_root), Path(output_dir)
    study = pd.read_csv(run_dir / "index.csv", dtype={"subject_id": str})
    excluded = set(study["subject_id"])

    stays = pd.read_csv(
        find_table(root, "icustays"),
        usecols=["subject_id", "hadm_id", "stay_id", "intime"],
        dtype=str,
    )
    admissions = pd.read_csv(
        find_table(root, "admissions"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "hospital_expire_flag"],
        dtype=str,
    )
    patients = pd.read_csv(
        find_table(root, "patients"),
        usecols=["subject_id", "anchor_age", "anchor_year"],
        dtype=str,
    )
    frame = stays.merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
    frame = frame.merge(patients, on="subject_id", validate="many_to_one")
    for column in ("intime", "admittime", "dischtime", "deathtime"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    flow = [{"stage": "icu_stays_with_admission", "stays": len(frame)}]

    def keep(mask: pd.Series, stage: str) -> None:
        nonlocal frame
        frame = frame[mask.to_numpy()]
        flow.append({"stage": stage, "stays": len(frame)})

    keep(~frame["subject_id"].isin(excluded), "patient_not_in_study_cohort")
    keep(frame["intime"].notna() & frame["admittime"].notna() & frame["dischtime"].notna(),
         "valid_timestamps")
    age = (pd.to_numeric(frame["anchor_age"], errors="coerce")
           + frame["admittime"].dt.year - pd.to_numeric(frame["anchor_year"], errors="coerce"))
    keep(age.ge(min_age), "adult")

    # The same consistency rule the pipeline applies to the study cohort.
    expire = pd.to_numeric(frame["hospital_expire_flag"], errors="coerce")
    death = frame["deathtime"]
    consistent = (
        expire.isin([0, 1])
        & (~expire.eq(0) | death.isna())
        & (death.isna() | ((death >= frame["admittime"]) & (death <= frame["dischtime"])))
    )
    keep(consistent, "valid_mortality_and_death_time")

    # Alive and still admitted at the prediction time: a record that stops early
    # would otherwise leak the outcome through its own absence.
    prediction_time = frame["intime"] + pd.to_timedelta(float(hours), unit="h")
    cutoff = frame["dischtime"].where(frame["deathtime"].isna(), frame["deathtime"])
    keep(prediction_time < cutoff, "outcome_undetermined_at_prediction_time")

    frame = frame.assign(
        order=frame["subject_id"].map(lambda value: _unit_interval(seed, value))
    ).sort_values(["order", "subject_id", "intime"])
    if max_stays is not None and len(frame) > max_stays:
        # Whole patients, in hash order, until the budget is met.
        boundary = frame["order"].iloc[max_stays - 1]
        keep(frame["order"] <= boundary, "sampled_whole_patients")

    # Applied after sampling, so cutting the file short removes stays from the
    # sample instead of silently drawing a different one.
    covered = _last_covered_subject(root)
    subject_number = pd.to_numeric(frame["subject_id"])
    truncated_at = None
    if covered is not None and len(frame) and covered < int(pd.to_numeric(stays["subject_id"]).max()):
        truncated_at = covered
        keep(subject_number < covered, "patient_covered_by_chartevents")

    assert not (set(frame["subject_id"]) & excluded)
    prediction_time = frame["intime"] + pd.to_timedelta(float(hours), unit="h")
    in_val = frame["subject_id"].map(lambda value: _unit_interval(seed + 1, value)) < val_fraction
    index = pd.DataFrame({
        "subject_id": frame["subject_id"],
        "hadm_id": frame["hadm_id"],
        "stay_id": frame["stay_id"],
        "admittime": frame["admittime"],
        # The extraction reads these two names; here they are the prediction time.
        "study_time": prediction_time,
        "hours_since_admission": (prediction_time - frame["admittime"]).dt.total_seconds() / 3600,
        "label": pd.to_numeric(frame["hospital_expire_flag"]).astype(int),
        "split": np.where(in_val, EXTERNAL_SPLITS[1], EXTERNAL_SPLITS[0]),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "index.csv", index=False, lineterminator="\n")
    summary = {
        "purpose": "clinical encoder pretraining; no patient of the study cohort is included",
        "study_run_dir": str(run_dir.resolve()),
        "study_index_sha256": sha256(run_dir / "index.csv"),
        "study_patients_excluded": len(excluded),
        "observation_hours": float(hours),
        "seed": seed,
        # Set when chartevents stops before the last ICU patient: only patients
        # below this subject_id have their vital signs in the file.
        "chartevents_truncated_at_subject_id": truncated_at,
        "flow": flow,
        "stays": int(len(index)),
        "patients": int(index["subject_id"].nunique()),
        "deaths": int(index["label"].sum()),
        "prevalence": float(index["label"].mean()),
        "per_split": {
            name: {
                "stays": int(part["label"].size),
                "patients": int(part["subject_id"].nunique()),
                "deaths": int(part["label"].sum()),
            }
            for name, part in index.groupby("split")
        },
    }
    (output_dir / "cohort.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _external_arrays(
    cohort_dir: Path, source: str | Path, timesteps: int
) -> tuple[pd.DataFrame, TableClinicalProvider, np.ndarray, np.ndarray]:
    """Dense [N, K, T] values and mask for every external stay that has data."""
    index = pd.read_csv(
        cohort_dir / "index.csv", dtype={"subject_id": str, "hadm_id": str, "stay_id": str}
    )
    provider = TableClinicalProvider(source, num_timesteps=timesteps)
    rows = index.to_dict(orient="records")
    # A stay with no extracted row carries no physiology to learn from, and the
    # provider would stand in a placeholder observation for it.
    present = np.array([provider.has_observations(row) for row in rows])
    index = index[present].reset_index(drop=True)
    rows = [row for row, keep in zip(rows, present) if keep]

    shape = (len(rows), len(provider.variable_names), provider.num_timesteps)
    values = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    for position, row in enumerate(rows):
        values[position], mask[position] = provider.features(row)
    return index, provider, values, mask


@torch.inference_mode()
def _external_scores(model, values: torch.Tensor, mask: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(values), batch_size):
        batch = slice(start, start + batch_size)
        logits = model({"clinical": values[batch], "clinical_mask": mask[batch]})
        scores.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(scores).numpy().astype(np.float64)


def pretrain_clinical(
    cohort_dir: str | Path,
    cfg: TrainingConfig,
    output_path: str | Path,
    *,
    source: str | Path | None = None,
    epochs: int = 60,
    batch_size: int = 256,
    patience: int = 8,
    device: str = "auto",
) -> dict[str, Any]:
    """Fit the clinical-only model on the external cohort and save its encoder.

    cfg is the study's clinical-only config, so the encoder is the module the
    study experiments instantiate. Early stopping reads external validation only.
    """
    from .model import build_model
    from .predict import resolve_device
    from .trainer import set_seed

    if cfg.modalities != ("clinical",):
        raise ValueError("Pretraining needs a clinical-only config")
    if cfg.head.num_outputs != 1:
        raise ValueError("Pretraining supports the single-logit head only")
    cohort_dir = Path(cohort_dir)
    source = Path(source) if source else cohort_dir / "clinical_features.csv"
    cfg = replace(cfg, clinical_encoder=replace(cfg.clinical_encoder, pretrained=None, freeze=False))

    set_seed(cfg.seed)
    target = resolve_device(device)
    index, provider, values, mask = _external_arrays(cohort_dir, source, cfg.data.clinical_timesteps)
    fitting = index["split"].eq(EXTERNAL_SPLITS[0]).to_numpy()
    rows = index.to_dict(orient="records")
    normalizer = ClinicalNormalizer.fit(
        provider, [row for row, keep in zip(rows, fitting) if keep], split="train"
    )
    normalizer.fitted_on = "external pretraining cohort (train part)"
    scaled = torch.from_numpy(normalizer.apply(values, mask))
    observed = torch.from_numpy(mask)
    labels = torch.from_numpy(index["label"].to_numpy(dtype=np.float32))

    train_rows = torch.from_numpy(np.flatnonzero(fitting))
    val_rows = torch.from_numpy(np.flatnonzero(~fitting))
    x_train, m_train, y_train = (t[train_rows].to(target) for t in (scaled, observed, labels))
    x_val, m_val = scaled[val_rows].to(target), observed[val_rows].to(target)
    y_val = labels[val_rows].numpy().astype(int)

    model = build_model(
        cfg, num_variables=len(provider.variable_names), num_timesteps=provider.num_timesteps
    ).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    generator = torch.Generator().manual_seed(cfg.seed)

    history: list[dict[str, Any]] = []
    best_loss, best_state, best_epoch, left = None, None, 0, patience
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(target)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            logits = model({"clinical": x_train[batch], "clinical_mask": m_train[batch]})
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            total += float(loss) * len(batch)

        scores = _external_scores(model, x_val, m_val, batch_size)
        clipped = np.clip(scores, 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(y_val * np.log(clipped) + (1 - y_val) * np.log(1 - clipped)))
        metrics = selection_metrics(y_val, scores)
        history.append({
            "epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss,
            "val_auroc": metrics["auroc"], "val_auprc": metrics["auprc"],
        })
        print(f"[pretrain {cfg.clinical_encoder.name}] epoch {epoch}/{epochs} "
              f"train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_auroc={metrics['auroc']:.4f}", flush=True)
        if best_loss is None or val_loss < best_loss:
            best_loss, best_epoch, left = val_loss, epoch, patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                break

    model.load_state_dict(best_state)
    scores = _external_scores(model, x_val, m_val, batch_size)
    external = {
        **selection_metrics(y_val, scores),
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "fitting_stays": int(fitting.sum()),
        "fitting_deaths": int(index.loc[fitting, "label"].sum()),
        "stays_without_data_dropped": int(
            len(pd.read_csv(cohort_dir / "index.csv", usecols=["stay_id"])) - len(index)
        ),
    }
    cohort_summary = json.loads((cohort_dir / "cohort.json").read_text(encoding="utf-8"))
    payload = {
        "format": CHECKPOINT_FORMAT,
        "encoder": asdict(cfg.clinical_encoder),
        "encoder_state": {k: v.cpu() for k, v in model.clinical_encoder.state_dict().items()},
        # The whole clinical-only model, so the external model can be scored on
        # the study cohort without fitting anything there.
        "model_config": cfg.to_dict(),
        "model_state": best_state,
        "geometry": {
            "num_variables": len(provider.variable_names),
            "num_timesteps": provider.num_timesteps,
        },
        "clinical_normalizer": normalizer.to_dict(),
        "external_validation": external,
        "history": history,
        "cohort": cohort_summary,
        "clinical_source_sha256": sha256(source),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report = {
        "checkpoint": str(output_path),
        "encoder": cfg.clinical_encoder.name,
        "external_validation": external,
        "study_patients_excluded": cohort_summary["study_patients_excluded"],
        "study_index_sha256": cohort_summary["study_index_sha256"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps({**report, "history": history}, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretrained clinical encoder not found: {path}. Build it with "
            "'python -m bn5212_training.pretrain fit'."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path.name} is not a {CHECKPOINT_FORMAT} file")
    return payload


def load_pretrained_encoder(encoder: torch.nn.Module, cfg: ClinicalEncoderConfig) -> None:
    """Copy the pretrained weights into an encoder of the same shape."""
    payload = _read_checkpoint(cfg.pretrained)
    stored, wanted = payload["encoder"], asdict(cfg)
    differing = [key for key in _ENCODER_SHAPE if stored[key] != wanted[key]]
    if differing:
        raise ValueError(
            f"{Path(cfg.pretrained).name} holds a different clinical encoder: "
            + ", ".join(f"{key}={stored[key]!r} (config has {wanted[key]!r})" for key in differing)
        )
    encoder.load_state_dict(payload["encoder_state"])


def load_pretrained_normalizer(path: str | Path) -> ClinicalNormalizer:
    """The statistics the pretrained encoder's inputs were scaled with."""
    return ClinicalNormalizer.from_dict(_read_checkpoint(path)["clinical_normalizer"])


def score_study_cohort(
    checkpoint: str | Path,
    cfg: TrainingConfig,
    output_dir: str | Path,
    *,
    n_splits: int = 5,
    bootstrap_samples: int = 2000,
    device: str = "auto",
) -> dict[str, Any]:
    """Apply the external clinical model to the study cohort without fitting on it.

    Train+val rows are reported in the cross-validation folds; val and test
    predictions are written for the benchmark project to score.
    """
    from .crossval import _patient_table, fold_summary, make_folds, write_pooled_report
    from .data import TrainingDataset, make_loader
    from .model import build_model
    from .predict import resolve_device, run_inference, write_predictions

    payload = _read_checkpoint(checkpoint)
    target = resolve_device(device)
    model_cfg = config_from_dict(payload["model_config"])
    model = build_model(model_cfg, **payload["geometry"], load_pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(target)

    normalizer = ClinicalNormalizer.from_dict(payload["clinical_normalizer"])
    provider = build_provider(cfg.data, variable_names=normalizer.variable_names)

    def predict(splits) -> dict[str, np.ndarray]:
        dataset = TrainingDataset(
            cfg.data.run_dir, splits, data_pipeline_path=cfg.data.data_pipeline_path,
            provider=provider, normalizer=normalizer, load_image=False,
        )
        loader = make_loader(dataset, batch_size=64, num_workers=0, shuffle=False, seed=cfg.seed)
        return run_inference(model, loader, target)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scored = predict(("train", "val"))
    pooled = pd.DataFrame({
        "sample_id": scored["sample_id"], "subject_id": scored["subject_id"],
        "hadm_id": scored["hadm_id"], "label": scored["label"], "y_score": scored["y_score"],
    })
    folds = make_folds(_patient_table(cfg), n_splits=n_splits, seed=cfg.seed, stratify=True)
    fold_of = {str(subject): index for index, fold in enumerate(folds) for subject in fold}
    pooled["fold"] = pooled["subject_id"].astype(str).map(fold_of)
    per_fold = [
        fold_summary(pooled[pooled["fold"].eq(index)], index, len(fold))
        for index, fold in enumerate(folds)
    ]
    report_cfg = replace(cfg, experiment=f"{cfg.experiment}_external")
    pooled_metrics = write_pooled_report(
        root, report_cfg, pooled, per_fold,
        run_id=root.name, n_splits=n_splits, inner_splits=None,
        bootstrap_samples=bootstrap_samples,
    )
    for split in ("val", "test"):
        write_predictions(predict(split), root / f"predictions_{split}.csv")
    (root / "run_manifest.json").write_text(
        json.dumps({
            "evaluation": "external clinical model applied to the study cohort; "
                          "no parameter was fitted on study patients",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "external_validation": payload["external_validation"],
            "study_patients_excluded_from_pretraining": payload["cohort"]["study_patients_excluded"],
            "pooled_metrics": pooled_metrics,
            "per_fold": per_fold,
        }, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {"run_dir": str(root), "pooled": pooled_metrics, "per_fold": per_fold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    cohort = commands.add_parser("cohort", help="Select external ICU stays for pretraining")
    cohort.add_argument("--run-dir", required=True, help="Study run; its patients are excluded")
    cohort.add_argument("--mimic-root", required=True)
    cohort.add_argument("--output", required=True)
    cohort.add_argument("--max-stays", type=int, default=20_000)
    cohort.add_argument("--val-fraction", type=float, default=0.15)
    cohort.add_argument("--seed", type=int, default=5212)

    fit = commands.add_parser("fit", help="Pretrain the clinical-only model on the external cohort")
    fit.add_argument("--cohort", required=True, help="Directory written by the cohort command")
    fit.add_argument("--config", required=True, help="The study's clinical-only config")
    fit.add_argument("--clinical-source", help="Default <cohort>/clinical_features.csv")
    fit.add_argument("--output", help="Checkpoint path; default <cohort>/clinical_encoder.pt")
    fit.add_argument("--encoder", help="Override clinical_encoder.name")
    fit.add_argument("--epochs", type=int, default=60)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--patience", type=int, default=8)
    fit.add_argument("--device", default="auto")

    score = commands.add_parser("score", help="Score the study cohort with the external model")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--config", required=True, help="The study's clinical-only config")
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="auto")

    args = parser.parse_args()
    if args.command == "cohort":
        summary = build_external_cohort(
            args.run_dir, args.mimic_root, args.output,
            max_stays=args.max_stays, val_fraction=args.val_fraction, seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "fit":
        overrides = {"clinical_encoder.name": args.encoder} if args.encoder else None
        cfg = load_config(args.config, overrides)
        output = args.output or str(Path(args.cohort) / "clinical_encoder.pt")
        report = pretrain_clinical(
            args.cohort, cfg, output, source=args.clinical_source, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience, device=args.device,
        )
        print(json.dumps(report, indent=2, default=str))
    else:
        result = score_study_cohort(
            args.checkpoint, load_config(args.config), args.output, device=args.device
        )
        pooled, interval = result["pooled"], result["pooled"]["auroc_ci"]
        print(f"study train+val n={pooled['n']} positives={pooled['n_positive']}")
        print(f"AUROC {pooled['auroc']:.3f}  95% CI [{interval['lower']:.3f}, {interval['upper']:.3f}]")
        print(f"AUPRC {pooled['auprc']:.3f}   Brier {pooled['brier']:.3f}")
        print(f"Run directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
