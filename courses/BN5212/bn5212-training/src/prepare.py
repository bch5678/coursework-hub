"""Prepare the original training inputs by reusing the project's data pipeline."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys

from . import upstream
from .config import FusionConfig, TrainingConfig, load_config
from .manifest import sha256

ITEM_MAP_PATH = Path(__file__).resolve().parents[1] / "configs" / "clinical_items.json"


def prepare_inputs(
    cfg: TrainingConfig,
    *,
    pipeline_config: str | Path | None = None,
    download_manifest: str | Path | None = None,
    mimic_root: str | Path | None = None,
    pretrain: bool = False,
    pretrain_epochs: int = 60,
    expected_index_sha256: str | None = None,
) -> tuple[TrainingConfig, dict]:
    from .cache import ImageCache, build_image_cache
    from .extract import extract_clinical_features, find_table

    if pretrain_epochs < 1:
        raise ValueError("pretrain_epochs must be positive")
    upstream.load_pipeline(cfg.data.data_pipeline_path)
    pipeline_settings = None
    if pipeline_config:
        settings_module = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.config")
        pipeline_settings = settings_module.load_config(pipeline_config)
    run_dir = Path(cfg.data.run_dir).resolve()
    if pipeline_settings:
        declared_output = Path(pipeline_settings["paths"]["output_dir"]).resolve()
        if declared_output != run_dir:
            raise ValueError(
                f"Training run_dir {run_dir} differs from pipeline output_dir {declared_output}; "
                "set --run-dir to the same output directory"
            )
    root = Path(mimic_root).resolve() if mimic_root else (
        Path(pipeline_settings["paths"]["mimic_iv_root"]) if pipeline_settings else None
    )
    stages = {}
    if download_manifest:
        downloader = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.download")
        print(f"[prepare] Processing authorized download manifest: {Path(download_manifest).resolve()}", flush=True)
        downloader.download_manifest(download_manifest)
        stages["downloads"] = "processed"
    if not (run_dir / "SUCCESS.json").is_file():
        if run_dir.exists():
            raise ValueError(f"Incomplete dataset directory {run_dir}; refusing to overwrite it")
        if pipeline_settings is None:
            raise ValueError(
                "Building the dataset needs --pipeline-config pointing to a configured raw-data "
                "pipeline JSON. Project .7z bundles contain derived tables/pooled vectors, not "
                "the raw radiographs needed to reconstruct patch features."
            )
        image_root = Path(pipeline_settings["paths"]["image_root"])
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"Raw CXR image_root is missing: {image_root}. Set paths.image_root in the pipeline config "
                "to the original radiographs; pooled vectors in the project .7z bundles cannot recover them."
            )
        for name in ("admissions", "patients"):
            find_table(root, name)
        if (pipeline_settings["cohort"].get("unit", "admission") == "icu_stay"
                or pipeline_settings["label"]["kind"] == "icu_transfer"):
            find_table(root, "icustays")
        print(f"[prepare] Building frozen dataset: {run_dir}", flush=True)
        builder = importlib.import_module(f"{upstream.PACKAGE_ALIAS}.data.pipeline")
        builder.build(pipeline_config)
        stages["dataset"] = "built"
    else:
        print(f"[prepare] Reusing frozen dataset: {run_dir}", flush=True)
        stages["dataset"] = "reused"

    success = json.loads((run_dir / "SUCCESS.json").read_text(encoding="utf-8"))
    if success.get("status") != "complete" or not success.get("artifacts_sha256"):
        raise ValueError(f"Invalid dataset completion manifest: {run_dir / 'SUCCESS.json'}")
    for filename, checksum in success["artifacts_sha256"].items():
        if sha256(run_dir / filename) != checksum:
            raise ValueError(f"Changed frozen dataset artifact: {filename}")
    probe = upstream.dataset_module(cfg.data.data_pipeline_path).MimicCXRDataset(run_dir, "train")
    index_hash = probe.spec["index_sha256"]
    if cfg.data.task is not None and probe.spec.get("task") != cfg.data.task:
        raise ValueError(f"Expected dataset task {cfg.data.task!r}, got {probe.spec.get('task')!r}")
    if expected_index_sha256 and index_hash != expected_index_sha256:
        raise ValueError(f"Baseline index hash mismatch: expected {expected_index_sha256}, got {index_hash}")
    snapshot_path = run_dir / "config.snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else pipeline_settings
    if snapshot is None:
        raise ValueError("Clinical preparation needs the dataset's config.snapshot.json or --pipeline-config")
    unit = snapshot["cohort"].get("unit", "admission")
    hours = snapshot["alignment"].get("icu_observation_hours", 48) if unit == "icu_stay" else cfg.data.clinical_timesteps
    transfer = probe.spec.get("task") == "icu_transfer"
    if transfer:
        if cfg.data.clinical_window_anchor != "cxr" or cfg.data.clinical_window_hours != 48:
            raise ValueError("ICU transfer needs a CXR-anchored 48-hour clinical window")
        if cfg.clinical_encoder.pretrained or pretrain:
            raise ValueError("ICU transfer does not reuse mortality pretraining; use the transfer config without --pretrain-clinical")
        hours = cfg.data.clinical_window_hours
    data = replace(cfg.data, run_dir=str(run_dir))

    def require_clinical_root() -> Path:
        if root is None or not root.is_dir():
            raise FileNotFoundError("Missing MIMIC-IV tables; supply --mimic-root or a valid pipeline config")
        return root

    if cfg.uses_clinical and data.clinical_provider == "table":
        source = Path(data.clinical_source) if data.clinical_source else run_dir.parent / f"{run_dir.name}_clinical.csv"
        sidecar = source.with_suffix(source.suffix + ".preparation.json")
        metadata = {"index_sha256": index_hash, "cohort_unit": unit, "max_hours": hours}
        if transfer:
            metadata.update(task="icu_transfer", clinical_window_anchor="cxr", feature_schema="16labs_age_sex/1")
        if source.is_file():
            if transfer and not sidecar.is_file():
                raise ValueError(f"Transfer clinical table needs its provenance sidecar: {sidecar}")
            if sidecar.is_file() and json.loads(sidecar.read_text()) != metadata:
                raise ValueError(f"Clinical table was prepared for different data/settings: {source}")
            stages["clinical"] = "reused"
        else:
            print(f"[prepare] Extracting clinical table: {source}", flush=True)
            if transfer:
                from .transfer import extract_transfer_features

                extract_transfer_features(run_dir, require_clinical_root(), source, hours=int(hours))
            else:
                extract_clinical_features(
                    run_dir=run_dir, mimic_root=require_clinical_root(), output_path=source,
                    item_map_path=ITEM_MAP_PATH, unit=unit, max_hours=hours,
                )
            sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            stages["clinical"] = "extracted"
        data = replace(data, clinical_source=str(source.resolve()))

    if cfg.uses_clinical and cfg.clinical_encoder.pretrained:
        checkpoint = Path(cfg.clinical_encoder.pretrained)
        if checkpoint.is_file():
            stages["clinical_pretraining"] = "reused"
        elif not pretrain:
            raise FileNotFoundError(
                f"Clinical checkpoint is missing: {checkpoint}. Use --pretrain-clinical to reproduce "
                "external pretraining, or provide the original --clinical-checkpoint."
            )
        else:
            from .pretrain import build_external_cohort, pretrain_clinical

            if unit != "icu_stay":
                raise ValueError("Automatic external clinical pretraining requires the original ICU-stay protocol")
            external = checkpoint.parent / "external_cohort"
            cohort_manifest = external / "cohort.json"
            if external.exists():
                metadata = json.loads(cohort_manifest.read_text()) if cohort_manifest.is_file() else {}
                if (metadata.get("study_index_sha256") != index_hash
                        or metadata.get("seed") != cfg.seed
                        or metadata.get("observation_hours") != hours):
                    raise ValueError(f"External pretraining cohort does not match this study: {external}")
            else:
                build_external_cohort(run_dir, require_clinical_root(), external, hours=hours, seed=cfg.seed)
            external_features = external / "clinical_features.csv"
            if not external_features.is_file():
                extract_clinical_features(
                    run_dir=external, mimic_root=require_clinical_root(), output_path=external_features,
                    item_map_path=ITEM_MAP_PATH, unit="icu_stay", max_hours=hours,
                )
            clinical_cfg = replace(
                cfg, data=data, modalities=("clinical",),
                fusion=FusionConfig(name="clinical_only", embed_dim=cfg.clinical_encoder.embed_dim, pooling="mean"),
            )
            print(f"[prepare] Pretraining clinical encoder outside study patients: {checkpoint}", flush=True)
            pretrain_clinical(external, clinical_cfg, checkpoint, source=external_features,
                             epochs=pretrain_epochs, device=cfg.device)
            stages["clinical_pretraining"] = "trained"

    if cfg.uses_image:
        cache_path = Path(data.image_cache) if data.image_cache else run_dir.parent / f"{run_dir.name}_images.json"
        if cache_path.is_file():
            cache = ImageCache(cache_path, probe.spec)
            if not cache.path.with_suffix(".npy").is_file():
                raise FileNotFoundError(f"Image cache array is missing: {cache.path.with_suffix('.npy')}")
            stages["image_cache"] = "reused"
        else:
            print(f"[prepare] Caching decoded images: {cache_path}", flush=True)
            build_image_cache(run_dir, cache_path, data_pipeline_path=cfg.data.data_pipeline_path)
            stages["image_cache"] = "built"
        data = replace(data, image_cache=str(cache_path.resolve()))
    return replace(cfg, data=data), {"index_sha256": index_hash, "cohort_unit": unit, "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pipeline-config")
    parser.add_argument("--download-manifest")
    parser.add_argument("--mimic-root")
    parser.add_argument("--pretrain-clinical", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            parser.error("--set expects SECTION.KEY=VALUE")
        try:
            overrides[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key.strip()] = value
    cfg, report = prepare_inputs(
        load_config(args.config, overrides), pipeline_config=args.pipeline_config,
        download_manifest=args.download_manifest,
        mimic_root=args.mimic_root, pretrain=args.pretrain_clinical,
        pretrain_epochs=args.pretrain_epochs, expected_index_sha256=args.expected_index_sha256,
    )
    destination = Path(args.output_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
    destination.with_suffix(".preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
