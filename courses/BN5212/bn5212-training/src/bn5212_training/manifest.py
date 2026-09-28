"""Run provenance.

The benchmark project requires every model version to declare its git commit,
dataset index hash, checkpoint SHA-256, seed and software environment. Recording
them at training time is what makes a result reproducible later, so the manifest
is written by the trainer rather than reconstructed by hand afterwards.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: str | Path | None = None) -> str | None:
    """Current commit, or None outside a git checkout."""
    directory = Path(repo) if repo else Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_is_dirty(repo: str | Path | None = None) -> bool | None:
    directory = Path(repo) if repo else Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():  # pragma: no cover - depends on the machine
            payload["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        payload["torch"] = None
    try:
        import numpy

        payload["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    return payload


def dataset_provenance(run_dir: str | Path) -> dict[str, Any]:
    """Identify the frozen dataset run this model was trained against."""
    run_dir = Path(run_dir).resolve()
    payload: dict[str, Any] = {"run_dir": str(run_dir)}
    spec_path = run_dir / "dataset_spec.json"
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "schema_version": spec.get("schema_version"),
                "task": spec.get("task"),
                "index_sha256": spec.get("index_sha256"),
                "split_seed": spec.get("split_seed"),
                "split_unit": spec.get("split_unit"),
                "dataset_versions": spec.get("dataset_versions"),
                "loader": spec.get("loader"),
            }
        )
    index_path = run_dir / "index.csv"
    if index_path.is_file():
        payload["index_sha256_recomputed"] = sha256(index_path)
    return payload


def build_manifest(
    *,
    config: Mapping[str, Any],
    config_fingerprint: str,
    run_dir: str | Path,
    seed: int,
    experiment: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "seed": seed,
        "config_fingerprint": config_fingerprint,
        "config": dict(config),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "dataset": dataset_provenance(run_dir),
        "environment": environment(),
        **(dict(extra) if extra else {}),
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path
