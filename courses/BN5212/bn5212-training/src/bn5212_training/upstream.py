"""Import the frozen data pipeline without installing it.

The pipeline exposes a top-level `src` package that would collide with this
project's own `src`, so it is loaded under the private name `bn5212_pipeline`.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

PACKAGE_ALIAS = "bn5212_pipeline"
ENV_VAR = "BN5212_DATA_PIPELINE"
DEFAULT_SIBLING = "bn5212-data-pipeline"


def _candidate_roots(explicit: str | os.PathLike[str] | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        candidates.append(Path(from_env))
    # courses/BN5212/bn5212-training/src/bn5212_training/upstream.py -> courses/BN5212
    course_dir = Path(__file__).resolve().parents[3]
    candidates.append(course_dir / DEFAULT_SIBLING)
    return candidates


def data_pipeline_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the pipeline project root that contains src/data/dataset.py."""
    tried = []
    for candidate in _candidate_roots(explicit):
        root = candidate.expanduser().resolve()
        tried.append(str(root))
        if (root / "src" / "data" / "dataset.py").is_file():
            return root
    raise FileNotFoundError(
        "Could not locate the bn5212-data-pipeline project. Set the "
        f"{ENV_VAR} environment variable or pass data_pipeline_path in the "
        f"training config. Tried: {tried}"
    )


class _PipelineFinder:
    """Makes `bn5212_pipeline` importable by the normal machinery.

    A DataLoader worker on Windows starts by spawn and must import the pipeline
    itself, so a sys.modules entry alone is not enough.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != PACKAGE_ALIAS:
            return None  # submodules resolve through the package search locations
        try:
            src_dir = data_pipeline_root() / "src"
        except FileNotFoundError:
            return None
        return importlib.util.spec_from_file_location(
            PACKAGE_ALIAS,
            src_dir / "__init__.py",
            submodule_search_locations=[str(src_dir)],
        )


def _install_finder() -> None:
    if not any(isinstance(finder, _PipelineFinder) for finder in sys.meta_path):
        sys.meta_path.append(_PipelineFinder())


# Installed at import time so that a spawned worker, which imports this module
# while unpickling the dataset, can resolve the pipeline package too.
_install_finder()


def load_pipeline(explicit: str | os.PathLike[str] | None = None) -> ModuleType:
    """Load and cache the pipeline package under the PACKAGE_ALIAS name."""
    if PACKAGE_ALIAS in sys.modules:
        return sys.modules[PACKAGE_ALIAS]
    src_dir = data_pipeline_root(explicit) / "src"
    # Record the resolved root so spawned workers find the same checkout even when
    # they are started from a different working directory.
    os.environ.setdefault(ENV_VAR, str(src_dir.parent))
    spec = importlib.util.spec_from_file_location(
        PACKAGE_ALIAS,
        src_dir / "__init__.py",
        submodule_search_locations=[str(src_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build an import spec for {src_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_ALIAS] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(PACKAGE_ALIAS, None)
        raise
    return module


def dataset_module(explicit: str | os.PathLike[str] | None = None) -> ModuleType:
    """Return the pipeline's dataset module (MimicCXRDataset, make_dataloader)."""
    load_pipeline(explicit)
    return importlib.import_module(f"{PACKAGE_ALIAS}.data.dataset")
