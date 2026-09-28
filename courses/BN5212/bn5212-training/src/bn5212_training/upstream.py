"""Import the frozen data pipeline without installing it or shadowing `src`.

The sibling project bn5212-data-pipeline exposes its code as a top-level `src`
package, which would collide with this project's own `src` directory. We load it
under the private name `bn5212_pipeline` instead, so relative imports inside the
pipeline keep resolving while nothing global is renamed.
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


def load_pipeline(explicit: str | os.PathLike[str] | None = None) -> ModuleType:
    """Load and cache the pipeline package under the PACKAGE_ALIAS name."""
    if PACKAGE_ALIAS in sys.modules:
        return sys.modules[PACKAGE_ALIAS]
    src_dir = data_pipeline_root(explicit) / "src"
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
