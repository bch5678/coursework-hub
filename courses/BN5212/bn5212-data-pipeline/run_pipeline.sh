#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# PYTHON can point to a server environment with the desired CPU/CUDA PyTorch.
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" run_pipeline.py "$@" --test-loader
