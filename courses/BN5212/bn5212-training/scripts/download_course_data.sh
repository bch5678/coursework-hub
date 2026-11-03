#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# Install this project with pip install -e . so bn5212_training maps to src/.
mkdir -p "$PROJECT_DIR/data/download_logs"
"${PYTHON_BIN:-python}" -u -m bn5212_training.dropbox_download "$@" \
  2>&1 | tee -a "$PROJECT_DIR/data/download_logs/dropbox.log"
