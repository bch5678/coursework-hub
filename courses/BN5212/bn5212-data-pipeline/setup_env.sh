#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo 'Ready. Run: PYTHON=.venv/bin/python bash run_pipeline.sh --config config/default.json'
