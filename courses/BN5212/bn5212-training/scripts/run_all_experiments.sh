#!/usr/bin/env bash
# Run every experiment against one frozen dataset run, then build the comparison table.
#
#   ./scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh --preset real \
#       --run-dir /srv/derived/bn5212/mortality_v1 \
#       --clinical-source /srv/derived/bn5212/clinical_features.csv.gz --device cuda
#
# The synthetic preset produces no results, only proof that the code path works.
set -euo pipefail

PRESET="synthetic"
RUN_DIR=""
CLINICAL_SOURCE=""
DEVICE=""
RUN_ID=""
EXPERIMENTS="clinical_only cxr_only concat_fusion metra_joint cross_attention"

while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --clinical-source) CLINICAL_SOURCE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
TRAIN=".venv/bin/bn5212-train"
SUMMARIZE=".venv/bin/bn5212-summarize"
[ -x "$TRAIN" ] || { echo "bn5212-train not found. Create .venv and pip install -e '.[test]' first." >&2; exit 1; }

CONFIG_DIR="configs"
[ "$PRESET" = "synthetic" ] && CONFIG_DIR="configs/synthetic"
[ -n "$RUN_ID" ] || RUN_ID="$(date -u +%Y%m%d-%H%M%S)"

if [ "$PRESET" = "synthetic" ] && [ -z "$RUN_DIR" ]; then
  RUN_DIR="../bn5212-data-pipeline/demo/png/processed"
  [ -f "$RUN_DIR/SUCCESS.json" ] || {
    echo "Synthetic fixture missing. See README, 'Run training locally', step 1." >&2; exit 1; }
fi
[ -n "$RUN_DIR" ] || { echo "--run-dir is required for the real preset" >&2; exit 1; }

echo "preset=$PRESET  run_dir=$RUN_DIR  run_id=$RUN_ID"
echo

COMPLETED=""
for EXPERIMENT in $EXPERIMENTS; do
  CONFIG="$CONFIG_DIR/$EXPERIMENT.json"
  [ -f "$CONFIG" ] || { echo "skip $EXPERIMENT (no $CONFIG)"; continue; }

  set -- --config "$CONFIG" --run-dir "$RUN_DIR" --run-id "$RUN_ID"
  [ -n "$DEVICE" ] && set -- "$@" --device "$DEVICE"
  [ -n "$CLINICAL_SOURCE" ] && set -- "$@" \
    --set data.clinical_provider=table --set "data.clinical_source=$CLINICAL_SOURCE"

  echo "=== $EXPERIMENT ==="
  "$TRAIN" "$@"
  COMPLETED="$COMPLETED outputs/$EXPERIMENT/$RUN_ID"
  echo
done

if [ -n "$COMPLETED" ]; then
  # shellcheck disable=SC2086
  "$SUMMARIZE" $COMPLETED --output "outputs/comparison-$RUN_ID"
  echo
  cat "outputs/comparison-$RUN_ID.md"
  echo
  echo "Validation numbers for model selection only."
  echo "Reported results come from benchmark-evaluation."
fi
