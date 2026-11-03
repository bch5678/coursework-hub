#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_EXECUTABLE="${PYTHON_BIN:-python}"
BASE_CONFIG="configs/icu_transfer/cross_attention_adapter.json"
BASE_CONFIG_EXPLICIT=0
TASK=""
OUTPUT_DIR="outputs/three_multimodal"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
BOOTSTRAP_SAMPLES=2000
THRESHOLD_METHOD="youden_j"
EVALUATION_UNIT="admission"
RESUME=0
INCLUDE_UNIMODAL=0
FREEZE_FROM_UNIMODAL=0
PREPARE=0
PREPARE_ONLY=0
PIPELINE_CONFIG=""
DOWNLOAD_MANIFEST=""
MIMIC_ROOT=""
PRETRAIN_CLINICAL=0
PRETRAIN_EPOCHS=60
EXPECTED_INDEX_SHA256=""
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<'HELP'
Run concat -> joint attention -> cross-attention adapter, sequentially.
Each model: train + validation -> save weights/predictions -> benchmark test.

bash scripts/run_three_multimodal.sh [options]
  --base-config FILE            Shared multimodal config (defaults to ICU-transfer v3)
  --task TASK                   icu_transfer selects matching transfer data/training configs
  --run-id NAME                 Batch name; use the same name with --resume
  --output-dir DIR              Default: outputs/three_multimodal
  --run-dir DIR                 Frozen dataset run
  --clinical-source FILE        Existing clinical CSV
  --clinical-checkpoint FILE    Compatible pretrained clinical encoder
  --image-cache FILE            Existing decoded-image cache metadata
  --device DEVICE              cpu, cuda or auto (otherwise inherit base config)
  --seed NUMBER                Shared training/bootstrap seed
  --set SECTION.KEY=VALUE       Shared override, repeatable
  --bootstrap-samples NUMBER    Default: 2000
  --threshold-method METHOD    youden_j (default), fixed_0.5 or f1
  --evaluation-unit UNIT       admission (default) or sample
  --freeze-from-unimodal       Train baselines first, then freeze their encoders for fusion
  --include-unimodal           Also run clinical-only and CXR-only on the shared dataset
  --resume                     Skip completed training/evaluation stages
  --prepare                    Prepare/reuse the original pipeline inputs before training
  --prepare-only               Prepare inputs without training or benchmark scoring
  --pipeline-config FILE       Raw-data pipeline config; also enables --prepare
  --download-manifest FILE     Download/verify/extract authorized files before preparation
  --mimic-root DIR             MIMIC-IV root used for missing clinical features
  --pretrain-clinical           Reproduce a missing external clinical encoder checkpoint
  --pretrain-epochs NUMBER      Default: 60
  --expected-index-sha256 HASH  Stop unless the dataset matches the original baseline index

All relative paths are resolved from the training project directory.
PYTHON_BIN selects the Python interpreter; CUDA_VISIBLE_DEVICES selects the GPU.
HELP
      exit 0 ;;
    --freeze-from-unimodal) FREEZE_FROM_UNIMODAL=1; INCLUDE_UNIMODAL=1; shift ;;
    --include-unimodal) INCLUDE_UNIMODAL=1; shift ;;
    --resume) RESUME=1; shift ;;
    --prepare) PREPARE=1; shift ;;
    --prepare-only) PREPARE=1; PREPARE_ONLY=1; shift ;;
    --pretrain-clinical) PREPARE=1; PRETRAIN_CLINICAL=1; shift ;;
    --base-config|--task|--run-id|--output-dir|--run-dir|--clinical-source|--clinical-checkpoint|--image-cache|--device|--seed|--set|--bootstrap-samples|--threshold-method|--evaluation-unit|--pipeline-config|--download-manifest|--mimic-root|--pretrain-epochs|--expected-index-sha256)
      [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --base-config) BASE_CONFIG="$2"; BASE_CONFIG_EXPLICIT=1 ;;
        --task) TASK="$2" ;;
        --run-id) RUN_ID="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --run-dir) OVERRIDES+=("data.run_dir=$2") ;;
        --clinical-source) OVERRIDES+=("data.clinical_source=$2" "data.clinical_provider=table") ;;
        --clinical-checkpoint) OVERRIDES+=("clinical_encoder.pretrained=$2") ;;
        --image-cache) OVERRIDES+=("data.image_cache=$2") ;;
        --device) OVERRIDES+=("device=$2") ;;
        --seed) OVERRIDES+=("seed=$2") ;;
        --set) OVERRIDES+=("$2") ;;
        --bootstrap-samples) BOOTSTRAP_SAMPLES="$2" ;;
        --threshold-method) THRESHOLD_METHOD="$2" ;;
        --evaluation-unit) EVALUATION_UNIT="$2" ;;
        --pipeline-config) PIPELINE_CONFIG="$2"; PREPARE=1 ;;
        --download-manifest) DOWNLOAD_MANIFEST="$2"; PREPARE=1 ;;
        --mimic-root) MIMIC_ROOT="$2" ;;
        --pretrain-epochs) PRETRAIN_EPOCHS="$2" ;;
        --expected-index-sha256) EXPECTED_INDEX_SHA256="$2" ;;
      esac
      shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TASK" && "$BASE_CONFIG_EXPLICIT" -eq 0 ]]; then
  TASK="icu_transfer"
fi

if [[ -n "$TASK" ]]; then
  [[ "$TASK" = "icu_transfer" ]] || { printf 'Unsupported task preset: %s\n' "$TASK" >&2; exit 2; }
  [[ "$BASE_CONFIG_EXPLICIT" -eq 1 ]] || BASE_CONFIG="configs/icu_transfer/cross_attention_adapter.json"
  if [[ "$PREPARE" -eq 1 && -z "$PIPELINE_CONFIG" ]]; then
    PIPELINE_CONFIG="configs/preparation/icu_transfer_pipeline.json"
  fi
  OVERRIDES+=("data.task=icu_transfer" "fusion.concat_image_pooling=patch_mean" "fusion.concat_project=false")
fi

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { printf 'Invalid run-id\n' >&2; exit 2; }
[[ "$BOOTSTRAP_SAMPLES" =~ ^[1-9][0-9]*$ ]] || { printf 'bootstrap-samples must be positive\n' >&2; exit 2; }
[[ "$THRESHOLD_METHOD" =~ ^(youden_j|fixed_0.5|f1)$ ]] || { printf 'Invalid threshold method\n' >&2; exit 2; }
[[ "$EVALUATION_UNIT" =~ ^(admission|sample)$ ]] || { printf 'Invalid evaluation unit\n' >&2; exit 2; }

# Training imports use the editable package mapping from pyproject.toml.
export PYTHONPATH="$PROJECT_DIR/../benchmark-evaluation/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
BATCH_DIR="$OUTPUT_DIR/$RUN_ID"

if [[ "$PREPARE" -eq 1 ]]; then
  PREPARED_CONFIG="$PROJECT_DIR/data/prepared_configs/$RUN_ID.json"
  PREPARE_ARGS=(--config "$BASE_CONFIG" --output-config "$PREPARED_CONFIG" --pretrain-epochs "$PRETRAIN_EPOCHS")
  [[ -z "$PIPELINE_CONFIG" ]] || PREPARE_ARGS+=(--pipeline-config "$PIPELINE_CONFIG")
  [[ -z "$DOWNLOAD_MANIFEST" ]] || PREPARE_ARGS+=(--download-manifest "$DOWNLOAD_MANIFEST")
  [[ -z "$MIMIC_ROOT" ]] || PREPARE_ARGS+=(--mimic-root "$MIMIC_ROOT")
  [[ -z "$EXPECTED_INDEX_SHA256" ]] || PREPARE_ARGS+=(--expected-index-sha256 "$EXPECTED_INDEX_SHA256")
  [[ "$PRETRAIN_CLINICAL" -eq 0 ]] || PREPARE_ARGS+=(--pretrain-clinical)
  for OVERRIDE in "${OVERRIDES[@]}"; do
    PREPARE_ARGS+=(--set "$OVERRIDE")
  done
  mkdir -p "$PROJECT_DIR/data/preparation_logs"
  "$PYTHON_EXECUTABLE" -u -m bn5212_training.prepare "${PREPARE_ARGS[@]}" \
    2>&1 | tee -a "$PROJECT_DIR/data/preparation_logs/$RUN_ID.log"
  BASE_CONFIG="$PREPARED_CONFIG"
  OVERRIDES=()
  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    printf 'Prepared inputs/config: %s\n' "$PREPARED_CONFIG"
    exit 0
  fi
fi

DATA_RUN="$("$PYTHON_EXECUTABLE" - "$BASE_CONFIG" "$BATCH_DIR" "$RESUME" \
  "$BOOTSTRAP_SAMPLES" "$THRESHOLD_METHOD" "$EVALUATION_UNIT" "$INCLUDE_UNIMODAL" "$FREEZE_FROM_UNIMODAL" "${OVERRIDES[@]}" <<'PY'
import copy
import json
from pathlib import Path
import sys

from bn5212_training.config import config_from_dict, load_config
from bn5212_benchmark.io import load_dataset_index

base_path, destination, resume, samples, threshold, unit, include_unimodal, freeze_from_unimodal, *items = sys.argv[1:]
overrides = {}
for item in items:
    key, separator, value = item.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError(f"Expected SECTION.KEY=VALUE, got {item!r}")
    if key in {"experiment", "modalities", "output_dir", "fusion.name", "fusion.direction"}:
        raise ValueError(f"{key} is controlled by the three-model script")
    try:
        overrides[key] = json.loads(value)
    except json.JSONDecodeError:
        overrides[key] = value
base = load_config(base_path, overrides).to_dict()
if set(base["modalities"]) != {"clinical", "cxr"}:
    raise ValueError("The shared base config must enable clinical and cxr")
if not 0 <= base["seed"] < 2**32:
    raise ValueError("Seed must be in [0, 2**32)")
run_dir = str(Path(base["data"]["run_dir"]).resolve())
required = ("SUCCESS.json", "dataset_spec.json", "index.csv")
missing = [name for name in required if not (Path(run_dir) / name).is_file()]
if missing:
    state = "directory does not exist" if not Path(run_dir).is_dir() else "dataset has not been published completely"
    raise SystemExit(
        f"Dataset run is unavailable: {run_dir}\n"
        f"Reason: {state}; missing: {', '.join(missing)}\n"
        "Pass --run-dir /absolute/path/to/the/published/dataset-run, using the same run as your baselines.\n"
        "An embeddings .pt/.npz directory or a raw MIMIC directory is not a published dataset run.\n"
        "If no published run exists, finish the data pipeline first; do not create SUCCESS.json manually."
        "\nYou can chain it here with --prepare --pipeline-config /path/to/server.local.json."
    )
index, spec = load_dataset_index(run_dir)
if base["data"]["task"] is not None and base["data"]["task"] != spec.get("task"):
    raise ValueError(f"Expected dataset task {base['data']['task']!r}, got {spec.get('task')!r}")
paths = [base["clinical_encoder"]["pretrained"], base["data"]["image_cache"]]
if base["data"]["clinical_provider"] == "table":
    if not base["data"]["clinical_source"]:
        raise ValueError("The table provider requires clinical_source")
    paths.append(base["data"]["clinical_source"])
for value in paths:
    if value and not Path(value).is_file():
        raise FileNotFoundError(f"Missing input file: {value}")
batch = Path(destination).resolve()
if freeze_from_unimodal == "1" and (base["freeze_encoders"] or base["clinical_checkpoint"] or base["image_checkpoint"]):
    raise ValueError("Two-stage training requires a base config without study checkpoint sources")
configs = {}
models = [("concat", "concat_mlp"), ("joint", "joint_self_attention"),
          ("cross_attention_adapter", "cross_attention_adapter")]
if include_unimodal == "1":
    models = [("clinical_only", "clinical_only"), ("cxr_only", "image_only")] + models
for name, fusion in models:
    payload = copy.deepcopy(base)
    payload["data"]["run_dir"] = run_dir
    payload["experiment"] = name
    if fusion in {"clinical_only", "image_only"}:
        payload["modalities"] = ["clinical"] if fusion == "clinical_only" else ["cxr"]
    elif freeze_from_unimodal == "1":
        payload["freeze_encoders"] = True
        payload["clinical_checkpoint"] = str(batch / "training/clinical_only/fit/checkpoint_best.pt")
        payload["image_checkpoint"] = str(batch / "training/cxr_only/fit/checkpoint_best.pt")
    payload["output_dir"] = str(batch / "training")
    payload["fusion"].update(name=fusion, direction="clinical_to_image")
    if fusion == "image_only" and freeze_from_unimodal == "1":
        payload["fusion"]["pooling"] = "patch_mean"
    payload["notes"] += f" | three-model comparison: {name}; shared base settings"
    configs[name] = config_from_dict(payload).to_dict()
plan = {"configs": configs, "index_sha256": spec["index_sha256"], "python": sys.executable,
        "benchmark": {"bootstrap_samples": int(samples), "threshold_method": threshold,
                      "evaluation_unit": unit, "seed": base["seed"]}}
plan_path = batch / "plan.json"
if batch.exists():
    if resume != "1":
        raise FileExistsError(f"{batch} already exists; choose a new run-id or add --resume")
    if not plan_path.is_file() or json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume requires the same configuration, inputs and benchmark options")
else:
    (batch / "configs").mkdir(parents=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
for name, payload in configs.items():
    (batch / "configs" / f"{name}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(run_dir)
PY
)"

mkdir -p "$BATCH_DIR/logs"
trap 'printf "Failed at line %s; check %s/logs\n" "$LINENO" "$BATCH_DIR" >&2' ERR
MODELS=(concat joint cross_attention_adapter)
if [[ "$INCLUDE_UNIMODAL" -eq 1 ]]; then
  MODELS=(clinical_only cxr_only "${MODELS[@]}")
fi
BENCHMARK_DIRS=()
TRAIN_DIRS=()
BOOTSTRAP_SEED="$("$PYTHON_EXECUTABLE" -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$BATCH_DIR/configs/${MODELS[0]}.json")"
printf 'Model comparison run: %s\nDataset: %s\n' "$BATCH_DIR" "$DATA_RUN"

for MODEL in "${MODELS[@]}"; do
  TRAIN_ROOT="$BATCH_DIR/training/$MODEL/fit"
  BENCHMARK_ROOT="$BATCH_DIR/benchmark/$MODEL"
  CHECKPOINT="$TRAIN_ROOT/checkpoint_best.pt"
  printf '\n=== %s: training + validation ===\n' "$MODEL"
  if [[ -f "$TRAIN_ROOT/TRAIN_SUCCESS" ]]; then
    for ARTIFACT in checkpoint_best.pt predictions_val.csv predictions_test.csv run_manifest.json summary.csv; do
      [[ -f "$TRAIN_ROOT/$ARTIFACT" ]] || { printf 'Missing training artifact: %s\n' "$TRAIN_ROOT/$ARTIFACT" >&2; exit 1; }
    done
    printf 'Training already complete; keeping saved weights.\n'
  else
    [[ ! -e "$BENCHMARK_ROOT" ]] || { printf 'Refusing to retrain weights with existing benchmark results.\n' >&2; exit 1; }
    "$PYTHON_EXECUTABLE" -u -c 'from bn5212_training.cli import train_main; train_main()' \
      --config "$BATCH_DIR/configs/$MODEL.json" \
      --output-dir "$BATCH_DIR/training" --run-id fit \
      2>&1 | tee -a "$BATCH_DIR/logs/$MODEL.train.log"
    [[ -f "$CHECKPOINT" && -f "$TRAIN_ROOT/run_manifest.json" && -f "$TRAIN_ROOT/predictions_test.csv" ]]
    touch "$TRAIN_ROOT/TRAIN_SUCCESS"
  fi

  printf '\n=== %s: benchmark test ===\n' "$MODEL"
  if [[ -f "$BENCHMARK_ROOT/SUCCESS.json" ]]; then
    printf 'Benchmark already complete; keeping immutable report.\n'
  else
    [[ ! -e "$BENCHMARK_ROOT" ]] || { printf 'Incomplete benchmark directory; use a new run-id.\n' >&2; exit 1; }
    BOOTSTRAP_SEED="$("$PYTHON_EXECUTABLE" -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$BATCH_DIR/configs/$MODEL.json")"
    "$PYTHON_EXECUTABLE" -u -c 'from bn5212_benchmark.cli import evaluate_main; evaluate_main()' \
      --run-dir "$DATA_RUN" \
      --val-predictions "$TRAIN_ROOT/predictions_val.csv" \
      --test-predictions "$TRAIN_ROOT/predictions_test.csv" \
      --checkpoint "$CHECKPOINT" --model-name "$MODEL" --model-version "$RUN_ID" \
      --output-dir "$BENCHMARK_ROOT" --threshold-method "$THRESHOLD_METHOD" \
      --evaluation-unit "$EVALUATION_UNIT" --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      --bootstrap-seed "$BOOTSTRAP_SEED" \
      2>&1 | tee -a "$BATCH_DIR/logs/$MODEL.benchmark.log"
  fi
  BENCHMARK_DIRS+=("$BENCHMARK_ROOT")
  TRAIN_DIRS+=("$TRAIN_ROOT")
done

"$PYTHON_EXECUTABLE" -u -c 'from bn5212_training.cli import summarize_main; summarize_main()' \
  "${TRAIN_DIRS[@]}" --output "$BATCH_DIR/validation_comparison" \
  2>&1 | tee -a "$BATCH_DIR/logs/summary.log"
"$PYTHON_EXECUTABLE" -u -c 'from bn5212_benchmark.cli import compare_main; compare_main()' \
  "${BENCHMARK_DIRS[@]}" --output-csv "$BATCH_DIR/benchmark_leaderboard.csv" \
  --output-html "$BATCH_DIR/benchmark_leaderboard.html" \
  2>&1 | tee -a "$BATCH_DIR/logs/summary.log"
if [[ "$INCLUDE_UNIMODAL" -eq 1 ]]; then
  "$PYTHON_EXECUTABLE" -u -m bn5212_training.paired_comparison "$BATCH_DIR" \
    --samples "$BOOTSTRAP_SAMPLES" --seed "$BOOTSTRAP_SEED" \
    2>&1 | tee -a "$BATCH_DIR/logs/summary.log"
fi
printf '\nFinished. Weights: %s/training/<model>/fit/checkpoint_best.pt\n' "$BATCH_DIR"
printf 'Benchmark reports: %s/benchmark/<model>/report.html\n' "$BATCH_DIR"
printf 'Combined results: %s/benchmark_leaderboard.csv\n' "$BATCH_DIR"
