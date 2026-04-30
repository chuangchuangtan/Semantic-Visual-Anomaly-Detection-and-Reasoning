#!/usr/bin/env bash
set -euo pipefail

MODEL=""
TEST_JSON=""
IMAGE_ROOT=""
OUTPUT_DIR="runs/$(date +%Y%m%d_%H%M%S)"
PROMPT_FILE=""
RAW_RESPONSE_ONLY=0
BASE_PORT=8000
INSTANCES=1
CONCURRENCY=4
PATH_MODE="basename"
MAX_RETRIES=3
GT_JSON=""
METRIC_SCRIPT=""
THRESHOLDS="0.7,0.8,0.9"
ALPHA="0.5"
USE_GPU=0
EXTRA_INFER_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_pipeline.sh \
    --model MODEL_ID \
    --test-json data/test.json \
    --image-root /path/to/images \
    --instances 4 \
    --concurrency 32

Options:
  --model ID                  Model id served by vLLM. Required.
  --test-json PATH            Dataset JSON. Required.
  --image-root DIR            Image directory. Required.
  --prompt-file PATH          Optional prompt text file passed to infer_dataset.py.
  --raw-response-only         Keep original model response text and skip response parsing.
  --output-dir DIR            Run directory. Default: runs/YYYYmmdd_HHMMSS.
  --base-port PORT            First vLLM port. Default: 8000.
  --instances NUM             Number of vLLM instances. Default: 1.
  --concurrency NUM           Concurrent requests. Default: 4.
  --path-mode MODE            basename, relative, or absolute. Default: basename.
  --max-retries NUM           Retries per image. Default: 3.
  --gt-json PATH              Optional ground-truth JSON for metric evaluation.
  --metric-script PATH        Optional semap_semf1_metric.py path.
  --thresholds LIST           Metric thresholds. Default: 0.7,0.8,0.9.
  --alpha NUM                 Metric alpha. Default: 0.5.
  --use-gpu                   Forward --use_gpu to metric script.
  --help                      Show this help.

Everything after -- is forwarded to scripts/infer_dataset.py.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --test-json) TEST_JSON="$2"; shift 2 ;;
    --image-root|--projection-root) IMAGE_ROOT="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --raw-response-only) RAW_RESPONSE_ONLY=1; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --instances|--gpus) INSTANCES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --path-mode) PATH_MODE="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --gt-json) GT_JSON="$2"; shift 2 ;;
    --metric-script) METRIC_SCRIPT="$2"; shift 2 ;;
    --thresholds) THRESHOLDS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --use-gpu) USE_GPU=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_INFER_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$TEST_JSON" || -z "$IMAGE_ROOT" ]]; then
  echo "Missing required --model, --test-json, or --image-root." >&2
  usage >&2
  exit 2
fi

if [[ "$RAW_RESPONSE_ONLY" == "1" && -n "$GT_JSON" && -n "$METRIC_SCRIPT" ]]; then
  echo "--raw-response-only cannot be combined with SemAP/SemF1 evaluation because metrics expect structured parsed predictions." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
PRED_NDJSON="$OUTPUT_DIR/predictions.ndjson"
ERROR_NDJSON="$OUTPUT_DIR/errors.ndjson"
MISSING_JSON="$OUTPUT_DIR/missing_images.json"
PRED_JSON="$OUTPUT_DIR/predictions.json"
METRIC_OUTPUT_ROOT="$OUTPUT_DIR/metric_runs"

echo "[1/3] Running inference -> $PRED_NDJSON"
infer_cmd=(
  python3 scripts/infer_dataset.py
  --test-json "$TEST_JSON"
  --image-root "$IMAGE_ROOT"
  --path-mode "$PATH_MODE"
  --out "$PRED_NDJSON"
  --error-out "$ERROR_NDJSON"
  --missing-out "$MISSING_JSON"
  --model "$MODEL"
  --base-port "$BASE_PORT"
  --instances "$INSTANCES"
  --concurrency "$CONCURRENCY"
  --max-retries "$MAX_RETRIES"
  --resume
)
if [[ -n "$PROMPT_FILE" ]]; then
  infer_cmd+=(--prompt-file "$PROMPT_FILE")
fi
if [[ "$RAW_RESPONSE_ONLY" == "1" ]]; then
  infer_cmd+=(--raw-response-only)
fi
if [[ "${#EXTRA_INFER_ARGS[@]}" -gt 0 ]]; then
  infer_cmd+=("${EXTRA_INFER_ARGS[@]}")
fi
"${infer_cmd[@]}"

echo "[2/3] Merging predictions -> $PRED_JSON"
merge_cmd=(
  python3 scripts/merge_results.py
  --input "$PRED_NDJSON"
  --output "$PRED_JSON"
)
if [[ "$RAW_RESPONSE_ONLY" == "1" ]]; then
  merge_cmd+=(--output-format response)
else
  merge_cmd+=(--prefer-parsed)
fi
"${merge_cmd[@]}"

echo "[INFO] Run artifacts are in: $OUTPUT_DIR"
