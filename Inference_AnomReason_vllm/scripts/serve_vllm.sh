#!/usr/bin/env bash
set -euo pipefail

MODEL=""
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
BASE_PORT=8000
LOG_DIR=""
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
STAGGER_SECONDS=30
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/serve_vllm.sh --model /path/to/model --gpus 0,1,2,3 [options]

Options:
  --model PATH_OR_ID             Model path or Hugging Face id. Required.
  --gpus LIST                    Physical GPU ids, comma-separated. Default: CUDA_VISIBLE_DEVICES or 0.
  --base-port PORT               First vLLM port. Default: 8000.
  --log-dir DIR                  Log directory. Default: runs/vllm_YYYYmmdd_HHMMSS.
  --gpu-memory-utilization NUM   vLLM GPU memory fraction. Default: 0.9.
  --max-model-len NUM            vLLM max model length. Default: 8192.
  --stagger-seconds NUM          Sleep between server launches. Default: 30.
  --dry-run                      Print commands without starting servers.
  --help                         Show this help.

Everything after -- is forwarded to `vllm serve`.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --stagger-seconds) STAGGER_SECONDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Missing required --model." >&2
  usage >&2
  exit 2
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/vllm_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
echo "[INFO] Starting ${#GPU_ARRAY[@]} vLLM server(s). Logs: $LOG_DIR"

for index in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$index]}"
  port=$((BASE_PORT + index))
  log_file="$LOG_DIR/vllm_gpu${gpu}_port${port}.log"
  cmd=(
    vllm serve "$MODEL"
    --port "$port"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
  )
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi
  echo "[INFO] GPU $gpu -> http://127.0.0.1:$port/v1"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1 &\n' "$log_file"
  else
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
    echo "$!" > "$LOG_DIR/vllm_gpu${gpu}_port${port}.pid"
    if [[ "$index" -lt $((${#GPU_ARRAY[@]} - 1)) ]]; then
      sleep "$STAGGER_SECONDS"
    fi
  fi
done

echo "[INFO] Done. Check logs with: tail -f $LOG_DIR/vllm_gpu*_port*.log"
