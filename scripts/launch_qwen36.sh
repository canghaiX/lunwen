#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
  --context-length "${CONTEXT_LENGTH:-32768}" \
  --reasoning-parser qwen3
