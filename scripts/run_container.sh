#!/usr/bin/env bash
set -euo pipefail
docker build -t lunwen:latest .
docker run --rm -it --gpus all --name lunwen \
  -e TRANSFORMERS_OFFLINE=1 \
  -e REVIMPACT_MODEL_PATH=/models/Qwen/Qwen3-0.6B \
  -v "$(pwd):/workspace/lunwen" \
  -v /data/models:/models:ro \
  -w /workspace/lunwen lunwen:latest bash
