#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-adapters/muffin-0.5b}"

exec .venv/bin/mlx_lm.server \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --host 127.0.0.1 \
  --port 8080
