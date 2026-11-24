#!/usr/bin/env bash
# 3B QLoRA on the 8 GB M1 Mac.
#
# Usage:
#   ./fine_tune/train_3b_mac.sh                 # 2000 iters, resumes if found
#   ITERS=4000 ./fine_tune/train_3b_mac.sh
#   MAX_SEQ=256 ./fine_tune/train_3b_mac.sh     # if memory pressure is high
#
# Needs: model already downloaded (mlx-community/Qwen2.5-3B-Instruct-4bit)
# and data in data/mlx3/{train,valid}.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
DATA="${DATA:-data/mlx3}"
ADAPTER="${ADAPTER:-adapters/muffin-3b}"
ITERS="${ITERS:-2000}"
MAX_SEQ="${MAX_SEQ:-512}"
BATCH="${BATCH:-1}"
LAYERS="${LAYERS:-16}"
LR="${LR:-5e-5}"

mkdir -p "$ADAPTER" logs

RESUME=()
if [[ -f "$ADAPTER/adapters.safetensors" ]]; then
  echo "resuming from $ADAPTER/adapters.safetensors"
  RESUME=(--resume-adapter-file "$ADAPTER/adapters.safetensors")
fi

exec .venv/bin/python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --batch-size "$BATCH" \
  --iters "$ITERS" \
  --val-batches 2 \
  --learning-rate "$LR" \
  --max-seq-length "$MAX_SEQ" \
  --mask-prompt \
  --grad-checkpoint \
  --num-layers "$LAYERS" \
  --adapter-path "$ADAPTER" \
  --steps-per-report 50 \
  --steps-per-eval 500 \
  --save-every 500 \
  "${RESUME[@]}"
