#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_DIR LLAMA_CPP_DIR [PORT]" >&2
  exit 2
fi

MODEL_DIR="$1"
LLAMA_CPP_DIR="$2"
PORT="${3:-8082}"
MODEL="$MODEL_DIR/Qwen3-0.6B-Q4_0.gguf"
SERVER="$LLAMA_CPP_DIR/llama-server"

if [[ ! -f "$MODEL" ]]; then
  echo "Model is missing: $MODEL" >&2
  exit 2
fi
if [[ ! -x "$SERVER" ]]; then
  echo "llama-server is missing or not executable: $SERVER" >&2
  exit 2
fi

exec "$SERVER" \
  -m "$MODEL" \
  -c 512 \
  -ngl 99 \
  --reasoning off \
  --reasoning-format none \
  --host 127.0.0.1 \
  --port "$PORT" \
  --metrics
