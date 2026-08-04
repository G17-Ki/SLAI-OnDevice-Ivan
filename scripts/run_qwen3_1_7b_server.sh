#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_DIR LLAMA_CPP_DIR [PORT]" >&2
  exit 2
fi

MODEL_DIR="$1"
LLAMA_CPP_DIR="$2"
PORT="${3:-8084}"
MODEL_FILE="${MODEL_FILE:-Qwen3-1.7B-Q4_0.gguf}"
MODEL="$MODEL_DIR/$MODEL_FILE"
SERVER="$LLAMA_CPP_DIR/llama-server"
REASONING="${REASONING:-off}"
REASONING_FORMAT="${REASONING_FORMAT:-none}"
REASONING_BUDGET="${REASONING_BUDGET:-}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model is missing: $MODEL" >&2
  exit 2
fi
if [[ ! -x "$SERVER" ]]; then
  echo "llama-server is missing or not executable: $SERVER" >&2
  exit 2
fi

server_args=(
  -m "$MODEL"
  -c 512
  -ngl 99
  --reasoning "$REASONING"
  --reasoning-format "$REASONING_FORMAT"
  --host 127.0.0.1
  --port "$PORT"
  --metrics
)

if [[ -n "$REASONING_BUDGET" ]]; then
  server_args+=(--reasoning-budget "$REASONING_BUDGET")
fi

exec "$SERVER" "${server_args[@]}"
