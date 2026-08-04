#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/lfm25-1.2b-instruct}"
REVISION="047e06635fbe71469926b35ea414537245218200"
MODEL_FILE="LFM2.5-1.2B-Instruct-Q4_0.gguf"
MODEL_SHA256="2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
URL="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/${REVISION}/${MODEL_FILE}"

mkdir -p "$OUTPUT_DIR"
PARTIAL="$OUTPUT_DIR/$MODEL_FILE.download"
FINAL="$OUTPUT_DIR/$MODEL_FILE"

if [[ ! -f "$FINAL" ]]; then
  while ! curl --http1.1 --fail --location --continue-at - --output "$PARTIAL" "$URL"; do
    sleep 3
  done
  mv "$PARTIAL" "$FINAL"
fi

actual="$(shasum -a 256 "$FINAL" | awk '{print $1}')"
if [[ "$actual" != "$MODEL_SHA256" ]]; then
  echo "SHA-256 mismatch for $FINAL: expected $MODEL_SHA256, got $actual" >&2
  exit 1
fi

printf 'Verified %s\n' "$FINAL"
