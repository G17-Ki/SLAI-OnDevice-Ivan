#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/qwen3-1.7b}"
REVISION="90862c4b9d2787eaed51d12237eafdfe7c5f6077"
MODEL_FILE="Qwen3-1.7B-Q8_0.gguf"
MODEL_SHA256="061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a"
URL="https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/${REVISION}/${MODEL_FILE}"

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
