#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/qwen3-1.7b-q4}"
MODEL_REPO="unsloth/Qwen3-1.7B-GGUF"
MODEL_REVISION="d7f544eead698dbd1f15126ef60b45a1e1933222"
MODEL_FILE="Qwen3-1.7B-Q4_0.gguf"
MODEL_SHA256="c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5"
MODEL_URL="https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/$MODEL_FILE"

mkdir -p "$OUTPUT_DIR"
temporary="$OUTPUT_DIR/$MODEL_FILE.download"
final="$OUTPUT_DIR/$MODEL_FILE"

if [[ ! -f "$final" ]]; then
  while ! curl --http1.1 --fail --location --continue-at - \
    --output "$temporary" "$MODEL_URL"; do
    sleep 3
  done
  mv "$temporary" "$final"
fi

actual="$(shasum -a 256 "$final" | awk '{print $1}')"
if [[ "$actual" != "$MODEL_SHA256" ]]; then
  echo "SHA-256 mismatch for $MODEL_FILE" >&2
  echo "expected: $MODEL_SHA256" >&2
  echo "actual:   $actual" >&2
  exit 1
fi

curl --http1.1 --fail --location \
  --output "$OUTPUT_DIR/README.md" \
  "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/README.md"

echo "Verified Qwen3-1.7B Q4_0: $final"
