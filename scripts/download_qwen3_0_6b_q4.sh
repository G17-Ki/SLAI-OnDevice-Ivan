#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/qwen3-0.6b-q4}"
MODEL_REPO="unsloth/Qwen3-0.6B-GGUF"
MODEL_REVISION="50968a4468ef4233ed78cd7c3de230dd1d61a56b"
MODEL_FILE="Qwen3-0.6B-Q4_0.gguf"
MODEL_SHA256="33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face 'hf' command is required." >&2
  echo "Install it with: python3 -m pip install huggingface_hub" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
hf download "$MODEL_REPO" \
  "$MODEL_FILE" README.md \
  --revision "$MODEL_REVISION" \
  --local-dir "$OUTPUT_DIR"

if command -v sha256sum >/dev/null 2>&1; then
  echo "$MODEL_SHA256  $OUTPUT_DIR/$MODEL_FILE" | sha256sum -c -
else
  actual="$(shasum -a 256 "$OUTPUT_DIR/$MODEL_FILE" | awk '{print $1}')"
  if [[ "$actual" != "$MODEL_SHA256" ]]; then
    echo "SHA-256 mismatch for $MODEL_FILE" >&2
    echo "expected: $MODEL_SHA256" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
fi

echo "Verified Qwen3-0.6B Q4_0: $OUTPUT_DIR/$MODEL_FILE"
