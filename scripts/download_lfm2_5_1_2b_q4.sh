#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/lfm2.5-1.2b-instruct-q4}"
MODEL_REPO="LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
MODEL_REVISION="047e06635fbe71469926b35ea414537245218200"
MODEL_FILE="LFM2.5-1.2B-Instruct-Q4_0.gguf"
MODEL_SHA256="2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face 'hf' command is required." >&2
  echo "Install it with: python3 -m pip install huggingface_hub" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  hf download "$MODEL_REPO" \
    "$MODEL_FILE" README.md LICENSE \
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

echo "Verified LFM2.5-1.2B-Instruct Q4_0: $OUTPUT_DIR/$MODEL_FILE"
