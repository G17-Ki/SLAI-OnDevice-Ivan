#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/qwen3.5-0.8b-q4}"
MODEL_REPO="ggml-org/Qwen3.5-0.8B-GGUF"
MODEL_REVISION="8fea620810c4afa23dd6443f999a48574c1611a3"
MODEL_FILE="Qwen3.5-0.8B-Q4_0.gguf"
MODEL_SHA256="57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"
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

echo "Verified Qwen3.5-0.8B Q4_0: $final"
