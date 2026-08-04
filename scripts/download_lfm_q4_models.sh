#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/lfm25-audio-q4}"
MODEL_REPO="LiquidAI/LFM2.5-Audio-1.5B-GGUF"
MODEL_REVISION="7d525f883a077e20afb782f2ff618edcae0e39e4"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face 'hf' command is required." >&2
  echo "Install it with: python3 -m pip install huggingface_hub" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
hf download "$MODEL_REPO" \
  LFM2.5-Audio-1.5B-Q4_0.gguf \
  mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf \
  vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf \
  tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf \
  LICENSE README.md \
  --revision "$MODEL_REVISION" \
  --local-dir "$OUTPUT_DIR"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$OUTPUT_DIR" && sha256sum -c "$REPO_ROOT/android-app/model-files.sha256")
else
  (cd "$OUTPUT_DIR" && shasum -a 256 -c "$REPO_ROOT/android-app/model-files.sha256")
fi

echo "Verified Q4 model files: $OUTPUT_DIR"
