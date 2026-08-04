#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="${1:?Usage: run_lfm_jp_server.sh BUNDLE_DIR [PORT] [extra runner arguments ...]}"
PORT="${2:-18081}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

RUNNER="$BUNDLE_DIR/runners/llama-liquid-audio-macos-arm64/llama-liquid-audio-server"
MODEL="$BUNDLE_DIR/LFM2.5-Audio-1.5B-JP-Q4_0.gguf"
MMPROJ="$BUNDLE_DIR/mmproj-LFM2.5-Audio-1.5B-JP-Q4_0.gguf"
VOCODER="$BUNDLE_DIR/vocoder-LFM2.5-Audio-1.5B-JP-Q4_0.gguf"
TOKENIZER="$BUNDLE_DIR/tokenizer-LFM2.5-Audio-1.5B-JP-Q4_0.gguf"

for file in "$RUNNER" "$MODEL" "$MMPROJ" "$VOCODER" "$TOKENIZER"
do
  if [[ ! -f "$file" ]]; then
    printf 'Missing Japanese runner/model file: %s\n' "$file" >&2
    exit 2
  fi
done

exec "$RUNNER" \
  -m "$MODEL" \
  -mm "$MMPROJ" \
  -mv "$VOCODER" \
  --tts-speaker-file "$TOKENIZER" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --log-colors off \
  --perf \
  "$@"
