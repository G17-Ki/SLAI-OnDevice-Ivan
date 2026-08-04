#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
MODEL_REPO="LiquidAI/LFM2.5-Audio-1.5B-GGUF"
MODEL_REVISION="7d525f883a077e20afb782f2ff618edcae0e39e4"
ARCHIVE_REL="runners/llama-liquid-audio-android-arm64.zip"
CACHE_DIR="${1:-$LFM_DEMO_WORK_ROOT/downloads/lfm-android-runner}"
JNI_DIR="$REPO_ROOT/android-app/app/src/main/jniLibs/arm64-v8a"
NOTICE_DIR="$REPO_ROOT/android-app/third_party/llama-liquid-audio"

if ! command -v hf >/dev/null 2>&1; then
  echo "Install the Hugging Face CLI with: python3 -m pip install huggingface_hub" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR" "$JNI_DIR" "$NOTICE_DIR"
hf download "$MODEL_REPO" "$ARCHIVE_REL" \
  --revision "$MODEL_REVISION" \
  --local-dir "$CACHE_DIR"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
unzip -q "$CACHE_DIR/$ARCHIVE_REL" -d "$TEMP_DIR"
RUNNER_DIR="$TEMP_DIR/llama-liquid-audio-android-arm64"

cp "$RUNNER_DIR"/*.so "$JNI_DIR/"
cp "$RUNNER_DIR/llama-liquid-audio-server" "$JNI_DIR/liblfmserver.so"
cp "$RUNNER_DIR"/LICENSE* "$NOTICE_DIR/"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$JNI_DIR" && sha256sum -c "$REPO_ROOT/android-app/runner-files.sha256")
else
  (cd "$JNI_DIR" && shasum -a 256 -c "$REPO_ROOT/android-app/runner-files.sha256")
fi

echo "Verified Android ARM64 runner libraries from $MODEL_REPO@$MODEL_REVISION"
