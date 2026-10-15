#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE="ai.liquid.lfmdemo.dev"
ACTIVITY="$PACKAGE/.MainActivity"
APK="$REPO_ROOT/android-app/releases/lfm-audio-demo-0.4.13-expressive-lfm-debug.apk"
MODEL_DIR="${LFM_MODEL_DIR:-}"
REMOTE_STAGE="/data/local/tmp/lfm-audio-model-install"

usage() {
  echo "usage: $0 --model-dir DIR [--apk FILE]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --apk) APK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "Pass --model-dir or set LFM_MODEL_DIR." >&2
  usage
  exit 2
fi
if [[ ! -f "$APK" ]]; then
  echo "APK not found: $APK" >&2
  exit 2
fi
if ! command -v adb >/dev/null 2>&1; then
  echo "adb is required and must be on PATH." >&2
  exit 2
fi
if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
  echo "Connect one Android phone, enable USB debugging, and accept its authorization prompt." >&2
  exit 2
fi

FILES=(
  LFM2.5-Audio-1.5B-Q4_0.gguf
  mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf
  vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf
  tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf
)
for name in "${FILES[@]}"; do
  if [[ ! -f "$MODEL_DIR/$name" ]]; then
    echo "Model file missing: $MODEL_DIR/$name" >&2
    exit 2
  fi
done

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$MODEL_DIR" && sha256sum -c "$REPO_ROOT/android-app/model-files.sha256")
else
  (cd "$MODEL_DIR" && shasum -a 256 -c "$REPO_ROOT/android-app/model-files.sha256")
fi

adb install -r "$APK"
adb shell "mkdir -p '$REMOTE_STAGE'"
adb shell "run-as '$PACKAGE' mkdir -p files/models"

for name in "${FILES[@]}"; do
  echo "Installing $name..."
  adb push "$MODEL_DIR/$name" "$REMOTE_STAGE/$name"
  adb shell "run-as '$PACKAGE' cp '$REMOTE_STAGE/$name' 'files/models/$name'"
  adb shell "rm -f '$REMOTE_STAGE/$name'"
done

adb shell pm grant "$PACKAGE" android.permission.RECORD_AUDIO
adb shell pm grant "$PACKAGE" android.permission.POST_NOTIFICATIONS 2>/dev/null || true
adb shell am force-stop "$PACKAGE"
adb shell am start -n "$ACTIVITY"

echo "Application and LFM model installation complete."
echo "Deploy the ASR companion before using Expressive LFM."
echo "Deploy the Qwen companion as well for Exact text and multilingual responses."
echo "Keep USB connected after the companions report healthy."
echo "A disconnect or reboot stops both companions; reconnect and restart them over ADB."
