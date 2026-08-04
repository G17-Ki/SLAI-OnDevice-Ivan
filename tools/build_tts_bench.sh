#!/usr/bin/env bash
# Build the standalone TTS bench APK.
#
# Separate Gradle project from android-app/; building this cannot affect the
# validated LFM demo or its release APK.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/android-tts-bench"
WORKERS="${ANDROID_BUILD_MAX_WORKERS:-2}"

if [[ -z "${JAVA_HOME:-}" ]]; then
  for candidate in \
    /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    /usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home; do
    [[ -x "$candidate/bin/java" ]] && export JAVA_HOME="$candidate" && break
  done
  if [[ -z "${JAVA_HOME:-}" && -x /usr/libexec/java_home ]]; then
    JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || true)"
    export JAVA_HOME
  fi
fi
[[ -n "${JAVA_HOME:-}" ]] || { echo "JDK 17 not found; set JAVA_HOME." >&2; exit 2; }

if [[ -z "${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}" ]]; then
  for candidate in "$HOME/Library/Android/sdk" "$HOME/.cache/lmf-demo/android-sdk"; do
    [[ -d "$candidate/platforms" ]] && export ANDROID_HOME="$candidate" && break
  done
fi
[[ -n "${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}" ]] || {
  echo "Set ANDROID_HOME to an SDK with platform android-35." >&2; exit 2; }

# The sherpa-onnx runtime is not stored in this repository.
export SHERPA_AAR_DIR="${SHERPA_AAR_DIR:-$HOME/.cache/lmf-demo/sherpa-aar}"
if [[ ! -f "$SHERPA_AAR_DIR/sherpa-onnx-1.13.4.aar" ]]; then
  echo "sherpa-onnx AAR missing; fetching..."
  "$REPO_ROOT/tools/fetch_sherpa_aar.sh"
fi

echo "JAVA_HOME=$JAVA_HOME"
echo "ANDROID_HOME=${ANDROID_HOME:-$ANDROID_SDK_ROOT}"
echo "SHERPA_AAR_DIR=$SHERPA_AAR_DIR"

"$APP_DIR/gradlew" --no-daemon --max-workers="$WORKERS" -p "$APP_DIR" :app:assembleDebug

APK="$APP_DIR/app/build/outputs/apk/debug/app-debug.apk"
echo
echo "APK:    $APK"
echo "size:   $(du -h "$APK" | cut -f1)"
echo "sha256: $(shasum -a 256 "$APK" | cut -d' ' -f1)"
