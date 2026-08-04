#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_ROOT/android-app"
BUILD_WORKERS="${ANDROID_BUILD_MAX_WORKERS:-2}"

if [[ ! "$BUILD_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ANDROID_BUILD_MAX_WORKERS must be a positive integer." >&2
  exit 2
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
  if [[ -x /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/bin/java ]]; then
    export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
  elif [[ -x /usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/bin/java ]]; then
    export JAVA_HOME=/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
  elif [[ -x /usr/libexec/java_home ]]; then
    export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
  fi
fi
if [[ -z "${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}" ]]; then
  echo "Set ANDROID_HOME or ANDROID_SDK_ROOT to an Android SDK containing API 35." >&2
  exit 2
fi

export SHERPA_AAR_DIR="${SHERPA_AAR_DIR:-$HOME/.cache/lmf-demo/sherpa-aar}"
if [[ ! -f "$SHERPA_AAR_DIR/sherpa-onnx-1.13.4.aar" ]]; then
  echo "sherpa-onnx 1.13.4 AAR missing; fetching the pinned external dependency..."
  "$REPO_ROOT/tools/fetch_sherpa_aar.sh"
fi

"$APP_DIR/gradlew" \
  --no-daemon \
  --max-workers="$BUILD_WORKERS" \
  -p "$APP_DIR" \
  :app:assembleDebug
echo "APK: $APP_DIR/app/build/outputs/apk/debug/app-debug.apk"
