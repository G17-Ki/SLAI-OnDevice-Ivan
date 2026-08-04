#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE="ai.liquid.lfmdemo"
STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${1:-$REPO_ROOT/reports/android_phone/device/$STAMP}"
APK="$REPO_ROOT/android-app/releases/lfm-audio-demo-0.1.0-debug.apk"

if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
  echo "No authorized Android device." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
"$SCRIPT_DIR/inspect_android_target.sh" >"$OUTPUT_DIR/device.txt"
adb shell dumpsys package "$PACKAGE" >"$OUTPUT_DIR/package.txt" 2>&1 || true
adb shell 'ps -A -o USER,PID,PPID,NAME,ARGS' >"$OUTPUT_DIR/processes.txt" 2>&1 || true
adb shell dumpsys meminfo "$PACKAGE" >"$OUTPUT_DIR/app_meminfo.txt" 2>&1 || true
adb shell dumpsys battery >"$OUTPUT_DIR/battery.txt" 2>&1 || true
adb shell dumpsys thermalservice >"$OUTPUT_DIR/thermalservice.txt" 2>&1 || true
adb exec-out run-as "$PACKAGE" cat files/lfm-server.log >"$OUTPUT_DIR/lfm-server.log" 2>&1 || true

if [[ -f "$APK" ]]; then
  unzip -l "$APK" | sed "1s|$APK|$(basename "$APK")|" >"$OUTPUT_DIR/apk_contents.txt"
  (cd "$(dirname "$APK")" && shasum -a 256 "$(basename "$APK")") >"$OUTPUT_DIR/apk.sha256"
fi

model_pid="$(adb shell pidof liblfmserver.so 2>/dev/null || true)"
model_pid="$(printf '%s' "$model_pid" | tr -d '\r' | awk '{print $1}')"
if [[ -n "$model_pid" ]]; then
  adb exec-out run-as "$PACKAGE" cat "/proc/$model_pid/status" >"$OUTPUT_DIR/model_status.txt" 2>&1 || true
  adb exec-out run-as "$PACKAGE" cat "/proc/$model_pid/cmdline" \
    | tr '\0' '\n' >"$OUTPUT_DIR/model_cmdline.txt" 2>&1 || true
fi

echo "$OUTPUT_DIR"
