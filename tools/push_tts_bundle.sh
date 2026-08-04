#!/usr/bin/env bash
# Install the TTS bench APK and push the candidate model bundle to the phone.
#
# This path was used for the published device benchmark. It remains intentionally
# explicit because the transfer is large and must be verified before results are
# trusted.
#
# The bundle goes to the app's own external files directory, which needs no
# runtime storage permission and is removed when the app is uninstalled:
#   /sdcard/Android/data/ai.liquid.ttsbench/files/tts-eval/
#
# It does not touch the LFM demo app or its data.
set -Eeuo pipefail

PKG="ai.liquid.ttsbench"
REMOTE_ROOT="/sdcard/Android/data/${PKG}/files/tts-eval"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APK="${TTS_BENCH_APK:-$REPO_ROOT/android-tts-bench/app/build/outputs/apk/debug/app-debug.apk}"
BUNDLE="${TTS_BENCH_BUNDLE:-}"
ADB="${ADB:-adb}"

usage() {
  cat <<EOF
usage: TTS_BENCH_BUNDLE=/path/to/device-bundle $0 [--apk-only|--bundle-only]

  TTS_BENCH_BUNDLE  staged bundle directory (from stage_device_bundle.py)
  TTS_BENCH_APK     override the APK path
  ADB               override the adb binary
EOF
}

MODE="all"
case "${1:-}" in
  --apk-only) MODE="apk" ;;
  --bundle-only) MODE="bundle" ;;
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) usage; exit 2 ;;
esac

if ! command -v "$ADB" >/dev/null 2>&1; then
  echo "adb not found. Set ADB=/path/to/adb." >&2
  exit 2
fi

DEVICES="$("$ADB" devices | awk 'NR>1 && $2=="device" {print $1}')"
COUNT="$(printf '%s' "$DEVICES" | grep -c . || true)"
if [[ "$COUNT" -eq 0 ]]; then
  echo "No authorised device. Connect the phone, enable USB debugging, and accept the prompt." >&2
  exit 3
fi
if [[ "$COUNT" -gt 1 ]]; then
  echo "More than one device attached; set ANDROID_SERIAL to choose one:" >&2
  printf '  %s\n' $DEVICES >&2
  exit 3
fi
echo "device: $DEVICES"
"$ADB" shell getprop ro.product.model
"$ADB" shell getprop ro.build.version.release

if [[ "$MODE" != "bundle" ]]; then
  [[ -f "$APK" ]] || { echo "APK not found: $APK (run tools/build_tts_bench.sh)" >&2; exit 2; }
  echo "installing $(basename "$APK") ($(du -h "$APK" | cut -f1))"
  # -r keeps existing app data; -d allows a downgrade during iteration.
  "$ADB" install -r -d "$APK"
fi

if [[ "$MODE" != "apk" ]]; then
  [[ -n "$BUNDLE" && -d "$BUNDLE" ]] || {
    echo "Set TTS_BENCH_BUNDLE to the staged bundle directory." >&2; exit 2; }
  [[ -f "$BUNDLE/candidates.json" ]] || {
    echo "$BUNDLE has no candidates.json - stage it with stage_device_bundle.py first." >&2; exit 2; }

  SIZE_KB="$(du -sk "$BUNDLE" | cut -f1)"
  FREE_KB="$("$ADB" shell df /sdcard | awk 'NR==2 {print $4}' | tr -d '\r')"
  echo "bundle: $((SIZE_KB / 1024)) MiB   free on /sdcard: $((FREE_KB / 1024)) MiB"
  if [[ -n "$FREE_KB" && "$FREE_KB" -lt "$((SIZE_KB + 262144))" ]]; then
    echo "Not enough free space (want the bundle plus 256 MiB headroom)." >&2
    exit 4
  fi

  # Creating the directory via run-as is not possible on a non-debuggable
  # build, so the app is launched once to create its files dir first.
  "$ADB" shell am start -n "${PKG}/.MainActivity" >/dev/null 2>&1 || true
  sleep 2
  "$ADB" shell mkdir -p "$REMOTE_ROOT"

  echo "pushing bundle (this takes several minutes over USB 2)..."
  "$ADB" push "$BUNDLE/." "$REMOTE_ROOT/"

  echo "verifying..."
  REMOTE_COUNT="$("$ADB" shell "find $REMOTE_ROOT -type f | wc -l" | tr -d '\r')"
  LOCAL_COUNT="$(find "$BUNDLE" -type f | wc -l | tr -d ' ')"
  echo "  files on device: $REMOTE_COUNT   local: $LOCAL_COUNT"
  if [[ "$REMOTE_COUNT" != "$LOCAL_COUNT" ]]; then
    echo "  WARNING: file counts differ; re-run the push before trusting results." >&2
  fi
  "$ADB" shell "ls -l $REMOTE_ROOT/candidates.json"
fi

cat <<EOF

Done. Open "LFM TTS Bench" on the phone.

  Speak A / Speak B   play one candidate
  A / B               play A then B back to back for comparison
  Run all (silent)    sweep every candidate for latency and stability
  Export JSON         writes to $REMOTE_ROOT/../results/

Pull results with:
  $ADB pull /sdcard/Android/data/${PKG}/files/results ./device-results
EOF
