#!/usr/bin/env bash
set -Eeuo pipefail

ADB_BIN="${ADB:-adb}"
SERIAL=""
APK_SOURCE=""
OFFLINE=0
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/tts"
PACKAGE="com.reecedunn.espeak"
VERSION_CODE="22"
APK_NAME="com.reecedunn.espeak_22.apk"
APK_URL="https://f-droid.org/repo/com.reecedunn.espeak_22.apk"
APK_BYTES="10446659"
APK_SHA256="0a7822fec54d7f7ae759ffedc9afeaa77f809587b171fb2101998787821ec790"

usage() {
  printf '%s\n' \
    'Usage: scripts/install_espeak_phone.sh [--serial SERIAL] [--adb PATH] [--apk PATH] [--offline]' \
    '' \
    'Installs the exact audited F-Droid eSpeak 1.52.0 package used by the demo.' \
    'The APK is cached outside the repository and verified by byte count and SHA-256.'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) SERIAL="$2"; shift 2 ;;
    --adb) ADB_BIN="$2"; shift 2 ;;
    --apk) APK_SOURCE="$2"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v "$ADB_BIN" >/dev/null || { printf 'adb not found: %s\n' "$ADB_BIN" >&2; exit 2; }
command -v shasum >/dev/null || { printf '%s\n' 'shasum is required.' >&2; exit 2; }

verify_apk() {
  local apk="$1"
  [[ -f "$apk" ]] || return 1
  local bytes digest
  bytes="$(stat -f '%z' "$apk" 2>/dev/null || stat -c '%s' "$apk")"
  digest="$(shasum -a 256 "$apk" | awk 'NR == 1 {print $1}')"
  [[ "$bytes" == "$APK_BYTES" && "$digest" == "$APK_SHA256" ]]
}

mkdir -p "$CACHE_ROOT"
APK_PATH="$CACHE_ROOT/$APK_NAME"
if [[ -n "$APK_SOURCE" ]]; then
  verify_apk "$APK_SOURCE" || {
    printf '%s\n' 'The supplied eSpeak APK does not match the pinned release.' >&2
    exit 3
  }
  APK_PATH="$APK_SOURCE"
elif ! verify_apk "$APK_PATH"; then
  [[ "$OFFLINE" -eq 0 ]] || {
    printf 'Pinned APK is absent or invalid in offline mode: %s\n' "$APK_PATH" >&2
    exit 3
  }
  temporary="$APK_PATH.partial.$$"
  trap 'rm -f "$temporary"' EXIT
  curl --fail --location --retry 4 --retry-all-errors --output "$temporary" "$APK_URL"
  verify_apk "$temporary" || {
    printf '%s\n' 'Downloaded eSpeak APK failed integrity verification.' >&2
    exit 3
  }
  mv "$temporary" "$APK_PATH"
  trap - EXIT
fi

if [[ -z "$SERIAL" ]]; then
  devices="$($ADB_BIN devices | awk 'NR > 1 && $2 == "device" {print $1}')"
  count="$(printf '%s\n' "$devices" | awk 'NF {count++} END {print count + 0}')"
  [[ "$count" -eq 1 ]] || {
    printf 'Expected exactly one connected adb device, found %s; use --serial.\n' "$count" >&2
    exit 4
  }
  SERIAL="$(printf '%s\n' "$devices" | awk 'NF {print; exit}')"
fi

ADB_CMD=("$ADB_BIN" -s "$SERIAL")
"${ADB_CMD[@]}" install -r "$APK_PATH"
PACKAGE_FACTS="$("${ADB_CMD[@]}" shell dumpsys package "$PACKAGE" | tr -d '\r')"
printf '%s\n' "$PACKAGE_FACTS" | grep -q "versionCode=$VERSION_CODE" || {
  printf 'Installed package did not report expected versionCode %s.\n' "$VERSION_CODE" >&2
  exit 5
}
printf 'Installed %s versionCode %s on %s.\n' "$PACKAGE" "$VERSION_CODE" "$SERIAL"
printf 'APK SHA-256: %s\n' "$APK_SHA256"
