#!/usr/bin/env bash
# Resumable, verified pull of the bench app's results from the device.
#
# Mirrors tools/push_bundle_resumable.sh for the same reason: this device's adb
# link drops under sustained load, so a single bulk `adb pull` gets part of the
# way and reports failure. This pulls file by file, verifies by size, retries,
# and skips anything already local and correct — so it can be re-run at any
# time, including while a sweep is still writing.
#
# usage: DEST=/path/to/device tools/pull_device_results.sh
set -uo pipefail

PKG=ai.liquid.ttsbench
REMOTE=/sdcard/Android/data/$PKG/files/results
ADB="${ADB:-adb}"
DEST="${DEST:?set DEST}"
RETRIES="${PULL_RETRIES:-4}"

wait_link() {
  for _ in $(seq 1 40); do
    [[ "$("$ADB" get-state 2>/dev/null || true)" == "device" ]] && return 0
    "$ADB" kill-server >/dev/null 2>&1 || true; sleep 3
    "$ADB" start-server >/dev/null 2>&1 || true; sleep 3
  done
  return 1
}

wait_link || { echo "no device" >&2; exit 3; }
mkdir -p "$DEST/results" "$DEST/wav"

listing="$("$ADB" shell "find $REMOTE -type f -exec stat -c '%s %n' {} +" </dev/null 2>/dev/null | tr -d '\r')"
[[ -n "$listing" ]] || { echo "could not list remote results" >&2; exit 4; }

total=0; got=0; skipped=0; failed=0
while IFS= read -r line <&3; do
  [[ -z "$line" ]] && continue
  size="${line%% *}"
  path="${line#* }"
  rel="${path#"$REMOTE"/}"
  # WAVs go to wav/, everything else to results/, matching the analysis scripts.
  case "$rel" in
    wav/*) local_path="$DEST/$rel" ;;
    *)     local_path="$DEST/results/$rel" ;;
  esac
  total=$((total + 1))

  if [[ -f "$local_path" ]]; then
    have=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path")
    if [[ "$have" == "$size" ]]; then skipped=$((skipped + 1)); continue; fi
  fi

  mkdir -p "$(dirname "$local_path")"
  ok=0
  for attempt in $(seq 1 "$RETRIES"); do
    if ! "$ADB" get-state </dev/null >/dev/null 2>&1; then wait_link || break; fi
    if "$ADB" pull "$path" "$local_path" </dev/null >/dev/null 2>&1; then
      have=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path" 2>/dev/null || echo 0)
      [[ "$have" == "$size" ]] && { ok=1; break; }
    fi
    sleep 2
  done
  if [[ "$ok" == "1" ]]; then got=$((got + 1)); else failed=$((failed + 1)); echo "  FAILED: $rel" >&2; fi
done 3<<< "$listing"

echo "remote files: $total   pulled: $got   already local: $skipped   failed: $failed"
[[ "$failed" -eq 0 ]]
