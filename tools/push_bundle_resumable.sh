#!/usr/bin/env bash
# Resumable, verified push of the TTS model bundle.
#
# A single `adb push` of the whole 1.5 GiB tree failed at 59% with
# "failed to read copy response: EOF" and knocked the device offline. Large
# single transfers over this link are not reliable, so this pushes one file at
# a time, verifies each by size, retries, and skips anything already correct —
# so an interrupted run is resumed rather than restarted.
#
# usage: TTS_BENCH_BUNDLE=/path/to/device-bundle tools/push_bundle_resumable.sh
set -Eeuo pipefail

PKG="ai.liquid.ttsbench"
REMOTE="/sdcard/Android/data/${PKG}/files/tts-eval"
ADB="${ADB:-adb}"
BUNDLE="${TTS_BENCH_BUNDLE:?set TTS_BENCH_BUNDLE}"
MAX_RETRIES="${PUSH_MAX_RETRIES:-4}"

wait_for_device() {
  for _ in $(seq 1 60); do
    [[ "$("$ADB" get-state 2>/dev/null || true)" == "device" ]] && return 0
    sleep 5
  done
  return 1
}

reset_adb() {
  echo "  ! link lost, resetting adb" >&2
  "$ADB" kill-server >/dev/null 2>&1 || true
  sleep 3
  "$ADB" start-server >/dev/null 2>&1 || true
  wait_for_device || { echo "device did not come back" >&2; exit 3; }
}

wait_for_device || { echo "no device" >&2; exit 3; }
"$ADB" shell mkdir -p "$REMOTE"

# One `ls -lR`-style pass instead of a stat per file: an adb shell round trip is
# ~50 ms and the bundle has hundreds of files.
echo "reading remote sizes..."
REMOTE_SIZES="$(mktemp)"
trap 'rm -f "$REMOTE_SIZES"' EXIT
"$ADB" shell "find $REMOTE -type f -printf '%s %P\n' 2>/dev/null || find $REMOTE -type f -exec stat -c '%s %n' {} +" \
  2>/dev/null | tr -d '\r' > "$REMOTE_SIZES" || true

remote_size_of() {
  local rel="$1"
  awk -v want="$rel" '{ $1=$1; size=$1; $1=""; sub(/^ /,""); if ($0==want || $0=="'"$REMOTE"'/"want) { print size; exit } }' \
    "$REMOTE_SIZES" 2>/dev/null | head -1
}

total=0; pushed=0; skipped=0; failed=0
while IFS= read -r -d '' local_file <&3; do
  rel="${local_file#"$BUNDLE"/}"
  total=$((total + 1))
  local_size=$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file")

  remote_size="$(awk -v want="$rel" '{ size=$1; sub(/^[0-9]+ /, ""); if ($0 == want) { print size; exit } }' "$REMOTE_SIZES")"
  if [[ -n "${remote_size:-}" && "$remote_size" == "$local_size" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  ok=0
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    if ! "$ADB" get-state </dev/null >/dev/null 2>&1; then reset_adb; fi
    "$ADB" shell mkdir -p "$REMOTE/$(dirname "$rel")" </dev/null >/dev/null 2>&1 || true
    if "$ADB" push "$local_file" "$REMOTE/$rel" </dev/null >/dev/null 2>&1; then
      actual="$("$ADB" shell "stat -c%s '$REMOTE/$rel' 2>/dev/null" </dev/null | tr -d '\r')"
      if [[ "$actual" == "$local_size" ]]; then ok=1; break; fi
      echo "  size mismatch for $rel ($actual != $local_size), retry $attempt" >&2
    else
      echo "  push failed for $rel, retry $attempt" >&2
      reset_adb
    fi
    sleep 2
  done

  if [[ "$ok" == "1" ]]; then
    pushed=$((pushed + 1))
    printf '  [%4d] %s (%s)\n' "$pushed" "$rel" "$(numfmt --to=iec "$local_size" 2>/dev/null || echo "${local_size}B")"
  else
    failed=$((failed + 1))
    echo "  FAILED: $rel" >&2
  fi
done 3< <(find "$BUNDLE" -type f -print0)

echo
echo "files: $total   pushed: $pushed   already correct: $skipped   failed: $failed"
[[ "$failed" -eq 0 ]] || exit 6

REMOTE_COUNT="$("$ADB" shell "find $REMOTE -type f | wc -l" | tr -d '\r')"
echo "remote file count: $REMOTE_COUNT (local $total)"
[[ "$REMOTE_COUNT" == "$total" ]] || { echo "count mismatch" >&2; exit 7; }
echo "bundle verified"
