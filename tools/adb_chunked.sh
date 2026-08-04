#!/usr/bin/env bash
# Move files over an adb link that cannot sustain a bulk transfer.
#
# This device's USB link dies partway through anything large: a 64 KiB read
# completes, a 512 KiB read returns 79 002 of 524 288 bytes and takes the device
# offline. `adb pull`/`adb push` are therefore unusable here, but short
# `adb shell` commands keep working. This moves data in small blocks, verifies
# each one, and re-establishes the link when it drops.
#
#   adb_chunked.sh pull  <remote> <local>
#   adb_chunked.sh push  <local>  <remote>
#
# BLOCK_BYTES tunes the block size (default 65536, the largest observed to
# survive). Both directions verify the finished file by md5 against the source.
set -uo pipefail

ADB="${ADB:-adb}"
BLOCK="${BLOCK_BYTES:-65536}"
MAX_TRIES="${CHUNK_MAX_TRIES:-8}"

wait_link() {
  for _ in $(seq 1 30); do
    [[ "$("$ADB" get-state 2>/dev/null || true)" == "device" ]] && return 0
    "$ADB" kill-server >/dev/null 2>&1 || true; sleep 2
    "$ADB" start-server >/dev/null 2>&1 || true; sleep 4
    "$ADB" reconnect >/dev/null 2>&1 || true; sleep 2
  done
  return 1
}

remote_size() { "$ADB" shell "stat -c%s '$1'" </dev/null 2>/dev/null | tr -d '\r'; }
remote_md5()  { "$ADB" shell "md5sum '$1'" </dev/null 2>/dev/null | tr -d '\r' | awk '{print $1}'; }
local_md5()   { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }

do_pull() {
  local remote="$1" local_path="$2"
  wait_link || { echo "no device" >&2; return 3; }
  local size want
  size="$(remote_size "$remote")"; want="$(remote_md5 "$remote")"
  [[ -n "$size" ]] || { echo "cannot stat $remote" >&2; return 4; }
  local blocks=$(( (size + BLOCK - 1) / BLOCK ))
  echo "pull $remote -> $local_path ($size bytes, $blocks x $BLOCK)"

  local tmp="${local_path}.part"
  mkdir -p "$(dirname "$local_path")"
  : > "$tmp"
  local i
  for (( i = 0; i < blocks; i++ )); do
    local got=0 try
    for (( try = 1; try <= MAX_TRIES; try++ )); do
      wait_link || return 3
      # Always rewind to this block's offset first: a failed read still appends
      # its partial bytes, and without truncating, the next attempt appends on
      # top of them and the file silently desynchronises.
      /usr/bin/env python3 -c "import sys;f=open(sys.argv[1],'r+b');f.truncate(int(sys.argv[2]));f.close()" \
        "$tmp" "$(( i * BLOCK ))" 2>/dev/null
      # base64 keeps the payload inside the text channel that still works;
      # a raw exec-out of the same size trips the same failure as adb pull.
      "$ADB" exec-out "dd if='$remote' bs=$BLOCK count=1 skip=$i 2>/dev/null | base64" \
        </dev/null 2>/dev/null | tr -d '\r\n' | base64 -d >> "$tmp" 2>/dev/null
      local now expect
      now=$(stat -f%z "$tmp" 2>/dev/null || stat -c%s "$tmp")
      expect=$(( (i + 1) * BLOCK )); (( expect > size )) && expect=$size
      if [[ "$now" == "$expect" ]]; then got=1; break; fi
      sleep 1
    done
    (( got )) || { echo "block $i failed after $MAX_TRIES tries" >&2; return 5; }
    (( i % 16 == 0 )) && printf '  %d/%d\n' "$i" "$blocks"
  done

  mv "$tmp" "$local_path"
  local have; have="$(local_md5 "$local_path")"
  if [[ -n "$want" && "$have" != "$want" ]]; then
    echo "md5 mismatch: $have != $want" >&2; return 6
  fi
  echo "pulled and verified ($have)"
}

do_push() {
  local local_path="$1" remote="$2"
  [[ -f "$local_path" ]] || { echo "no such file: $local_path" >&2; return 2; }
  wait_link || { echo "no device" >&2; return 3; }
  local size want
  size=$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path")
  want="$(local_md5 "$local_path")"
  local blocks=$(( (size + BLOCK - 1) / BLOCK ))
  echo "push $local_path -> $remote ($size bytes, $blocks x $BLOCK)"

  "$ADB" shell "mkdir -p '$(dirname "$remote")' && rm -f '$remote'" </dev/null >/dev/null 2>&1
  local i
  for (( i = 0; i < blocks; i++ )); do
    local b64
    b64=$(dd if="$local_path" bs=$BLOCK count=1 skip=$i 2>/dev/null | base64 | tr -d '\n')
    local got=0 try
    for (( try = 1; try <= MAX_TRIES; try++ )); do
      wait_link || return 3
      if "$ADB" shell "echo '$b64' | base64 -d >> '$remote'" </dev/null >/dev/null 2>&1; then
        local now expect
        now="$(remote_size "$remote")"
        expect=$(( (i + 1) * BLOCK )); (( expect > size )) && expect=$size
        if [[ "$now" == "$expect" ]]; then got=1; break; fi
        "$ADB" shell "truncate -s $(( i * BLOCK )) '$remote'" </dev/null >/dev/null 2>&1
      fi
      sleep 1
    done
    (( got )) || { echo "block $i failed after $MAX_TRIES tries" >&2; return 5; }
    (( i % 16 == 0 )) && printf '  %d/%d\n' "$i" "$blocks"
  done

  local have; have="$(remote_md5 "$remote")"
  if [[ "$have" != "$want" ]]; then echo "md5 mismatch: $have != $want" >&2; return 6; fi
  echo "pushed and verified ($have)"
}

case "${1:?usage: adb_chunked.sh pull|push src dst}" in
  pull) do_pull "${2:?remote}" "${3:?local}" ;;
  push) do_push "${2:?local}" "${3:?remote}" ;;
  *) echo "usage: adb_chunked.sh pull|push src dst" >&2; exit 2 ;;
esac
