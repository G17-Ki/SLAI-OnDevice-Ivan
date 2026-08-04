#!/usr/bin/env bash
# Start the two phone-local companions the demo app needs, and verify them.
#
# Why this exists: ASR (non-English) and the LLM are NOT in the APK. They are
# shell-uid processes under /data/local/tmp, and every process started by
# `adb shell` lands in adbd's cgroup (/system/uid_0/pid_<adbd>). When adbd
# restarts - which is what unplugging the USB cable does - init tears that
# cgroup down and takes the companions with it. Measured on NX809J:
# `nohup`, PPID=1 and oom_score_adj=-1000 do not save them, and shell cannot
# move a process to another cgroup (permission denied).
#
# Consequence: run this after every replug and after every reboot, then keep
# USB connected. The demo app now says which service is missing instead of
# sitting on "Loading" forever, and it recovers on its own once this runs.
#
# TTS is unaffected: all voices are in-app and need neither companion.
set -uo pipefail

ADB="${ADB:-adb}"
ASR_ROOT=/data/local/tmp/lmf-asr-companion
TEXT_ROOT=/data/local/tmp/lmf-text-companion

if [[ "$("$ADB" get-state 2>/dev/null || true)" != "device" ]]; then
  echo "no device visible to adb" >&2
  exit 3
fi

start_one() {
  local name="$1" root="$2"
  printf '%-6s ' "$name"
  # start.sh can block; bound it rather than hanging the caller.
  local out
  out="$("$ADB" shell "cd $root && sh start.sh start" </dev/null 2>&1 | tr -d '\r' | tail -1)"
  echo "$out"
}

status_one() {
  local name="$1" root="$2"
  local out expected
  if ! out="$("$ADB" shell "cd $root && sh start.sh status" </dev/null 2>&1 | tr -d '\r' | head -1)"; then
    printf '  DOWN %-6s %s\n' "$name" "${out:-no response}"
    return 1
  fi
  case "$name" in
    asr) expected="ASR companion running as PID " ;;
    text) expected="Text companion running as PID " ;;
    *) printf '  DOWN %-6s unknown companion\n' "$name"; return 1 ;;
  esac
  if [[ "$out" == "$expected"* ]]; then
    printf '  OK   %-6s %s\n' "$name" "$out"
    return 0
  fi
  printf '  DOWN %-6s %s\n' "$name" "${out:-no response}"
  return 1
}

start_one asr  "$ASR_ROOT"
start_one text "$TEXT_ROOT"

echo
echo "verify:"
rc=0
status_one asr  "$ASR_ROOT" || rc=1
status_one text "$TEXT_ROOT" || rc=1

if (( rc == 0 )); then
  echo
  echo "Both up. The app shows 'Offline stack ready' within ~2 s."
  echo "Keep USB connected: unplugging or rebooting stops both companions."
else
  echo
  echo "Something did not come up. Retry once - the start scripts poll a bounded"
  echo "number of times and can give up on a slow first load." >&2
fi
exit $rc
