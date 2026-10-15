#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE="ai.liquid.lfmdemo.dev"
ACTIVITY="$PACKAGE/.MainActivity"
REMOTE="/data/local/tmp/lfm-runtime.args"
RESTART=1

usage() {
  echo "usage: $0 [--no-restart] [--clear | -- ARG ...]" >&2
}

if [[ "${1:-}" == "--no-restart" ]]; then
  RESTART=0
  shift
fi

if [[ "$(adb get-state 2>/dev/null || true)" != "device" ]]; then
  echo "No authorized Android device." >&2
  exit 2
fi

if [[ "${1:-}" == "--clear" ]]; then
  adb shell "run-as '$PACKAGE' rm -f files/runtime.args"
  shift
elif [[ "${1:-}" == "--" ]]; then
  shift
  TEMP_FILE="$(mktemp)"
  trap 'rm -f "$TEMP_FILE"' EXIT
  printf '%s\n' "$@" >"$TEMP_FILE"
  adb push "$TEMP_FILE" "$REMOTE" >/dev/null
  adb shell "run-as '$PACKAGE' cp '$REMOTE' files/runtime.args"
  adb shell "rm -f '$REMOTE'"
  set --
else
  usage
  exit 2
fi

if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

if [[ "$RESTART" == "1" ]]; then
  adb shell am force-stop "$PACKAGE"
  adb shell am start -n "$ACTIVITY" >/dev/null
fi

echo "Runtime configuration updated."
