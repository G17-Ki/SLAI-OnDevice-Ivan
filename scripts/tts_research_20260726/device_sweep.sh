#!/usr/bin/env bash
# Drive an unattended on-device TTS sweep and wait for it to finish.
#
# The bench app's autorun mode is launched with `am start` extras, writes one
# JSON line per measurement, and drops a `.done` marker when finished. This
# script polls that marker rather than guessing at a duration, and fails loudly
# if the app dies rather than reporting a partial sweep as complete.
#
# usage: device_sweep.sh TAG [extra am args...]
#   TAG              names the output files on the device
#   SWEEP_REPEATS    repeats per phrase (default 3)
#   SWEEP_PLAY       true to play audio and measure real underruns (default false)
#   SWEEP_SAVE_WAV   true to write one WAV per candidate/phrase (default false)
#   SWEEP_THREADS    comma-separated thread counts, "0" = use candidates.json
#   SWEEP_CANDIDATES comma-separated ids, or "all"
#   SWEEP_PHRASES    comma-separated phrase ids, or "all"
#   SWEEP_TIMEOUT_S  give up after this long (default 14400)
set -Eeuo pipefail

PKG="ai.liquid.ttsbench"
FILES="/sdcard/Android/data/${PKG}/files"
RESULTS="${FILES}/results"
ADB="${ADB:-adb}"

TAG="${1:?usage: device_sweep.sh TAG}"
REPEATS="${SWEEP_REPEATS:-3}"
PLAY="${SWEEP_PLAY:-false}"
SAVE_WAV="${SWEEP_SAVE_WAV:-false}"
THREADS="${SWEEP_THREADS:-0}"
CANDIDATES="${SWEEP_CANDIDATES:-all}"
PHRASES="${SWEEP_PHRASES:-all}"
TIMEOUT_S="${SWEEP_TIMEOUT_S:-14400}"

# The USB link on this device drops repeatedly under sustained load. Every
# host-side adb call must tolerate that: the sweep itself runs inside the app
# and keeps going regardless, so a dropped link is a host inconvenience, not a
# lost measurement. Aborting here previously killed two whole phases.
wait_link() {
  for _ in $(seq 1 60); do
    [[ "$("$ADB" get-state 2>/dev/null || true)" == "device" ]] && return 0
    "$ADB" kill-server >/dev/null 2>&1 || true
    sleep 3
    "$ADB" start-server >/dev/null 2>&1 || true
    sleep 3
  done
  return 1
}
wait_link || { echo "no device after retries" >&2; exit 3; }

# A stale marker from a previous sweep with the same tag would make this script
# return immediately with someone else's results.
"$ADB" shell "rm -f ${RESULTS}/autorun_${TAG}_*.done" >/dev/null 2>&1 || true

"$ADB" shell am force-stop "$PKG"
sleep 1
"$ADB" shell am start -n "${PKG}/.MainActivity" \
  --es mode autorun \
  --es tag "$TAG" \
  --ei repeats "$REPEATS" \
  --ez play "$PLAY" \
  --ez save_wav "$SAVE_WAV" \
  --es threads "$THREADS" \
  --es candidates "$CANDIDATES" \
  --es phrases "$PHRASES" >/dev/null

echo "sweep '$TAG' launched (repeats=$REPEATS play=$PLAY threads=$THREADS candidates=$CANDIDATES)"

start=$(date +%s)
last_done=-1
stall=0
while true; do
  sleep 20
  elapsed=$(( $(date +%s) - start ))
  if [[ "$elapsed" -gt "$TIMEOUT_S" ]]; then
    echo "TIMEOUT after ${elapsed}s" >&2
    "$ADB" shell "cat ${RESULTS}/autorun_status.json" 2>/dev/null || true
    exit 4
  fi

  if ! "$ADB" get-state </dev/null >/dev/null 2>&1; then wait_link || true; fi
  marker="$("$ADB" shell "ls ${RESULTS}/autorun_${TAG}_*.done 2>/dev/null" </dev/null 2>/dev/null | tr -d '\r' | head -1 || true)"
  if [[ -n "$marker" ]]; then
    echo "sweep complete after ${elapsed}s"
    "$ADB" shell "cat $marker"
    exit 0
  fi

  # Progress and liveness. If the process is gone and no marker exists, the app
  # died; saying so beats waiting out the full timeout.
  status="$("$ADB" shell "cat ${RESULTS}/autorun_status.json 2>/dev/null" </dev/null | tr -d '\r' || true)"
  completed="$(printf '%s' "$status" | sed -n 's/.*"completed_units": *\([0-9]*\).*/\1/p' | head -1 || true)"
  planned="$(printf '%s' "$status" | sed -n 's/.*"planned_units": *\([0-9]*\).*/\1/p' | head -1 || true)"
  current="$(printf '%s' "$status" | sed -n 's/.*"current": *"\([^"]*\)".*/\1/p' | head -1 || true)"
  [[ -n "${completed:-}" ]] && echo "  [${elapsed}s] ${completed}/${planned:-?}  ${current:-}"

  if ! "$ADB" shell "pidof $PKG" </dev/null >/dev/null 2>&1; then
    if [[ "${completed:-0}" == "${last_done:-0}" ]]; then
      stall=$((stall + 1))
    else
      stall=0
    fi
    # Six consecutive stalled polls (~2 min) before calling it a crash: a
    # dropped link makes pidof fail too, and that is not the app dying.
    if [[ "$stall" -ge 6 ]]; then
      echo "app gone and no progress for ~2 min — treating as a crash" >&2
      "$ADB" shell "cat ${RESULTS}/autorun_status.json" </dev/null 2>/dev/null || true
      exit 5
    fi
  else
    stall=0
  fi
  last_done="${completed:-0}"
done
