#!/usr/bin/env bash
# Unattended overnight on-device profiling: run every remaining phase in order,
# recording progress so an interrupted night can be resumed rather than redone.
#
# Each phase writes a `.phase-done` stamp. Re-running the script skips completed
# phases, so this is safe to restart after a USB drop, a reboot, or an
# orchestration-process restart.
#
# Phases:
#   B  main sweep            every candidate x every phrase, no playback
#   D  thread sweep          1/2/4/8 threads for the leading candidates
#   E  playback sweep        real AudioTrack first-audible and underruns
#   T  thermal soak          repeated long synthesis, watching for throttling
#   P  pull                  device results and WAVs to the host
set -uo pipefail

RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
PKG=ai.liquid.ttsbench
FILES=/sdcard/Android/data/$PKG/files
RESULTS=$FILES/results
STAMPS="$RUN_ROOT/device/stamps"
LOG="$RUN_ROOT/device/overnight.log"
mkdir -p "$STAMPS" "$RUN_ROOT/device"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

wait_for_device() {
  for _ in $(seq 1 120); do
    [[ "$(adb get-state 2>/dev/null || true)" == "device" ]] && return 0
    sleep 5
  done
  return 1
}

# The USB link on this device drops under sustained load; every phase boundary
# re-establishes it rather than assuming it survived.
recover_link() {
  if ! adb get-state </dev/null >/dev/null 2>&1; then
    say "  link down, resetting adb"
    adb kill-server >/dev/null 2>&1 || true; sleep 3
    adb start-server >/dev/null 2>&1 || true
    wait_for_device || { say "  device did not return"; return 1; }
  fi
  return 0
}

snapshot_thermal() {
  local label="$1"
  {
    echo "--- $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    adb shell dumpsys battery </dev/null 2>/dev/null | grep -E "level|temperature" | tr -d '\r'
    adb shell "cat /sys/class/thermal/thermal_zone0/temp" </dev/null 2>/dev/null | tr -d '\r'
    adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq" </dev/null 2>/dev/null | tr -d '\r'
    adb shell "cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq" </dev/null 2>/dev/null | tr -d '\r'
  } >> "$RUN_ROOT/device/thermal.log" 2>&1
}

run_phase() {
  local phase="$1"; shift
  local stamp="$STAMPS/$phase.phase-done"
  if [[ -f "$stamp" ]]; then say "phase $phase already done, skipping"; return 0; fi
  recover_link || return 1
  snapshot_thermal "before-$phase"
  say "phase $phase starting: $*"
  if "$@"; then
    snapshot_thermal "after-$phase"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$stamp"
    say "phase $phase DONE"
    return 0
  fi
  # The link drops often enough that one failure is usually the cable, not the
  # measurement. Retry a phase twice before giving up on it.
  say "phase $phase failed, retrying once"
  sleep 20
  recover_link || { say "phase $phase FAILED (device gone)"; return 1; }
  if "$@"; then
    snapshot_thermal "after-$phase"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$stamp"
    say "phase $phase DONE (on retry)"
    return 0
  fi
  say "phase $phase FAILED (continuing with the rest)"
  return 1
}

sweep() {
  local tag="$1"; shift
  env "$@" bash "$REPO/scripts/tts_research_20260726/device_sweep.sh" "$tag"
}

wait_existing_sweep() {
  local tag="$1" timeout_s="$2" start elapsed marker
  start="$(date +%s)"
  while true; do
    recover_link || return 1
    marker="$(adb shell "ls $RESULTS/autorun_${tag}_*.done 2>/dev/null" </dev/null 2>/dev/null | tr -d '\r' | head -1 || true)"
    if [[ -n "$marker" ]]; then
      say "  existing $tag sweep complete"
      adb shell "cat $marker" </dev/null 2>/dev/null || true
      return 0
    fi
    if ! adb shell "pidof $PKG" </dev/null >/dev/null 2>&1; then
      say "  existing $tag sweep stopped without a completion marker"
      return 1
    fi
    elapsed=$(( $(date +%s) - start ))
    if (( elapsed > timeout_s )); then
      say "  existing $tag sweep timed out after ${elapsed}s"
      return 1
    fi
    sleep 20
  done
}

# ---- B: main sweep (may already be running/complete) ----------------------
phase_main() {
  if adb shell "ls $RESULTS/autorun_main_*.done" </dev/null >/dev/null 2>&1; then
    say "  main sweep already complete"; return 0
  fi
  if adb shell "pidof $PKG" </dev/null >/dev/null 2>&1; then
    say "  main sweep already running, waiting"
    wait_existing_sweep main 14400
    return $?
  fi
  sweep main SWEEP_REPEATS=3 SWEEP_SAVE_WAV=true SWEEP_TIMEOUT_S=14400
}

# ---- D: thread sweep -------------------------------------------------------
# Only the candidates still in contention; sweeping ten engines across four
# thread counts would spend the night on configurations nobody will ship.
phase_threads() {
  sweep threads \
    SWEEP_REPEATS=3 \
    SWEEP_THREADS=1,2,4,8 \
    SWEEP_CANDIDATES=matcha_zh_baker,melotts_zh,piper_zh_huayan,supertonic3_s5,piper_ru_dmitri \
    SWEEP_PHRASES=clause,answer,long \
    SWEEP_TIMEOUT_S=14400
}

# ---- E: playback sweep -----------------------------------------------------
# play=true is the only way to get a real first-audible time and a real
# underrun count; the host could only simulate both.
phase_playback() {
  sweep playback \
    SWEEP_REPEATS=2 \
    SWEEP_PLAY=true \
    SWEEP_PHRASES=clause,answer,long \
    SWEEP_TIMEOUT_S=14400
}

# ---- T: thermal soak -------------------------------------------------------
# Sustained load on the leading candidates, to see whether RTF degrades as the
# SoC heats. A single cold measurement cannot answer that.
phase_soak() {
  sweep soak \
    SWEEP_REPEATS=25 \
    SWEEP_CANDIDATES=matcha_zh_baker,supertonic3_s5 \
    SWEEP_PHRASES=long \
    SWEEP_TIMEOUT_S=14400
}

# ---- P: pull ---------------------------------------------------------------
phase_pull() {
  mkdir -p "$RUN_ROOT/device/results"
  recover_link || return 1
  adb pull "$RESULTS" "$RUN_ROOT/device/" </dev/null >/dev/null 2>&1 || {
    say "  bulk pull failed, falling back to per-file"
    for f in $(adb shell "ls $RESULTS" </dev/null | tr -d '\r'); do
      adb pull "$RESULTS/$f" "$RUN_ROOT/device/results/" </dev/null >/dev/null 2>&1 || true
    done
  }
  say "  pulled $(find "$RUN_ROOT/device/results" -type f 2>/dev/null | wc -l | tr -d ' ') files"
}

# ---- R: re-measure what sweep B could not ---------------------------------
# Sweep B ran with a build whose memory numbers were stale and whose file check
# rejected Kokoro's comma-separated lexicon paths. Rather than leave holes,
# re-run Kokoro entirely and re-measure memory for every candidate on one
# representative phrase.
phase_remeasure() {
  sweep kokoro \
    SWEEP_REPEATS=3 \
    SWEEP_SAVE_WAV=true \
    SWEEP_CANDIDATES=kokoro_zh_fp32,kokoro_zh_int8 \
    SWEEP_TIMEOUT_S=7200 || return 1
  sweep memory \
    SWEEP_REPEATS=2 \
    SWEEP_PHRASES=clause,long \
    SWEEP_TIMEOUT_S=7200
}

# The fixed APK once failed to install because the link dropped during
# `adb install`, and the next phase then ran against the stale build and
# silently reproduced the bug it was meant to fix. Verify before measuring.
verify_apk() {
  local local_apk="$REPO/android-tts-bench/app/build/outputs/apk/debug/app-debug.apk"
  [[ -f "$local_apk" ]] || { say "no local APK to verify against"; return 0; }
  local want; want="$(md5 -q "$local_apk" 2>/dev/null || md5sum "$local_apk" | cut -d' ' -f1)"
  for attempt in 1 2 3; do
    recover_link || return 1
    local path got
    path="$(adb shell "pm path $PKG" </dev/null 2>/dev/null | tr -d '\r' | sed 's/package://')"
    got="$(adb shell "md5sum $path" </dev/null 2>/dev/null | tr -d '\r' | awk '{print $1}')"
    if [[ "$got" == "$want" ]]; then say "  APK verified ($want)"; return 0; fi
    say "  installed APK is stale ($got != $want), reinstalling (attempt $attempt)"
    adb install -r -d "$local_apk" </dev/null >/dev/null 2>&1 || true
    sleep 3
  done
  say "  COULD NOT INSTALL THE CURRENT APK - measurements would use a stale build"
  return 1
}

say "=== overnight orchestration starting ==="
verify_apk || { say "aborting: stale APK"; exit 2; }
run_phase R phase_remeasure
run_phase B phase_main
run_phase D phase_threads
run_phase E phase_playback
run_phase T phase_soak
run_phase P phase_pull
say "=== overnight orchestration finished ==="
touch "$STAMPS/ALL.phase-done"
