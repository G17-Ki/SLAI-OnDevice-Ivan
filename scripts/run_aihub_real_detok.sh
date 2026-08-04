#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
PYTHON="${LFM_AIHUB_PYTHON:-$LFM_DEMO_WORK_ROOT/venvs/aihub/bin/python}"
RUNNER="$SCRIPT_DIR/submit_component_aihub.py"
DEVICE="QCS8550 (Proxy)"
DEVICE_OS="12"
STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/aihub/real-detok-$STAMP"
PROBE_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/real-detok-probes"
LOG="$OUTPUT_DIR/run.log"
mkdir -p "$OUTPUT_DIR"
exec >"$LOG" 2>&1

status=0
for frames in 4 8; do
  "$PYTHON" "$RUNNER" \
    --component "detokenizer-real-t${frames}" \
    --device "$DEVICE" \
    --device-os "$DEVICE_OS" \
    --device-attribute framework:qnn \
    --model "$LFM_DEMO_WORK_ROOT/lfm-feasibility/exports/detok_neural_probe_codes_t${frames}.pt2" \
    --io "$PROBE_DIR/detok_real_turn1_t${frames}.npz" \
    --input codes \
    --output '#0=log_abs' \
    --output '#1=angle' \
    --runtime strict-npu \
    --output-dir "$OUTPUT_DIR" || status=$?
done
exit "$status"
