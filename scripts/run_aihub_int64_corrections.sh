#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
PYTHON="${LFM_AIHUB_PYTHON:-$LFM_DEMO_WORK_ROOT/venvs/aihub/bin/python}"
RUNNER="$SCRIPT_DIR/submit_component_aihub.py"
EXPORTS="$LFM_DEMO_WORK_ROOT/lfm-feasibility/exports"
DEVICE="QCS8550 (Proxy)"
DEVICE_OS="12"
STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/aihub/int64-corrections-$STAMP"
LOG="$OUTPUT_DIR/run.log"

mkdir -p "$OUTPUT_DIR"
exec >"$LOG" 2>&1

"$PYTHON" "$RUNNER" \
  --component depth-decoder \
  --device "$DEVICE" --device-os "$DEVICE_OS" --device-attribute framework:qnn \
  --model "$EXPORTS/depth_decoder_hidden1x2048.pt2" \
  --io "$EXPORTS/depth_decoder_hidden1x2048.npz" \
  --input hidden --output '#0=tokens' --output '#1=next_audio_embedding' \
  --runtime strict-npu --output-dir "$OUTPUT_DIR"

"$PYTHON" "$RUNNER" \
  --component detokenizer-t4 \
  --device "$DEVICE" --device-os "$DEVICE_OS" --device-attribute framework:qnn \
  --model "$EXPORTS/detok_neural_probe_codes_t4.pt2" \
  --io "$EXPORTS/detok_neural_probe_codes_t4.npz" \
  --input codes --output '#0=log_abs' --output '#1=angle' \
  --runtime strict-npu --output-dir "$OUTPUT_DIR"
