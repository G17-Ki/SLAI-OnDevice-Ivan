#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
PYTHON="${LFM_AIHUB_PYTHON:-$LFM_DEMO_WORK_ROOT/venvs/aihub/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/aihub/detok-t8-correction-$STAMP"
mkdir -p "$OUTPUT_DIR"
exec >"$OUTPUT_DIR/run.log" 2>&1

"$PYTHON" "$SCRIPT_DIR/submit_component_aihub.py" \
  --component detokenizer-t8 \
  --device "QCS8550 (Proxy)" --device-os 12 --device-attribute framework:qnn \
  --model "$LFM_DEMO_WORK_ROOT/lfm-feasibility/exports/detok_neural_probe_codes_t8.pt2" \
  --io "$LFM_DEMO_WORK_ROOT/lfm-feasibility/exports/detok_neural_probe_codes_t8.npz" \
  --input codes --output '#0=log_abs' --output '#1=angle' \
  --runtime strict-npu --output-dir "$OUTPUT_DIR"
