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
OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/aihub/component-matrix-$STAMP"
LOG="$OUTPUT_DIR/run.log"
STATUS_TSV="$OUTPUT_DIR/status.tsv"

mkdir -p "$OUTPUT_DIR"
exec >"$LOG" 2>&1
printf 'component\tstrict_rc\tdiagnostic_rc\n' >"$STATUS_TSV"

run_component() {
  local component="$1"
  local model="$2"
  local io="$3"
  shift 3
  local strict_rc diagnostic_rc

  printf '\n===== %s strict-npu =====\n' "$component"
  set +e
  "$PYTHON" "$RUNNER" \
    --component "$component" \
    --device "$DEVICE" \
    --device-os "$DEVICE_OS" \
    --device-attribute framework:qnn \
    --model "$model" \
    --io "$io" \
    --output-dir "$OUTPUT_DIR" \
    --runtime strict-npu \
    "$@"
  strict_rc=$?
  set -e

  diagnostic_rc=0
  if [[ "$strict_rc" -ne 0 ]]; then
    printf '\n===== %s ort-diagnostic =====\n' "$component"
    set +e
    "$PYTHON" "$RUNNER" \
      --component "$component" \
      --device "$DEVICE" \
      --device-os "$DEVICE_OS" \
      --device-attribute framework:qnn \
      --model "$model" \
      --io "$io" \
      --output-dir "$OUTPUT_DIR" \
      --runtime ort-diagnostic \
      "$@"
    diagnostic_rc=$?
    set -e
  fi
  printf '%s\t%s\t%s\n' "$component" "$strict_rc" "$diagnostic_rc" >>"$STATUS_TSV"
}

run_component backbone-conv-prefill \
  "$EXPORTS/lfm2_layer0_conv_seq16.pt2" \
  "$EXPORTS/lfm2_layer0_conv_seq16.npz" \
  --input hidden_states --output '#0=output'

run_component backbone-attention-prefill \
  "$EXPORTS/lfm2_layer2_attention_seq16.pt2" \
  "$EXPORTS/lfm2_layer2_attention_seq16.npz" \
  --input hidden_states --output '#0=output'

run_component backbone-conv-cached-decode \
  "$EXPORTS/lfm2_layer0_conv_decode_cache.pt2" \
  "$EXPORTS/lfm2_layer0_conv_decode_cache.npz" \
  --input hidden_states --input conv_cache \
  --output '#0=output' --output '#1=updated_conv_cache'

run_component backbone-attention-cached-decode \
  "$EXPORTS/lfm2_layer2_attention_decode_kv16_past8.pt2" \
  "$EXPORTS/lfm2_layer2_attention_decode_kv16_past8.npz" \
  --input hidden_states --input key_cache --input value_cache \
  --output '#0=output' --output '#1=updated_key_cache' --output '#2=updated_value_cache'

run_component depth-decoder \
  "$EXPORTS/depth_decoder_hidden1x2048.pt2" \
  "$EXPORTS/depth_decoder_hidden1x2048.npz" \
  --input hidden --output '#0=tokens' --output '#1=next_audio_embedding'

run_component detokenizer-t4 \
  "$EXPORTS/detok_neural_probe_codes_t4.pt2" \
  "$EXPORTS/detok_neural_probe_codes_t4.npz" \
  --input codes --output '#0=log_abs' --output '#1=angle'

run_component detokenizer-t8 \
  "$EXPORTS/detok_neural_probe_codes_t8.pt2" \
  "$EXPORTS/detok_neural_probe_codes_t8.npz" \
  --input codes --output '#0=log_abs' --output '#1=angle'

printf '\nComponent matrix finished.\n'
