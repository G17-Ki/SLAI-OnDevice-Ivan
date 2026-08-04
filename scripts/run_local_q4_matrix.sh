#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT_FROM_ENV="${LFM_DEMO_WORK_ROOT:-}"
source "$SCRIPT_DIR/path_config.sh"
ROOT="${1:-$LFM_DEMO_REPO_ROOT}"
if [[ $# -gt 0 && -z "$WORK_ROOT_FROM_ENV" ]]; then
  WORK_ROOT="$ROOT/work"
else
  WORK_ROOT="$LFM_DEMO_WORK_ROOT"
fi
REPEATS="${REPEATS:-5}"
OUT="${LFM_Q4_REPORT_DIR:-$ROOT/reports/local_q4_matrix}"
RAW="$OUT/raw"
MODEL_DIR="${LFM_Q4_MODEL_DIR:-$WORK_ROOT/models/lfm25-audio-gguf}"
CLI="${LFM_Q4_CLI:-$MODEL_DIR/runners/llama-liquid-audio-macos-arm64/llama-liquid-audio-cli}"
ASSET_DIR="${LFM_Q4_ASSET_DIR:-$ROOT/vendor/liquid-audio/assets}"

mkdir -p "$RAW"

configs=(
  "f16_fa_on:f16:f16:on"
  "q8_fa_on:q8_0:q8_0:on"
  "q4_fa_on:q4_0:q4_0:on"
  "f16_fa_off:f16:f16:off"
)
audios=("question" "asr")

printf '%s\n' "started $(date -u +%Y-%m-%dT%H:%M:%SZ) repeats=$REPEATS" > "$OUT/status.txt"

failures=0
for spec in "${configs[@]}"; do
  IFS=: read -r config cache_k cache_v flash <<< "$spec"
  for audio_name in "${audios[@]}"; do
    for run in $(seq 1 "$REPEATS"); do
      log="$RAW/${config}__${audio_name}__run${run}.log"
      if rg -q "=== GENERATED TEXT ===" "$log" 2>/dev/null; then
        continue
      fi
      printf '%s\n' "running config=$config audio=$audio_name repeat=$run $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/status.txt"
      if /usr/bin/time -p "$CLI" \
        -fit off \
        -m "$MODEL_DIR/LFM2.5-Audio-1.5B-Q4_0.gguf" \
        --mmproj "$MODEL_DIR/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" \
        -mv "$MODEL_DIR/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" \
        --tts-speaker-file "$MODEL_DIR/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" \
        -sys 'Perform ASR.' \
        --audio "$ASSET_DIR/${audio_name}.wav" \
        -n 128 \
        --temp 0 \
        --seed 1 \
        --perf \
        --offline \
        --log-colors off \
        --cache-type-k "$cache_k" \
        --cache-type-v "$cache_v" \
        --flash-attn "$flash" \
        > "$log" 2>&1; then
        printf '%s\n' "passed config=$config audio=$audio_name repeat=$run" >> "$OUT/status.txt"
      else
        failures=$((failures + 1))
        printf '%s\n' "failed config=$config audio=$audio_name repeat=$run log=$log" >> "$OUT/status.txt"
      fi
    done
  done
done

printf '%s\n' "finished $(date -u +%Y-%m-%dT%H:%M:%SZ) failures=$failures" >> "$OUT/status.txt"
exit "$failures"
