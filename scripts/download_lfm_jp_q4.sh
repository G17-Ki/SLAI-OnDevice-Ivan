#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/models/lfm25-audio-jp-q4}"
MODEL_REPO="LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF"
MODEL_REVISION="c8410f9538b2e5695cf574dd5d3be54a4a3aed72"
SAMPLE_REPO="Liquid4All/liquid-audio"
SAMPLE_REVISION="19e65845923a7f136442c95137884ec61eb386aa"
CHECKSUMS="$SCRIPT_DIR/lfm_jp_q4.sha256"

download_verified() {
  local url="$1"
  local relative_path="$2"
  local destination="$OUTPUT_DIR/$relative_path"
  local partial="$destination.partial"
  local expected
  local actual
  local attempt=1
  expected="$(awk -v path="$relative_path" '$2 == path { print $1 }' "$CHECKSUMS")"
  if [[ -f "$destination" && -n "$expected" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      actual="$(sha256sum "$destination" | awk '{ print $1 }')"
    else
      actual="$(shasum -a 256 "$destination" | awk '{ print $1 }')"
    fi
    if [[ "$actual" == "$expected" ]]; then
      printf 'Already verified: %s\n' "$relative_path"
      return
    fi
  fi

  # Never resume into a file that is presented as complete. A prior interrupted
  # range transfer can have the expected byte count but still contain a corrupt
  # segment. Keep resumable bytes under a .partial suffix, verify them, and only
  # then publish the final filename.
  if [[ -f "$destination" ]]; then
    if [[ ! -f "$partial" ]]; then
      mv "$destination" "$partial"
    else
      rm -f "$destination"
    fi
  fi
  mkdir -p "$(dirname "$destination")"
  while true; do
    if curl \
      --http1.1 \
      --fail \
      --location \
      --retry 12 \
      --retry-all-errors \
      --retry-delay 2 \
      --connect-timeout 30 \
      --continue-at - \
      --output "$partial" \
      "$url"
    then
      if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$partial" | awk '{ print $1 }')"
      else
        actual="$(shasum -a 256 "$partial" | awk '{ print $1 }')"
      fi
      if [[ "$actual" == "$expected" ]]; then
        mv "$partial" "$destination"
        break
      fi
      printf 'Checksum mismatch after transfer; retrying fresh: %s\n' \
        "$relative_path" >&2
      rm -f "$partial"
    fi
    if [[ "$attempt" -ge 100 ]]; then
      printf 'Download failed after %d resumable attempts: %s\n' "$attempt" "$relative_path" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    printf 'Download interrupted; resuming %s (attempt %d/100).\n' \
      "$relative_path" "$attempt" >&2
    sleep 3
  done
}

mkdir -p "$OUTPUT_DIR"

for relative_path in \
  LFM2.5-Audio-1.5B-JP-Q4_0.gguf \
  mmproj-LFM2.5-Audio-1.5B-JP-Q4_0.gguf \
  vocoder-LFM2.5-Audio-1.5B-JP-Q4_0.gguf \
  tokenizer-LFM2.5-Audio-1.5B-JP-Q4_0.gguf \
  runners/llama-liquid-audio-android-arm64.zip \
  runners/llama-liquid-audio-macos-arm64.zip \
  README.md \
  LICENSE
do
  download_verified \
    "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/$relative_path" \
    "$relative_path"
done

for sample in question_jp.wav asr_jp.wav
do
  download_verified \
    "https://raw.githubusercontent.com/$SAMPLE_REPO/$SAMPLE_REVISION/assets/$sample" \
    "assets/$sample"
done

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$OUTPUT_DIR" && sha256sum -c "$CHECKSUMS")
else
  (cd "$OUTPUT_DIR" && shasum -a 256 -c "$CHECKSUMS")
fi

for platform in macos-arm64 android-arm64
do
  archive="$OUTPUT_DIR/runners/llama-liquid-audio-$platform.zip"
  unzip -q -o "$archive" -d "$OUTPUT_DIR/runners"
done

find "$OUTPUT_DIR/runners" -type f \
  \( -name 'llama-liquid-audio-cli' -o -name 'llama-liquid-audio-server' \) \
  -exec chmod +x {} +

printf 'Verified Japanese Q4 bundle from %s@%s\n' "$MODEL_REPO" "$MODEL_REVISION"
printf 'Verified official samples from %s@%s\n' "$SAMPLE_REPO" "$SAMPLE_REVISION"
printf 'Bundle: %s\n' "$OUTPUT_DIR"
