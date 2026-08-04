#!/bin/zsh

set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h:h}
destination=${1:-"$repo_root/models/supertonic3-int8"}
archive="$destination/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"
model_dir="$destination/sherpa-onnx-supertonic-3-tts-int8-2026-05-11"

archive_url='https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2'
archive_sha256='82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427'
model_license_url='https://huggingface.co/Supertone/supertonic-3/raw/3cadd1ee6394adea1bd021217a0e650ede09a323/LICENSE'
model_license_sha256='0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f'

mkdir -p "$destination"

if [[ ! -f "$archive" ]]; then
  curl --location --fail --retry 5 --output "$archive" "$archive_url"
fi

printf '%s  %s\n' "$archive_sha256" "$archive" | shasum -a 256 -c -

if [[ ! -d "$model_dir" ]]; then
  tar -xjf "$archive" -C "$destination"
fi

curl --location --fail --retry 5 \
  --output "$model_dir/MODEL_LICENSE_OPENRAIL-M.txt" \
  "$model_license_url"
printf '%s  %s\n' \
  "$model_license_sha256" \
  "$model_dir/MODEL_LICENSE_OPENRAIL-M.txt" | shasum -a 256 -c -

echo "$model_dir"
