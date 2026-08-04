#!/bin/zsh

set -euo pipefail

model_name=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
archive_name=$model_name.tar.bz2
archive_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$archive_name
expected_sha256=7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e
destination=${1:-${SENSEVOICE_MODEL_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/sensevoice}}
archive=$destination/$archive_name
partial=$archive.part
model_dir=$destination/$model_name

mkdir -p $destination

if [[ ! -f $archive ]]; then
  curl -L --fail --show-error --retry 3 --continue-at - \
    --output $partial $archive_url
  mv $partial $archive
fi

actual_sha256=$(shasum -a 256 $archive | awk '{print $1}')
if [[ $actual_sha256 != $expected_sha256 ]]; then
  print -u2 "SenseVoice archive checksum mismatch"
  print -u2 "expected: $expected_sha256"
  print -u2 "actual:   $actual_sha256"
  exit 1
fi

if [[ ! -f $model_dir/model.int8.onnx || ! -f $model_dir/tokens.txt ]]; then
  tar -xjf $archive -C $destination
fi

[[ -f $model_dir/model.int8.onnx ]]
[[ -f $model_dir/tokens.txt ]]
print $model_dir
