#!/bin/zsh

set -euo pipefail

model_name=sherpa-onnx-qnn-SM8850-binary-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8
archive_name=$model_name.tar.bz2
archive_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models-qnn-binary/$archive_name
expected_sha256=ecbc1ffba39f8e23582b79a199d12e8455425a22ef7b6b18c535ce25fcff2d64
destination=${1:-${SENSEVOICE_QNN_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lmf-demo/sensevoice-qnn-sm8850}}
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
  print -u2 "SM8850 SenseVoice QNN archive checksum mismatch"
  print -u2 "expected: $expected_sha256"
  print -u2 "actual:   $actual_sha256"
  exit 1
fi

if [[ ! -f $model_dir/model.bin || ! -f $model_dir/tokens.txt ]]; then
  tar -xjf $archive -C $destination
fi

for required in model.bin tokens.txt info.txt LICENSE README.md; do
  if [[ ! -f $model_dir/$required ]]; then
    print -u2 "missing expected file: $model_dir/$required"
    exit 1
  fi
done

print $model_dir
