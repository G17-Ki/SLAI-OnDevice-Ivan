#!/usr/bin/env bash
# Fetch the sherpa-onnx Android runtime used by the experimental demo TTS
# backend and android-tts-bench.
#
# The AAR is 48 MiB of prebuilt native code and is deliberately kept out of this
# repository. It is pinned by version and SHA-256 so the device runs the same
# engine build the host measurements were taken with.
set -Eeuo pipefail

VERSION="1.13.4"
SHA256="03f9c4df965f21c71269365a7951a7f23b5696fddd093fa318c80d65550ab780"
DEST="${SHERPA_AAR_DIR:-${HOME}/.cache/lmf-demo/sherpa-aar}"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v${VERSION}/sherpa-onnx-${VERSION}.aar"

mkdir -p "$DEST"
FILE="$DEST/sherpa-onnx-${VERSION}.aar"

if [[ ! -f "$FILE" ]]; then
  echo "downloading sherpa-onnx ${VERSION} AAR..."
  curl --location --fail --retry 3 --output "$FILE" "$URL"
fi

echo "${SHA256}  ${FILE}" | shasum -a 256 -c -
echo "SHERPA_AAR_DIR=$DEST"
