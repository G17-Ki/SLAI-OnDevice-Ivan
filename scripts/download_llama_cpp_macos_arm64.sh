#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/path_config.sh"
OUTPUT_DIR="${1:-$LFM_DEMO_WORK_ROOT/runtimes/llama-cpp-b10083}"
VERSION="b10083"
ARCHIVE="llama-${VERSION}-bin-macos-arm64.tar.gz"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${VERSION}/${ARCHIVE}"
SHA256="b6c4b7ff81cd6cc54be156ba8d5cb7ec930e8a0efa333454107089c5ffce6998"

mkdir -p "$OUTPUT_DIR"
curl -fL --retry 3 --output "$OUTPUT_DIR/$ARCHIVE" "$URL"

actual="$(shasum -a 256 "$OUTPUT_DIR/$ARCHIVE" | awk '{print $1}')"
if [[ "$actual" != "$SHA256" ]]; then
  echo "SHA-256 mismatch for $ARCHIVE" >&2
  echo "expected: $SHA256" >&2
  echo "actual:   $actual" >&2
  exit 1
fi

tar -xzf "$OUTPUT_DIR/$ARCHIVE" -C "$OUTPUT_DIR"
test -x "$OUTPUT_DIR/llama-${VERSION}/llama-server"
test -x "$OUTPUT_DIR/llama-${VERSION}/llama-cli"
echo "Verified llama.cpp $VERSION macOS ARM64 runtime: $OUTPUT_DIR/llama-${VERSION}"
