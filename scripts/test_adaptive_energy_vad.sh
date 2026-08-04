#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lfm-vad-test.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

javac \
  -encoding UTF-8 \
  -d "$BUILD_DIR" \
  "$ROOT/android-app/app/src/main/java/ai/liquid/lfmdemo/AdaptiveEnergyVad.java" \
  "$ROOT/scripts/AdaptiveEnergyVadTest.java"

java -cp "$BUILD_DIR" ai.liquid.lfmdemo.AdaptiveEnergyVadTest
