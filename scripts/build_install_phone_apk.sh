#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$SCRIPT_DIR/build_android_apk.sh"
"$SCRIPT_DIR/install_native_phone_demo.sh" \
  --apk "$REPO_ROOT/android-app/app/build/outputs/apk/debug/app-debug.apk" \
  "$@"
