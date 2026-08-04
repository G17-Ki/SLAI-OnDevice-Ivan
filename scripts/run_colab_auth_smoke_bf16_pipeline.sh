#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  printf 'Usage: %s [--dry-run]\n' "${0##*/}" >&2
  exit 2
fi

printf '%s\n' 'Step 1/3: Colab account authorization'
bash "$SCRIPT_DIR/run_colab_auth_refresh.sh" "$@"

printf '%s\n' 'Step 2/3: L4 smoke test'
bash "$SCRIPT_DIR/run_colab_smoke.sh" "$@"

printf '%s\n' 'Step 3/3: full pinned BF16 golden run'
bash "$SCRIPT_DIR/run_colab_bf16_golden.sh" "$@"

printf '%s\n' 'Colab auth, smoke test, and BF16 golden pipeline completed.'
