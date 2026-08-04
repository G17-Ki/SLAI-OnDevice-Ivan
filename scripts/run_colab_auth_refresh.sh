#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
lfm_require_colab_bundle

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'Colab bundle: %s\nWork root: %s\n' \
    "$COLAB_BUNDLE" "$LFM_DEMO_WORK_ROOT"
  exit 0
fi

export COLAB_PYTHON="${COLAB_PYTHON:-/opt/homebrew/bin/python3.13}"
set -a
source "$COLAB_BUNDLE/.colab_env"
set +a

# Seed the private home once. The OAuth flow then replaces token.json with the
# refreshed token. Subsequent smoke/full-run wrappers deliberately preserve it.
colab_env_materialize
colab_env_bootstrap_cli

lfm_find_colab_oauth_config

HOME="$COLAB_HOME" "$COLAB_CLI" \
  --client-oauth-config "$OFFICIAL_OAUTH_CONFIG" \
  --auth "$COLAB_AUTH" sessions
