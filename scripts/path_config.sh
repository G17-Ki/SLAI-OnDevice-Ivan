#!/usr/bin/env bash
# Shared checkout-relative paths for shell launchers. Source this file.

_lfm_path_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFM_DEMO_REPO_ROOT="$(cd "$_lfm_path_config_dir/.." && pwd)"
LFM_DEMO_WORK_ROOT="${LFM_DEMO_WORK_ROOT:-$LFM_DEMO_REPO_ROOT/work}"
LFM_DEMO_VENDOR_ROOT="$LFM_DEMO_REPO_ROOT/vendor/liquid-audio"
HF_HOME="${HF_HOME:-$LFM_DEMO_WORK_ROOT/cache/huggingface}"
export LFM_DEMO_REPO_ROOT LFM_DEMO_WORK_ROOT LFM_DEMO_VENDOR_ROOT HF_HOME
unset _lfm_path_config_dir

lfm_require_colab_bundle() {
  if [[ -z "${COLAB_CLI_BUNDLE:-}" ]]; then
    printf '%s\n' \
      'Set COLAB_CLI_BUNDLE to the private colab-cli bundle directory.' >&2
    return 2
  fi
  COLAB_BUNDLE="$(cd "$COLAB_CLI_BUNDLE" 2>/dev/null && pwd)" || {
    printf 'COLAB_CLI_BUNDLE is not a readable directory: %s\n' \
      "$COLAB_CLI_BUNDLE" >&2
    return 2
  }
  if [[ ! -f "$COLAB_BUNDLE/.colab_env" ]]; then
    printf 'COLAB_CLI_BUNDLE is missing .colab_env: %s\n' \
      "$COLAB_BUNDLE" >&2
    return 2
  fi
  export COLAB_BUNDLE
}

lfm_find_colab_oauth_config() {
  local candidate
  OFFICIAL_OAUTH_CONFIG=""
  for candidate in \
    "$COLAB_VENV"/lib/python*/site-packages/colab_cli/oauth_config.json
  do
    if [[ -f "$candidate" ]]; then
      OFFICIAL_OAUTH_CONFIG="$candidate"
      export OFFICIAL_OAUTH_CONFIG
      return 0
    fi
  done
  printf 'Official Colab OAuth config not found below: %s\n' \
    "$COLAB_VENV" >&2
  return 2
}
