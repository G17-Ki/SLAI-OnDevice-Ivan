#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/path_config.sh"
lfm_require_colab_bundle
CONFIG="$SCRIPT_DIR/colab_lfm_bf16_config.json"
BASE_RUNNER="$SCRIPT_DIR/colab_lfm_bf16_runner.py"
RUNNER="$SCRIPT_DIR/colab_candidate_quality_runner.py"
LAUNCHER="$SCRIPT_DIR/colab_candidate_quality_launcher.py"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'Runner: %s\nBase runner: %s\nConfig: %s\nColab bundle: %s\nWork root: %s\n' \
    "$RUNNER" "$BASE_RUNNER" "$CONFIG" "$COLAB_BUNDLE" \
    "$LFM_DEMO_WORK_ROOT"
  exit 0
fi

set -a
source "$COLAB_BUNDLE/.colab_env"
set +a
colab_env_bootstrap_cli

if [[ ! -f "$COLAB_HOME/.config/colab-cli/token.json" ]]; then
  printf '%s\n' 'Bundle-local Colab token is missing.' >&2
  exit 3
fi
lfm_find_colab_oauth_config
COLAB_ARGS=(--client-oauth-config "$OFFICIAL_OAUTH_CONFIG" --auth "$COLAB_AUTH")
AUTH_CHECK_LOG="$(mktemp -t lfm-candidate-auth.XXXXXX)"
chmod 600 "$AUTH_CHECK_LOG"
if ! HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" sessions >"$AUTH_CHECK_LOG" 2>&1; then
  rm -f "$AUTH_CHECK_LOG"
  printf '%s\n' 'Colab authorization is not ready.' >&2
  exit 3
fi
rm -f "$AUTH_CHECK_LOG"

STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
GPU="${REQUESTED_COLAB_GPU:-L4}"
SESSION="${COLAB_SESSION_PREFIX}-candidate-quality-$STAMP"
LOCAL_OUTPUT_DIR="$LFM_DEMO_WORK_ROOT/lfm-feasibility/artifacts/colab-candidate-quality-$STAMP"
LOCAL_ZIP="$LOCAL_OUTPUT_DIR/candidate_quality_result.zip"
mkdir -p "$LOCAL_OUTPUT_DIR"
SESSION_CREATED=0

stop_session() {
  local rc=$?
  if [[ "$SESSION_CREATED" -eq 1 ]]; then
    HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" stop --session "$SESSION" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap stop_session EXIT INT TERM HUP

SESSION_CREATED=1
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" new --gpu "$GPU" --session "$SESSION"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" status --session "$SESSION"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$CONFIG" /content/colab_lfm_bf16_config.json
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$BASE_RUNNER" /content/colab_lfm_bf16_runner.py
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$RUNNER" /content/colab_candidate_quality_runner.py
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$LFM_DEMO_REPO_ROOT/reports/colab_bf16/interleaved_turn1.wav" /content/lfm_turn1.wav
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$LFM_DEMO_REPO_ROOT/reports/colab_bf16/interleaved_turn2.wav" /content/lfm_turn2.wav
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$LFM_DEMO_REPO_ROOT/reports/colab_bf16/interleaved_turn1.txt" /content/lfm_turn1.txt
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" upload --session "$SESSION" "$LFM_DEMO_REPO_ROOT/reports/colab_bf16/interleaved_turn2.txt" /content/lfm_turn2.txt

set +e
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" exec --session "$SESSION" --file "$LAUNCHER" --timeout 10800
REMOTE_RC=$?
set -e
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" download --session "$SESSION" /content/candidate_quality_result.zip "$LOCAL_ZIP"
HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" stop --session "$SESSION"
SESSION_CREATED=0
trap - EXIT INT TERM HUP

printf 'Candidate quality archive: %s\n' "$LOCAL_ZIP"
exit "$REMOTE_RC"
