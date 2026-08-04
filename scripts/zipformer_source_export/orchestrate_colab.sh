#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/orchestrate_colab.sh"
EXPORTER="$SCRIPT_DIR/export_sources.py"
REMOTE_WORKER="$SCRIPT_DIR/remote_worker.py"
REMOTE_LAUNCHER="$SCRIPT_DIR/remote_launcher.py"
REMOTE_PROBE="$SCRIPT_DIR/remote_probe.py"

MODE="dry-run"
CONFIRM_CLOUD=0
OUTPUT_DIR=""
COLAB_BUNDLE="${COLAB_BUNDLE_DIR:-}"
GPU="${REQUESTED_COLAB_GPU:-T4}"
POLL_SECONDS="${COLAB_POLL_SECONDS:-45}"
TIMEOUT_SECONDS="${COLAB_TIMEOUT_SECONDS:-14400}"
SESSION_NAME=""
RESUME_CHECKPOINT=""

usage() {
  printf '%s\n' \
    'Offline:  orchestrate_colab.sh --dry-run' \
    'Launch:   orchestrate_colab.sh --launch --confirm-cloud --colab-bundle PATH' \
    'Resume:   add --resume-checkpoint PATH_TO_LATEST_CHECKPOINT_TAR' \
    '' \
    'A real launch creates a detached host tmux session. The internal --run mode' \
    'is guarded and cannot allocate Colab unless both --confirm-cloud and TMUX exist.'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --launch)
      MODE="launch"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --confirm-cloud)
      CONFIRM_CLOUD=1
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --colab-bundle)
      COLAB_BUNDLE="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --resume-checkpoint)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for required in "$EXPORTER" "$REMOTE_WORKER" "$REMOTE_LAUNCHER" "$REMOTE_PROBE"; do
  if [[ ! -f "$required" ]]; then
    printf 'Missing harness file: %s\n' "$(basename "$required")" >&2
    exit 2
  fi
done

if [[ ! "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 15 || POLL_SECONDS > 300 )); then
  printf '%s\n' '--poll-seconds must be an integer from 15 through 300.' >&2
  exit 2
fi
if [[ ! "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( TIMEOUT_SECONDS < 900 || TIMEOUT_SECONDS > 43200 )); then
  printf '%s\n' '--timeout-seconds must be an integer from 900 through 43200.' >&2
  exit 2
fi
case "$GPU" in
  T4|L4) ;;
  *)
    printf '%s\n' 'This CPU-side export permits only T4 or L4; T4 is sufficient.' >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "dry-run" ]]; then
  python3 "$EXPORTER" --self-check --threads 4 >/dev/null
  python3 - "$GPU" "$POLL_SECONDS" "$TIMEOUT_SECONDS" <<'PY'
import json
import sys
print(json.dumps({
    "status": "dry_run_passed",
    "cloud_action": False,
    "components": ["decoder", "joiner", "encoder"],
    "authoritative_source": "QAIHM 0.58 component.serialize TorchScript",
    "onnx": {"opset": 17, "dynamic_axes": False, "local_comparison": True},
    "checkpoint_resume": True,
    "gpu_for_future_launch": sys.argv[1],
    "poll_seconds": int(sys.argv[2]),
    "timeout_seconds": int(sys.argv[3]),
    "real_launch_requires": ["--launch", "--confirm-cloud", "detached tmux", "private colab-cli bundle"],
}, indent=2, sort_keys=True))
PY
  exit 0
fi

if [[ "$CONFIRM_CLOUD" -ne 1 ]]; then
  printf '%s\n' 'Refusing cloud allocation without --confirm-cloud.' >&2
  exit 3
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')"
if [[ -z "$SESSION_NAME" ]]; then
  SESSION_NAME="zipformer-source-$STAMP"
fi
if [[ ! "$SESSION_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf '%s\n' '--session may contain only letters, digits, dot, underscore, and hyphen.' >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$PROJECT_DIR/artifacts/zipformer-source-export/$SESSION_NAME"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    printf '%s\n' 'The requested resume checkpoint does not exist.' >&2
    exit 2
  fi
  RESUME_CHECKPOINT="$(cd "$(dirname "$RESUME_CHECKPOINT")" && pwd)/$(basename "$RESUME_CHECKPOINT")"
  python3 - "$RESUME_CHECKPOINT" <<'PY'
from pathlib import Path, PurePosixPath
import sys, tarfile
path = Path(sys.argv[1])
with tarfile.open(path, "r:*") as archive:
    members = archive.getmembers()
if not members or len(members) > 1000:
    raise SystemExit("resume checkpoint has an invalid member count")
size = 0
names = set()
for member in members:
    pure = PurePosixPath(member.name)
    if pure.is_absolute() or ".." in pure.parts or not member.isfile():
        raise SystemExit("resume checkpoint contains an unsafe member")
    if member.name in names:
        raise SystemExit("resume checkpoint contains duplicate members")
    names.add(member.name)
    size += member.size
if size > 4 * 1024 * 1024 * 1024:
    raise SystemExit("resume checkpoint expands beyond the safety limit")
if "SHA256SUMS" not in names:
    raise SystemExit("resume checkpoint is missing SHA256SUMS")
PY
fi

if [[ "$MODE" == "launch" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    printf '%s\n' 'tmux is required for a Colab launch.' >&2
    exit 4
  fi
  if tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
    printf 'tmux session already exists: %s\n' "$SESSION_NAME" >&2
    exit 4
  fi
  if [[ -z "$COLAB_BUNDLE" || ! -f "$COLAB_BUNDLE/.colab_env" ]]; then
    printf '%s\n' 'Set --colab-bundle to the private colab-cli bundle directory.' >&2
    exit 4
  fi
  TMUX_COMMAND=(
    "$SCRIPT_PATH"
    --run
    --confirm-cloud
    --output-dir "$OUTPUT_DIR"
    --colab-bundle "$COLAB_BUNDLE"
    --gpu "$GPU"
    --poll-seconds "$POLL_SECONDS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --session "$SESSION_NAME"
  )
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TMUX_COMMAND+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi
  tmux new-session -d -s "$SESSION_NAME" "${TMUX_COMMAND[@]}"
  printf 'Detached tmux session: %s\n' "$SESSION_NAME"
  printf 'Progress log: %s\n' "$OUTPUT_DIR/orchestrator.log"
  printf 'Latest checkpoint: %s\n' "$OUTPUT_DIR/latest-checkpoint.tar"
  exit 0
fi

if [[ "$MODE" != "run" ]]; then
  printf '%s\n' 'Invalid orchestrator mode.' >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  printf '%s\n' 'Cloud orchestration is allowed only inside an active tmux session.' >&2
  exit 5
fi
if [[ -z "$COLAB_BUNDLE" || ! -f "$COLAB_BUNDLE/.colab_env" ]]; then
  printf '%s\n' 'The private colab-cli bundle is unavailable.' >&2
  exit 4
fi

exec >>"$OUTPUT_DIR/orchestrator.log" 2>&1
STATUS_FILE="$OUTPUT_DIR/orchestrator_status.json"
LATEST_PROBE="$OUTPUT_DIR/latest-probe.json"
LATEST_CHECKPOINT="$OUTPUT_DIR/latest-checkpoint.tar"

write_local_status() {
  local stage="$1"
  local detail="${2:-}"
  python3 - "$STATUS_FILE" "$stage" "$SESSION_NAME" "$detail" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile
path = Path(sys.argv[1])
payload = {
    "format_version": 1,
    "stage": sys.argv[2],
    "session": sys.argv[3],
    "detail": sys.argv[4],
    "updated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, path)
PY
}

write_local_status "initializing" "no cloud session created yet"

set -a
# shellcheck source=/dev/null
source "$COLAB_BUNDLE/.colab_env"
set +a
if ! colab_env_bootstrap_cli >/dev/null 2>&1; then
  write_local_status "failed" "Colab CLI bootstrap failed"
  printf '%s\n' 'Colab CLI bootstrap failed.' >&2
  exit 4
fi
if [[ ! -f "$COLAB_HOME/.config/colab-cli/token.json" ]]; then
  if ! colab_env_materialize >/dev/null 2>&1; then
    write_local_status "failed" "Colab credentials are not materialized"
    printf '%s\n' 'Colab credentials are not ready; refresh the private bundle.' >&2
    exit 4
  fi
fi

# The CLI needs only its private files. Do not propagate unrelated remote-job
# tokens or the portable base64 credential snapshots to child processes.
unset HF_TOKEN GITHUB_TOKEN WANDB_API_KEY
unset COLAB_CLIENT_OAUTH_CONFIG_B64 COLAB_TOKEN_JSON_B64

OFFICIAL_OAUTH_CONFIG=""
for candidate in "$COLAB_VENV"/lib/python*/site-packages/colab_cli/oauth_config.json; do
  if [[ -f "$candidate" ]]; then
    OFFICIAL_OAUTH_CONFIG="$candidate"
    break
  fi
done
if [[ -z "$OFFICIAL_OAUTH_CONFIG" ]]; then
  write_local_status "failed" "official Colab OAuth configuration is missing"
  printf '%s\n' 'The installed Colab CLI OAuth configuration is missing.' >&2
  exit 4
fi
COLAB_ARGS=(--client-oauth-config "$OFFICIAL_OAUTH_CONFIG" --auth "$COLAB_AUTH")

safe_colab() {
  local raw
  raw="$(mktemp "$OUTPUT_DIR/.colab-cli.XXXXXX")"
  chmod 600 "$raw"
  if HOME="$COLAB_HOME" "$COLAB_CLI" "${COLAB_ARGS[@]}" "$@" >"$raw" 2>&1; then
    rm -f "$raw"
    return 0
  fi
  rm -f "$raw"
  return 1
}

if ! safe_colab sessions; then
  write_local_status "failed" "Colab authorization check failed"
  printf '%s\n' 'Colab authorization is not ready. Refresh it interactively.' >&2
  exit 4
fi

SESSION_CREATED=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ "$SESSION_CREATED" -eq 1 ]]; then
    safe_colab stop --session "$SESSION_NAME" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

write_local_status "allocating" "requesting $GPU runtime"
if ! safe_colab new --gpu "$GPU" --session "$SESSION_NAME"; then
  write_local_status "failed" "Colab runtime allocation failed"
  printf '%s\n' 'Colab runtime allocation failed.' >&2
  exit 6
fi
SESSION_CREATED=1
write_local_status "uploading_harness" "runtime allocated"

if ! safe_colab status --session "$SESSION_NAME"; then
  write_local_status "failed" "new Colab session did not become ready"
  exit 6
fi
if ! safe_colab upload --session "$SESSION_NAME" "$EXPORTER" /content/export_sources.py; then
  write_local_status "failed" "exporter upload failed"
  exit 6
fi
if ! safe_colab upload --session "$SESSION_NAME" "$REMOTE_WORKER" /content/remote_worker.py; then
  write_local_status "failed" "worker upload failed"
  exit 6
fi
if ! safe_colab upload --session "$SESSION_NAME" "$REMOTE_LAUNCHER" /content/remote_launcher.py; then
  write_local_status "failed" "launcher upload failed"
  exit 6
fi
if ! safe_colab upload --session "$SESSION_NAME" "$REMOTE_PROBE" /content/remote_probe.py; then
  write_local_status "failed" "probe upload failed"
  exit 6
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  write_local_status "uploading_resume_checkpoint" "validated local checkpoint"
  if ! safe_colab upload --session "$SESSION_NAME" "$RESUME_CHECKPOINT" /content/zipformer-source-resume.tar; then
    write_local_status "failed" "resume checkpoint upload failed"
    exit 6
  fi
fi
if ! safe_colab exec --session "$SESSION_NAME" --file "$REMOTE_LAUNCHER" --timeout 120; then
  write_local_status "failed" "detached worker launch failed"
  exit 6
fi

write_local_status "running" "remote worker launched"
LAST_GENERATION=0
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))

download_checkpoint() {
  local expected_sha="$1"
  local expected_size="$2"
  local generation="$3"
  local partial="$OUTPUT_DIR/.checkpoint-generation-$generation.partial.tar"
  rm -f "$partial"
  if ! safe_colab download --session "$SESSION_NAME" /content/zipformer-source-v058/checkpoint.tar "$partial"; then
    rm -f "$partial"
    return 1
  fi
  if ! python3 - "$partial" "$expected_sha" "$expected_size" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib
import sys
import tarfile
path = Path(sys.argv[1])
expected_sha = sys.argv[2]
expected_size = int(sys.argv[3])
if path.stat().st_size != expected_size:
    raise SystemExit("checkpoint size mismatch")
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha:
    raise SystemExit("checkpoint digest mismatch")
with tarfile.open(path, "r:*") as archive:
    members = archive.getmembers()
if not members or len(members) > 1000:
    raise SystemExit("checkpoint member count is invalid")
names = set()
for member in members:
    pure = PurePosixPath(member.name)
    if pure.is_absolute() or ".." in pure.parts or not member.isfile():
        raise SystemExit("checkpoint contains an unsafe member")
    if member.name in names:
        raise SystemExit("checkpoint contains duplicate members")
    names.add(member.name)
if "SHA256SUMS" not in names:
    raise SystemExit("checkpoint is missing SHA256SUMS")
PY
  then
    rm -f "$partial"
    return 1
  fi
  mv "$partial" "$LATEST_CHECKPOINT"
  printf '%s  %s\n' "$expected_sha" "$(basename "$LATEST_CHECKPOINT")" > "$LATEST_CHECKPOINT.sha256"
  LAST_GENERATION="$generation"
  return 0
}

while true; do
  if (( $(date +%s) >= DEADLINE )); then
    write_local_status "timed_out" "latest verified checkpoint retained locally"
    printf '%s\n' 'Colab export reached its host timeout; latest checkpoint was retained.' >&2
    exit 7
  fi
  if ! safe_colab status --session "$SESSION_NAME"; then
    write_local_status "session_lost" "latest verified checkpoint retained locally"
    printf '%s\n' 'The Colab session is no longer reachable.' >&2
    exit 7
  fi
  if safe_colab exec --session "$SESSION_NAME" --file "$REMOTE_PROBE" --timeout 90; then
    PROBE_PARTIAL="$OUTPUT_DIR/.probe.partial.json"
    rm -f "$PROBE_PARTIAL"
    if safe_colab download --session "$SESSION_NAME" /content/zipformer-source-v058/probe.json "$PROBE_PARTIAL"; then
      PROBE_FIELDS="$(python3 - "$PROBE_PARTIAL" <<'PY'
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text())
values = [
    str(payload.get("state", "unknown")),
    str(payload.get("checkpoint_generation", 0)),
    str(payload.get("checkpoint_sha256") or "-"),
    str(payload.get("checkpoint_size_bytes", 0)),
    str(payload.get("active_component") or "-"),
    ",".join(str(value) for value in payload.get("completed_components", [])) or "-",
]
print("|".join(values))
PY
)"
      IFS='|' read -r REMOTE_STATE GENERATION CHECKPOINT_SHA CHECKPOINT_SIZE ACTIVE_COMPONENT COMPLETED_COMPONENTS <<< "$PROBE_FIELDS"
      mv "$PROBE_PARTIAL" "$LATEST_PROBE"
      write_local_status "remote_$REMOTE_STATE" "active=$ACTIVE_COMPONENT completed=$COMPLETED_COMPONENTS checkpoint=$GENERATION"
      if [[ "$GENERATION" =~ ^[0-9]+$ ]] \
        && (( GENERATION > LAST_GENERATION )) \
        && [[ "$CHECKPOINT_SHA" =~ ^[0-9a-f]{64}$ ]] \
        && [[ "$CHECKPOINT_SIZE" =~ ^[0-9]+$ ]] \
        && (( CHECKPOINT_SIZE > 0 )); then
        if download_checkpoint "$CHECKPOINT_SHA" "$CHECKPOINT_SIZE" "$GENERATION"; then
          write_local_status "checkpoint_downloaded" "generation=$GENERATION remote_state=$REMOTE_STATE"
        fi
      fi
      if [[ "$REMOTE_STATE" == "succeeded" ]]; then
        safe_colab stop --session "$SESSION_NAME" >/dev/null 2>&1 || true
        SESSION_CREATED=0
        trap - EXIT INT TERM HUP
        write_local_status "succeeded" "final checkpoint generation=$LAST_GENERATION"
        printf 'Zipformer source export completed. Checkpoint: %s\n' "$LATEST_CHECKPOINT"
        exit 0
      fi
      if [[ "$REMOTE_STATE" == "failed" || "$REMOTE_STATE" == "stale" ]]; then
        write_local_status "failed" "remote_state=$REMOTE_STATE latest checkpoint generation=$LAST_GENERATION"
        printf '%s\n' 'The remote exporter stopped before a successful completion.' >&2
        exit 8
      fi
    fi
  fi
  sleep "$POLL_SECONDS"
done
