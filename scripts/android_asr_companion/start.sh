#!/system/bin/sh
# Phone-side lifecycle manager. It only signals the PID it created and only
# after /proc confirms the expected loopback listener command.

umask 077
SCRIPT_DIR=${0%/*}
if [ "$SCRIPT_DIR" = "$0" ]; then
  SCRIPT_DIR=.
fi
SCRIPT_DIR=$(CDPATH='' cd -- "$SCRIPT_DIR" 2>/dev/null && pwd)
ASR_COMPANION_ROOT=${ASR_COMPANION_ROOT:-$SCRIPT_DIR}
PID_FILE=${ASR_PID_FILE:-$ASR_COMPANION_ROOT/asr-companion.pid}
LOG_FILE=${ASR_LOG_FILE:-$ASR_COMPANION_ROOT/asr-companion.log}
HANDLER=$ASR_COMPANION_ROOT/handler.sh

is_expected_pid() {
  pid=$1
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  [ -r "/proc/$pid/cmdline" ] || return 1
  command_line=$(toybox tr '\000' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  case "$command_line" in
    *"nc -s 127.0.0.1 -p 6010 -L sh handler.sh"*) return 0 ;;
    *) return 1 ;;
  esac
}

read_pid() {
  [ -f "$PID_FILE" ] || return 1
  pid=$(toybox head -n 1 "$PID_FILE" | toybox tr -d ' \t\r\n')
  is_expected_pid "$pid"
}

health() {
  response="$ASR_COMPANION_ROOT/health.$$.tmp"
  trap 'toybox rm -f "$response"' 0 1 2 15
  printf 'GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n' | \
    toybox nc -w 3 127.0.0.1 6010 > "$response" 2>/dev/null
  result=$?
  if [ "$result" -eq 0 ] && toybox grep -q '"ok":true' "$response"; then
    toybox cat "$response"
    toybox rm -f "$response"
    trap - 0 1 2 15
    return 0
  fi
  toybox cat "$response" 2>/dev/null
  toybox rm -f "$response"
  trap - 0 1 2 15
  return 1
}

start_service() {
  if read_pid; then
    echo "ASR companion already running as PID $pid"
    health
    return $?
  fi
  if [ -f "$PID_FILE" ]; then
    echo "Removing stale ASR companion PID file; no process was signalled" >&2
    toybox rm -f "$PID_FILE"
  fi
  [ -f "$HANDLER" ] || { echo "Missing handler: $HANDLER" >&2; return 2; }
  toybox mkdir -p "$ASR_COMPANION_ROOT/tmp"
  (
    cd "$ASR_COMPANION_ROOT" || exit 2
    if [ -f ./env.sh ]; then
      # shellcheck disable=SC1091
      . ./env.sh
    fi
    exec nohup toybox nc -s 127.0.0.1 -p 6010 -L sh handler.sh
  ) >> "$LOG_FILE" 2>&1 </dev/null &
  launched_pid=$!
  printf '%s\n' "$launched_pid" > "$PID_FILE"
  tries=0
  while [ "$tries" -lt 12 ]; do
    if ! is_expected_pid "$launched_pid"; then
      echo "ASR companion exited during startup; inspect $LOG_FILE" >&2
      toybox rm -f "$PID_FILE"
      return 3
    fi
    if health >/dev/null; then
      echo "ASR companion ready on 127.0.0.1:6010 (PID $launched_pid)"
      return 0
    fi
    sleep 1
    tries=$((tries + 1))
  done
  echo "ASR companion did not become healthy; inspect $LOG_FILE" >&2
  stop_service
  return 4
}

stop_service() {
  if ! read_pid; then
    echo "ASR companion is not running; no process was signalled"
    toybox rm -f "$PID_FILE"
    return 0
  fi
  target_pid=$pid
  kill "$target_pid" 2>/dev/null || true
  tries=0
  while [ "$tries" -lt 10 ]; do
    if ! is_expected_pid "$target_pid"; then
      toybox rm -f "$PID_FILE"
      echo "ASR companion stopped"
      return 0
    fi
    sleep 1
    tries=$((tries + 1))
  done
  echo "ASR companion PID $target_pid did not stop; refusing to escalate" >&2
  return 5
}

case "${1:-status}" in
  start) start_service ;;
  stop) stop_service ;;
  restart)
    stop_service && start_service
    ;;
  status)
    if read_pid; then
      echo "ASR companion running as PID $pid"
      health
    else
      echo "ASR companion is not running"
      exit 1
    fi
    ;;
  health) health ;;
  log) toybox tail -n "${2:-100}" "$LOG_FILE" ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|health|log [lines]}" >&2
    exit 2
    ;;
esac
