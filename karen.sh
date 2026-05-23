#!/usr/bin/env bash
# Karen — control script
# Usage:
#   ./karen.sh start              — start bot in background
#   ./karen.sh stop               — stop bot
#   ./karen.sh status             — show running status + current config
#   ./karen.sh log                — tail live log
#   ./karen.sh strategy <mode>    — hot-swap strategy (mean_reversion | trend_following)
#   ./karen.sh trading <on|off>   — enable or disable trading without restart

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
LOG_FILE="$SCRIPT_DIR/logs/karen.log"
PID_FILE="$SCRIPT_DIR/data/karen.pid"

# ── helpers ───────────────────────────────────────────────────────────────────

get_pid() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return
    fi
  fi
  # fallback: search by process name
  pgrep -f "python -m karen.main" 2>/dev/null | head -1 || true
}

# Edit .env in-place (preserves inode so watchdog detects the change)
set_env_var() {
  local key="$1" val="$2"
  python3 - <<EOF
import re, sys
path = "$ENV_FILE"
content = open(path).read()
new_content = re.sub(r'^(${key}\s*=\s*).*$', r'\g<1>${val}', content, flags=re.MULTILINE)
if content == new_content:
    sys.exit(1)
with open(path, 'r+') as f:
    f.seek(0); f.write(new_content); f.truncate()
EOF
}

get_env_var() {
  grep -E "^$1\s*=" "$ENV_FILE" | cut -d= -f2- | tr -d ' '
}

# ── commands ──────────────────────────────────────────────────────────────────

cmd_start() {
  local pid
  pid=$(get_pid)
  if [[ -n "$pid" ]]; then
    echo "[karen] Already running (PID $pid)"
    return
  fi

  cd "$SCRIPT_DIR"
  [[ -f ".venv/bin/activate" ]] && source .venv/bin/activate
  mkdir -p "$SCRIPT_DIR/data" "$SCRIPT_DIR/logs"

  nohup python -m karen.main >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "[karen] Started (PID $!)"
  echo "[karen] Log: tail -f $LOG_FILE"
}

cmd_stop() {
  local pid
  pid=$(get_pid)
  if [[ -z "$pid" ]]; then
    echo "[karen] Not running"
    return
  fi
  kill "$pid"
  rm -f "$PID_FILE"
  echo "[karen] Stopped (PID $pid)"
}

cmd_status() {
  local pid
  pid=$(get_pid)
  local strategy trading testnet
  strategy=$(get_env_var STRATEGY_MODE)
  trading=$(get_env_var TRADING_ENABLED)
  testnet=$(get_env_var BINANCE_TESTNET)

  echo "──────────────────────────────"
  if [[ -n "$pid" ]]; then
    echo "  Status   : RUNNING (PID $pid)"
  else
    echo "  Status   : STOPPED"
  fi
  echo "  Strategy : $strategy"
  echo "  Trading  : $trading"
  echo "  Testnet  : $testnet"
  echo "──────────────────────────────"
}

cmd_log() {
  tail -f "$LOG_FILE"
}

cmd_strategy() {
  local mode="${1:-}"
  if [[ "$mode" != "mean_reversion" && "$mode" != "trend_following" ]]; then
    echo "Usage: $0 strategy <mean_reversion|trend_following>"
    exit 1
  fi

  local current
  current=$(get_env_var STRATEGY_MODE)
  if [[ "$current" == "$mode" ]]; then
    echo "[karen] Strategy already set to '$mode' — no change"
    return
  fi

  if set_env_var STRATEGY_MODE "$mode"; then
    echo "[karen] Strategy → $mode (hot-reload in ~2s)"
    sleep 2
    grep "Strategy mode changed\|Config reloaded" "$LOG_FILE" | tail -3
  else
    echo "[karen] ERROR: could not update STRATEGY_MODE in $ENV_FILE"
    exit 1
  fi
}

cmd_trading() {
  local val="${1:-}"
  if [[ "$val" != "on" && "$val" != "off" ]]; then
    echo "Usage: $0 trading <on|off>"
    exit 1
  fi

  local bool_val
  [[ "$val" == "on" ]] && bool_val="true" || bool_val="false"

  local current
  current=$(get_env_var TRADING_ENABLED)
  if [[ "$current" == "$bool_val" ]]; then
    echo "[karen] Trading already $val — no change"
    return
  fi

  if set_env_var TRADING_ENABLED "$bool_val"; then
    echo "[karen] Trading → $val (hot-reload in ~2s)"
    sleep 2
    grep "Trading enabled\|Config reloaded" "$LOG_FILE" | tail -3
  else
    echo "[karen] ERROR: could not update TRADING_ENABLED in $ENV_FILE"
    exit 1
  fi
}

# ── dispatch ──────────────────────────────────────────────────────────────────

case "${1:-}" in
  start)    cmd_start ;;
  stop)     cmd_stop ;;
  status)   cmd_status ;;
  log)      cmd_log ;;
  strategy) cmd_strategy "${2:-}" ;;
  trading)  cmd_trading "${2:-}" ;;
  *)
    echo "Usage: $0 {start|stop|status|log|strategy|trading}"
    echo ""
    echo "  start                      start bot in background"
    echo "  stop                       stop bot"
    echo "  status                     show running status + current config"
    echo "  log                        tail live log"
    echo "  strategy mean_reversion    hot-swap to Mean Reversion"
    echo "  strategy trend_following   hot-swap to Trend Following"
    echo "  trading on                 enable trading (hot-reload)"
    echo "  trading off                disable trading (hot-reload)"
    exit 1
    ;;
esac
