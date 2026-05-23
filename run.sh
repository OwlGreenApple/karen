#!/usr/bin/env bash
# Karen — start the trading bot + dashboard
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Build frontend if node_modules are present ─────────────────────────────────
FRONTEND_DIR="$SCRIPT_DIR/karen/dashboard/frontend"
STATIC_DIR="$SCRIPT_DIR/karen/dashboard/static"

if [[ -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "[karen] Building dashboard frontend..."
  cd "$FRONTEND_DIR"
  npm run build
  cd "$SCRIPT_DIR"
  echo "[karen] Frontend built → $STATIC_DIR"
else
  echo "[karen] Skipping frontend build (run 'npm install' in $FRONTEND_DIR first)"
fi

# ── Activate venv if present ──────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# ── Ensure data directory exists ──────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/data"

# ── Start bot (dashboard server is embedded) ──────────────────────────────────
echo "[karen] Starting Karen..."
exec python -m karen.main
