#!/usr/bin/env bash
#
# Bracket Tracker — Mac launcher
# Installs everything needed and opens the app in your browser.
# Run this once to set up, and again any time you want to use the app.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PORT=5000

# ── macOS check ───────────────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script only supports macOS."
    exit 1
fi

echo "=== Bracket Tracker ==="
echo ""

# ── Homebrew ──────────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew (this is a one-time step)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# ── Python 3.11+ ──────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(sys.version_info >= (3, 11))" 2>/dev/null || echo "False")
        if [[ "$ver" == "True" ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "Installing Python 3.11 via Homebrew..."
    brew install python@3.11
    PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
fi

echo "Using Python: $($PYTHON --version)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── Python dependencies ───────────────────────────────────────────────────────
echo "Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Playwright / Chromium ─────────────────────────────────────────────────────
# Only reinstall if chromium is not already present
if ! python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium.launch(); b.close(); p.stop()" &>/dev/null 2>&1; then
    echo "Installing Chromium browser (one-time, ~150 MB)..."
    playwright install chromium
fi

# ── Check port is free ────────────────────────────────────────────────────────
if lsof -iTCP:"$PORT" -sTCP:LISTEN -t &>/dev/null 2>&1; then
    echo ""
    echo "Port $PORT is already in use. Opening existing app..."
    open "http://localhost:$PORT"
    exit 0
fi

# ── Open browser after app starts ─────────────────────────────────────────────
(
    for i in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if curl -s "http://localhost:$PORT" &>/dev/null; then
            open "http://localhost:$PORT"
            break
        fi
    done
) &

# ── Start the app ─────────────────────────────────────────────────────────────
echo ""
echo "Starting Bracket Tracker at http://localhost:$PORT"
echo "Press Ctrl+C to stop."
echo ""

cd "$SCRIPT_DIR"
PORT="$PORT" python app.py
