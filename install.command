#!/bin/bash
# -------------------------------------------------------
# Bracket Tracker — One-Time Installer for Mac
# Double-click this file to install.
# -------------------------------------------------------

set -e
cd "$(dirname "$0")"

echo ""
echo "============================================"
echo "  Bracket Tracker — Installer"
echo "============================================"
echo ""

# --- Check for Python 3.9+ ---
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed."
    echo ""
    echo "Installing Xcode Command Line Tools (includes Python 3)..."
    echo "A system dialog may appear — click 'Install' and wait for it to finish."
    echo ""
    xcode-select --install 2>/dev/null || true
    echo ""
    echo "After the install finishes, double-click this file again."
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "Python $PYTHON_VERSION found, but 3.9+ is required."
    echo "Please install a newer Python from https://www.python.org/downloads/"
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

echo "Found Python $PYTHON_VERSION"

# --- Create virtual environment ---
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements-local.txt -q

# --- Install Playwright Chromium ---
echo ""
echo "Downloading browser (this may take a minute)..."
.venv/bin/playwright install chromium

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  To start the app, double-click:"
echo "    Bracket Tracker.command"
echo "============================================"
echo ""
read -n 1 -s -r -p "Press any key to close..."
