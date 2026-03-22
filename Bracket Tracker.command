#!/bin/bash
# -------------------------------------------------------
# Bracket Tracker — Launcher
# Double-click this file to start the app.
# Close this Terminal window to stop it.
# -------------------------------------------------------

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Not installed yet! Double-click 'install.command' first."
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

source .venv/bin/activate

echo ""
echo "============================================"
echo "  Bracket Tracker is starting..."
echo "  Opening your browser to localhost:8080"
echo ""
echo "  Close this window to stop the app."
echo "============================================"
echo ""

# Open browser after a short delay (gives Flask time to start)
(sleep 2 && open "http://localhost:8080") &

python app.py
